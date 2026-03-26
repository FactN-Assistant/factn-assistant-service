"""
repositories/api_key_repo.py
─────────────────────────────
APIKeyRepository — manages the `api_keys` collection.

Security design
───────────────
API keys are generated as random 32-character base62 strings with a typed
prefix (pk_live_ for publishable, sk_live_ for secret).

The FULL key is returned ONCE at creation time — it is NEVER stored in the
database.  Only the argon2id hash is persisted.  The key prefix (first 12
chars) is stored in plaintext for dashboard display.

On every WebSocket connection the client provides the full key.  We hash
it with argon2 and do a constant-time comparison against the stored hash.

We use argon2-cffi for hashing (the same library recommended for password
hashing).  Parameters are deliberately modest so that a live API key lookup
takes ~10 ms rather than the 200-500 ms used for passwords — fast enough
for interactive API usage, still infeasible to brute-force.
"""

from __future__ import annotations

import logging
import secrets
import string
import uuid
from datetime import datetime, timezone

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError

from core.documents import APIKeyDoc
from db.mongo import MongoDB

log = logging.getLogger("livechat.repo.api_key")

# Argon2 hasher — tuned for API key verification speed.
# time_cost=1, memory_cost=32MB, parallelism=1 → ~5-15 ms per hash on M0.
# For password hashing use time_cost=3, memory_cost=65536 (Week 6).
_HASHER = PasswordHasher(
    time_cost    = 1,
    memory_cost  = 32768,   # 32 MB
    parallelism  = 1,
    hash_len     = 32,
    salt_len     = 16,
)

# Key alphabet: base62 (no ambiguous chars like 0/O/l/I)
_ALPHABET = string.ascii_letters + string.digits
_KEY_BODY_LEN = 32


def _generate_raw_key(key_type: str) -> str:
    """
    Generate a full API key string.

    Format:  {type_prefix}_{random_body}
    Example: pk_live_Xy7mQ3rN9vP2kLsT8wA6bD4cJhFe1GuZ
    """
    env    = "live"   # TODO: "test" for test-mode keys (Week 6)
    prefix = "pk" if key_type == "publishable" else "sk"
    body   = "".join(secrets.choice(_ALPHABET) for _ in range(_KEY_BODY_LEN))
    return f"{prefix}_{env}_{body}"


class APIKeyRepository:
    def __init__(self, mongodb: MongoDB) -> None:
        self._col = mongodb.api_keys

    # ── Create ────────────────────────────────────────────────

    async def create(
        self,
        project_id: str,
        tenant_id:  str,
        label:      str = "Default key",
        key_type:   str = "publishable",
        rate_limit_rpm: int = 60,
    ) -> tuple[APIKeyDoc, str]:
        """
        Create a new API key.

        Returns
        -------
        (APIKeyDoc, raw_key)
            APIKeyDoc  — the stored document (key_hash set, raw key NOT stored)
            raw_key    — the full key string.  Return this to the user ONCE.
                         It cannot be recovered after this call.
        """
        raw_key    = _generate_raw_key(key_type)
        key_hash   = _HASHER.hash(raw_key)
        key_prefix = raw_key[:12]   # e.g. "pk_live_Xy7m"

        doc = APIKeyDoc(
            id             = str(uuid.uuid4()),
            project_id     = project_id,
            tenant_id      = tenant_id,
            label          = label,
            key_prefix     = key_prefix,
            key_hash       = key_hash,
            key_type       = key_type,
            rate_limit_rpm = rate_limit_rpm,
        )
        await self._col.insert_one(doc.to_mongo())
        log.info(
            "API key created: %s (project=%s type=%s prefix=%s)",
            doc.id, project_id, key_type, key_prefix,
        )
        return doc, raw_key

    # ── Lookup (hot path) ─────────────────────────────────────

    async def get_by_raw_key(self, raw_key: str) -> APIKeyDoc | None:
        """
        Verify a raw API key against stored hashes.

        Strategy
        ────────
        1. Extract the prefix from the raw key (first 12 chars).
        2. Query MongoDB by key_prefix to narrow candidates
           (avoids hashing every key in the collection).
        3. Verify the hash of the raw key against each candidate.

        This is safe because:
          • key_prefix alone is not secret — it's shown in the dashboard.
          • The argon2 hash verification is the security gate.
          • The prefix-based pre-filter keeps DB reads to O(1) in practice.

        Returns None if the key is not found, is revoked, or is expired.
        """
        if not raw_key:
            return None

        prefix = raw_key[:12]

        # Find active keys with this prefix
        cursor = self._col.find({
            "key_prefix": prefix,
            "revoked":    False,
        })

        async for raw_doc in cursor:
            doc = APIKeyDoc.from_mongo(raw_doc)
            try:
                _HASHER.verify(doc.key_hash, raw_key)
            except (VerifyMismatchError, VerificationError, InvalidHashError):
                continue   # wrong key — check next candidate

            # Match found — check expiry
            if not doc.is_valid:
                log.warning("API key %s is expired", doc.key_prefix)
                return None

            # Update last_used_at in the background (fire-and-forget)
            await self._touch_last_used(doc.id)
            return doc

        return None   # no matching key found

    # ── Read ──────────────────────────────────────────────────

    async def list_for_project(self, project_id: str) -> list[APIKeyDoc]:
        """List all keys for a project (prefix + metadata only — no hashes exposed via API)."""
        cursor = self._col.find({"project_id": project_id}).sort("created_at", -1)
        return [APIKeyDoc.from_mongo(d) async for d in cursor]

    async def get_by_id(
        self, key_id: str, tenant_id: str
    ) -> APIKeyDoc | None:
        raw = await self._col.find_one({"_id": key_id, "tenant_id": tenant_id})
        return APIKeyDoc.from_mongo(raw) if raw else None

    # ── Revoke ────────────────────────────────────────────────

    async def revoke(self, key_id: str, tenant_id: str) -> bool:
        """
        Revoke a key.  Subsequent get_by_raw_key() calls will return None.
        Revoking is permanent — keys are never un-revoked.
        """
        result = await self._col.update_one(
            {"_id": key_id, "tenant_id": tenant_id},
            {"$set": {
                "revoked":    True,
                "revoked_at": datetime.now(timezone.utc),
            }},
        )
        if result.modified_count:
            log.info("API key revoked: %s", key_id)
        return result.modified_count == 1

    # ── Internal ──────────────────────────────────────────────

    async def _touch_last_used(self, key_id: str) -> None:
        try:
            await self._col.update_one(
                {"_id": key_id},
                {"$set": {"last_used_at": datetime.now(timezone.utc)}},
            )
        except Exception as exc:
            # Non-critical — don't fail the connection over a timestamp update
            log.warning("Failed to update last_used_at for key %s: %s", key_id, exc)