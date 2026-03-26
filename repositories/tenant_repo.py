"""
repositories/tenant_repo.py
────────────────────────────
TenantRepository — CRUD for the `tenants` collection.

All methods are async and return domain objects (TenantDoc), never raw
Motor dicts.  Callers never touch Motor directly.

Error handling
──────────────
  DuplicateKeyError  → raised as-is; the API layer translates to HTTP 409.
  Document not found → returns None (callers check and raise HTTP 404).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from pymongo.errors import DuplicateKeyError

from core.documents import TenantDoc
from db.mongo import MongoDB

log = logging.getLogger("livechat.repo.tenant")


class TenantRepository:
    def __init__(self, mongodb: MongoDB) -> None:
        self._col = mongodb.tenants

    # ── Create ────────────────────────────────────────────────

    async def create(self, name: str, email: str) -> TenantDoc:
        """
        Insert a new tenant.

        Raises DuplicateKeyError if the email is already registered.
        The password_hash field is empty — Week 6 auth layer fills it in
        after calling argon2 hashing.
        """
        doc = TenantDoc(
            id    = str(uuid.uuid4()),
            name  = name,
            email = email.lower().strip(),
        )
        await self._col.insert_one(doc.to_mongo())
        log.info("Tenant created: %s (%s)", doc.id, email)
        return doc

    # ── Read ──────────────────────────────────────────────────

    async def get_by_id(self, tenant_id: str) -> TenantDoc | None:
        raw = await self._col.find_one({"_id": tenant_id})
        return TenantDoc.from_mongo(raw) if raw else None

    async def get_by_email(self, email: str) -> TenantDoc | None:
        raw = await self._col.find_one({"email": email.lower().strip()})
        return TenantDoc.from_mongo(raw) if raw else None

    async def list_all(self, limit: int = 100, skip: int = 0) -> list[TenantDoc]:
        cursor = self._col.find({}).skip(skip).limit(limit)
        return [TenantDoc.from_mongo(d) async for d in cursor]

    # ── Update ────────────────────────────────────────────────

    async def update_password_hash(
        self, tenant_id: str, password_hash: str
    ) -> bool:
        result = await self._col.update_one(
            {"_id": tenant_id},
            {"$set": {
                "password_hash": password_hash,
                "updated_at":    datetime.now(timezone.utc),
            }},
        )
        return result.modified_count == 1

    async def set_active(self, tenant_id: str, active: bool) -> bool:
        result = await self._col.update_one(
            {"_id": tenant_id},
            {"$set": {
                "is_active":  active,
                "updated_at": datetime.now(timezone.utc),
            }},
        )
        return result.modified_count == 1

    # ── Delete ────────────────────────────────────────────────

    async def delete(self, tenant_id: str) -> bool:
        """Hard delete — prefer set_active(False) in production."""
        result = await self._col.delete_one({"_id": tenant_id})
        return result.deleted_count == 1