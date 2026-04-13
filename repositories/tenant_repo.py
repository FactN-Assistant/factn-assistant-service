"""
repositories/tenant_repo.py
────────────────────────────
TenantRepository — CRUD for the `tenants` collection.

All methods are async and return domain objects (TenantDoc), never raw
Motor dicts.  Callers never touch Motor directly.

New Changes for plans
──────────────────────────────
  suspend()     set is_suspended = True (quota exceeded / billing lapsed)
  unsuspend()   set is_suspended = False (payment received / quota reset)

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

    async def update_plan(self, tenant_id: str, plan: str) -> bool:
        """
        Update a tenant's plan tier.
        Called from the Stripe webhook handler when a subscription changes.
        """
        result = await self._col.update_one(
            {"_id": tenant_id},
            {"$set": {
                "plan":       plan,
                "updated_at": datetime.now(timezone.utc),
            }},
        )
        if result.modified_count:
            log.info("Tenant plan updated: %s → %s", tenant_id, plan)
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

    async def suspend(self, tenant_id: str) -> bool:
        """
        Suspend a tenant account.
        Called when daily token quota is exceeded or billing lapses.
        Suspended tenants:
          • Cannot open new WebSocket sessions (rejected at handshake)
          • Can still access the dashboard REST API
          • Can still manage projects and keys (but can't use them for chat)
        """
        result = await self._col.update_one(
            {"_id": tenant_id},
            {"$set": {
                "is_suspended": True,
                "updated_at":   datetime.now(timezone.utc),
            }},
        )
        if result.modified_count:
            log.warning("Tenant suspended: %s", tenant_id)
        return result.modified_count == 1

    async def unsuspend(self, tenant_id: str) -> bool:
        """
        Lift a suspension.
        Called when a Stripe payment succeeds or quota resets at midnight.
        """
        result = await self._col.update_one(
            {"_id": tenant_id},
            {"$set": {
                "is_suspended": False,
                "updated_at":   datetime.now(timezone.utc),
            }},
        )
        if result.modified_count:
            log.info("Tenant unsuspended: %s", tenant_id)
        return result.modified_count == 1

    # ── Delete ────────────────────────────────────────────────

    async def delete(self, tenant_id: str) -> bool:
        """Hard delete — prefer set_active(False) in production."""
        result = await self._col.delete_one({"_id": tenant_id})
        return result.deleted_count == 1