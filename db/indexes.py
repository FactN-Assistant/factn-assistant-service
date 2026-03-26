"""
db/indexes.py
─────────────
MongoDB index definitions.

Call ensure_indexes(mongodb) once at application startup (inside lifespan)
AFTER the MongoDB client is created.  Motor's create_index calls are
idempotent — safe to run on every boot.

Index strategy
──────────────
tenants
  email           unique — login lookup

projects
  tenant_id       list all projects for a tenant
  tenant_id + _id compound — fetch single project scoped to tenant (security)

api_keys
  key_hash        unique — the hot lookup path: every WS connection hashes
                  the provided key and looks it up here
  project_id      list keys for a project (dashboard)
  tenant_id       list all keys for a tenant

sessions
  project_id + started_at  — paginated session list per project (analytics)
  tenant_id                 — cross-project usage queries
  status                    — find all active sessions (admin ops)

All _id fields are UUID strings — no ObjectId.  This makes cross-service
references trivial and avoids ObjectId serialisation headaches in Pydantic.
"""

from __future__ import annotations

import logging

from pymongo import ASCENDING, DESCENDING

from .mongo import MongoDB

log = logging.getLogger("livechat.db.indexes")


async def ensure_indexes(mongodb: MongoDB) -> None:
    """Create all required indexes.  Safe to call on every startup."""
    log.info("Ensuring MongoDB indexes…")

    # ── tenants ───────────────────────────────────────────────
    await mongodb.tenants.create_index(
        [("email", ASCENDING)],
        unique=True,
        name="tenants_email_unique",
    )

    # ── projects ──────────────────────────────────────────────
    await mongodb.projects.create_index(
        [("tenant_id", ASCENDING)],
        name="projects_by_tenant",
    )
    await mongodb.projects.create_index(
        [("tenant_id", ASCENDING), ("_id", ASCENDING)],
        name="projects_tenant_id_compound",
    )

    # ── api_keys ──────────────────────────────────────────────
    # This is the HOTTEST index — every WebSocket handshake hits it.
    await mongodb.api_keys.create_index(
        [("key_hash", ASCENDING)],
        unique=True,
        name="api_keys_hash_unique",
    )
    await mongodb.api_keys.create_index(
        [("project_id", ASCENDING)],
        name="api_keys_by_project",
    )
    await mongodb.api_keys.create_index(
        [("tenant_id", ASCENDING)],
        name="api_keys_by_tenant",
    )
    # Partial index: only index non-revoked keys for faster active-key scans
    await mongodb.api_keys.create_index(
        [("key_hash", ASCENDING)],
        partialFilterExpression={"revoked": False},
        name="api_keys_active_hash",
    )

    # ── sessions ──────────────────────────────────────────────
    await mongodb.sessions.create_index(
        [("project_id", ASCENDING), ("started_at", DESCENDING)],
        name="sessions_by_project_date",
    )
    await mongodb.sessions.create_index(
        [("tenant_id", ASCENDING)],
        name="sessions_by_tenant",
    )
    await mongodb.sessions.create_index(
        [("status", ASCENDING)],
        name="sessions_by_status",
    )

    log.info("MongoDB indexes ensured")