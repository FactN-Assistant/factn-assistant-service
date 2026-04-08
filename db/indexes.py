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

This was migrated to PyMongo Async removing Motor due to the reason Motor is about to deprecate.
PyMongo Async note: create_index is awaitable, same as Motor.

Changes for Plans
──────────────────────────────
  tenants  — added partial index on is_suspended for fast suspension queries.
  sessions — added compound (tenant_id, started_at) for daily token quota
             aggregation in get_daily_token_usage().
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
    # Partial index: only indexes documents where is_suspended=True.
    # Used by admin queries to list all currently suspended tenants.
    await mongodb.tenants.create_index(
        [("is_suspended", ASCENDING)],
        partialFilterExpression={"is_suspended": True},
        name="tenants_suspended",
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
    await mongodb.api_keys.create_index(
        [("key_prefix", ASCENDING)],
        partialFilterExpression={"revoked": False},
        name="api_keys_active_prefix",
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
    # Compound index for daily token quota aggregation.
    # get_daily_token_usage() filters on tenant_id + started_at (today).
    # This index serves that query without a full collection scan.
    await mongodb.sessions.create_index(
        [("tenant_id", ASCENDING), ("started_at", DESCENDING)],
        name="sessions_tenant_date",
    )

    # ── auth_tokens ───────────────────────────────────────────
    await mongodb.db["auth_tokens"].create_index(
        [("expires_at", ASCENDING)],
        expireAfterSeconds=0,
        name="auth_tokens_ttl",
    )
    await mongodb.db["auth_tokens"].create_index(
        [("token_hash", ASCENDING)],
        unique=True,
        name="auth_tokens_hash_unique",
    )
 
    log.info("MongoDB indexes ensured")