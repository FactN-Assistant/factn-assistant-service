"""
db/mongo.py
───────────
MongoDB connection management using Motor (async PyMongo driver).

Pattern
───────
A single AsyncIOMotorClient is created at application startup (lifespan)
and stored on app.state.  All repositories receive the MongoDB instance —
they never create their own connections.

Collections  (one database, four collections)
─────────────────────────────────────────────
  tenants    — customer organisation accounts
  projects   — chatbot projects owned by tenants
  api_keys   — API keys linked to projects
  sessions   — session records written on session close

See db/indexes.py for the index definitions that are applied on first boot.
"""

from __future__ import annotations

import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

log = logging.getLogger("livechat.db.mongo")

# Collection name constants — import these everywhere instead of
# hardcoding strings so a typo is a NameError, not a silent bug.
COL_TENANTS  = "tenants"
COL_PROJECTS = "projects"
COL_API_KEYS = "api_keys"
COL_SESSIONS = "sessions"


class MongoDB:
    """
    Thin wrapper around the Motor client exposing typed collection accessors.
    Instantiated once in lifespan and stored on app.state.
    """

    def __init__(self, uri: str, db_name: str) -> None:
        self._client: AsyncIOMotorClient = AsyncIOMotorClient(
            uri,
            # Keep the pool small for Atlas free tier (M0) which caps at 500
            # concurrent connections.  Raise maxPoolSize to 50 on M10+.
            maxPoolSize=10,
            minPoolSize=1,
            serverSelectionTimeoutMS=10_000,
            connectTimeoutMS=10_000,
            socketTimeoutMS=30_000,
        )
        self._db: AsyncIOMotorDatabase = self._client[db_name]
        log.info("MongoDB client created (db=%s)", db_name)

    # ── Collection accessors ───────────────────────────────────

    @property
    def tenants(self):
        return self._db[COL_TENANTS]

    @property
    def projects(self):
        return self._db[COL_PROJECTS]

    @property
    def api_keys(self):
        return self._db[COL_API_KEYS]

    @property
    def sessions(self):
        return self._db[COL_SESSIONS]

    @property
    def db(self) -> AsyncIOMotorDatabase:
        """Raw database — for index creation and admin ops."""
        return self._db

    # ── Lifecycle ──────────────────────────────────────────────

    async def ping(self) -> bool:
        """Return True if Atlas is reachable.  Used in /health."""
        try:
            await self._client.admin.command("ping")
            return True
        except Exception as exc:
            log.error("MongoDB ping failed: %s", exc)
            return False

    def close(self) -> None:
        self._client.close()
        log.info("MongoDB client closed")