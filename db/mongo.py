"""
db/mongo.py
───────────
MongoDB connection using PyMongo's native async API.

Motor → PyMongo Async migration (Motor deprecated May 2025)
────────────────────────────────────────────────────────────
  Motor                         PyMongo Async
  ──────────────────────────    ──────────────────────────────
  from motor.motor_asyncio      from pymongo
  import AsyncIOMotorClient     import AsyncMongoClient
  AsyncIOMotorClient(uri)       AsyncMongoClient(uri)           ← same args
  await col.find_one(...)       await col.find_one(...)         ← identical
  cursor.to_list(0)             cursor.to_list(None)            ← 0 invalid!
  async for doc in cursor:      async for doc in cursor:        ← identical

Key constraints:
  • AsyncMongoClient does NOT accept an io_loop parameter.
  • AsyncMongoClient is NOT thread-safe — one event loop only.
    FastAPI's lifespan guarantees this.
  • to_list(None) = unlimited; to_list(N) = at most N docs.
"""

from __future__ import annotations

import logging

from pymongo import AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

log = logging.getLogger("livechat.db.mongo")

COL_TENANTS  = "tenants"
COL_PROJECTS = "projects"
COL_API_KEYS = "api_keys"
COL_SESSIONS = "sessions"


class MongoDB:
    """
    Thin wrapper around AsyncMongoClient exposing typed collection accessors.
    Instantiated once in lifespan, stored on app.state.
    """

    def __init__(self, uri: str, db_name: str) -> None:
        self._client: AsyncMongoClient = AsyncMongoClient(
            uri,
            maxPoolSize=10,       # Atlas M0 free tier caps at 500 total
            minPoolSize=1,
            serverSelectionTimeoutMS=10_000,
            connectTimeoutMS=10_000,
            socketTimeoutMS=30_000,
        )
        self._db: AsyncDatabase = self._client[db_name]
        log.info("MongoDB (PyMongo Async) client created (db=%s)", db_name)

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
    def db(self) -> AsyncDatabase:
        return self._db

    async def ping(self) -> bool:
        try:
            await self._client.admin.command("ping")
            return True
        except Exception as exc:
            log.error("MongoDB ping failed: %s", exc)
            return False

    def close(self) -> None:
        self._client.close()
        log.info("MongoDB client closed")