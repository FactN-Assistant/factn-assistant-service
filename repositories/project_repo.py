"""
repositories/project_repo.py
─────────────────────────────
ProjectRepository — CRUD for the `projects` collection + the critical
get_config_by_api_key() hot-path that every WebSocket connection uses.

Hot path (every new WebSocket connection)
─────────────────────────────────────────
  1. APIKeyRepository.get_by_key_hash(hash)   → APIKeyDoc
  2. ProjectRepository.get_config_for_key(doc) → ProjectConfig  ← this file
     a. Check Redis cache  (sub-millisecond on hit)
     b. On miss: read MongoDB, write back to Redis, return
  3. SessionManager.get_or_create(session_id, project_config)

The Redis cache means that after the first connection for a project, all
subsequent connections pay only a Redis lookup, not a MongoDB round-trip.

Cache invalidation
──────────────────
Call redis_client.invalidate_project_config(project_id) whenever a project
document is mutated (update_* methods below call it automatically).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from core.documents import ProjectDoc
from core.schemas import ProjectConfig, ToolDefinition, VoiceConfig, VADConfig
from db.mongo import MongoDB
from db.redis_client import RedisClient

log = logging.getLogger("livechat.repo.project")

# How long a ProjectConfig lives in Redis before a fresh DB read is forced.
_CACHE_TTL = 60  # seconds


class ProjectRepository:
    def __init__(self, mongodb: MongoDB, redis: RedisClient) -> None:
        self._col   = mongodb.projects
        self._redis = redis

    # ── Hot path ──────────────────────────────────────────────

    async def get_config_for_key(self, api_key_doc) -> ProjectConfig | None:
        """
        Resolve a ProjectConfig from an APIKeyDoc.

        Checks Redis first, falls through to MongoDB on a miss.
        Returns None if the project doesn't exist or is inactive.

        This is called on EVERY WebSocket connection — keep it fast.
        """
        project_id = api_key_doc.project_id

        # 1. Redis cache hit?
        cached = await self._redis.get_project_config(project_id)
        if cached is not None:
            log.debug("Project config cache HIT: %s", project_id)
            return _dict_to_project_config(cached)

        # 2. MongoDB read
        log.debug("Project config cache MISS: %s — reading MongoDB", project_id)
        doc = await self.get_by_id(project_id, tenant_id=api_key_doc.tenant_id)
        if doc is None or not doc.is_active:
            return None

        # 3. Populate cache
        config = doc.to_project_config()
        await self._redis.set_project_config(
            project_id, _project_config_to_dict(config), ttl=_CACHE_TTL
        )
        return config

    async def get_config_by_id(
        self,
        project_id: str,
        tenant_id:  str,
    ) -> ProjectConfig | None:
        """
        Resolve a ProjectConfig directly from project_id + tenant_id.

        Used by the ephemeral token path in api/chat.py where we have
        the project_id from the token payload but no APIKeyDoc to pass
        to get_config_for_key().

        Follows the same Redis-first cache pattern as get_config_for_key().
        """
        # 1. Redis cache hit?
        cached = await self._redis.get_project_config(project_id)
        if cached is not None:
            log.debug("Project config cache HIT (by id): %s", project_id)
            return _dict_to_project_config(cached)

        # 2. MongoDB read — scope to tenant_id for security
        log.debug("Project config cache MISS (by id): %s", project_id)
        doc = await self.get_by_id(project_id, tenant_id=tenant_id)
        if doc is None or not doc.is_active:
            return None

        # 3. Populate cache
        config = doc.to_project_config()
        await self._redis.set_project_config(
            project_id, _project_config_to_dict(config), ttl=_CACHE_TTL
        )
        return config

    # ── Create ────────────────────────────────────────────────

    async def create(
        self,
        tenant_id:     str,
        name:          str,
        system_prompt: str,
        **kwargs: Any,
    ) -> ProjectDoc:
        doc = ProjectDoc(
            id            = str(uuid.uuid4()),
            tenant_id     = tenant_id,
            name          = name,
            system_prompt = system_prompt,
            **kwargs,
        )
        await self._col.insert_one(doc.to_mongo())
        log.info(
            "Project created: %s (tenant=%s name=%r)",
            doc.id, tenant_id, name,
        )
        return doc

    # ── Read ──────────────────────────────────────────────────

    async def get_by_id(
        self,
        project_id: str,
        tenant_id:  str | None = None,
    ) -> ProjectDoc | None:
        """
        Fetch a project by ID, optionally scoped to a tenant_id.
        Always pass tenant_id from authenticated contexts to prevent
        cross-tenant data access.
        """
        query: dict[str, Any] = {"_id": project_id}
        if tenant_id:
            query["tenant_id"] = tenant_id
        raw = await self._col.find_one(query)
        return ProjectDoc.from_mongo(raw) if raw else None

    async def list_for_tenant(
        self,
        tenant_id: str,
        limit:     int = 50,
        skip:      int = 0,
    ) -> list[ProjectDoc]:
        cursor = (
            self._col.find({"tenant_id": tenant_id})
            .sort("created_at", -1)
            .skip(skip)
            .limit(limit)
        )
        return [ProjectDoc.from_mongo(d) async for d in cursor]

    # ── Update ────────────────────────────────────────────────

    async def update(
        self,
        project_id: str,
        tenant_id:  str,
        fields:     dict[str, Any],
    ) -> bool:
        """
        Partial update.  fields is a dict of top-level document fields to
        set.  Always invalidates the Redis cache so the next WS connection
        picks up the new config immediately.
        """
        fields["updated_at"] = datetime.now(timezone.utc)
        result = await self._col.update_one(
            {"_id": project_id, "tenant_id": tenant_id},
            {"$set": fields},
        )
        if result.modified_count:
            await self._redis.invalidate_project_config(project_id)
            log.info("Project updated: %s", project_id)
        return result.modified_count == 1

    async def touch_last_accessed(self, project_id: str, tenant_id: str) -> None:
        """
        Non-blocking update of last_accessed without invalidating the config
        cache — this field is only used for dashboard sorting and does not
        affect runtime behaviour.
        """
        await self._col.update_one(
            {"_id": project_id, "tenant_id": tenant_id},
            {"$set": {"last_accessed": datetime.now(timezone.utc)}},
        )
        log.debug("Project last_accessed touched: %s", project_id)

    async def set_tools(
        self,
        project_id: str,
        tenant_id:  str,
        tools:      list[ToolDefinition],
    ) -> bool:
        """Replace the entire tools array atomically."""
        return await self.update(
            project_id, tenant_id,
            {"tools": [t.model_dump() for t in tools]},
        )

    # ── Delete ────────────────────────────────────────────────

    async def soft_delete(self, project_id: str, tenant_id: str) -> bool:
        """
        Mark project inactive.  Existing sessions continue until they idle
        out; new WebSocket connections will be rejected (project not found).
        """
        result = await self.update(
            project_id, tenant_id,
            {"is_active": False},
        )
        if result:
            await self._redis.invalidate_project_config(project_id)
        return result


# ── Serialisation helpers (ProjectConfig ↔ plain dict for Redis) ──────────
# We can't store Pydantic models in Redis directly.  These helpers convert
# to/from plain dicts that json.dumps() can handle.

def _project_config_to_dict(cfg: ProjectConfig) -> dict[str, Any]:
    return cfg.model_dump()


def _dict_to_project_config(d: dict[str, Any]) -> ProjectConfig:
    # Nested models (VoiceConfig, VADConfig, ToolDefinition) are reconstructed
    # automatically by Pydantic's model_validate.
    return ProjectConfig.model_validate(d)