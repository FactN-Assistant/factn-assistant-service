"""
db/redis_client.py
──────────────────
Redis connection management using redis-py's async client.

Two use-cases in Week 5
───────────────────────
1.  Project config cache
    Key:   project_config:{project_id}
    Value: JSON-serialised ProjectConfig
    TTL:   60 s  (REDIS_CONFIG_CACHE_TTL)

    When the WebSocket endpoint resolves an API key it first checks Redis.
    On a miss it reads MongoDB, then writes back to Redis.  This keeps the
    "hot path" (new WebSocket connection) at sub-millisecond latency after
    the first lookup.

2.  Rate limit counters
    Key:   rate_limit:{api_key_prefix}:{window_start_minute}
    Value: integer request count (INCR + EXPIRE)
    TTL:   60 s

    A sliding-window counter per API key.  The limit itself (requests per
    minute) is stored on the APIKey document and checked here.
    Rate limiting is enforced in the WebSocket handshake before the Gemini
    session is opened.

Ephemeral tokens (Week 6) will also live here — the structure is already
designed to accommodate them with no schema changes.

Pattern
───────
A single Redis async client is created at startup (lifespan) and stored
on app.state.  All consumers import RedisClient from here and receive the
instance via dependency injection or app.state.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis

log = logging.getLogger("livechat.db.redis")

# Key prefixes — always use these constants, never bare strings.
PREFIX_PROJECT_CACHE = "project_config"
PREFIX_RATE_LIMIT    = "rate_limit"


class RedisClient:
    """
    Thin async wrapper around redis-py.
    Instantiated once in lifespan, stored on app.state.
    """

    def __init__(self, url: str) -> None:
        # decode_responses=True means all values come back as str, not bytes.
        # For binary data (audio) use raw ws connections, not Redis.
        self._redis = aioredis.from_url(
            url,
            decode_responses=True,
            max_connections=20,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        log.info("Redis client created")

    # ── Project config cache ───────────────────────────────────

    async def get_project_config(self, project_id: str) -> dict[str, Any] | None:
        """
        Return the cached project config dict or None on a cache miss.
        The caller (ProjectRepository) is responsible for re-populating.
        """
        key  = f"{PREFIX_PROJECT_CACHE}:{project_id}"
        raw  = await self._redis.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            log.warning("Corrupt project cache for %s — evicting", project_id)
            await self._redis.delete(key)
            return None

    async def set_project_config(
        self,
        project_id: str,
        data:       dict[str, Any],
        ttl:        int = 60,
    ) -> None:
        """Cache a project config dict with a TTL (seconds)."""
        key = f"{PREFIX_PROJECT_CACHE}:{project_id}"
        await self._redis.set(key, json.dumps(data), ex=ttl)

    async def invalidate_project_config(self, project_id: str) -> None:
        """
        Evict the cached config for a project.
        Call this whenever a project's config is updated in MongoDB so the
        next WebSocket connection picks up the new prompt/tools/voice.
        """
        await self._redis.delete(f"{PREFIX_PROJECT_CACHE}:{project_id}")
        log.info("Project config cache invalidated: %s", project_id)

    # ── Rate limit counters ────────────────────────────────────

    async def check_and_increment_rate_limit(
        self,
        key_prefix: str,
        limit:      int,
        window_s:   int = 60,
    ) -> tuple[bool, int]:
        """
        Sliding-window rate limiter using a single Redis key per minute.

        Returns
        -------
        (allowed, current_count)
            allowed=True  → request is within limit
            allowed=False → limit exceeded, caller should reject
        """
        import time
        window = int(time.time()) // window_s
        redis_key = f"{PREFIX_RATE_LIMIT}:{key_prefix}:{window}"

        # Atomic increment — if key doesn't exist Redis creates it at 0 first
        count = await self._redis.incr(redis_key)
        if count == 1:
            # First request in this window — set the TTL
            await self._redis.expire(redis_key, window_s * 2)

        allowed = count <= limit
        if not allowed:
            log.warning(
                "Rate limit exceeded: key=%s count=%d limit=%d",
                key_prefix, count, limit,
            )
        return allowed, count

    # ── Generic helpers ────────────────────────────────────────

    async def ping(self) -> bool:
        """Return True if Redis is reachable.  Used in /health."""
        try:
            return await self._redis.ping()
        except Exception as exc:
            log.error("Redis ping failed: %s", exc)
            return False

    async def close(self) -> None:
        await self._redis.aclose()
        log.info("Redis client closed")