"""
db/redis_client.py
──────────────────
Redis connection management using redis-py's async client.
 
Three responsibilities
──────────────────────
1.  Project config cache
    Key:   project_config:{project_id}
    Value: JSON-serialised ProjectConfig
    TTL:   60 s
 
2.  Rate limit counters
    Key:   rate_limit:{key_prefix}:{window_start_minute}
    Value: integer count (INCR + EXPIRE)
    TTL:   120 s
 
3.  Ephemeral tokens  ← added Week 8
    Key:   ephemeral_token:{token}
    Value: JSON blob  { project_id, tenant_id, api_key_id,
                        rate_limit_rpm, metadata, issued_at,
                        expires_at }
    TTL:   configurable per token (1–300 s, default 60 s)
 
Ephemeral token — single-use enforcement
─────────────────────────────────────────
redeem_ephemeral_token() uses a Lua script executed via EVAL/EVALSHA
to atomically GET-then-DELETE the token key.  Because Lua scripts run
atomically inside Redis, two concurrent WebSocket connections racing
with the same token cannot both succeed — the first caller gets the
payload, the second always gets nil (None in Python).
 
This eliminates any race condition that a two-command GET + DEL
approach would have.

Pattern
───────
A single Redis async client is created at startup (lifespan) and stored
on app.state.  All consumers import RedisClient from here and receive the
instance via dependency injection or app.state.
"""

from __future__ import annotations
 
import json
import logging
import secrets
import time
from typing import Any
 
import redis.asyncio as aioredis
 
log = logging.getLogger("livechat.db.redis")
 
PREFIX_PROJECT_CACHE   = "project_config"
PREFIX_RATE_LIMIT      = "rate_limit"
PREFIX_EPHEMERAL_TOKEN = "ephemeral_token"
 
EPHEMERAL_TOKEN_DEFAULT_TTL = 60    # seconds
EPHEMERAL_TOKEN_MAX_TTL     = 300   # seconds
 
# Atomic GET-and-DELETE Lua script.
# Returns the stored value if the key existed, else nil.
_LUA_REDEEM = """
local val = redis.call('GET', KEYS[1])
if val then
    redis.call('DEL', KEYS[1])
    return val
end
return nil
"""
 
 
class RedisClient:
    """
    Thin async wrapper around redis-py.
    Instantiated once in lifespan, stored on app.state.
    """
 
    def __init__(self, url: str) -> None:
        self._redis = aioredis.from_url(
            url,
            decode_responses=True,
            max_connections=20,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        # register_script() pre-registers the Lua script so subsequent calls
        # use EVALSHA (sends only the hash, not the full script body).
        self._redeem_script = self._redis.register_script(_LUA_REDEEM)
        log.info("Redis client created")
 
    # ── Project config cache ───────────────────────────────────
 
    async def get_project_config(self, project_id: str) -> dict[str, Any] | None:
        key = f"{PREFIX_PROJECT_CACHE}:{project_id}"
        raw = await self._redis.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            log.warning("Corrupt project cache for %s — evicting", project_id)
            await self._redis.delete(key)
            return None
 
    async def set_project_config(
        self, project_id: str, data: dict[str, Any], ttl: int = 60
    ) -> None:
        key = f"{PREFIX_PROJECT_CACHE}:{project_id}"
        await self._redis.set(key, json.dumps(data), ex=ttl)
 
    async def invalidate_project_config(self, project_id: str) -> None:
        await self._redis.delete(f"{PREFIX_PROJECT_CACHE}:{project_id}")
        log.info("Project config cache invalidated: %s", project_id)
 
    # ── Rate limit counters ────────────────────────────────────
 
    async def check_and_increment_rate_limit(
        self, key_prefix: str, limit: int, window_s: int = 60
    ) -> tuple[bool, int]:
        window    = int(time.time()) // window_s
        redis_key = f"{PREFIX_RATE_LIMIT}:{key_prefix}:{window}"
        count     = await self._redis.incr(redis_key)
        if count == 1:
            await self._redis.expire(redis_key, window_s * 2)
        allowed = count <= limit
        if not allowed:
            log.warning(
                "Rate limit exceeded: key=%s count=%d limit=%d",
                key_prefix, count, limit,
            )
        return allowed, count
 
    # ── Ephemeral tokens ───────────────────────────────────────
 
    async def create_ephemeral_token(
        self,
        project_id:     str,
        tenant_id:      str,
        api_key_id:     str,
        rate_limit_rpm: int,
        ttl:            int               = EPHEMERAL_TOKEN_DEFAULT_TTL,
        metadata:       dict[str, Any]    = None,
    ) -> tuple[str, int]:
        """
        Generate a single-use ephemeral token and store it in Redis.
 
        Parameters
        ──────────
        project_id       project this token grants access to
        tenant_id        owner tenant (written into session records)
        api_key_id       secret key that authorised this issuance
        rate_limit_rpm   inherited from the secret key; enforced at connect
        ttl              lifetime seconds (1–300, default 60)
        metadata         arbitrary dict attached by the customer's backend
                         e.g. {"user_id": "u_123", "locale": "en"}
 
        Returns
        ───────
        (raw_token, expires_at_unix_timestamp)
        """
        ttl        = max(1, min(ttl, EPHEMERAL_TOKEN_MAX_TTL))
        raw_token  = secrets.token_urlsafe(32)
        issued_at  = int(time.time())
        expires_at = issued_at + ttl
 
        payload: dict[str, Any] = {
            "project_id":     project_id,
            "tenant_id":      tenant_id,
            "api_key_id":     api_key_id,
            "rate_limit_rpm": rate_limit_rpm,
            "issued_at":      issued_at,
            "expires_at":     expires_at,
            "metadata":       metadata or {},
        }
 
        key = f"{PREFIX_EPHEMERAL_TOKEN}:{raw_token}"
        await self._redis.set(key, json.dumps(payload), ex=ttl)
 
        log.info(
            "Ephemeral token created (project=%s ttl=%ds)",
            project_id, ttl,
        )
        return raw_token, expires_at
 
    async def redeem_ephemeral_token(
        self, raw_token: str
    ) -> dict[str, Any] | None:
        """
        Atomically read and delete an ephemeral token (single-use).
 
        The Lua script makes GET + DEL one atomic Redis operation.
        Only the first caller wins — any concurrent or replay attempt
        receives None.
 
        Returns the stored payload dict, or None if the token:
          • does not exist (never issued)
          • was already redeemed (deleted by a previous call)
          • expired (Redis TTL auto-deleted it)
        """
        key = f"{PREFIX_EPHEMERAL_TOKEN}:{raw_token}"
        raw = await self._redeem_script(keys=[key])
 
        if raw is None:
            log.warning(
                "Ephemeral token not found / already redeemed: %s…",
                raw_token[:8],
            )
            return None
 
        try:
            payload = json.loads(raw)
            log.info(
                "Ephemeral token redeemed (project=%s)",
                payload.get("project_id"),
            )
            return payload
        except json.JSONDecodeError:
            log.error("Corrupt ephemeral token payload — discarding")
            return None
 
    async def peek_ephemeral_token(
        self, raw_token: str
    ) -> dict[str, Any] | None:
        """
        Read a token payload WITHOUT consuming it.
        Used by POST /v1/tokens/rotate to verify the old token before
        issuing a replacement.
        """
        key = f"{PREFIX_EPHEMERAL_TOKEN}:{raw_token}"
        raw = await self._redis.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
 
    async def get_ephemeral_token_ttl(self, raw_token: str) -> int:
        """
        Return remaining TTL (seconds) of a token.
        Returns -2 if the key does not exist.
        """
        return await self._redis.ttl(
            f"{PREFIX_EPHEMERAL_TOKEN}:{raw_token}"
        )
 
    # ── Generic helpers ────────────────────────────────────────
 
    async def ping(self) -> bool:
        try:
            return await self._redis.ping()
        except Exception as exc:
            log.error("Redis ping failed: %s", exc)
            return False
 
    async def close(self) -> None:
        await self._redis.aclose()
        log.info("Redis client closed")