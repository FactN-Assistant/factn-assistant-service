"""
api/chat.py
───────────
The single WebSocket endpoint every client connects to.

Week 5 changes from Week 2-3
─────────────────────────────
  • api_key is now RESOLVED against the database (APIKeyRepository)
  • ProjectConfig is loaded via ProjectRepository (Redis-cached hot path)
  • Per-key rate limiting enforced at handshake time (RedisClient)
  • WebSocket is closed with a clear 4xxx code on auth / rate-limit failure
    so clients can distinguish rejection reasons without polling
    
Week 6 changes from Week 5
─────────────────────────────
  • api_key is now RESOLVED against the database (APIKeyRepository)
  • ProjectConfig is loaded via ProjectRepository (Redis-cached hot path)
  • Per-key rate limiting enforced at handshake time (RedisClient)
  • WebSocket is closed with a clear 4xxx code on auth / rate-limit failure
    so clients can distinguish rejection reasons without polling
    
New changes for plans
──────────────────────────────
  _resolve_project()  now enforces:
    1. CORS — checks the request Origin header against project.allowed_origins.
               Empty list = allow all (open project).
               Mismatch = close with 4006.
    2. Tenant suspension — suspended tenants cannot open new sessions.
               Close code 4007.
    3. Daily token quota — checks today's token usage against the plan limit.
               Exceeded = close code 4008.
 
  ws_chat()  now catches MaxSessionsExceededError from SessionManager
             and closes with 4004.

Connection URL
──────────────
  wss://host/v1/chat?api_key=<key>[&session_id=<uuid>]
  wss://host/v1/chat?token=<ephemeral>[&session_id=<uuid>]
 
  Exactly one of api_key or token must be provided.
  token is single-use — it is deleted from Redis on the first
  successful WebSocket handshake.

Close codes used
────────────────
  4001  Missing or invalid API key / token
  4002  Project not found or inactive
  4003  Rate limit exceeded
  4004  Max concurrent sessions reached for this project
  4005  Ephemeral token already redeemed or expired
  4006  Origin not allowed (CORS)
  4007  Tenant account suspended
  4008  Daily token quota exceeded
"""
from __future__ import annotations
 
import asyncio
import json
import logging
import uuid
 
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
 
from core import plan_limits
from core.schemas import ProjectConfig
from core.session_manager import MaxSessionsExceededError, SessionManager
from core.session_state import FrameKind, InputFrame, _OUTBOX_STOP
from db.redis_client import RedisClient
from repositories import Repositories
 
log = logging.getLogger("livechat.ws")
 
router = APIRouter()
 
 
# ──────────────────────────────────────────────────────────────
# CORS origin check
# ──────────────────────────────────────────────────────────────
 
def _origin_allowed(origin: str | None, allowed_origins: list[str]) -> bool:
    """
    Return True when the request origin is permitted.
 
    Rules:
      • allowed_origins is empty → allow everything (open project)
      • origin header is missing → allow (non-browser clients, curl, etc.)
      • origin is in the whitelist → allow
      • otherwise → deny
    """
    if not allowed_origins:
        return True
    if origin is None:
        return True
    return origin in allowed_origins
 
 
# ──────────────────────────────────────────────────────────────
# Handshake — resolve project config from credentials
# ──────────────────────────────────────────────────────────────
 
async def _resolve_project(
    ws:      WebSocket,
    api_key: str | None,
    token:   str | None,
) -> tuple[ProjectConfig, str, str] | None:
    """
    Validate credentials and return (ProjectConfig, api_key_id, tenant_plan).

    Extended from the previous version with four additional checks:
      1. CORS origin enforcement
      2. Tenant suspension check
      3. Daily token quota check
 
    Returns None and closes the WebSocket (with an explanatory code) if
    anything fails.  Callers must return immediately on None.
    """
    repos: Repositories = ws.app.state.repos
    redis: RedisClient  = ws.app.state.redis
 
    # ── 1. Exactly one credential required ────────────────────
    if not api_key and not token:
        await ws.close(
            code=4001,
            reason="Either api_key or token query parameter is required.",
        )
        return None
 
    if api_key and token:
        log.debug("Both api_key and token provided — using token")
        api_key = None
 
    # ── Resolve config and api_key_id from whichever credential ──
    config:     ProjectConfig | None = None
    api_key_id: str                  = ""
    tenant_plan: str                 = "free"

    # ══════════════════════════════════════════════════════════
    # PATH A: Ephemeral token
    # ══════════════════════════════════════════════════════════
    if token:
        payload = await redis.redeem_ephemeral_token(token)
        if payload is None:
            await ws.close(
                code=4005,
                reason="Ephemeral token is invalid, already redeemed, or expired.",
            )
            return None
 
        project_id     = payload["project_id"]
        tenant_id      = payload["tenant_id"]
        api_key_id     = payload["api_key_id"]
        rate_limit_rpm = payload.get("rate_limit_rpm", 60)
 
        allowed, count = await redis.check_and_increment_rate_limit(
            key_prefix = f"tok:{project_id}",
            limit      = rate_limit_rpm,
            window_s   = 60,
        )
        if not allowed:
            await ws.close(
                code=4003,
                reason=f"Rate limit exceeded ({rate_limit_rpm} req/min).",
            )
            return None
 
        config = await repos.projects.get_config_by_id(
            project_id = project_id,
            tenant_id  = tenant_id,
        )
        if config is None:
            await ws.close(code=4002, reason="Project not found or inactive.")
            return None

        # Fetch tenant for suspension + plan checks
        tenant = await repos.tenants.get_by_id(tenant_id)
        if tenant is None:
            await ws.close(code=4001, reason="Tenant account not found.")
            return None
        tenant_plan = tenant.plan
 
        log.info("Token auth: project=%s api_key_id=%s", project_id, api_key_id)

    # ══════════════════════════════════════════════════════════
    # PATH B: Long-lived API key
    # ══════════════════════════════════════════════════════════
    else:
        key_doc = await repos.api_keys.get_by_raw_key(api_key)
        if key_doc is None:
            await ws.close(code=4001, reason="Invalid or revoked API key.")
            return None
 
        allowed, count = await redis.check_and_increment_rate_limit(
            key_prefix = key_doc.key_prefix,
            limit      = key_doc.rate_limit_rpm,
            window_s   = 60,
        )
        if not allowed:
            await ws.close(
                code=4003,
                reason=f"Rate limit exceeded ({key_doc.rate_limit_rpm} req/min).",
            )
            return None

        config = await repos.projects.get_config_for_key(key_doc)
        if config is None:
            await ws.close(code=4002, reason="Project not found or inactive.")
            return None
 
        api_key_id = key_doc.id
 
        # Fetch tenant for suspension + plan checks
        tenant = await repos.tenants.get_by_id(key_doc.tenant_id)
        if tenant is None:
            await ws.close(code=4001, reason="Tenant account not found.")
            return None
        tenant_plan = tenant.plan

    # ── 2. CORS origin enforcement ────────────────────────────
    origin = ws.headers.get("origin")
    if not _origin_allowed(origin, config.allowed_origins):
        log.warning(
            "CORS rejected: origin=%s project=%s allowed=%s",
            origin, config.project_id, config.allowed_origins,
        )
        await ws.close(
            code=4006,
            reason=f"Origin '{origin}' is not allowed for this project.",
        )
        return None

    # ── 3. Tenant suspension check ────────────────────────────
    if tenant.is_suspended:
        log.warning(
            "Suspended tenant attempted WebSocket: tenant=%s project=%s",
            tenant.id, config.project_id,
        )
        await ws.close(
            code=4007,
            reason="Account suspended. Please check your billing or contact support.",
        )
        return None

    # ── 4. Daily token quota check ────────────────────────────
    quota = plan_limits.daily_token_quota(tenant_plan)
    if quota is not None:
        today_tokens = await repos.sessions.get_daily_token_usage(tenant.id)
        if today_tokens >= quota:
            log.warning(
                "Daily token quota exceeded: tenant=%s plan=%s quota=%d used=%d",
                tenant.id, tenant_plan, quota, today_tokens,
            )
            await ws.close(
                code=4008,
                reason=(
                    f"Daily token quota of {quota:,} tokens exceeded. "
                    "Quota resets at midnight UTC."
                ),
            )
            return None

    return config, api_key_id, tenant_plan
 
 
# ──────────────────────────────────────────────────────────────
# Forward loop  (outbox → WebSocket)
# ──────────────────────────────────────────────────────────────
 
async def _forward_loop(ws: WebSocket, state) -> None:
    while True:
        item = await state.outbox.get()
 
        if item is _OUTBOX_STOP:
            break
 
        kind, payload = item
 
        try:
            match kind:
                case "audio_pcm":
                    await ws.send_bytes(payload)
 
                case "session_ready":
                    await ws.send_text(
                        json.dumps({"type": "session_ready", **payload})
                    )
 
                case "assistant_text":
                    await ws.send_text(
                        json.dumps({"type": "assistant_text", "text": payload})
                    )
 
                case "user_transcript":
                    await ws.send_text(
                        json.dumps({"type": "user_transcript", "text": payload})
                    )
 
                case "tool_call":
                    await ws.send_text(
                        json.dumps({"type": "tool_call", **payload})
                    )
 
                case "turn_complete":
                    await ws.send_text(json.dumps({"type": "turn_complete"}))
 
                case "speaker_mode_updated":
                    await ws.send_text(
                        json.dumps({"type": "speaker_mode_updated", "enabled": payload})
                    )
 
                case "error":
                    await ws.send_text(
                        json.dumps({"type": "error", "message": payload})
                    )
 
                case "session_ended":
                    await ws.send_text(json.dumps({"type": "session_ended"}))
 
                case _:
                    log.warning(
                        "[%s] forward_loop: unknown frame kind %r",
                        state.session_id, kind,
                    )
 
        except WebSocketDisconnect:
            log.info("[%s] WebSocket disconnected during send", state.session_id)
            break
        except Exception as exc:
            log.error("[%s] forward_loop send error: %s", state.session_id, exc)
            break
 
 
# ──────────────────────────────────────────────────────────────
# Receive loop  (WebSocket → inbox)
# ──────────────────────────────────────────────────────────────
 
async def _receive_loop(ws: WebSocket, state) -> None:
    sid          = state.session_id
    voice_active = False
 
    try:
        while True:
            try:
                message = await ws.receive()
            except WebSocketDisconnect:
                log.info("[%s] client disconnected", sid)
                return
 
            if "bytes" in message and message["bytes"]:
                if not voice_active:
                    log.debug("[%s] audio bytes outside voice turn — ignoring", sid)
                    continue
                chunk: bytes = message["bytes"]
                state.touch()
                try:
                    state.inbox.put_nowait(InputFrame(FrameKind.AUDIO_CHUNK, chunk))
                except asyncio.QueueFull:
                    log.warning("[%s] inbox full — dropping audio chunk", sid)
                continue
 
            raw_text = message.get("text", "")
            if not raw_text:
                continue
 
            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({
                    "type":    "error",
                    "message": "Invalid JSON — could not parse message.",
                }))
                continue
 
            msg_type = payload.get("type")
 
            match msg_type:
 
                case "text_input":
                    text = (payload.get("text") or "").strip()
                    if not text:
                        continue
                    state.touch()
                    try:
                        state.inbox.put_nowait(InputFrame(FrameKind.TEXT, text))
                    except asyncio.QueueFull:
                        await _send_busy(ws)
 
                case "voice_start":
                    if voice_active:
                        log.debug("[%s] voice_start while already active — resetting", sid)
                    voice_active = True
                    state.touch()
                    try:
                        state.inbox.put_nowait(InputFrame(FrameKind.ACTIVITY_START, None))
                    except asyncio.QueueFull:
                        await _send_busy(ws)
 
                case "voice_end":
                    if not voice_active:
                        log.debug("[%s] voice_end without active turn — ignored", sid)
                        continue
                    voice_active = False
                    state.touch()
                    try:
                        state.inbox.put_nowait(InputFrame(FrameKind.ACTIVITY_END, None))
                    except asyncio.QueueFull:
                        await _send_busy(ws)
 
                case "set_speaker":
                    enabled = bool(payload.get("enabled", False))
                    try:
                        state.inbox.put_nowait(InputFrame(FrameKind.SET_SPEAKER, enabled))
                    except asyncio.QueueFull:
                        await _send_busy(ws)
 
                case "interrupt":
                    voice_active = False
                    log.debug("[%s] interrupt", sid)
 
                case "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
 
                case _:
                    await ws.send_text(json.dumps({
                        "type":    "error",
                        "message": f"Unknown message type: {msg_type!r}",
                    }))
 
    except Exception as exc:
        log.error("[%s] receive_loop unexpected error: %s", sid, exc)
 
 
async def _send_busy(ws: WebSocket) -> None:
    try:
        await ws.send_text(json.dumps({
            "type":    "error",
            "message": "Server busy — try again shortly.",
        }))
    except Exception:
        pass
 
 
# ──────────────────────────────────────────────────────────────
# WebSocket endpoint
# ──────────────────────────────────────────────────────────────
 
@router.websocket("/v1/chat")
async def ws_chat(
    ws:         WebSocket,
    api_key:    str | None = Query(default=None, alias="api_key"),
    token:      str | None = Query(default=None, alias="token"),
    session_id: str | None = Query(default=None, alias="session_id"),
) -> None:
    """
    Unified chat WebSocket endpoint.
 
    api_key     Long-lived publishable key.  Validated against DB; rate-limited.
    token       Single-use ephemeral token.  Redeemed atomically from Redis.
    session_id  Optional UUID to resume an existing Gemini session.
 
    Exactly one of api_key or token must be provided.
 
    New close codes vs previous version:
      4006  Origin not allowed (CORS enforcement)
      4007  Tenant account suspended
      4008  Daily token quota exceeded
    """
    await ws.accept()
 
    resolved = await _resolve_project(ws, api_key, token)
    if resolved is None:
        return
 
    project, api_key_id, tenant_plan = resolved
    sid     = session_id or str(uuid.uuid4())
    manager: SessionManager = ws.app.state.session_manager
 
    log.info(
        "[%s] WebSocket connected (project=%s tenant=%s plan=%s)",
        sid, project.project_id, project.tenant_id, tenant_plan,
    )
 
    try:
        state = await manager.get_or_create(sid, project, api_key_id=api_key_id)
    except MaxSessionsExceededError:
        await ws.close(
            code=4004,
            reason=(
                f"This project has reached its maximum concurrent session limit "
                f"of {project.max_concurrent_sessions}."
            ),
        )
        return
 
    try:
        await asyncio.gather(
            _receive_loop(ws, state),
            _forward_loop(ws, state),
        )
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log.error("[%s] unhandled WebSocket error: %s", sid, exc)
    finally:
        log.info("[%s] WebSocket handler exiting — Gemini session persists", sid)