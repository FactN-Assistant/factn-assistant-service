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

Connection URL
──────────────
  wss://host/v1/chat?api_key=<key>[&session_id=<uuid>]

Close codes used
────────────────
  4001  Missing or invalid API key
  4002  Project not found or inactive
  4003  Rate limit exceeded
  4004  Max concurrent sessions reached for this project
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from core.schemas import ProjectConfig
from core.session_manager import SessionManager
from core.session_state import FrameKind, InputFrame, _OUTBOX_STOP
from db.redis_client import RedisClient
from repositories import Repositories

log = logging.getLogger("livechat.ws")

router = APIRouter()


# ──────────────────────────────────────────────────────────────
# Handshake — resolve project config from API key
# ──────────────────────────────────────────────────────────────

async def _resolve_project(
    ws:      WebSocket,
    api_key: str | None,
) -> ProjectConfig | None:
    """
    Validate the API key and return the resolved ProjectConfig.

    Returns None and closes the WebSocket (with an explanatory code) if
    anything fails.  Callers must return immediately on None.
    """
    repos:  Repositories = ws.app.state.repos
    redis:  RedisClient  = ws.app.state.redis

    # ── 1. Key provided? ──────────────────────────────────────
    if not api_key:
        await ws.close(code=4001, reason="Missing api_key query parameter.")
        return None

    # ── 2. Lookup key in DB ───────────────────────────────────
    key_doc = await repos.api_keys.get_by_raw_key(api_key)
    if key_doc is None:
        log.warning("Invalid or revoked API key: prefix=%s", api_key[:12])
        await ws.close(code=4001, reason="Invalid or revoked API key.")
        return None

    # ── 3. Rate limit check ───────────────────────────────────
    allowed, count = await redis.check_and_increment_rate_limit(
        key_prefix = key_doc.key_prefix,
        limit      = key_doc.rate_limit_rpm,
        window_s   = 60,
    )
    if not allowed:
        log.warning(
            "Rate limit exceeded: key=%s count=%d limit=%d",
            key_doc.key_prefix, count, key_doc.rate_limit_rpm,
        )
        await ws.close(
            code=4003,
            reason=f"Rate limit exceeded ({key_doc.rate_limit_rpm} req/min).",
        )
        return None

    # ── 4. Load project config (Redis-cached) ─────────────────
    config = await repos.projects.get_config_for_key(key_doc)
    if config is None:
        log.warning(
            "Project not found or inactive: project_id=%s", key_doc.project_id
        )
        await ws.close(code=4002, reason="Project not found or inactive.")
        return None

    return config


# ──────────────────────────────────────────────────────────────
# Forward loop  (outbox → WebSocket)
# ──────────────────────────────────────────────────────────────

async def _forward_loop(ws: WebSocket, state) -> None:
    """
    Drain the session outbox and write frames to the client WebSocket.
    Exits on _OUTBOX_STOP sentinel or WebSocket disconnect.
    """
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
    """
    Read frames from the client WebSocket and push InputFrames onto inbox.

    Binary frames  → AUDIO_CHUNK (only while voice_active)
    Text frames    → parsed JSON control messages
    """
    sid          = state.session_id
    voice_active = False

    try:
        while True:
            try:
                message = await ws.receive()
            except WebSocketDisconnect:
                log.info("[%s] client disconnected", sid)
                return

            # ── Binary: raw PCM ────────────────────────────────
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

            # ── Text: JSON control messages ────────────────────
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
    session_id: str | None = Query(default=None, alias="session_id"),
) -> None:
    """
    Unified chat WebSocket endpoint.

    api_key     Identifies the project.  Validated against DB; rate-limited.
    session_id  Optional UUID to resume an existing Gemini session.
                Omit (or pass a new UUID) for a brand-new conversation.
    """
    await ws.accept()

    # ── Resolve project from API key ───────────────────────────
    project = await _resolve_project(ws, api_key)
    if project is None:
        return  # WebSocket already closed with a 4xxx code

    sid     = session_id or str(uuid.uuid4())
    manager: SessionManager = ws.app.state.session_manager

    log.info(
        "[%s] WebSocket connected (project=%s tenant=%s)",
        sid, project.project_id, project.tenant_id,
    )

    state = await manager.get_or_create(sid, project)

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