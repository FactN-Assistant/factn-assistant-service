"""
api/chat.py
───────────
The single WebSocket endpoint that every client connects to.

Connection URL
──────────────
  wss://host/v1/chat?api_key=<key>[&session_id=<uuid>]

  api_key     Required.  In this phase it is read and logged but NOT
              validated — auth is added in Week 5-6.
              TODO (Week 5-6): resolve ProjectConfig from DB via api_key.

  session_id  Optional UUID.  Omit to start a new session.
              Provide the same UUID on reconnect to resume context
              (Gemini session stays alive as long as the runner is up).

Wire protocol
─────────────
Client → Server (JSON text frames)
  {"type": "text_input",  "text": "..."}
  {"type": "voice_start"}
  {"type": "voice_end"}
  {"type": "set_speaker", "enabled": bool}
  {"type": "interrupt"}
  {"type": "ping"}

Client → Server (binary frames)
  Raw 16-bit PCM at 16 kHz, little-endian.
  Must arrive only between voice_start and voice_end.

Server → Client (JSON text frames)
  {"type": "session_ready",        "session_id": "...", "speaker_mode": false, ...}
  {"type": "user_transcript",      "text": "..."}
  {"type": "assistant_text",       "text": "..."}
  {"type": "tool_call",            "tool": "...", "args": {...}, "result": {...}}
  {"type": "turn_complete"}
  {"type": "speaker_mode_updated", "enabled": bool}
  {"type": "error",                "code": "...", "message": "..."}
  {"type": "session_ended",        "reason": "..."}
  {"type": "pong"}

Server → Client (binary frames)
  Raw 16-bit PCM at 24 kHz — ONLY when speaker_mode is True.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from core.session_manager import SessionManager
from core.session_state import FrameKind, InputFrame, _OUTBOX_STOP

log = logging.getLogger("livechat.ws")

router = APIRouter()


# ──────────────────────────────────────────────────────────────
# Forward loop  (outbox → WebSocket)
# ──────────────────────────────────────────────────────────────

async def _forward_loop(ws: WebSocket, state) -> None:
    """
    Drain the session outbox and write frames to the client WebSocket.

    Runs concurrently with _receive_loop via asyncio.gather().
    Exits when it dequeues the _OUTBOX_STOP sentinel (runner shut down)
    or when the WebSocket disconnects mid-send.
    """
    while True:
        item = await state.outbox.get()

        if item is _OUTBOX_STOP:
            break  # runner is done — close the forward direction

        kind, payload = item

        try:
            match kind:
                case "audio_pcm":
                    # Raw PCM binary — no JSON wrapper
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
            log.info(
                "[%s] WebSocket disconnected during send", state.session_id
            )
            break

        except Exception as exc:
            log.error(
                "[%s] forward_loop send error: %s", state.session_id, exc
            )
            break


# ──────────────────────────────────────────────────────────────
# Receive loop  (WebSocket → inbox)
# ──────────────────────────────────────────────────────────────

async def _receive_loop(ws: WebSocket, state) -> None:
    """
    Read frames from the client WebSocket and push InputFrames onto the
    session inbox.

    Binary frames → AUDIO_CHUNK (only while voice_active is True)
    Text frames   → parsed as JSON control messages
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

            # ── Binary: raw PCM → enqueue immediately, no buffering ────
            if "bytes" in message and message["bytes"]:
                if not voice_active:
                    # Guard: ignore stray audio chunks outside a voice turn
                    log.debug(
                        "[%s] audio bytes outside voice turn — ignoring", sid
                    )
                    continue

                chunk: bytes = message["bytes"]
                state.touch()
                try:
                    state.inbox.put_nowait(
                        InputFrame(FrameKind.AUDIO_CHUNK, chunk)
                    )
                except asyncio.QueueFull:
                    # Drop the chunk rather than blocking the receive loop
                    log.warning("[%s] inbox full — dropping audio chunk", sid)
                continue

            # ── Text: JSON control messages ────────────────────────────
            raw_text = message.get("text", "")
            if not raw_text:
                continue

            try:
                payload = json.loads(raw_text)
            except json.JSONDecodeError:
                await ws.send_text(
                    json.dumps({
                        "type":    "error",
                        "message": "Invalid JSON — could not parse message.",
                    })
                )
                continue

            msg_type = payload.get("type")

            match msg_type:

                # ── Text turn ─────────────────────────────────────────
                case "text_input":
                    text = (payload.get("text") or "").strip()
                    if not text:
                        continue
                    state.touch()
                    try:
                        state.inbox.put_nowait(InputFrame(FrameKind.TEXT, text))
                    except asyncio.QueueFull:
                        await _send_busy(ws)

                # ── Voice start ───────────────────────────────────────
                case "voice_start":
                    if voice_active:
                        log.debug(
                            "[%s] voice_start while already active — resetting",
                            sid,
                        )
                    voice_active = True
                    state.touch()
                    log.debug("[%s] voice_start", sid)
                    try:
                        state.inbox.put_nowait(
                            InputFrame(FrameKind.ACTIVITY_START, None)
                        )
                    except asyncio.QueueFull:
                        await _send_busy(ws)

                # ── Voice end ─────────────────────────────────────────
                case "voice_end":
                    if not voice_active:
                        log.debug(
                            "[%s] voice_end without active turn — ignored", sid
                        )
                        continue
                    voice_active = False
                    state.touch()
                    log.debug("[%s] voice_end", sid)
                    try:
                        state.inbox.put_nowait(
                            InputFrame(FrameKind.ACTIVITY_END, None)
                        )
                    except asyncio.QueueFull:
                        await _send_busy(ws)

                # ── Speaker toggle ────────────────────────────────────
                case "set_speaker":
                    enabled = bool(payload.get("enabled", False))
                    try:
                        state.inbox.put_nowait(
                            InputFrame(FrameKind.SET_SPEAKER, enabled)
                        )
                    except asyncio.QueueFull:
                        await _send_busy(ws)

                # ── Interrupt ─────────────────────────────────────────
                case "interrupt":
                    # Reset voice state on the receive side immediately.
                    # No stray audio chunks will be forwarded because
                    # voice_active is now False.
                    voice_active = False
                    log.debug("[%s] interrupt", sid)
                    # NOTE: a future enhancement could also cancel _recv_task
                    # in the runner via a dedicated INTERRUPT FrameKind.

                # ── Keepalive ─────────────────────────────────────────
                case "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))

                # ── Unknown ───────────────────────────────────────────
                case _:
                    await ws.send_text(
                        json.dumps({
                            "type":    "error",
                            "message": f"Unknown message type: {msg_type!r}",
                        })
                    )

    except Exception as exc:
        log.error("[%s] receive_loop unexpected error: %s", sid, exc)


async def _send_busy(ws: WebSocket) -> None:
    try:
        await ws.send_text(
            json.dumps({
                "type":    "error",
                "message": "Server busy — try again shortly.",
            })
        )
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
    # Injected via app.state in main.py
) -> None:
    """
    Unified chat WebSocket endpoint.

    api_key     Identifies which project config to load.
                ⚠ NOT VALIDATED YET — auth added in Week 5-6.
                TODO: resolve ProjectConfig from DB using api_key.

    session_id  Optional UUID to resume an existing session.
                Omit (or pass a new UUID) for a brand-new conversation.
    """
    await ws.accept()

    sid = session_id or str(uuid.uuid4())
    log.info("[%s] WebSocket connected (api_key=%s)", sid, api_key)

    manager: SessionManager = ws.app.state.session_manager
    project = ws.app.state.demo_project_config  # TODO: resolve from DB via api_key

    state = await manager.get_or_create(sid, project)

    try:
        # Run receive and forward loops concurrently.
        # When either exits (disconnect or session end) gather() cancels the other.
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