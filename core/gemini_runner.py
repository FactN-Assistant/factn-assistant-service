"""
core/gemini_runner.py
─────────────────────
Gemini Live API session runner.

Key design decisions
────────────────────
1.  build_gemini_config()
    Constructs a types.LiveConnectConfig from the project's schema at
    session-open time.  This means different projects get different system
    prompts, voices, and tool sets without restarting any server process.

2.  _consume_gemini_responses()
    Drains Gemini's async generator until turn_complete (or session close).
    Runs as either:
      • an awaited call  — for text turns (sequential, keeps ordering simple)
      • a background Task — for voice turns (concurrent with PCM forwarding)

3.  session_runner()
    The main loop per session.  It pops InputFrames from the inbox and
    drives Gemini.  Voice-turn concurrency model:

        ACTIVITY_START  → send activity_start to Gemini
                          spawn _recv_task (concurrent response consumer)

        AUDIO_CHUNK(s)  → forward each PCM chunk to Gemini immediately
                          _recv_task drains Gemini in parallel

        ACTIVITY_END    → send activity_end to Gemini
                          await _recv_task to finish

    This pipelines the user's speaking with Gemini's generation, saving
    the full utterance duration in latency (typically 2-8 s).

Week 6 additions
────────────────
1.  Token usage tracking
    Gemini Live API surfaces cumulative token counts via
    message.usage_metadata on server messages (not on every message —
    the server sends them periodically).  Fields used:
      usage.total_token_count              — total tokens so far
      usage.prompt_token_count             — input side
      usage.response_tokens_details        — per-modality output breakdown
 
    We capture the LAST seen values from usage_metadata and store them
    on the SessionState so the finally block can write accurate totals
    to MongoDB.
 
2.  Session record persistence
    session_runner now accepts an optional SessionRepository and calls
    session_repo.close_session() in its finally block.  This is the fix
    for sessions never being written to the sessions collection.
 
    The session_repo is None-safe — if it hasn't been wired in (e.g. in
    tests) the write is skipped silently.
    
New Changes for plans
──────────────────────────────
  session_runner()  now accepts project_webhook_url and project_webhook_secret
                    parameters and calls session_events.deliver_session_started()
                    after the Gemini session opens and
                    session_events.deliver_session_closed() in the finally block
                    alongside the MongoDB session record write.
 
  Both deliveries are fire-and-forget (asyncio.create_task inside
  session_events) so they never block the session lifecycle path.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from datetime import datetime, timezone

from google import genai
from google.genai import types

from .schemas import ProjectConfig
from .session_events import deliver_session_started, deliver_session_closed
from .session_state import SessionState, FrameKind, InputFrame, _OUTBOX_STOP
from .tool_executor import ToolExecutor

log = logging.getLogger("livechat.runner")


# ──────────────────────────────────────────────────────────────
# Config builder
# ──────────────────────────────────────────────────────────────

def build_gemini_config(project: ProjectConfig) -> types.LiveConnectConfig:
    """Translate a ProjectConfig into the Gemini SDK's LiveConnectConfig."""
    function_declarations = [
        {
            "name":        t.name,
            "description": t.description,
            "parameters":  t.parameters,
        }
        for t in project.tools
    ]

    config_kwargs: dict[str, Any] = dict(
        response_modalities=["AUDIO"],
        system_instruction=project.system_prompt,
        output_audio_transcription={},
        input_audio_transcription={},
        thinking_config=types.ThinkingConfig(
            thinking_budget=0,
            include_thoughts=False,
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=project.voice_config.voice_name,
                )
            )
        ),
        temperature=0.7,
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(
                disabled=True
            )
        ),
    )

    if function_declarations:
        config_kwargs["tools"] = [{"function_declarations": function_declarations}]

    return types.LiveConnectConfig(**config_kwargs)


# ──────────────────────────────────────────────────────────────
# Tool call handler
# ──────────────────────────────────────────────────────────────

async def _handle_tool_call(
    gsession:  Any,
    state:     SessionState,
    tool_call: Any,
    executor:  ToolExecutor,
) -> None:
    sid = state.session_id
    function_responses: list[types.FunctionResponse] = []

    for fc in tool_call.function_calls:
        args = dict(fc.args) if fc.args else {}

        result = await executor.execute(
            name=fc.name,
            args=args,
            session_id=sid,
            call_id=getattr(fc, "id", ""),
        )

        state.tool_calls += 1

        await state.send_outbox((
            "tool_call",
            {
                "tool":    fc.name,
                "args":    args,
                "result":  result,
                "call_id": getattr(fc, "id", ""),
            },
        ))

        function_responses.append(
            types.FunctionResponse(
                id=getattr(fc, "id", ""),
                name=fc.name,
                response={"result": result},
            )
        )

    try:
        await gsession.send_tool_response(function_responses=function_responses)
    except Exception as exc:
        log.error("[%s] failed to send tool response: %s", sid, exc)
        await state.send_outbox(("error", f"Tool response failed: {exc}"))


# ──────────────────────────────────────────────────────────────
# Usage metadata capture
# ──────────────────────────────────────────────────────────────

def _capture_usage(state: SessionState, usage: Any) -> None:
    if usage is None:
        return
 
    total    = getattr(usage, "total_token_count",      0) or 0
    prompt   = getattr(usage, "prompt_token_count",     0) or 0
    response = getattr(usage, "candidates_token_count", 0) or 0
 
    if total and not prompt and not response:
        state.output_tokens = total
    else:
        state.input_tokens  = prompt
        state.output_tokens = response
 
    log.debug(
        "[%s] usage_metadata: total=%d prompt=%d response=%d",
        state.session_id, total, prompt, response,
    )
 
 
# ──────────────────────────────────────────────────────────────
# Gemini response consumer
# ──────────────────────────────────────────────────────────────

async def _consume_gemini_responses(
    gsession:  Any,
    state:     SessionState,
    executor:  ToolExecutor,
) -> bool:
    sid = state.session_id

    try:
        async for response in gsession.receive():
 
            if response.usage_metadata:
                _capture_usage(state, response.usage_metadata)
 
            sc = response.server_content

            if sc is not None:
                if sc.model_turn and sc.model_turn.parts:
                    for part in sc.model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            if await state.get_speaker_mode():
                                await state.send_outbox(
                                    ("audio_pcm", part.inline_data.data)
                                )

                if sc.output_transcription and sc.output_transcription.text:
                    await state.send_outbox(
                        ("assistant_text", sc.output_transcription.text)
                    )

                if sc.input_transcription and sc.input_transcription.text:
                    await state.send_outbox(
                        ("user_transcript", sc.input_transcription.text)
                    )

                if sc.interrupted:
                    log.info("[%s] Gemini interrupted generation", sid)
                    state.turns += 1
                    await state.send_outbox(("turn_complete", None))
                    return True

                if sc.turn_complete:
                    state.turns += 1
                    await state.send_outbox(("turn_complete", None))
                    log.info("[%s] turn_complete (turns=%d)", sid, state.turns)
                    return True

            if response.tool_call:
                await _handle_tool_call(gsession, state, response.tool_call, executor)

        return False

    except asyncio.CancelledError:
        raise

    except Exception as exc:
        log.error("[%s] error receiving from Gemini: %s", sid, exc)
        await state.send_outbox(("error", f"Gemini receive error: {exc}"))
        return False


# ──────────────────────────────────────────────────────────────
# Session runner
# ──────────────────────────────────────────────────────────────

async def session_runner(
    state:        SessionState,
    client:       genai.Client,
    project:      ProjectConfig,
    session_repo: Any = None,   # SessionRepository | None
) -> None:
    """
    Opens ONE Gemini Live session and processes turns until stopped.
 
    Delivers session lifecycle webhooks to project.webhook_url on open
    and close.  Both deliveries are fire-and-forget and never block the
    session lifecycle path.
    """
    sid           = state.session_id
    status        = "closed"
    error_message: str | None = None

    # Pull webhook config from the project config (stored on ProjectConfig
    # via ProjectDoc.to_project_config — added in this release).
    webhook_url    = getattr(project, "webhook_url",    None)
    webhook_secret = getattr(project, "webhook_secret", None)
 
    log.info("[%s] runner starting (project=%s)", sid, project.project_id)

    gemini_config = build_gemini_config(project)
    executor      = ToolExecutor(project)

    _recv_task: asyncio.Task | None = None

    try:
        async with client.aio.live.connect(
            model=project.gemini_model, config=gemini_config
        ) as gsession:

            log.info("[%s] Gemini session open", sid)
 
            # Notify customer that session is live
            await deliver_session_started(
                webhook_url    = webhook_url,
                webhook_secret = webhook_secret,
                session_id     = sid,
                project_id     = project.project_id,
                tenant_id      = project.tenant_id,
                started_at     = state.started_at,
            )
 
            await state.send_outbox((
                "session_ready",
                {
                    "session_id":   sid,
                    "speaker_mode": state.speaker_mode,
                    "project_id":   project.project_id,
                    "project_name": project.name,
                },
            ))

            while True:
                try:
                    frame: InputFrame = await asyncio.wait_for(
                        state.inbox.get(),
                        timeout=float(project.session_ttl_seconds),
                    )
                except TimeoutError:
                    log.info("[%s] idle timeout — closing session", sid)
                    status = "timeout"
                    break

                state.touch()

                if frame.kind == FrameKind.STOP:
                    log.info("[%s] stop signal received", sid)
                    break

                if frame.kind == FrameKind.SET_SPEAKER:
                    await state.set_speaker_mode(bool(frame.payload))
                    await state.send_outbox(("speaker_mode_updated", frame.payload))
                    log.info("[%s] speaker_mode → %s", sid, frame.payload)
                    continue

                if frame.kind == FrameKind.ACTIVITY_START:
                    log.debug("[%s] activity_start → Gemini", sid)

                    # Cancel any lingering recv task from a prior voice turn
                    if _recv_task is not None and not _recv_task.done():
                        _recv_task.cancel()
                        try:
                            await _recv_task
                        except (asyncio.CancelledError, Exception):
                            pass
                        _recv_task = None

                    try:
                        await gsession.send_realtime_input(
                            activity_start=types.ActivityStart()
                        )
                    except Exception as exc:
                        log.error("[%s] activity_start failed: %s", sid, exc)
                        await state.send_outbox(("error", f"activity_start failed: {exc}"))
                        continue

                    _recv_task = asyncio.create_task(
                        _consume_gemini_responses(gsession, state, executor),
                        name=f"recv-{sid}",
                    )
                    continue

                if frame.kind == FrameKind.AUDIO_CHUNK:
                    try:
                        await gsession.send_realtime_input(
                            audio=types.Blob(
                                data=frame.payload,
                                mime_type="audio/pcm;rate=16000",
                            )
                        )
                    except Exception as exc:
                        log.warning("[%s] audio chunk send failed: %s", sid, exc)
                    continue

                if frame.kind == FrameKind.ACTIVITY_END:
                    log.debug("[%s] activity_end → Gemini", sid)
                    try:
                        await gsession.send_realtime_input(
                            activity_end=types.ActivityEnd()
                        )
                    except Exception as exc:
                        log.error("[%s] activity_end failed: %s", sid, exc)
                        await state.send_outbox(("error", f"activity_end failed: {exc}"))

                    if _recv_task is not None:
                        try:
                            session_alive = await _recv_task
                        except Exception as exc:
                            log.error("[%s] recv task raised: %s", sid, exc)
                            session_alive = False
                        _recv_task = None
                        if not session_alive:
                            log.warning("[%s] Gemini session ended during voice turn", sid)
                            break
                    continue

                if frame.kind == FrameKind.TEXT:
                    # Drop text input while a voice turn is active.
                    # Cancelling the in-flight voice receiver here can leave
                    # stale Gemini events that break the next normal text turn.
                    if _recv_task is not None and not _recv_task.done():
                        log.debug("[%s] dropping text input — voice turn active", sid)
                        continue

                    try:
                        await gsession.send_client_content(
                            turns=[
                                types.Content(
                                    role="user",
                                    parts=[types.Part(text=frame.payload)],
                                )
                            ],
                            turn_complete=True,
                        )
                    except Exception as exc:
                        log.error("[%s] text send failed: %s", sid, exc)
                        await state.send_outbox(("error", f"Failed to send to Gemini: {exc}"))
                        continue

                    session_alive = await _consume_gemini_responses(
                        gsession, state, executor
                    )
                    if not session_alive:
                        log.warning("[%s] Gemini session ended after text turn", sid)
                        break

    except asyncio.CancelledError:
        log.info("[%s] runner cancelled", sid)
        status = "closed"
        if _recv_task and not _recv_task.done():
            _recv_task.cancel()

    except Exception as exc:
        log.exception("[%s] runner fatal error: %s", sid, exc)
        status = "error"
        error_message = str(exc)
        await state.send_outbox(("error", f"Session error: {exc}"))

    finally:
        if _recv_task and not _recv_task.done():
            _recv_task.cancel()

        ended_at = datetime.now(timezone.utc)
        duration = (ended_at - state.started_at).total_seconds()

        # ── Persist session record ─────────────────────────────
        if session_repo is not None:
            try:
                await session_repo.close_session(
                    session_id    = sid,
                    project_id    = project.project_id,
                    tenant_id     = project.tenant_id,
                    started_at    = state.started_at,
                    status        = status,
                    turns         = state.turns,
                    tool_calls    = state.tool_calls,
                    input_tokens  = state.input_tokens,
                    output_tokens = state.output_tokens,
                    error_message = error_message,
                    api_key_id    = state.api_key_id,
                )
            except Exception as exc:
                log.error("[%s] failed to write session record: %s", sid, exc)

        # ── Deliver session closed webhook ─────────────────────
        await deliver_session_closed(
            webhook_url      = webhook_url,
            webhook_secret   = webhook_secret,
            session_id       = sid,
            project_id       = project.project_id,
            tenant_id        = project.tenant_id,
            status           = status,
            started_at       = state.started_at,
            ended_at         = ended_at,
            duration_seconds = round(duration, 2),
            turns            = state.turns,
            tool_calls       = state.tool_calls,
            input_tokens     = state.input_tokens,
            output_tokens    = state.output_tokens,
            error_message    = error_message,
        )

        log.info(
            "[%s] runner shutdown (status=%s turns=%d input_tokens=%d output_tokens=%d)",
            sid, status, state.turns, state.input_tokens, state.output_tokens,
        )
        await state.send_outbox(("session_ended", None))
        await state.send_outbox(_OUTBOX_STOP)