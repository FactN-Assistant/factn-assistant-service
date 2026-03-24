"""
core/gemini_runner.py
─────────────────────
Everything that talks directly to the Google Gemini Live API.

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
    the full utterance duration in latency (typically 2–8 s).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from google import genai
from google.genai import types

from .schemas import ProjectConfig
from .session_state import SessionState, FrameKind, InputFrame, _OUTBOX_STOP
from .tool_executor import ToolExecutor

log = logging.getLogger("livechat.runner")


# ──────────────────────────────────────────────────────────────
# Config builder
# ──────────────────────────────────────────────────────────────

def build_gemini_config(project: ProjectConfig) -> types.LiveConnectConfig:
    """
    Translate a ProjectConfig into the Gemini SDK's LiveConnectConfig.

    Called once per session at open time — never at request time — so
    changing a project's config in the database only affects new sessions.
    """
    # Convert our ToolDefinition objects to Gemini's function_declarations
    # format (plain dicts that match the JSON Schema subset).
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
                disabled=True  # we drive VAD manually via activity_start/end
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
    """
    Execute each function_call in the batch, push results onto the outbox
    (so the client sees what happened), then send all responses back to
    Gemini in one send_tool_response call.
    """
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

        # Forward the tool call + result to the client for transparency
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
# Gemini response consumer
# ──────────────────────────────────────────────────────────────

async def _consume_gemini_responses(
    gsession:  Any,
    state:     SessionState,
    executor:  ToolExecutor,
) -> bool:
    """
    Drain Gemini's response stream for one turn.

    Returns
    -------
    True   turn_complete or interrupted → session still alive
    False  generator exhausted without turn_complete → Gemini closed session
    """
    sid = state.session_id

    try:
        async for response in gsession.receive():
            sc = response.server_content

            if sc is not None:
                # ── Audio PCM ─────────────────────────────────────────
                if sc.model_turn and sc.model_turn.parts:
                    for part in sc.model_turn.parts:
                        if part.inline_data and part.inline_data.data:
                            if await state.get_speaker_mode():
                                await state.send_outbox(
                                    ("audio_pcm", part.inline_data.data)
                                )

                # ── Output transcription (assistant speech → text) ─────
                if sc.output_transcription and sc.output_transcription.text:
                    await state.send_outbox(
                        ("assistant_text", sc.output_transcription.text)
                    )
                    log.debug(
                        "[%s] assistant_text: %r",
                        sid, sc.output_transcription.text,
                    )

                # ── Input transcription (user speech → text) ──────────
                if sc.input_transcription and sc.input_transcription.text:
                    await state.send_outbox(
                        ("user_transcript", sc.input_transcription.text)
                    )
                    log.debug(
                        "[%s] user_transcript: %r",
                        sid, sc.input_transcription.text,
                    )

                # ── Barge-in / interrupted ────────────────────────────
                if sc.interrupted:
                    log.info("[%s] Gemini interrupted generation", sid)
                    await state.send_outbox(("turn_complete", None))
                    return True

                # ── Normal turn complete ───────────────────────────────
                if sc.turn_complete:
                    await state.send_outbox(("turn_complete", None))
                    log.info("[%s] turn_complete", sid)
                    return True

            # ── Tool calls ────────────────────────────────────────────
            if response.tool_call:
                await _handle_tool_call(gsession, state, response.tool_call, executor)

        # Generator exhausted without turn_complete → Gemini closed the WS
        return False

    except asyncio.CancelledError:
        raise  # propagate so the runner can clean up

    except Exception as exc:
        log.error("[%s] error receiving from Gemini: %s", sid, exc)
        await state.send_outbox(("error", f"Gemini receive error: {exc}"))
        return False


# ──────────────────────────────────────────────────────────────
# Session runner  (one per session, lives inside a Task)
# ──────────────────────────────────────────────────────────────

async def session_runner(
    state:   SessionState,
    client:  genai.Client,
    project: ProjectConfig,
) -> None:
    """
    Opens ONE Gemini Live session per session_id and processes turns until:
      • a STOP frame is received
      • the inbox idles past session_ttl_seconds
      • Gemini closes the session
      • an unrecoverable error occurs

    Voice-turn concurrency
    ──────────────────────
    ACTIVITY_START received
        → send activity_start to Gemini
        → spawn _recv_task as a background Task
          (Gemini may start generating immediately)

    AUDIO_CHUNKs received rapidly
        → each chunk forwarded to Gemini with send_realtime_input
        → _recv_task drains Gemini's output concurrently

    ACTIVITY_END received
        → send activity_end to Gemini (utterance complete)
        → await _recv_task (drain any remaining output)

    Text turns are sequential: send → await _consume directly.
    """
    sid = state.session_id
    log.info("[%s] runner starting (project=%s)", sid, project.project_id)

    gemini_config = build_gemini_config(project)
    executor      = ToolExecutor(project)

    _recv_task: asyncio.Task | None = None

    try:
        async with client.aio.live.connect(
            model=project.gemini_model, config=gemini_config
        ) as gsession:

            log.info("[%s] Gemini session open", sid)
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
                # Block waiting for the next inbound frame.
                # If the inbox is idle longer than the project TTL, close.
                try:
                    frame: InputFrame = await asyncio.wait_for(
                        state.inbox.get(),
                        timeout=float(project.session_ttl_seconds),
                    )
                except TimeoutError:
                    log.info(
                        "[%s] idle timeout (%ss) — closing session",
                        sid, project.session_ttl_seconds,
                    )
                    break

                state.touch()

                # ── STOP ──────────────────────────────────────────────
                if frame.kind == FrameKind.STOP:
                    log.info("[%s] stop signal received", sid)
                    break

                # ── Speaker toggle ─────────────────────────────────────
                if frame.kind == FrameKind.SET_SPEAKER:
                    await state.set_speaker_mode(bool(frame.payload))
                    await state.send_outbox(("speaker_mode_updated", frame.payload))
                    log.info("[%s] speaker_mode → %s", sid, frame.payload)
                    continue

                # ── Voice: activity start ──────────────────────────────
                if frame.kind == FrameKind.ACTIVITY_START:
                    log.debug("[%s] activity_start → Gemini", sid)
                    try:
                        await gsession.send_realtime_input(
                            activity_start=types.ActivityStart()
                        )
                    except Exception as exc:
                        log.error("[%s] activity_start failed: %s", sid, exc)
                        await state.send_outbox(
                            ("error", f"activity_start failed: {exc}")
                        )
                        continue

                    # Spawn the response consumer concurrently so it runs
                    # while audio chunks are still being forwarded.
                    _recv_task = asyncio.create_task(
                        _consume_gemini_responses(gsession, state, executor),
                        name=f"recv-{sid}",
                    )
                    continue

                # ── Voice: forward PCM chunk ───────────────────────────
                if frame.kind == FrameKind.AUDIO_CHUNK:
                    try:
                        await gsession.send_realtime_input(
                            audio=types.Blob(
                                data=frame.payload,
                                mime_type="audio/pcm;rate=16000",
                            )
                        )
                    except Exception as exc:
                        # One dropped chunk is survivable
                        log.warning("[%s] audio chunk send failed: %s", sid, exc)
                    continue

                # ── Voice: activity end ────────────────────────────────
                if frame.kind == FrameKind.ACTIVITY_END:
                    log.debug("[%s] activity_end → Gemini", sid)
                    try:
                        await gsession.send_realtime_input(
                            activity_end=types.ActivityEnd()
                        )
                    except Exception as exc:
                        log.error("[%s] activity_end failed: %s", sid, exc)
                        await state.send_outbox(
                            ("error", f"activity_end failed: {exc}")
                        )

                    # Now drain the full response for this voice turn
                    if _recv_task is not None:
                        try:
                            session_alive = await _recv_task
                        except Exception as exc:
                            log.error("[%s] recv task raised: %s", sid, exc)
                            session_alive = False
                        _recv_task = None
                        if not session_alive:
                            log.warning(
                                "[%s] Gemini session ended during voice turn", sid
                            )
                            break
                    continue

                # ── Text input ─────────────────────────────────────────
                if frame.kind == FrameKind.TEXT:
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
                        await state.send_outbox(
                            ("error", f"Failed to send to Gemini: {exc}")
                        )
                        continue

                    # Text turns are sequential — no concurrent task needed
                    session_alive = await _consume_gemini_responses(
                        gsession, state, executor
                    )
                    if not session_alive:
                        log.warning(
                            "[%s] Gemini session ended after text turn", sid
                        )
                        break

    except asyncio.CancelledError:
        log.info("[%s] runner cancelled", sid)
        if _recv_task and not _recv_task.done():
            _recv_task.cancel()

    except Exception as exc:
        log.exception("[%s] runner fatal error: %s", sid, exc)
        await state.send_outbox(("error", f"Session error: {exc}"))

    finally:
        if _recv_task and not _recv_task.done():
            _recv_task.cancel()
        log.info("[%s] runner shutting down", sid)
        await state.send_outbox(("session_ended", None))
        await state.send_outbox(_OUTBOX_STOP)