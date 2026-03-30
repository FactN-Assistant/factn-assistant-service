"""
core/session_state.py
─────────────────────
Low-level session primitives: the queue-based inbox/outbox pair that
decouples the WebSocket receive loop, the Gemini runner, and the WebSocket
forward loop into three independent async coroutines.

                  ┌──────────────────┐
  WebSocket RX ──►│  inbox  (queue)  │──► session_runner ──► Gemini
                  └──────────────────┘
                  ┌──────────────────┐
        Gemini ──►│  outbox (queue)  │──► forward_loop ──► WebSocket TX
                  └──────────────────┘

Nothing here touches the database, authentication, or Gemini directly.

Week 6 additions
────────────────
SessionState now carries live counters that the session runner updates
throughout the session lifetime:
 
  turns           incremented after every turn_complete
  tool_calls      incremented every time a tool is executed
  input_tokens    accumulated from usage_metadata on Gemini responses
  output_tokens   accumulated from usage_metadata on Gemini responses
  started_at      set at SessionState creation — used for duration calc
 
These counters are read by session_runner in its finally block to write
the SessionDoc to MongoDB via SessionRepository.close_session().
 
The Gemini Live API surfaces token counts via message.usage_metadata
on server messages.  The field is a UsageMetadata object with:
  total_token_count         total tokens used in the session so far
  response_tokens_details   list of ModalityTokenCount per modality
 
Because usage_metadata is cumulative (total since session start) we
store the last seen total and compute a per-turn delta if needed.  For
session-level logging we record the final cumulative total.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from datetime import datetime, timezone

log = logging.getLogger("livechat.session_state")

INBOX_MAX_SIZE  = 512
OUTBOX_MAX_SIZE = 256


class FrameKind(StrEnum):
    TEXT           = "text"
    AUDIO_CHUNK    = "audio_chunk"
    ACTIVITY_START = "activity_start"
    ACTIVITY_END   = "activity_end"
    SET_SPEAKER    = "set_speaker"
    STOP           = "stop"

@dataclass(slots=True)
class InputFrame:
    kind:    FrameKind
    payload: Any

# Sentinel placed on the outbox when the runner is completely done.
# The forward_loop watches for this and exits.
_OUTBOX_STOP = object()


# ──────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────

@dataclass
class SessionState:
    """
    All mutable state for one active chat session.
 
    In addition to the inbox/outbox queues and speaker flag (unchanged
    from Weeks 2-3), this now holds live counters that are written to
    MongoDB as a SessionDoc when the session closes.
    """
    session_id:  str
    project_id:  str

    inbox:  asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=INBOX_MAX_SIZE)
    )
    outbox: asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=OUTBOX_MAX_SIZE)
    )

    _speaker: bool         = field(default=False, repr=False)
    _lock:    asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    last_active: float = field(default_factory=time.monotonic)
    worker_task: asyncio.Task | None = field(default=None, repr=False)

    # ── Session-level counters ─────────────────────────────────
    started_at:    datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    turns:         int = field(default=0)    # incremented per turn_complete
    tool_calls:    int = field(default=0)    # incremented per tool executed
    # Token counts from Gemini usage_metadata (cumulative totals)
    input_tokens:  int = field(default=0)
    output_tokens: int = field(default=0)
    # api_key_id is set by the WS endpoint after key resolution so the
    # session record can reference which key opened this session.
    api_key_id:    str = field(default="")

    # ── Speaker mode ──────────────────────────────────────────
    @property
    def speaker_mode(self) -> bool:
        return self._speaker

    async def set_speaker_mode(self, enabled: bool) -> None:
        async with self._lock:
            self._speaker = enabled

    async def get_speaker_mode(self) -> bool:
        async with self._lock:
            return self._speaker

    # ── Liveness ──────────────────────────────────────────────
    def touch(self) -> None:
        self.last_active = time.monotonic()

    def is_idle(self, ttl: float) -> bool:
        return (time.monotonic() - self.last_active) > ttl

    # ── Outbox ────────────────────────────────────────────────
    async def send_outbox(self, frame: tuple | object) -> None:
        try:
            self.outbox.put_nowait(frame)
        except asyncio.QueueFull:
            log.warning("[%s] outbox full — dropping frame", self.session_id)
 