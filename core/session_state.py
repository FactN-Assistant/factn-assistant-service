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
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

log = logging.getLogger("livechat.session_state")

# Configurable via environment in main.py but kept as module-level defaults
# so unit tests don't need a full app context.
INBOX_MAX_SIZE  = 512   # audio chunks arrive fast — keep this large
OUTBOX_MAX_SIZE = 256


# ──────────────────────────────────────────────────────────────
# Frame types
# ──────────────────────────────────────────────────────────────

class FrameKind(StrEnum):
    TEXT           = "text"
    AUDIO_CHUNK    = "audio_chunk"    # raw 16-bit PCM at 16 kHz
    ACTIVITY_START = "activity_start" # client opened mic
    ACTIVITY_END   = "activity_end"   # client closed mic
    SET_SPEAKER    = "set_speaker"    # toggle audio PCM output
    STOP           = "stop"           # shut down the runner cleanly


@dataclass(slots=True)
class InputFrame:
    """One message on the inbox queue."""
    kind:    FrameKind
    payload: Any  # str | bytes | bool | None


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

    session_id  globally-unique identifier (UUID4 string)
    project_id  which project config this session belongs to — used for
                metrics, audit logging, and future DB lookups
    """
    session_id:  str
    project_id:  str

    inbox:  asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=INBOX_MAX_SIZE)
    )
    outbox: asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=OUTBOX_MAX_SIZE)
    )

    # Speaker mode: when True the runner pushes raw PCM onto the outbox.
    # Guarded by an asyncio.Lock so the runner and the WS handler can
    # toggle it safely from different coroutines.
    _speaker: bool          = field(default=False, repr=False)
    _lock:    asyncio.Lock  = field(default_factory=asyncio.Lock, repr=False)

    last_active: float                = field(default_factory=time.monotonic)
    worker_task: asyncio.Task | None  = field(default=None, repr=False)

    # ── Speaker mode accessors ─────────────────────────────────

    @property
    def speaker_mode(self) -> bool:
        return self._speaker

    async def set_speaker_mode(self, enabled: bool) -> None:
        async with self._lock:
            self._speaker = enabled

    async def get_speaker_mode(self) -> bool:
        async with self._lock:
            return self._speaker

    # ── Liveness helpers ───────────────────────────────────────

    def touch(self) -> None:
        """Update last-activity timestamp on any inbound frame."""
        self.last_active = time.monotonic()

    def is_idle(self, ttl: float) -> bool:
        return (time.monotonic() - self.last_active) > ttl

    # ── Outbox helper ──────────────────────────────────────────

    async def send_outbox(self, frame: tuple | object) -> None:
        """
        Non-blocking put onto the outbox.  Drops the frame if the queue is
        full rather than back-pressuring the Gemini receive loop.
        """
        try:
            self.outbox.put_nowait(frame)
        except asyncio.QueueFull:
            log.warning("[%s] outbox full — dropping frame", self.session_id)