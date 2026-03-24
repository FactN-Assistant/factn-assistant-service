"""
core/session_manager.py
───────────────────────
Lifecycle management for all active sessions on this server node.

get_or_create(session_id, project)
    Returns an existing SessionState (session resumption) or creates a
    new one and spawns a session_runner Task.

    In the multi-tenant final version (Week 5-6) the project argument will
    be resolved from the database using the caller's API key before this
    method is called.  The signature here is already prepared for that.

Horizontal scaling note
───────────────────────
Each backend node has its own in-process SessionManager.  NGINX sticky
sessions route WebSocket reconnections to the same node.  Cross-node
events (interrupt, speaker toggle) will be delivered via Redis Pub/Sub
in the scaling sprint.  For now a single node is assumed.
"""

from __future__ import annotations

import asyncio
import logging

from google import genai

from .gemini_runner import session_runner
from .schemas import ProjectConfig
from .session_state import FrameKind, InputFrame, SessionState

log = logging.getLogger("livechat.session_manager")

# Stale-session sweep interval (seconds)
_CLEANUP_INTERVAL = 60


class SessionManager:
    def __init__(self, gemini_client: genai.Client) -> None:
        self._client:   genai.Client           = gemini_client
        self._sessions: dict[str, SessionState] = {}
        self._lock:     asyncio.Lock            = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    # ── Lifecycle ─────────────────────────────────────────────

    async def start(self) -> None:
        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(), name="session-cleanup"
        )
        log.info("SessionManager started")

    async def stop(self) -> None:
        log.info("SessionManager stopping — terminating all sessions")
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        async with self._lock:
            sids = list(self._sessions.keys())
        for sid in sids:
            await self._terminate(sid)

        log.info("SessionManager stopped")

    # ── Session access ────────────────────────────────────────

    async def get_or_create(
        self,
        session_id: str,
        project:    ProjectConfig,
    ) -> SessionState:
        """
        Return an existing session (resume) or open a fresh Gemini session.

        Thread-safe: the lock ensures only one coroutine creates a session
        for a given session_id even under concurrent reconnect storms.
        """
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is not None:
                state.touch()
                log.info("[%s] resuming existing session", session_id)
                return state

            state = SessionState(
                session_id=session_id,
                project_id=project.project_id,
            )
            self._sessions[session_id] = state

        # Spawn the runner outside the lock — it may take a moment to open
        # the Gemini WebSocket and we don't want to hold the lock that long.
        task = asyncio.create_task(
            session_runner(state, self._client, project),
            name=f"runner-{session_id}",
        )
        state.worker_task = task

        # Auto-remove from registry when the runner finishes
        task.add_done_callback(
            lambda _: asyncio.create_task(self._on_runner_done(session_id))
        )

        log.info(
            "[%s] new session created (project=%s)",
            session_id, project.project_id,
        )
        return state

    # ── Metrics ───────────────────────────────────────────────

    async def active_session_count(self) -> int:
        async with self._lock:
            return len(self._sessions)

    async def get_active_session_ids(self) -> list[str]:
        async with self._lock:
            return list(self._sessions.keys())

    # ── Internal ──────────────────────────────────────────────

    async def _on_runner_done(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)
        log.info("[%s] session removed from registry", session_id)

    async def _terminate(self, session_id: str) -> None:
        """
        Gracefully shut down a session: signal the runner via STOP frame,
        then cancel the task if it hasn't exited within a short grace period.
        """
        async with self._lock:
            state = self._sessions.pop(session_id, None)
        if state is None:
            return

        try:
            state.inbox.put_nowait(InputFrame(kind=FrameKind.STOP, payload=None))
        except asyncio.QueueFull:
            pass

        if state.worker_task and not state.worker_task.done():
            # Give the task 2 s to handle the STOP frame before cancelling
            try:
                await asyncio.wait_for(
                    asyncio.shield(state.worker_task), timeout=2.0
                )
            except (asyncio.TimeoutError, asyncio.CancelledError):
                state.worker_task.cancel()
                try:
                    await state.worker_task
                except (asyncio.CancelledError, Exception):
                    pass

        log.info("[%s] session terminated", session_id)

    async def _cleanup_loop(self) -> None:
        """Periodically sweep and terminate idle sessions."""
        while True:
            await asyncio.sleep(_CLEANUP_INTERVAL)
            async with self._lock:
                stale = [
                    sid
                    for sid, s in self._sessions.items()
                    if s.is_idle(300)  # fallback TTL — overridden per project in runner
                ]
            for sid in stale:
                log.info("[%s] cleaning up idle session", sid)
                await self._terminate(sid)