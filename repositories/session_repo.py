"""
repositories/session_repo.py
─────────────────────────────
SessionRepository — writes and queries the `sessions` collection.

Write pattern
─────────────
Sessions are written ONCE when a Gemini session closes (append-only).
The session_runner calls session_repo.close_session() in its finally block.
There are no update operations — if something goes wrong mid-session the
record is written with status="error".

Read pattern
────────────
Analytics queries:  list_for_project(), usage_summary().
These are called from REST endpoints (Week 10) and the dashboard.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from core.documents import SessionDoc
from db.mongo import MongoDB

log = logging.getLogger("livechat.repo.session")


class SessionRepository:
    def __init__(self, mongodb: MongoDB) -> None:
        self._col = mongodb.sessions

    # ── Write (called by session runner on close) ─────────────

    async def close_session(
        self,
        session_id:    str,
        project_id:    str,
        tenant_id:     str,
        started_at:    datetime,
        status:        str = "closed",
        turns:         int = 0,
        tool_calls:    int = 0,
        error_message: str | None = None,
        api_key_id:    str = "",
    ) -> SessionDoc:
        """
        Persist a session record on close.

        Called from session_runner's finally block so it always runs —
        even on unexpected errors.  Non-critical: if this write fails the
        session still closed cleanly, we just lose the analytics record.
        """
        ended_at = datetime.now(timezone.utc)
        duration = (ended_at - started_at).total_seconds()

        doc = SessionDoc(
            _id              = session_id,
            project_id       = project_id,
            tenant_id        = tenant_id,
            api_key_id       = api_key_id,
            status           = status,         # type: ignore[arg-type]
            started_at       = started_at,
            ended_at         = ended_at,
            duration_seconds = round(duration, 2),
            turns            = turns,
            tool_calls       = tool_calls,
            error_message    = error_message,
        )
        try:
            await self._col.insert_one(doc.to_mongo())
            log.info(
                "Session record written: %s (status=%s turns=%d duration=%.1fs)",
                session_id, status, turns, duration,
            )
        except Exception as exc:
            # Never crash the shutdown path over an analytics write
            log.error("Failed to write session record %s: %s", session_id, exc)

        return doc

    # ── Read ──────────────────────────────────────────────────

    async def list_for_project(
        self,
        project_id: str,
        limit:      int = 50,
        skip:       int = 0,
    ) -> list[SessionDoc]:
        cursor = (
            self._col.find({"project_id": project_id})
            .sort("started_at", -1)
            .skip(skip)
            .limit(limit)
        )
        return [SessionDoc.from_mongo(d) async for d in cursor]

    async def get_by_id(
        self, session_id: str, project_id: str
    ) -> SessionDoc | None:
        raw = await self._col.find_one(
            {"_id": session_id, "project_id": project_id}
        )
        return SessionDoc.from_mongo(raw) if raw else None

    async def usage_summary(
        self,
        project_id: str,
        since:      datetime,
    ) -> dict:
        """
        Aggregate basic usage metrics for a project since a given datetime.
        Returns a plain dict — used by the /usage REST endpoint (Week 10).
        """
        pipeline = [
            {"$match": {
                "project_id": project_id,
                "started_at": {"$gte": since},
            }},
            {"$group": {
                "_id":              None,
                "total_sessions":   {"$sum": 1},
                "total_turns":      {"$sum": "$turns"},
                "total_tool_calls": {"$sum": "$tool_calls"},
                "total_duration_s": {"$sum": "$duration_seconds"},
                "error_count":      {"$sum": {
                    "$cond": [{"$eq": ["$status", "error"]}, 1, 0]
                }},
            }},
        ]
        results = await self._col.aggregate(pipeline).to_list(length=1)
        if not results:
            return {
                "total_sessions":   0,
                "total_turns":      0,
                "total_tool_calls": 0,
                "total_duration_s": 0.0,
                "error_count":      0,
            }
        r = results[0]
        r.pop("_id", None)
        return r