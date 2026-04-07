"""
repositories/session_repo.py
─────────────────────────────
SessionRepository — append-only session record writes + analytics queries.

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

PyMongo Async note
──────────────────
cursor.to_list() in PyMongo Async requires an explicit length argument.
Use to_list(None) to retrieve all results (no hard limit).
Use to_list(N) to cap at N documents.
Never use to_list(0) — this is invalid in PyMongo Async (was valid in Motor).

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

    # ── Write ─────────────────────────────────────────────────

    async def close_session(
        self,
        session_id:    str,
        project_id:    str,
        tenant_id:     str,
        started_at:    datetime,
        status:        str = "closed",
        turns:         int = 0,
        tool_calls:    int = 0,
        input_tokens:  int = 0,
        output_tokens: int = 0,
        error_message: str | None = None,
        api_key_id:    str = "",
    ) -> SessionDoc:
        """
        Persist a session record on close.  Always called from the
        session_runner's finally block — runs even on error/cancel.
 
        Non-critical: write failures are logged but never re-raised so
        the session shutdown path is never blocked.
        """
        ended_at = datetime.now(timezone.utc)
        duration = (ended_at - started_at).total_seconds()

        doc = SessionDoc(
            _id              = session_id,
            project_id       = project_id,
            tenant_id        = tenant_id,
            api_key_id       = api_key_id,
            status           = status,           # type: ignore[arg-type]
            started_at       = started_at,
            ended_at         = ended_at,
            duration_seconds = round(duration, 2),
            turns            = turns,
            tool_calls       = tool_calls,
            input_tokens     = input_tokens,
            output_tokens    = output_tokens,
            error_message    = error_message,
        )
        try:
            await self._col.insert_one(doc.to_mongo())
            log.info(
                "Session record written: %s "
                "(status=%s turns=%d input_tokens=%d output_tokens=%d duration=%.1fs)",
                session_id, status, turns, input_tokens, output_tokens, duration,
            )
        except Exception as exc:
            log.error("Failed to write session record %s: %s", session_id, exc)

        return doc

    # ── Read ──────────────────────────────────────────────────

    async def list_for_project(
        self,
        project_id: str,
        limit:      int = 50,
        skip:       int = 0,
        status:     str | None = None,
    ) -> list[SessionDoc]:
        """
        List session records for a project, newest first.
 
        status  optional filter: "closed" | "error" | "timeout"
                omit to return all statuses
        """
        query: dict = {"project_id": project_id}
        if status is not None:
            query["status"] = status
 
        cursor = (
            self._col.find(query)
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
                "total_input_tokens":  {"$sum": "$input_tokens"},
                "total_output_tokens": {"$sum": "$output_tokens"},
                "error_count":      {"$sum": {
                    "$cond": [{"$eq": ["$status", "error"]}, 1, 0]
                }},
            }},
        ]
        # PyMongo Async: to_list(None) replaces to_list(length=1) from Motor
        results = await self._col.aggregate(pipeline).to_list(None)
        if not results:
            return {
                "total_sessions":      0,
                "total_turns":         0,
                "total_tool_calls":    0,
                "total_duration_s":    0.0,
                "total_input_tokens":  0,
                "total_output_tokens": 0,
                "error_count":         0,
            }
        r = results[0]
        r.pop("_id", None)
        return r