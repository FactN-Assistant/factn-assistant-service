"""
core/session_events.py
──────────────────────
Session lifecycle webhook delivery.

Purpose
───────
When a Gemini session closes the platform delivers a POST to the
customer's project webhook_url (if configured) with a session.closed
event payload.  This lets customers reconcile their own records with
platform session data without polling the sessions API.

Event types
───────────
  session.started   Sent when the Gemini session is fully open and
                    session_ready has been sent to the client.
  session.closed    Sent in the session_runner finally block alongside
                    the MongoDB session record write.
  session.error     Same timing as session.closed but status = "error".

Delivery guarantees
───────────────────
Best-effort with one retry.  If delivery fails after two attempts the
failure is logged but never re-raised — it must never block the session
shutdown path.

The same HMAC-SHA256 signing used for tool webhooks is applied here so
customers can verify these events with the same secret.

Why not reuse ToolExecutor
──────────────────────────
ToolExecutor is tied to a live gsession and built for sub-second
synchronous tool call turnarounds.  Session lifecycle events are
fire-and-forget background deliveries with a different payload shape
and longer acceptable latency.  Keeping them separate maintains clear
responsibility boundaries.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

log = logging.getLogger("livechat.session_events")

# Two attempts total: immediate + one retry after a short delay.
_MAX_ATTEMPTS  = 2
_RETRY_DELAY_S = 1.5
_TIMEOUT_S     = 8.0   # generous — these are fire-and-forget


def _build_payload(
    event:           str,
    session_id:      str,
    project_id:      str,
    tenant_id:       str,
    status:          str,
    started_at:      datetime,
    ended_at:        datetime | None = None,
    duration_seconds: float          = 0.0,
    turns:           int             = 0,
    tool_calls:      int             = 0,
    input_tokens:    int             = 0,
    output_tokens:   int             = 0,
    error_message:   str | None      = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "event":      event,
        "session_id": session_id,
        "project_id": project_id,
        "tenant_id":  tenant_id,
        "status":     status,
        "started_at": started_at.isoformat(),
    }
    if ended_at is not None:
        payload["ended_at"]         = ended_at.isoformat()
        payload["duration_seconds"] = round(duration_seconds, 2)
        payload["turns"]            = turns
        payload["tool_calls"]       = tool_calls
        payload["input_tokens"]     = input_tokens
        payload["output_tokens"]    = output_tokens
        payload["total_tokens"]     = input_tokens + output_tokens
    if error_message:
        payload["error_message"] = error_message
    return payload


def _sign(body: str, secret: str) -> str:
    return "sha256=" + hmac.new(
        secret.encode(), body.encode(), hashlib.sha256
    ).hexdigest()


async def _deliver(
    webhook_url:    str,
    webhook_secret: str | None,
    payload:        dict[str, Any],
    session_id:     str,
    event:          str,
) -> None:
    """
    POST payload to webhook_url with optional HMAC signing.
    Retries once on transient failure.  Never raises.
    """
    body = json.dumps(payload, separators=(",", ":"))
    headers: dict[str, str] = {
        "Content-Type":       "application/json",
        "X-LiveChat-Event":   event,
        "X-LiveChat-Session": session_id,
    }
    if webhook_secret:
        headers["X-LiveChat-Signature"] = _sign(body, webhook_secret)

    async with httpx.AsyncClient() as client:
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                resp = await client.post(
                    webhook_url,
                    content=body,
                    headers=headers,
                    timeout=_TIMEOUT_S,
                )
                if resp.status_code < 400:
                    log.info(
                        "[%s] lifecycle webhook delivered: event=%s status=%d",
                        session_id, event, resp.status_code,
                    )
                    return
                log.warning(
                    "[%s] lifecycle webhook HTTP %d (attempt %d/%d)",
                    session_id, resp.status_code, attempt, _MAX_ATTEMPTS,
                )
            except Exception as exc:
                log.warning(
                    "[%s] lifecycle webhook error (attempt %d/%d): %s",
                    session_id, attempt, _MAX_ATTEMPTS, exc,
                )
            if attempt < _MAX_ATTEMPTS:
                await asyncio.sleep(_RETRY_DELAY_S)

    log.error(
        "[%s] lifecycle webhook failed after %d attempts: event=%s url=%s",
        session_id, _MAX_ATTEMPTS, event, webhook_url,
    )


# ── Public API ─────────────────────────────────────────────────

async def deliver_session_started(
    webhook_url:    str | None,
    webhook_secret: str | None,
    session_id:     str,
    project_id:     str,
    tenant_id:      str,
    started_at:     datetime,
) -> None:
    """
    Fire-and-forget: deliver a session.started event.
    Called immediately after Gemini session is open.
    """
    if not webhook_url:
        return
    payload = _build_payload(
        event      = "session.started",
        session_id = session_id,
        project_id = project_id,
        tenant_id  = tenant_id,
        status     = "active",
        started_at = started_at,
    )
    # Fire and forget — don't await in the hot path
    asyncio.create_task(
        _deliver(webhook_url, webhook_secret, payload, session_id, "session.started"),
        name=f"webhook-started-{session_id}",
    )


async def deliver_session_closed(
    webhook_url:     str | None,
    webhook_secret:  str | None,
    session_id:      str,
    project_id:      str,
    tenant_id:       str,
    status:          str,
    started_at:      datetime,
    ended_at:        datetime,
    duration_seconds: float,
    turns:           int,
    tool_calls:      int,
    input_tokens:    int,
    output_tokens:   int,
    error_message:   str | None = None,
) -> None:
    """
    Deliver a session.closed or session.error event.
    Called from session_runner finally block — must never raise.
    """
    if not webhook_url:
        return
    event = "session.error" if status == "error" else "session.closed"
    payload = _build_payload(
        event            = event,
        session_id       = session_id,
        project_id       = project_id,
        tenant_id        = tenant_id,
        status           = status,
        started_at       = started_at,
        ended_at         = ended_at,
        duration_seconds = duration_seconds,
        turns            = turns,
        tool_calls       = tool_calls,
        input_tokens     = input_tokens,
        output_tokens    = output_tokens,
        error_message    = error_message,
    )
    # Still fire-and-forget but we create the task before runner exits
    asyncio.create_task(
        _deliver(webhook_url, webhook_secret, payload, session_id, event),
        name=f"webhook-closed-{session_id}",
    )