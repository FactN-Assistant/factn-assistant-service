"""
core/tool_executor.py
─────────────────────
Executes tool calls on behalf of a project.
 
Two execution modes (set per-tool in ProjectConfig)
────────────────────────────────────────────────────
  static   Return the pre-configured JSON stored in the tool definition.
           Instant, no network call. Good for demos, mocks, deterministic
           responses (e.g. "get_current_time" always returns the same mock).
 
  webhook  POST the tool arguments to the customer's HTTPS endpoint.
           The platform signs the request with HMAC-SHA256 so the customer
           can verify authenticity. Retry logic is applied for transient
           failures.
 
Webhook retry strategy (Week 9 addition)
─────────────────────────────────────────
Transient failures (network errors, HTTP 5xx, HTTP 429) are retried with
exponential backoff. Permanent failures (HTTP 4xx except 429, bad JSON,
misconfigured URL) are NOT retried — retrying them would just waste time.
 
Retry schedule (defaults):
  Attempt 1: immediately
  Attempt 2: 1.0s delay
  Attempt 3: 2.0s delay
  Total wall-clock budget ≤ tool.timeout_ms (enforced by outer wait_for)
 
The customer's tool.timeout_ms controls the TOTAL time budget for all
attempts combined, not the per-attempt timeout. A single httpx request
uses tool.timeout_ms / max_attempts as its connect/read timeout so that
at least one full retry is possible within the budget.
 
HMAC-SHA256 signing
─────────────────────
Every webhook request carries:
  X-LiveChat-Signature: sha256=<hex_digest>
 
The digest is computed over the raw request body (compact JSON, no extra
spaces). Customers verify by computing the same HMAC with their webhook
secret and comparing using hmac.compare_digest() (constant-time).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
from typing import Any

import httpx

from .schemas import ProjectConfig, ToolDefinition

log = logging.getLogger("livechat.tool_executor")
 
# Retry configuration
_MAX_ATTEMPTS   = 3
_BASE_DELAY_S   = 1.0   # seconds between attempt 1 → 2
_BACKOFF_FACTOR = 2.0   # multiplied each retry: 1s, 2s, 4s (if budget allows)

# HTTP status codes that are retryable (transient failures)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class ToolExecutor:
    """Resolves and executes tool calls for one project session."""

    def __init__(self, project: ProjectConfig) -> None:
        self._project_id = project.project_id
        # Build name → definition lookup once at session open
        self._tools: dict[str, ToolDefinition] = {
            t.name: t for t in project.tools
        }

    async def execute(
        self,
        name:       str,
        args:       dict[str, Any],
        session_id: str,
        call_id:    str = "",
    ) -> dict[str, Any]:
        """
        Dispatch a single tool call. Always returns a dict — never raises.
        Errors are surfaced as {"error": "..."} so Gemini receives a
        well-formed function response instead of a crash.
        """
        tool = self._tools.get(name)
        if tool is None:
            log.warning("[%s] unknown tool requested: %s", session_id, name)
            return {"error": f"Unknown tool: {name!r}"}

        log.info(
            "[%s] executing tool %s (mode=%s)",
            session_id, name, tool.execution_mode,
        )

        if tool.execution_mode == "static":
            return self._execute_static(tool)

        if tool.execution_mode == "webhook":
            return await self._execute_webhook(tool, args, session_id, call_id)

        return {"error": f"Unknown execution mode: {tool.execution_mode!r}"}

    # ── Static mode ───────────────────────────────────────────

    def _execute_static(self, tool: ToolDefinition) -> dict[str, Any]:
        """
        Return the stored static_response from the project config.
        This is the Week 9 task 4 implementation — no network call,
        just return the JSON the customer configured in the dashboard.
        """
        if tool.static_response is None:
            return {"status": "ok"}
        return tool.static_response

    # ── Webhook mode ──────────────────────────────────────────

    async def _execute_webhook(
        self,
        tool:       ToolDefinition,
        args:       dict[str, Any],
        session_id: str,
        call_id:    str,
    ) -> dict[str, Any]:
        """
        POST tool arguments to the customer's webhook URL with HMAC signing
        and exponential backoff retry for transient failures.
 
        Total time across all attempts is bounded by tool.timeout_ms.
        """
        if not tool.webhook_url:
            return {"error": "No webhook URL configured for this tool."}

        payload = {
            "tool_name":  tool.name,
            "call_id":    call_id,
            "arguments":  args,
            "session_id": session_id,
            "project_id": self._project_id,
        }
        # Compact JSON — no extra whitespace — for consistent HMAC computation
        body = json.dumps(payload, separators=(",", ":"))

        headers: dict[str, str] = {
            "Content-Type":       "application/json",
            "X-LiveChat-Session": session_id,
            "X-LiveChat-Project": self._project_id,
        }

        # HMAC-SHA256 signature — customers verify with their webhook_secret
        if tool.webhook_secret:
            sig = hmac.new(
                tool.webhook_secret.encode(),
                body.encode(),
                hashlib.sha256,
            ).hexdigest()
            headers["X-LiveChat-Signature"] = f"sha256={sig}"

        total_budget_s  = tool.timeout_ms / 1_000
        # Per-attempt timeout: divide budget across attempts, leaving room
        # for retry delays. Minimum 1 second per attempt.
        per_attempt_s   = max(1.0, total_budget_s / _MAX_ATTEMPTS)
 
        last_error: str = "No attempts made"
 
        async with httpx.AsyncClient() as client:
            for attempt in range(1, _MAX_ATTEMPTS + 1):
                try:
                    resp = await client.post(
                        tool.webhook_url,
                        content = body,
                        headers = headers,
                        timeout = per_attempt_s,
                    )
 
                    # ── Success ───────────────────────────────────────
                    if resp.status_code < 400:
                        try:
                            data = resp.json()
                        except Exception:
                            log.error(
                                "[%s] webhook %s returned non-JSON body (HTTP %d)",
                                session_id, tool.name, resp.status_code,
                            )
                            return {
                                "error": (
                                    f"Webhook returned HTTP {resp.status_code} "
                                    "but the response body was not valid JSON."
                                )
                            }
                        log.info(
                            "[%s] webhook %s succeeded (HTTP %d, attempt %d/%d)",
                            session_id, tool.name, resp.status_code,
                            attempt, _MAX_ATTEMPTS,
                        )
                        # Convention: customer wraps result in {"result": {...}}
                        return data.get("result", data)
 
                    # ── Retryable HTTP error (5xx, 429) ───────────────
                    if resp.status_code in _RETRYABLE_STATUS:
                        last_error = (
                            f"HTTP {resp.status_code} on attempt {attempt}"
                        )
                        log.warning(
                            "[%s] webhook %s retryable error: HTTP %d (attempt %d/%d)",
                            session_id, tool.name, resp.status_code,
                            attempt, _MAX_ATTEMPTS,
                        )
                        # Respect Retry-After header if the server sends one
                        retry_after = _parse_retry_after(
                            resp.headers.get("Retry-After"), per_attempt_s
                        )
                        if attempt < _MAX_ATTEMPTS:
                            await asyncio.sleep(
                                min(retry_after, total_budget_s / 2)
                            )
                        continue
 
                    # ── Permanent HTTP error (4xx except 429) ─────────
                    # Do not retry — the request is malformed or unauthorised
                    log.error(
                        "[%s] webhook %s permanent error: HTTP %d (not retrying)",
                        session_id, tool.name, resp.status_code,
                    )
                    return {
                        "error": (
                            f"Webhook returned HTTP {resp.status_code}. "
                            "Check the webhook URL and authentication."
                        )
                    }
 
                # ── Network / timeout errors (retryable) ──────────────
                except httpx.TimeoutException:
                    last_error = f"Timeout on attempt {attempt}"
                    log.warning(
                        "[%s] webhook %s timed out (attempt %d/%d, %.1fs budget)",
                        session_id, tool.name, attempt, _MAX_ATTEMPTS,
                        per_attempt_s,
                    )
 
                except httpx.ConnectError as exc:
                    last_error = f"Connection error on attempt {attempt}: {exc}"
                    log.warning(
                        "[%s] webhook %s connection error (attempt %d/%d): %s",
                        session_id, tool.name, attempt, _MAX_ATTEMPTS, exc,
                    )
 
                except Exception as exc:
                    # Unexpected error — log and do not retry
                    log.exception(
                        "[%s] webhook %s unexpected error", session_id, tool.name
                    )
                    return {"error": str(exc)}
 
                # ── Backoff before next attempt ────────────────────────
                if attempt < _MAX_ATTEMPTS:
                    delay = _BASE_DELAY_S * (_BACKOFF_FACTOR ** (attempt - 1))
                    log.info(
                        "[%s] webhook %s retrying in %.1fs (attempt %d/%d)",
                        session_id, tool.name, delay, attempt + 1, _MAX_ATTEMPTS,
                    )
                    await asyncio.sleep(delay)
 
        # All attempts exhausted
        log.error(
            "[%s] webhook %s failed after %d attempts. Last error: %s",
            session_id, tool.name, _MAX_ATTEMPTS, last_error,
        )
        return {
            "error": (
                f"Webhook failed after {_MAX_ATTEMPTS} attempts. "
                f"Last error: {last_error}"
            )
        }
 
 
# ── Internal helpers ──────────────────────────────────────────
 
def _parse_retry_after(header_value: str | None, fallback: float) -> float:
    """
    Parse a Retry-After header value (integer seconds or HTTP-date).
    Returns the fallback if the header is absent or unparseable.
    """
    if header_value is None:
        return fallback
    try:
        return max(0.0, float(header_value))
    except (ValueError, TypeError):
        return fallback
