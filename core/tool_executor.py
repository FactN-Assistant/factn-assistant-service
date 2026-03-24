"""
core/tool_executor.py
─────────────────────
Executes tool calls on behalf of a project.

Two modes (set per-tool in the ProjectConfig):

  static   Return a pre-configured JSON response stored in the tool
           definition.  Great for demos, mocks, or deterministic tools.

  webhook  POST the tool arguments to the customer's HTTPS endpoint.
           The platform signs the body with HMAC-SHA256 so the customer
           can verify authenticity.  Timeout is enforced per plan limits.

ToolExecutor is instantiated once per session inside session_runner so
that project config is captured at session-open time — consistent with
how Gemini config is built.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any

import httpx

from .schemas import ProjectConfig, ToolDefinition

log = logging.getLogger("livechat.tool_executor")


class ToolExecutor:
    """Resolves and executes tool calls for one project."""

    def __init__(self, project: ProjectConfig) -> None:
        self._project_id = project.project_id
        # Build a name → definition lookup once
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
        Dispatch a single tool call.  Always returns a dict — never raises.
        Errors are surfaced as {"error": "..."} so Gemini receives a
        well-formed function response rather than a crash.
        """
        tool = self._tools.get(name)
        if tool is None:
            log.warning("[%s] unknown tool requested: %s", session_id, name)
            return {"error": f"Unknown tool: {name!r}"}

        log.info(
            "[%s] executing tool %s (mode=%s) args=%s",
            session_id, name, tool.execution_mode, args,
        )

        if tool.execution_mode == "static":
            return self._execute_static(tool)

        if tool.execution_mode == "webhook":
            return await self._execute_webhook(tool, args, session_id, call_id)

        return {"error": f"Unknown execution mode: {tool.execution_mode!r}"}

    # ── Static ────────────────────────────────────────────────

    def _execute_static(self, tool: ToolDefinition) -> dict[str, Any]:
        if tool.static_response is None:
            return {"status": "ok"}
        return tool.static_response

    # ── Webhook ───────────────────────────────────────────────

    async def _execute_webhook(
        self,
        tool:       ToolDefinition,
        args:       dict[str, Any],
        session_id: str,
        call_id:    str,
    ) -> dict[str, Any]:
        if not tool.webhook_url:
            return {"error": "No webhook URL configured for this tool."}

        payload = {
            "tool_name":  tool.name,
            "call_id":    call_id,
            "arguments":  args,
            "session_id": session_id,
            "project_id": self._project_id,
        }
        body = json.dumps(payload, separators=(",", ":"))

        headers: dict[str, str] = {
            "Content-Type":       "application/json",
            "X-LiveChat-Session": session_id,
            "X-LiveChat-Project": self._project_id,
        }

        # HMAC-SHA256 signature so customers can verify the request origin
        if tool.webhook_secret:
            sig = hmac.new(
                tool.webhook_secret.encode(),
                body.encode(),
                hashlib.sha256,
            ).hexdigest()
            headers["X-LiveChat-Signature"] = f"sha256={sig}"

        timeout_s = tool.timeout_ms / 1_000

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    tool.webhook_url,
                    content=body,
                    headers=headers,
                    timeout=timeout_s,
                )
                resp.raise_for_status()
                data = resp.json()
                # Convention: customer returns {"result": {...}}
                return data.get("result", data)

        except httpx.TimeoutException:
            log.error("[%s] webhook timeout for tool %s", session_id, tool.name)
            return {"error": f"Tool {tool.name!r} timed out after {tool.timeout_ms} ms."}

        except httpx.HTTPStatusError as exc:
            log.error(
                "[%s] webhook HTTP %s for tool %s",
                session_id, exc.response.status_code, tool.name,
            )
            return {"error": f"Webhook returned HTTP {exc.response.status_code}."}

        except Exception as exc:
            log.exception("[%s] webhook error for tool %s", session_id, tool.name)
            return {"error": str(exc)}