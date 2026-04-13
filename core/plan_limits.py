"""
core/plan_limits.py
────────────────────
Single source of truth for all plan-based limits.

HOW TO EDIT LIMITS
──────────────────
This is the ONE file you touch to change any plan limit.
Every enforcement point in the application reads from this file.

Example: raise free plan concurrent sessions from 2 → 3:
  Just change:  "concurrent_sessions": 2
  To:           "concurrent_sessions": 3
  That's it. No other file needs to change.

Limit fields
────────────
  concurrent_sessions   Max simultaneous WebSocket sessions per project.
                        None = unlimited (enterprise).
  session_ttl_seconds   Max idle timeout a tenant can set on a project.
                        Actual session TTL is min(project.session_ttl, plan.session_ttl).
  projects              Max number of projects a tenant can own.
                        None = unlimited.
  tools_per_project     Max tool definitions per project.
                        None = unlimited.
  daily_token_quota     Max total tokens (input + output) per tenant per day.
                        None = unlimited.
  webhook_timeout_ms    Max allowed timeout_ms on webhook tools.
                        Customer-set tool.timeout_ms is capped at this value.
  rate_limit_rpm        Max rate_limit_rpm a tenant can set on an API key.
                        None = unlimited.

Adding a new plan
─────────────────
Just add a new key to PLAN_LIMITS.  All enforcement helpers below
read from this dict, so no other code changes are needed.

Adding a new limit field
────────────────────────
1. Add the field to every plan dict below (use None for unlimited).
2. Add a getter function following the pattern of the existing ones.
3. Call the getter from the enforcement point.
"""

from __future__ import annotations

from typing import Any

# ── Plan definitions ──────────────────────────────────────────
# Edit these values to change plan limits.
# None means "unlimited" for every field that supports it.

PLAN_LIMITS: dict[str, dict[str, Any]] = {
    "free": {
        "concurrent_sessions":  2,
        "session_ttl_seconds":  60,
        "projects":             1,
        "tools_per_project":    3,
        "daily_token_quota":    10_000,
        "webhook_timeout_ms":   3_000,
        "rate_limit_rpm":       30,
    },
    "starter": {
        "concurrent_sessions":  20,
        "session_ttl_seconds":  300,
        "projects":             5,
        "tools_per_project":    10,
        "daily_token_quota":    500_000,
        "webhook_timeout_ms":   5_000,
        "rate_limit_rpm":       120,
    },
    "pro": {
        "concurrent_sessions":  100,
        "session_ttl_seconds":  1_800,
        "projects":             25,
        "tools_per_project":    30,
        "daily_token_quota":    5_000_000,
        "webhook_timeout_ms":   15_000,
        "rate_limit_rpm":       600,
    },
    "enterprise": {
        "concurrent_sessions":  None,      # unlimited
        "session_ttl_seconds":  86_400,    # 24 hours max
        "projects":             None,      # unlimited
        "tools_per_project":    None,      # unlimited
        "daily_token_quota":    None,      # unlimited
        "webhook_timeout_ms":   30_000,
        "rate_limit_rpm":       None,      # unlimited
    },
}

# Fallback when an unknown plan string is encountered.
# Defaults to the most restrictive tier rather than crashing.
_FALLBACK_PLAN = "free"


# ── Internal helper ───────────────────────────────────────────

def _get(plan: str, field: str) -> Any:
    """
    Return the limit value for a plan + field combination.
    Falls back to the free plan if the plan string is unrecognised.
    """
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS[_FALLBACK_PLAN])
    return limits.get(field, PLAN_LIMITS[_FALLBACK_PLAN].get(field))


# ── Public getters ─────────────────────────────────────────────
# One function per limit.  Call these from enforcement points —
# never access PLAN_LIMITS directly in other modules.

def max_concurrent_sessions(plan: str) -> int | None:
    """
    Max simultaneous WebSocket sessions per project.
    Returns None for unlimited (enterprise).
    """
    return _get(plan, "concurrent_sessions")


def max_session_ttl_seconds(plan: str) -> int:
    """
    Maximum idle-timeout a tenant may configure on a project.
    The effective TTL is min(project.session_ttl_seconds, this value).
    """
    return _get(plan, "session_ttl_seconds")


def max_projects(plan: str) -> int | None:
    """
    Maximum number of projects a tenant may own.
    Returns None for unlimited.
    """
    return _get(plan, "projects")


def max_tools_per_project(plan: str) -> int | None:
    """
    Maximum tool definitions allowed per project.
    Returns None for unlimited.
    """
    return _get(plan, "tools_per_project")


def daily_token_quota(plan: str) -> int | None:
    """
    Maximum total tokens (input + output) a tenant may consume per day.
    Returns None for unlimited.
    """
    return _get(plan, "daily_token_quota")


def max_webhook_timeout_ms(plan: str) -> int:
    """
    Maximum allowed timeout_ms on a webhook tool.
    Customer-set values are capped at this limit.
    """
    return _get(plan, "webhook_timeout_ms")


def max_rate_limit_rpm(plan: str) -> int | None:
    """
    Maximum rate_limit_rpm a tenant may set on an API key.
    Returns None for unlimited.
    """
    return _get(plan, "rate_limit_rpm")


def get_all_limits(plan: str) -> dict[str, Any]:
    """
    Return all limits for a plan as a plain dict.
    Used by the /v1/auth/me endpoint to expose limits to the dashboard.
    """
    return dict(PLAN_LIMITS.get(plan, PLAN_LIMITS[_FALLBACK_PLAN]))


def is_within_concurrent_sessions(plan: str, current_count: int) -> bool:
    """
    Return True if opening one more session is within the plan limit.
    Always returns True for unlimited plans.
    """
    limit = max_concurrent_sessions(plan)
    if limit is None:
        return True
    return current_count < limit


def is_within_project_limit(plan: str, current_count: int) -> bool:
    """Return True if creating one more project is within the plan limit."""
    limit = max_projects(plan)
    if limit is None:
        return True
    return current_count < limit


def is_within_tool_limit(plan: str, current_count: int) -> bool:
    """Return True if adding one more tool is within the plan limit."""
    limit = max_tools_per_project(plan)
    if limit is None:
        return True
    return current_count < limit


def effective_session_ttl(plan: str, requested_ttl: int) -> int:
    """
    Return the effective session TTL: the lesser of the requested TTL
    and the plan's maximum.  This prevents free-tier tenants from setting
    a 7200-second TTL on their projects.
    """
    return min(requested_ttl, max_session_ttl_seconds(plan))


def effective_webhook_timeout_ms(plan: str, requested_ms: int) -> int:
    """
    Return the effective webhook timeout: the lesser of the requested value
    and the plan's maximum.
    """
    return min(requested_ms, max_webhook_timeout_ms(plan))


def effective_rate_limit_rpm(plan: str, requested_rpm: int) -> int:
    """
    Return the effective rate limit: the lesser of the requested value
    and the plan's maximum (if capped).
    """
    limit = max_rate_limit_rpm(plan)
    if limit is None:
        return requested_rpm
    return min(requested_rpm, limit)