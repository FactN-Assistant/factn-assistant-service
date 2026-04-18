"""
api/plans.py
────────────
Plan information endpoints.

GET /v1/plans          list all available plans with their limits
GET /v1/plans/current  return the authenticated tenant's current plan and limits
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel

from core import plan_limits as pl
from core.documents import TenantDoc

from .dependencies import get_current_tenant

log = logging.getLogger("livechat.api.plans")

router = APIRouter(prefix="/v1/plans", tags=["plans"])


# ── Response schemas ──────────────────────────────────────────

class PlanResponse(BaseModel):
    """Single plan tier with all its limits."""
    name:                   str
    concurrent_sessions:    int | None
    session_ttl_seconds:    int
    projects:               int | None
    tools_per_project:      int | None
    daily_token_quota:      int | None
    webhook_timeout_ms:     int
    rate_limit_rpm:         int | None


class CurrentPlanResponse(BaseModel):
    """The tenant's active plan with limits."""
    plan:   str
    limits: PlanResponse


# ── Routes ────────────────────────────────────────────────────

@router.get("", status_code=status.HTTP_200_OK)
async def list_plans() -> list[PlanResponse]:
    """
    Return every available plan tier and its limits.

    This endpoint is public (no auth required) so marketing pages
    and the upgrade modal can display plan comparisons.
    """
    result: list[PlanResponse] = []
    for plan_name, limits in pl.PLAN_LIMITS.items():
        result.append(
            PlanResponse(
                name=plan_name,
                concurrent_sessions=limits["concurrent_sessions"],
                session_ttl_seconds=limits["session_ttl_seconds"],
                projects=limits["projects"],
                tools_per_project=limits["tools_per_project"],
                daily_token_quota=limits["daily_token_quota"],
                webhook_timeout_ms=limits["webhook_timeout_ms"],
                rate_limit_rpm=limits["rate_limit_rpm"],
            )
        )
    return result


@router.get("/current", status_code=status.HTTP_200_OK)
async def get_current_plan(
    tenant: TenantDoc = Depends(get_current_tenant),
) -> CurrentPlanResponse:
    """
    Return the authenticated tenant's current plan and its limits.
    """
    limits = pl.get_all_limits(tenant.plan)
    return CurrentPlanResponse(
        plan=tenant.plan,
        limits=PlanResponse(
            name=tenant.plan,
            concurrent_sessions=limits["concurrent_sessions"],
            session_ttl_seconds=limits["session_ttl_seconds"],
            projects=limits["projects"],
            tools_per_project=limits["tools_per_project"],
            daily_token_quota=limits["daily_token_quota"],
            webhook_timeout_ms=limits["webhook_timeout_ms"],
            rate_limit_rpm=limits["rate_limit_rpm"],
        ),
    )
