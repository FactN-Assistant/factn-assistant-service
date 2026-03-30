"""
api/dependencies.py
────────────────────
FastAPI dependency functions shared across all API routers.

get_current_tenant()
─────────────────────
Extracts and validates the access token from the Authorization header
(Bearer scheme) or from the access_token httpOnly cookie.  Returns a
TenantDoc on success.  Raises HTTP 401 on any failure.

Used with FastAPI's Depends() in every protected route:

    @router.get("/v1/projects")
    async def list_projects(
        request: Request,
        tenant:  TenantDoc = Depends(get_current_tenant),
    ):
        ...

Cookie vs header
────────────────
Browser dashboard clients use httpOnly cookies set by /v1/auth/login.
Server-to-server API clients send the token as a Bearer header.
The dependency checks the header first, then falls back to the cookie.
"""

from __future__ import annotations

import logging

from fastapi import Cookie, Depends, Header, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.auth import decode_access_token
from core.documents import TenantDoc
from repositories import Repositories

log = logging.getLogger("livechat.deps")

_bearer = HTTPBearer(auto_error=False)


async def get_current_tenant(
    request:     Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    access_token_cookie: str | None = Cookie(default=None, alias="access_token"),
) -> TenantDoc:
    """
    Resolve the current authenticated tenant from access token.

    Token priority:
      1. Authorization: Bearer <token>   (API clients, server-side)
      2. access_token cookie             (browser dashboard)
    """
    # 1. Prefer Authorization header
    raw_token: str | None = None
    if credentials and credentials.credentials:
        raw_token = credentials.credentials
    elif access_token_cookie:
        raw_token = access_token_cookie

    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(raw_token)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token is invalid or expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    tenant_id: str = payload.get("sub", "")
    repos: Repositories = request.app.state.repos
    tenant = await repos.tenants.get_by_id(tenant_id)

    if tenant is None or not tenant.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tenant account not found or deactivated.",
        )

    return tenant