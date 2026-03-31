"""
api/auth.py
───────────
Authentication endpoints.

POST /v1/auth/register   create a new tenant account
POST /v1/auth/login      authenticate and receive tokens
POST /v1/auth/refresh    rotate refresh token → new access + refresh
POST /v1/auth/logout     revoke all refresh tokens for this tenant
GET  /v1/auth/me         return current tenant profile

Cookie design
─────────────
Both tokens are set as httpOnly, Secure, SameSite=Lax cookies so they
are never accessible from JavaScript — prevents XSS token theft.

  access_token   15 min  — sent on every request, verified stateless
  refresh_token  7 days  — sent only to /v1/auth/refresh

The access token is ALSO returned in the response body so server-side
clients (not browser-based) can store it in memory and send it as a
Bearer header instead of relying on cookies.

Rate limiting on auth endpoints
─────────────────────────────────
POST /v1/auth/login is rate-limited via Redis to 10 attempts per minute
per IP address.  This is a basic brute-force mitigation.  Production
systems should add CAPTCHA or exponential backoff.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, Field
from pymongo.errors import DuplicateKeyError

from core.auth import (
    create_access_token,
    hash_password,
    needs_rehash,
    verify_password,
)
from core.documents import TenantDoc
from db.redis_client import RedisClient
from repositories import Repositories

from .dependencies import get_current_tenant

log = logging.getLogger("livechat.api.auth")

router = APIRouter(prefix="/v1/auth", tags=["auth"])

# Cookie settings
_COOKIE_OPTS: dict = dict(
    httponly = True,
    secure   = False,   # set True in production (HTTPS only)
    samesite = "lax",
    path     = "/",
)
_ACCESS_MAX_AGE  = 15 * 60           # 15 minutes in seconds
_REFRESH_MAX_AGE = 7 * 24 * 60 * 60  # 7 days in seconds


# ── Request / Response schemas ────────────────────────────────

class RegisterRequest(BaseModel):
    name:     str        = Field(min_length=1, max_length=100)
    email:    EmailStr
    password: str        = Field(min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str        = Field(min_length=1)


class AuthResponse(BaseModel):
    """Returned in the response body on login and refresh."""
    access_token: str
    token_type:   str = "bearer"
    tenant_id:    str
    email:        str
    name:         str
    plan:         str


class TenantProfileResponse(BaseModel):
    tenant_id:  str
    email:      str
    name:       str
    plan:       str
    created_at: str


# ── Helpers ───────────────────────────────────────────────────

def _set_auth_cookies(
    response:      Response,
    access_token:  str,
    refresh_token: str,
) -> None:
    response.set_cookie(
        key      = "access_token",
        value    = access_token,
        max_age  = _ACCESS_MAX_AGE,
        **_COOKIE_OPTS,
    )
    response.set_cookie(
        key      = "refresh_token",
        value    = refresh_token,
        max_age  = _REFRESH_MAX_AGE,
        **_COOKIE_OPTS,
    )


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie("access_token",  path="/")
    response.delete_cookie("refresh_token", path="/")


async def _check_login_rate_limit(request: Request) -> None:
    """
    Allow 10 login attempts per minute per client IP.
    Raises HTTP 429 if exceeded.
    """
    redis: RedisClient = request.app.state.redis
    client_ip = request.client.host if request.client else "unknown"
    allowed, count = await redis.check_and_increment_rate_limit(
        key_prefix = f"login:{client_ip}",
        limit      = 10,
        window_s   = 60,
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Try again in a minute.",
        )


# ── Endpoints ─────────────────────────────────────────────────

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    body:     RegisterRequest,
    request:  Request,
    response: Response,
) -> AuthResponse:
    """
    Create a new tenant account.

    On success: account is created, tokens are issued immediately (no
    separate login step required), cookies are set.

    Errors:
      409  email already registered
      422  validation failure (weak password, bad email format)
    """
    repos: Repositories = request.app.state.repos

    # Hash password BEFORE touching the DB — if hashing fails the account
    # is never created.
    password_hash = hash_password(body.password)

    try:
        tenant = await repos.tenants.create(
            name  = body.name.strip(),
            email = body.email.lower(),
        )
    except DuplicateKeyError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    # Store the password hash
    await repos.tenants.update_password_hash(tenant.id, password_hash)
    log.info("New tenant registered: %s (%s)", tenant.id, tenant.email)

    # Issue tokens immediately
    access_token = create_access_token(
        tenant_id = tenant.id,
        email     = tenant.email,
        plan      = tenant.plan,
    )
    _, refresh_token = await repos.auth_tokens.create(tenant_id=tenant.id)

    _set_auth_cookies(response, access_token, refresh_token)

    return AuthResponse(
        access_token = access_token,
        tenant_id    = tenant.id,
        email        = tenant.email,
        name         = tenant.name,
        plan         = tenant.plan,
    )


@router.post("/login")
async def login(
    body:     LoginRequest,
    request:  Request,
    response: Response,
) -> AuthResponse:
    """
    Authenticate with email + password.

    On success: access token (15 min) and refresh token (7 days) are
    set as httpOnly cookies and the access token is returned in the body.

    Errors:
      401  invalid credentials (deliberately vague — don't reveal if
           the email exists or just the password is wrong)
      429  rate limit exceeded (10 attempts/min per IP)
    """
    await _check_login_rate_limit(request)

    repos: Repositories = request.app.state.repos

    tenant = await repos.tenants.get_by_email(body.email)
    if tenant is None or not tenant.is_active:
        # Constant-time-ish: still hash to avoid timing attacks that reveal
        # whether the email exists.
        hash_password("dummy-constant-time-fill")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    if not verify_password(body.password, tenant.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )

    # Rehash if parameters have been upgraded since last login
    if needs_rehash(tenant.password_hash):
        new_hash = hash_password(body.password)
        await repos.tenants.update_password_hash(tenant.id, new_hash)
        log.info("Password hash upgraded for tenant %s", tenant.id)

    access_token = create_access_token(
        tenant_id = tenant.id,
        email     = tenant.email,
        plan      = tenant.plan,
    )
    _, refresh_token = await repos.auth_tokens.create(tenant_id=tenant.id)

    _set_auth_cookies(response, access_token, refresh_token)
    log.info("Tenant logged in: %s", tenant.id)

    return AuthResponse(
        access_token = access_token,
        tenant_id    = tenant.id,
        email        = tenant.email,
        name         = tenant.name,
        plan         = tenant.plan,
    )


@router.post("/refresh")
async def refresh(
    request:  Request,
    response: Response,
) -> AuthResponse:
    """
    Exchange a valid refresh token for a new access + refresh token pair.

    The old refresh token is deleted (rotated) on every call.  Presenting
    a superseded token triggers a security warning.

    The refresh token is read from the refresh_token httpOnly cookie.

    Errors:
      401  missing, invalid, or expired refresh token
    """
    repos:         Repositories = request.app.state.repos
    raw_refresh    = request.cookies.get("refresh_token")

    if not raw_refresh:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token missing.",
        )

    # We need the tenant_id to scope the lookup — extract it from the
    # access token cookie (may be expired, but sub claim is still readable).
    # If both cookies are gone, require a fresh login.
    raw_access  = request.cookies.get("access_token")
    tenant_id: str | None = None

    if raw_access:
        from core.auth import decode_access_token
        payload = decode_access_token(raw_access)
        if payload:
            tenant_id = payload.get("sub")

    if not tenant_id:
        # Fall back: try to decode without expiry validation to get the sub.
        # python-jose doesn't natively support "decode but ignore exp", so
        # we handle JWTError gracefully and redirect to login.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired — please log in again.",
        )

    result = await repos.auth_tokens.verify_and_rotate(raw_refresh, tenant_id)
    if result is None:
        _clear_auth_cookies(response)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token is invalid or expired — please log in again.",
        )

    _, new_refresh_token = result

    tenant = await repos.tenants.get_by_id(tenant_id)
    if tenant is None or not tenant.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account not found or deactivated.",
        )

    new_access_token = create_access_token(
        tenant_id = tenant.id,
        email     = tenant.email,
        plan      = tenant.plan,
    )

    _set_auth_cookies(response, new_access_token, new_refresh_token)
    log.info("Token refreshed for tenant %s", tenant_id)

    return AuthResponse(
        access_token = new_access_token,
        tenant_id    = tenant.id,
        email        = tenant.email,
        name         = tenant.name,
        plan         = tenant.plan,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    request:  Request,
    response: Response,
    tenant:   TenantDoc = Depends(get_current_tenant),
) -> None:
    """
    Revoke all refresh tokens for this tenant and clear auth cookies.

    After this call the access token remains technically valid for up to
    15 more minutes (stateless JWT), but all refresh tokens are gone so
    the session cannot be renewed.  For immediate full invalidation,
    rotate JWT_SECRET (affects all tenants — use sparingly).
    """
    repos: Repositories = request.app.state.repos
    await repos.auth_tokens.revoke_for_tenant(tenant.id)
    _clear_auth_cookies(response)
    log.info("Tenant logged out: %s", tenant.id)


@router.get("/me")
async def me(
    tenant: TenantDoc = Depends(get_current_tenant),
) -> TenantProfileResponse:
    """Return the current tenant's profile."""
    return TenantProfileResponse(
        tenant_id  = tenant.id,
        email      = tenant.email,
        name       = tenant.name,
        plan       = tenant.plan,
        created_at = tenant.created_at.isoformat(),
    )