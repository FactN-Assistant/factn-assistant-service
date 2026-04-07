"""
api/tokens.py
─────────────
Ephemeral token issuance and rotation endpoints.

Who calls these
───────────────
These endpoints are called exclusively by the CUSTOMER'S BACKEND SERVER,
never by a browser.  The customer's server holds a secret key (sk_live_)
and uses it to mint short-lived tokens for their end-users.

  POST /v1/tokens          mint a new ephemeral token
  POST /v1/tokens/rotate   replace an existing token before it expires

Authentication
──────────────
Both endpoints require a SECRET key (sk_live_...) in the Authorization
header.  Publishable keys are rejected with 403.  This is enforced because
only server-side code should ever be minting tokens.

Flow
────
1.  Customer backend calls POST /v1/tokens with their secret key.
2.  Platform validates the key, resolves the project, generates a 43-char
    random token, stores it in Redis with the requested TTL, returns the
    token to the backend.
3.  Customer backend passes the token to the browser (via API response,
    cookie, or URL parameter).
4.  Browser opens WebSocket: wss://api.livechat.io/v1/chat?token=<value>
5.  Platform redeems (atomically deletes) the token on successful handshake.
6.  The token is now dead — cannot be reused.

Rotation flow (task 4 of Week 8)
─────────────────────────────────
The customer's backend can call POST /v1/tokens/rotate with an existing
(not yet redeemed) token to:
  • verify it is still valid
  • delete it
  • issue a fresh token with a new TTL

This allows the customer to refresh tokens before they expire in long
page-load scenarios.  Rotation is NOT triggered automatically by the
platform — the customer's backend must implement the refresh logic.

Recommended rotation strategy: call /v1/tokens/rotate when the remaining
TTL of the current token drops below 30 seconds.  The response includes
the remaining_ttl field to make this easy to implement.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from db.redis_client import (
    EPHEMERAL_TOKEN_DEFAULT_TTL,
    EPHEMERAL_TOKEN_MAX_TTL,
    RedisClient,
)
from repositories import Repositories

log = logging.getLogger("livechat.api.tokens")

router = APIRouter(prefix="/v1/tokens", tags=["tokens"])

_bearer = HTTPBearer(auto_error=False)


# ── Request / Response schemas ────────────────────────────────

class CreateTokenRequest(BaseModel):
    """
    Body for POST /v1/tokens.

    ttl_seconds   How long the token is valid.  The browser must open its
                  WebSocket before this elapses.  Default 60 s, max 300 s.

    metadata      Arbitrary key-value pairs the customer wants attached to
                  the session record (e.g. user_id, plan tier, locale).
                  Stored in the Redis payload, copied into SessionDoc on
                  session close.  Max 10 keys, string values only.
    """
    ttl_seconds: int                  = Field(
        default = EPHEMERAL_TOKEN_DEFAULT_TTL,
        ge      = 1,
        le      = EPHEMERAL_TOKEN_MAX_TTL,
    )
    metadata:    dict[str, str]       = Field(default_factory=dict)

    class Config:
        # Reject extra fields — metadata should be in the metadata dict,
        # not at the top level.
        extra = "forbid"


class TokenResponse(BaseModel):
    """
    Returned by POST /v1/tokens and POST /v1/tokens/rotate.
    The ephemeral_token is the value the browser sends as ?token=<value>.
    """
    ephemeral_token: str
    expires_at:      int    # unix timestamp
    ttl_seconds:     int
    project_id:      str


class RotateTokenRequest(BaseModel):
    """Body for POST /v1/tokens/rotate."""
    current_token: str  = Field(min_length=1)
    ttl_seconds:   int  = Field(
        default = EPHEMERAL_TOKEN_DEFAULT_TTL,
        ge      = 1,
        le      = EPHEMERAL_TOKEN_MAX_TTL,
    )


class RotateTokenResponse(BaseModel):
    """
    Returned by POST /v1/tokens/rotate.
    The old token is deleted; the new token must be passed to the browser.
    """
    ephemeral_token: str
    expires_at:      int
    ttl_seconds:     int
    project_id:      str
    previous_remaining_ttl: int  # how many seconds the old token had left


# ── Auth helper ───────────────────────────────────────────────

async def _require_secret_key(request: Request) -> tuple:
    """
    Extract and validate the secret API key from the Authorization header.

    Returns (key_doc, project_config) on success.
    Raises HTTP 401/403 on failure.

    This is intentionally NOT using the get_current_tenant dependency
    because token issuance is authenticated with an API key (sk_live_),
    not a JWT.  The customer's backend server holds the secret key —
    there is no logged-in tenant session here.
    """
    repos:  Repositories = request.app.state.repos
    redis:  RedisClient  = request.app.state.redis

    # Extract raw key from Authorization: Bearer header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required (Bearer sk_live_...).",
            headers={"WWW-Authenticate": "Bearer"},
        )

    raw_key = auth_header[len("Bearer "):].strip()
    if not raw_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key missing from Authorization header.",
        )

    # Validate key against DB
    key_doc = await repos.api_keys.get_by_raw_key(raw_key)
    if key_doc is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
        )

    # Enforce secret key requirement — publishable keys cannot mint tokens
    if key_doc.key_type != "secret":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Ephemeral tokens can only be minted with a secret key "
                "(sk_live_...).  Publishable keys are for browser clients only."
            ),
        )

    # Resolve the project config
    config = await repos.projects.get_config_for_key(key_doc)
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found or inactive.",
        )

    return key_doc, config


# ── Endpoints ─────────────────────────────────────────────────

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_token(
    body:    CreateTokenRequest,
    request: Request,
) -> TokenResponse:
    """
    Mint a new single-use ephemeral token.

    Must be called from the customer's backend server using a secret key.
    The returned ephemeral_token should be passed to the browser and used
    exactly once to open a WebSocket connection.

    The token cannot be recovered after this response — if it is lost
    before the browser uses it, call this endpoint again to get a new one.
    """
    key_doc, config = await _require_secret_key(request)
    redis: RedisClient = request.app.state.redis

    # Validate metadata size — prevent abuse of the Redis payload
    if len(body.metadata) > 10:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="metadata may contain at most 10 keys.",
        )

    raw_token, expires_at = await redis.create_ephemeral_token(
        project_id     = config.project_id,
        tenant_id      = config.tenant_id,
        api_key_id     = key_doc.id,
        rate_limit_rpm = key_doc.rate_limit_rpm,
        ttl            = body.ttl_seconds,
        metadata       = dict(body.metadata),
    )

    log.info(
        "Token minted: project=%s key=%s ttl=%ds",
        config.project_id, key_doc.key_prefix, body.ttl_seconds,
    )

    return TokenResponse(
        ephemeral_token = raw_token,
        expires_at      = expires_at,
        ttl_seconds     = body.ttl_seconds,
        project_id      = config.project_id,
    )


@router.post("/rotate")
async def rotate_token(
    body:    RotateTokenRequest,
    request: Request,
) -> RotateTokenResponse:
    """
    Replace an existing (unredeemed) token with a fresh one.

    This is the rotation flow described in Week 8 task 4.  Call this when
    the remaining TTL of the current token drops below your buffer threshold
    (e.g. 30 seconds) and the browser hasn't connected yet.

    The old token is atomically read and deleted; a new token is issued
    in its place.  If the old token has already been redeemed or has
    expired, HTTP 404 is returned — mint a fresh token instead.

    Both the old token validation and new token creation are done with the
    same secret key so cross-project token rotation is impossible.
    """
    key_doc, config = await _require_secret_key(request)
    redis: RedisClient = request.app.state.redis

    # Peek at the old token to get its remaining TTL — without consuming it
    old_ttl = await redis.get_ephemeral_token_ttl(body.current_token)

    if old_ttl < 0:
        # -2 means key doesn't exist; -1 means no TTL (shouldn't happen)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Current token not found — it may have already been redeemed "
                "or expired.  Mint a new token with POST /v1/tokens."
            ),
        )

    # Peek at the payload to verify it belongs to the same project/tenant
    old_payload = await redis.peek_ephemeral_token(body.current_token)
    if old_payload is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Current token payload is missing — mint a new token.",
        )

    # Security: verify the token being rotated belongs to the same project
    # as the secret key being used.  Prevents one project from rotating
    # another project's tokens.
    if old_payload.get("project_id") != config.project_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot rotate a token that belongs to a different project.",
        )

    # Atomically consume the old token
    consumed = await redis.redeem_ephemeral_token(body.current_token)
    if consumed is None:
        # Was redeemed between peek and redeem (very rare race condition)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Token was redeemed between rotation check and deletion "
                "(race condition).  The browser may have already connected."
            ),
        )

    # Issue the new token, preserving the original metadata
    raw_token, expires_at = await redis.create_ephemeral_token(
        project_id     = config.project_id,
        tenant_id      = config.tenant_id,
        api_key_id     = key_doc.id,
        rate_limit_rpm = key_doc.rate_limit_rpm,
        ttl            = body.ttl_seconds,
        metadata       = old_payload.get("metadata", {}),
    )

    log.info(
        "Token rotated: project=%s old_ttl=%ds new_ttl=%ds",
        config.project_id, old_ttl, body.ttl_seconds,
    )

    return RotateTokenResponse(
        ephemeral_token         = raw_token,
        expires_at              = expires_at,
        ttl_seconds             = body.ttl_seconds,
        project_id              = config.project_id,
        previous_remaining_ttl  = old_ttl,
    )