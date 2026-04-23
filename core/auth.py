"""
core/auth.py
────────────
Authentication helpers: password hashing and JWT management.

JWT design
──────────
Two-token scheme:

  Access token   Short-lived (15 min), stateless JWT.  Carries tenant_id,
                 email, and plan in the payload.  Verified by decoding
                 with the shared secret — no DB lookup on every request.

  Refresh token  Long-lived (7 days), opaque random string.  Stored as
                 an argon2 hash in the auth_tokens MongoDB collection.
                 Used only at POST /v1/auth/refresh.

JWT claims
──────────
  sub   tenant_id (UUID string)
  email tenant email
  plan  plan tier
  type  "access" | "refresh"  ← distinguishes the two token types
  exp   expiry (standard claim, checked by python-jose automatically)
  iat   issued-at

Why python-jose
───────────────
python-jose is the standard JWT library for FastAPI.  It supports HS256
(HMAC-SHA256) which is the correct algorithm for symmetric server-to-server
tokens where the same secret signs and verifies.

Password hashing
────────────────
argon2-cffi with higher work factors than API key hashing — passwords are
worth the extra 200–500 ms because they are presented infrequently.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Any

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError
from jose import jwt, JWTError
from core import config

log = logging.getLogger("livechat.auth")

# ── JWT configuration ─────────────────────────────────────────
JWT_SECRET    = config.JWT_SECRET
JWT_ALGORITHM = config.JWT_ALGORITHM
ACCESS_TOKEN_TTL_MINUTES  = config.ACCESS_TOKEN_TTL_MINUTES
REFRESH_TOKEN_TTL_DAYS    = config.REFRESH_TOKEN_TTL_DAYS

# ── Password hasher (stronger than API key hasher) ────────────
_PWD_HASHER = PasswordHasher(
    time_cost   = 3,
    memory_cost = 65536,   # 64 MB
    parallelism = 2,
    hash_len    = 32,
    salt_len    = 16,
)


# ── Password helpers ──────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Return an argon2 hash of the plaintext password."""
    return _PWD_HASHER.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    """
    Return True if plain matches the stored hash.
    Returns False (never raises) on mismatch or bad hash.
    """
    try:
        _PWD_HASHER.verify(hashed, plain)
        return True
    except (VerifyMismatchError, VerificationError, Exception):
        return False


def needs_rehash(hashed: str) -> bool:
    """
    Return True if the stored hash was created with old parameters and
    should be re-hashed on next successful login.
    """
    return _PWD_HASHER.check_needs_rehash(hashed)


# ── JWT helpers ───────────────────────────────────────────────

def create_access_token(tenant_id: str, email: str, plan: str) -> str:
    """
    Create a 15-minute access token.

    The token is stateless — verification requires only the JWT_SECRET,
    not a database lookup.
    """
    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "sub":   tenant_id,
        "email": email,
        "plan":  plan,
        "type":  "access",
        "iat":   now,
        "exp":   now + timedelta(minutes=ACCESS_TOKEN_TTL_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token_payload(tenant_id: str) -> dict[str, Any]:
    """
    Return the JWT payload for a refresh token.

    NOTE: The actual raw token stored in the cookie is the opaque random
    string from AuthTokenRepository.create(), NOT a JWT.  This function
    is kept for reference — in our design refresh tokens are opaque, not
    JWTs, because they are validated against the database anyway.
    """
    # Unused in current flow — refresh tokens are opaque strings.
    # Kept for documentation purposes.
    return {}


def decode_access_token(token: str) -> dict[str, Any] | None:
    """
    Decode and verify an access token.

    Returns the payload dict on success, None on any failure
    (expired, bad signature, wrong type, malformed).
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            log.warning("Token type mismatch — expected 'access'")
            return None
        return payload
    except JWTError as exc:
        log.debug("JWT decode failed: %s", exc)
        return None


def get_tenant_id_from_token(token: str) -> str | None:
    """Convenience wrapper — returns tenant_id or None."""
    payload = decode_access_token(token)
    return payload.get("sub") if payload else None