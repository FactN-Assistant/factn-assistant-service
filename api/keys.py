"""
api/keys.py
───────────
API key management endpoints.  All routes require a valid access token.

The full raw key is returned ONCE on creation — it cannot be recovered.
Subsequent GET requests return only the key_prefix and metadata.

POST /v1/projects/{project_id}/keys         create a new key
GET  /v1/projects/{project_id}/keys         list keys for a project
DELETE /v1/projects/{project_id}/keys/{id}  revoke a key
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel

from core.documents import TenantDoc
from repositories import Repositories

from .dependencies import get_current_tenant

log = logging.getLogger("livechat.api.keys")

router = APIRouter(prefix="/v1/projects", tags=["api-keys"])


# ── Schemas ───────────────────────────────────────────────────

class CreateKeyRequest(BaseModel):
    label:          str = "Default key"
    key_type:       str = "publishable"   # "publishable" | "secret"
    rate_limit_rpm: int = 60


class KeyCreatedResponse(BaseModel):
    """Returned ONCE on key creation — includes the raw key."""
    key_id:         str
    raw_key:        str   # SHOW ONCE — never stored, cannot be recovered
    key_prefix:     str
    key_type:       str
    label:          str
    rate_limit_rpm: int
    created_at:     str


class KeySummaryResponse(BaseModel):
    """Safe to return on list — never includes the raw key or hash."""
    key_id:         str
    key_prefix:     str
    key_type:       str
    label:          str
    rate_limit_rpm: int
    revoked:        bool
    created_at:     str
    last_used_at:   str | None


# ── Endpoints ─────────────────────────────────────────────────

@router.post("/{project_id}/keys", status_code=status.HTTP_201_CREATED)
async def create_key(
    project_id: str,
    body:       CreateKeyRequest,
    request:    Request,
    tenant:     TenantDoc = Depends(get_current_tenant),
) -> KeyCreatedResponse:
    """
    Generate a new API key for a project.

    Returns the raw key ONCE — store it securely.  It cannot be recovered.
    """
    repos: Repositories = request.app.state.repos

    # Verify project belongs to this tenant
    project = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")

    if body.key_type not in ("publishable", "secret"):
        raise HTTPException(
            status_code=422,
            detail="key_type must be 'publishable' or 'secret'.",
        )

    key_doc, raw_key = await repos.api_keys.create(
        project_id     = project_id,
        tenant_id      = tenant.id,
        label          = body.label,
        key_type       = body.key_type,
        rate_limit_rpm = body.rate_limit_rpm,
    )

    log.info(
        "API key created: %s (project=%s type=%s)",
        key_doc.key_prefix, project_id, body.key_type,
    )

    return KeyCreatedResponse(
        key_id         = key_doc.id,
        raw_key        = raw_key,
        key_prefix     = key_doc.key_prefix,
        key_type       = key_doc.key_type,
        label          = key_doc.label,
        rate_limit_rpm = key_doc.rate_limit_rpm,
        created_at     = key_doc.created_at.isoformat(),
    )


@router.get("/{project_id}/keys")
async def list_keys(
    project_id: str,
    request:    Request,
    tenant:     TenantDoc = Depends(get_current_tenant),
) -> list[KeySummaryResponse]:
    """List all API keys for a project (prefix only — no raw keys or hashes)."""
    repos: Repositories = request.app.state.repos

    project = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")

    keys = await repos.api_keys.list_for_project(project_id)
    return [
        KeySummaryResponse(
            key_id         = k.id,
            key_prefix     = k.key_prefix,
            key_type       = k.key_type,
            label          = k.label,
            rate_limit_rpm = k.rate_limit_rpm,
            revoked        = k.revoked,
            created_at     = k.created_at.isoformat(),
            last_used_at   = k.last_used_at.isoformat() if k.last_used_at else None,
        )
        for k in keys
    ]


@router.delete(
    "/{project_id}/keys/{key_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def revoke_key(
    project_id: str,
    key_id:     str,
    request:    Request,
    tenant:     TenantDoc = Depends(get_current_tenant),
) -> None:
    """
    Revoke an API key permanently.

    Revocation takes effect immediately for all new WebSocket connections.
    Existing active sessions are NOT forcibly closed — they continue until
    the session TTL expires or the client disconnects.
    """
    repos: Repositories = request.app.state.repos

    project = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found.")

    revoked = await repos.api_keys.revoke(key_id, tenant_id=tenant.id)
    if not revoked:
        raise HTTPException(status_code=404, detail="Key not found or already revoked.")

    log.info("API key revoked: %s (project=%s)", key_id, project_id)