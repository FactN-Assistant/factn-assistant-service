"""
api/projects.py
───────────────
Project CRUD + Tool CRUD + synchronous test endpoint + sub-resource PUT endpoints.

Route map
─────────
  POST   /v1/projects                              create project
  GET    /v1/projects                              list all projects for tenant
  GET    /v1/projects/{id}                         get single project
  PATCH  /v1/projects/{id}                         update project fields
  DELETE /v1/projects/{id}                         soft-delete project

  POST   /v1/projects/{id}/tools                   add a tool
  GET    /v1/projects/{id}/tools                   list tools
  PATCH  /v1/projects/{id}/tools/{tool_name}       update a tool
  DELETE /v1/projects/{id}/tools/{tool_name}       remove a tool

  POST   /v1/projects/{id}/test                    synchronous test turn

All routes require a valid access token (Depends(get_current_tenant)).
All write operations invalidate the Redis project config cache automatically
via ProjectRepository.update() — the next WebSocket connection picks up
the new configuration within milliseconds.

Tool schema validation
──────────────────────
Tool parameters are validated as a JSON Schema object (OpenAPI 3.0 subset).
validate_tool_parameters() enforces the constraints Gemini requires:
  • top-level type must be "object"
  • properties must be a dict
  • each property must have a type field
  • type values must be one of the Gemini-supported primitives
  • required must be a list of strings that exist in properties

Test endpoint design
────────────────────
POST /v1/projects/{id}/test sends a single text message to a temporary
Gemini session, waits for the full response, and returns it synchronously.
This lets dashboard users validate their system prompt and tools without
connecting a WebSocket client.

The test session is completely isolated from production sessions — it uses
a fresh Gemini connection that is closed immediately after the turn.

Changes from previous version
──────────────────────────────
  POST /v1/projects
    — Enforces plan.max_projects limit.
    — Caps session_ttl_seconds to plan.max_session_ttl_seconds.
    — Caps rate_limit_rpm to plan.max_rate_limit_rpm.
 
  POST /v1/projects/{id}/tools
    — Enforces plan.max_tools_per_project limit.
    — Caps tool timeout_ms to plan.max_webhook_timeout_ms.
 
  POST /v1/projects/{id}/keys (unchanged — in api/keys.py)
    — rate_limit_rpm now capped there too (see api/keys.py).
 
  NEW: PUT /v1/projects/{id}/system-prompt
  NEW: PUT /v1/projects/{id}/voice-config
  NEW: PUT /v1/projects/{id}/webhook-config
    — Dedicated endpoints for the three sidebar sections that users
      iterate on most.  Each is atomic, independently retryable, and
      produces a clean audit trail.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime, timezone, timedelta

from core import plan_limits
from core.documents import ProjectDoc, TenantDoc
from core.gemini_runner import build_gemini_config
from core.schemas import (
    ToolDefinition,
    VADConfig,
    VoiceConfig,
)
from repositories import Repositories

from .dependencies import get_current_tenant

log = logging.getLogger("livechat.api.projects")

router = APIRouter(prefix="/v1/projects", tags=["projects"])

_VALID_PARAM_TYPES = {"string", "number", "integer", "boolean", "array", "object"}

_SUPPORTED_MODELS = {
    "gemini-2.5-flash-native-audio-preview-12-2025",
    "gemini-2.0-flash-live-001",
}


# ── Tool parameter validation ─────────────────────────────────

def validate_tool_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        raise ValueError("parameters must be a JSON object.")
    if parameters.get("type") != "object":
        raise ValueError('parameters.type must be "object".')
    properties = parameters.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError("parameters.properties must be a JSON object.")
    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            raise ValueError(f'Property "{prop_name}" must be a JSON object.')
        prop_type = prop_schema.get("type")
        if not prop_type:
            raise ValueError(f'Property "{prop_name}" is missing a "type" field.')
        if prop_type not in _VALID_PARAM_TYPES:
            raise ValueError(
                f'Property "{prop_name}" has unsupported type "{prop_type}". '
                f"Supported: {sorted(_VALID_PARAM_TYPES)}."
            )
        if "enum" in prop_schema:
            if not isinstance(prop_schema["enum"], list) or not prop_schema["enum"]:
                raise ValueError(f'Property "{prop_name}".enum must be a non-empty list.')
    required = parameters.get("required", [])
    if not isinstance(required, list):
        raise ValueError("parameters.required must be a list of strings.")
    for r in required:
        if not isinstance(r, str):
            raise ValueError("Every entry in parameters.required must be a string.")
        if r not in properties:
            raise ValueError(f'Required field "{r}" not in parameters.properties.')
    return parameters


# ── Request / Response schemas ────────────────────────────────

class VoiceConfigRequest(BaseModel):
    voice_name:    str  = "Kore"
    language_code: str  = "en-US"
    enabled:       bool = True

    @field_validator("language_code", mode="before")
    @classmethod
    def validate_language_code(cls, v: str) -> str:
        """Coerce empty strings to the default value before validation."""
        if isinstance(v, str):
            v = v.strip()
        return v or "en-US"


class VADConfigRequest(BaseModel):
    mode: str = "manual"

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("manual", "auto"):
            raise ValueError('mode must be "manual" or "auto".')
        return v


class CreateProjectRequest(BaseModel):
    name:                    str   = Field(min_length=1, max_length=100)
    description:             str   = Field(default="", max_length=500)
    system_prompt:           str   = Field(min_length=1, max_length=32_000)
    gemini_model:            str   = "gemini-2.5-flash-native-audio-preview-12-2025"
    voice_config:            VoiceConfigRequest  = Field(default_factory=VoiceConfigRequest)
    vad_config:              VADConfigRequest    = Field(default_factory=VADConfigRequest)
    session_ttl_seconds:     int   = Field(default=300, ge=30, le=7200)
    max_concurrent_sessions: int   = Field(default=10, ge=1, le=500)
    rate_limit_rpm:          int   = Field(default=60, ge=1, le=1000)
    webhook_url:             str | None = None
    webhook_secret:          str | None = None
    allowed_origins:         list[str]  = Field(default_factory=list)

    @field_validator("gemini_model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in _SUPPORTED_MODELS:
            raise ValueError(f'Unsupported model "{v}". Supported: {sorted(_SUPPORTED_MODELS)}.')
        return v


class UpdateProjectRequest(BaseModel):
    name:                    str | None   = Field(default=None, min_length=1, max_length=100)
    description:             str | None   = Field(default=None, max_length=500)
    system_prompt:           str | None   = Field(default=None, min_length=1, max_length=32_000)
    gemini_model:            str | None   = None
    voice_config:            VoiceConfigRequest | None = None
    vad_config:              VADConfigRequest   | None = None
    session_ttl_seconds:     int | None   = Field(default=None, ge=30, le=7200)
    max_concurrent_sessions: int | None   = Field(default=None, ge=1, le=500)
    rate_limit_rpm:          int | None   = Field(default=None, ge=1, le=1000)
    webhook_url:             str | None   = None
    webhook_secret:          str | None   = None
    allowed_origins:         list[str] | None = None

    @field_validator("gemini_model")
    @classmethod
    def validate_model(cls, v: str | None) -> str | None:
        if v is not None and v not in _SUPPORTED_MODELS:
            raise ValueError(f'Unsupported model "{v}".')
        return v


# ── Sub-resource request schemas (new PUT endpoints) ──────────

class SystemPromptRequest(BaseModel):
    """Body for PUT /v1/projects/{id}/system-prompt."""
    system_prompt: str = Field(min_length=1, max_length=32_000)


class VoiceConfigUpdateRequest(BaseModel):
    """Body for PUT /v1/projects/{id}/voice-config."""
    voice_name:    str  = Field(default="Kore",  min_length=1, max_length=64)
    language_code: str  = Field(default="en-US", min_length=2, max_length=16)
    enabled:       bool = True
    vad_mode:      str  = "manual"

    @field_validator("language_code", mode="before")
    @classmethod
    def validate_language_code(cls, v: str) -> str:
        """Coerce empty strings to the default value before validation."""
        if isinstance(v, str):
            v = v.strip()
        return v or "en-US"

    @field_validator("vad_mode")
    @classmethod
    def validate_vad(cls, v: str) -> str:
        if v not in ("manual", "auto"):
            raise ValueError('vad_mode must be "manual" or "auto".')
        return v
 
 
class WebhookConfigRequest(BaseModel):
    """Body for PUT /v1/projects/{id}/webhook-config."""
    webhook_url:     str | None  = None
    webhook_secret:  str | None  = None
    allowed_origins: list[str]   = Field(default_factory=list)


class ProjectResponse(BaseModel):
    project_id:              str
    tenant_id:               str
    name:                    str
    description:             str
    system_prompt:           str
    gemini_model:            str
    voice_config:            dict
    vad_config:              dict
    tools:                   list[dict]
    webhook_url:             str | None
    allowed_origins:         list[str]
    session_ttl_seconds:     int
    max_concurrent_sessions: int
    rate_limit_rpm:          int
    is_active:               bool
    created_at:              str
    updated_at:              str

    @classmethod
    def from_doc(cls, doc: ProjectDoc) -> "ProjectResponse":
        return cls(
            project_id              = doc.id,
            tenant_id               = doc.tenant_id,
            name                    = doc.name,
            description             = doc.description,
            system_prompt           = doc.system_prompt,
            gemini_model            = doc.gemini_model,
            voice_config            = doc.voice_config.model_dump(),
            vad_config              = doc.vad_config.model_dump(),
            tools                   = [
                {k: v for k, v in t.model_dump().items() if k != "webhook_secret"}
                for t in doc.tools
            ],
            webhook_url             = doc.webhook_url,
            allowed_origins         = doc.allowed_origins,
            session_ttl_seconds     = doc.session_ttl_seconds,
            max_concurrent_sessions = doc.max_concurrent_sessions,
            rate_limit_rpm          = doc.rate_limit_rpm,
            is_active               = doc.is_active,
            created_at              = doc.created_at.isoformat(),
            updated_at              = doc.updated_at.isoformat(),
        )


# ── Tool request / response schemas ──────────────────────────

class CreateToolRequest(BaseModel):
    name:            str  = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    description:     str  = Field(min_length=1, max_length=1000)
    parameters:      dict[str, Any]
    execution_mode:  str  = "static"
    static_response: dict[str, Any] | None = None
    webhook_url:     str | None = None
    webhook_secret:  str | None = None
    timeout_ms:      int  = Field(default=5000, ge=100, le=30_000)

    @field_validator("execution_mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("static", "webhook"):
            raise ValueError('execution_mode must be "static" or "webhook".')
        return v

    @field_validator("parameters")
    @classmethod
    def validate_params(cls, v: dict) -> dict:
        return validate_tool_parameters(v)

    @model_validator(mode="after")
    def validate_execution_fields(self) -> "CreateToolRequest":
        if self.execution_mode == "static" and self.static_response is None:
            raise ValueError('static_response is required when execution_mode is "static".')
        if self.execution_mode == "webhook" and not self.webhook_url:
            raise ValueError('webhook_url is required when execution_mode is "webhook".')
        return self


class UpdateToolRequest(BaseModel):
    description:     str | None             = Field(default=None, min_length=1, max_length=1000)
    parameters:      dict[str, Any] | None  = None
    execution_mode:  str | None             = None
    static_response: dict[str, Any] | None  = None
    webhook_url:     str | None             = None
    webhook_secret:  str | None             = None
    timeout_ms:      int | None             = Field(default=None, ge=100, le=30_000)

    @field_validator("execution_mode")
    @classmethod
    def validate_mode(cls, v: str | None) -> str | None:
        if v is not None and v not in ("static", "webhook"):
            raise ValueError('execution_mode must be "static" or "webhook".')
        return v

    @field_validator("parameters")
    @classmethod
    def validate_params(cls, v: dict | None) -> dict | None:
        if v is not None:
            return validate_tool_parameters(v)
        return v


class ToolResponse(BaseModel):
    name:            str
    description:     str
    parameters:      dict[str, Any]
    execution_mode:  str
    static_response: dict[str, Any] | None
    webhook_url:     str | None
    timeout_ms:      int


class TestRequest(BaseModel):
    text:            str = Field(min_length=1, max_length=4000)
    timeout_seconds: int = Field(default=30, ge=5, le=120)


class TestResponse(BaseModel):
    assistant_text: str
    tool_calls:     list[dict]
    input_tokens:   int
    output_tokens:  int
    latency_ms:     int


# ── Helper ────────────────────────────────────────────────────

def _get_project_or_404(doc: ProjectDoc | None, project_id: str) -> ProjectDoc:
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project '{project_id}' not found.",
        )
    return doc


# ══════════════════════════════════════════════════════════════
# PROJECT CRUD
# ══════════════════════════════════════════════════════════════

@router.post("", status_code=status.HTTP_201_CREATED)
async def create_project(
    body:    CreateProjectRequest,
    request: Request,
    tenant:  TenantDoc = Depends(get_current_tenant),
) -> ProjectResponse:
    """
    Create a new chatbot project.
 
    Plan enforcement
    ────────────────
    • Checks the tenant's current project count against plan.max_projects.
    • Caps session_ttl_seconds to plan.max_session_ttl_seconds.
    • Caps rate_limit_rpm to plan.max_rate_limit_rpm.
    """
    repos: Repositories = request.app.state.repos
 
    # ── Plan: project count limit ──────────────────────────────
    existing_projects = await repos.projects.list_for_tenant(tenant.id, limit=1000)
    if not plan_limits.is_within_project_limit(tenant.plan, len(existing_projects)):
        limit = plan_limits.max_projects(tenant.plan)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Your {tenant.plan} plan allows a maximum of {limit} project(s). "
                "Upgrade your plan to create more projects."
            ),
        )

    # ── Plan: cap session TTL and rate limit ───────────────────
    effective_ttl = plan_limits.effective_session_ttl(tenant.plan, body.session_ttl_seconds)
    effective_rpm = plan_limits.effective_rate_limit_rpm(tenant.plan, body.rate_limit_rpm)

    doc = await repos.projects.create(
        tenant_id               = tenant.id,
        name                    = body.name,
        system_prompt           = body.system_prompt,
        description             = body.description,
        gemini_model            = body.gemini_model,
        voice_config            = VoiceConfig(**body.voice_config.model_dump()),
        vad_config              = VADConfig(mode=body.vad_config.mode),
        session_ttl_seconds     = effective_ttl,
        max_concurrent_sessions = body.max_concurrent_sessions,
        rate_limit_rpm          = effective_rpm,
        webhook_url             = body.webhook_url,
        webhook_secret          = body.webhook_secret,
        allowed_origins         = body.allowed_origins,
    )
    log.info(
        "Project created: %s (tenant=%s plan=%s ttl=%d rpm=%d)",
        doc.id, tenant.id, tenant.plan, effective_ttl, effective_rpm,
    )
    return ProjectResponse.from_doc(doc)


@router.get("")
async def list_projects(
    request: Request,
    limit:   int = 50,
    skip:    int = 0,
    tenant:  TenantDoc = Depends(get_current_tenant),
) -> list[ProjectResponse]:
    repos: Repositories = request.app.state.repos
    docs = await repos.projects.list_for_tenant(
        tenant_id = tenant.id,
        limit     = min(limit, 100),
        skip      = skip,
    )
    return [ProjectResponse.from_doc(d) for d in docs]


@router.get("/{project_id}")
async def get_project(
    project_id: str,
    request:    Request,
    tenant:     TenantDoc = Depends(get_current_tenant),
) -> ProjectResponse:
    repos: Repositories = request.app.state.repos
    doc = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    return ProjectResponse.from_doc(_get_project_or_404(doc, project_id))


@router.patch("/{project_id}")
async def update_project(
    project_id: str,
    body:       UpdateProjectRequest,
    request:    Request,
    tenant:     TenantDoc = Depends(get_current_tenant),
) -> ProjectResponse:
    """
    Partially update a project — name, description, model, session limits.

    Plan enforcement: session_ttl_seconds and rate_limit_rpm are capped
    to the tenant's plan limits if provided.
    """
    repos: Repositories = request.app.state.repos

    doc = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    _get_project_or_404(doc, project_id)

    updates: dict[str, Any] = {}
    if body.name                    is not None: updates["name"]                    = body.name
    if body.description             is not None: updates["description"]             = body.description
    if body.system_prompt           is not None: updates["system_prompt"]           = body.system_prompt
    if body.gemini_model            is not None: updates["gemini_model"]            = body.gemini_model
    if body.max_concurrent_sessions is not None: updates["max_concurrent_sessions"] = body.max_concurrent_sessions
    if body.webhook_url             is not None: updates["webhook_url"]             = body.webhook_url
    if body.webhook_secret          is not None: updates["webhook_secret"]          = body.webhook_secret
    if body.allowed_origins         is not None: updates["allowed_origins"]         = body.allowed_origins
    if body.voice_config            is not None: updates["voice_config"]            = body.voice_config.model_dump()
    if body.vad_config              is not None: updates["vad_config"]              = body.vad_config.model_dump()

    # Cap plan-sensitive fields
    if body.session_ttl_seconds is not None:
        updates["session_ttl_seconds"] = plan_limits.effective_session_ttl(
            tenant.plan, body.session_ttl_seconds
        )
    if body.rate_limit_rpm is not None:
        updates["rate_limit_rpm"] = plan_limits.effective_rate_limit_rpm(
            tenant.plan, body.rate_limit_rpm
        )

    if not updates:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No fields provided to update.",
        )

    await repos.projects.update(project_id, tenant.id, updates)
    updated = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    return ProjectResponse.from_doc(updated)  # type: ignore[arg-type]


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    request:    Request,
    tenant:     TenantDoc = Depends(get_current_tenant),
) -> None:
    repos: Repositories = request.app.state.repos
    doc = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    _get_project_or_404(doc, project_id)
    await repos.projects.soft_delete(project_id, tenant.id)
    log.info("Project soft-deleted: %s (tenant=%s)", project_id, tenant.id)


# ══════════════════════════════════════════════════════════════
# SUB-RESOURCE PUT ENDPOINTS  (new in this release)
# Each maps to one sidebar section in the dashboard.
# All three follow the same pattern:
#   1. Verify project ownership
#   2. Validate the sub-resource payload
#   3. Write only the relevant fields via repos.projects.update()
#   4. Return the full updated ProjectResponse
# ══════════════════════════════════════════════════════════════

@router.put("/{project_id}/system-prompt")
async def update_system_prompt(
    project_id: str,
    body:       SystemPromptRequest,
    request:    Request,
    tenant:     TenantDoc = Depends(get_current_tenant),
) -> ProjectResponse:
    """
    Replace the project system prompt.

    Maps to the System Prompt sidebar tab save button.
    Invalidates the Redis project config cache — next session picks up
    the new prompt immediately.
    """
    repos: Repositories = request.app.state.repos
    doc = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    _get_project_or_404(doc, project_id)

    await repos.projects.update(
        project_id, tenant.id,
        {"system_prompt": body.system_prompt},
    )
    log.info("System prompt updated: project=%s tenant=%s", project_id, tenant.id)
 
    updated = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    return ProjectResponse.from_doc(updated)  # type: ignore[arg-type]


@router.put("/{project_id}/voice-config")
async def update_voice_config(
    project_id: str,
    body:       VoiceConfigUpdateRequest,
    request:    Request,
    tenant:     TenantDoc = Depends(get_current_tenant),
) -> ProjectResponse:
    """
    Replace the project voice and VAD configuration.
 
    Maps to the Voice Config sidebar tab save button.
    Stores voice_config and vad_config as a single atomic update.
    Next Gemini session opened for this project uses the new voice.
    """
    repos: Repositories = request.app.state.repos
    doc = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    _get_project_or_404(doc, project_id)
 
    await repos.projects.update(
        project_id, tenant.id,
        {
            "voice_config": {
                "voice_name":    body.voice_name,
                "language_code": body.language_code,
                "enabled":       body.enabled,
            },
            "vad_config": {"mode": body.vad_mode},
        },
    )
    log.info("Voice config updated: project=%s tenant=%s", project_id, tenant.id)
 
    updated = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    return ProjectResponse.from_doc(updated)  # type: ignore[arg-type]


@router.put("/{project_id}/webhook-config")
async def update_webhook_config(
    project_id: str,
    body:       WebhookConfigRequest,
    request:    Request,
    tenant:     TenantDoc = Depends(get_current_tenant),
) -> ProjectResponse:
    """
    Replace the project webhook and CORS configuration.

    Maps to the Webhook / Security sidebar tab save button.
    Updates webhook_url, webhook_secret, and allowed_origins atomically.
    CORS enforcement on new WebSocket connections takes effect immediately
    (Redis cache is invalidated, allowed_origins is on ProjectConfig).
    """
    repos: Repositories = request.app.state.repos
    doc = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    _get_project_or_404(doc, project_id)
 
    await repos.projects.update(
        project_id, tenant.id,
        {
            "webhook_url":     body.webhook_url,
            "webhook_secret":  body.webhook_secret,
            "allowed_origins": body.allowed_origins,
        },
    )
    log.info("Webhook config updated: project=%s tenant=%s", project_id, tenant.id)

    updated = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    return ProjectResponse.from_doc(updated)  # type: ignore[arg-type]


# ══════════════════════════════════════════════════════════════
# TOOL CRUD
# ══════════════════════════════════════════════════════════════

@router.post("/{project_id}/tools", status_code=status.HTTP_201_CREATED)
async def add_tool(
    project_id: str,
    body:       CreateToolRequest,
    request:    Request,
    tenant:     TenantDoc = Depends(get_current_tenant),
) -> ToolResponse:
    """
    Add a new tool definition to a project.

    Plan enforcement
    ────────────────
    • Checks tool count against plan.max_tools_per_project.
    • Caps tool timeout_ms to plan.max_webhook_timeout_ms.
    """
    repos: Repositories = request.app.state.repos

    doc = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    _get_project_or_404(doc, project_id)

    if any(t.name == body.name for t in doc.tools):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A tool named '{body.name}' already exists in this project.",
        )

    # ── Plan: tool count limit ─────────────────────────────────
    if not plan_limits.is_within_tool_limit(tenant.plan, len(doc.tools)):
        limit = plan_limits.max_tools_per_project(tenant.plan)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Your {tenant.plan} plan allows a maximum of {limit} tool(s) per project. "
                "Upgrade your plan to add more tools."
            ),
        )

    # ── Plan: cap webhook timeout ──────────────────────────────
    effective_timeout = plan_limits.effective_webhook_timeout_ms(tenant.plan, body.timeout_ms)

    new_tool = ToolDefinition(
        name            = body.name,
        description     = body.description,
        parameters      = body.parameters,
        execution_mode  = body.execution_mode,
        static_response = body.static_response,
        webhook_url     = body.webhook_url,
        webhook_secret  = body.webhook_secret,
        timeout_ms      = effective_timeout,
    )

    updated_tools = list(doc.tools) + [new_tool]
    await repos.projects.set_tools(project_id, tenant.id, updated_tools)
    log.info(
        "Tool added: %s → project %s (plan=%s timeout_ms=%d)",
        body.name, project_id, tenant.plan, effective_timeout,
    )

    return _tool_to_response(new_tool)


@router.get("/{project_id}/tools")
async def list_tools(
    project_id: str,
    request:    Request,
    tenant:     TenantDoc = Depends(get_current_tenant),
) -> list[ToolResponse]:
    repos: Repositories = request.app.state.repos
    doc = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    _get_project_or_404(doc, project_id)
    return [_tool_to_response(t) for t in doc.tools]


@router.patch("/{project_id}/tools/{tool_name}")
async def update_tool(
    project_id: str,
    tool_name:  str,
    body:       UpdateToolRequest,
    request:    Request,
    tenant:     TenantDoc = Depends(get_current_tenant),
) -> ToolResponse:
    repos: Repositories = request.app.state.repos

    doc = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    _get_project_or_404(doc, project_id)

    tool_index = next(
        (i for i, t in enumerate(doc.tools) if t.name == tool_name), None
    )
    if tool_index is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found in project '{project_id}'.",
        )

    existing = doc.tools[tool_index]
    updated_fields = existing.model_dump()

    if body.description     is not None: updated_fields["description"]     = body.description
    if body.parameters      is not None: updated_fields["parameters"]      = body.parameters
    if body.execution_mode  is not None: updated_fields["execution_mode"]  = body.execution_mode
    if body.static_response is not None: updated_fields["static_response"] = body.static_response
    if body.webhook_url     is not None: updated_fields["webhook_url"]     = body.webhook_url
    if body.webhook_secret  is not None: updated_fields["webhook_secret"]  = body.webhook_secret
    if body.timeout_ms      is not None:
        # Cap updated timeout to plan limit
        updated_fields["timeout_ms"] = plan_limits.effective_webhook_timeout_ms(
            tenant.plan, body.timeout_ms
        )

    updated_tool = ToolDefinition.model_validate(updated_fields)

    if updated_tool.execution_mode == "static" and updated_tool.static_response is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='static_response required when execution_mode is "static".',
        )
    if updated_tool.execution_mode == "webhook" and not updated_tool.webhook_url:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='webhook_url required when execution_mode is "webhook".',
        )

    new_tools = list(doc.tools)
    new_tools[tool_index] = updated_tool
    await repos.projects.set_tools(project_id, tenant.id, new_tools)
    log.info("Tool updated: %s in project %s", tool_name, project_id)

    return _tool_to_response(updated_tool)


@router.delete(
    "/{project_id}/tools/{tool_name}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_tool(
    project_id: str,
    tool_name:  str,
    request:    Request,
    tenant:     TenantDoc = Depends(get_current_tenant),
) -> None:
    repos: Repositories = request.app.state.repos

    doc = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    _get_project_or_404(doc, project_id)

    new_tools = [t for t in doc.tools if t.name != tool_name]
    if len(new_tools) == len(doc.tools):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_name}' not found in project '{project_id}'.",
        )

    await repos.projects.set_tools(project_id, tenant.id, new_tools)
    log.info("Tool deleted: %s from project %s", tool_name, project_id)


# ══════════════════════════════════════════════════════════════
# TEST ENDPOINT
# ══════════════════════════════════════════════════════════════

@router.post("/{project_id}/test")
async def test_project(
    project_id: str,
    body:       TestRequest,
    request:    Request,
    tenant:     TenantDoc = Depends(get_current_tenant),
) -> TestResponse:
    repos:         Repositories = request.app.state.repos
    gemini_client: genai.Client = request.app.state.session_manager._client

    doc = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    _get_project_or_404(doc, project_id)

    if not doc.is_active:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Cannot test an inactive project.",
        )

    project_config = doc.to_project_config()
    gemini_config  = build_gemini_config(project_config)

    try:
        result = await asyncio.wait_for(
            _run_test_turn(gemini_client, project_config.gemini_model, gemini_config, body.text),
            timeout=float(body.timeout_seconds),
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Gemini did not respond within {body.timeout_seconds} seconds.",
        )
    except Exception as exc:
        log.error("Test turn failed for project %s: %s", project_id, exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Gemini error: {exc}",
        )

    return result


async def _run_test_turn(
    client:  genai.Client,
    model:   str,
    config:  types.LiveConnectConfig,
    text:    str,
) -> TestResponse:
    import time
    start_ms = int(time.monotonic() * 1000)

    assistant_parts: list[str]  = []
    tool_calls_log:  list[dict] = []
    input_tokens  = 0
    output_tokens = 0

    async with client.aio.live.connect(model=model, config=config) as gsession:
        await gsession.send_client_content(
            turns=[types.Content(role="user", parts=[types.Part(text=text)])],
            turn_complete=True,
        )

        async for response in gsession.receive():
            if response.usage_metadata:
                usage = response.usage_metadata
                input_tokens  = getattr(usage, "prompt_token_count", 0) or 0
                # Try response_token_count first (newer SDK)
                output_tokens = getattr(usage, "response_token_count", 0) or 0
                # Fall back to response_tokens_details (Live API modality breakdown)
                if not output_tokens:
                    details = getattr(usage, "response_tokens_details", None)
                    if details:
                        output_tokens = sum(getattr(d, "token_count", 0) or 0 for d in details)
                # Last resort: derive from total - prompt
                if not output_tokens:
                    total = getattr(usage, "total_token_count", 0) or 0
                    if total and input_tokens:
                        output_tokens = total - input_tokens
                    elif total:
                        output_tokens = total

            sc = response.server_content
            if sc is not None:
                if sc.output_transcription and sc.output_transcription.text:
                    assistant_parts.append(sc.output_transcription.text)
                if sc.model_turn and sc.model_turn.parts:
                    for part in sc.model_turn.parts:
                        if part.text:
                            assistant_parts.append(part.text)
                if sc.turn_complete or sc.interrupted:
                    break

            if response.tool_call:
                function_responses = []
                for fc in response.tool_call.function_calls:
                    args   = dict(fc.args) if fc.args else {}
                    result = {"note": "Test mode — static response only."}
                    tool_calls_log.append({"tool": fc.name, "args": args, "result": result})
                    function_responses.append(
                        types.FunctionResponse(
                            id=getattr(fc, "id", ""),
                            name=fc.name,
                            response={"result": result},
                        )
                    )
                await gsession.send_tool_response(function_responses=function_responses)

    latency_ms = int(time.monotonic() * 1000) - start_ms
    return TestResponse(
        assistant_text = " ".join(assistant_parts).strip(),
        tool_calls     = tool_calls_log,
        input_tokens   = input_tokens,
        output_tokens  = output_tokens,
        latency_ms     = latency_ms,
    )


# ══════════════════════════════════════════════════════════════
# SESSION ANALYTICS
# ══════════════════════════════════════════════════════════════

class SessionSummaryResponse(BaseModel):
    session_id:       str
    project_id:       str
    status:           str
    started_at:       str
    ended_at:         str
    duration_seconds: float
    turns:            int
    tool_calls:       int
    input_tokens:     int
    output_tokens:    int
    api_key_id:       str
    error_message:    str | None


class SessionListResponse(BaseModel):
    sessions: list[SessionSummaryResponse]
    total:    int
    limit:    int
    skip:     int


class UsageSummaryResponse(BaseModel):
    project_id:           str
    since:                str
    until:                str
    total_sessions:       int
    total_turns:          int
    total_tool_calls:     int
    total_duration_s:     float
    avg_duration_s:       float
    total_input_tokens:   int
    total_output_tokens:  int
    total_tokens:         int
    error_count:          int
    error_rate_pct:       float
 
 
@router.get("/{project_id}/sessions")
async def list_sessions(
    project_id: str,
    request:    Request,
    limit:      int = 50,
    skip:       int = 0,
    status:     str | None = None,
    tenant:     TenantDoc = Depends(get_current_tenant),
) -> SessionListResponse:
    repos: Repositories = request.app.state.repos
    doc = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    _get_project_or_404(doc, project_id)
    limit = max(1, min(limit, 200))
    sessions = await repos.sessions.list_for_project(
        project_id=project_id, limit=limit, skip=skip, status=status,
    )
    return SessionListResponse(
        sessions=[
            SessionSummaryResponse(
                session_id       = s.id,
                project_id       = s.project_id,
                status           = s.status,
                started_at       = s.started_at.isoformat(),
                ended_at         = s.ended_at.isoformat(),
                duration_seconds = s.duration_seconds,
                turns            = s.turns,
                tool_calls       = s.tool_calls,
                input_tokens     = s.input_tokens,
                output_tokens    = s.output_tokens,
                api_key_id       = s.api_key_id,
                error_message    = s.error_message,
            )
            for s in sessions
        ],
        total=len(sessions), limit=limit, skip=skip,
    )


@router.get("/{project_id}/sessions/{session_id_path}")
async def get_session(
    project_id:      str,
    session_id_path: str,
    request:         Request,
    tenant:          TenantDoc = Depends(get_current_tenant),
) -> SessionSummaryResponse:
    repos: Repositories = request.app.state.repos
    doc = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    _get_project_or_404(doc, project_id)
    session = await repos.sessions.get_by_id(session_id_path, project_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id_path}' not found.",
        )
    return SessionSummaryResponse(
        session_id       = session.id,
        project_id       = session.project_id,
        status           = session.status,
        started_at       = session.started_at.isoformat(),
        ended_at         = session.ended_at.isoformat(),
        duration_seconds = session.duration_seconds,
        turns            = session.turns,
        tool_calls       = session.tool_calls,
        input_tokens     = session.input_tokens,
        output_tokens    = session.output_tokens,
        api_key_id       = session.api_key_id,
        error_message    = session.error_message,
    )


@router.get("/{project_id}/usage")
async def get_usage(
    project_id: str,
    request:    Request,
    days:       int = 30,
    tenant:     TenantDoc = Depends(get_current_tenant),
) -> UsageSummaryResponse:
    repos: Repositories = request.app.state.repos
    doc = await repos.projects.get_by_id(project_id, tenant_id=tenant.id)
    _get_project_or_404(doc, project_id)
    days  = max(1, min(days, 365))
    since = datetime.now(timezone.utc) - timedelta(days=days)
    until = datetime.now(timezone.utc)
    summary = await repos.sessions.usage_summary(project_id=project_id, since=since)
    total_sessions = summary.get("total_sessions", 0)
    total_dur      = summary.get("total_duration_s", 0.0)
    error_count    = summary.get("error_count", 0)
    total_in       = summary.get("total_input_tokens", 0)
    total_out      = summary.get("total_output_tokens", 0)
    return UsageSummaryResponse(
        project_id          = project_id,
        since               = since.isoformat(),
        until               = until.isoformat(),
        total_sessions      = total_sessions,
        total_turns         = summary.get("total_turns", 0),
        total_tool_calls    = summary.get("total_tool_calls", 0),
        total_duration_s    = round(total_dur, 2),
        avg_duration_s      = round(total_dur / total_sessions, 2) if total_sessions else 0.0,
        total_input_tokens  = total_in,
        total_output_tokens = total_out,
        total_tokens        = total_in + total_out,
        error_count         = error_count,
        error_rate_pct      = round(error_count / total_sessions * 100, 1) if total_sessions else 0.0,
    )


# ── Internal helper ───────────────────────────────────────────

def _tool_to_response(tool: ToolDefinition) -> ToolResponse:
    return ToolResponse(
        name            = tool.name,
        description     = tool.description,
        parameters      = tool.parameters,
        execution_mode  = tool.execution_mode,
        static_response = tool.static_response,
        webhook_url     = tool.webhook_url,
        timeout_ms      = tool.timeout_ms,
    )
 