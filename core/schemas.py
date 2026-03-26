"""
core/schemas.py
───────────────
Pydantic models used by the session engine at runtime.

These are PURE DATA SHAPES — no database or auth logic.
The repository layer populates them by converting MongoDB documents.

Relationship to core/documents.py
──────────────────────────────────
  documents.py  = full MongoDB document shapes (all DB fields)
  schemas.py    = lean runtime shapes (only what Gemini runner needs)

  ProjectDoc.to_project_config() bridges the two.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Tool definition
# ──────────────────────────────────────────────────────────────

class ToolDefinition(BaseModel):
    """
    One function the Gemini model may call.

    execution_mode
        "static"  → return static_response directly (demos / mocks)
        "webhook" → POST to webhook_url with the call arguments
    """
    name:            str
    description:     str
    parameters:      dict[str, Any]                   # raw JSON Schema
    execution_mode:  Literal["static", "webhook"] = "static"
    static_response: dict[str, Any] | None         = None
    webhook_url:     str | None                    = None
    webhook_secret:  str | None                    = None  # HMAC-SHA256 key
    timeout_ms:      int                           = 5_000


# ──────────────────────────────────────────────────────────────
# Sub-configs
# ──────────────────────────────────────────────────────────────

class VoiceConfig(BaseModel):
    enabled:       bool = True
    voice_name:    str  = "Kore"
    language_code: str  = "en-US"


class VADConfig(BaseModel):
    """
    mode="manual"  backend drives activity_start / activity_end signals.
    mode="auto"    Gemini's built-in VAD (we keep manual as the default).
    """
    mode: Literal["auto", "manual"] = "manual"


# ──────────────────────────────────────────────────────────────
# Project config  (the core multi-tenant runtime unit)
# ──────────────────────────────────────────────────────────────

class ProjectConfig(BaseModel):
    """
    Everything the session engine needs to open and drive a Gemini session.

    tenant_id is included so the session runner can tag session records
    correctly when it calls session_repo.close_session() on shutdown.
    It is NOT passed to Gemini — it's purely for internal bookkeeping.
    """
    project_id:              str
    tenant_id:               str  = ""   # populated from DB in Week 5+
    name:                    str
    system_prompt:           str
    gemini_model:            str  = "gemini-2.5-flash-native-audio-preview-12-2025"
    voice_config:            VoiceConfig  = Field(default_factory=VoiceConfig)
    vad_config:              VADConfig    = Field(default_factory=VADConfig)
    tools:                   list[ToolDefinition] = Field(default_factory=list)
    session_ttl_seconds:     int = 300
    max_concurrent_sessions: int = 100
    rate_limit_rpm:          int = 60