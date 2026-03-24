"""
core/schemas.py
───────────────
Pydantic models that describe everything a Project needs to run a Gemini
Live session.  No database or auth logic here — these are pure data shapes.

When the database layer arrives (Week 5-6) these same models will be
populated by deserialising MongoDB documents.  Until then, a demo project
is constructed from environment variables in main.py.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# Tool definition
# ──────────────────────────────────────────────────────────────

class ToolDefinition(BaseModel):
    """
    Describes one function the Gemini model may call.

    parameters  Raw JSON Schema object (OpenAPI 3.0 subset) — passed
                verbatim to Gemini's function_declarations.

    execution_mode
        "static"  → return static_response directly (good for demos / mocks)
        "webhook" → POST to webhook_url with the call arguments
    """
    name:            str
    description:     str
    parameters:      dict[str, Any]          # raw JSON Schema
    execution_mode:  Literal["static", "webhook"] = "static"
    static_response: dict[str, Any] | None   = None
    webhook_url:     str | None              = None
    webhook_secret:  str | None              = None   # HMAC-SHA256 key
    timeout_ms:      int                     = 5_000


# ──────────────────────────────────────────────────────────────
# Sub-configs
# ──────────────────────────────────────────────────────────────

class VoiceConfig(BaseModel):
    enabled:       bool = True
    voice_name:    str  = "Kore"     # Gemini prebuilt voice
    language_code: str  = "en-US"


class VADConfig(BaseModel):
    """
    Voice Activity Detection.

    mode="manual"  → backend sends activity_start / activity_end driven by
                     the client's voice_start / voice_end frames (our default).
    mode="auto"    → Gemini's built-in VAD (disabled in manual mode).
    """
    mode: Literal["auto", "manual"] = "manual"


# ──────────────────────────────────────────────────────────────
# Project config  (the core multi-tenant unit)
# ──────────────────────────────────────────────────────────────

class ProjectConfig(BaseModel):
    """
    All configuration required to open and drive a Gemini Live session
    for one customer project.

    This will later be loaded from MongoDB by looking up the API key.
    For now it is built from environment variables via `get_demo_config()`
    in main.py.
    """
    project_id:              str
    name:                    str
    system_prompt:           str
    gemini_model:            str = "gemini-2.5-flash-native-audio-preview-12-2025"
    voice_config:            VoiceConfig  = Field(default_factory=VoiceConfig)
    vad_config:              VADConfig    = Field(default_factory=VADConfig)
    tools:                   list[ToolDefinition] = Field(default_factory=list)
    session_ttl_seconds:     int = 300
    max_concurrent_sessions: int = 100