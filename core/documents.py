"""
core/documents.py
─────────────────
Pydantic models that map 1:1 to MongoDB documents.

Naming convention
─────────────────
  TenantDoc    — stored in `tenants` collection
  ProjectDoc   — stored in `projects` collection
  APIKeyDoc    — stored in `api_keys` collection
  SessionDoc   — stored in `sessions` collection

Relationship to core/schemas.py
────────────────────────────────
  schemas.py   holds runtime config shapes (ProjectConfig, ToolDefinition…)
               used by the session engine.  No DB awareness.

  documents.py holds the full MongoDB document shapes, including fields
               that the session engine doesn't need (tenant billing info,
               audit timestamps, key hashes, etc.).

  The repository layer converts between the two:
    ProjectDoc  ──(to_project_config)──►  ProjectConfig   (session engine)
    ProjectConfig is NEVER written back to MongoDB — mutations go through
    ProjectDoc.

MongoDB _id strategy
────────────────────
All _id fields are plain UUID strings (str), not ObjectId.  This makes
Pydantic serialisation trivial and cross-service references readable.
Use str(uuid.uuid4()) when inserting new documents.

Serialisation note
──────────────────
All models use model_config = ConfigDict(populate_by_name=True) so that
both field aliases (_id) and Python names (id) work during construction.
When writing to MongoDB use .to_mongo() which renames `id` → `_id`.
When reading from MongoDB pass the raw dict directly — Motor returns `_id`
and populate_by_name allows that via the Field(alias="_id") pattern.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .schemas import ProjectConfig, ToolDefinition, VoiceConfig, VADConfig


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


# ──────────────────────────────────────────────────────────────
# Tenant
# ──────────────────────────────────────────────────────────────

class TenantDoc(BaseModel):
    """
    One customer organisation.

    password_hash   argon2 hash — added in Week 6 (auth).  Stored as empty
                    string here so the schema is stable from Week 5 onward.
    plan            billing plan — enforces session / token limits.
    """
    model_config = ConfigDict(populate_by_name=True)

    id:             str      = Field(default_factory=_new_id, alias="_id")
    name:           str
    email:          str
    password_hash:  str      = ""          # populated in Week 6
    plan:           Literal["free", "starter", "pro", "enterprise"] = "free"
    created_at:     datetime = Field(default_factory=_now)
    updated_at:     datetime = Field(default_factory=_now)
    is_active:      bool     = True

    def to_mongo(self) -> dict[str, Any]:
        """Return a dict suitable for Motor insert/replace operations."""
        d = self.model_dump(by_alias=True)
        return d

    @classmethod
    def from_mongo(cls, doc: dict[str, Any]) -> "TenantDoc":
        return cls.model_validate(doc)


# ──────────────────────────────────────────────────────────────
# Project
# ──────────────────────────────────────────────────────────────

class ProjectDoc(BaseModel):
    """
    One chatbot project belonging to a tenant.

    tools            list of ToolDefinition — stored as plain dicts in Mongo,
                     deserialized back to ToolDefinition objects by from_mongo().
    webhook_url      customer's HTTPS endpoint for webhook-mode tools.
    webhook_secret   HMAC-SHA256 signing key for outbound webhook calls.
    allowed_origins  CORS whitelist — enforced at WS upgrade (Week 8).
    """
    model_config = ConfigDict(populate_by_name=True)

    id:                      str      = Field(default_factory=_new_id, alias="_id")
    tenant_id:               str
    name:                    str
    description:             str      = ""
    system_prompt:           str
    gemini_model:            str      = "gemini-2.5-flash-native-audio-preview-12-2025"
    voice_config:            VoiceConfig  = Field(default_factory=VoiceConfig)
    vad_config:              VADConfig    = Field(default_factory=VADConfig)
    tools:                   list[ToolDefinition] = Field(default_factory=list)
    webhook_url:             str | None   = None
    webhook_secret:          str | None   = None
    allowed_origins:         list[str]    = Field(default_factory=list)
    session_ttl_seconds:     int          = 300
    max_concurrent_sessions: int          = 10
    rate_limit_rpm:          int          = 60   # requests per minute per API key
    created_at:              datetime     = Field(default_factory=_now)
    updated_at:              datetime     = Field(default_factory=_now)
    is_active:               bool         = True

    def to_mongo(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)

    @classmethod
    def from_mongo(cls, doc: dict[str, Any]) -> "ProjectDoc":
        return cls.model_validate(doc)

    def to_project_config(self) -> ProjectConfig:
        """
        Convert to the runtime ProjectConfig used by the session engine.

        This is the bridge between the database world and the Gemini runner.
        Called by ProjectRepository.get_config_by_api_key() — which is the
        hot path every WebSocket connection takes.
        """
        return ProjectConfig(
            project_id              = self.id,
            tenant_id               = self.tenant_id,
            name                    = self.name,
            system_prompt           = self.system_prompt,
            gemini_model            = self.gemini_model,
            voice_config            = self.voice_config,
            vad_config              = self.vad_config,
            tools                   = self.tools,
            session_ttl_seconds     = self.session_ttl_seconds,
            max_concurrent_sessions = self.max_concurrent_sessions,
        )


# ──────────────────────────────────────────────────────────────
# API Key
# ──────────────────────────────────────────────────────────────

class APIKeyDoc(BaseModel):
    """
    An API key granting access to one project.

    key_hash    argon2 hash of the full key — the only form stored in DB.
                The full key is returned ONCE at creation and never stored.
    key_prefix  first 12 characters of the full key — safe to display in the
                dashboard so users can identify which key is which.
    key_type    "publishable" → client-side use, chat:connect permission only.
                "secret"      → server-side use, full permissions.
    """
    model_config = ConfigDict(populate_by_name=True)

    id:           str      = Field(default_factory=_new_id, alias="_id")
    project_id:   str
    tenant_id:    str
    label:        str      = "Default key"
    key_prefix:   str      = ""   # e.g. "pk_live_xxxx" (first 12 chars)
    key_hash:     str      = ""   # argon2 hash — populated by APIKeyRepository
    key_type:     Literal["publishable", "secret"] = "publishable"
    rate_limit_rpm: int    = 60
    revoked:      bool     = False
    revoked_at:   datetime | None = None
    created_at:   datetime = Field(default_factory=_now)
    last_used_at: datetime | None = None
    expires_at:   datetime | None = None

    def to_mongo(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)

    @classmethod
    def from_mongo(cls, doc: dict[str, Any]) -> "APIKeyDoc":
        return cls.model_validate(doc)

    @property
    def is_valid(self) -> bool:
        """Quick in-memory validity check (not a substitute for hash verification)."""
        if self.revoked:
            return False
        if self.expires_at and self.expires_at < _now():
            return False
        return True


# ──────────────────────────────────────────────────────────────
# Session
# ──────────────────────────────────────────────────────────────

class SessionDoc(BaseModel):
    """
    Persisted record written when a Gemini session closes.

    This is append-only — sessions are never updated after being written.
    Analytics queries run against this collection.
    """
    model_config = ConfigDict(populate_by_name=True)

    id:                   str      = Field(alias="_id")   # == the session_id UUID
    project_id:           str
    tenant_id:            str
    api_key_id:           str      = ""
    status:               Literal["closed", "error", "timeout"] = "closed"
    started_at:           datetime
    ended_at:             datetime = Field(default_factory=_now)
    duration_seconds:     float    = 0.0
    turns:                int      = 0
    tool_calls:           int      = 0
    error_message:        str | None = None

    def to_mongo(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)

    @classmethod
    def from_mongo(cls, doc: dict[str, Any]) -> "SessionDoc":
        return cls.model_validate(doc)