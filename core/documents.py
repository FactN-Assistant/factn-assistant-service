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

Week 6 additions
────────────────
  SessionDoc  — added input_tokens, output_tokens fields
  AuthTokenDoc — new: refresh token store for JWT auth (Week 6)
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
    model_config = ConfigDict(populate_by_name=True)

    id:             str      = Field(default_factory=_new_id, alias="_id")
    name:           str
    email:          str
    password_hash:  str      = ""
    plan:           Literal["free", "starter", "pro", "enterprise"] = "free"
    created_at:     datetime = Field(default_factory=_now)
    updated_at:     datetime = Field(default_factory=_now)
    is_active:      bool     = True

    def to_mongo(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)

    @classmethod
    def from_mongo(cls, doc: dict[str, Any]) -> "TenantDoc":
        return cls.model_validate(doc)


# ──────────────────────────────────────────────────────────────
# Project
# ──────────────────────────────────────────────────────────────

class ProjectDoc(BaseModel):
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
    rate_limit_rpm:          int          = 60
    created_at:              datetime     = Field(default_factory=_now)
    updated_at:              datetime     = Field(default_factory=_now)
    is_active:               bool         = True

    def to_mongo(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)

    @classmethod
    def from_mongo(cls, doc: dict[str, Any]) -> "ProjectDoc":
        return cls.model_validate(doc)

    def to_project_config(self) -> ProjectConfig:
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
            rate_limit_rpm          = self.rate_limit_rpm,
        )


# ──────────────────────────────────────────────────────────────
# API Key
# ──────────────────────────────────────────────────────────────

class APIKeyDoc(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id:             str      = Field(default_factory=_new_id, alias="_id")
    project_id:     str
    tenant_id:      str
    label:          str      = "Default key"
    key_prefix:     str      = ""
    key_hash:       str      = ""
    key_type:       Literal["publishable", "secret"] = "publishable"
    rate_limit_rpm: int      = 60
    revoked:        bool     = False
    revoked_at:     datetime | None = None
    created_at:     datetime = Field(default_factory=_now)
    last_used_at:   datetime | None = None
    expires_at:     datetime | None = None

    def to_mongo(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)

    @classmethod
    def from_mongo(cls, doc: dict[str, Any]) -> "APIKeyDoc":
        return cls.model_validate(doc)

    @property
    def is_valid(self) -> bool:
        if self.revoked:
            return False
        if self.expires_at and self.expires_at < _now():
            return False
        return True

# ──────────────────────────────────────────────────────────────
# Session (append-only analytics record)
# ──────────────────────────────────────────────────────────────

class SessionDoc(BaseModel):
    """
    Written once when a Gemini session closes.  Never updated.
 
    Token fields
    ────────────
    input_tokens   accumulated from usage_metadata.prompt_token_count
    output_tokens  accumulated from usage_metadata.candidates_token_count
 
    The Gemini Live API reports these cumulatively via usage_metadata
    on server messages.  We capture the LAST values before session close.
    For audio-only sessions total_token_count may be the only value
    available; in that case it is stored in output_tokens.
    """
    model_config = ConfigDict(populate_by_name=True)

    id:               str      = Field(alias="_id")
    project_id:       str
    tenant_id:        str
    api_key_id:       str      = ""
    status:           Literal["closed", "error", "timeout"] = "closed"
    started_at:       datetime
    ended_at:         datetime = Field(default_factory=_now)
    duration_seconds: float    = 0.0
    turns:            int      = 0
    tool_calls:       int      = 0
    input_tokens:     int      = 0
    output_tokens:    int      = 0
    error_message:    str | None = None
 
    def to_mongo(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)

    @classmethod
    def from_mongo(cls, doc: dict[str, Any]) -> "SessionDoc":
        return cls.model_validate(doc)
    
# ──────────────────────────────────────────────────────────────
# Auth Token  (Week 6 — refresh token store)
# ──────────────────────────────────────────────────────────────
 
class AuthTokenDoc(BaseModel):
    """
    Persisted refresh token.
 
    Stored in the auth_tokens collection with a TTL index on expires_at
    so MongoDB auto-deletes expired tokens.  The token_hash is an
    argon2 hash of the raw refresh token — same security model as API keys.
 
    On logout or token rotation, the document is deleted explicitly
    (doesn't wait for the TTL to expire).
    """
    model_config = ConfigDict(populate_by_name=True)
 
    id:           str      = Field(default_factory=_new_id, alias="_id")
    tenant_id:    str
    token_hash:   str                # argon2 hash of raw refresh token
    token_family: str      = ""      # rotation family — detect reuse attacks
    created_at:   datetime = Field(default_factory=_now)
    expires_at:   datetime           # TTL index on this field
 
    def to_mongo(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)
 
    @classmethod
    def from_mongo(cls, doc: dict[str, Any]) -> "AuthTokenDoc":
        return cls.model_validate(doc)
