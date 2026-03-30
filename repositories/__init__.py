"""
repositories/__init__.py
─────────────────────────
Exports a single Repositories dataclass that bundles all four repositories.
Stored on app.state so any route handler can access them via:

    repos: Repositories = request.app.state.repos
    project = await repos.projects.get_by_id(...)

This avoids passing four separate objects through every function signature.
"""

from __future__ import annotations

from dataclasses import dataclass

from db.mongo import MongoDB
from db.redis_client import RedisClient

from .api_key_repo import APIKeyRepository
from .auth_token_repo import AuthTokenRepository
from .project_repo import ProjectRepository
from .session_repo import SessionRepository
from .tenant_repo import TenantRepository


@dataclass(frozen=True)
class Repositories:
    tenants:     TenantRepository
    projects:    ProjectRepository
    api_keys:    APIKeyRepository
    sessions:    SessionRepository
    auth_tokens: AuthTokenRepository

    @classmethod
    def create(cls, mongodb: MongoDB, redis: RedisClient) -> "Repositories":
        return cls(
            tenants     = TenantRepository(mongodb),
            projects    = ProjectRepository(mongodb, redis),
            api_keys    = APIKeyRepository(mongodb),
            sessions    = SessionRepository(mongodb),
            auth_tokens = AuthTokenRepository(mongodb),
        )
 