#!/usr/bin/env python3
"""
scripts/seed.py
───────────────
Bootstrap script — creates an initial Tenant, Project, and API key so you
can test the WebSocket endpoint without a dashboard.

Usage
─────
  python scripts/seed.py

Output
──────
  Prints the raw API key once.  Copy it into test_client.html's API Key
  field and connect.  The key cannot be recovered after this script exits.

Run again
─────────
  Safe to run multiple times — if a tenant with SEED_EMAIL already exists
  it skips tenant/project creation and just adds a new key to the existing
  project.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from core.auth import hash_password
from db.indexes import ensure_indexes
from db.mongo import MongoDB
from db.redis_client import RedisClient
from repositories import Repositories

SEED_EMAIL    = "admin@example.com"
SEED_PASSWORD = "changeme123"       # change before sharing
SEED_NAME     = "Demo Tenant"
PROJECT_NAME  = "Demo Assistant"
SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant. "
    "Keep answers concise and conversational."
)


async def main() -> None:
    mongo_uri = os.environ["MONGO_URI"]
    mongo_db  = os.environ.get("MONGO_DB_NAME", "livechat_dev")
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")

    mongodb = MongoDB(uri=mongo_uri, db_name=mongo_db)
    redis   = RedisClient(url=redis_url)
    repos   = Repositories.create(mongodb, redis)

    await ensure_indexes(mongodb)

    # ── Tenant ────────────────────────────────────────────────
    tenant = await repos.tenants.get_by_email(SEED_EMAIL)
    if tenant is None:
        tenant = await repos.tenants.create(name=SEED_NAME, email=SEED_EMAIL)
        pwd_hash = hash_password(SEED_PASSWORD)
        await repos.tenants.update_password_hash(tenant.id, pwd_hash)
        print(f"Created tenant:   {tenant.id}  ({tenant.email})")
        print(f"  Login password: {SEED_PASSWORD}")
    else:
        print(f"Existing tenant:  {tenant.id}  ({tenant.email})")

    # ── Project ───────────────────────────────────────────────
    projects = await repos.projects.list_for_tenant(tenant.id, limit=1)
    if not projects:
        project = await repos.projects.create(
            tenant_id     = tenant.id,
            name          = PROJECT_NAME,
            system_prompt = SYSTEM_PROMPT,
        )
        print(f"Created project:  {project.id}  ({project.name})")
    else:
        project = projects[0]
        print(f"Existing project: {project.id}  ({project.name})")

    # ── API Key ───────────────────────────────────────────────
    key_doc, raw_key = await repos.api_keys.create(
        project_id = project.id,
        tenant_id  = tenant.id,
        label      = "Dev test key",
        key_type   = "publishable",
    )

    print()
    print("=" * 60)
    print("  API KEY (copy this — shown only once):")
    print(f"  {raw_key}")
    print("=" * 60)
    print(f"\n  Prefix:     {key_doc.key_prefix}")
    print(f"  Project ID: {project.id}")
    print(f"  Tenant ID:  {tenant.id}")
    print("\nPaste the API key into test_client.html → API Key field.\n")

    await redis.close()
    mongodb.close()


if __name__ == "__main__":
    asyncio.run(main())