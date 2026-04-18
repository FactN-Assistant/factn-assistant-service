"""
main.py
───────
Application entry point.

Week 5 changes from Week 2-3
─────────────────────────────
  • MongoDB client (Motor) initialised in lifespan
  • Redis client initialised in lifespan
  • MongoDB indexes ensured on startup
  • Repositories bundle attached to app.state
  • demo_project_config REMOVED — all project resolution now goes through
    the repository layer via api/chat.py
  • /health endpoint extended with DB ping results
  
Week 6 changes from Week 5
───────────────────────────
  • Auth router (POST /v1/auth/*) registered
  • API keys router (POST|GET|DELETE /v1/projects/*/keys) registered
  • SessionManager constructed with session_repo so every session close
    writes a SessionDoc to MongoDB
  • JWT_SECRET checked at startup
  • Motor removed — PyMongo AsyncMongoClient used throughout
  
Week 7 changes from Week 6
───────────────────────────
  • Projects router registered (full CRUD + tool CRUD + test endpoint)
  • Version bumped to 0.7.0
  
Week 8 changes from Week 7
───────────────────────────
  • Tokens router registered (POST /v1/tokens, POST /v1/tokens/rotate)
  • Version bumped to 0.8.0

Changes from previous version
──────────────────────────────
  • Version bumped to 0.9.0
  • /health endpoint now exposes plan_limits summary for ops monitoring
  • All routers remain the same — no new routers needed for this release
    (plan enforcement, CORS, suspension, and lifecycle webhooks are
    implemented inside existing routers and the session manager)
 
Running locally
───────────────
  cp .env.example .env    # fill in MONGO_URI, REDIS_URL, GEMINI_API_KEY, JWT_SECRET
  pip install -r requirements.txt
  python scripts/seed.py
  uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from google import genai

from api.auth import router as auth_router
from api.tokens import router as tokens_router
from api.projects import router as projects_router
from api.plans import router as plans_router
from api.chat import router as chat_router
from api.keys import router as keys_router
from core.plan_limits import PLAN_LIMITS
from core.session_manager import SessionManager
from db.indexes import ensure_indexes
from db.mongo import MongoDB
from db.redis_client import RedisClient
from repositories import Repositories

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("livechat")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup order
    ─────────────
    1.  MongoDB (PyMongo AsyncMongoClient)
    2.  MongoDB indexes ensured
    3.  Redis
    4.  Repositories bundle
    5.  SessionManager
    6.  Attach all to app.state
 
    Shutdown order (reverse)
    ─────────────────────────
    1.  SessionManager.stop()
    2.  Redis.close()
    3.  MongoDB.close()
    """
    mongo_uri  = os.environ.get("MONGO_URI", "")
    mongo_db   = os.environ.get("MONGO_DB_NAME", "livechat_dev")
    redis_url  = os.environ.get("REDIS_URL", "redis://localhost:6379")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    jwt_secret = os.environ.get("JWT_SECRET", "")

    if not mongo_uri:
        log.warning("MONGO_URI not set — database calls will fail.")
    if not gemini_key:
        log.warning("GEMINI_API_KEY not set — Gemini calls will fail.")
    if not jwt_secret or jwt_secret == "change-me-in-production":
        log.warning(
            "JWT_SECRET is not set or is default — "
            "set a strong random secret in production!"
        )

    # ── MongoDB ───────────────────────────────────────────────
    mongodb = MongoDB(uri=mongo_uri, db_name=mongo_db)
    await ensure_indexes(mongodb)

    # ── Redis ─────────────────────────────────────────────────
    redis = RedisClient(url=redis_url)

    # ── Repositories ──────────────────────────────────────────
    repos = Repositories.create(mongodb, redis)

    # ── Gemini + SessionManager ───────────────────────────────
    gemini_client = genai.Client(api_key=gemini_key)

    manager = SessionManager(
        gemini_client = gemini_client,
        session_repo  = repos.sessions,
    )
    await manager.start()

    # ── Attach to app.state ───────────────────────────────────
    app.state.mongodb         = mongodb
    app.state.redis           = redis
    app.state.repos           = repos
    app.state.session_manager = manager

    log.info(
        "Application startup complete (db=%s) — plan tiers: %s",
        mongo_db,
        ", ".join(PLAN_LIMITS.keys()),
    )
    yield

    # ── Shutdown ──────────────────────────────────────────────
    log.info("Application shutting down")
    await manager.stop()
    await redis.close()
    mongodb.close()
    log.info("Application shutdown complete")


app = FastAPI(
    title       = "LiveChat API Platform",
    version     = "0.9.0",
    description = (
        "Multi-tenant AI Chatbot-as-a-Service powered by Google Gemini Live API. "
        "v0.9.0 adds plan enforcement, CORS, tenant suspension, "
        "daily token quotas, session lifecycle webhooks, and "
        "dedicated sub-resource endpoints for system prompt, voice config, "
        "and webhook config."
    ),
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_methods     = ["*"],
    allow_headers     = ["*"],
    allow_credentials = True,
)

# ── Routers ───────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(projects_router)
app.include_router(plans_router)
app.include_router(tokens_router)
app.include_router(chat_router)
app.include_router(keys_router)


@app.get("/")
async def root() -> dict:
    return {
        "service": app.title,
        "version": app.version,
        "status":  "ok",
        "plans":   list(PLAN_LIMITS.keys()),
    }


@app.get("/health")
async def health(request: Request) -> dict:
    mongodb:  MongoDB        = request.app.state.mongodb
    redis:    RedisClient    = request.app.state.redis
    manager:  SessionManager = request.app.state.session_manager
 
    mongo_ok = await mongodb.ping()
    redis_ok = await redis.ping()

    return {
        "status":          "ok" if (mongo_ok and redis_ok) else "degraded",
        "mongodb":         "ok" if mongo_ok else "unreachable",
        "redis":           "ok" if redis_ok else "unreachable",
        "active_sessions": await manager.active_session_count(),
        "session_ids":     await manager.get_active_session_ids(),
        "version":         app.version,
    }