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

Running locally
───────────────
  cp .env.example .env    # fill in MONGO_URI, REDIS_URL, GEMINI_API_KEY
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
from api.projects import router as projects_router
from api.chat import router as chat_router
from api.keys import router as keys_router
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
    5.  SessionManager  ← now receives session_repo for session logging
    6.  Attach all to app.state
 
    Shutdown order (reverse)
    ─────────────────────────
    1.  SessionManager.stop()  — gracefully closes all Gemini sessions
    2.  Redis.close()
    3.  MongoDB.close()
    """

    # ── Validate required env vars ────────────────────────────
    mongo_uri = os.environ.get("MONGO_URI", "")
    mongo_db  = os.environ.get("MONGO_DB_NAME", "livechat_dev")
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
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

    # Pass session_repo so every session close writes to MongoDB
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

    log.info("Application startup complete (db=%s)", mongo_db)
    yield

    # ── Shutdown ──────────────────────────────────────────────
    log.info("Application shutting down")
    await manager.stop()
    await redis.close()
    mongodb.close()
    log.info("Application shutdown complete")


app = FastAPI(
    title       = "LiveChat API Platform",
    version     = "0.7.0",
    description = "Multi-tenant AI chatbot-as-a-service",
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
app.include_router(chat_router)
app.include_router(keys_router)


@app.get("/")
async def root() -> dict:
    return {"service": "LiveChat API Platform", "version": "0.7.0", "status": "ok"}


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
    }