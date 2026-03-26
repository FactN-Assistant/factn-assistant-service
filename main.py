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

from api.chat import router as chat_router
from core.session_manager import SessionManager
from db.indexes import ensure_indexes
from db.mongo import MongoDB
from db.redis_client import RedisClient
from repositories import Repositories

load_dotenv()

# ──────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("livechat")


# ──────────────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup
    ───────
    1. Connect MongoDB (Motor async client)
    2. Connect Redis
    3. Ensure all collection indexes exist
    4. Build Repositories bundle
    5. Start SessionManager (spawns Gemini sessions)

    Shutdown
    ────────
    1. Stop SessionManager (gracefully closes all active Gemini sessions)
    2. Close Redis connection
    3. Close MongoDB connection
    """

    # ── MongoDB ───────────────────────────────────────────────
    mongo_uri = os.environ.get("MONGO_URI", "")
    mongo_db  = os.environ.get("MONGO_DB_NAME", "livechat_dev")
    if not mongo_uri:
        log.warning("MONGO_URI is not set — database calls will fail.")

    mongodb = MongoDB(uri=mongo_uri, db_name=mongo_db)
    await ensure_indexes(mongodb)

    # ── Redis ─────────────────────────────────────────────────
    redis_url = os.environ.get("REDIS_URL")
    redis     = RedisClient(url=redis_url)

    # ── Repositories ──────────────────────────────────────────
    repos = Repositories.create(mongodb, redis)

    # ── Gemini + SessionManager ───────────────────────────────
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        log.warning("GEMINI_API_KEY is not set — Gemini calls will fail.")

    gemini_client = genai.Client(api_key=gemini_api_key)
    manager       = SessionManager(gemini_client)
    await manager.start()

    # ── Attach to app.state ───────────────────────────────────
    app.state.mongodb          = mongodb
    app.state.redis            = redis
    app.state.repos            = repos
    app.state.session_manager  = manager

    log.info("Application startup complete (db=%s)", mongo_db)

    yield  # ← application runs here

    # ── Shutdown ──────────────────────────────────────────────
    log.info("Application shutting down")
    await manager.stop()
    await redis.close()
    mongodb.close()
    log.info("Application shutdown complete")


# ──────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "LiveChat API Platform",
    version     = "0.5.0",
    description = "Multi-tenant AI chatbot-as-a-service",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # tighten to per-project origins in Week 8
    allow_methods     = ["*"],
    allow_headers     = ["*"],
    allow_credentials = True,
)

# ── Routers ───────────────────────────────────────────────────
app.include_router(chat_router)


# ──────────────────────────────────────────────────────────────
# Core endpoints
# ──────────────────────────────────────────────────────────────

@app.get("/")
async def root() -> dict:
    return {
        "service": "LiveChat API Platform",
        "version": "0.5.0",
        "status":  "ok",
    }


@app.get("/health")
async def health(request: Request) -> dict:
    mongodb:  MongoDB          = request.app.state.mongodb
    redis:    RedisClient      = request.app.state.redis
    manager:  SessionManager   = request.app.state.session_manager

    mongo_ok = await mongodb.ping()
    redis_ok = await redis.ping()

    return {
        "status":          "ok" if (mongo_ok and redis_ok) else "degraded",
        "mongodb":         "ok" if mongo_ok else "unreachable",
        "redis":           "ok" if redis_ok else "unreachable",
        "active_sessions": await manager.active_session_count(),
        "session_ids":     await manager.get_active_session_ids(),
    }