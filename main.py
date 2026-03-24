"""
main.py
───────
Application entry point.

What lives here
───────────────
  • FastAPI app creation with lifespan (startup / shutdown)
  • SessionManager initialisation (one per process)
  • Demo ProjectConfig loaded from .env  ← replaced by DB lookup in Week 5-6
  • Router registration
  • CORS middleware

Running locally
───────────────
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
from core.schemas import ProjectConfig, ToolDefinition, VoiceConfig, VADConfig
from core.session_manager import SessionManager

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
# Demo project config  (Week 2-4 placeholder)
# ──────────────────────────────────────────────────────────────
# When the database layer arrives (Week 5-6) this whole section is replaced
# by a repository call:  project = await ProjectRepo.get_by_api_key(api_key)
# ──────────────────────────────────────────────────────────────

def _build_demo_project() -> ProjectConfig:
    """
    Build a ProjectConfig from environment variables.

    Set these in your .env file — see .env.example for defaults.
    """
    system_prompt = os.getenv(
        "DEMO_SYSTEM_PROMPT",
        (
            "You are a helpful, friendly assistant. "
            "Keep answers concise and conversational."
        ),
    )

    # Demo tools — mix of static (instant mock responses) so you can see
    # the full tool-call flow in the test client without a real backend.
    demo_tools = [
        ToolDefinition(
            name="get_current_time",
            description="Returns the current date and time in UTC.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            execution_mode="static",
            static_response={
                "utc": "2026-03-23T10:00:00Z",
                "note": "This is a static demo response.",
            },
        ),
        ToolDefinition(
            name="search_knowledge_base",
            description=(
                "Searches the internal knowledge base for information on a topic."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                },
                "required": ["query"],
            },
            execution_mode="static",
            static_response={
                "results": [
                    {
                        "title":   "Demo Article",
                        "snippet": "This is a placeholder result from the static demo tool.",
                    }
                ]
            },
        ),
    ]

    return ProjectConfig(
        project_id    = os.getenv("DEMO_PROJECT_ID",   "demo-project-001"),
        name          = os.getenv("DEMO_PROJECT_NAME",  "Demo Assistant"),
        system_prompt = system_prompt,
        gemini_model  = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash-native-audio-preview-12-2025",
        ),
        voice_config  = VoiceConfig(
            voice_name = os.getenv("DEMO_VOICE_NAME", "Kore"),
        ),
        vad_config            = VADConfig(mode="manual"),
        tools                 = demo_tools,
        session_ttl_seconds   = int(os.getenv("SESSION_TTL", "300")),
        max_concurrent_sessions = int(os.getenv("MAX_SESSIONS", "100")),
    )


# ──────────────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the SessionManager on boot, tear it down on shutdown."""
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        log.warning("GEMINI_API_KEY is not set — Gemini calls will fail.")

    gemini_client = genai.Client(api_key=gemini_api_key)

    manager = SessionManager(gemini_client)
    await manager.start()

    # Attach shared state to the app so routers can access it via request.app.state
    app.state.session_manager    = manager
    app.state.demo_project_config = _build_demo_project()

    log.info("Application startup complete")
    log.info("Demo project: %s", app.state.demo_project_config.name)
    log.info(
        "Demo tools: %s",
        [t.name for t in app.state.demo_project_config.tools],
    )

    yield  # ← app is running here

    log.info("Application shutting down")
    await manager.stop()
    log.info("Application shutdown complete")


# ──────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "LiveChat API Platform",
    version     = "0.2.0",       # Week 2-3 implementation
    description = "Multi-tenant AI chatbot-as-a-service — core session engine",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # Tighten to per-project origins in Week 8
    allow_methods     = ["*"],
    allow_headers     = ["*"],
    allow_credentials = True,
)

# ── Routers ───────────────────────────────────────────────────
app.include_router(chat_router)


# ── Health & root endpoints (preserved from Week 1) ───────────

@app.get("/")
async def root() -> dict:
    return {
        "service": "LiveChat API Platform",
        "version": "0.2.0",
        "status":  "ok",
    }


@app.get("/health")
async def health(request: Request) -> dict:
    manager: SessionManager = request.app.state.session_manager
    return {
        "status":          "ok",
        "active_sessions": await manager.active_session_count(),
        "session_ids":     await manager.get_active_session_ids(),
    }