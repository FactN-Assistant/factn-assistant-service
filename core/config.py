"""
core/config.py
──────────────
Centralized environment-based configuration for the entire application.

All environment variables are loaded here at startup and imported by other
modules. This ensures:
  • Single source of truth for all config
  • Environment validation at startup
  • Easy defaults and logging
  • No scattered os.getenv() calls throughout codebase

Environment Variables
──────────────────────
MONGO_URI
  MongoDB connection string.
  Required: Yes
  Default: None

MONGO_DB_NAME
  MongoDB database name.
  Required: No
  Default: "livechat_dev"

REDIS_URL
  Redis connection URL.
  Required: No
  Default: "redis://localhost:6379"

GEMINI_API_KEY
  Google Gemini API key for LLM operations.
  Required: Yes
  Default: None

JWT_SECRET
  Secret key for JWT token generation and validation.
  Required: Yes (production should use strong random secret)
  Default: None

GEMINI_MODEL
  Gemini Live API model name (supports AUDIO response_modality exclusively).
  Required: No
  Default: "gemini-2.5-flash-native-audio-preview-12-2025"
"""

import os

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI", "")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "livechat_dev")
MONGO_MAX_POOL_SIZE = 10
MONGO_MIN_POOL_SIZE = 10
MONGO_SERVER_SELECTION_TIMEOUT = 10_000
MONGO_CONNECT_TIMEOUT = 10_000
MONGO_SOCKET_TIMEOUT = 30_000

# Redis Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_TTL_MINUTES = 15
REFRESH_TOKEN_TTL_DAYS = 7 

# Google Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv(
  "GEMINI_MODEL",
  "gemini-2.5-flash-native-audio-preview-12-2025",
)