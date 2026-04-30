# Schema & Architecture Change Log
**Platform:** Multi-Tenant AI API Gateway (ChatN / FactN Assistant)
**Document Date:** 2026-04-30
**Status:** Agreed — Pending Implementation

---

## Table of Contents

1. [Ephemeral Token Flow — Full Redesign](#1-ephemeral-token-flow--full-redesign)
2. [Database Schema Changes](#2-database-schema-changes)
   - [ephemeral_tokens — Modified](#21-ephemeral_tokens--modified)
   - [project_configs — Modified](#22-project_configs--modified)
   - [sessions — Modified](#23-sessions--modified)
3. [Redis Key Convention — Finalized](#3-redis-key-convention--finalized)
4. [Conversation History Strategy — Finalized](#4-conversation-history-strategy--finalized)
5. [No-Change Tables](#5-no-change-tables)
6. [Constraints to Enforce in Code](#6-constraints-to-enforce-in-code)

---

## 1. Ephemeral Token Flow — Full Redesign

### What Changed and Why

The previous design left the token issuance endpoint unspecified and used a generic `/auth/token` path. The single-use enforcement mechanism was also underspecified. The corrected flow is:

### Issuance (REST API Service)

```
POST /api/v1/projects/{project_id}/tokens
Authorization: Bearer <long_lived_api_key>
Body: { "scope": "session:connect" }
```

**Validation steps performed by REST API Service:**
1. Hash inbound API key → Redis lookup (fast path) → PostgreSQL fallback
2. Validate: key is active, not expired, belongs to `project_id` in URL path
3. Validate: project is active
4. Validate: subscription plan allows token issuance (rate check)
5. Check concurrent session count vs plan limit → reject if already at limit
6. Generate `raw_token` = CSPRNG (256 bits minimum)
7. `token_hash` = HMAC-SHA256(raw_token, server_secret)
8. `Redis.SET(token_hash, {project_id, scope, issued_to_ip, expires_at}, TTL=60s)`
9. `INSERT INTO ephemeral_tokens` (audit record, outcome = NULL)
10. Return `{"token": "<raw_token>", "expires_in": 60}` — raw token shown exactly once

### Connection (Session Gateway Service)

```
# WebSocket (browser cannot set headers — query param is the only option)
wss://host/ws?project_id=xxx&token=<raw_token>

# SSE (standard HTTP — Authorization header preferred)
GET /sse?project_id=xxx
Authorization: Bearer <raw_token>
```

**Validation steps performed by Session Gateway:**
1. Extract token from query param (WS) or Authorization header (SSE)
2. `token_hash` = HMAC-SHA256(inbound_token, server_secret)
3. `payload = Redis.GETDEL(token_hash)` ← **atomic get-and-delete**
4. If nil → HTTP 401, reject connection
5. Defensive check: `expires_at < now()` → HTTP 401
6. Validate `scope == 'session:connect'`
7. Validate `project_id` in payload matches `project_id` in request
8. `UPDATE ephemeral_tokens SET used_at = now(), outcome = 'used'`
9. Continue with session initialization

### Background Cleanup Job

```sql
-- Runs every hour
UPDATE ephemeral_tokens
SET outcome = 'expired'
WHERE outcome IS NULL AND expires_at < now();

-- Purge records older than 24 hours
DELETE FROM ephemeral_tokens
WHERE expires_at < now() - INTERVAL '24 hours';
```

### Key Security Decisions

| Decision | Detail |
|---|---|
| Raw token storage | Never stored anywhere — shown to client once only |
| Redis stores | HMAC-SHA256 hash of token, not the raw token |
| Single-use enforcement | Redis `GETDEL` (atomic) — not SETNX, not GET + DELETE |
| Query param risk (WS) | Accepted industry tradeoff — mitigated by 60s TTL and single-use |
| SSE | Uses Authorization header — not query param |
| Nginx log masking | Configure load balancer to redact `token` query parameter from access logs |
| Scope | Enum — currently only `session:connect`. Field exists for future expansion |

---

## 2. Database Schema Changes

### 2.1 `ephemeral_tokens` — Modified

#### Fields to REMOVE

| Field | Reason |
|---|---|
| `is_used BOOL` | Replaced by `used_at` + `outcome` which carry more information |

#### Fields to ADD

| Field | Type | Default | Description |
|---|---|---|---|
| `scope` | `TEXT NOT NULL` | `'session:connect'` | Token permission scope. Validated at connection time by Session Gateway. Currently only `session:connect` is valid. |
| `used_at` | `TIMESTAMPTZ` | `NULL` | Timestamp when the token was consumed at connection time. NULL means unused. |
| `outcome` | `TEXT` | `NULL` | Final state of the token. Values: `NULL` (pending), `'used'` (consumed at connection), `'expired'` (TTL elapsed, never used). Set by background job for expired tokens. |

#### Fields Already Correct (no change needed)

| Field | Note |
|---|---|
| `token_hash TEXT UQ` | Correct — stores hash, never raw token |
| `master_key UUID FK apikeys.id` | Correct — links token back to the API key used to issue it |
| `project_id UUID FK` | Correct — token is always scoped to a specific project |
| `client_ip TEXT` | Correct — maps to `issued_to_ip` in the Redis payload |
| `expires_at TIMESTAMPTZ` | Correct |
| `created_at TIMESTAMPTZ` | Correct |

#### Final `ephemeral_tokens` Schema

```sql
CREATE TABLE ephemeral_tokens (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id  UUID NOT NULL REFERENCES projects(id),
    master_key  UUID NOT NULL REFERENCES api_keys(id),
    token_hash  TEXT NOT NULL UNIQUE,
    scope       TEXT NOT NULL DEFAULT 'session:connect',
    client_ip   INET NOT NULL,
    expires_at  TIMESTAMPTZ NOT NULL,
    used_at     TIMESTAMPTZ,          -- NULL until consumed
    outcome     TEXT,                 -- NULL | 'used' | 'expired'
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_ephemeral_tokens_project_id ON ephemeral_tokens(project_id);
CREATE INDEX idx_ephemeral_tokens_expires_at ON ephemeral_tokens(expires_at);
```

---

### 2.2 `project_configs` — Modified

#### Fields to ADD

| Field | Type | Default | Description |
|---|---|---|---|
| `history_turn_limit` | `INT NOT NULL` | `10` | Maximum number of conversation rounds (user + assistant pairs) retained in Redis per session. When this limit is exceeded, the oldest user+assistant pair is dropped. Configurable per project. The system prompt is never counted in this limit and is never stored in the Redis history array. |

#### Why This Field Exists

HTTP Orchestrator adapters (Chat Completions, Gemini Generate Content) maintain conversation history in Redis for the duration of a session. Without a size limit this array grows unboundedly with each turn, eventually exceeding the model's context window and causing provider-side errors. The `history_turn_limit` caps the array at a meaningful size. WS Bridge adapters (Realtime, Gemini Live) do not use this field — the provider holds all state.

#### Trimming Behaviour (enforced in adapter code, not in DB)

- A **turn** = one user message + one assembled assistant response
- When `turn_count > history_turn_limit`: drop the oldest user+assistant pair
- Always drop in complete pairs — never leave a dangling user message
- System prompt is prepended at request-build time from `project_configs.system_prompt` — it is NOT stored in the Redis history array

#### Final addition to `project_configs`

```sql
ALTER TABLE project_configs
ADD COLUMN history_turn_limit INT NOT NULL DEFAULT 10;
```

---

### 2.3 `sessions` — Modified

The current ERD has a `STATUS` section header but no actual `status` column with defined values. Several fields required by the session teardown flow and admin force-close flow are also missing.

#### Fields to ADD

| Field | Type | Default | Nullable | Description |
|---|---|---|---|---|
| `status` | `TEXT NOT NULL` | `'validating'` | No | Current state of the session. See state machine values below. |
| `node_id` | `TEXT` | `NULL` | Yes | Identifier of the gateway node handling this session. Required for Redis pub/sub admin force-close routing. |
| `close_code` | `INT` | `NULL` | Yes | WebSocket close code sent at session end (1000, 4001–4006, 4029). NULL for SSE sessions or sessions not yet closed. |
| `error_code` | `TEXT` | `NULL` | Yes | Machine-readable error code if session ended due to an error (e.g., `provider_unavailable`, `session_timeout`, `key_revoked`). NULL for clean closes. |
| `error_message` | `TEXT` | `NULL` | Yes | Human-readable error description. NULL for clean closes. |
| `ephemeral_token_id` | `UUID FK ephemeral_tokens.id` | `NULL` | Yes | The ephemeral token used to open this session. NULL if the session was opened with a long-lived API key directly. |
| `connection_type` | `TEXT NOT NULL` | `'websocket'` | No | How the End Client connected. Values: `'websocket'`, `'sse'`. |

#### `status` Field — Valid State Machine Values

| Value | Description |
|---|---|
| `'validating'` | Credential is being verified. Connection not yet upgraded. |
| `'initializing'` | Credential valid. Loading config, retrieving secrets, instantiating adapter. |
| `'active'` | Session fully established. Waiting for client messages. |
| `'responding'` | LLM is generating a response. Streaming in progress. |
| `'awaiting_tool_result'` | LLM invoked a tool. Waiting for End Client to return the result. |
| `'closing'` | Teardown sequence in progress. |
| `'closed'` | Session fully torn down. All resources released. Final state. |
| `'rejected'` | Connection was rejected before or during upgrade (auth failure, limit exceeded). |
| `'failed'` | Session terminated due to an unrecoverable error (provider unavailable, fatal protocol error). |

#### Final additions to `sessions`

```sql
ALTER TABLE sessions
ADD COLUMN status              TEXT NOT NULL DEFAULT 'validating',
ADD COLUMN node_id             TEXT,
ADD COLUMN close_code          INT,
ADD COLUMN error_code          TEXT,
ADD COLUMN error_message       TEXT,
ADD COLUMN ephemeral_token_id  UUID REFERENCES ephemeral_tokens(id),
ADD COLUMN connection_type     TEXT NOT NULL DEFAULT 'websocket';

CREATE INDEX idx_sessions_status     ON sessions(status);
CREATE INDEX idx_sessions_project_id ON sessions(project_id);
CREATE INDEX idx_sessions_node_id    ON sessions(node_id);
```

---

## 3. Redis Key Convention — Finalized

All Redis keys must follow this naming convention. No exceptions.

```
# Session registry (written at INITIALIZING, deleted at teardown)
session:{session_id}                         → JSON hash: project_id, user_id, adapter_type, model_id, node_id, started_at

# Conversation history — Chat Completions and Gemini Generate adapters only
session:{session_id}:history                 → JSON: { "turns": [...], "turn_count": N }

# Response ID — Responses API adapter only
session:{session_id}:response_id             → plain string: "resp_abc123"

# Usage counters (flushed every 30 seconds, final flush at teardown)
session:{session_id}:usage                   → hash: input_tokens, output_tokens, audio_in_secs, audio_out_secs, tool_calls_count

# Project config cache (invalidated immediately on config update via REST API)
project:{project_id}:config                  → JSON project configuration

# API key fast lookup (written on first DB hit, invalidated on revocation)
apikey:{key_hash}                            → JSON: project_id, is_active, expires_at

# Ephemeral token (GETDEL at connection time — atomic, self-deleting)
token:{token_hash}                           → JSON: project_id, scope, issued_to_ip, expires_at

# Rate limiting — sliding window counters
ratelimit:apikey:{key_hash}:conns            → sliding window counter (WS connections per second)
ratelimit:ip:{ip_address}:requests          → sliding window counter (REST requests per minute)

# Admin pub/sub channel
admin:commands                               → pub/sub channel for force-close and admin commands
```

---

## 4. Conversation History Strategy — Finalized

### Principle

> Store only completed, meaningful turns. Never store streaming chunks. Never write conversation data to PostgreSQL under any circumstances.

### Per-Adapter Strategy

| Adapter | Redis Stores | What Is Never Stored |
|---|---|---|
| OpenAI Realtime API (WS Bridge) | Nothing | Chunks, turns, audio |
| Gemini Live API (WS Bridge) | Nothing | Chunks, turns, audio |
| OpenAI Chat Completions (HTTP Orchestrator) | Full `messages[]` array (assembled turns only) | Raw SSE chunks |
| Gemini Generate Content (HTTP Orchestrator) | Full `contents[]` array (assembled turns only) | Raw SSE chunks |
| OpenAI Responses API (HTTP Orchestrator) | Only `previous_response_id` (a short string) | Any message content |

### Assembly Mechanism

Streaming chunks are accumulated **in adapter memory only** during a turn. Nothing is written to Redis until `TextDoneEvent` fires with the fully assembled response.

```
Turn lifecycle:
  1. User message arrives → write user turn to Redis immediately
  2. Provider call made with full history payload
  3. SSE/WSS chunks arrive → accumulate in _currentAssemblyBuffer (memory only)
  4. TextDoneEvent fires → fullText = _currentAssemblyBuffer.ToString()
  5. Write assembled assistant turn to Redis
  6. Increment turn_count
  7. If turn_count > history_turn_limit → trim oldest pair
  8. Refresh TTL on history key
  9. Clear _currentAssemblyBuffer
```

### History Format in Redis

**Chat Completions adapter** (OpenAI format):
```json
{
  "turns": [
    {"role": "user",      "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ],
  "turn_count": 1
}
```

**Gemini Generate adapter** (Gemini format):
```json
{
  "turns": [
    {"role": "user",  "parts": [{"text": "What is the capital of France?"}]},
    {"role": "model", "parts": [{"text": "The capital of France is Paris."}]}
  ],
  "turn_count": 1
}
```

**Responses API adapter**:
```json
"resp_abc123xyz"
```

### Trimming Rule

When `turn_count > history_turn_limit`:
- Drop the **oldest user+assistant pair** (2 entries from the front of `turns[]`)
- Always drop in pairs — never leave an orphaned user or assistant entry
- System prompt is **never** in the `turns[]` array — it is loaded from `project_configs.system_prompt` and prepended at request-build time only

### Cleanup at Teardown

```
DEL session:{session_id}:history        # Chat Completions, Gemini Generate
DEL session:{session_id}:response_id    # Responses API
# WS Bridge adapters: no history key exists, nothing to delete
```

Explicit `DEL` is always called — do not rely on TTL expiry for cleanup.

### Session Recovery Behaviour (Node Failure)

Session recovery after gateway node crash is **not supported in v1.0**. The correct behaviour is:
- Node crash → End Client detects WebSocket/SSE disconnect
- End Client must open a new session
- New session starts with empty history for HTTP Orchestrator adapters
- This is documented as a known platform behaviour, not a defect

---

## 5. No-Change Tables

The following tables required no schema changes based on the conversations in this session. They are listed here for completeness.

| Table | Status |
|---|---|
| `users` | No changes |
| `projects` | No changes |
| `api_keys` | No changes |
| `tools` | No changes |
| `llm_providers` | No changes |
| `llm_provider_apis` | No changes |
| `llm_models` | No changes |
| `audit_logs` | No changes |

---

## 6. Constraints to Enforce in Code

These are architectural constraints that cannot be expressed as database constraints alone. They must be enforced at the repository or service layer.

| ID | Constraint | Where Enforced |
|---|---|---|
| C-01 | Every query on tenant-owned resources MUST include `user_id` or `project_id` predicate | Repository layer |
| C-02 | Raw API keys and raw ephemeral tokens MUST never be written to any database column | Service layer |
| C-03 | Conversation message content (text, audio) MUST never be written to PostgreSQL | Adapter layer + code review |
| C-04 | Redis `GETDEL` (not GET + DELETE) MUST be used for ephemeral token consumption | Session Gateway |
| C-05 | SSE connections MUST extract ephemeral token from Authorization header, not query param | Session Gateway |
| C-06 | WebSocket connections MAY use query param for ephemeral token (browser limitation) | Session Gateway + Nginx log masking |
| C-07 | Conversation history trimming MUST drop complete user+assistant pairs only | Adapter base class |
| C-08 | System prompt MUST NOT be stored in the Redis history array | Adapter base class |
| C-09 | `history_turn_limit` from `project_configs` MUST be respected by all HTTP Orchestrator adapters | Adapter base class |
| C-10 | Ephemeral tokens MUST be audit-logged (PostgreSQL) alongside the Redis fast-path entry | REST API Service |
| C-11 | The OpenAI Assistants API MUST NOT be integrated | Adapter Factory — reject at registration |
| C-12 | Session state transitions MUST only occur through the Session State Machine | Session Manager |

---

*End of change log. All changes above reflect decisions agreed in the architecture review session dated 2026-04-30.*
