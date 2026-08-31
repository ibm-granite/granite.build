# Chat API

The chat endpoints expose the AutoTuneX **assistant** — a tool-calling agent that answers
questions and performs actions (listing jobs, creating configurations, starting tuning
runs) on the caller's behalf. This page documents the chat endpoints under the
`/api/v1/chat` prefix.

Both endpoints run the same single turn; they differ only in how the result is delivered.
`POST /api/v1/chat` runs the turn to completion and returns the joined text in one JSON
body. `POST /api/v1/chat/stream` runs the same turn but streams it as
[Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events),
so a client can render per-tool status and token-by-token output as it arrives.

See [overview.md](overview.md) for shared conventions (the `ProblemDetail` error shape,
base URL, status codes) and [authentication.md](authentication.md) for how a caller is
resolved to a principal.

## Authentication and scope

Both endpoints require authentication — the `get_principal` router dependency `401`s an
unauthenticated caller before either body runs. The resolved principal scopes every tool
call the agent makes: a tool the assistant runs on your behalf sees only your own jobs,
configurations, and datasets, exactly as the corresponding REST endpoint would.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/chat` | Run one chat turn and return the joined response |
| `POST` | `/api/v1/chat/stream` | Run one chat turn and stream it as server-sent events |

---

## POST /api/v1/chat

Run one chat turn to completion and return the joined assistant text. Returns `200` with a
`ChatResponse`.

### Request body — `ChatRequest`

| Field | Type | Required | Default | Notes |
| --- | --- | --- | --- | --- |
| `messages` | `ChatMessage[]` | yes | — | The conversation history, ending with the new user turn |
| `context` | object | no | `{}` | Opaque conversation context, echoed back on the streaming endpoint's trailing `context` frame |
| `thread_id` | string \| null | no | `null` | Stable conversation id enabling server-side tool-result memory across turns (see below) |

Each `ChatMessage` is:

| Field | Type | Notes |
| --- | --- | --- |
| `role` | string | Message role, `user` or `assistant` |
| `content` | string \| null | Message text |

When `thread_id` is omitted, the full client-sent `messages` history is replayed on every
turn and nothing is persisted server-side. When a `thread_id` is supplied, the server keeps
per-thread memory (namespaced by the caller's identity, so two callers passing the same
`thread_id` never share state), and only the newest user message needs to be sent on
subsequent turns.

This per-thread memory is held **in-process**: it is not shared across workers and is lost on
restart — a single-process affordance, not durable storage. A multi-worker deployment would
need a shared cross-worker store, which is not built today; until then, either run a single
worker or omit `thread_id` and replay the full `messages` history each turn (the stateless
mode above). The idle lifetime and total number of remembered threads are bounded by
`AUTOTUNEX_CHAT_THREAD_TTL_SECONDS` and `AUTOTUNEX_CHAT_MAX_THREADS` (see
[../operations/configuration.md](../operations/configuration.md)).

A turn is also bounded in *work*: `AUTOTUNEX_CHAT_MAX_TOOL_ITERATIONS` (default
`8`) caps how many tool-call rounds the agent runs before it stops without making
a further model call. Hitting that cap is **not** reported as a failure — the turn
ends with an ordinary `done` frame on the streaming endpoint, and an ordinary
`200` here — so a turn truncated by the cap is indistinguishable from one that
finished because the model was done. Raise it if legitimate multi-step requests
are being cut short.

```bash
curl -X POST https://example.com/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-key>" \
  -d '{
    "messages": [
      { "role": "user", "content": "List my datasets" }
    ]
  }'
```

### Response `200` — `ChatResponse`

| Field | Type | Notes |
| --- | --- | --- |
| `output` | string | The assistant's joined response text |
| `context` | object | Updated conversation context (empty on the blocking endpoint) |

```json
{
  "output": "You have 2 datasets: **support-tickets** and **product-docs**.",
  "context": {}
}
```

If the turn fails cleanly (for example, an upstream LLM error mid-turn), `output` carries
the assistant's error message rather than raising — this endpoint never surfaces an
exception the agent already turned into a clean event.

### Notable statuses

| Status | When |
| --- | --- |
| `401` | No credential, or a credential that failed to verify |
| `503` | No LLM provider is configured on the server (see below) |

---

## POST /api/v1/chat/stream

Run one chat turn and stream it as Server-Sent Events. The response has content type
`text/event-stream` and carries three headers: `Cache-Control: no-cache`,
`Connection: keep-alive`, and `X-Accel-Buffering: no`. It is the last of those that
actually stops a proxy from buffering the stream into one delayed chunk — nginx (and
the proxies that honor the same header) reads `X-Accel-Buffering: no` to pass frames
straight through; `Cache-Control` only keeps the response from being cached.

The request body is the same `ChatRequest` shape as `POST /api/v1/chat`.

```bash
curl -N -X POST https://example.com/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your-key>" \
  -d '{
    "messages": [
      { "role": "user", "content": "Start a tuning job on support-tickets" }
    ],
    "thread_id": "conv-123"
  }'
```

### Frames

Every SSE frame is a single `data:` line carrying a JSON object. The object's `type` field
selects which other keys are present — the frame kind is inside the JSON payload, not in an
SSE `event:` field.

| `type` | Other keys | Meaning |
| --- | --- | --- |
| `token` | `text` (string) | A fragment of assistant text; concatenate these in order to rebuild the reply |
| `tool_start` | `name` (string), `label` (string) | A tool call began; `label` is a human-readable status line |
| `tool_end` | `name` (string) | That tool call finished |
| `refresh` | `target` (string) | A write tool succeeded; the UI should refresh the named view — `tunings` or `configs` |
| `error` | `message` (string) | The turn failed; no more turn events follow (no `done` is emitted after an `error`) |
| `done` | — | The turn is over |
| `context` | `context` (object) | Always the **last** frame; echoes the caller's `context` plus the last user message's text under `last_input` |

The `context` frame is always sent last, whether the turn succeeded, failed cleanly, or
raised. A clean turn ends with a `done` frame followed by `context`; a failed turn ends
with an `error` frame followed by `context` (never `done`).

Only two tools emit `refresh`, and each maps to exactly one view: `start_tuning_job`
sends `target: "tunings"`, and `create_config` sends `target: "configs"`. Every other
tool, read-only ones included, emits `tool_start` and `tool_end` but no `refresh`. The
frame is also **suppressed on failure**: a tool whose result begins with `Error`
finishes with a plain `tool_end` and no `refresh`, so receiving one means the write
actually landed and the named view has something new to show.

```text
data: {"type": "tool_start", "name": "list_datasets", "label": "Looking up your datasets…"}

data: {"type": "tool_end", "name": "list_datasets"}

data: {"type": "token", "text": "You have 2 datasets: "}

data: {"type": "token", "text": "**support-tickets** and **product-docs**."}

data: {"type": "done"}

data: {"type": "context", "context": {"last_input": "List my datasets"}}
```

### Notable statuses

| Status | When |
| --- | --- |
| `401` | No credential, or a credential that failed to verify |
| `503` | No LLM provider is configured on the server (see below) |

Because the response is streamed, a failure that occurs **during** the turn cannot change
an already-sent HTTP status. Such failures are delivered as an `error` frame inside the
stream rather than as a `5xx` — a `200` with a stream that ends in an `error` frame is the
normal shape of a mid-turn failure.

## The `503` case

Both endpoints resolve a chat service before running, and that service is built only when
an LLM provider is configured (`llm_base_url`, `llm_api_key`, and `llm_model` set together).
When none is configured, the endpoint returns `503 Service Unavailable` with the standard
`ProblemDetail` body **before** either body runs — the assistant is an optional feature, so
an unconfigured deployment answers with a request-time `503` rather than failing at startup.

```json
{
  "type": "about:blank",
  "title": "Service Unavailable",
  "status": 503,
  "detail": "The LLM intelligence feature is not configured on this server."
}
```

## See also

- [overview.md](overview.md) — base URL, error shape, status codes
- [authentication.md](authentication.md) — how a caller is resolved to a principal
- [mcp.md](mcp.md) — the same tool registry exposed over the Model Context Protocol
