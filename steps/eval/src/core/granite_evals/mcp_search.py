"""Minimal MCP web-search client (streamable HTTP), shared by the families that search
the web: bfcl's web_search category and the agentic tools in agent_tools."""

from __future__ import annotations

import json
import random
import re
import threading
import time

SEARCH_TOOL = "google_pse_search"


_RESULT_HEAD = re.compile(r"^\s*\d+\.\s+(.*)$")


def parse_search_text(text: str) -> list[dict[str, str]]:
    """Parse google_pse_search's text result ("Found N results for: q", then
    numbered blocks: title / URL / snippet) into BFCL's {title, href, body}."""
    results = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.strip().splitlines()]
        if len(lines) < 2:
            continue
        m = _RESULT_HEAD.match(lines[0])
        if not m or not lines[1].startswith(("http://", "https://")):
            continue
        results.append({"title": m.group(1).strip(), "href": lines[1], "body": " ".join(lines[2:]).strip()})
    return results


def _jsonrpc_payload(response) -> dict:
    """A JSON-RPC message from a streamable-HTTP response (plain JSON or SSE)."""
    if "text/event-stream" in response.headers.get("content-type", ""):
        for line in response.text.splitlines():
            if line.startswith("data:"):
                data = line[5:].strip()
                if data:
                    msg = json.loads(data)
                    if "result" in msg or "error" in msg:
                        return msg
        raise RuntimeError("no JSON-RPC result in event stream")
    return response.json()


class MCPSearch:
    """Minimal streamable-HTTP MCP client for one tool (thread-safe, re-initializes
    the session on expiry, retries throttling and server errors with backoff)."""

    def __init__(self, url: str, *, client=None, max_attempts: int = 8, backoff_s: float = 2.0):
        import httpx

        self.url = url
        self.client = client or httpx.Client(timeout=60.0)
        self.max_attempts = max_attempts
        self.backoff_s = backoff_s
        self.session: str | None = None
        self._lock = threading.RLock()
        self._id = 0
        self.stats = {"calls": 0, "errors": 0, "retries": 0, "throttled": 0, "empty": 0}

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.session:
            h["mcp-session-id"] = self.session
        return h

    def _next_id(self) -> int:
        with self._lock:
            self._id += 1
            return self._id

    def _initialize(self) -> None:
        self.session = None
        r = self.client.post(self.url, headers=self._headers(), json={
            "jsonrpc": "2.0", "id": self._next_id(), "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "granite-evals-bfcl", "version": "1"}},
        })
        r.raise_for_status()
        self.session = r.headers.get("mcp-session-id")
        self.client.post(self.url, headers=self._headers(),
                         json={"jsonrpc": "2.0", "method": "notifications/initialized"})

    def search(self, query: str, num_results: int) -> list[dict[str, str]]:
        self.stats["calls"] += 1
        body = {"jsonrpc": "2.0", "method": "tools/call",
                "params": {"name": SEARCH_TOOL, "arguments": {"query": query, "num_results": num_results}}}
        backoff = self.backoff_s
        last = ""
        for attempt in range(self.max_attempts):
            try:
                with self._lock:
                    if self.session is None:
                        self._initialize()
                r = self.client.post(self.url, headers=self._headers(), json=body | {"id": self._next_id()})
                if r.status_code in (400, 404) and "session" in r.text.lower():
                    with self._lock:
                        self.session = None
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                    continue
                if r.status_code == 429 or r.status_code >= 500:
                    self.stats["throttled" if r.status_code == 429 else "retries"] += 1
                    last = f"HTTP {r.status_code}"
                    raise _Retry()
                r.raise_for_status()
                msg = _jsonrpc_payload(r)
                if "error" in msg:
                    raise RuntimeError(f"MCP error: {msg['error']}")
                result = msg["result"]
                text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
                if result.get("isError"):
                    if "429" in text or "rate" in text.lower():
                        self.stats["throttled"] += 1
                        last = text[:200]
                        raise _Retry()
                    raise RuntimeError(text[:500])
                hits = parse_search_text(text)
                if not hits:
                    self.stats["empty"] += 1
                return hits
            except _Retry:
                pass
            except Exception as e:  # network errors: retry; the last one is reported
                last = f"{type(e).__name__}: {e}"
                if isinstance(e, RuntimeError) and str(e).startswith("MCP error"):
                    break
            self.stats["retries"] += 1
            time.sleep(backoff + random.uniform(0, backoff))
            backoff = min(backoff * 2, 60)
        self.stats["errors"] += 1
        raise RuntimeError(f"web search failed after {self.max_attempts} attempts: {last}")


class _Retry(Exception):
    pass
