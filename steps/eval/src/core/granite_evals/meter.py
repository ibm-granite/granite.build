"""Meter and cap what paid APIs (judges, user simulators) cost a run.

Benchmarks send every call to a paid OpenAI-compatible endpoint through a
local proxy: :func:`metered` maps the endpoint's base URL to the proxy's. The
proxy forwards the request unchanged (the client's own Authorization header
included; the proxy never sees a key otherwise), records the call's cost and
refuses new calls with HTTP 402 once the budget is spent.

The cost is the gateway's own ``x-litellm-response-cost`` header (
LiteLLM gateway sets it on every response). Without it, the cost is priced
from the response's token usage at deliberately high fallback prices.

Every call is appended to a JSONL ledger. A ledger shared by concurrent jobs
(on a shared filesystem, appends under ``lockf``) makes the budget global
across all of them: ``GRANITE_EVALS_SPEND_LEDGER`` names it, ``GRANITE_EVALS_SPEND_BUDGET_USD``
caps the ledger's total. A new call is refused once total + reserve reaches
the budget, so calls already in flight can't carry the total past it.
The ledger holds costs and token counts only, never prompts or keys.

The CLI starts a :class:`Meters` around each run: options named ``*_base_url``
with an http(s) value are rewritten to their proxy, and benchmarks route any
other paid endpoint (e.g. a class default) through :func:`metered`. The run's
spend goes to results.json as ``details.api_spend``; ``run_total_usd`` adds
the ledger's earlier attempts on the same output dir.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import threading
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

log = logging.getLogger(__name__)

# $ per million tokens (input, output), only for responses without a cost header.
# Set above list prices so a fallback overestimates.
FALLBACK_PRICES = {"haiku": (1.5, 7.5), "sonnet": (4.0, 20.0), "opus": (15.0, 75.0)}
DEFAULT_PRICE = (15.0, 75.0)
RESERVE_USD = 0.5
_HOP_HEADERS = {"host", "content-length", "accept-encoding", "connection", "transfer-encoding", "content-encoding"}


class Ledger:
    """Append-only JSONL of calls; ``total()`` is what every job has spent."""

    def __init__(self, path: Path | None):
        self.path = path
        self._lock = threading.Lock()

    def total(self, run: str | None = None) -> float:
        """What every job has spent, or only the calls tagged with output dir ``run``."""
        if not self.path or not self.path.exists():
            return 0.0
        total = 0.0
        for line in self.path.read_text().splitlines():
            try:
                entry = json.loads(line)
                if run is None or entry.get("run") == run:
                    total += float(entry.get("cost_usd", 0))
            except (ValueError, AttributeError):
                continue
        return total

    def record(self, entry: dict[str, Any]) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, sort_keys=True) + "\n"
        with self._lock, self.path.open("a") as f:
            fcntl.lockf(f, fcntl.LOCK_EX)
            f.write(line)
            f.flush()
            fcntl.lockf(f, fcntl.LOCK_UN)


def usage_cost(model: str, usage: dict[str, Any]) -> float:
    """Fallback price of a response's token usage."""
    price_in, price_out = next((p for k, p in FALLBACK_PRICES.items() if k in model.lower()), DEFAULT_PRICE)
    return (usage.get("prompt_tokens", 0) * price_in + usage.get("completion_tokens", 0) * price_out) / 1e6


def _usage_from_sse(body: bytes) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for line in body.splitlines():
        if line.startswith(b"data: {") and b'"usage"' in line:
            try:
                usage = json.loads(line[6:]).get("usage") or usage
            except ValueError:
                continue
    return usage


class Meter:
    """A proxy in front of one upstream endpoint."""

    def __init__(self, upstream: str, *, role: str, ledger: Ledger, budget_usd: float | None, tags: dict[str, Any]):
        self.upstream = upstream.rstrip("/")
        self.role = role
        self.ledger = ledger
        self.budget_usd = budget_usd
        self.tags = tags
        self.calls = 0
        self.refused = 0
        self.usd = 0.0
        self.tokens: dict[str, int] = defaultdict(int)
        self.by_model: dict[str, float] = defaultdict(float)
        self.estimated_calls = 0
        self._lock = threading.Lock()
        self._client = httpx.Client(timeout=httpx.Timeout(900, connect=30))
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        split = urlsplit(self.upstream)
        self._origin = f"{split.scheme}://{split.netloc}"
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}{split.path}"

    def start(self) -> Meter:
        self._thread.start()
        log.info("metering %s calls to %s via %s", self.role, self.upstream, self.url)
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._client.close()

    def over_budget(self) -> bool:
        return self.budget_usd is not None and self.ledger.total() + RESERVE_USD >= self.budget_usd

    def account(self, model: str, status: int, usage: dict[str, Any], cost_header: str | None) -> None:
        if cost_header is not None:
            cost, source = float(cost_header), "gateway"
        else:
            cost, source = usage_cost(model, usage), "fallback-price"
        details = usage.get("prompt_tokens_details") or {}
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **self.tags,
            "role": self.role,
            "model": model,
            "status": status,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "cached_tokens": details.get("cached_tokens", usage.get("cache_read_input_tokens", 0)) or 0,
            "cost_usd": cost,
            "cost_source": source,
        }
        with self._lock:
            self.calls += 1
            self.usd += cost
            self.by_model[model] += cost
            self.estimated_calls += source != "gateway"
            for k in ("prompt_tokens", "completion_tokens", "cached_tokens"):
                self.tokens[k] += entry[k]
        self.ledger.record(entry)

    def summary(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "upstream": self.upstream,
            "calls": self.calls,
            "refused_over_budget": self.refused,
            "usd": round(self.usd, 6),
            "usd_by_model": {k: round(v, 6) for k, v in self.by_model.items()},
            "tokens": dict(self.tokens),
            "calls_priced_by_fallback": self.estimated_calls,
        }

    def _handler(self):
        meter = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):  # the run log has the benchmark's own lines
                pass

            def _send(self, status: int, headers: dict[str, str], body: bytes) -> None:
                self.send_response(status)
                for k, v in headers.items():
                    if k.lower() not in _HOP_HEADERS:
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _forward(self, method: str) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_HEADERS}
                url = meter._origin + self.path
                if method == "GET":
                    r = meter._client.get(url, headers=headers)
                    return self._send(r.status_code, dict(r.headers), r.content)
                if meter.over_budget():
                    with meter._lock:
                        meter.refused += 1
                    log.error("granite spend budget of $%.2f reached; refusing %s call", meter.budget_usd, meter.role)
                    msg = {"error": {"message": "granite spend budget exhausted", "type": "budget_exhausted"}}
                    return self._send(402, {"Content-Type": "application/json"}, json.dumps(msg).encode())
                try:
                    request = json.loads(body or b"{}")
                except ValueError:
                    request = {}
                model = str(request.get("model", ""))
                if request.get("stream"):
                    # Ask for the usage chunk, so a stream can be priced too.
                    request.setdefault("stream_options", {})["include_usage"] = True
                    body = json.dumps(request).encode()
                r = meter._client.post(url, headers=headers, content=body)
                content = r.content
                if request.get("stream"):
                    usage = _usage_from_sse(content)
                else:
                    try:
                        usage = r.json().get("usage") or {}
                    except (ValueError, AttributeError):
                        usage = {}
                if r.status_code < 400 or r.headers.get("x-litellm-response-cost"):
                    meter.account(model, r.status_code, usage, r.headers.get("x-litellm-response-cost"))
                self._send(r.status_code, dict(r.headers), content)

            def do_GET(self):
                self._forward("GET")

            def do_POST(self):
                self._forward("POST")

        return Handler


class Meters:
    """The meters of one run; the CLI enters one around ``benchmark.run``."""

    _active: Meters | None = None

    def __init__(self, options: dict[str, Any], tags: dict[str, Any]):
        path = os.environ.get("GRANITE_EVALS_SPEND_LEDGER")
        budget = os.environ.get("GRANITE_EVALS_SPEND_BUDGET_USD")
        self.ledger = Ledger(Path(path) if path else None)
        self.budget_usd = float(budget) if budget else None
        self.options = options
        self.tags = tags
        self._meters: dict[tuple[str, str], Meter] = {}

    def url(self, upstream: str, role: str) -> str:
        # One meter per endpoint and role: a judge and a user simulator on the same
        # gateway are accounted apart.
        key = (upstream.rstrip("/"), role)
        if key not in self._meters:
            self._meters[key] = Meter(
                key[0], role=role, ledger=self.ledger, budget_usd=self.budget_usd, tags=self.tags
            ).start()
        return self._meters[key].url

    def __enter__(self) -> Meters:
        Meters._active = self
        for key, value in list(self.options.items()):
            if key.endswith("_base_url") and str(value).startswith(("http://", "https://")):
                self.options[key] = self.url(str(value), role=key.removesuffix("_base_url"))
        return self

    def __exit__(self, *exc) -> None:
        Meters._active = None
        for m in self._meters.values():
            m.close()

    def summary(self) -> dict[str, Any] | None:
        if not self._meters:
            return None
        meters = [m.summary() for m in self._meters.values()]
        return {
            "usd": round(sum(m["usd"] for m in meters), 6),
            "calls": sum(m["calls"] for m in meters),
            "refused_over_budget": sum(m["refused_over_budget"] for m in meters),
            "budget_usd": self.budget_usd,
            "ledger": str(self.ledger.path) if self.ledger.path else None,
            "ledger_total_usd": round(self.ledger.total(), 6) if self.ledger.path else None,
            # Every attempt and phase on this output dir: a job the queue requeued resumes
            # from saved results, so ``usd`` (this process) leaves out the earlier attempts.
            "run_total_usd": round(self.ledger.total(run=self.tags.get("run")), 6)
            if self.ledger.path and self.tags.get("run")
            else None,
            "endpoints": meters,
        }


def metered(upstream: str, role: str) -> str:
    """The URL a benchmark must use for a paid endpoint. Outside a CLI run
    (tests with fakes) there is nothing to meter and the URL is returned."""
    if Meters._active is None or not upstream.startswith(("http://", "https://")):
        return upstream
    return Meters._active.url(upstream, role)


def report(path: Path) -> dict[str, Any]:
    """Totals of a ledger by benchmark, model, role and job."""
    out: dict[str, Any] = {"usd": 0.0, "calls": 0}
    groups: dict[str, dict[str, float]] = {k: defaultdict(float) for k in ("benchmark", "model", "role", "job")}
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        cost = float(e.get("cost_usd", 0))
        out["usd"] += cost
        out["calls"] += 1
        for k, g in groups.items():
            g[str(e.get(k, "-"))] += cost
    out["usd"] = round(out["usd"], 4)
    out.update({f"by_{k}": {n: round(v, 4) for n, v in sorted(g.items())} for k, g in groups.items()})
    return out
