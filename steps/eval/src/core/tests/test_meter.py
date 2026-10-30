import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from granite_evals import meter


class _Upstream(BaseHTTPRequestHandler):
    """A LiteLLM-like gateway: every completion costs $0.01, and says so."""

    protocol_version = "HTTP/1.1"
    seen: list[dict] = []
    cost_header = True

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Upstream.seen.append({"auth": self.headers.get("Authorization"), "body": body})
        usage = {"prompt_tokens": 1000, "completion_tokens": 100}
        if body.get("stream"):
            chunks = [{"choices": [{"delta": {"content": "ok"}}]}, {"choices": [], "usage": usage}]
            out = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"
            ctype = "text/event-stream"
        else:
            out = json.dumps({"choices": [{"message": {"content": "ok"}}], "usage": usage}).encode()
            ctype = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        if _Upstream.cost_header:
            self.send_header("x-litellm-response-cost", "0.01")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def upstream():
    _Upstream.seen, _Upstream.cost_header = [], True
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def _chat(base_url, **extra):
    return httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": "Bearer k"},
        json={"model": "aws/claude-sonnet-5", "messages": [], **extra},
        timeout=10,
    )


def test_options_are_metered_and_spend_is_ledgered(upstream, tmp_path, monkeypatch):
    ledger = tmp_path / "spend.jsonl"
    monkeypatch.setenv("GRANITE_EVALS_SPEND_LEDGER", str(ledger))
    monkeypatch.delenv("GRANITE_EVALS_SPEND_BUDGET_USD", raising=False)
    options = {"judge_base_url": upstream, "judge_model": "aws/claude-sonnet-5", "user_base_url": "self"}
    with meter.Meters(options, {"benchmark": "b", "job": "1"}) as meters:
        assert options["judge_base_url"] != upstream and options["user_base_url"] == "self"
        r = _chat(options["judge_base_url"])
        assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "ok"
        assert _chat(options["judge_base_url"], stream=True).text.endswith("[DONE]\n\n")
        spend = meters.summary()
    assert _Upstream.seen[0]["auth"] == "Bearer k"  # the client's key passes through
    assert _Upstream.seen[1]["body"]["stream_options"] == {"include_usage": True}
    assert spend["calls"] == 2 and spend["usd"] == pytest.approx(0.02)
    assert spend["endpoints"][0]["tokens"]["prompt_tokens"] == 2000
    assert spend["ledger_total_usd"] == pytest.approx(0.02)
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert {r["cost_source"] for r in rows} == {"gateway"} and rows[0]["benchmark"] == "b"
    assert meter.report(ledger)["by_benchmark"] == {"b": 0.02}


def test_run_total_counts_earlier_attempts(upstream, tmp_path, monkeypatch):
    # A requeued score job resumes from saved judgements: its own process pays little,
    # the output dir's total still counts the first attempt.
    ledger = tmp_path / "spend.jsonl"
    monkeypatch.setenv("GRANITE_EVALS_SPEND_LEDGER", str(ledger))
    monkeypatch.delenv("GRANITE_EVALS_SPEND_BUDGET_USD", raising=False)
    for run, n in (("/runs/a", 2), ("/runs/other", 1), ("/runs/a", 1)):
        options = {"judge_base_url": upstream}
        with meter.Meters(options, {"benchmark": "b", "run": run}) as meters:
            for _ in range(n):
                _chat(options["judge_base_url"])
            spend = meters.summary()
    assert spend["usd"] == pytest.approx(0.01) and spend["run_total_usd"] == pytest.approx(0.03)
    assert spend["ledger_total_usd"] == pytest.approx(0.04)


def test_roles_on_one_gateway_are_accounted_apart(upstream, tmp_path, monkeypatch):
    # tau's user simulator and judge share the gateway URL.
    ledger = tmp_path / "spend.jsonl"
    monkeypatch.setenv("GRANITE_EVALS_SPEND_LEDGER", str(ledger))
    options = {"user_base_url": upstream, "judge_base_url": upstream}
    with meter.Meters(options, {"benchmark": "b"}) as meters:
        assert options["user_base_url"] != options["judge_base_url"]
        _chat(options["user_base_url"]), _chat(options["user_base_url"]), _chat(meter.metered(upstream, "judge"))
        spend = meters.summary()
    assert {e["role"]: e["calls"] for e in spend["endpoints"]} == {"user": 2, "judge": 1}
    assert meter.report(ledger)["by_role"] == {"user": pytest.approx(0.02), "judge": pytest.approx(0.01)}


def test_budget_is_shared_through_the_ledger(upstream, tmp_path, monkeypatch):
    ledger = tmp_path / "spend.jsonl"
    # Another job has already spent $49.49 of $50; the reserve leaves room for one call.
    ledger.write_text(json.dumps({"cost_usd": 49.49}) + "\n")
    monkeypatch.setenv("GRANITE_EVALS_SPEND_LEDGER", str(ledger))
    monkeypatch.setenv("GRANITE_EVALS_SPEND_BUDGET_USD", "50")
    with meter.Meters({}, {"benchmark": "b"}) as meters:
        url = meter.metered(upstream, "judge")
        assert _chat(url).status_code == 200
        r = _chat(url)
        assert r.status_code == 402 and r.json()["error"]["type"] == "budget_exhausted"
        spend = meters.summary()
    assert len(_Upstream.seen) == 1
    assert spend["refused_over_budget"] == 1 and spend["ledger_total_usd"] == pytest.approx(49.5)


def test_no_cost_header_is_priced_high(upstream, tmp_path, monkeypatch):
    _Upstream.cost_header = False
    monkeypatch.setenv("GRANITE_EVALS_SPEND_LEDGER", str(tmp_path / "spend.jsonl"))
    with meter.Meters({"judge_base_url": upstream}, {}) as meters:
        _chat(meters.options["judge_base_url"])
        spend = meters.summary()
    assert spend["endpoints"][0]["calls_priced_by_fallback"] == 1
    assert spend["usd"] == pytest.approx(meter.usage_cost("sonnet", {"prompt_tokens": 1000, "completion_tokens": 100}))
    assert spend["usd"] > (1000 * 2 + 100 * 10) / 1e6  # above Sonnet 5 list price


def test_metered_is_identity_outside_a_run():
    assert meter.metered("https://gw/v1", "judge") == "https://gw/v1"
    assert meter.metered("self", "judge") == "self"
