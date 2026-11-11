"""Scale AI MultiChallenge (extended suite, group "chat-if").

MultiChallenge (Sirdeshmukh et al. 2025, arXiv:2501.17399): 273 multi-turn
conversations ending in a user turn, each with a human-written YES/NO rubric
question about the final assistant reply, over four axes (INFERENCE_MEMORY,
INSTRUCTION_RETENTION, SELF_COHERENCE, RELIABLE_VERSION_EDITING). Data and
protocol: ekwinox117/multi-challenge@5ccefcca6a39020d66c1383c4e6a809cb07afa33
(``data/benchmark_questions.jsonl``, ``src/evaluator.py``,
``src/result_parser.py``), sha256-verified.

The served model continues each conversation (the whole ``CONVERSATION`` as
chat messages, no system prompt; sampling = the checkpoint's
generation_config). The judge sees only the final answer and the rubric
question (upstream's ``JUDGE_PROMPT``, temperature 0, structured output
``{reasoning, verdict: YES|NO}``); a response passes when the verdict equals
``PASS_CRITERIA``. Upstream's score: per axis, the fraction of questions with
a passing attempt; overall = the unweighted mean of the four axis scores.

``value`` = pass@1 (macro over axes, as upstream). With ``--repeats k`` (default
1) it is pass@1[avg-of-k]: each question's pass rate over its k attempts,
averaged per axis, then over axes; ``details.pass_at_k`` is upstream's own
score over those k attempts (a question passes if any attempt does).

Deviations (``details.deviations``): judge Claude Sonnet 5 via the gateway
instead of gemini-3.1-flash-lite (gemini-3.1-pro fallback); judge failures (errors, refusals, unparseable
verdicts) and generation errors are recorded failures left out of the score
(upstream counts them as NO), bounded by ``max_failed_frac``; an empty final
answer fails without a judge call.

Gold / sanity mode: ``--option responses=<model>`` grades the responses
upstream ships in ``data/final_model_responses/`` (claude-3-5-sonnet-20241022,
gpt-4o-2024-08-06, o1-preview; one attempt each) without a served model; the
paper's automatic-judge averages (Table 3) are in ``details.paper_reference``.
"""

from __future__ import annotations

import collections
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

from granite_evals import data
from granite_evals.benchmarks.judge_general import Judge
from granite_evals.benchmarks.safety import (
    JudgeFailure,
    ResponseJudgedBenchmark,
    judge_failure_reason,
    load_override,
    pinned_file,
    read_rows,
)
from granite_evals.registry import pass_at_k_record, register

log = logging.getLogger(__name__)

MC_REPO = "ekwinox117/multi-challenge"
MC_COMMIT = "5ccefcca6a39020d66c1383c4e6a809cb07afa33"
MC_QUESTIONS = ("data/benchmark_questions.jsonl", "06c72d1ffa387481910d100b146295f28ab68c4ba0032743eac5c9195edb596e")
MC_RESPONSES = {
    "claude-3-5-sonnet-20241022": "d83582385768fdc9006e0882dc9f11cb6ac8cdec06eae3a25a7e2ea88a4761fb",
    "gpt-4o-2024-08-06": "12579bda05fb4dcc7330b093d93cd3eec5d89005f3d1be559fe7457048a31b96",
    "o1-preview": "94398716d955e961db3b0d1ca676d8209f3246597dcf1ddd9930b3ad5adef255",
}
# arXiv:2501.17399 Table 3 (automatic evaluation, GPT-4o judge), percent. The paper names
# "Claude 3.5 Sonnet (June 2024)" in the table and "(October 2024)" in its appendix.
MC_PAPER_AUTO = {
    "claude-3-5-sonnet-20241022": {"INSTRUCTION_RETENTION": 61.43, "INFERENCE_MEMORY": 37.29,
                                   "RELIABLE_VERSION_EDITING": 26.83, "SELF_COHERENCE": 45.45, "average": 42.75},
    "gpt-4o-2024-08-06": {"INSTRUCTION_RETENTION": 12.86, "INFERENCE_MEMORY": 5.93,
                          "RELIABLE_VERSION_EDITING": 14.63, "SELF_COHERENCE": 18.18, "average": 12.9},
    "o1-preview": {"INSTRUCTION_RETENTION": 28.57, "INFERENCE_MEMORY": 38.98,
                   "RELIABLE_VERSION_EDITING": 39.02, "SELF_COHERENCE": 36.36, "average": 35.73},
}
MC_AXES = ("INFERENCE_MEMORY", "INSTRUCTION_RETENTION", "RELIABLE_VERSION_EDITING", "SELF_COHERENCE")

# src/evaluator.py JUDGE_PROMPT (verbatim), formatted with (response, TARGET_QUESTION).
MC_JUDGE_PROMPT = '''You are tasked with evaluating a model response to see if it meets a specific criteria.
The criteria will always be YES/NO evaluation.

The model response is as follows:
<MODEL_RESPONSE>
{}
</MODEL_RESPONSE>

The criteria that the model response must meet is as follows. Be VERY STRICT!:
<CRITERIA>
{}
</CRITERIA>

Print your reasoning followed by your verdict, either "YES" or "NO".'''

# Upstream's pydantic JudgeResponse as a JSON schema (OpenAI structured output).
MC_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "JudgeResponse",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"reasoning": {"type": "string"}, "verdict": {"type": "string", "enum": ["YES", "NO"]}},
            "required": ["reasoning", "verdict"],
            "additionalProperties": False,
        },
    },
}

_VERDICT_LABEL = re.compile(r"verdict\W{0,10}(YES|NO)\b", re.IGNORECASE)
_VERDICT_WORD = re.compile(r"\b(YES|NO)\b")


def parse_verdict(text: str) -> str | None:
    """YES / NO from a structured reply ``{"reasoning", "verdict"}``, else from
    text: the last "verdict: YES|NO", else the last standalone upper-case YES / NO."""
    body = text.strip()
    if body.startswith("```"):
        body = body.strip("`").removeprefix("json").strip()
    try:
        obj = json.loads(body)
        if isinstance(obj, dict) and str(obj.get("verdict", "")).upper() in ("YES", "NO"):
            return str(obj["verdict"]).upper()
    except ValueError:
        pass
    labelled = _VERDICT_LABEL.findall(text)
    if labelled:
        return labelled[-1].upper()
    words = _VERDICT_WORD.findall(text)
    return words[-1] if words else None


def macro(per_item: dict[str, float], axis_of: dict[str, str]) -> tuple[float, dict[str, float]]:
    """Upstream's overall score: per-axis mean, then the unweighted mean over axes."""
    axes: dict[str, list[float]] = collections.defaultdict(list)
    for qid, v in per_item.items():
        axes[axis_of[qid]].append(v)
    scores = {a: sum(v) / len(v) for a, v in sorted(axes.items())}
    return sum(scores.values()) / len(scores), scores


@register
class MultiChallenge(ResponseJudgedBenchmark):
    """MultiChallenge, judged pass/fail per conversation.

    Options (besides ResponseJudgedBenchmark's):
      responses=model|claude-3-5-sonnet-20241022|gpt-4o-2024-08-06|o1-preview
                         the served model (default) or upstream's shipped responses
      judge_structured=true
                         send upstream's structured output schema (response_format);
                         dropped once, and recorded, if the endpoint rejects it
    """

    id = "multi-challenge"
    metric = "pass@1 correct"
    dataset = f"github.com/{MC_REPO}"
    dataset_revision = MC_COMMIT
    gold_modes = tuple(MC_RESPONSES)
    metric_note = ("value = pass@1[avg-of-k] (k = repeats, default 1): per-question pass rate, mean per axis, "
                   "unweighted mean over the 4 axes (upstream result_parser); pass_at_k = any attempt passes")
    deviations = (
        "judge Claude Sonnet 5 via the gateway instead of gemini-3.1-flash-lite (gemini-3.1-pro fallback)",
        "judge failures and generation errors are recorded failures left out of the score "
        "(upstream counts them as NO), bounded by max_failed_frac",
        "an empty final answer fails without a judge call",
        "policy sampling = the checkpoint's generation_config (the paper used per-model temperatures)",
    )

    def __init__(self, config):
        super().__init__(config)
        self._lock = threading.Lock()
        self._structured: bool | None = None  # None until the option is read
        self.judge_adaptations: list[str] = []
        self._shipped: dict[str, list[str]] | None = None

    def load_items(self) -> tuple[list[dict], dict[str, Any]]:
        source, revision = self.dataset_source()
        provenance: dict[str, Any] = {"dataset": source, "dataset_revision": revision,
                                      "harness": f"github.com/{MC_REPO}@{MC_COMMIT} src/evaluator.py, "
                                                 "src/result_parser.py"}
        if (source, revision) == (self.dataset, MC_COMMIT):
            rel, sha = MC_QUESTIONS
            rows = read_rows(self._fetch(rel, sha))
            provenance["dataset_sha256"] = {rel: sha}
        else:
            rows = load_override(source, revision, "train")
        if self.responses != "model":
            if self.repeats != 1:
                raise SystemExit(f"multi-challenge: responses={self.responses} has one attempt per question; "
                                 "use --repeats 1")
            rel = f"data/final_model_responses/{self.responses}_responses.jsonl"
            self._shipped = {r["QUESTION_ID"]: list(r["RESPONSE"])
                             for r in read_rows(self._fetch(rel, MC_RESPONSES[self.responses]))}
            provenance["dataset_sha256"] = {**provenance.get("dataset_sha256", {}), rel: MC_RESPONSES[self.responses]}
            provenance["paper_reference"] = {"source": "arXiv:2501.17399 Table 3 (automatic, GPT-4o judge), percent",
                                             **MC_PAPER_AUTO[self.responses]}
        items = [{"id": r["QUESTION_ID"], "axis": r["AXIS"], "conversation": r["CONVERSATION"],
                  "target_question": r["TARGET_QUESTION"], "pass_criteria": r["PASS_CRITERIA"]} for r in rows]
        return data.take(items, self.config.limit, key="id"), provenance

    @staticmethod
    def _fetch(rel: str, sha: str) -> Path:
        url = f"https://raw.githubusercontent.com/{MC_REPO}/{MC_COMMIT}/{rel}"
        return pinned_file(url, sha, f"multi-challenge/{MC_COMMIT}/{rel}")

    def messages(self, item: dict) -> list[dict]:
        return [{"role": m["role"], "content": m["content"]} for m in item["conversation"]]

    def gold_response(self, item: dict, k: int) -> str | None:
        attempts = (self._shipped or {}).get(item["id"], [])
        return attempts[k] if k < len(attempts) else None  # missing: a recorded failure (upstream: NO)

    def empty_result(self, item: dict) -> dict | None:
        return {"verdict": None, "passed": False}

    # -- judge -------------------------------------------------------------------

    def judge_one(self, judge: Judge, item: dict, response: str) -> dict:
        messages = [{"role": "user", "content": MC_JUDGE_PROMPT.format(response, item["target_question"])}]
        with self._lock:
            if self._structured is None:
                self._structured = self.opt("judge_structured", True)
            structured = self._structured
        params: dict[str, Any] = {"temperature": 0}
        if structured:
            params["response_format"] = MC_RESPONSE_FORMAT
        try:
            text, usage = self.ask(judge, messages, **params)
        except Exception as e:
            if not (structured and getattr(e, "status_code", None) == 400 and "response_format" in str(e).lower()):
                raise
            with self._lock:
                if self._structured:
                    self._structured = False
                    self.judge_adaptations.append(f"dropped response_format: {str(e)[:300]}")
            log.warning("multi-challenge: judge rejected response_format; verdicts parsed from text")
            params.pop("response_format")
            text, usage = self.ask(judge, messages, **params)
        verdict = parse_verdict(text)
        if verdict is None:
            raise JudgeFailure(judge_failure_reason(text), text, usage)
        return {"verdict": verdict, "passed": verdict == item["pass_criteria"], "judge_output": text,
                "usage": usage}

    # -- score ---------------------------------------------------------------------

    def score(self, rows) -> dict[str, Any]:
        attempts: dict[str, list[bool]] = collections.defaultdict(list)
        axis_of: dict[str, str] = {}
        for item, _, r in rows:
            attempts[str(item["id"])].append(bool(r["passed"]))
            axis_of[str(item["id"])] = item["axis"]
        structured = self.opt("judge_structured", True) if self._structured is None else self._structured
        pass1, axes1 = macro({q: sum(a) / len(a) for q, a in attempts.items()}, axis_of)
        out: dict[str, Any] = {
            "value": pass1,
            "metric_name": f"pass@1[avg-of-{self.repeats}]" if self.repeats > 1 else "pass@1",
            "axis_scores": {a: round(v, 6) for a, v in axes1.items()},
            "axis_counts": dict(sorted(collections.Counter(axis_of.values()).items())),
            "judge_structured_output": bool(structured),
            "judge_request_adaptations": list(self.judge_adaptations),
            "judge_request": "upstream src/evaluator.py JUDGE_PROMPT as the only user message, temperature 0"
                             + (", response_format JudgeResponse{reasoning, verdict}" if structured else ""),
        }
        passk = pass1
        if self.repeats > 1:
            passk, axesk = macro({q: float(any(a)) for q, a in attempts.items()}, axis_of)
            by_k: dict[int, dict[str, float]] = collections.defaultdict(dict)
            for item, k, r in rows:
                by_k[k][str(item["id"])] = float(r["passed"])
            out.update({
                "pass_at_k_axis_scores": {a: round(v, 6) for a, v in axesk.items()},
                "per_repeat": [round(macro(by_k[k], axis_of)[0], 6) for k in sorted(by_k)],
            })
        out["pass_at_k"] = pass_at_k_record(self.repeats, pass1, passk, len(attempts),
                                            "macro over the 4 axes; pass@k: any attempt passes")
        return out
