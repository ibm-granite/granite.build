"""BFCL v3 (Berkeley Function Calling Leaderboard), Granite metric "overall_accuracy".

The v3 leaderboard's own harness, data and formula: ``bfcl-eval==2025.8.6.2``,
the last PyPI release before BFCL V4 (2025.8.25). Its 17 categories (4,441
entries: non-live AST, live AST, relevance / irrelevance, multi-turn) and its
"Overall Acc" (score/<handler>/data_overall.csv) define the number: the unweighted mean
of the live overall (entry-weighted over the six live categories), the
non-live overall (unweighted over simple, multiple, parallel, parallel
multiple and irrelevance) and the multi-turn overall (unweighted over four).

Not "v3 on the v4 package": V4 re-weights the overall and its multi-turn
data differ from v3's in 451 of 800 entries (questions, initial states,
excluded functions, ground truth) and in the simulators that replay them, so
the v3 categories of ``bfcl-v4``'s pinned release do not give v3 numbers.

v3 and v4 are one distribution (``bfcl_eval``) at two releases, so v3 lives in
its own side venv in the bfcl image (``GRANITE_BFCL_V3_PYTHON``, default
``/opt/bfcl-v3/bin/python``, hash-locked) and runs as a child process
(:mod:`bfcl_v3_child`, run with ``python -I`` so the job venv cannot leak in).
Two request formats (``--option handler=``). ``fc``, the default, is
``bfcl-v4``'s: FC mode against the vLLM endpoint, the checkpoint's own chat
template and tool parser, sampling only when set. ``legacy-granite`` is BFCL
prompt mode with the Granite 4.0 prompt format rendered by the handler and sent
as text to the completions endpoint, the format Granite 4.0 BFCL v3 numbers were
produced with (see :func:`bfcl_v3_child.granite4_prompt` for the template's
source); the handler parses the tool calls from the model's ``<tool_call>``
blocks itself. Both send ``temperature`` and ``top_p`` only when set
(``temperature=0.001`` is BFCL's own default, which those runs used).
``max_tokens``: ``fc`` sends it only when set; ``legacy-granite`` always sends
one, BFCL prompt mode's min(4096, context - prompt - 2) unless set,
counted by the serving vLLM (/tokenize, max_model_len), so it needs a vLLM endpoint.
Infrastructure errors: failure_policy, as for ``bfcl-v4`` (``max_failed_frac``).
No web search, no memory: v3 has neither.

Phases, resume and ``--repeats`` as in ``bfcl-v4``.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from granite_evals.benchmarks.bfcl import _pct, budget_record, infra_failure, prune_infra_errors
from granite_evals.benchmarks.bfcl_v3_child import (
    HARNESS_VERSIONS,
    INFRA_EXIT,
    MISSING_EXIT,
    DEFAULT_HANDLER,
    REGISTRY_NAMES,
    RESULT_PREFIX,
    TEST_IDS_FILE,
    DEFAULT_MAX_FAILED_FRAC,
    failed_entries,
    missing_results,
    registry_name,
    result_files,
    score_dir,
    selected_ids,
)
from granite_evals.registry import Benchmark, failure_policy, pass_at_k_record, register

log = logging.getLogger(__name__)

HARNESS = "bfcl-eval"
HARNESS_VERSION = "2025.8.6.2"
CHILD = Path(__file__).with_name("bfcl_v3_child.py")
CHILD_CONFIG = "granite_bfcl_v3_config.json"
PYTHON_ENV = "GRANITE_BFCL_V3_PYTHON"
DEFAULT_PYTHON = "/opt/bfcl-v3/bin/python"
HANDLER_DESCRIPTIONS = {
    "fc": "OpenAICompletionsHandler (FC) via the checkpoint's chat template and vLLM tool parser",
    "legacy-granite": "BFCL prompt mode, Granite 4.0 chat-template format (granite-4.0-micro@b77de16e), "
                      "completions endpoint, <tool_call> blocks parsed by the handler (extract_tool_calls), "
                      "max_tokens min(4096, context - prompt - 2) unless set",
}

OVERALL_GROUPS = {
    "non_live_ast": "Non-Live AST Acc",  # AST summary only; Overall uses non_live (below)
    "live": "Live Acc",
    "multi_turn": "Multi Turn Acc",
    "relevance": "Relevance Detection",
    "irrelevance": "Irrelevance Detection",
}
NON_LIVE_OVERALL = "Non_Live Overall Acc"  # data_non_live.csv: the non-live term of Overall Acc


def read_overall(score_dir: Path) -> dict[str, Any]:
    """BFCL v3's own "Overall Acc" and its group scores, as fractions."""
    with (score_dir / "data_overall.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"{score_dir}/data_overall.csv: expected one model row, found {len(rows)}")
    if None in rows[0]:  # BFCL writes unquoted: a "," in a cell shifts every later column
        raise RuntimeError(f"{score_dir}/data_overall.csv: more cells than columns (a ',' in the model name?)")
    groups = {k: _pct(rows[0][col]) for k, col in OVERALL_GROUPS.items()}
    non_live = score_dir / "data_non_live.csv"
    if non_live.exists():
        with non_live.open(newline="") as f:
            groups["non_live"] = _pct(next(csv.DictReader(f))[NON_LIVE_OVERALL])
    return {"overall": _pct(rows[0]["Overall Acc"]) or 0.0, "groups": groups}


def read_category_scores(score_dir: Path) -> dict[str, dict[str, Any]]:
    """Per-category header ({accuracy, correct_count, total_count}) of each v3 score file."""
    out = {}
    for path in sorted(score_dir.glob(f"{RESULT_PREFIX}*_score.json")):
        cat = path.name[len(RESULT_PREFIX):-len("_score.json")]
        with path.open() as f:
            header = json.loads(f.readline())
        out[cat] = {k: header[k] for k in ("accuracy", "correct_count", "total_count")}
    return out


@register
class BFCLv3(Benchmark):
    id = "bfcl-v3"
    metric = "overall_accuracy accuracy"
    default_repeats = 1
    extra = "bfcl"
    # The harness is in the side venv, not the job venv results.json reads
    # versions from; details.harness_versions records the side venv's.
    harness_packages = ()
    splittable = True
    dataset = f"https://pypi.org/project/{HARNESS}/{HARNESS_VERSION}/"
    dataset_revision = "sha256:fdb996ac963ed1c6fce1eb4884053aae4a6de282fb931665a46dbeed204a228d"

    def opt(self, key: str, default: Any) -> Any:
        value = self.config.options.get(key)
        return default if value is None else type(default)(value)

    def sampling(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, cast in (("temperature", float), ("top_p", float), ("max_tokens", int)):
            if key in self.config.options:
                out[key] = cast(self.config.options[key])
        return out

    def python(self) -> str:
        python = os.environ.get(PYTHON_ENV) or DEFAULT_PYTHON
        if not Path(python).exists():
            raise SystemExit(f"{self.id}: no BFCL v3 interpreter at {python} (the bfcl image's side venv; "
                             f"set {PYTHON_ENV} to a venv with {HARNESS}=={HARNESS_VERSION})")
        return python

    def max_tokens_record(self, handler: str) -> dict[str, Any]:
        """results.json's max_tokens: fc's is the step's served budget (bfcl.budget_record);
        legacy-granite sends its own with every request."""
        if handler == "fc":
            return budget_record(self.config.output_dir, self.config.options)
        explicit = self.config.options.get("max_tokens")
        return {"cap": int(explicit) if explicit else "min(4096, context - prompt - 2)",
                "source": "legacy-granite, with every request"}

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        if self.repeats != 1:
            raise SystemExit(f"{self.id} is a single-run metric (one BFCL result per test id); use --repeats 1")
        python = self.python()
        root = self.config.output_dir / "bfcl"
        root.mkdir(parents=True, exist_ok=True)
        categories = [c for c in self.opt("categories", "all").split(",") if c]
        handler = self.opt("handler", DEFAULT_HANDLER)
        if handler not in REGISTRY_NAMES:
            raise SystemExit(f"{self.id}: unknown handler {handler!r} (one of: {', '.join(REGISTRY_NAMES)})")
        child = {
            "served_model_name": served_model_name,
            "categories": categories,
            "limit": self.config.limit,
            "num_threads": self.config.workers,
            "sampling": self.sampling(),
            "include_input_log": self.opt("include_input_log", "false").lower() == "true",
            "phase": self.phase,
            "handler": handler,
            "max_failed_frac": self.opt("max_failed_frac", DEFAULT_MAX_FAILED_FRAC),
        }
        (root / CHILD_CONFIG).write_text(json.dumps(child, indent=2))
        if self.generating:
            for f in result_files(root, handler):
                if n := prune_infra_errors(f):
                    log.info("%s: %s: retrying %d entries that hit infrastructure errors", self.id, f.name, n)
        env = {
            **os.environ,
            "BFCL_PROJECT_ROOT": str(root),
            "OPENAI_BASE_URL": base_url,
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY") or "EMPTY",
            "TQDM_DISABLE": "1",
        }
        if not base_url:
            env.pop("OPENAI_BASE_URL")  # score phase: the handler is built, never called
        log.info("%s: categories=%s limit=%s workers=%d sampling=%s", self.id, ",".join(categories),
                 self.config.limit, self.config.workers, child["sampling"] or "generation_config")
        proc = subprocess.run([python, "-I", str(CHILD), str(root / CHILD_CONFIG)], env=env, check=False)
        if proc.returncode == MISSING_EXIT and not self.generating:
            missing = missing_results(root, handler)
            raise self.not_generated(f"{len(missing)} test ids (first: {', '.join(missing[:5])})")
        failed, total, by_cat = failed_entries(root, handler) if (root / TEST_IDS_FILE).exists() else ([], 0, {})
        if proc.returncode == INFRA_EXIT:
            raise infra_failure(self.id, failed, total, by_cat, child["max_failed_frac"], not self.generating)
        if proc.returncode != 0:
            raise RuntimeError(f"bfcl-v3 child process exited with {proc.returncode}")
        policy = {**failure_policy(self.id, "entries", len(failed), total, child["max_failed_frac"]),
                  "entries_failed_by_category": {c: f for c, (f, _) in by_cat.items() if f}}
        versions_path = root / HARNESS_VERSIONS
        versions = json.loads(versions_path.read_text()) if versions_path.exists() else {}
        if not self.scoring:
            missing = missing_results(root, handler)
            generated = len(selected_ids(root)) - len(missing)
            log.info("%s: generated %d entries, %d without a result, %d with an infrastructure error",
                     self.id, generated, len(missing), len(failed))
            return {"n": generated, "missing": len(missing), "categories": categories, "handler": handler, **policy,
                    "sampling": child["sampling"] or "checkpoint generation_config", "harness_versions": versions}

        scores = score_dir(root, handler)
        per_category = read_category_scores(scores / registry_name(handler))
        overall = read_overall(scores)
        for cat, s in per_category.items():
            log.info("%s: %s accuracy=%.4f (%d/%d)", self.id, cat, s["accuracy"], s["correct_count"], s["total_count"])
        log.info("%s: overall_accuracy=%.4f", self.id, overall["overall"])
        n = sum(s["total_count"] for s in per_category.values())
        return {
            "value": overall["overall"],
            "n": n,
            "dataset": self.dataset,
            "dataset_revision": self.dataset_revision,
            "harness": f"{HARNESS}=={HARNESS_VERSION}",
            "harness_versions": versions,
            "handler": {"name": handler, "how": HANDLER_DESCRIPTIONS[handler]},
            "sampling": child["sampling"] or "checkpoint generation_config",
            "max_tokens": self.max_tokens_record(handler),
            "categories": categories,
            "partial": self.config.limit is not None,
            **policy,  # entries_failed / entries_total / incomplete: failed entries are left out of the score
            "groups": overall["groups"],
            "pass_at_k": pass_at_k_record(1, overall["overall"], overall["overall"], n, "BFCL v3 Overall Acc, one sample"),
            "per_category": per_category,
        }
