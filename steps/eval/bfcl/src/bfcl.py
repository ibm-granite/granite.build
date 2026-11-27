"""BFCL v4 (Berkeley Function Calling Leaderboard), Granite metric "overall_accuracy".

The pinned ``bfcl-eval`` package does everything that defines the number: test
data (shipped inside the package, so the package pin is the dataset pin), tool
schemas, the multi-turn / memory / web-search execution environments, the AST
and state checkers, and the leaderboard formula. The headline value is BFCL's
own "Overall Acc" (score/data_overall.csv): a 10/10/10/30/40 weighted mix of
non-live AST, live AST, irrelevance, multi-turn and agentic (web search +
memory) over all 22 scoring categories.

Handler: function-calling (FC) mode through BFCL's ``OpenAICompletionsHandler``
against the vLLM endpoint, i.e. the checkpoint's own chat template and tool
parser do the tool-call formatting, as for every "(FC)" leaderboard entry. (BFCL's
prompt-mode Granite handler hardcodes the Granite 3/4.0 prompt format, which is
not Granite 4.2's.) The thin subclass here only changes what is sent: sampling
parameters are sent only when set as options, so the checkpoint's
generation_config applies by default (BFCL itself sends temperature=0.001).

Web search: BFCL's ``WebSearchAPI.search_engine_query`` calls SerpAPI's DuckDuckGo
engine. Here it is patched, in the BFCL child process only, to call a configurable
google-search MCP server (tool ``google_pse_search``) and map its results onto
the same ``{title, href, body}`` shape; ``fetch_url_content`` is BFCL's own. This
is a documented deviation: search results (and so web-search scores) are not
identical to SerpAPI/DuckDuckGo's.

BFCL runs in a child process (``python -m granite_evals.benchmarks.bfcl``) because
it reads ``BFCL_PROJECT_ROOT`` at import time and keeps state in module globals.
Generation resumes natively (finished test ids are skipped); inference errors
that look like infrastructure failures are dropped on restart so they are retried
(a memory category with one is regenerated whole). Those that remain are never
scored as model answers (they would read as wrong in AST categories and as
correct in irrelevance): failure_policy, as for the other infra-dependent
families, leaves them out of the score and marks the run incomplete, up to
``--option max_failed_frac`` (default 0.05). Above it, generate exits non-zero,
so the target retry resumes it, and the score phase refuses.

Phases: ``--phase generate`` runs BFCL's generation only (vLLM and the search
MCP server; results under ``bfcl/result/``); ``--phase score`` runs only BFCL's
evaluation on those files, which replays multi-turn calls in BFCL's local
simulators and never calls the model or web search. Every selected test id
needs a result: BFCL itself refuses a full evaluation with missing ones (and
silently drops them from a partial one), so a missing one fails the score phase.

Repeats: one generation per test id only (``--repeats 1``). BFCL keeps one
result per test id under one project root and its "Overall Acc" is a weighted
mix of category accuracies from its own score files, so k samples would mean k
BFCL runs and a re-implemented overall formula over them, i.e. a number BFCL
does not define. ``details.pass_at_k`` is the k = 1 record (both values the
headline) to keep results.json's shape.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from granite_evals.mcp_search import SEARCH_TOOL, MCPSearch, parse_search_text  # noqa: F401
from granite_evals.registry import Benchmark, failure_policy, pass_at_k_record, register

log = logging.getLogger(__name__)

HARNESS = "bfcl-eval"
HARNESS_VERSION = "2026.3.23"
REGISTRY_NAME = "granite-fc"  # BFCL model-registry key; no "_" (BFCL maps "_" <-> "/")
DEFAULT_SEARCH_MCP_URL = ""  # set --option search_mcp_url=<MCP search endpoint>
CHILD_CONFIG = "granite_bfcl_config.json"
SEARCH_STATS = "web_search_stats.json"
TEST_IDS_FILE = "test_case_ids_to_generate.json"  # BFCL's TEST_IDS_TO_GENERATE_PATH, under BFCL_PROJECT_ROOT
MISSING_EXIT = 3  # the child's exit code when the score phase finds test ids without a result
INFRA_EXIT = 4  # the child's exit code while too many results hold infrastructure errors
DEFAULT_MAX_FAILED_FRAC = 0.05  # as bird, tau, mcpmark and officeqa

EXCLUDED_CATEGORIES = {"memory_vector"}
"""The reference run excludes this category: granite-4.2's 128k context can't
support it. Filtered out of every default/collection selection (21/22 scoring
categories), still selectable with an explicit --option categories=."""

# An inference error with one of these in its message is an infrastructure
# failure, not a model failure: drop it on restart so the entry is regenerated.
# "vLLM endpoint error" marks bfcl-v3 legacy-granite's own vLLM calls (bfcl_v3_child).
INFRA_ERROR = re.compile(
    r"Connection error|APIConnectionError|APITimeoutError|Request timed out|InternalServerError"
    r"|Error code: 5\d\d|ServiceUnavailable|RemoteProtocolError|vLLM endpoint error",
)


@register
class BFCLv4(Benchmark):
    id = "bfcl-v4"
    metric = "overall_accuracy accuracy"
    default_repeats = 1
    extra = "bfcl"
    harness_packages = (HARNESS,)
    splittable = True
    # The data ships inside the pinned package (bfcl_eval/data), so the dataset
    # is the exact package release, pinned by its wheel's sha256.
    dataset = f"https://pypi.org/project/{HARNESS}/{HARNESS_VERSION}/"
    dataset_revision = "sha256:3bb6dfa5f0c68ad403c9ec50b00db2bb3b4cc9b38ab1ff33f48fe30d853d3a0a"

    def opt(self, key: str, default: Any) -> Any:
        value = self.config.options.get(key)
        return default if value is None else type(default)(value)

    def sampling(self) -> dict[str, Any]:
        """Sampling sent with every request; unset ones come from the checkpoint's
        generation_config (vLLM applies it)."""
        out: dict[str, Any] = {}
        for key, cast in (("temperature", float), ("top_p", float), ("max_tokens", int)):
            if key in self.config.options:
                out[key] = cast(self.config.options[key])
        return out

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        if self.repeats != 1:
            raise SystemExit(
                "bfcl-v4 is a single-run metric (one BFCL result per test id; Overall Acc is BFCL's weighted "
                "mix of category scores, undefined over k samples); use --repeats 1"
            )
        root = self.config.output_dir / "bfcl"
        root.mkdir(parents=True, exist_ok=True)
        categories = [c for c in self.opt("categories", "all_scoring").split(",") if c]
        child = {
            "served_model_name": served_model_name,
            "categories": categories,
            "limit": self.config.limit,
            "num_threads": self.config.workers,
            "sampling": self.sampling(),
            "search_mcp_url": self.opt("search_mcp_url", DEFAULT_SEARCH_MCP_URL),
            "include_input_log": self.opt("include_input_log", "false").lower() == "true",
            "phase": self.phase,
            "max_failed_frac": self.opt("max_failed_frac", DEFAULT_MAX_FAILED_FRAC),
        }
        (root / CHILD_CONFIG).write_text(json.dumps(child, indent=2))
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
        cmd = [sys.executable, "-m", "granite_evals.benchmarks.bfcl", str(root / CHILD_CONFIG)]
        proc = subprocess.run(cmd, env=env, check=False)
        if proc.returncode == MISSING_EXIT and not self.generating:
            missing = missing_results(root)
            raise self.not_generated(f"{len(missing)} test ids (first: {', '.join(missing[:5])})")
        failed, total, dead = failed_entries(root) if (root / TEST_IDS_FILE).exists() else ([], 0, [])
        if proc.returncode == INFRA_EXIT:
            raise infra_failure(self.id, failed, total, dead, child["max_failed_frac"], not self.generating)
        if proc.returncode != 0:
            raise RuntimeError(f"bfcl child process exited with {proc.returncode}")
        policy = failure_policy(self.id, "entries", len(failed), total, child["max_failed_frac"])
        stats_path = root / SEARCH_STATS
        search_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
        if not self.scoring:
            missing = missing_results(root)
            generated = len(scored_ids(root)) - len(missing)
            log.info("%s: generated %d entries, %d without a result, %d with an infrastructure error",
                     self.id, generated, len(missing), len(failed))
            return {"n": generated, "missing": len(missing), "categories": categories, **policy,
                    "sampling": child["sampling"] or "checkpoint generation_config", "web_search": search_stats}

        per_category = read_category_scores(root / "score" / REGISTRY_NAME)
        overall = read_overall(root / "score" / "data_overall.csv")
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
            "handler": "OpenAICompletionsHandler (FC) via vLLM tool parser",
            "sampling": child["sampling"] or "checkpoint generation_config",
            "categories": categories,
            "partial": self.config.limit is not None,
            **policy,  # entries_failed / entries_total / incomplete: failed entries are left out of the score
            "groups": overall["groups"],
            "pass_at_k": pass_at_k_record(1, overall["overall"], overall["overall"], n, "BFCL Overall Acc, one sample"),
            "per_category": per_category,
            "web_search_backend": {
                "type": f"mcp:{SEARCH_TOOL}",
                "url": child["search_mcp_url"],
                "replaces": "SerpAPI duckduckgo engine (BFCL default)",
                **search_stats,
            },
        }


# -- result parsing (parent side, no harness import) ------------------------


def _pct(text: str) -> float | None:
    text = text.strip()
    if not text or text == "N/A":
        return None
    return float(text.rstrip("%")) / 100


OVERALL_GROUPS = {
    "non_live": "Non-Live AST Acc",
    "live": "Live Acc",
    "multi_turn": "Multi Turn Acc",
    "web_search": "Web Search Acc",
    "memory": "Memory Acc",
    "relevance": "Relevance Detection",
    "irrelevance": "Irrelevance Detection",
}


def read_overall(path: Path) -> dict[str, Any]:
    """BFCL's own "Overall Acc" and group scores from data_overall.csv, as fractions."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise RuntimeError(f"{path}: expected one model row, found {len(rows)}")
    row = rows[0]
    if None in row:  # BFCL writes unquoted: a "," in a cell shifts every later column
        raise RuntimeError(f"{path}: more cells than columns (a ',' in the model name?)")
    return {
        "overall": _pct(row["Overall Acc"]) or 0.0,
        "groups": {k: _pct(row[col]) for k, col in OVERALL_GROUPS.items()},
    }


def read_category_scores(score_dir: Path) -> dict[str, dict[str, Any]]:
    """Per-category header ({accuracy, correct_count, total_count}) of each score file."""
    out = {}
    for path in sorted(score_dir.rglob("BFCL_v4_*_score.json")):
        cat = path.name[len("BFCL_v4_"):-len("_score.json")]
        with path.open() as f:
            header = json.loads(f.readline())
        out[cat] = {k: header[k] for k in ("accuracy", "correct_count", "total_count")}
    return out


# -- test selection (--limit) -----------------------------------------------


def select_ids(entries_by_category: dict[str, list[dict]], limit: int | None) -> dict[str, list[str]]:
    """First ``limit`` scored entries of every category, in the given (harness)
    order. Memory entries bring the prerequisite conversations they depend on."""
    out = {}
    for cat, entries in entries_by_category.items():
        scored = [e for e in entries if "prereq" not in e["id"]]
        chosen = scored if limit is None else scored[:limit]
        ids = {e["id"] for e in chosen}
        for e in chosen:
            ids.update(e.get("depends_on") or [])
        out[cat] = [e["id"] for e in entries if e["id"] in ids]
    return out


def scored_ids(root: Path) -> list[str]:
    """The selected test ids BFCL scores (memory prereq conversations are not scored)."""
    ids = json.loads((root / TEST_IDS_FILE).read_text())
    return [i for cat_ids in ids.values() for i in cat_ids if "prereq" not in i]


def missing_results(root: Path) -> list[str]:
    """Selected scored test ids with no entry in any result file under ``root``."""
    present = set()
    for path in (root / "result" / REGISTRY_NAME).rglob("BFCL_v4_*_result.json"):
        present.update(json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip())
    return [i for i in scored_ids(root) if i not in present]


def is_infra_error(rec: dict) -> bool:
    """A result BFCL recorded for an infrastructure failure, not a model answer."""
    res = rec.get("result")
    return isinstance(res, str) and res.startswith("Error during inference") and bool(INFRA_ERROR.search(res))


def _records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def infra_error_ids(root: Path) -> list[str]:
    """Selected test ids (memory prereqs included: their state feeds the scored
    entries) whose result is an infrastructure error."""
    selected = {i for cat_ids in json.loads((root / TEST_IDS_FILE).read_text()).values() for i in cat_ids}
    return [rec["id"] for path in sorted((root / "result" / REGISTRY_NAME).rglob("BFCL_v4_*_result.json"))
            for rec in _records(path) if rec["id"] in selected and is_infra_error(rec)]


# -- infrastructure errors: granite_evals' failure_policy ----------------------------
#
# An entry whose result is an infrastructure error is no model answer: it is left out
# of the score (scored, it would read as wrong in AST categories and as correct in
# irrelevance), and the run is marked incomplete, as failure_policy does for every
# infra-dependent family. Resume prunes and retries such entries. Generate fails
# (INFRA_EXIT, so the target retry resumes it), and score refuses, only while the
# failed fraction is above max_failed_frac, or when a category has nothing else left.


def failed_entries(root: Path) -> tuple[list[str], int, list[str]]:
    """(failed scored ids, scored ids in all, categories with every scored id failed).
    A scored entry fails on its own infrastructure error, or, in a memory category,
    on one in any of the category's prereq conversations (they set its state)."""
    ids = json.loads((root / TEST_IDS_FILE).read_text())
    bad = set(infra_error_ids(root))
    failed, dead = [], []
    for cat, cat_ids in ids.items():
        scored = [i for i in cat_ids if "prereq" not in i]
        prereq_failed = any(i in bad for i in cat_ids if "prereq" in i)
        cat_failed = [i for i in scored if prereq_failed or i in bad]
        failed += cat_failed
        if scored and len(cat_failed) == len(scored):
            dead.append(cat)
    return failed, sum(1 for v in ids.values() for i in v if "prereq" not in i), dead


def over_failure_limit(failed: int, total: int, max_frac: float) -> bool:
    """failure_policy's rule (granite_evals.registry), for a child to choose its exit."""
    return bool(failed) and (not total or failed / total > max_frac)


def infra_failure(benchmark: str, failed: list[str], total: int, dead: list[str], max_frac: float,
                  score_only: bool) -> SystemExit:
    """The parent's error when the child stopped on infrastructure errors (INFRA_EXIT)."""
    try:
        failure_policy(benchmark, "entries", len(failed), total, max_frac)
        why = f"{benchmark}: every entry of {', '.join(dead)} hit an infrastructure error, so it has none to score"
    except SystemExit as e:
        why = str(e)
    first = f" (first: {', '.join(failed[:5])})"
    if score_only:
        how = ("This build's generate target already succeeded, so a retry or `gb build restart` of the build "
               "reuses it and does not retry them. Start a new build (gb build start) with the same RUN_NAME and "
               "OUTPUT_ROOT: its generate resumes in this output dir and retries only these entries.")
    else:
        how = "A retry of this generate target (the recipes' MAX_RETRIES) resumes in this output dir and retries them."
    return SystemExit(f"{'refusing to score' if score_only else 'generation incomplete'}: {why}{first}. {how} "
                      f"To score without them instead, raise max_failed_frac (now {max_frac:g}).")


def check_failures(benchmark: str, root: Path, max_frac: float) -> set[str]:
    """Child side: exit INFRA_EXIT above the limit (or with a category left empty);
    otherwise the failed scored ids, which the score leaves out."""
    failed, total, dead = failed_entries(root)
    if over_failure_limit(len(failed), total, max_frac) or dead:
        print(f"{benchmark}: {len(failed)}/{total} entries hit infrastructure errors (first: {', '.join(failed[:5])})"
              f"{'; no entry left in ' + ', '.join(dead) if dead else ''}; above max_failed_frac={max_frac:g}",
              flush=True)
        sys.exit(INFRA_EXIT)
    if failed:
        print(f"{benchmark}: {len(failed)}/{total} entries hit infrastructure errors; left out of the score "
              f"(max_failed_frac={max_frac:g})", flush=True)
    return set(failed)


def prune_infra_errors(result_file: Path) -> int:
    """Drop results that failed for infrastructure reasons, so resume retries them."""
    if not result_file.exists():
        return 0
    keep, dropped = [], 0
    for line in result_file.read_text().splitlines():
        if not line.strip():
            continue
        if is_infra_error(json.loads(line)):
            dropped += 1
            continue
        keep.append(line)
    if dropped:
        result_file.write_text("".join(line + "\n" for line in keep))
    return dropped


# -- web search through the MCP server (client in granite_evals.mcp_search) --


def make_search_engine_query(search: MCPSearch):
    """A replacement for WebSearchAPI.search_engine_query with BFCL's signature
    and return shape ({title, href, body}, body dropped without snippets)."""

    def search_engine_query(self, keywords: str, max_results: int | None = 10, region: str | None = "wt-wt"):
        # region: the Google PSE tool has no region parameter; ignored.
        n = max(1, min(int(max_results or 10), 10))
        try:
            hits = search.search(keywords, n)
        except Exception as e:
            print(f"[WebSearchAPI/MCP] {e}", flush=True)
            return {"error": str(e)}
        if not hits:
            return {"error": "Failed to retrieve the search results from server. Please try again later."}
        if self.show_snippet:
            return hits[:n]
        return [{"title": h["title"], "href": h["href"]} for h in hits[:n]]

    return search_engine_query


# -- the BFCL child process ----------------------------------------------------


def _make_handler(sampling: dict[str, Any], progress: dict[str, int]):
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

    class GraniteFCHandler(OpenAICompletionsHandler):
        """BFCL's OpenAI-compatible FC handler; sends sampling only when configured
        and keeps the (vLLM) reasoning trace out of the chat history but in the log."""

        def _query_FC(self, inference_data: dict):
            message = inference_data["message"]
            tools = inference_data["tools"]
            inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
            kwargs = {"messages": message, "model": self.model_name, **sampling}
            if tools:
                kwargs["tools"] = tools
            return self.generate_with_backoff(**kwargs)

        def _parse_query_response_FC(self, api_response):
            data = super()._parse_query_response_FC(api_response)
            msg = api_response.choices[0].message
            history: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                history["tool_calls"] = [
                    {"id": tc.id, "type": tc.type,
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            data["model_responses_message_for_chat_history"] = history
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
            if reasoning:
                data["reasoning_content"] = reasoning
            return data

        def write(self, result, result_dir, update_mode=False):
            super().write(result, result_dir, update_mode=update_mode)
            items = result if isinstance(result, list) else [result]
            for item in items:
                progress["done"] += 1
                print(f"bfcl-v4: generated {progress['done']}/{progress['total']} {item['id']}", flush=True)

    return GraniteFCHandler


def child_main(config_path: str) -> None:
    from types import SimpleNamespace

    cfg = json.loads(Path(config_path).read_text())
    root = Path(os.environ["BFCL_PROJECT_ROOT"])

    from bfcl_eval._llm_response_generation import main as generation_main
    from bfcl_eval.constants.eval_config import TEST_IDS_TO_GENERATE_PATH
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main
    from bfcl_eval.eval_checker.multi_turn_eval.func_source_code import web_search
    from bfcl_eval.utils import (
        get_directory_structure_by_category,
        get_file_name_by_category,
        is_memory,
        load_dataset_entry,
        parse_test_category_argument,
        sort_key,
    )

    categories = [c for c in parse_test_category_argument(cfg["categories"]) if c not in EXCLUDED_CATEGORIES]
    # Collection names (e.g. "all_scoring") re-expand to the full set inside
    # generation_main/evaluation_main; pin the already-filtered category list
    # explicitly so every downstream BFCL call sees the same 21/22 categories.
    cfg["categories"] = categories
    entries = {c: sorted(load_dataset_entry(c), key=sort_key) for c in categories}
    ids = select_ids(entries, cfg["limit"])
    Path(TEST_IDS_TO_GENERATE_PATH).write_text(json.dumps(ids, indent=2))
    progress = {"done": 0, "total": sum(len(v) for v in ids.values())}
    print(f"bfcl-v4: {progress['total']} entries in {len(ids)} categories", flush=True)

    MODEL_CONFIG_MAPPING[REGISTRY_NAME] = ModelConfig(
        model_name=cfg["served_model_name"],
        # No "," in the name: BFCL writes its score CSVs unquoted, so one would
        # shift every column after "Model" in data_overall.csv.
        display_name=f"{cfg['served_model_name']} (FC granite)",
        url="",
        org="",
        license="",
        model_handler=_make_handler(cfg["sampling"], progress),
        is_fc_model=True,
        underscore_to_dot=True,  # OpenAI-style tool names: "." is sent as "_"
    )

    if cfg["phase"] == "score":
        # Score only what was generated; a missing result is no model failure to hide.
        if missing := missing_results(root):
            print(f"bfcl-v4: {len(missing)} test ids have no result (run --phase generate)", flush=True)
            sys.exit(MISSING_EXIT)
        failed = check_failures("bfcl-v4", root, cfg.get("max_failed_frac", DEFAULT_MAX_FAILED_FRAC))
        evaluate_without(root, cfg, failed, evaluation_main)
        return

    search = MCPSearch(cfg["search_mcp_url"])
    web_search.WebSearchAPI.search_engine_query = make_search_engine_query(search)

    result_root = root / "result" / REGISTRY_NAME
    for cat in categories:
        d = result_root / get_directory_structure_by_category(cat)
        f = d / get_file_name_by_category(cat, is_result_file=True)
        if is_memory(cat):
            # Memory prereq chains carry state, so never regenerate part of one: a
            # category with an infrastructure error is regenerated whole: its result
            # and prereq files, and its snapshot where BFCL's memory backend keeps it
            # (memory_api_metaclass: result/<model>/<category dir>/memory_snapshot;
            # BFCL's own overwrite path, result/<model>/memory_snapshot/<cat>, is
            # never written).
            files = [f, d / get_file_name_by_category(f"{cat}_prereq", is_result_file=True)]
            if n := sum(is_infra_error(rec) for p in files for rec in _records(p)):
                for p in files:
                    p.unlink(missing_ok=True)
                shutil.rmtree(d / "memory_snapshot", ignore_errors=True)
                print(f"bfcl-v4: {cat}: regenerating the category, {n} entries hit infrastructure errors", flush=True)
            continue
        if n := prune_infra_errors(f):
            print(f"bfcl-v4: {cat}: retrying {n} entries that hit infrastructure errors", flush=True)

    generation_main(SimpleNamespace(
        model=[REGISTRY_NAME], test_category=cfg["categories"], temperature=sampling_temperature(cfg),
        include_input_log=cfg["include_input_log"], exclude_state_log=False, num_gpus=1,
        num_threads=cfg["num_threads"], gpu_memory_utilization=0.9, backend="vllm",
        skip_server_setup=True, local_model_path=None, result_dir=None, allow_overwrite=False,
        run_ids=True, enable_lora=False, max_lora_rank=None, lora_modules=None,
    ))
    (root / SEARCH_STATS).write_text(json.dumps(search.stats, indent=2))
    # Above the limit, fail the job, so the target retry resumes and retries them.
    failed = check_failures("bfcl-v4", root, cfg.get("max_failed_frac", DEFAULT_MAX_FAILED_FRAC))
    if cfg["phase"] == "generate":
        return
    evaluate_without(root, cfg, failed, evaluation_main)


def evaluate_without(root: Path, cfg: dict, failed: set[str], evaluation_main) -> None:
    """BFCL's evaluation with the failed entries' results left out. Only the result
    files are filtered (BFCL aligns prompts and ground truth by position, and some
    ground-truth ids differ from the prompt ids); partial evaluation then scores the
    entries present, so a failed entry counts nowhere, irrelevance included."""
    if failed:
        from bfcl_eval.eval_checker import eval_runner

        load, results = eval_runner.load_file, (root / "result").resolve()

        def load_without_failed(path, *args, **kwargs):
            entries = load(path, *args, **kwargs)
            if Path(path).resolve().is_relative_to(results):
                entries = [e for e in entries if not (isinstance(e, dict) and e.get("id") in failed)]
            return entries

        eval_runner.load_file = load_without_failed
    evaluation_main([REGISTRY_NAME], cfg["categories"], None, None, cfg["limit"] is not None or bool(failed))


def sampling_temperature(cfg: dict) -> float:
    # Only used for BFCL's own bookkeeping; GraniteFCHandler sends cfg["sampling"].
    return float(cfg["sampling"].get("temperature", 0.0))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    child_main(sys.argv[1])
