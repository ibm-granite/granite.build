"""The BFCL v3 child process (``bfcl-v3``): BFCL's own generation and evaluation.

Runs under the bfcl-v3 side venv's interpreter (``python -I``), never the job
venv: v3 and v4 are the same distribution (``bfcl-eval``, import name
``bfcl_eval``) at different releases, so they cannot share one environment.
This file therefore imports only the standard library at module level and
``bfcl_eval`` inside functions -- the job venv still imports it, because the
benchmark registry imports every module of the package.

The pinned release is ``bfcl-eval==2025.8.6.2``, the last PyPI release before
BFCL V4 (2025.8.25): its data are ``BFCL_v3_*``, its "Overall Acc" is the v3
formula (unweighted mean of the live, non-live and multi-turn overalls).

What this changes in BFCL, and nothing else:

- the model: an ``OpenAICompletionsHandler`` (FC mode) registered under
  ``granite-fc`` that sends the served model's name and only the configured
  sampling (the checkpoint's generation_config applies by default), as
  ``bfcl-v4`` does;
- ``--limit``: v3 has no partial evaluation (it asserts a result for every
  entry of a category), so a limited score phase filters the category files to
  the selected ids before BFCL's own checkers and aggregation run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REGISTRY_NAME = "granite-fc"  # BFCL model-registry key; no "_" (BFCL maps "_" <-> "/")
TEST_IDS_FILE = "test_case_ids_to_generate.json"  # BFCL's TEST_IDS_TO_GENERATE_PATH, under BFCL_PROJECT_ROOT
HARNESS_VERSIONS = "harness_versions.json"
MISSING_EXIT = 3
RESULT_PREFIX = "BFCL_v3_"


def result_files(root: Path) -> list[Path]:
    """v3 keeps one flat result file per category under result/<model>/."""
    return sorted((root / "result" / REGISTRY_NAME).glob(f"{RESULT_PREFIX}*_result.json"))


def selected_ids(root: Path) -> list[str]:
    ids = json.loads((root / TEST_IDS_FILE).read_text())
    return [i for cat_ids in ids.values() for i in cat_ids]


def missing_results(root: Path) -> list[str]:
    """Selected test ids with no entry in any result file under ``root``."""
    present = set()
    for path in result_files(root):
        present.update(json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip())
    return [i for i in selected_ids(root) if i not in present]


def select_ids(entries_by_category: dict[str, list[dict]], limit: int | None) -> dict[str, list[str]]:
    """First ``limit`` entries of every category, in the given (harness) order."""
    return {cat: [e["id"] for e in (entries if limit is None else entries[:limit])]
            for cat, entries in entries_by_category.items()}


def make_handler(served_model_name: str, sampling: dict[str, Any], progress: dict[str, int]):
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler

    class GraniteFCHandler(OpenAICompletionsHandler):
        """BFCL v3's OpenAI-compatible FC handler. v3 builds handlers from the registry
        key (not ModelConfig.model_name), so the served name is sent explicitly."""

        def _query_FC(self, inference_data: dict):
            message = inference_data["message"]
            tools = inference_data["tools"]
            inference_data["inference_input_log"] = {"message": repr(message), "tools": tools}
            kwargs = {"messages": message, "model": served_model_name, **sampling}
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
            for item in result if isinstance(result, list) else [result]:
                progress["done"] += 1
                print(f"bfcl-v3: generated {progress['done']}/{progress['total']} {item['id']}", flush=True)

    return GraniteFCHandler


def restrict_evaluation_to(ids: set[str]) -> None:
    """Partial evaluation for v3: every file BFCL's evaluator loads (prompts,
    possible answers, results) is cut to the selected ids, so its per-category
    length checks hold and its scores cover exactly what was generated."""
    from bfcl_eval.eval_checker import eval_runner

    load = eval_runner.load_file

    def load_selected(*args, **kwargs):
        return [e for e in load(*args, **kwargs) if not isinstance(e, dict) or "id" not in e or e["id"] in ids]

    eval_runner.load_file = load_selected


def harness_versions() -> dict[str, str]:
    from importlib import metadata

    out = {}
    for name in ("bfcl-eval", "openai", "numpy"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out


def evaluate(cfg: dict) -> None:
    from bfcl_eval.eval_checker.eval_runner import main as evaluation_main

    if cfg["limit"] is not None:
        restrict_evaluation_to(set(cfg["selected"]))
    evaluation_main([REGISTRY_NAME], cfg["categories"], None, None)


def child_main(config_path: str) -> None:
    cfg = json.loads(Path(config_path).read_text())
    root = Path(os.environ["BFCL_PROJECT_ROOT"])

    from bfcl_eval._llm_response_generation import main as generation_main
    from bfcl_eval.constants.category_mapping import TEST_FILE_MAPPING
    from bfcl_eval.constants.eval_config import PROMPT_PATH, TEST_IDS_TO_GENERATE_PATH
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
    from bfcl_eval.utils import load_file, parse_test_category_argument, sort_key

    _, categories = parse_test_category_argument(cfg["categories"])
    cfg["categories"] = categories
    entries = {c: sorted(load_file(PROMPT_PATH / TEST_FILE_MAPPING[c]), key=sort_key) for c in categories}
    ids = select_ids(entries, cfg["limit"])
    Path(TEST_IDS_TO_GENERATE_PATH).write_text(json.dumps(ids, indent=2))
    cfg["selected"] = [i for v in ids.values() for i in v]
    progress = {"done": 0, "total": len(cfg["selected"])}
    print(f"bfcl-v3: {progress['total']} entries in {len(ids)} categories", flush=True)
    (root / HARNESS_VERSIONS).write_text(json.dumps(harness_versions(), indent=2))

    MODEL_CONFIG_MAPPING[REGISTRY_NAME] = ModelConfig(
        model_name=cfg["served_model_name"],
        # No "," in the name: BFCL writes its score CSVs unquoted.
        display_name=f"{cfg['served_model_name']} (FC granite)",
        url="",
        org="",
        license="",
        model_handler=make_handler(cfg["served_model_name"], cfg["sampling"], progress),
        input_price=None,
        output_price=None,
        is_fc_model=True,
        underscore_to_dot=True,  # OpenAI-style tool names: "." is sent as "_"
    )

    if cfg["phase"] == "score":
        if missing := missing_results(root):
            print(f"bfcl-v3: {len(missing)} test ids have no result (run --phase generate)", flush=True)
            sys.exit(MISSING_EXIT)
        evaluate(cfg)
        return

    generation_main(SimpleNamespace(
        model=[REGISTRY_NAME], test_category=categories,
        temperature=float(cfg["sampling"].get("temperature", 0.0)),  # bookkeeping only; the handler sends cfg["sampling"]
        include_input_log=cfg["include_input_log"], exclude_state_log=False, num_gpus=1,
        num_threads=cfg["num_threads"], gpu_memory_utilization=0.9, backend="vllm",
        skip_server_setup=True, local_model_path=None, result_dir=None, allow_overwrite=False,
        run_ids=True,
    ))
    if cfg["phase"] == "generate":
        return
    evaluate(cfg)


if __name__ == "__main__":
    child_main(sys.argv[1])
