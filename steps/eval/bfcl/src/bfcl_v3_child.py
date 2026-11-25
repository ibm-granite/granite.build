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

- the model, one of two handlers (``--option handler=``):

  ``fc`` (default): BFCL's ``OpenAICompletionsHandler`` in function-calling
  mode, as ``bfcl-v4``: the checkpoint's own chat template and vLLM's tool
  parser format the calls, and only the configured sampling is sent.

  ``legacy-granite``: BFCL prompt mode with the Granite 4.0 prompt format
  rendered by the handler itself and sent as raw text to the completions
  endpoint, the way Granite 4.0 BFCL v3 numbers were produced before this step
  (BFCL's prompt-mode OSS handler pattern; see :func:`granite4_prompt`). The
  model's tool calls are parsed from its ``<tool_call>`` text, not by vLLM.

- ``--limit``: v3 has no partial evaluation (it asserts a result for every
  entry of a category), so a limited score phase filters the category files to
  the selected ids before BFCL's own checkers and aggregation run.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# BFCL model-registry keys, one per handler (results land under result/<key>/).
# No "_" in a key: BFCL maps "_" <-> "/" in model names.
REGISTRY_NAMES = {"fc": "granite-fc", "legacy-granite": "granite-legacy"}
DEFAULT_HANDLER = "fc"
TEST_IDS_FILE = "test_case_ids_to_generate.json"  # BFCL's TEST_IDS_TO_GENERATE_PATH, under BFCL_PROJECT_ROOT
HARNESS_VERSIONS = "harness_versions.json"
MISSING_EXIT = 3
RESULT_PREFIX = "BFCL_v3_"


def registry_name(handler: str) -> str:
    if handler not in REGISTRY_NAMES:
        raise SystemExit(f"bfcl-v3: unknown handler {handler!r} (one of: {', '.join(REGISTRY_NAMES)})")
    return REGISTRY_NAMES[handler]


def result_files(root: Path, handler: str = DEFAULT_HANDLER) -> list[Path]:
    """v3 keeps one flat result file per category under result/<model>/."""
    return sorted((root / "result" / registry_name(handler)).glob(f"{RESULT_PREFIX}*_result.json"))


def score_dir(root: Path, handler: str = DEFAULT_HANDLER) -> Path:
    """BFCL's score dir for one handler: score/<handler>/, holding <model>/ and the
    CSVs. BFCL scores every model under its score dir into one data_overall.csv,
    so handlers that share a run dir must not share a score dir."""
    registry_name(handler)
    return root / "score" / handler


def selected_ids(root: Path) -> list[str]:
    ids = json.loads((root / TEST_IDS_FILE).read_text())
    return [i for cat_ids in ids.values() for i in cat_ids]


def missing_results(root: Path, handler: str = DEFAULT_HANDLER) -> list[str]:
    """Selected test ids with no entry in any result file under ``root``."""
    present = set()
    for path in result_files(root, handler):
        present.update(json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip())
    return [i for i in selected_ids(root) if i not in present]


def select_ids(entries_by_category: dict[str, list[dict]], limit: int | None) -> dict[str, list[str]]:
    """First ``limit`` entries of every category, in the given (harness) order."""
    return {cat: [e["id"] for e in (entries if limit is None else entries[:limit])]
            for cat, entries in entries_by_category.items()}


# -- legacy-granite: the Granite 4.0 prompt format (stdlib only, unit-tested) ---------
#
# Source: the Granite 4.0 chat template as published on Hugging Face,
# ibm-granite/granite-4.0-micro, chat_template.jinja at revision
# b77de16e64f1bdb2ad943402334ec3a76947ead4 (2025-10-01, Apache-2.0; the same
# template ships with every Granite 4.0 model). That revision predates the
# template's default system message (added 2025-10-07), as the earlier runs did.
# Rendered here for BFCL's inputs: tools are BFCL's own function docs, JSON-
# encoded with json.dumps (ASCII-escaped) like BFCL's Granite handlers do, where
# the template's tojson would keep non-ASCII characters; documents are unused.

G4_TOOLS_PREFIX = (
    "You are a helpful assistant with access to the following tools. You may call one or more tools to "
    "assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML "
    "tags:\n<tools>"
)
G4_TOOLS_SUFFIX = (
    "\n</tools>\n\nFor each tool call, return a json object with function name and arguments within "
    '<tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": '
    "<args-json-object>}\n</tool_call>. If a tool does not exist in the provided list of tools, notify the "
    "user that you do not have the ability to fulfill the request."
)
TOOL_CALL_RE = re.compile(r"<tool_call>\n(.*?)\n</tool_call>", re.DOTALL)
MAX_NEW_TOKENS = 4096  # BFCL's prompt-mode cap on requested tokens


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # the template's list-of-parts form: text parts, "\n"-joined
        return "\n".join(e["text"] for e in content if isinstance(e, dict) and e.get("type") == "text")
    return ""


def granite4_prompt(messages: list[dict], functions: list[dict]) -> str:
    """The Granite 4.0 chat template's rendering of ``messages`` with ``functions``
    as tools, plus the generation prompt (see the source note above)."""
    tools = ""
    if functions:
        tools = G4_TOOLS_PREFIX + "".join("\n" + json.dumps(f) for f in functions) + G4_TOOLS_SUFFIX
    if messages and messages[0]["role"] == "system":
        system = _text(messages[0]["content"]) + ("\n\n" + tools if tools else "")
    else:
        system = tools
    out = f"<|start_of_role|>system<|end_of_role|>{system}<|end_of_text|>\n" if system else ""
    for i, m in enumerate(messages):
        role, content = m["role"], _text(m.get("content"))
        if role == "user" or (role == "system" and i > 0):
            out += f"<|start_of_role|>{role}<|end_of_role|>{content}<|end_of_text|>\n"
        elif role == "assistant":
            out += f"<|start_of_role|>assistant<|end_of_role|>{content}"
            for j, call in enumerate(m.get("tool_calls") or []):
                if j or content:
                    out += "\n"
                call = call.get("function", call)
                args = call.get("arguments", {})
                out += '<tool_call>\n{"name": "' + call["name"] + '", "arguments": '
                out += (args if isinstance(args, str) else json.dumps(args)) + "}\n</tool_call>"
            out += "<|end_of_text|>\n"
        elif role == "tool":
            if i == 0 or messages[i - 1]["role"] != "tool":
                out += "<|start_of_role|>user<|end_of_role|>"
            out += f"\n<tool_response>\n{content}\n</tool_response>"
            if i == len(messages) - 1 or messages[i + 1]["role"] != "tool":
                out += "<|end_of_text|>\n"
    return out + "<|start_of_role|>assistant<|end_of_role|>"


def extract_tool_calls(text: str) -> list[dict]:
    """The ``{"name", "arguments"}`` objects in the model's ``<tool_call>`` blocks;
    a block that is not a JSON object is skipped."""
    calls = []
    for block in TOOL_CALL_RE.findall(text or ""):
        try:
            call = json.loads(block)
        except ValueError:
            continue
        if isinstance(call, dict) and isinstance(call.get("name"), str):
            calls.append({"name": call["name"], "arguments": call.get("arguments") or {}})
    return calls


def strip_reasoning(text: str) -> str:
    """The answer after a ``</think>`` block, if the model wrote one."""
    return text.split("</think>")[-1].lstrip("\n") if "</think>" in text else text


def max_new_tokens(context_length: int, prompt_tokens: int) -> int:
    """BFCL prompt mode's request size: what is left of the context, capped."""
    if context_length < prompt_tokens + 2:
        return 1000  # the prompt alone overflows; the server rejects it either way
    return min(MAX_NEW_TOKENS, context_length - prompt_tokens - 2)


class EndpointError(RuntimeError):
    """The serving vLLM failed a /tokenize or /v1/models call for a transient
    reason (5xx, connection, timeout) on every attempt. The message starts with a
    phrase ``bfcl.INFRA_ERROR`` matches, so the entry BFCL records as "Error during
    inference: ..." is retried on resume instead of being scored as a wrong answer."""


class VLLMTokens:
    """Prompt token counts and the context length from the serving vLLM itself
    (its /tokenize and /v1/models), so no tokenizer is loaded here."""

    ATTEMPTS = 3
    RETRY_WAIT = 2.0  # seconds, times the attempt number
    TIMEOUT = 600.0

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        self.model = model
        self._context: int | None = None

    def _json(self, url: str, body: dict | None = None) -> Any:
        """One JSON call, retried on transient failures. A 4xx is not transient (an
        endpoint the server does not have): it is raised at once, and :meth:`probe`
        turns it into a startup failure."""
        import http.client
        import time
        import urllib.error
        import urllib.request

        data = None if body is None else json.dumps(body).encode()
        for attempt in range(1, self.ATTEMPTS + 1):
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code < 500:
                    raise
                err = f"Error code: {e.code} from {url}"
            except (OSError, http.client.HTTPException) as e:  # URLError, TimeoutError, ConnectionError
                reason = getattr(e, "reason", e)
                if isinstance(reason, TimeoutError):
                    err = f"Request timed out: {url}"
                else:
                    err = f"Connection error: {url}: {type(reason).__name__}: {reason}"
            if attempt < self.ATTEMPTS:
                time.sleep(self.RETRY_WAIT * attempt)
        raise EndpointError(f"{err} ({self.ATTEMPTS} attempts)")

    def probe(self) -> None:
        """Fail the job now, not every entry later, if the endpoint cannot serve
        legacy-granite (no /tokenize or no max_model_len: not a vLLM server)."""
        try:
            context, count = self.context_length(), self.count("probe")
        except Exception as e:
            raise SystemExit(f"bfcl-v3: handler=legacy-granite needs vLLM's /tokenize and the max_model_len of "
                             f"/v1/models at {self.base_url}: {type(e).__name__}: {e}") from e
        print(f"bfcl-v3: {self.base_url}: max_model_len {context}, /tokenize ok ({count} tokens)", flush=True)

    def context_length(self) -> int:
        if self._context is None:
            models = self._json(f"{self.base_url}/models")["data"]
            model = next((m for m in models if m["id"] == self.model), models[0])
            self._context = int(model["max_model_len"])
        return self._context

    def count(self, prompt: str) -> int:
        out = self._json(f"{self.root}/tokenize", {"model": self.model, "prompt": prompt, "add_special_tokens": False})
        return int(out["count"])


# -- handlers (bfcl_eval imported here, in the side venv only) ------------------------


def _progress_write(cls, progress: dict[str, int]):
    def write(self, result, result_dir, update_mode=False):
        cls.write(self, result, result_dir, update_mode=update_mode)
        for item in result if isinstance(result, list) else [result]:
            progress["done"] += 1
            print(f"bfcl-v3: generated {progress['done']}/{progress['total']} {item['id']}", flush=True)

    return write


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

    GraniteFCHandler.write = _progress_write(OpenAICompletionsHandler, progress)
    return GraniteFCHandler


def make_legacy_granite_handler(served_model_name: str, sampling: dict[str, Any], progress: dict[str, int],
                                tokens: VLLMTokens | None = None):
    from bfcl_eval.model_handler.base_handler import BaseHandler
    from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
    from bfcl_eval.model_handler.model_style import ModelStyle
    from bfcl_eval.model_handler.utils import convert_to_function_call, func_doc_language_specific_pre_processing
    from openai import OpenAI

    def override(method):
        """Mark an override for BFCL's EnforceOverrides metaclass. (The overrides
        package's decorator finds base classes from the defining frame, which fails
        for a class defined inside a function.)"""
        method.__override__ = True
        return method

    class LegacyGraniteHandler(OSSHandler):
        """BFCL prompt mode, Granite 4.0 prompt format, against the step's vLLM.

        From BFCL's OSS handler: the prompting conversation (turns, execution
        results as tool messages, held-out functions as user messages). Changed:
        it uses the served endpoint (OPENAI_BASE_URL) instead of starting vLLM, so
        BFCL's threaded per-entry path runs it; the prompt is :func:`granite4_prompt`
        with no BFCL system prompt added (the template carries the tool
        instructions); and sampling is sent only when configured."""

        def __init__(self, model_name, temperature) -> None:
            super().__init__(model_name, temperature)
            self.model_style = ModelStyle.OpenAI_Completions  # BFCL's threaded path, no server spin-up
            self.client = OpenAI()  # OPENAI_BASE_URL / OPENAI_API_KEY from the parent
            self.tokens = tokens or VLLMTokens(os.environ.get("OPENAI_BASE_URL", ""), served_model_name)

        @override
        def inference(self, test_entry: dict, include_input_log: bool, exclude_state_log: bool):
            return BaseHandler.inference(self, test_entry, include_input_log, exclude_state_log)

        @override
        def decode_ast(self, result, language="Python"):
            return [{c["name"]: dict(c["arguments"])} for c in extract_tool_calls(result)
                    if isinstance(c["arguments"], dict)]

        @override
        def decode_execute(self, result):
            return convert_to_function_call([{c["name"]: c["arguments"]} for c in extract_tool_calls(result)
                                             if isinstance(c["arguments"], dict)])

        @override
        def _format_prompt(self, messages, function):
            return granite4_prompt(messages, function)

        @override
        def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
            category = test_entry["id"].rsplit("_", 1)[0]
            return {"message": [], "function": func_doc_language_specific_pre_processing(test_entry["function"], category)}

        @override
        def _query_prompting(self, inference_data: dict):
            import time

            prompt = self._format_prompt(inference_data["message"], inference_data["function"])
            inference_data["inference_input_log"] = {"formatted_prompt": prompt}
            kwargs = {"model": served_model_name, "prompt": prompt, **sampling}
            if "max_tokens" not in kwargs:
                kwargs["max_tokens"] = max_new_tokens(self.tokens.context_length(), self.tokens.count(prompt))
            start = time.time()
            response = self.client.completions.create(**kwargs, timeout=72000)
            return response, time.time() - start

        @override
        def _parse_query_response_prompting(self, api_response) -> dict:
            text = strip_reasoning(api_response.choices[0].text)
            calls = extract_tool_calls(text)
            history = ({"role": "assistant", "content": "", "tool_calls": calls} if calls
                       else {"role": "assistant", "content": text})
            return {
                "model_responses": text,
                "model_responses_message_for_chat_history": history,
                "input_token": api_response.usage.prompt_tokens,
                "output_token": api_response.usage.completion_tokens,
            }

        @override
        def _add_assistant_message_prompting(self, inference_data: dict, model_response_data: dict) -> dict:
            inference_data["message"].append(model_response_data["model_responses_message_for_chat_history"])
            return inference_data

    LegacyGraniteHandler.write = _progress_write(OSSHandler, progress)
    return LegacyGraniteHandler


def build_handler(cfg: dict, progress: dict[str, int]):
    """The handler class for ``cfg["handler"]``. legacy-granite needs vLLM's own
    endpoints for every request, so a generating child checks them once, first."""
    if cfg["handler"] == "fc":
        return make_handler(cfg["served_model_name"], cfg["sampling"], progress)
    tokens = VLLMTokens(os.environ.get("OPENAI_BASE_URL", ""), cfg["served_model_name"])
    if cfg["phase"] != "score":
        tokens.probe()
    return make_legacy_granite_handler(cfg["served_model_name"], cfg["sampling"], progress, tokens)


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
    scores = score_dir(Path(os.environ["BFCL_PROJECT_ROOT"]), cfg["handler"])
    scores.mkdir(parents=True, exist_ok=True)
    evaluation_main([registry_name(cfg["handler"])], cfg["categories"], None, str(scores))


def child_main(config_path: str) -> None:
    cfg = json.loads(Path(config_path).read_text())
    cfg.setdefault("handler", DEFAULT_HANDLER)
    registry = registry_name(cfg["handler"])
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
    print(f"bfcl-v3: {progress['total']} entries in {len(ids)} categories, handler {cfg['handler']}", flush=True)
    (root / HARNESS_VERSIONS).write_text(json.dumps(harness_versions(), indent=2))

    fc = cfg["handler"] == "fc"
    MODEL_CONFIG_MAPPING[registry] = ModelConfig(
        model_name=cfg["served_model_name"],
        # No "," in the name: BFCL writes its score CSVs unquoted.
        display_name=f"{cfg['served_model_name']} ({'FC' if fc else 'prompt'} {cfg['handler']})",
        url="",
        org="",
        license="",
        model_handler=build_handler(cfg, progress),
        input_price=None,
        output_price=None,
        is_fc_model=fc,
        # FC: OpenAI tool names cannot hold ".", BFCL sends "_" and maps it back.
        # Prompt mode: the model sees and writes the functions' own names.
        underscore_to_dot=fc,
    )

    if cfg["phase"] == "score":
        if missing := missing_results(root, cfg["handler"]):
            print(f"bfcl-v3: {len(missing)} test ids have no result (run --phase generate)", flush=True)
            sys.exit(MISSING_EXIT)
        evaluate(cfg)
        return

    generation_main(SimpleNamespace(
        model=[registry], test_category=categories,
        temperature=float(cfg["sampling"].get("temperature", 0.0)),  # bookkeeping only; handlers send cfg["sampling"]
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
