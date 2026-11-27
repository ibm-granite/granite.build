"""The bfcl step's FC output budget, decided before ``granite-evals run`` serves the model.

Run by the step template (``python -m granite_evals.benchmarks.bfcl_serving MODEL_PATH
BASE_URL OPTIONS``). It prints one ``fc_budget=<cap>:<source>`` line, which the template
passes as an ``--option`` so results.json records the decision, then the extra ``vllm
serve`` arguments, one per line (none, or the budget).

The budget: FC sends no max_tokens, so a checkpoint that loops runs to the end of its
context, into the client's 600 s timeout, an infrastructure error instead of its (wrong)
answer. Served with ``--override-generation-config '{"max_new_tokens":16384}'``, vLLM 0.27.1
caps a request that sends no max_tokens at min(16384, the context left) (get_max_tokens).
A served default, not a request field: vLLM rejects a request whose prompt plus its
max_tokens exceeds the context (renderers/params.py), so a client-side default would turn
long multi-turn prompts into errors.

It is applied only when nothing else sets the output length:
- not with ``base_url`` (an endpoint the step does not start);
- not with ``handler=legacy-granite``, which sends its own min(4096, ctx - prompt - 2);
- not with an explicit ``max_tokens`` option, which is sent with every request;
- not when the checkpoint's generation_config.json sets ``max_new_tokens`` (vLLM applies
  it; the override would replace it, in either direction) or ``max_length`` (vLLM ignores
  it, but the checkpoint asks for a total length the budget would not honour);
- not when that file cannot be read (recorded as unknown, never guessed).

Options are parsed as the CLI parses them (granite_evals.cli): split on any whitespace,
the last value of a key wins.

Why 16,384: the largest per-request FC output in the four bfcl-v3 calibration runs
(granite-4.0-micro and -h-small, BFCL v3) was 8,697 tokens (micro, multi_turn_long_context_132).
There is no such evidence for bfcl-v4 or for other checkpoints.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

FC_MAX_TOKENS = 16384


def last_option(options: str, key: str) -> str | None:
    """The value of ``key`` in a ``key=value`` options string: the last one wins, and any
    whitespace separates, as the CLI's --option list is built and read."""
    value = None
    for word in options.split():
        k, sep, v = word.partition("=")
        if sep and k == key:
            value = v
    return value


def read_generation_config(model_path: str) -> dict:
    """The checkpoint's generation_config.json ({} when it has none). A local directory is
    read directly; anything else is a Hugging Face repo id, read through the hub cache (as
    vLLM itself resolves it, honouring HF_HOME / HF_HUB_OFFLINE)."""
    local = Path(model_path)
    if local.is_dir():
        f = local / "generation_config.json"
        return json.loads(f.read_text()) if f.is_file() else {}
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return json.loads(Path(hf_hub_download(model_path, "generation_config.json")).read_text())
    except EntryNotFoundError:
        return {}


def plan(model_path: str, base_url: str, options: str) -> tuple[str, list[str]]:
    """(the fc_budget record "<cap>:<source>", the vllm serve arguments)."""
    if base_url:
        return "none:base_url", []
    if (last_option(options, "handler") or "fc") == "legacy-granite":
        return "none:legacy-granite", []
    if (explicit := last_option(options, "max_tokens")) is not None:
        return f"{explicit}:max_tokens-option", []
    try:
        config = read_generation_config(model_path)
    except Exception as e:  # noqa: BLE001  never guess: no budget, recorded as unknown
        return f"unknown:generation_config-unreadable-{type(e).__name__}", []
    if config.get("max_new_tokens") is not None:
        return f"{config['max_new_tokens']}:checkpoint-max_new_tokens", []
    if config.get("max_length") is not None:
        return "context-left:checkpoint-max_length", []
    override = json.dumps({"max_new_tokens": FC_MAX_TOKENS}, separators=(",", ":"))
    return f"{FC_MAX_TOKENS}:step-default", [f"--override-generation-config={override}"]


def describe(record: str | None) -> dict:
    """results.json's max_tokens record from the fc_budget option."""
    if not record:
        return {"cap": "unknown", "source": "not recorded (granite-evals run outside the bfcl step)"}
    cap, _, source = record.partition(":")
    notes = {
        "step-default": "the bfcl step's served default (--override-generation-config max_new_tokens)",
        "base_url": "none: an existing endpoint (base_url), not served by the step",
        "legacy-granite": "none served: legacy-granite sends its own with every request",
        "max_tokens-option": "the max_tokens option, sent with every request",
        "checkpoint-max_new_tokens": "the checkpoint's generation_config max_new_tokens (vLLM applies it)",
        "checkpoint-max_length": "none served: the checkpoint's generation_config sets max_length (vLLM ignores it)",
    }
    return {"cap": int(cap) if cap.isdigit() else cap, "source": notes.get(source, source)}


def main(argv: list[str]) -> int:
    model_path, base_url, options = (argv + ["", "", ""])[:3]
    record, args = plan(model_path, base_url, options)
    print(f"fc_budget={record}")
    for a in args:
        print(a)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
