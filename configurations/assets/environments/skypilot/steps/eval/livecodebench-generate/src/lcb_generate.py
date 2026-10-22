#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate LiveCodeBench completions for any chat model with vLLM.

LiveCodeBench's own runner (``lcb_runner.runner.main``) only accepts models listed
in its ``lm_styles.py``, and that listing also chooses the prompt format. This
script uses LiveCodeBench as a library instead, so any model works:

* the problems come from LiveCodeBench's loader, read from a local copy of the
  dataset (``lcb_dataset``) and narrowed to a release and date window;
* each prompt is LiveCodeBench's generic chat prompt (its system message and
  question template), formatted with **the model's own chat template**;
* generation runs in vLLM;
* code is pulled out of each output with LiveCodeBench's ``extract_code``, after
  dropping any reasoning that ends in ``</think>``. The scorer executes
  ``code_list`` exactly as given, so raw output (code inside a fence) would score 0.

Output directory:

* ``completions.json``: ``[{"question_id", "code_list"}]``, the scorer's input;
* ``raw_outputs.json``: ``[{"question_id", "output_list"}]``, untouched generations;
* ``generation_meta.json``: the settings, versions and a hash of the problem set,
  which ``eval/livecodebench-score`` checks before scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import lcb_dataset

COMPLETIONS = "completions.json"
RAW = "raw_outputs.json"
META = "generation_meta.json"
THINK_END = "</think>"


def question_ids_sha256(ids: Sequence[str]) -> str:
    """Hash a problem set independently of order (the score step recomputes it)."""
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def prompt_opens_reasoning(prompt: str) -> bool:
    """Whether the rendered prompt itself opens a reasoning block.

    Some chat templates (granite-4.2's, with thinking on) end the generation
    prompt with ``<think>\\n``. vLLM returns only the generated text, so such an
    output contains ``</think>`` but never the opening tag. A template with
    thinking off ends with ``<think></think>`` instead, which opens nothing.
    """
    tail = prompt.rstrip()
    return tail.endswith("<think>")


def strip_reasoning(text: str, reasoning_open: bool = False) -> str:
    """Return the answer part of an output: what follows the last ``</think>``.

    A reasoning model writes its scratch work first; any code fence inside it is a
    draft, not the answer.

    :param text: the generated text.
    :param reasoning_open: the prompt already opened ``<think>`` (see
        :func:`prompt_opens_reasoning`). Then an output with no ``</think>`` was
        cut off mid-thought, and has no answer: ``""``.
    """
    if THINK_END in text:
        return text.rsplit(THINK_END, 1)[1]
    if reasoning_open or "<think>" in text:
        return ""
    return text


def extract(text: str, reasoning_open: bool = False) -> str:
    """The code LiveCodeBench would score for ``text``, or ``""`` if there is none."""
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code

    # OpenAIChat takes the fence path: the text between the last two ``` lines.
    return extract_code(strip_reasoning(text, reasoning_open), LMStyle.OpenAIChat)


def build_messages(problem: Any) -> List[Dict[str, str]]:
    """LiveCodeBench's generic chat prompt for one problem."""
    from lcb_runner.prompts.code_generation import (
        PromptConstants,
        get_generic_question_template_answer,
    )

    return [
        {"role": "system", "content": PromptConstants.SYSTEM_MESSAGE_GENERIC},
        {"role": "user", "content": get_generic_question_template_answer(problem)},
    ]


def render_prompts(
    tokenizer: Any, problems: Sequence[Any], enable_thinking: Optional[bool]
) -> List[str]:
    """Format every problem with the model's chat template.

    ``enable_thinking`` is passed only when set: templates without the switch
    ignore unknown keywords, but leaving it out keeps the template's own default.
    """
    extra = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    return [
        tokenizer.apply_chat_template(
            build_messages(p), tokenize=False, add_generation_prompt=True, **extra
        )
        for p in problems
    ]


def assemble(
    problems: Sequence[Any],
    outputs: Sequence[Sequence[str]],
    reasoning_open: bool = False,
) -> tuple[list, list, Dict[str, Any]]:
    """Pair each problem with its outputs; count the samples with no extractable code."""
    if len(outputs) != len(problems):
        raise RuntimeError(
            f"got outputs for {len(outputs)} of {len(problems)} problems"
        )
    completions, raw = [], []
    samples = empty = 0
    for p, outs in zip(problems, outputs):
        codes = [extract(o, reasoning_open) for o in outs]
        samples += len(codes)
        empty += sum(1 for c in codes if not c.strip())
        completions.append({"question_id": p.question_id, "code_list": codes})
        raw.append({"question_id": p.question_id, "output_list": list(outs)})
    stats = {
        "problems": len(problems),
        "samples": samples,
        "empty_extractions": empty,
        "empty_fraction": (empty / samples) if samples else 0.0,
    }
    return completions, raw, stats


def parse_bool(value: str) -> Optional[bool]:
    v = value.strip().lower()
    if v == "":
        return None
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise ValueError(f"enable_thinking must be true, false or empty, not {value!r}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--release-version", required=True)
    ap.add_argument("--start-date", default="")
    ap.add_argument("--end-date", default="")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--enable-thinking", default="")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument(
        "--gpu-memory-utilization", type=float, default=0.0, help="0 = vLLM's default"
    )
    ap.add_argument("--max-model-len", type=int, default=0, help="0 = the model's own")
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--lcb-commit", default="")
    return ap.parse_args(argv)


def generate(args: argparse.Namespace, prompts: List[str]) -> List[List[str]]:
    """Run vLLM over the prompts; one list of ``n`` outputs per prompt."""
    from vllm import LLM, SamplingParams

    kwargs: Dict[str, Any] = {
        "model": args.model_dir,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.gpu_memory_utilization > 0:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.max_model_len > 0:
        kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**kwargs)
    params = SamplingParams(
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    results = llm.generate(prompts, params)
    return [[o.text for o in r.outputs] for r in results]


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.n < 1:
        raise ValueError("n must be at least 1")
    if args.max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    thinking = parse_bool(args.enable_thinking)
    start, end = args.start_date or None, args.end_date or None

    lcb_dataset.install(args.dataset_dir)
    from lcb_runner.benchmarks.code_generation import load_code_generation_dataset

    problems = load_code_generation_dataset(
        args.release_version, start_date=start, end_date=end
    )
    if not problems:
        raise ValueError(
            f"no problems in {args.release_version} for window {start}..{end}"
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=args.trust_remote_code
    )
    prompts = render_prompts(tokenizer, problems, thinking)
    # One template renders every prompt the same way, so the first one decides.
    reasoning_open = prompt_opens_reasoning(prompts[0])
    outputs = generate(args, prompts)
    completions, raw, stats = assemble(problems, outputs, reasoning_open)
    stats["reasoning_open"] = reasoning_open

    import vllm  # for the version only; generate() already imported it

    meta = {
        "release_version": args.release_version,
        "start_date": start,
        "end_date": end,
        "model": args.model_dir,
        "n": args.n,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "enable_thinking": thinking,
        "lcb_commit": args.lcb_commit,
        "vllm_version": vllm.__version__,
        "prompt": "lcb generic chat prompt + model chat template",
        "question_ids_sha256": question_ids_sha256([p.question_id for p in problems]),
        **stats,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / COMPLETIONS).write_text(json.dumps(completions) + "\n")
    (out / RAW).write_text(json.dumps(raw) + "\n")
    (out / META).write_text(json.dumps(meta, indent=2) + "\n")
    print("eval/livecodebench-generate: " + json.dumps(stats))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ValueError as e:
        print(f"eval/livecodebench-generate: ERROR {e}", file=sys.stderr)
        sys.exit(2)
