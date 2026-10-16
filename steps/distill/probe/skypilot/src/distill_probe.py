#!/usr/bin/env python3
"""Three cheap questions that gate the whole granite4-350m distillation family.

Each is answered inside the distillation image in about two minutes, and none trains or
produces anything: the answer is a `PROBE VERDICT` line in the log.

  load-student   Does granite-4.0-350m load at all in this image? It is granitemoehybrid,
                 which makes transformers lazily fetch kernels-community/mamba-ssm, whose
                 CACHED build imports GreedySearchDecoderOnlyOutput and therefore cannot
                 import under transformers 5.8.0 (measured, build 96868683). Every model
                 the reference distillation runs use is dense, which is why they never hit
                 this. If it fails, the fix is a prefetched HF_HOME -- the steps expose
                 workload.hf_home -- not a different student.

  tokenizer      Does the checkpoint's declared tokenizer_class change how text segments?
                 Both granite 4.x models declare GPT2Tokenizer over a trained
                 Sequence[Split(regex), ByteLevel] pre_tokenizer, and align's premise is
                 that AutoTokenizer honours the class and discards the Split (26.1 vs 3.29
                 PPL/token, measured upstream). It decides whether the comparison against
                 the recorded after-SFT eval row needs a tokenizer-control column: our
                 export pins PreTrainedTokenizerFast, the recorded row was produced from a
                 directory declaring GPT2Tokenizer.

  tokenizer-fit  Which segmentation do THIS checkpoint's weights prefer? `tokenizer` shows
                 the two paths disagree; this says whether pinning is a FIX or a
                 REGRESSION. Pretraining used tokenizer.json (the pinned segmentation);
                 the SFT stage most likely went through AutoTokenizer (the as-is one). One
                 number, two GPU-minutes instead of 27 eval jobs.

Was three inline `command` targets in recipes/granite4-350m/lsf/distill-probe. torch and
transformers are imported inside each probe, so the module imports without them.

Usage: distill_probe.py <probe> --checkpoint <hf_dir> [--corpus <jsonl>] [--rows N]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

PROBES = ("load-student", "tokenizer", "tokenizer-fit")

# Only the small files. The weights are irrelevant to tokenization.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.jinja",
)

PINNED_CLASS = "PreTrainedTokenizerFast"

# Chosen to hit the Split regex the override would discard: contractions, a leading
# space, digits, a newline, and the granite role marker, which must stay a single token
# either way. A set of plain ASCII words would report SAME whether or not the override
# fired.
SPLIT_PROBES = (
    "It's 42 degrees.",
    " leading space",
    "a\nb",
    "don't 1234 tokenize",
    "<|start_of_role|>assistant<|end_of_role|>",
)


def stage_tokenizer_copies(src: Path, work: Path) -> tuple[Path, Path]:
    """Copy the checkpoint's tokenizer files into work/asis and work/pinned, and rewrite
    tokenizer_class in the pinned copy only.

    The real checkpoint is never written to: writing tokenizer_class into it would
    silently change what the recorded eval row refers to.
    """
    asis, pinned = work / "asis", work / "pinned"
    asis.mkdir(parents=True)
    pinned.mkdir(parents=True)
    for name in TOKENIZER_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, asis / name)
            shutil.copy2(src / name, pinned / name)
        else:
            print(f"PROBE absent {name}", flush=True)
    cfg_path = pinned / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    print("PROBE declared_class", cfg.get("tokenizer_class"), flush=True)
    cfg["tokenizer_class"] = PINNED_CLASS
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"PROBE rewrote tokenizer_class -> {PINNED_CLASS}", flush=True)
    return asis, pinned


def load_student(checkpoint: str) -> int:
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    print(
        "PROBE transformers",
        transformers.__version__,
        "torch",
        torch.__version__,
        flush=True,
    )
    cfg = AutoConfig.from_pretrained(checkpoint)
    # logits_scaling gates the fused JSD kernel and tie_word_embeddings a ZeRO-3
    # checkpoint hazard, so they are read once here rather than guessed later.
    print(
        "PROBE model_type",
        cfg.model_type,
        "logits_scaling",
        getattr(cfg, "logits_scaling", None),
        "tie_word_embeddings",
        getattr(cfg, "tie_word_embeddings", None),
        flush=True,
    )
    # The eager fallback runs after a failure rather than instead of the first attempt:
    # "loads but FA2 does not resolve" is a different diagnosis from "does not load", and
    # they have different fixes. FA2 is probed because render_sft_config.py selects it
    # for granitemoehybrid and sft.py aborts every rank on a mismatch, so the SFT-control
    # arm depends on a kernel distill-gold never asks for.
    ok = []
    for attn in ("flash_attention_2", "eager"):
        try:
            m = AutoModelForCausalLM.from_pretrained(
                checkpoint, attn_implementation=attn, dtype=torch.bfloat16
            )
            n = sum(p.numel() for p in m.parameters())
            print(f"PROBE LOAD_OK attn={attn} params={n}", flush=True)
            ok.append(attn)
            del m
        except Exception:  # pylint: disable=broad-except
            print(f"PROBE LOAD_FAIL attn={attn}", flush=True)
            traceback.print_exc()
    print(f"PROBE VERDICT loaded={','.join(ok) or 'NONE'}", flush=True)
    print("PROBE DONE load-student", flush=True)
    return 0


def tokenizer(checkpoint: str) -> int:
    from transformers import AutoTokenizer

    work = Path(tempfile.mkdtemp())
    print(f"PROBE scratch {work}", flush=True)
    asis_dir, pinned_dir = stage_tokenizer_copies(Path(checkpoint), work)
    asis = AutoTokenizer.from_pretrained(str(asis_dir))
    pinned = AutoTokenizer.from_pretrained(str(pinned_dir))
    for name, tk in (("as-is", asis), ("pinned", pinned)):
        print(
            f"PROBE {name} class={type(tk).__name__} "
            f"pre_tokenizer={tk.backend_tokenizer.pre_tokenizer}",
            flush=True,
        )
    differing = 0
    for s in SPLIT_PROBES:
        a = asis(s)["input_ids"]
        b = pinned(s)["input_ids"]
        same = a == b
        differing += not same
        print(f"PROBE {'SAME' if same else 'DIFF'} {s!r}", flush=True)
        print(f"      as-is  {a}", flush=True)
        print(f"      pinned {b}", flush=True)
    print(f"PROBE VERDICT differing={differing}/{len(SPLIT_PROBES)}", flush=True)
    print(
        "PROBE DONE tokenizer -- differing>0 means the recorded eval row was "
        "produced with a mis-segmenting tokenizer, and the comparison needs a "
        "tokenizer-control column",
        flush=True,
    )
    return 0


def read_conversations(corpus: str, rows: int) -> list[list[dict]]:
    """The first `rows` records that carry a non-empty `messages` list."""
    out: list[list[dict]] = []
    with open(corpus) as fh:
        for line in fh:
            if len(out) >= rows:
                break
            r = json.loads(line)
            if r.get("messages"):
                out.append(r["messages"])
    return out


def nll_per_byte(total_nll: float, total_bytes: int) -> float:
    """NLL PER BYTE, not per token. Two tokenizations of the same text produce different
    token counts, so perplexity-per-token is not comparable between them -- the coarser
    segmentation wins by construction. Bytes are the invariant."""
    return total_nll / max(total_bytes, 1)


def tokenizer_fit(checkpoint: str, corpus: str, rows: int) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    work = Path(tempfile.mkdtemp())
    stage_tokenizer_copies(Path(checkpoint), work)
    convs = read_conversations(corpus, rows)
    print(f"PROBE fit rows={len(convs)} from {corpus}", flush=True)

    model = (
        AutoModelForCausalLM.from_pretrained(
            checkpoint, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        .cuda()
        .eval()
    )

    def render(tk, msgs):
        try:
            return tk.apply_chat_template(msgs, tokenize=False)
        except Exception:  # pylint: disable=broad-except
            return "\n".join(
                m.get("content", "") for m in msgs if isinstance(m.get("content"), str)
            )

    per_byte = {}
    for name in ("asis", "pinned"):
        tk = AutoTokenizer.from_pretrained(str(work / name))
        tot_nll, tot_bytes, tot_tokens, skipped = 0.0, 0, 0, 0
        for msgs in convs:
            text = render(tk, msgs)
            nbytes = len(text.encode("utf-8"))
            ids = tk(text, return_tensors="pt", truncation=True, max_length=4096)[
                "input_ids"
            ].cuda()
            if ids.shape[1] < 2:
                skipped += 1
                continue
            with torch.no_grad():
                out = model(ids, labels=ids)
            # HF returns the MEAN nll over the (n-1) predicted positions, so summing raw
            # losses would weight a short row equally with a long one.
            n_pred = ids.shape[1] - 1
            tot_nll += out.loss.float().item() * n_pred
            tot_bytes += nbytes
            tot_tokens += ids.shape[1]
        per_byte[name] = nll_per_byte(tot_nll, tot_bytes)
        print(
            f"PROBE fit {name}: class={type(tk).__name__} tokens={tot_tokens} "
            f"bytes={tot_bytes} skipped={skipped} "
            f"nll_per_byte={per_byte[name]:.5f} "
            f"nll_per_token={tot_nll / max(tot_tokens - len(convs), 1):.5f}",
            flush=True,
        )

    a, b = per_byte["asis"], per_byte["pinned"]
    better = "pinned" if b < a else "asis"
    print(
        f"PROBE VERDICT nll_per_byte asis={a:.5f} pinned={b:.5f} "
        f"prefers={better} ratio={max(a, b) / max(min(a, b), 1e-9):.3f}",
        flush=True,
    )
    print(
        "PROBE DONE tokenizer-fit -- prefers=pinned means the SFT stage "
        "introduced the mismatch and pinning restores consistency with "
        "pretraining; prefers=asis means pinning is a regression and the "
        "export must NOT pin for the comparison to hold",
        flush=True,
    )
    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("probe", choices=PROBES)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--corpus", default="")
    p.add_argument("--rows", type=int, default=200)
    args = p.parse_args(argv)
    if not args.checkpoint:
        p.error("--checkpoint is empty")
    if args.probe == "tokenizer-fit" and not args.corpus:
        p.error("tokenizer-fit needs --corpus")
    if args.rows < 1:
        p.error("--rows must be >= 1")
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.probe == "load-student":
        return load_student(args.checkpoint)
    if args.probe == "tokenizer":
        return tokenizer(args.checkpoint)
    return tokenizer_fit(args.checkpoint, args.corpus, args.rows)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
