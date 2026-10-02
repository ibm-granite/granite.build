#
# PORTED, not authored here. Upstream source of truth:
#   repo   github.ibm.com/Herbert-Woisetschlaeger/gb-steps-collection-post-training
#   path   steps/distill-hf-export/src/export_hf_model.py
#   commit 70c1550a171aa8e09a9ad9047a5bf763c39e8579
#
# Verbatim apart from `black`/`isort` reflow, which CI requires repo-wide, and ONE
# behaviour divergence recorded below. Keep it otherwise verbatim so re-syncing upstream
# stays a three-way merge.
#
# DIVERGENCE: `normalise_tokenizer_json` + its wiring in `export`, which clear the
# trainer's live truncation/padding state out of the published `tokenizer.json`. Upstream
# copies that file verbatim, so every model it publishes carries the trainer's last
# truncation budget (`max_length: 466` on build df8512e0's stage-1 export). See
# STRIP_TOKENIZER_JSON_RUNTIME_KEYS for why that is not cosmetic. This belongs upstream;
# until it lands there, the divergence lives here and the three-way merge has to keep it.
#
# DIVERGENCE: `tokenizer_pin_problems` + its call in `verify`, which assert that the
# published tokenizer, loaded the way a consumer loads it, segments exactly as its own
# tokenizer.json does. See TOKENIZER_PIN_PROBES. Same status as the one above.
#
# It imports gb_steps_post_training.distillation at MODULE scope, which is delivered at
# RUN time from the checkout named by code_config (see step-template.yaml). That is why
# the tests for this file are gated on GB_DISTILL_CODE_DIR — see test/conftest.py.
#
"""Turn a GOLD training output directory into a publishable HF model directory.

This step is a SELECTOR/PRUNER, not a format converter. That is a verified property, not
an assumption: the DeepSpeed config used by `distill-gold-train` sets
`zero3_save_16bit_model: true`, so `Trainer.save_model` already writes HF-native
`model.safetensors` at the checkpoint root. Confirmed directly on job 1136274's
checkpoint-25 -- a 704786224-byte `model.safetensors` (350M params x 2 bytes) sitting
beside the DeepSpeed `global_step25/` shard tree.

So there are exactly four jobs here, and none of them is a weight conversion:

  1. SELECT   which checkpoint-N is the release. The trainer does not decide this.
  2. PRUNE    the resumable-run state (DeepSpeed shards, optimizer, scheduler, RNG). This
              is also nearly all of the size difference -- ZeRO optimizer state is several
              times the weights.
  3. NORMALISE the DEFAULTS the trainer saved for its own use, of which there are three
              and all of them are about what a consumer gets when it asks for nothing:
                - `gold.py:345` sets `padding_side="left"` because the trainer generates.
                  That is right for generation and wrong as a published default, since a
                  right-padding consumer trusting `tokenizer_config.json` will silently pad
                  on the wrong side.
                - `chat_template.jinja:13` defaults `enable_thinking` to True, so
                  `apply_chat_template(..., add_generation_prompt=True)` hands out the
                  REASONING prompt. For a student distilled on a corpus with no think traces
                  that is the prefix the weights never saw. `--chat-template-thinking default-off` flips
                  it; see THINKING_POLICIES, which also records why this step refuses to
                  delete the empty `<think></think>` blocks themselves.
                - `tokenizer.json` carries the trainer's live `truncation` state, because
                  saving a tokenizer serialises what it was last configured to do. A
                  consumer reading that file through the raw `tokenizers` backend inherits
                  a truncation budget belonging to one training batch. See
                  STRIP_TOKENIZER_JSON_RUNTIME_KEYS.
              All are OPT-IN or reported, never silent: this step's contract is that the
              export manifest says exactly which defaults it changed.
  4. ASSERT   the pruned directory actually loads.

Deliberately NOT a job here: grafting in a tokenizer. `gold.py:454` passes
`processing_class=tokenizer` and loads it from the *student* path (`gold.py:341`), so each
checkpoint is self-describing and carries the retagged tokenizer the corpus was built
with. The step confirms that rather than repairing it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

# The shared step-completion contract, vendored into this step's image (see Dockerfile). A hard
# import and not a try/except: a missing gate would silently turn resume off for this step, and a
# step that re-exports a 700 MB directory every time it is asked is the cheaper failure to notice
# than one that skips work it should have done.
from gb_steps_post_training.distillation import export_state, step_state

# Files that make up a published model. An explicit KEEP list rather than a DROP list:
# with a DROP list, anything a future transformers/TRL release starts writing would be
# published by default, and this directory is the thing users download.
KEEP_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "model.safetensors.index.json",  # present only when the weights are sharded
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
)

# Sharded weights: model-00001-of-0000N.safetensors. Matched by prefix+suffix rather than
# listed, since N is not known ahead of time.
KEEP_SHARD_PREFIX = "model-"
KEEP_SHARD_SUFFIX = ".safetensors"

# Resumable-run state. Named explicitly so the step's own report can say what it dropped
# and why, rather than "everything not in KEEP_FILES".
PRUNE_KNOWN = (
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "trainer_state.json",
    "training_args.bin",
    "zero_to_fp32.py",
    "latest",
)
PRUNE_DIR_PREFIXES = ("global_step",)

# Load-time kwargs that transformers persists into tokenizer_config.json. Neither is
# tokenizer configuration; both were observed in job 1136274's checkpoint-25
# (`local_files_only` and `is_local`), and `is_local` was already present in the student
# source. Publishing a model whose tokenizer config pins local_files_only=true is wrong.
STRIP_TOKENIZER_KEYS = ("local_files_only", "is_local")

# A transformers-5-only `tokenizer_class`, and the portable spelling of the same
# tokenizer.
#
# transformers 5 reorganised the tokenization stack and records its fast-tokenizer
# backend as `tokenizer_class: "TokenizersBackend"`. A consumer on transformers 4
# cannot resolve that name and raises
#   ValueError: Tokenizer class TokenizersBackend does not exist or is not currently
#   imported
# before reading a single token. The distillation steps run transformers 5.8.0, so
# every model this step publishes carries the pin: bfcl-eval, which ships a
# transformers 4 image, died on it at base_oss_handler.py:109 in build 30a99c4b.
#
# PreTrainedTokenizerFast exists in BOTH generations and loads `tokenizer.json`
# directly, so this rewrite changes the spelling and not the tokenizer. Only this one
# name is rewritten: guessing at an unfamiliar class would change which tokenizer a
# consumer instantiates, which is the kind of silent substitution this step refuses
# everywhere else.
TOKENIZER_CLASS_REWRITES = {"TokenizersBackend": "PreTrainedTokenizerFast"}

# RUNTIME state that `save_pretrained` serialises into `tokenizer.json` itself.
#
# A trained tokenizer.json carries `truncation` and `padding` as explicit nulls. But the
# trainer's in-process tokenizer had truncation switched on for its own batching, and
# saving the tokenizer writes that live state into the file -- so the published model
# arrives with a budget nobody asked for. Observed on build df8512e0's stage-1 export:
#   "truncation": {"direction": "Right", "max_length": 466, ...}
# where the aligned student's was null. 466 is one training batch's longest sequence; it
# describes the trainer's last call, not the model.
#
# It is not cosmetic. transformers' PreTrainedTokenizerFast calls `no_truncation()` when a
# caller does not ask for truncation, so a transformers consumer is unaffected -- but a
# consumer that loads the file through the raw backend
# (`tokenizers.Tokenizer.from_file`, which is also how several serving stacks read it)
# inherits the 466-token limit and silently truncates every longer prompt.
#
# NOT in this list, deliberately: `post_processor`. The trainer writes an identity
# `TemplateProcessing` where align had null, and that one is genuinely inert -- it inserts
# no special tokens. A post_processor decides which specials get added at encode time, so
# rewriting one would be the kind of silent substitution this step refuses everywhere
# else. It is left exactly as the checkpoint had it.
STRIP_TOKENIZER_JSON_RUNTIME_KEYS = ("truncation", "padding")

# THE ROPE RELOCATION, and the portability rule that generalises it.
#
# transformers 5 moved RoPE settings out of top-level config keys into a nested
# `rope_parameters` dict. The value is not corrupted by that move, it is RELOCATED -- which
# is why a key-by-key diff of a 4.57.6 config against a 5.8.0 one does not show it:
# `rope_theta` is not DIFFERENT, it is MISSING ON ONE SIDE.
#
#   key               align/retagged_student (4.57.6)  export-N (5.8.0)
#   rope_theta        10000000                         <ABSENT>
#   rope_parameters   <ABSENT>                         {rope_theta: 10000000, ...}
#
# Every consumer on the transformers-4 schema -- vLLM 0.11.0, which the whole eval stack
# serves through, and bfcl-eval -- reads the top-level field, does not find it, and falls
# back to the class default. Measured directly under transformers 4.55.4:
#
#   align/retagged_student        rope_theta -> 10000000    <- healthy
#   export-N as written          rope_theta ->     10000.0  <- 1000x too small
#   export-N with the hoist      rope_theta -> 10000000    <- repaired
#
# A RoPE base 1000x too small destroys long-range position information while leaving local
# coherence intact, so the published models wrote correct code and correct arithmetic and
# never emitted a `<tool_call>`: BFCL 0.0000 on 400/400 across every arm in this epic, from
# a single training step onward, while `eval-transfer` -- which loads through transformers
# and therefore reads `rope_parameters` correctly -- reported healthy divergences digit for
# digit. Two harnesses, one checkpoint, opposite verdicts. Restoring the one key returned
# BFCL 0.8025 against a 0.8000-0.8125 baseline.
#
# HOIST, NOT MOVE. The nested dict stays exactly as the checkpoint had it, so
# transformers-5 consumers are unaffected; the top-level key is added for the older schema.
# Verified inert on the newer one: transformers 5.8.0's config does not define a
# `rope_theta` attribute at all, so the added key changes nothing it resolves. (It is also
# dropped again if a transformers-5 process re-serialises the config -- this step publishes
# a final artifact, so nothing downstream does that, but a future step that re-saves a
# config through transformers 5 would silently undo the hoist.)
#
# WHY THE RULE IS NOT `rope_theta`. A fix naming one field protects one field, and the next
# transformers reorganisation lands this same bug silently somewhere else. The rule is
# structural instead: any NUMERIC scalar inside a nested `*_parameters` dict must also
# appear top-level with an equal value. `assert_config_portable` enforces it on what is
# actually written, which is what makes the guard outlive this particular field.
NESTED_PARAM_SUFFIX = "_parameters"

# Non-numeric leaves are deliberately NOT mirrored. `rope_parameters.rope_type: "default"`
# is not a transformers-4 top-level field; inventing one could change which RoPE
# implementation a v4 consumer selects, which is the class of silent substitution this step
# refuses everywhere else. Numbers are safe because a number means the same thing to both
# generations -- it is the LOOKUP that moved, not the semantics.

# The trainer's gradient-checkpointing flag, which has no business on a published inference
# artifact. vLLM manages its own KV cache so this appears inert in the eval path, but a
# transformers consumer calling `generate()` on a config that says use_cache=false gets no
# KV cache and a quadratic decode.
CONFIG_USE_CACHE_KEY = "use_cache"

# transformers 5's JSON spelling for a float that has no JSON literal:
#   "time_step_limit": [0.0, {"__float__": "Infinity"}]
# There is no strict-JSON spelling of infinity, so this cannot be REPAIRED into a portable
# number -- transformers 4.55.4 loads it and hands back a `dict` where a float belongs, and
# any consumer doing arithmetic on `config.time_step_limit[1]` gets a TypeError. The
# portable move is therefore to DROP the key and let the consumer's own class default
# apply, which is exactly what the healthy 4.57.6-authored `align/retagged_student` config
# does -- it carries none of the three `time_step_*` keys at all.
#
# Safe here because the value IS the class default on both generations, verified:
# transformers 5.8.0 resolves an absent `time_step_limit` to `(0.0, inf)`, the same pair
# the sentinel spells. `verify()` re-checks that per export rather than trusting this note.
#
# The neighbours are left alone on purpose. `time_step_min: 0.001` / `time_step_max: 0.1`
# are plain portable floats, and they are also both exactly the 5.8.0 class defaults, so
# there is nothing to fix. They are not inconsistent with a `time_step_limit` of [0.0, inf]
# either, which is what it looks like at a glance: the limit clamps dt at runtime while
# min/max bound its initialisation, so the stock default set is all three together.
_FLOAT_SENTINEL_KEY = "__float__"

PADDING_SIDES = ("right", "left", "keep")

# The published chat template's THINKING POLICY.
#
# `chat_template.jinja:13` reads
#   {%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}
# so a consumer that calls apply_chat_template(..., add_generation_prompt=True) and says
# nothing about thinking gets the REASONING generation prompt:
#   enable_thinking=True   ->  '<|im_start|>assistant\n<think>\n'      (template line 191)
#   enable_thinking=False  ->  '<|im_start|>assistant\n<think></think>' (template line 193)
#
# For a student distilled on a corpus with no think traces those are not equivalent, and the
# DEFAULT is the wrong one. Every labelled assistant span in that corpus begins with the
# literal `<think></think>` -- template line 90 prepends it to any assistant turn carrying no
# think markers, which job 1141677 measured as 120/120 spans. So the model was trained to emit
# `</think>` immediately after `<think>`, and was never once conditioned on `<think>\n`: the
# newline between the two markers does not occur anywhere in its training data. `default-off`
# flips line 13 so the published default is the shape the weights actually saw.
#
# WHY THERE IS NO `strip-injection` POLICY. The obvious reading of "strip the empty think
# block" is to delete line 90's injection, and that would be a defect. `<think></think>` is
# written at FIVE sites in this template, and they are not one feature:
#
#   line  90  INJECTOR     prepends it to a plain assistant turn with no think markers.
#   line 113  CANONICALISER tool-call turn, truncated history: rebuilds the turn with an empty
#                          block in place of the real trace.
#   line 119  INJECTOR     tool-call turn whose content is not a string: emits a bare block.
#   line 147  CANONICALISER plain turn, truncated history: same collapse as 113.
#   line 193  PROMPT       the enable_thinking=False generation prompt.
#
# `truncate_history_thinking` also defaults True (line 18), so 113 and 147 fire on their own
# for every assistant turn before the last user turn. Deleting only line 90 therefore renders
# ONE conversation two ways -- older turns keep the empty block via 113/147, the newest turn
# loses it, and the generation prompt still emits it at 193 -- which is a prefix shape no
# training example had. The empty block is load-bearing structure in this template, not
# decoration, so this step will not remove it. It changes which DEFAULT the consumer lands on
# and records that it did.
THINKING_POLICIES = ("keep", "default-off")

# NOT to be confused with distill-corpus-prep's `think_policy` (keep | strip | require),
# which is a decision about the TRAINING DATA -- whether reasoning traces are kept in the
# supervised targets at all. This one edits a DEFAULT in the published chat template and
# touches no data. They are related in one direction only: a corpus prepared with
# think_policy=strip produces a student that should be published with
# chat_template_thinking=default-off. The flag is spelled in full for that reason -- two
# neighbouring step keys called `thinking` and `think_policy` is how a recipe author sets
# the wrong one.

# Matched after stripping surrounding whitespace, and required to occur exactly once. A
# substring match on `enable_thinking` would also hit line 190's `{%- if enable_thinking %}`,
# and this repo has already been bitten by identifying a thing by substring.
_THINKING_LINE_ON = "{%- set enable_thinking = enable_thinking if enable_thinking is defined else True %}"
_THINKING_LINE_OFF = "{%- set enable_thinking = enable_thinking if enable_thinking is defined else False %}"

# Invariant guard for the five sites above: `default-off` must not change how many times the
# empty block is written. If a future edit to this function starts deleting them, this is what
# says so, at export time, instead of the published model doing it quietly.
_EMPTY_THINK = "<think></think>"


# Strings the published tokenizer must segment exactly as its own tokenizer.json does.
#
# granite 4.x checkpoints declare `tokenizer_class: GPT2Tokenizer` over a trained
# Sequence[Split, ByteLevel] pre_tokenizer. AutoTokenizer honours the declaration, builds
# the GPT2 converter and discards the trained Split, so the model is evaluated on a
# segmentation it was never trained on -- while still reporting is_fast=True. align pins
# `tokenizer_class` to fix that and TOKENIZER_CLASS_REWRITES keeps the pin portable, but a
# class NAME is only a label. These probes are the behavioural check: the ids AutoTokenizer
# produces against the ids tokenizer.json produces with no transformers in the path. Code,
# runs of whitespace, role markers and non-Latin text are where the two segmentations part.
TOKENIZER_PIN_PROBES = (
    "def fibonacci(n: int) -> int:\n    return n if n < 2 else fibonacci(n-1)+fibonacci(n-2)",
    "The quick brown fox jumps over 42 lazy dogs.",
    "  leading and   internal   whitespace\t\tand a tab",
    "<|start_of_role|>user<|end_of_role|>Hello<|end_of_text|>",
    "Ceci n'est pas une pipe \u2014 na\u00efve caf\u00e9, \u65e5\u672c\u8a9e\u30c6\u30ad\u30b9\u30c8, \U0001f680",
    "1234567890 0x1F 3.14159e-7",
)


class ExportError(Exception):
    """Raised for any condition under which this step must not publish."""


def select_checkpoint(train_output_dir: Path, which: str = "latest") -> Path:
    """Pick the checkpoint to publish.

    `which` is "latest", a bare "checkpoint-N", or a path. "latest" means highest step
    number, NOT newest mtime: a resumed run rewrites older checkpoints' mtimes, so mtime
    ordering can select a checkpoint that is not the furthest along.
    """
    if which and which not in ("latest",):
        cand = Path(which)
        if not cand.is_absolute():
            cand = train_output_dir / which
        if not cand.is_dir():
            raise ExportError(f"requested checkpoint does not exist: {cand}")
        return cand

    if not train_output_dir.is_dir():
        raise ExportError(f"train_output_dir is not a directory: {train_output_dir}")

    found = [p for p in train_output_dir.glob("checkpoint-*") if p.is_dir()]
    if not found:
        raise ExportError(
            f"no checkpoint-* directory under {train_output_dir}. Nothing to export -- "
            "either training never reached save_steps, or output_dir is wrong."
        )

    def step_of(p: Path) -> int:
        tail = p.name.split("-", 1)[1]
        if not tail.isdigit():
            raise ExportError(
                f"cannot order checkpoint directory {p.name!r}: expected checkpoint-<int>."
            )
        return int(tail)

    return max(found, key=step_of)


def _is_kept(name: str) -> bool:
    if name in KEEP_FILES:
        return True
    # Sharded weights, e.g. model-00002-of-00003.safetensors.
    return name.startswith(KEEP_SHARD_PREFIX) and name.endswith(KEEP_SHARD_SUFFIX)


def classify(checkpoint: Path) -> dict[str, list[str]]:
    """Split a checkpoint's entries into keep / prune / unknown.

    "unknown" exists so that a file neither list anticipated is SURFACED rather than
    silently published or silently dropped. Both silent outcomes are wrong for a directory
    users download.
    """
    keep: list[str] = []
    prune: list[str] = []
    unknown: list[str] = []
    for entry in sorted(checkpoint.iterdir(), key=lambda p: p.name):
        name = entry.name
        if entry.is_dir():
            if any(name.startswith(pfx) for pfx in PRUNE_DIR_PREFIXES):
                prune.append(name)
            else:
                unknown.append(name)
        elif _is_kept(name):
            keep.append(name)
        elif name in PRUNE_KNOWN:
            prune.append(name)
        else:
            unknown.append(name)
    return {"keep": keep, "prune": prune, "unknown": unknown}


def _is_number(value: Any) -> bool:
    # bool is an int subclass in Python and `use_cache: true` must not be mistaken for a
    # number to mirror.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _nested_param_dicts(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        k: v
        for k, v in cfg.items()
        if k.endswith(NESTED_PARAM_SUFFIX) and isinstance(v, dict)
    }


def _find_float_sentinel(value: Any) -> bool:
    """True if `value` contains a transformers-5 non-finite-float sentinel anywhere."""
    if isinstance(value, dict):
        if _FLOAT_SENTINEL_KEY in value:
            return True
        return any(_find_float_sentinel(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_find_float_sentinel(v) for v in value)
    return False


def published_config_needs_repair(dest: Path) -> str | None:
    """Return why `dest`'s already-published config.json fails today's contract, or None.

    THE RESUME GATE'S BLIND SPOT. `export_state.expectation()` keys a skip on the source
    checkpoint and the policy flags. It has no component for this step's own normalisation
    contract, so when that contract gains a rule -- as it did when the RoPE hoist was added
    -- every export already on disk still hashes equal, the gate says SKIP, and the broken
    config survives its own fix. That is not hypothetical: it is how all 38 mis-served
    export directories in this epic would have stayed mis-served after the bug was found.

    Adding a version component to the expectation would be the tidier fix, but that key is
    built in `gb_steps_post_training`, which this repo does not own and must not vendor
    (see conftest.py). So the check lives here instead and is narrower on purpose: a skip
    is valid only if what is already published still passes the guard it would be published
    under today.

    Silent when there is nothing published -- no dest, or no config.json in it. The resume
    gate is already going to run the export in that case and a second complaint adds noise.
    Unreadable JSON reads as "needs repair", never as a crash inside the gate.
    """
    cfg_path = dest / "config.json"
    if not cfg_path.is_file():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError) as exc:
        return f"{cfg_path} could not be read as JSON ({exc})"
    if not isinstance(cfg, dict):
        return f"{cfg_path} is not a JSON object"
    try:
        assert_config_portable(cfg)
    except ExportError as exc:
        return str(exc)
    return None


def normalise_model_config(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return (normalised config.json, list of human-readable changes).

    Three jobs, all of them about what a consumer on the OTHER transformers generation
    resolves; see the NESTED_PARAM_SUFFIX block for the measurements behind each.

      1. MIRROR every numeric scalar from a nested `*_parameters` dict to a top-level key.
      2. RESTORE use_cache.
      3. DROP any key carrying a non-finite-float sentinel.

    Refuses rather than repairs when a top-level key already exists and DISAGREES with its
    nested twin: that is two live values for one field, and picking one silently is how a
    model gets served at a RoPE base its weights never saw.
    """
    out = dict(raw)
    changes: list[str] = []

    for parent, nested in sorted(_nested_param_dicts(raw).items()):
        for key, value in sorted(nested.items()):
            if not _is_number(value):
                continue
            if key in out:
                if out[key] != value:
                    raise ExportError(
                        f"config.json: {parent}.{key}={value!r} disagrees with top-level "
                        f"{key}={out[key]!r}. This step will not guess which one a "
                        "consumer should get -- they resolve on different transformers "
                        "generations, so shipping both means shipping two models."
                    )
                continue
            out[key] = value
            changes.append(
                f"hoisted {parent}.{key} -> top-level {key}={value!r} (kept nested; "
                "transformers 5 reads the dict, every transformers-4-schema consumer "
                "reads the top-level key and otherwise silently takes the class default)"
            )

    if out.get(CONFIG_USE_CACHE_KEY) is False:
        out[CONFIG_USE_CACHE_KEY] = True
        changes.append(
            f"{CONFIG_USE_CACHE_KEY} False -> True (the trainer disables the KV cache for "
            "gradient checkpointing; that is training state, not a published default)"
        )

    for key in [k for k, v in sorted(out.items()) if _find_float_sentinel(v)]:
        stale = out.pop(key)
        changes.append(
            f"dropped {key}={stale!r} (a transformers-5 {_FLOAT_SENTINEL_KEY!r} sentinel; "
            "no other JSON consumer parses it into a float, and infinity has no strict-"
            "JSON spelling, so the portable form is absence + the consumer's class default)"
        )

    return out, changes


def diff_resolved_numeric(source: dict[str, Any], dest: dict[str, Any]) -> list[str]:
    """Return a human-readable line per numeric field whose RESOLVED value moved.

    The second half of the config guard, and the one that keeps `normalise_model_config`
    honest about the keys it DROPS. Dropping `time_step_limit` is only safe because an
    absent key resolves to the same `(0.0, inf)` the sentinel spelled; this compares what
    the two directories actually resolve, so a dropped key that was NOT its class default
    fails the export instead of quietly changing the published model.

    Only fields the SOURCE had, and only numbers. This step ADDS top-level keys on purpose
    -- that is the hoist -- so a key present only in `dest` is not drift. Non-numerics are
    out of scope: `dtype` and `architectures` are compared by the loads in `verify()`
    itself, and a string-valued field that legitimately differs (transformers_version) is
    not a statement about the weights.

    Takes plain dicts rather than config objects so it is unit-testable on a box with no
    transformers, which is the property this whole module is arranged around.
    """
    drift: list[str] = []
    for key, want in sorted(source.items()):
        if not _is_number(want):
            continue
        if key not in dest:
            drift.append(
                f"{key}: source resolves {want!r}, exported config has no such field"
            )
            continue
        got = dest[key]
        if _is_number(got) and got == want:
            continue
        drift.append(f"{key}: source resolves {want!r}, export resolves {got!r}")
    return drift


def assert_config_portable(cfg: dict[str, Any]) -> None:
    """Raise unless `cfg` resolves the same way on both transformers generations.

    THE GUARD THAT OUTLIVES THIS BUG. A resolved-value comparison between the written
    config and the source checkpoint cannot catch the RoPE relocation at all: both files
    are read by the SAME transformers inside this container, 5.8.0 reads the nested dict,
    and both sides resolve 10000000 whether or not the top-level key was ever written. The
    bug is a disagreement between transformers GENERATIONS, and the only side of it
    available here is the schema -- which is checkable in pure JSON, with no transformers 4
    to import.

    So the invariant is structural, not a list of field names: every numeric scalar in a
    nested `*_parameters` dict is mirrored top-level, and no key carries a sentinel a
    non-transformers-5 parser cannot read.
    """
    for parent, nested in sorted(_nested_param_dicts(cfg).items()):
        for key, value in sorted(nested.items()):
            if not _is_number(value):
                continue
            if key not in cfg:
                raise ExportError(
                    f"config.json is not schema-portable: {parent}.{key}={value!r} has no "
                    f"top-level {key}. A transformers-4-schema consumer (vLLM, bfcl-eval) "
                    f"reads the top-level key, will not find it, and will silently serve "
                    f"the class default instead of {value!r}."
                )
            if cfg[key] != value:
                raise ExportError(
                    f"config.json is not schema-portable: {parent}.{key}={value!r} but "
                    f"top-level {key}={cfg[key]!r}. The two generations would serve "
                    "different models from one directory."
                )

    for key, value in sorted(cfg.items()):
        if _find_float_sentinel(value):
            raise ExportError(
                f"config.json is not schema-portable: {key}={value!r} carries a "
                f"{_FLOAT_SENTINEL_KEY!r} sentinel, which only transformers 5 parses as a "
                "float. Drop the key and let the consumer's class default apply."
            )


def normalise_tokenizer_config(
    raw: dict[str, Any], *, padding_side: str = "right"
) -> tuple[dict[str, Any], list[str]]:
    """Return (normalised config, list of human-readable changes).

    Returns changes rather than logging them so the caller can record exactly what it did
    in the export manifest -- "record that it did" is part of this step's contract.
    """
    if padding_side not in PADDING_SIDES:
        raise ExportError(
            f"padding_side={padding_side!r} is not one of {list(PADDING_SIDES)}."
        )

    out = dict(raw)
    changes: list[str] = []

    for key in STRIP_TOKENIZER_KEYS:
        if key in out:
            del out[key]
            changes.append(f"stripped {key} (a load-time kwarg, not tokenizer config)")

    current_class = out.get("tokenizer_class")
    portable = TOKENIZER_CLASS_REWRITES.get(current_class) if current_class else None
    if portable is not None:
        out["tokenizer_class"] = portable
        changes.append(
            f"tokenizer_class {current_class!r} -> {portable!r} "
            "(the transformers 5 backend name no consumer on transformers 4 can resolve)"
        )

    if padding_side != "keep":
        before = out.get("padding_side")
        if before != padding_side:
            out["padding_side"] = padding_side
            changes.append(
                f"padding_side {before!r} -> {padding_side!r} "
                "(the trainer sets 'left' because it generates; see gold.py:345)"
            )

    return out, changes


def normalise_tokenizer_json(
    raw: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Return (normalised tokenizer.json, list of human-readable changes).

    Clears the trainer's live truncation/padding state and touches nothing else -- the
    vocabulary, merges, pre_tokenizer, added_tokens and post_processor are the tokenizer
    and are none of this step's business. Set to None rather than deleted, because that
    is how a clean tokenizer.json spells "no truncation configured".
    """
    out = dict(raw)
    changes: list[str] = []

    for key in STRIP_TOKENIZER_JSON_RUNTIME_KEYS:
        stale = out.get(key)
        if stale is None:
            continue
        out[key] = None
        detail = ""
        if isinstance(stale, dict) and "max_length" in stale:
            detail = f": max_length={stale['max_length']}"
        changes.append(
            f"cleared {key}{detail} (the trainer's live tokenizer state, serialised "
            "into tokenizer.json; a raw tokenizers consumer would inherit it)"
        )

    return out, changes


def normalise_chat_template(
    text: str, *, chat_template_thinking: str = "keep"
) -> tuple[str, list[str]]:
    """Return (normalised template, list of human-readable changes).

    `chat_template_thinking` is one of THINKING_POLICIES; see the block comment on THINKING_POLICIES for
    what each one means and, more importantly, for the policy that is deliberately absent.

    Refuses rather than repairs. A template whose `enable_thinking` default line is missing or
    duplicated is not a template this function understands, and editing it on a guess would
    change the published model's generation prompt -- the one string every consumer's first
    token is conditioned on.
    """
    if chat_template_thinking not in THINKING_POLICIES:
        raise ExportError(
            f"chat_template_thinking={chat_template_thinking!r} is not one of "
            f"{list(THINKING_POLICIES)}."
        )
    if chat_template_thinking == "keep":
        return text, []

    lines = text.splitlines(keepends=True)
    hits_on = [i for i, ln in enumerate(lines) if ln.strip() == _THINKING_LINE_ON]
    hits_off = [i for i, ln in enumerate(lines) if ln.strip() == _THINKING_LINE_OFF]

    if len(hits_on) + len(hits_off) != 1:
        raise ExportError(
            f"chat_template.jinja: expected exactly one `enable_thinking` default line, "
            f"found {len(hits_on)} defaulting True and {len(hits_off)} defaulting False. "
            "This step will not guess which one sets the published generation prompt. "
            "Expected, ignoring indentation:\n"
            f"  {_THINKING_LINE_ON}"
        )
    if hits_off:
        return text, [
            f"enable_thinking already defaults to False (line {hits_off[0] + 1}); "
            "policy 'default-off' had nothing to change"
        ]

    i = hits_on[0]
    before = lines[i]
    indent = before[: len(before) - len(before.lstrip())]
    newline = before[len(before.rstrip("\r\n")) :]
    lines[i] = indent + _THINKING_LINE_OFF + newline
    out = "".join(lines)

    # The five-site invariant. `default-off` edits one default; it must not add or remove any
    # empty think block.
    got, want = out.count(_EMPTY_THINK), text.count(_EMPTY_THINK)
    if got != want:
        raise ExportError(
            f"internal: 'default-off' changed the number of literal {_EMPTY_THINK} sites "
            f"from {want} to {got}. That policy is only allowed to flip one default; the "
            "empty block is load-bearing structure in this template (see THINKING_POLICIES)."
        )

    return out, [
        f"enable_thinking default True -> False (line {i + 1}); the generation prompt a "
        "consumer gets without asking becomes '<|im_start|>assistant\\n<think></think>', "
        "which is the prefix every labelled span in the training corpus began with"
    ]


def export(
    checkpoint: Path,
    dest: Path,
    *,
    padding_side: str = "right",
    chat_template_thinking: str = "keep",
    allow_unknown: bool = False,
) -> dict[str, Any]:
    """Copy the publishable subset of `checkpoint` into `dest`, normalising as it goes."""
    if not checkpoint.is_dir():
        raise ExportError(f"checkpoint is not a directory: {checkpoint}")

    parts = classify(checkpoint)
    if "model.safetensors" not in parts["keep"] and not any(
        n.startswith(KEEP_SHARD_PREFIX) and n.endswith(KEEP_SHARD_SUFFIX)
        for n in parts["keep"]
    ):
        raise ExportError(
            f"{checkpoint} has no model.safetensors (nor sharded model-*.safetensors). "
            "Either zero3_save_16bit_model was not set on the training run -- in which "
            "case the weights exist only inside global_step*/ and this step genuinely "
            "would need a converter -- or this is not a checkpoint directory."
        )
    if parts["unknown"] and not allow_unknown:
        raise ExportError(
            f"unrecognised entries in {checkpoint}: {parts['unknown']}. This step refuses "
            "to guess whether they belong in a published model. Add them to KEEP_FILES or "
            "PRUNE_KNOWN in export_hf_model.py, or pass --allow-unknown to drop them."
        )

    dest.mkdir(parents=True, exist_ok=True)
    changes: list[str] = []
    for name in parts["keep"]:
        src = checkpoint / name
        if name == "config.json":
            # Rewritten only when there is something to change, same discipline as
            # tokenizer.json and the chat template below: a config the trainer already
            # wrote correctly (align's, on transformers 4) must stay byte-identical so
            # `diff -r` against the checkpoint does not report a change this step did not
            # make.
            raw_c = json.loads(src.read_text())
            norm_c, mc_changes = normalise_model_config(raw_c)
            if mc_changes:
                (dest / name).write_text(
                    json.dumps(norm_c, indent=2, ensure_ascii=False) + "\n"
                )
            else:
                shutil.copy2(src, dest / name)
            # Asserted on what was actually WRITTEN, not on what normalise returned, so a
            # future edit to the serialisation cannot slip past the guard.
            assert_config_portable(json.loads((dest / name).read_text()))
            changes.extend(f"config.json: {c}" for c in mc_changes)
        elif name == "tokenizer_config.json":
            raw = json.loads(src.read_text())
            norm, tc_changes = normalise_tokenizer_config(
                raw, padding_side=padding_side
            )
            (dest / name).write_text(
                json.dumps(norm, indent=2, ensure_ascii=False) + "\n"
            )
            changes.extend(f"tokenizer_config.json: {c}" for c in tc_changes)
        elif name == "tokenizer.json":
            # Rewritten only when there is something to clear. This file is ~7 MB of
            # vocabulary, it was written by the Rust `tokenizers` serialiser, and
            # json.dumps does not reproduce that formatting -- so a round-trip on a clean
            # tokenizer would reformat the whole file and make `diff -r` against the
            # checkpoint report a change this step did not make. Same discipline as the
            # chat template below.
            raw_j = json.loads(src.read_text())
            norm_j, tj_changes = normalise_tokenizer_json(raw_j)
            if tj_changes:
                (dest / name).write_text(
                    json.dumps(norm_j, indent=2, ensure_ascii=False) + "\n"
                )
            else:
                shutil.copy2(src, dest / name)
            changes.extend(f"tokenizer.json: {c}" for c in tj_changes)
        elif name == "chat_template.jinja":
            # Written rather than copy2'd only when the policy actually changes something, so
            # that `keep` leaves a byte-identical file with the checkpoint's own mtime -- a
            # rewritten-but-unchanged template would make `diff -r` against the checkpoint
            # report a difference this step did not make.
            raw_t = src.read_text()
            norm_t, ct_changes = normalise_chat_template(
                raw_t, chat_template_thinking=chat_template_thinking
            )
            if norm_t == raw_t:
                shutil.copy2(src, dest / name)
            else:
                (dest / name).write_text(norm_t)
            changes.extend(f"chat_template.jinja: {c}" for c in ct_changes)
        else:
            shutil.copy2(src, dest / name)

    manifest = {
        "source_checkpoint": str(checkpoint),
        "kept": parts["keep"],
        "pruned": parts["prune"],
        "dropped_unrecognised": parts["unknown"] if allow_unknown else [],
        "normalisations": changes,
        "padding_side_policy": padding_side,
        "chat_template_thinking_policy": chat_template_thinking,
    }
    (dest / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def verify(
    dest: Path,
    *,
    expect_tokenizer_from: Path | None = None,
    source_checkpoint: Path | None = None,
) -> list[str]:
    """Load the exported dir and confirm it works. Imports transformers LAZILY so that
    classify/normalise/export stay unit-testable on a CPU box with no transformers."""
    notes: list[str] = []
    from transformers import AutoConfig, AutoTokenizer  # noqa: PLC0415

    cfg = AutoConfig.from_pretrained(str(dest))

    # RESOLVED-VALUE DRIFT against the checkpoint this was cut from. Note what this can and
    # cannot do: both configs are read by the SAME transformers here, so it CANNOT catch the
    # RoPE relocation -- 5.8.0 reads the nested dict and resolves both sides to 10000000
    # whether or not the top-level key was written. `assert_config_portable`, which runs in
    # export() and needs no transformers, is what catches that.
    #
    # What this DOES catch is the risk normalise_model_config takes when it DROPS a key: an
    # absent `time_step_limit` is only safe because the class default is the same
    # `(0.0, inf)` the sentinel spelled. This asserts that per export rather than trusting
    # the comment which says so.
    if source_checkpoint is not None:
        src_cfg = AutoConfig.from_pretrained(str(source_checkpoint))
        drift = diff_resolved_numeric(src_cfg.to_dict(), cfg.to_dict())
        if drift:
            raise ExportError(
                "the exported config does not resolve to the same numbers as its source "
                f"checkpoint {source_checkpoint}:\n  "
                + "\n  ".join(drift)
                + "\nA normalisation changed a value that affects inference. Publishing "
                "this would serve a model the training run never produced."
            )
        notes.append(
            f"config resolves identically to {source_checkpoint} on every numeric field"
        )
    tok = AutoTokenizer.from_pretrained(str(dest))
    notes.append(f"tokenizer loaded: {type(tok).__name__}, vocab={len(tok)}")

    tokenizer_json = dest / "tokenizer.json"
    if tokenizer_json.is_file():
        from tokenizers import Tokenizer  # noqa: PLC0415

        raw = Tokenizer.from_file(str(tokenizer_json))
        problems = tokenizer_pin_problems(
            type(tok).__name__,
            tok.is_fast,
            lambda text: tok(text, add_special_tokens=False)["input_ids"],
            lambda text: raw.encode(text, add_special_tokens=False).ids,
        )
        if problems:
            raise ExportError(
                "the exported tokenizer does not segment the way its own tokenizer.json "
                "does:\n  "
                + "\n  ".join(problems)
                + "\nEvery consumer that loads it through AutoTokenizer would see text "
                "segmented differently from training."
            )
        notes.append(
            f"tokenizer segments as tokenizer.json on all {len(TOKENIZER_PIN_PROBES)} probes"
        )

    if expect_tokenizer_from is not None:
        ref = AutoTokenizer.from_pretrained(str(expect_tokenizer_from))
        probe = "<|im_start|>user\nhello<|im_end|>\n"
        got = tok(probe, add_special_tokens=False)["input_ids"]
        want = ref(probe, add_special_tokens=False)["input_ids"]
        if got != want:
            raise ExportError(
                "exported tokenizer does not agree with the expected (retagged student) "
                f"tokenizer at {expect_tokenizer_from}: {got[:12]} != {want[:12]}. The "
                "corpus was tokenized with one tokenizer; publishing another silently "
                "changes what the model was trained to expect."
            )
        notes.append(f"tokenizer identity matches {expect_tokenizer_from}")
    return notes


def tokenizer_pin_problems(
    loaded_class: str,
    is_fast: bool,
    encode: Callable[[str], list[int]],
    encode_raw: Callable[[str], list[int]],
    probes: Iterable[str] = TOKENIZER_PIN_PROBES,
) -> list[str]:
    """Why the published tokenizer would not segment as tokenizer.json does; empty if it would.

    A pure function over the two encoders so it is testable with no transformers installed.
    `verify()` supplies AutoTokenizer's `encode` and `tokenizers.Tokenizer`'s `encode_raw`.
    """
    problems: list[str] = []
    if loaded_class.startswith("GPT2Tokenizer"):
        problems.append(
            f"AutoTokenizer resolves {loaded_class}, so the tokenizer_class pin did not take"
        )
    elif not is_fast:
        problems.append(
            f"AutoTokenizer resolves the slow {loaded_class}, which cannot be reading "
            "tokenizer.json"
        )
    for probe in probes:
        got, want = encode(probe), encode_raw(probe)
        if got != want:
            problems.append(
                f"{probe[:40]!r} segments as {got[:12]} (len {len(got)}), tokenizer.json "
                f"as {want[:12]} (len {len(want)})"
            )
    return problems


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-output-dir", required=True)
    p.add_argument("--dest", required=True)
    # Empty string is accepted as a synonym for 'latest' so the step-template can pass
    # every flag UNCONDITIONALLY. The alternative -- wrapping optional flags in jinja
    # {% if %} inside a backslash-continued shell command -- puts a line continuation
    # next to a conditionally-empty line, where one flag going empty silently swallows
    # the following flag. Templates should not have to be clever about whitespace.
    p.add_argument(
        "--checkpoint",
        default="latest",
        help="'latest' or '' (highest step number), a bare checkpoint-N, or a path.",
    )
    p.add_argument("--padding-side", default="right", choices=PADDING_SIDES)
    # Defaults to 'keep': which generation prompt a published model hands out is a decision
    # about the artifact, and this step's job is to make it explicit and recorded, not to make
    # it. 'default-off' is the right value for a student distilled on a corpus with no think
    # traces -- see THINKING_POLICIES for the measurement and for why "just strip the empty
    # block" is not on this list.
    p.add_argument(
        "--chat-template-thinking", default="keep", choices=THINKING_POLICIES
    )
    # BooleanOptionalAction, for the same reason: it gives --allow-unknown/--no-allow-unknown
    # and --verify/--no-verify, so a template renders one or the other and never omits
    # the line. It also makes an explicit `--no-verify` in a recipe readable as a
    # decision rather than as an oversight.
    p.add_argument(
        "--allow-unknown", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument(
        "--expect-tokenizer-from",
        default="",
        help="Retagged student dir. If given, --verify asserts token-id agreement with it.",
    )
    p.add_argument(
        "--verify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Load the exported dir (needs transformers).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        ckpt = select_checkpoint(Path(args.train_output_dir), args.checkpoint)
        dest = Path(args.dest)
        ref = Path(args.expect_tokenizer_from) if args.expect_tokenizer_from else None

        # THE RESUME GATE. Inline rather than in a wrapper script because this step has no
        # wrapper -- both launchers in step-template.yaml invoke this file directly -- so the
        # gate can only drift out of step with the export it guards if it is somewhere else.
        #
        # classify() is called here AND again inside export(). That is one listdir, deliberately
        # not factored out: the alternative is to pass parts into export() and have its own
        # contract depend on a caller doing the classification, which is worse for a function
        # that is also called from tests.
        parts = classify(ckpt)
        expectation = export_state.expectation(
            checkpoint=ckpt,
            keep=parts["keep"],
            dropped_unrecognised=parts["unknown"] if args.allow_unknown else [],
            padding_side=args.padding_side,
            chat_template_thinking=args.chat_template_thinking,
            verify=args.verify,
            expect_tokenizer_from=ref,
        )
        declared = export_state.declared_outputs(parts["keep"])
        decision = step_state.decide(
            dest, export_state.STEP_NAME, expectation, declared
        )
        for line in decision.lines:
            print(f"  {line}")
        if decision.kind == step_state.SKIP:
            # A SKIP is only honoured if the artifact already on disk still passes the guard
            # it would be published under TODAY. The expectation key covers the checkpoint
            # and the policy flags but not this step's normalisation contract, so without
            # this an export written before the RoPE hoist existed would be skipped forever
            # -- see published_config_needs_repair.
            stale = published_config_needs_repair(dest)
            if stale is None:
                # Exit 0, not 64. The 64 is step_state's internal vocabulary; the contract with the
                # launcher is that a successful no-op returns 0 so the LLMB_ARTIFACT_ID line after
                # this call still runs. A step that skips and exits non-zero starves its consumer.
                print(f"nothing to do: {dest} already holds this export.")
                return 0
            print(
                "  re-exporting: the expectation is unchanged but the published "
                f"config.json no longer meets this step's contract -- {stale}"
            )
        if decision.kind == step_state.REFUSE:
            print(
                f"FATAL [{export_state.STEP_NAME}]: refusing to overwrite {dest}. "
                "See the key named above.",
                file=sys.stderr,
            )
            return 2

        manifest = export(
            ckpt,
            Path(args.dest),
            padding_side=args.padding_side,
            chat_template_thinking=args.chat_template_thinking,
            allow_unknown=args.allow_unknown,
        )
        print(f"exported {ckpt} -> {args.dest}")
        print(f"  kept   : {len(manifest['kept'])} file(s)")
        print(f"  pruned : {manifest['pruned']}")
        for c in manifest["normalisations"]:
            print(f"  normalised: {c}")
        if args.verify:
            for note in verify(dest, expect_tokenizer_from=ref, source_checkpoint=ckpt):
                print(f"  verified: {note}")
        # Marked LAST, after verify(). An export that cannot be loaded must not be recorded as
        # complete -- otherwise the next run SKIPs and the broken directory becomes permanent.
        # The declared outputs are re-derived from the MANIFEST's kept list rather than reused
        # from `declared` above, so that what is recorded is what export() actually wrote.
        step_state.write_marker(
            dest,
            export_state.STEP_NAME,
            expectation,
            export_state.declared_outputs(manifest["kept"]),
        )
        print(
            f"  marked : {step_state.MARKER_NAME} written ({len(manifest['kept']) + 1} outputs)"
        )
    except ExportError as exc:
        print(f"FATAL [distill-hf-export]: {exc}", file=sys.stderr)
        return 2
    # No LLMB_ARTIFACT_ID line here: the step-template's launcher already emits it, and this
    # script printing it too meant ONE run announced the same id TWICE. Artifact emission
    # belongs to the launcher, because that is where the declared name lives -- keeping the
    # echo next to the outputs block is what stops the two from drifting apart. (Same
    # reasoning removed a duplicate from build_overlay.py, where the problem was first
    # measured: LSF job 1137372.)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
