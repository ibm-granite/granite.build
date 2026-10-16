#!/usr/bin/env python3
#
# PORTED, not authored here. Upstream source of truth:
#   repo   github.com/laminair/gb-steps-distillation
#   path   steps/distill-corpus-prep/src/prep_corpus.py
#   commit see code_config.expect_ref in step-template.yaml (not repeated here, because
#          two places to write a ref is one place for it to be stale)
#
# Verbatim apart from `black`/`isort` reflow, which CI requires repo-wide. Keep it that
# way so re-syncing upstream stays a three-way merge; behaviour changes belong upstream.
#
# --explode-assistant-turns, --explode-max-per-conv and --max-completion-length (with the
# four helpers they need -- _length_summary, prompt_budget_of, explode_assistant_turns,
# prompt_token_count -- and the per-record emit refactored out of build()'s loop, since one
# input record can now become several rows) were already present when this file was first
# vendored into gb-steps-distillation; there is no later re-sync to track for them, and no
# earlier upstream commit in THIS repo's history to name. (They trace back further, to the
# private source checkout's own commit 8a9ae32 -- "Build the unified 100-step round, and give
# ULD a teacher that renders every turn" -- but that checkout is not what code_config clones
# from, so its history is not this file's provenance chain.) Every one of the three flags is
# default-OFF: `train.jsonl` and the row sidecar come out byte-identical to a run with none of
# them set, and the manifest gains five keys that read `false`/`null`/`0`.
#
# ONE THING WORTH KNOWING ANYWAY: the two new policies are in expectation(), so an out_dir
# built by a copy of this file that predates them REFUSES on its marker ("DIFFERENT
# expectation") instead of reporting SKIP. That is the correct answer -- a fingerprint that
# omitted the flags would let a build with --explode-assistant-turns walk past an un-exploded
# corpus and call it done -- but it means adopting such an out_dir needs its marker removed
# once. Delete `out_dir/.step-done.json` (the path main() prints on an AlreadyDone) to adopt
# an existing corpus, or rebuild it.
#
# It imports gb_steps_post_training.distillation at module scope, which is delivered at
# RUN time from the checkout named by code_config (see step-template.yaml). That is why
# the tests for this file are gated on GB_DISTILL_CODE_DIR — see test/conftest.py.
#
"""Build a GOLD training corpus: filter, normalise and split a conversation dataset.

WHAT THE OUTPUT IS, precisely, because the plan doc got this wrong and the mistake is easy
to repeat. The corpus is a TEXT-level JSONL of `messages` conversations. It contains no
token ids. gold.py detects a `.json`/`.jsonl` dataset_name, reads it with pandas, wraps it
in a Dataset (gold.py:389-433) and the trainer tokenizes at train time.

SO WHY IS THE CORPUS TOKENIZER-SPECIFIC, and why does this step demand a --tokenizer at
all? Because every decision it makes about WHICH conversations survive is measured in
tokens:
  - a conversation is kept or dropped by its rendered length against --max-length;
  - the completion boundary is checked by rendering with return_assistant_tokens_mask=True
    and requiring a non-empty mask.
Both answers move when the tokenizer moves. The two families here differ in pre_tokenizer
(Sequence[Split(regex), ByteLevel] on granite-4.1 vs plain ByteLevel on granite-4.2), which
is easily enough to push examples across a length boundary. Nothing downstream re-checks
lengths, so a corpus built with the wrong tokenizer is not a crash: it is a training set
that quietly contains examples the trainer will truncate. That is what corpus_manifest.json's
`tokenizer_identity` exists to prevent -- distill-gold-train refuses a mismatch against the
student it is about to train.

WHY THE TOKENIZER IS LOADED WITHOUT AutoTokenizer. Same reason retag_student.py does not use
it: a Granite directory's tokenizer_config.json declares `tokenizer_class: "GPT2Tokenizer"`,
so AutoTokenizer builds THAT class, which rebuilds its backend from vocab+merges and installs
a plain ByteLevel(use_regex=True) -- discarding whatever pre_tokenizer tokenizer.json stored.
It does not error, and it still reports is_fast=True: CLASS IDENTITY is the mechanism, not
fast-versus-slow (docs/tokenizer_mismatch.md). Since this step's whole purpose is to measure
lengths with the tokenizer that will actually train, being wrong here would be self-defeating
in a way no test downstream would catch. So the tokenizer is built directly as
PreTrainedTokenizerFast(tokenizer_file=...) -- which takes the rule stored in tokenizer.json
and nothing else, exactly like the retag -- and the chat template is read from
chat_template.jinja by hand. Whether that stored rule is the one a MODEL was trained with is
a separate question, settled by measurement rather than by reading its files; this step only
has to agree with the student that will train, which the manifest is what enforces.

THE CHECK THAT EARNS ITS KEEP. --completion-boundary is verified, not merely recorded. A
chat template without `{% generation %}` markers yields an all-zero assistant mask, and
GOLD does not notice until sft.py:909 -- after model load, after the vLLM server is up,
i.e. after minutes of an expensive multi-node allocation. Rendering one conversation here
turns that into an error in seconds. If EVERY conversation has an empty mask the step fails
rather than emitting an empty corpus, because "0 examples kept" and "your template has no
generation markers" deserve different exit messages.

Example:
    python prep_corpus.py \
      --dataset data/distillation/bespoke_stratos_17k_think.jsonl \
      --tokenizer output/retagged_student \
      --out-dir output --max-length 4096 \
      --think-policy keep --completion-boundary last_message \
      --eval-fraction 0.02 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping

from gb_steps_post_training.distillation import step_state, tokenizer_identity

# Roles a conversation may use. `tool` is included because the granite-4.2 template renders
# tool results as their own turn; a record using it is valid input, not a malformed one.
VALID_ROLES = ("system", "user", "assistant", "tool")

# Role names that mean a VALID_ROLES role under a different spelling, mapped rather than
# dropped -- but COUNTED in the manifest, because renaming roles is a transformation of the
# data and an operator is entitled to see how much of it happened.
#
# FOUND ON REAL DATA (LSF job 1137876): `data/distillation/en_sft_4.1` spells tool results
# `tool_response`, and 10,665 of its messages use it. granite-4.2's chat template accepts
# only system/user/assistant/tool -- and its `tool` branch renders the content wrapped in
# literal `<tool_response>` tags (chat_template.jinja:176-178). So the two spellings are the
# same concept and the dataset simply predates the template's naming; dropping those
# conversations would discard every tool-result conversation in the corpus for a spelling.
#
# This map is deliberately tiny and explicit. An unrecognised role still fails as
# `bad_role:<r>`, because guessing at an unknown role is how a corpus ends up rendering
# something nobody intended.
ROLE_ALIASES = {"tool_response": "tool"}

THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

THINK_POLICIES = ("keep", "strip", "require")
LENGTH_POLICIES = ("drop", "truncate")
BOUNDARIES = ("last_message", "all_assistant")

DOCUMENTS_POLICIES = ("drop", "keep")

MANIFEST_NAME = "corpus_manifest.json"
STEP_NAME = "distill-corpus-prep"


def _declared_outputs(out_dir: Path) -> list[str]:
    """What this step promises to leave behind, as filenames relative to out_dir.

    eval.jsonl is conditional -- it is written only when eval_fraction > 0 -- so it is included
    only if it exists. Listing it unconditionally would make every zero-eval run REFUSE on the
    absence of a file it never produces, which is the sort of check that gets deleted rather than
    fixed. The manifest and the row sidecar are both small enough to be digested, so a corpus
    whose manifest was edited by hand does NOT read as complete.
    """
    names = ["train.jsonl", ROWS_NAME, MANIFEST_NAME]
    if (out_dir / "eval.jsonl").exists():
        names.insert(1, "eval.jsonl")
    return names


ROWS_NAME = "corpus_rows.jsonl"
ROW_ID_SCHEME = "blake2b-128/canonical-json"
# The key the id is written under when --emit-row-id is set. Named once, because it has to
# agree in three places that are edited at different times: what prep writes, what row_id()
# excludes from its own preimage, and what the trainer-side manifest reads back.
ROW_ID_FIELD = "row_id"


def template_renders_documents(tok) -> bool:
    """Does THIS tokenizer's chat template actually put `documents` into the prompt?

    The whole case for dropping grounded records rests on the template silently discarding
    them (audit-corpus-renderability.py: 7,013 of 811,172 on the deliverable corpus), which
    makes the assistant's answer reference text the student will never see -- measured at a
    45.1% floor for verbatim >=12-word lifts from the discarded document. That is training
    hallucination on purpose.

    But that is a fact about a TEMPLATE, and templates are the thing this project patches
    most often. If a future retag renders documents properly, dropping those records would
    throw away 0.86% of the corpus to fix a defect that no longer exists. So the premise is
    re-measured on every run instead of being trusted: render one synthetic record whose
    document contains a sentinel that cannot occur by chance, and look for it.

    Returns True if the sentinel survives into the rendered text. A template that raises on
    `documents` counts as NOT rendering them, which is the conservative answer -- it is the
    same outcome for the student either way.
    """
    sentinel = "ZQX-DOCUMENT-SENTINEL-8f21"
    rec = [
        {"role": "user", "content": "summarise"},
        {"role": "assistant", "content": "ok"},
    ]
    try:
        out = tok.apply_chat_template(
            rec,
            tokenize=False,
            add_generation_prompt=False,
            documents=[{"title": "t", "text": sentinel}],
        )
    except Exception:
        return False
    if isinstance(out, list):
        out = out[0] if out else ""
    return sentinel in str(out)


def row_id(record: dict) -> str:
    """A stable content-addressed id for one conversation record.

    WHY THIS IS NEEDED AT ALL: the source corpus has NO id field. Its rows are exactly
    `messages` / `tools` / `documents` (measured over all 2,000 probe rows and the
    deliverable corpus's schema), so "which data point was used during training" has no
    answer until we manufacture one. A row's own content is the only identity available,
    which makes a content hash not a workaround but the only correct key: it is stable
    across reruns, across machines, and across a reshuffle, and it needs no counter that
    a resumed or sharded run could desynchronise.

    Canonical JSON (sorted keys, no incidental whitespace) so that two records differing
    only in key order or serialisation hash the same -- otherwise the same conversation
    read from two files would get two ids and the manifest would claim to have trained on
    something it did not. `ensure_ascii=False` so the bytes hashed are the text's own
    UTF-8, not an escaping artefact that a future writer could change without changing
    the data.

    blake2b truncated to 128 bits: 16 bytes is ~2e-29 collision probability over a
    million rows, which is far below the rate at which any other part of this pipeline is
    wrong, and it keeps the sidecar readable. Not sha256 only because the digest_size
    parameter makes the truncation explicit rather than a slice someone later "tidies".

    ROW_ID_FIELD IS EXCLUDED FROM THE HASH, and that is what makes the id usable rather
    than merely present. With --emit-row-id the id is written INTO the emitted record so it
    travels with the data all the way to the trainer; if the id were part of its own preimage
    that stamp would be impossible (the value would have to be known before it was computed),
    and hashing a row of train.jsonl would give an answer that matched nothing. Excluding the
    field instead makes two useful things true: stamping is idempotent, and anyone holding
    train.jsonl can recompute a row's id and check it against the one stamped there. That is
    a verifiable claim about provenance instead of a number to be trusted.
    """
    if ROW_ID_FIELD in record:
        record = {k: v for k, v in record.items() if k != ROW_ID_FIELD}
    canon = json.dumps(
        record, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.blake2b(canon, digest_size=16).hexdigest()


class PrepError(Exception):
    """A corpus the step refuses to build. Message is the operator-facing explanation."""


class AlreadyDone(Exception):
    """This exact corpus is already in out_dir. Carries the manifest that was found.

    Distinct from PrepError because the two mean opposite things to a restarted recipe: PrepError
    is "stop", this is "walk past step 1 and get on with step 2". Before this existed the shell
    wrapper refused on the mere presence of train.jsonl (prep-corpus.sh:108), which is the same
    answer for "identical corpus already built" and "different policies, do not overwrite" -- and
    a preemption during training could not restart the recipe without a human deleting 3.1 GB of
    correct output.
    """

    def __init__(self, manifest: dict, lines: list[str]) -> None:
        super().__init__("; ".join(lines))
        self.manifest = manifest
        self.lines = lines


def _length_summary(values: list[int]) -> dict[str, Any]:
    """max/mean/p50/p90 of a non-empty length list, by nearest-rank on the sorted values.

    Nearest-rank rather than interpolation, and clamped to the last index: these are token
    counts to compare against an integer budget, so a p90 of 24575.5 would be a length no row
    has. numpy is not imported by this step and will not be imported for four numbers.
    """
    ordered = sorted(values)
    idx90 = min(int(len(ordered) * 0.9), len(ordered) - 1)
    return {
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 1),
        "p50": ordered[len(ordered) // 2],
        "p90": ordered[idx90],
    }


def prompt_budget_of(args: argparse.Namespace) -> int | None:
    """The prompt allowance, DERIVED -- never a literal, and never stored as one.

    ONE FUNCTION, THREE CALLERS (the filter, the manifest, the expectation), because the number
    is a difference of two configured values and a fourth place to write `24576` is a fourth
    place for it to be wrong. The trainer computes it the same way at
    custom_gold_trainer.py:2045 (`args.max_length - args.max_completion_length`), so the pinned
    settings for the unified sweep -- max_length 32768, max_completion_length 8192 -- give 24576
    here without 24576 appearing anywhere in this file.

    None means "no prompt filter": --max-completion-length defaults to 0, which is also how the
    trainer spells it off (`args.max_completion_length is None` at :2036 takes the plain
    filter). Under None, measure() does not render the prompt at all, so every corpus built
    before this option existed rebuilds byte-identically.
    """
    mcl = int(getattr(args, "max_completion_length", 0) or 0)
    if mcl <= 0:
        return None
    return int(args.max_length) - mcl


def expectation(args: argparse.Namespace, identity: str) -> dict:
    """Everything that changes the bytes this step writes, and nothing that does not.

    The contract is the same one `policies` has in the manifest, and it is deliberately the
    SUPERSET of it: `policies` answers "how was this corpus shaped", while an expectation must also
    pin WHICH dataset and WHICH tokenizer, because the same policies over a different input are a
    different corpus. --hf-home is excluded on purpose: it moves a cache, not an output.

    tokenizer_identity rather than the tokenizer PATH, for the reason the manifest already uses it:
    a path can be repointed at a retagged directory with the same name, and it is the vocabulary
    that decides the token counts and the drop set.
    """
    return {
        "dataset": args.dataset,
        "dataset_split": args.dataset_split,
        "dataset_config": args.dataset_config or None,
        "tokenizer_identity": identity,
        "policies": {
            "max_length": args.max_length,
            "length_policy": args.length_policy,
            "think_policy": args.think_policy,
            "completion_boundary": args.completion_boundary,
            "min_messages": args.min_messages,
            "documents_policy": args.documents_policy,
            "emit_row_id": args.emit_row_id,
            # IN THE FINGERPRINT, not merely in the manifest. Explosion changes which rows
            # exist, so a rerun that flips it produces a different corpus; if it were absent
            # here the step would match the previous expectation and SKIP, leaving an
            # un-exploded corpus behind a config that asks for an exploded one. That is the
            # same shape as the finished-output_dir resume trap in gold.py:506 -- work
            # skipped while reporting success.
            "explode_assistant_turns": args.explode_assistant_turns,
            "explode_max_per_conv": (
                args.explode_max_per_conv if args.explode_assistant_turns else None
            ),
            # IN THE FINGERPRINT for the same reason explosion is, and the trap is sharper
            # here because the flag's whole purpose is to REMOVE rows. Absent from the
            # expectation, turning the filter on would match the previous run's fingerprint
            # and SKIP -- leaving an unfiltered corpus on disk behind a config that asks for a
            # filtered one, and reporting success. The consumer then trains four arms on rows
            # the other two silently drop, which is precisely the incomparability this filter
            # was added to remove.
            #
            # BOTH keys, not just the derived one: `prompt_budget` alone would make
            # (max_length 32768, mcl 8192) and (max_length 24576, mcl 0) fingerprint-equal on
            # this axis, and they are different corpora -- the first also drops rows over
            # 32768 in total, the second over 24576.
            "max_completion_length": (
                int(args.max_completion_length)
                if prompt_budget_of(args) is not None
                else None
            ),
            "prompt_budget": prompt_budget_of(args),
        },
        "seed": args.seed,
        "eval_fraction": args.eval_fraction,
        "max_examples": args.max_examples,
        "shard": {
            "index": int(getattr(args, "shard_index", 0) or 0),
            "count": int(getattr(args, "shard_count", 1) or 1),
        },
    }


# ------------------------------------------------------------------ loading


def load_records(dataset: str, split: str, config: str, hf_home: str) -> Iterator[dict]:
    """Yield raw records from a JSONL file or an HF dataset id.

    A local path wins over a hub id when both could match, and it is checked FIRST rather
    than by catching a hub error: a typo'd path that happens to look like `org/name` would
    otherwise become an outbound network call, which on this cluster means a long timeout
    instead of an immediate "no such file".
    """
    path = Path(dataset)
    if path.is_file():
        with path.open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    raise PrepError(
                        f"{dataset}:{lineno} is not valid JSON: {exc}"
                    ) from exc
        return
    if path.exists():
        raise PrepError(
            f"--dataset {dataset} exists but is not a file. "
            "Pass the .jsonl itself, or an HF dataset id."
        )
    # Deferred import: the hub path is the rarer one and datasets is a heavy import that a
    # local-JSONL run should not pay for (and, offline, should not need installed).
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise PrepError(
            f"--dataset {dataset} is not a local file, so it is treated as an HF dataset "
            f"id, but `datasets` is not importable: {exc}"
        ) from exc
    if hf_home:
        os.environ.setdefault("HF_HOME", hf_home)
    ds = load_dataset(dataset, config or None, split=split)
    for row in ds:
        yield dict(row)


def load_tokenizer(tokenizer_dir: Path):
    """PreTrainedTokenizerFast built straight from tokenizer.json, with the .jinja template.

    Never AutoTokenizer -- see the module docstring. Returns (tokenizer, template_source).
    """
    from transformers import PreTrainedTokenizerFast

    tok_file = tokenizer_dir / "tokenizer.json"
    if not tok_file.is_file():
        raise PrepError(
            f"{tok_file} is missing. --tokenizer must be a directory holding a "
            "fast tokenizer (a model dir or distill-tokenizer-align's "
            "retagged_student / *_overlay)."
        )
    tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_file))

    jinja = tokenizer_dir / "chat_template.jinja"
    if jinja.is_file():
        tok.chat_template = jinja.read_text()
        return tok, str(jinja)
    # transformers 5.x treats chat_template.jinja as canonical, but a tokenizer_config from
    # an older export may still carry the template inline; accept it rather than failing on
    # a corpus that could legitimately be built.
    cfg = tokenizer_dir / "tokenizer_config.json"
    if cfg.is_file():
        inline = json.loads(cfg.read_text()).get("chat_template")
        if inline:
            tok.chat_template = inline
            return tok, f"{cfg} (inline chat_template)"
    raise PrepError(
        f"no chat template in {tokenizer_dir} (looked for chat_template.jinja and an inline "
        "chat_template in tokenizer_config.json). Lengths and the completion boundary are "
        "both measured on the RENDERED conversation, so this step cannot proceed without "
        "one. distill-tokenizer-align installs it via --chat-template."
    )


# ------------------------------------------------------------------ normalising


def normalise(
    record: dict,
    *,
    think_policy: str,
    boundary: str,
    min_messages: int,
    stats: dict | None = None,
    documents_policy: str = "keep",
) -> tuple[dict | None, str]:
    """Return (record, "") or (None, reason). Pure: does not touch the tokenizer.

    Kept separate from the length check so the cheap structural rejections happen before
    any rendering -- and so the reasons are testable without a tokenizer at all.
    """
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return None, "no_messages"
    if len(messages) < min_messages:
        return None, "too_few_messages"

    out: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict):
            return None, "malformed_message"
        role, content = msg.get("role"), msg.get("content")
        if role in ROLE_ALIASES:
            if stats is not None:
                stats[f"role_alias:{role}->{ROLE_ALIASES[role]}"] = (
                    stats.get(f"role_alias:{role}->{ROLE_ALIASES[role]}", 0) + 1
                )
            role = ROLE_ALIASES[role]
        if role not in VALID_ROLES:
            return None, f"bad_role:{role}"
        if not isinstance(content, str):
            # tool_calls-only assistant turns exist in some corpora; they are legitimate
            # data but not something this step knows how to length-check or strip think
            # tags from, so they are refused loudly rather than silently coerced to "".
            return None, "non_string_content"
        out.append({"role": role, "content": content})

    assistants = [m for m in out if m["role"] == "assistant"]
    if not assistants:
        return None, "no_assistant_turn"
    if not any(m["role"] == "user" for m in out):
        return None, "no_user_turn"

    # THE COMPLETION-BOUNDARY CONTRACT, enforced here rather than assumed. Under
    # last_message_only GOLD supervises exactly the final turn, so a conversation whose
    # final turn is NOT the assistant's contributes no loss at all -- it would be trained
    # on as a no-op, wasting the sample and skewing the reported example count.
    if boundary == "last_message" and out[-1]["role"] != "assistant":
        return None, "last_message_not_assistant"

    has_think = any("<think>" in m["content"] for m in assistants)
    if think_policy == "require" and not has_think:
        return None, "no_think_block"
    if think_policy == "strip":
        for m in out:
            if m["role"] == "assistant":
                m["content"] = THINK_RE.sub("", m["content"]).strip()
        if any(not m["content"] for m in out if m["role"] == "assistant"):
            return None, "empty_after_strip"

    kept = {"messages": out}
    # Pass through the two optional fields the granite chat templates consume. Dropping
    # them would silently change what the rendered prompt looks like relative to the source
    # dataset, which is the kind of difference that shows up only as a worse eval.
    #
    # COERCED, NOT FORWARDED VERBATIM, and that distinction cost a job (LSF 1137876, where
    # 395,007 of 405,672 en_sft_4.1 records died as `template_error:ValueError`). That
    # dataset stores `tools` as the JSON *string* `"[]"`, not as a list.
    # `apply_chat_template` iterates `tools` and demands each element be a dict or a
    # callable, so a string is iterated CHARACTER BY CHARACTER -- `'['` is neither, and it
    # raises. Nothing about the message says "your tools field is a string", which is why
    # the error is now surfaced verbatim (see `measure`).
    #
    # An empty tools/documents list is OMITTED rather than passed as `[]`. On this template
    # the two are equivalent (the tool-calling preamble is gated on truthiness), but a
    # record whose only difference from a plain conversation is an empty list should not
    # depend on a template's falsiness handling to render identically.
    for extra in ("documents", "tools"):
        raw = record.get(extra)
        if raw is None:
            continue
        # `documents_policy` is checked BEFORE the parse below, but only for a value that
        # parses to a non-empty list -- a record with `documents: []` or `"[]"` is not a
        # grounded record and must not be counted as one. That is why this cannot simply
        # test truthiness of the raw field: the string "[]" is truthy.
        if extra == "documents" and documents_policy == "drop":
            probe = raw
            if isinstance(probe, str):
                try:
                    probe = json.loads(probe)
                except ValueError:
                    probe = None
            if isinstance(probe, list) and probe:
                return None, "documents_not_renderable"
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return None, f"bad_{extra}"
        if not isinstance(raw, list):
            return None, f"bad_{extra}"
        if not raw:
            continue
        if not all(isinstance(item, dict) for item in raw):
            return None, f"bad_{extra}"
        kept[extra] = raw
    return kept, ""


def explode_assistant_turns(
    rec: dict, *, max_per_conv: int, seed_key: str
) -> list[dict]:
    """Turn one multi-turn conversation into several prefixes, each ending on an assistant.

    WHY THIS EXISTS -- it makes supervision symmetric across objectives, which no flag can.
    Under `last_message_only: true` the trainer supervises exactly the final assistant turn,
    so a conversation with 4 assistant turns contributes 1 and wastes 3. Under `false` it
    supervises all 4. The published sweep therefore trained ULD on 53.4% of the answer
    tokens its GOLD arms saw, because ULD's teacher path FORCES last_message_only: the
    off-policy render at custom_gold_trainer.py:3296 builds the teacher's view as
    `msgs[:-1]` against `teacher_full = msgs`, making the teacher's completion structurally
    the last message and nothing else. That is a confound in the objective comparison,
    not a tuning choice.

    Exploding at prep time removes it without touching the alignment core. A conversation
    whose assistant turns sit at indices k1..kn becomes up to n rows, row j being
    `messages[:kj+1]`. Then `msgs[:-1]` is exactly right for EVERY row on BOTH sides, and
    `last_message_only: true` can be set for all six arms at once -- so the arms differ in
    objective and nothing else. The alternative, teaching the teacher path to render
    per-turn, means editing the code that slices at a single SCALAR `teacher_prompt_length`
    shared across the batch (:3360-3460): per-row multi-segment breaks that contract, and
    that same core has already produced a loss of `student_logits.sum() * 0.0` -- an exact
    zero with a live graph, so job 1492353 advanced every step, wrote checkpoints, reported
    success, and trained nothing. This path cannot fail that way.

    THE LAST ASSISTANT TURN IS ALWAYS KEPT, which is what makes the change auditable: the
    full conversation is one of the emitted rows, so the exploded corpus is a SUPERSET of
    the un-exploded one and previously-trained numbers stay comparable. The remaining
    `max_per_conv - 1` slots are filled at random from the earlier turns rather than by
    taking the first ones, because taking the first would supervise only conversation
    openings -- systematically the easiest turns, before any tool result has come back.

    THE CAP IS A COST BOUND, NOT A PREFERENCE. Prefixes are re-encoded once per emitted row,
    so forward tokens grow with the number of variants while SUPERVISED tokens do not. At
    `max_per_conv 2` the tool axis roughly doubles its forward cost; uncapped, a 20-turn
    conversation would cost 20x for the same supervision.

    SEEDED PER ROW, from the row's own id rather than from a shared RNG, so the choice does
    not depend on iteration order. That matters here specifically: `--shard-count` splits
    the input round-robin, so a shared RNG would hand the same conversation different
    variants depending on which shard drew it, and the corpus would stop being a function
    of (dataset, seed).

    Returns the variants in ascending prefix length. A conversation with fewer than two
    assistant turns cannot be exploded and comes back unchanged as a single-element list --
    which is every row of the reasoning and instruction-following axes.
    """
    msgs = rec["messages"]
    idx = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    if len(idx) < 2 or max_per_conv < 1:
        return [rec]
    if max_per_conv >= len(idx):
        chosen = list(idx)
    else:
        # Already stable across runs without the sha256: CPython seeds Random from a str
        # via its bytes plus their sha512, not via hash(), so PYTHONHASHSEED does not
        # reach it. The wrapper is kept only to leave the chosen turns byte-for-byte
        # unchanged against corpora already built with it.
        rng = random.Random(hashlib.sha256(seed_key.encode("utf-8")).hexdigest())
        chosen = sorted(rng.sample(idx[:-1], max_per_conv - 1) + [idx[-1]])
    out = []
    for k in chosen:
        variant = dict(rec)
        variant["messages"] = [dict(m) for m in msgs[: k + 1]]
        out.append(variant)
    return out


# ------------------------------------------------------------------ measuring


def last_mask_span(mask) -> int:
    """Length of the FINAL contiguous run of 1s in an assistant mask.

    WHY THIS EXISTS. granite-4.2's chat template wraps EVERY assistant turn in
    `{% generation %}` (chat_template.jinja:99,141,152,157), so `assistant_masks` covers all
    of them. GOLD under `last_message_only=True` -- the setting this step's default
    `completion_boundary: last_message` corresponds to -- computes loss on only the FINAL
    assistant turn. Summing the whole mask therefore reports more supervised tokens than
    the trainer will actually supervise, by exactly the earlier assistant turns.

    The error is invisible on single-turn data, which is why it survived the first scale run
    (LSF job 1137876): `bespoke_stratos_17k` is one user turn and one assistant turn, so
    last_message and all_assistant produced byte-identical corpora and identical token
    stats. On a multi-turn corpus the same code would have overstated the supervised token
    count without any signal that it had.
    """
    end = next((i for i in range(len(mask) - 1, -1, -1) if mask[i]), None)
    if end is None:
        return 0
    start = end
    while start > 0 and mask[start - 1]:
        start -= 1
    return end - start + 1


def prompt_token_count(record: dict, tok) -> int:
    """Length of the PROMPT alone, rendered exactly as custom_gold_trainer.py:1961-1971 does.

    THIS IS A SECOND COPY OF THE TRAINER'S RENDER, AND THAT IS THE COST OF THE FEATURE. The
    objection is recorded at the --emit-row-id comment in build(): a predicate re-implemented
    here can drift from the trainer's, and a corpus that claims to be pre-filtered while
    filtering on a different rule is worse than one that makes no claim. Three things pay for
    it:

      1. A parity check (checks/prompt-budget-parity.py) imports BOTH this function and
         the trainer's real prepare/filter path and asserts they agree row by row on real
         corpus rows. Not a copy of the logic on either side -- the actual two call sites,
         so drift fails a check instead of shipping. KNOWN GAP: that check lives only in the
         private checkout these files were first ported from. It is NOT in
         gb-steps-distillation at `code_config.expect_ref`, and nothing in this repository
         runs it, so from here the parity rests on this file and the delivered trainer
         being the same pinned revision -- which they are -- not on a check that fails.
      2. Every kwarg here is the trainer's, including per-row `render_thinking` (the trainer's
         `row_thinking`, :1924) and the `documents` non-list coercion (:1902-1904, pandas NaN).
         The one difference is deliberate and inert: the trainer reads `tools` back out of an
         Arrow string column, this reads it before the emit re-serialises it, so both branches
         of its coercion are handled.
      3. The alternative is worse, which is the actual argument. The trainer's filter fires
         only when `lmbda != 0.0` (:2036), so the on-policy arms train on a strictly smaller
         row set than the off-policy ones -- from the same corpus, with no record of the
         difference. Six arms whose only intended difference is the loss would then differ in
         their data too, and the sweep would not be comparable. Filtering at build time is
         what makes one corpus mean one row set for all six arms.

    Returns a token count. Raises whatever the template raises; the caller attributes it.
    """
    prompt_messages = record["messages"][:-1]
    documents = record.get("documents")
    if not isinstance(documents, (list, tuple)):
        documents = []
    tools_raw = record.get("tools")
    if isinstance(tools_raw, str):
        tools = json.loads(tools_raw or "[]")
    elif isinstance(tools_raw, (list, tuple)):
        tools = list(tools_raw)
    else:
        tools = []
    ids = tok.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        documents=documents,
        tools=tools,
        enable_thinking=bool(record.get("render_thinking", False)),
        **(record.get("chat_template_kwargs") or {}),
    )
    # SHAPE-CHECKED BEFORE len(), because len() on the wrong shape is silently a small number
    # rather than an error, and a budget test against a small number keeps everything. Two
    # shapes have actually been returned by this API: a BatchEncoding (mapping), where len()
    # counts KEYS -- 2 -- and produced the "keep-rate 1.0 with p50=2" signature in an earlier
    # survey; and a list of per-conversation lists, where len() counts CONVERSATIONS -- 1.
    # Both are indistinguishable from a very short prompt downstream. The trainer's own
    # `len(x["prompts"])` at :2045 has the same exposure, so a mismatch here is a real finding
    # about the trainer and not a local inconvenience: it is raised, not worked around.
    if isinstance(ids, Mapping) or (ids and isinstance(ids[0], (list, tuple))):
        raise PrepError(
            f"apply_chat_template(tokenize=True, return_dict=False) returned "
            f"{type(ids).__name__}, not a flat token list, so len() would not be a token "
            f"count. The trainer's prompt filter (custom_gold_trainer.py:2045) takes len() of "
            f"this same call's result -- check that filter before changing this one."
        )
    return len(ids)


def measure(
    record: dict,
    tok,
    *,
    max_length: int,
    length_policy: str,
    boundary: str = "all_assistant",
    prompt_budget: int | None = None,
    detail: dict | None = None,
    totals: dict | None = None,
) -> tuple[dict | None, str, int, int]:
    """Render + tokenize one record. Returns (record|None, reason, n_tokens, n_target).

    n_target is the number of tokens THE TRAINER WILL SUPERVISE under `boundary`: the final
    assistant turn for `last_message`, every assistant turn for `all_assistant`. A record
    with none of them is dropped -- it is a sample the trainer would compute no loss on.

    `prompt_budget`, when given, additionally drops a record whose PROMPT alone renders to
    >= budget tokens, on the trainer's own predicate -- see the block at the end of this
    function and prompt_token_count(). None (the default) skips the extra render entirely, so a
    corpus built without it is byte-identical to one built before the option existed.

    `totals`, when given, gets THIS record's all-assistant mask count under
    `mask_tokens_all_assistant` -- overwritten per call, not accumulated, so the caller can
    add it up over KEPT records only. Accumulating here would silently mix in the records
    that were then dropped for length, and the resulting "unused signal" figure would be
    compared against a kept-only total and be nonsense. The figure matters because the gap
    between it and total_target_tokens is the argument for `all_assistant` on multi-turn
    data, so it is measured rather than left to guesswork.
    """
    kwargs = dict(
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    if record.get("tools") is not None:
        kwargs["tools"] = record["tools"]
    if record.get("documents") is not None:
        kwargs["documents"] = record["documents"]
    try:
        enc = tok.apply_chat_template([record["messages"]], **kwargs)
    except (
        Exception
    ) as exc:  # template errors are data-dependent, not programmer errors
        # The TYPE goes in the drop histogram (a bounded set, so the histogram stays
        # readable) and the first MESSAGE goes in `detail`, which build() reports and puts
        # in the manifest. Without the message, `template_error:ValueError` on 395,007
        # records says only "the template refused nearly everything" -- diagnosing it meant
        # a second job. One line of text would have been enough; now it is kept.
        if detail is not None:
            detail.setdefault(f"template_error:{type(exc).__name__}", str(exc)[:600])
        return None, f"template_error:{type(exc).__name__}", 0, 0

    ids = enc["input_ids"][0]
    mask = (enc.get("assistant_masks") or [[]])[0]
    n_tokens = len(ids)
    n_all = sum(mask)
    # The boundary scopes what is COUNTED, exactly as it scopes what the trainer optimises.
    n_target = n_all if boundary == "all_assistant" else last_mask_span(mask)
    if totals is not None:
        totals["mask_tokens_all_assistant"] = n_all

    # Tested on n_all, not n_target: an all-zero mask means the TEMPLATE has no generation
    # markers, which is a different fault from a boundary that happens to select nothing,
    # and build() reports the two differently.
    if n_all == 0:
        return None, "empty_assistant_mask", n_tokens, 0
    outcome = ""
    if n_tokens > max_length:
        if length_policy == "drop":
            return None, "over_max_length", n_tokens, n_target
        # truncate keeps the record but the caller is told, because truncation at the token
        # level cannot be reflected back into `messages` without re-detokenising -- so the
        # emitted record is the untruncated text and the trainer truncates it identically.
        # Recorded in the manifest as `truncated` so the count is not invisible.
        outcome = "truncated"

    # THE PROMPT BUDGET, tested LAST and so on a set that is disjoint from over_max_length.
    # Ordering is the whole design here. A row can fail both tests, and putting this first
    # would move rows out of `over_max_length` into `over_prompt_budget` -- silently changing
    # what a count means between two corpora built by the same command at different times.
    # Last, the two counts partition cleanly: `over_max_length` is "too long in total" and
    # `over_prompt_budget` is "fits in total, but its prompt alone leaves no room for the
    # completion". The second is the one a recipe can act on, by raising max_length or lowering
    # max_completion_length; the first cannot be fixed by rebalancing the split.
    #
    # `>=` MIRRORS THE TRAINER'S STRICT `<` KEEP-TEST (custom_gold_trainer.py:2045), which is
    # why the allowance at max_length 32768 / max_completion_length 8192 is 24,575 tokens and
    # not 24,576. Dropping here on `>` would keep exactly the rows at length == budget, which
    # the trainer then drops for the on-policy arms -- one row set for four arms and a
    # different one for two, which is the failure this filter exists to prevent.
    #
    # Also tested on a TRUNCATED-policy keep, deliberately. The trainer's prompt filter does not
    # consult length_policy: a row it keeps and truncates in total can still have an
    # over-budget prompt, and it drops that row.
    if prompt_budget is not None:
        try:
            n_prompt = prompt_token_count(record, tok)
        except PrepError:
            raise  # a shape fault, not a data fault -- see prompt_token_count()
        except Exception as exc:
            # A SEPARATE reason from the full render's `template_error:*`, for two reasons: the
            # empty-mask template-fault detector in build() compares its count against
            # n_rendered and must not see prompt-render faults, and a prompt-only failure means
            # something specific -- add_generation_prompt=True is the only difference between
            # the two calls, so this points at the template's generation block.
            if detail is not None:
                detail.setdefault(
                    f"prompt_template_error:{type(exc).__name__}", str(exc)[:600]
                )
            return (
                None,
                f"prompt_template_error:{type(exc).__name__}",
                n_tokens,
                n_target,
            )
        if totals is not None:
            totals["prompt_tokens"] = n_prompt
        if n_prompt >= prompt_budget:
            return None, "over_prompt_budget", n_tokens, n_target
    return record, outcome, n_tokens, n_target


# ------------------------------------------------------------------ driver


def build(args: argparse.Namespace) -> dict[str, Any]:
    tok_dir = Path(args.tokenizer)
    tok, template_src = load_tokenizer(tok_dir)
    try:
        identity = tokenizer_identity.read(tok_dir)
    except tokenizer_identity.IdentityError as exc:
        raise PrepError(str(exc)) from exc
    if identity is None:
        raise PrepError(f"cannot determine a tokenizer identity for {tok_dir}")

    renders_documents = template_renders_documents(tok)
    if args.documents_policy == "drop" and renders_documents:
        print(
            f"WARNING: --documents-policy drop, but {tok_dir}'s chat template DOES render "
            "`documents`. The reason for dropping grounded records was that the template "
            "discarded them; on this template it does not, so dropping them throws away "
            "usable grounding. Re-read the policy before trusting this corpus.",
            file=sys.stderr,
        )
    if args.documents_policy == "keep" and not renders_documents:
        print(
            "WARNING: --documents-policy keep, and this template DISCARDS `documents`. "
            "Grounded records will be rendered without their grounding, so the assistant "
            "answer references text the student never sees -- measured at a 45.1% floor "
            "for verbatim >=12-word lifts. This is the hallucination-training case.",
            file=sys.stderr,
        )

    shard_count = int(getattr(args, "shard_count", 1) or 1)
    shard_index = int(getattr(args, "shard_index", 0) or 0)
    if shard_count < 1:
        raise PrepError(f"--shard-count must be >= 1, got {shard_count}")
    if not 0 <= shard_index < shard_count:
        raise PrepError(f"--shard-index {shard_index} is outside 0..{shard_count - 1}")
    if shard_count > 1:
        # Both of these are refused rather than approximated, and for the same reason: they are
        # GLOBAL operations over the kept set, and a shard cannot see the kept set.
        #
        # --eval-fraction draws its split from a shuffle of this process's kept indices. Sharded,
        # that is K independent draws over K disjoint subsets -- still a valid split of the
        # corpus, but not the split (seed, n) names, so it would silently stop being
        # reproducible from the manifest. Prep the corpus sharded with no eval split and draw
        # the split once afterwards, or prep unsharded.
        if args.eval_fraction > 0:
            raise PrepError(
                "--eval-fraction with --shard-count > 1 would draw K independent splits over K "
                "disjoint subsets, so the manifest's (seed, eval_fraction) would no longer "
                "reproduce it. Prep sharded with --eval-fraction 0 and split afterwards."
            )
        # --max-examples means "stop after N kept". Per shard that is N*K, and which rows they
        # are depends on each shard's own drop rate -- so the same command yields a different
        # corpus at a different shard count, which is exactly what a cap is used to avoid.
        if args.max_examples:
            raise PrepError(
                f"--max-examples {args.max_examples} with --shard-count {shard_count} would keep "
                f"up to {args.max_examples * shard_count} rows, chosen differently at every "
                "shard count. Cap the input before prep, or prep unsharded."
            )

    # REFUSED, not clamped. A non-positive allowance would drop the entire corpus and then fail
    # on "0 of N records survived filtering", pointing at --max-length and the think policy --
    # the message that check prints -- rather than at the split that actually caused it.
    prompt_budget = prompt_budget_of(args)
    if prompt_budget is not None and prompt_budget <= 0:
        raise PrepError(
            f"--max-completion-length {args.max_completion_length} leaves no room for a prompt "
            f"inside --max-length {args.max_length} (allowance {prompt_budget}). The trainer "
            "splits max_length between prompt and completion rather than spending it on either, "
            "so the completion length must be well under the total."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Is this already built? Asked HERE -- after the tokenizer identity and the shard arguments
    # are known, since both are part of what "already built" means, and before the loop that
    # costs 0.7 h on the full corpus.
    want = expectation(args, identity)
    verdict = step_state.decide(out_dir, STEP_NAME, want, _declared_outputs(out_dir))
    if verdict.kind == step_state.SKIP:
        manifest_path = out_dir / MANIFEST_NAME
        raise AlreadyDone(json.loads(manifest_path.read_text()), verdict.lines)
    if verdict.kind == step_state.REFUSE:
        raise PrepError("\n  ".join(verdict.lines))

    drops: dict[str, int] = {}
    # Two side-channels out of the per-record helpers, both optional so neither changes the
    # helpers' contracts: `notes` counts transformations applied (role aliases), `detail`
    # keeps the first message behind each template-error type.
    notes: dict[str, int] = {}
    detail: dict[str, str] = {}
    totals: dict[str, int] = {}
    n_mask_all = 0
    kept: list[dict] = []
    # One entry per EMITTED ROW, kept or dropped -- which is one entry per input record
    # unless --explode-assistant-turns is on, where a conversation contributes one entry per
    # prefix and they share a `src_id`, separated by `explode_index`. This is the
    # per-datapoint training manifest; corpus_manifest.json stays aggregate. Two files rather
    # than one because a 811,172-row array inside the manifest would make the file that every
    # consumer reads for tokenizer_identity unreadable, and the aggregate counts are what most
    # consumers want. The sidecar is ~90 bytes/row, so ~73 MB at full corpus scale.
    rows: list[dict] = []
    lengths: list[int] = []
    targets: list[int] = []
    # Populated only under --max-completion-length. Kept rows only, so the percentiles describe
    # the corpus that exists rather than the one before filtering -- and the manifest says how
    # close the survivors run to the budget, which is what decides whether the split is set
    # right. A p90 far below the budget means the budget cost nothing; a p90 just under it means
    # the next corpus will lose rows to a small change in max_completion_length.
    prompt_lengths: list[int] = []
    n_in = 0
    n_truncated = 0
    n_rendered = 0
    # Explosion bookkeeping. Both stay 0 without --explode-assistant-turns, and both go in
    # the manifest: the forward-token cost of this feature is `n_explode_variants /
    # n_exploded_convs`, and a reader who cannot see that number cannot price the run.
    n_exploded_convs = 0
    n_explode_variants = 0

    def _emit(rec, rid, explode_index, reason, n_tok, n_tgt):
        """Serialise one surviving record into the corpus and the sidecar.

        A closure over build()'s accumulators rather than a module-level helper: it
        mutates six of them, and threading those through a signature would make the
        call site longer than the body. It exists at all because ONE INPUT RECORD CAN
        PRODUCE SEVERAL ROWS under --explode-assistant-turns, so this ran once per
        record and now runs once per emitted row.

        `explode_index` is None for an unexploded record, and the sidecar entry then
        omits the field entirely -- so a corpus built without the flag has a
        byte-identical sidecar to one built before the flag existed.
        """
        nonlocal n_mask_all
        # Accumulated here, over kept records only -- see measure()'s docstring.
        n_mask_all += totals.get("mask_tokens_all_assistant", 0)
        if "prompt_tokens" in totals:
            prompt_lengths.append(totals["prompt_tokens"])
        # `tools` GOES BACK TO A STRING BEFORE IT IS EMITTED, and `documents` does not.
        # normalise() parses both so measure() can render them (a string tools field is
        # iterated character by character by the template -- LSF 1137876), but the TRAINER
        # reads the two fields differently: it json.loads `tools`
        # (custom_gold_trainer.py:1745, again at :3075) and forwards `documents` to the
        # template as-is (:1750). Emitting the parsed list therefore made the deliverable
        # corpus crash the trainer's own preprocessing on the first tools-bearing row --
        # 17.5% of en-sft-4.1-0.2-16K, none of them in the 2,000-row probe head, so every
        # run so far was green. Found by upstream's checks/collator-masking.py on job 1162592.
        # Serialised HERE, before row_id(), so the id still hashes exactly the bytes that
        # reach train.jsonl and a holder of that file can still recompute it.
        # Only when present: an absent column reads as null and the trainer defaults it.
        if isinstance(rec.get("tools"), list):
            rec["tools"] = json.dumps(rec["tools"], ensure_ascii=False)
        # `out_id` hashes the EMITTED record, not the source one, and both are kept.
        # normalise() TRANSFORMS records (role aliases, think stripping, tools parsed from
        # their JSON string), so the two ids genuinely differ for any transformed row.
        # Recording only src_id would leave a holder of train.jsonl unable to look a row up;
        # recording only out_id would break the link back to the source dataset. Provenance
        # has to be traceable from BOTH ends or it answers only half the question.
        oid = row_id(rec)
        if args.emit_row_id:
            # THE ID TRAVELS WITH THE ROW. This is the difference between a manifest that
            # says what prep emitted and one that can say what the TRAINER consumed.
            #
            # The trainer applies its own row filter, and an arm-dependent one:
            # custom_gold_trainer.py:2045 additionally drops prompts over
            # `max_length - max_completion_length`, but only when lmbda != 0.0 (:2036).
            # Predicting that from here would mean re-implementing the trainer's prompt render
            # (add_generation_prompt=True, per-row enable_thinking, its tokenizer copy, its
            # template) in a second place that can drift from the first -- and a provenance
            # record that has silently drifted is worse than none, because it still looks
            # authoritative. Carrying the id instead lets the trainer answer from the dataset
            # it actually built, which is the only place the question has a true answer.
            #
            # THAT OBJECTION NOW HAS AN ANSWER, and --max-completion-length takes it: the
            # render IS re-implemented (prompt_token_count), because the alternative was two
            # different row sets across the six arms of one sweep. What makes it safe is not
            # confidence, it is a parity check asserting prep's predicate against the
            # TRAINER'S OWN, so drift fails a check -- see prompt_token_count() for where that
            # check lives, and why this repository does not run it. This comment's
            # argument still holds for its own subject: the row ids remain the record of what
            # the trainer consumed, since only the trainer knows what its arm did.
            #
            # Safe as an extra column: the tokenize map (:1814) sets no remove_columns, and
            # select_columns runs only under packing, which these configs do not use -- so
            # the field survives to trainer.train_dataset. It is also inert for rendering,
            # which reads only messages/tools/documents/chat_template_kwargs.
            rec[ROW_ID_FIELD] = oid
        entry = {
            "src_id": rid,
            "out_id": oid,
            "disposition": "kept",
            "reason": reason or "",
            "tokens": n_tok,
            "target_tokens": n_tgt,
            "kept_index": len(kept),
        }
        if explode_index is not None:
            entry["explode_index"] = explode_index
        rows.append(entry)
        kept.append(rec)
        lengths.append(n_tok)
        targets.append(n_tgt)

    # Counts EVERY record read, including other shards'. n_in counts only this shard's, so
    # that the sidecar's one-entry-per-assigned-record invariant still means what it says.
    g_in = 0
    for raw in load_records(
        args.dataset, args.dataset_split, args.dataset_config, args.hf_home
    ):
        if shard_count > 1:
            mine = g_in % shard_count == shard_index
            g_in += 1
            if not mine:
                continue
        n_in += 1
        rid = row_id(raw)
        rec, reason = normalise(
            raw,
            think_policy=args.think_policy,
            boundary=args.completion_boundary,
            min_messages=args.min_messages,
            stats=notes,
            documents_policy=args.documents_policy,
        )
        if rec is None:
            drops[reason] = drops.get(reason, 0) + 1
            rows.append({"src_id": rid, "disposition": "dropped", "reason": reason})
            continue
        # ONE INPUT RECORD CAN BECOME SEVERAL EMITTED ROWS. Without
        # --explode-assistant-turns this is always [rec] and every count below means
        # exactly what it meant before this flag existed. With it, the sidecar's invariant
        # changes from one-entry-per-input-record to one-entry-per-emitted-row: `src_id`
        # repeats across an exploded conversation's rows and `explode_index` separates
        # them. The manifest records that (see the `explode` block below) rather than
        # leaving a reader to infer it from duplicate ids.
        if args.explode_assistant_turns:
            variants = explode_assistant_turns(
                rec, max_per_conv=args.explode_max_per_conv, seed_key=rid
            )
            if len(variants) > 1:
                n_exploded_convs += 1
                n_explode_variants += len(variants)
        else:
            variants = [rec]

        for v_i, rec in enumerate(variants):
            exploded = len(variants) > 1
            if exploded:
                # RE-VALIDATED, not trusted because its parent was valid. A prefix is a
                # different conversation: it can fall under --min-messages, lose the only
                # <think> block under --think-policy require, or -- where a conversation's
                # first assistant turn precedes any user turn -- have no user turn at all.
                # Re-running the same contract is the whole reason a variant cannot enter
                # the corpus on its parent's credentials. Idempotent on an already
                # normalised record: the role aliases and the think-strip have no second
                # effect, and `tools` is already a list, which normalise accepts.
                rec, v_reason = normalise(
                    rec,
                    think_policy=args.think_policy,
                    boundary=args.completion_boundary,
                    min_messages=args.min_messages,
                    stats=notes,
                    documents_policy=args.documents_policy,
                )
                if rec is None:
                    # Namespaced, because these rejections have no counterpart in the
                    # un-exploded run: a prefix failing must never be read as the source
                    # dataset rejecting a record.
                    key = f"explode_variant:{v_reason}"
                    drops[key] = drops.get(key, 0) + 1
                    rows.append(
                        {
                            "src_id": rid,
                            "explode_index": v_i,
                            "disposition": "dropped",
                            "reason": key,
                        }
                    )
                    continue
            rec, reason, n_tok, n_tgt = measure(
                rec,
                tok,
                max_length=args.max_length,
                length_policy=args.length_policy,
                boundary=args.completion_boundary,
                prompt_budget=prompt_budget,
                detail=detail,
                totals=totals,
            )
            n_rendered += 1
            if rec is None:
                # NOT namespaced, deliberately, unlike the normalise rejections above. The
                # template-fault detector below fires on
                # `drops["empty_assistant_mask"] == n_rendered`, and that check is the one
                # thing standing between a template with no {% generation %} markers and a
                # corpus of entirely unsupervised rows. Namespacing these would blind it on
                # any exploded corpus -- trading a real safety check for tidier bookkeeping.
                drops[reason] = drops.get(reason, 0) + 1
                entry = {"src_id": rid, "disposition": "dropped", "reason": reason}
                if exploded:
                    entry["explode_index"] = v_i
                rows.append(entry)
                continue
            if reason == "truncated":
                n_truncated += 1
            _emit(rec, rid, v_i if exploded else None, reason, n_tok, n_tgt)
        # Checked after the variant loop, not inside it, so an exploded conversation is
        # emitted whole or not at all. Stopping mid-conversation would leave the corpus
        # holding a prefix whose siblings were cut by the cap -- a sample the cap chose
        # rather than the seed, and one that no rerun would reproduce.
        if args.max_examples and len(kept) >= args.max_examples:
            break

    # An all-zero assistant mask across the board is a TEMPLATE fault, not a data fault, and
    # it is the single most likely way this step is misconfigured -- so it gets its own
    # error. Reported before the "kept 0" check because it explains it.
    if n_rendered and drops.get("empty_assistant_mask", 0) == n_rendered:
        raise PrepError(
            f"every one of {n_rendered} rendered conversations produced an EMPTY assistant "
            f"mask. That is a chat-template fault, not a data fault: the template at "
            f"{template_src} has no {{% generation %}} markers, so nothing marks the "
            "completion span. GOLD would fail on this at sft.py:909 after loading the model "
            "and starting vLLM. Use a template with generation markers (see "
            "templates/chatml_granite_42_generation.jinja)."
        )
    if not kept:
        # THE ADVICE NAMES THE FLAG THAT DID IT. Listing three innocent flags is worse than
        # listing none: an operator whose corpus was emptied by the prompt budget reads
        # "check --max-length", widens it -- which makes the allowance LARGER and so looks like
        # the fix -- and gets the same zero, because `max_length - max_completion_length` is
        # still smaller than every prompt. The budget is spelled out arithmetically for the
        # same reason: it is a difference, not a setting, so quoting only the two inputs leaves
        # the reader to do the subtraction that surprised them in the first place.
        advice = (
            "Check --max-length, --think-policy and --completion-boundary against the "
            "dataset's actual shape."
        )
        if drops.get("over_prompt_budget"):
            advice = (
                f"{drops['over_prompt_budget']} of them were dropped by the PROMPT BUDGET: "
                f"--max-length {args.max_length} minus --max-completion-length "
                f"{args.max_completion_length} leaves {prompt_budget} tokens for the prompt, "
                f"and every prompt rendered longer. Widening --max-length raises that "
                f"allowance; lowering --max-completion-length raises it too, at the cost of a "
                f"shorter generation. " + advice
            )
        raise PrepError(
            f"0 of {n_in} records survived filtering. Drop reasons: {drops or 'none'}. "
            + "".join(f"First {k}: {v} " for k, v in detail.items())
            + advice
        )

    # Deterministic split from an explicit seed. Shuffling INDICES rather than the records
    # keeps the operation identical whether the corpus fits in memory comfortably or not,
    # and makes the split reproducible from (seed, n) alone.
    order = list(range(len(kept)))
    random.Random(args.seed).shuffle(order)
    n_eval = int(round(len(order) * args.eval_fraction))
    if args.eval_fraction > 0 and n_eval == 0:
        # Silently emitting an empty eval split would look like "eval was requested and
        # produced nothing", which reads as a bug downstream. Round up to one instead.
        n_eval = 1
    if n_eval >= len(order):
        raise PrepError(
            f"--eval-fraction {args.eval_fraction} would put all {len(order)} kept examples "
            "in the eval split, leaving nothing to train on."
        )
    eval_idx, train_idx = order[:n_eval], order[n_eval:]

    splits: dict[str, dict] = {}
    for name, idx in (("train", train_idx), ("eval", eval_idx)):
        if name == "eval" and not idx:
            continue
        dest = out_dir / f"{name}.jsonl"
        with dest.open("w") as fh:
            for i in sorted(idx):  # sorted: stable file order for a given index set
                fh.write(json.dumps(kept[i], ensure_ascii=False) + "\n")
        splits[name] = {"path": str(dest.resolve()), "examples": len(idx)}

    # The split is drawn AFTER the loop, so the sidecar's kept rows learn their split here.
    # Keyed by kept_index rather than by position in `rows`, because `rows` also holds the
    # dropped records and the two lists are deliberately different lengths.
    by_kept = {r["kept_index"]: r for r in rows if r["disposition"] == "kept"}
    for name, idx in (("train", train_idx), ("eval", eval_idx)):
        for i in idx:
            by_kept[i]["split"] = name

    manifest = {
        # THE KEY distill-gold-train COMPARES. Same name, same scheme, one function.
        "tokenizer_identity": identity,
        "tokenizer_path": str(tok_dir.resolve()),
        "chat_template_source": template_src,
        # An OBSERVATION about the template, not a policy -- which is why it sits here and
        # not in `policies`, whose contract is "every knob that changed the output" and is
        # asserted exactly by the tests. Measured on this run's tokenizer rather than
        # assumed (template_renders_documents()); it is what makes documents_policy
        # interpretable, since "drop" is only the right call while this is false.
        "template_renders_documents": renders_documents,
        "format": "messages-jsonl",
        "tokenized": False,  # spelled out: the corpus holds text, not ids
        "dataset": args.dataset,
        "dataset_split": args.dataset_split,
        "dataset_config": args.dataset_config or None,
        "policies": {
            "max_length": args.max_length,
            "length_policy": args.length_policy,
            "think_policy": args.think_policy,
            "completion_boundary": args.completion_boundary,
            "min_messages": args.min_messages,
            "documents_policy": args.documents_policy,
            "emit_row_id": args.emit_row_id,
            "explode_assistant_turns": args.explode_assistant_turns,
            # Recorded even when explosion is off, because "built without explosion" and
            # "built before the flag existed" are different claims and only one is checkable.
            "explode_max_per_conv": (
                args.explode_max_per_conv if args.explode_assistant_turns else None
            ),
            # Same key set as expectation()'s `policies`, asserted by
            # test_the_manifest_and_the_expectation_agree_on_which_keys_are_policies. The
            # derived value is recorded ALONGSIDE its input rather than left to be recomputed:
            # a consumer holding this manifest can check the trainer's `max_length -
            # max_completion_length` against the number this corpus was actually filtered on,
            # which is the comparison that catches a config drifting away from its data.
            "max_completion_length": (
                int(args.max_completion_length) if prompt_budget is not None else None
            ),
            "prompt_budget": prompt_budget,
        },
        "seed": args.seed,
        "eval_fraction": args.eval_fraction,
        # Always present, even for an unsharded run: a consumer must be able to tell "this is
        # the whole corpus" from "this is a piece of one" without inferring it from a missing
        # key. merge_shards.py refuses a set that does not agree on `count` or that is missing
        # an `index`.
        "shard": {"index": shard_index, "count": shard_count},
        "counts": {
            "input": n_in,
            "rendered": n_rendered,
            # 0/0 when explosion is off. `explode_variants / exploded_conversations` is the
            # forward-token multiplier this feature costs, and it is the number that decides
            # whether --explode-max-per-conv is set right for a given corpus.
            "exploded_conversations": n_exploded_convs,
            "explode_variants": n_explode_variants,
            "kept": len(kept),
            "truncated": n_truncated,
            "dropped": sum(drops.values()),
            "drop_reasons": dict(sorted(drops.items())),
            # Present only when something fired, so an empty-clean run keeps a clean
            # manifest. `transformations` is what was CHANGED (as opposed to dropped), and
            # it belongs in the manifest for the same reason drop_reasons does: a consumer
            # comparing this corpus to its source dataset needs to know.
            **({"transformations": dict(sorted(notes.items()))} if notes else {}),
            **({"template_errors": dict(sorted(detail.items()))} if detail else {}),
        },
        "token_stats": {
            "total_tokens": sum(lengths),
            "total_target_tokens": sum(targets),
            "max_tokens": max(lengths),
            "mean_tokens": round(sum(lengths) / len(lengths), 1),
            # Scoped by completion_boundary -- these are the tokens the trainer will
            # actually compute loss on, not every generation-marked token. See
            # last_mask_span().
            "mean_target_tokens": round(sum(targets) / len(targets), 1),
            # Every assistant turn, regardless of boundary. Equal to the two above under
            # `all_assistant`; under `last_message` the gap is the supervised signal the
            # boundary discards, which is the number a recipe needs to choose between them.
            "total_mask_tokens_all_assistant": n_mask_all,
            # Present only under --max-completion-length, so a corpus built without it keeps a
            # manifest identical to one built before the flag existed. PERCENTILES, not just a
            # max: the max is one row and says nothing about how much room the split left, while
            # p90 against `prompt_budget` is the headroom figure -- and the pair (p90, max) is
            # what distinguishes a distribution clipped by a generator's own ceiling from a
            # heavy-tailed one, which is the difference that decided the 32768 window.
            **(
                {"prompt_tokens": _length_summary(prompt_lengths)}
                if prompt_lengths
                else {}
            ),
        },
        "splits": splits,
    }
    # Written BEFORE the manifest, and the manifest carries its sha256. So a manifest that
    # exists is a manifest whose sidecar is already complete and whose digest was taken over
    # the finished file -- a consumer that finds both can verify it has the pair that were
    # produced together, rather than a sidecar from one run beside a manifest from another.
    rows_path = out_dir / ROWS_NAME
    with rows_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    h = hashlib.sha256()
    with rows_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)

    manifest["rows"] = {
        "path": str(rows_path.resolve()),
        "sha256": h.hexdigest(),
        "entries": len(rows),
        "row_id_scheme": ROW_ID_SCHEME,
        # Stated positively rather than left to be inferred from the corpus: a consumer
        # asking "can I trace what was trained on?" gets a yes/no here instead of having to
        # open train.jsonl and look for a field.
        "id_field": ROW_ID_FIELD if args.emit_row_id else None,
        # A row is in the training set iff disposition == "kept" AND split == "train".
        # Spelled out because "kept" alone is the wrong answer whenever eval_fraction > 0,
        # and that is a mistake a consumer makes once and never notices.
        "training_rows": sum(
            1 for r in rows if r["disposition"] == "kept" and r.get("split") == "train"
        ),
    }
    # An entry per EMITTED ROW, which is one per input record until --explode-assistant-turns
    # turns a conversation into several. Cheap to assert, and it is the invariant the whole
    # file rests on.
    #
    # THE EXPECTED COUNT IS DERIVED, NOT WIDENED. Every variant contributes exactly one entry
    # -- kept, or dropped at its re-validation, or dropped at measure -- so an exploded build
    # carries `explode_variants - exploded_conversations` entries beyond n_in and not one
    # more. Relaxing this to `>=` under explosion would have kept the check's shape while
    # giving up the only thing it does: catching a path out of the variant loop that records
    # nothing. (Upstream still compares against n_in here, which makes any exploded build
    # raise this error on an otherwise complete sidecar; fixed in this copy and reported.)
    expected_rows = n_in + (n_explode_variants - n_exploded_convs)
    if len(rows) != expected_rows and not args.max_examples:
        raise PrepError(
            f"sidecar has {len(rows)} entries, expected {expected_rows} for {n_in} input "
            f"records assigned to shard {shard_index}/{shard_count} "
            f"({n_explode_variants} variants from {n_exploded_convs} exploded "
            "conversations) -- a record was neither kept nor recorded as dropped, so the "
            "per-row manifest is incomplete and must not be published as one"
        )

    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    # LAST, and atomically. The marker is what a restarted recipe reads to decide whether to walk
    # past this step, so it must not be able to exist beside a half-written corpus -- which is why
    # it goes after the manifest, which itself goes after train.jsonl and the sidecar.
    step_state.write_marker(out_dir, STEP_NAME, want, _declared_outputs(out_dir))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="path to a conversations .jsonl, or an HF dataset id",
    )
    p.add_argument(
        "--dataset-split", default="train", help="HF split (ignored for a file)"
    )
    p.add_argument("--dataset-config", default="", help="HF config name, if any")
    p.add_argument(
        "--tokenizer",
        required=True,
        help="tokenizer directory whose lengths define this corpus -- normally "
        "distill-tokenizer-align's retagged_student",
    )
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="must match distill-gold-train's max_length; a corpus filtered at a "
        "different length silently contains examples the trainer truncates",
    )
    # WHY A CORPUS-BUILD FLAG NAMED AFTER A TRAINER ARGUMENT. `max_length` is SPLIT between
    # prompt and completion by the trainer, not spent on whichever the row needs: at
    # custom_gold_trainer.py:2045 a row survives only when its rendered PROMPT is strictly
    # shorter than `max_length - max_completion_length`. That filter is gated on `lmbda != 0.0`
    # (:2036), so the on-policy arms train on fewer rows than the off-policy ones from the same
    # corpus, with nothing on disk recording the difference. Applying it here makes one corpus
    # mean one row set for every arm, and puts the removal count in the manifest.
    #
    # Default 0 = OFF, so this changes no existing corpus and no existing command. Set it to the
    # trainer's own max_completion_length; the allowance is derived (prompt_budget_of), so the
    # number 24576 never appears in this file.
    p.add_argument(
        "--max-completion-length",
        type=int,
        default=0,
        help="the trainer's max_completion_length (0 = no prompt filter). Drops "
        "rows whose rendered prompt is >= max_length minus this, which is the "
        "trainer's own per-row test for the on-policy arms -- applied at build "
        "time so every arm trains on the same rows. Counted in the manifest as "
        "`over_prompt_budget`.",
    )
    p.add_argument(
        "--length-policy",
        choices=LENGTH_POLICIES,
        default="drop",
        help="drop: refuse over-length conversations (default -- a truncated "
        "assistant turn teaches an unterminated answer). truncate: keep "
        "them and let the trainer truncate, counted in the manifest.",
    )
    p.add_argument(
        "--think-policy",
        choices=THINK_POLICIES,
        default="keep",
        help="keep: leave <think> blocks in assistant turns. strip: remove them. "
        "require: drop conversations that have none (reasoning-only corpus).",
    )
    p.add_argument(
        "--completion-boundary",
        choices=BOUNDARIES,
        default="last_message",
        help="last_message: the final turn must be the assistant's, matching "
        "GOLD's last_message_only. all_assistant: every assistant turn is "
        "supervised.",
    )
    # WHY THIS IS NOT JUST `--completion-boundary all_assistant`. That flag changes WHICH
    # tokens the trainer supervises inside one row, and it cannot be used for the ULD arms:
    # ULD's teacher render (custom_gold_trainer.py:3296) builds the teacher's view as
    # `msgs[:-1]`, so the teacher's completion IS the last message and `last_message_only`
    # is forced true. Explosion instead changes the ROWS, so `last_message` supervises every
    # assistant turn across the corpus while each individual row still ends on one -- which
    # is the only way all six arms of a unified sweep can share a supervision scope.
    p.add_argument(
        "--explode-assistant-turns",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="emit one row per assistant turn (each row a prefix ending on that "
        "turn) instead of one row per conversation. Makes "
        "--completion-boundary last_message supervise every assistant turn "
        "across the corpus. Single-assistant conversations are unaffected.",
    )
    p.add_argument(
        "--explode-max-per-conv",
        type=int,
        default=2,
        help="cap on rows emitted per conversation when exploding (default 2). "
        "The last assistant turn is always one of them, so the exploded "
        "corpus is a superset of the un-exploded one; the rest are drawn at "
        "random, seeded from the row id. A cost bound: prefixes are "
        "re-encoded per row, so forward tokens scale with this and "
        "supervised tokens do not.",
    )
    p.add_argument("--eval-fraction", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="stop after N KEPT examples (0 = no limit). For smoke runs.",
    )
    p.add_argument("--min-messages", type=int, default=2)
    # Default "drop", which is NOT the identity behaviour, and deliberately so: on every
    # template in this project the grounding is silently discarded, so "keep" is the option
    # that quietly corrupts the corpus while looking like the safe choice. A default that
    # has to be overridden to do the harmful thing is the right way round. Both settings are
    # recorded in the manifest and both warn when they disagree with the measured template.
    # Default OFF, and deliberately not yet the default even though the manifest wants it.
    # Adding a column to the emitted corpus changes what reaches the collator, and no job has
    # yet proven the trainer tolerates it end to end. The evidence is encouraging rather than
    # conclusive -- custom_gold_trainer.py:1771 already carries `tools` through as a string
    # column for exactly this Arrow-compatibility reason -- so this flips to default-on once a
    # real run has trained with it, not before.
    # BooleanOptionalAction and not store_true, so steps/distill-corpus-prep/step-template.yaml
    # can pass the flag UNCONDITIONALLY -- same reason as build_overlay.py's --verify. A
    # store_true has no negative form, so the template's only option is a conditional that
    # renders to nothing, which leaves the preceding line's `\` continuing into a blank line.
    # That parses today and is one edit away from swallowing the next argument. Nothing calls
    # this with a value, so `--emit-row-id` keeps meaning exactly what it meant.
    p.add_argument(
        "--emit-row-id",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="write each emitted row's content id into the row itself, under "
        f"{ROW_ID_FIELD!r}, so the trainer can report which rows it actually "
        "consumed after its own arm-dependent filtering. Excluded from its own "
        "hash, so the stamp is idempotent and any holder of train.jsonl can "
        "recompute and verify it.",
    )
    p.add_argument(
        "--documents-policy",
        choices=DOCUMENTS_POLICIES,
        default="drop",
        help="drop: discard records with non-empty `documents`, because the "
        "chat template discards the grounding and the answer depends on it "
        "(7,013/811,172 = 0.86% on the deliverable corpus, 45.1% floor on "
        "verbatim dependence). keep: render them anyway.",
    )
    # ---------------------------------------------------------------- sharding
    # A full-corpus prep was measured at 303 input rows/s single-process -- ~0.7 h for
    # 811,172 rows -- so sharding is NOT needed at this corpus size and was deliberately not
    # built when that measurement came in. It is built now for the size AFTER this one: the
    # cost is linear in rows and entirely CPU-bound in the tokenizer, so a 5x corpus is a 3.5 h
    # serial job on a preemptable queue, which is a job that gets killed rather than a job that
    # finishes.
    #
    # ASSIGNMENT IS BY MODULO OVER THE INPUT STREAM, not by byte range. Every shard reads the
    # whole file and tokenizes only its own rows; reading 3.4 GB is IO-bound and cheap next to
    # the tokenization, and modulo needs no index, no line-offset table, and no assumption that
    # the input is seekable -- so the same code path works for an HF dataset as for a jsonl.
    #
    # The property worth having, and the reason for the round-robin merge in merge_shards.py:
    # K shards merged are BYTE-IDENTICAL to one unsharded run. Shard s holds input rows
    # s, s+K, s+2K, ..., each in input order, so a K-way lockstep merge reconstructs the global
    # input order exactly. That turns "is the sharded path correct?" into a diff, which is a
    # question a test can answer, instead of a distribution argument.
    p.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="split the input across this many independent processes (default 1, "
        "i.e. no sharding). Merge with merge_shards.py.",
    )
    p.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="which shard THIS process handles, 0-based. Takes input records where "
        "index %% shard-count == shard-index.",
    )
    p.add_argument("--hf-home", default="", help="HF_HOME for the hub path")
    return p


def out_dir_marker(out_dir: str) -> Path:
    return Path(out_dir) / step_state.MARKER_NAME


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.eval_fraction < 1.0:
        print("ERROR: --eval-fraction must be in [0.0, 1.0)", file=sys.stderr)
        return 2
    try:
        m = build(args)
    except AlreadyDone as exc:
        # rc 0. A recipe restarted after a preemption in a later step must be able to walk past
        # this one, and "the work you asked for is already here" is a success, not a failure.
        print(f"=== corpus already built -> {args.out_dir}")
        for line in exc.lines:
            print(f"  {line}")
        c = exc.manifest["counts"]
        print(
            f"  kept {c['kept']} of {c['input']} "
            f"(dropped {c['dropped']}, truncated {c['truncated']})"
        )
        print(f"  tokenizer identity : {exc.manifest['tokenizer_identity']}")
        print(f"  delete {out_dir_marker(args.out_dir)} to rebuild deliberately")
        return 0
    except PrepError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    c, t, pol = m["counts"], m["token_stats"], m["policies"]
    print(f"\n=== corpus built -> {args.out_dir}")
    print(f"  tokenizer identity : {m['tokenizer_identity']}")
    print(
        f"  kept {c['kept']} of {c['input']} "
        f"(dropped {c['dropped']}, truncated {c['truncated']})"
    )
    for reason, n in c["drop_reasons"].items():
        print(f"    drop {reason}: {n}")
    for label, n in c.get("transformations", {}).items():
        print(f"    applied {label}: {n} messages")
    for kind, msg in c.get("template_errors", {}).items():
        print(f"    first {kind}: {msg}")
    print(
        f"  tokens: mean {t['mean_tokens']} max {t['max_tokens']} "
        f"target-mean {t['mean_target_tokens']}"
    )
    unused = t["total_mask_tokens_all_assistant"] - t["total_target_tokens"]
    if unused > 0:
        pct = 100.0 * unused / t["total_mask_tokens_all_assistant"]
        print(
            f"  NOTE: completion_boundary={pol['completion_boundary']} leaves "
            f"{unused} assistant tokens ({pct:.1f}%) unsupervised. "
            f"completion_boundary=all_assistant would use them."
        )
    for name, s in m["splits"].items():
        print(f"  {name}: {s['examples']} -> {s['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
