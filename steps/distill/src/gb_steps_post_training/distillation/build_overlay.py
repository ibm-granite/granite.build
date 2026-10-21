#!/usr/bin/env python3
"""
Build a tokenizer *overlay*: a mirror directory holding the fast tokenizer of a model
directory and nothing that can make transformers fall back off it.

WHY THIS FILE EXISTS AT ALL. Until now it did not. This was, in an earlier plan, described
as "the single most reproducibility-critical undocumented step in the pipeline", and it was
undocumented in the strongest sense: many recipe configs pointed at `teacher_overlays/<teacher>`
paths that a human created BY HAND. Nothing in the repository could rebuild them. That is the
exact failure this collection exists to end -- an artifact every run depends on, whose
construction lives only in someone's shell history.

WHAT GOES WRONG WITHOUT IT -- MEASURED, NOT ASSUMED. A Granite model directory ships a
`tokenizer_config.json` carrying `tokenizer_class: "GPT2Tokenizer"`. transformers honours
that key and builds a GPT2Tokenizer whose pre_tokenizer is GPT-2's plain ByteLevel,
REPLACING the pre_tokenizer stored in tokenizer.json. It does not error. It silently
segments text with a pre_tokenizer the model was never trained with.

Whether that is harmless or catastrophic depends on the model:

  granite-4.2-3b (teacher)       tokenizer.json pre_tokenizer = ByteLevel
                                 -> the override is a no-op. Nothing diverges.
  granite-4.1-3b-base (student)  pre_tokenizer = Sequence[Split(regex), ByteLevel]
                                 -> the override DISCARDS the Split regex. 4/6 probes
                                    diverge. Ordinary prose still matches, so the damage
                                    concentrates in punctuation- and markup-dense text --
                                    which is to say, exactly the ChatML turn boundaries.

A direct measurement isolated the cause by materialising each variant separately
(transformers 5.8.0, tokenizers 0.22.1), for both models:

  files present                          class              pre_tokenizer used   ids
  tokenizer.json only                    TokenizersBackend  from tokenizer.json  match
  + vocab.json + merges.txt              TokenizersBackend  from tokenizer.json  match
  + tokenizer_config.json                GPT2Tokenizer      GPT-2 ByteLevel      DIVERGE
  + tokenizer_config.json minus that key TokenizersBackend  from tokenizer.json  match

Note that NONE of those variant dirs contained a `config.json`, which turned out to matter
a great deal -- see CONFIG_KEYS_FORCED, whose measurement varied that file and found
that its `model_type: granite` revives the override through
TOKENIZER_MAPPING_NAMES["granite"] even when `tokenizer_class` is absent entirely. The
table above is therefore sound but scoped to tokenizer-only directories, which is what an
overlay is. It is the reason this module pins the key rather than dropping it.

Three consequences, the first two contradicting what this module originally asserted:

  1. Under transformers 5.8 the legacy `vocab.json` / `merges.txt` sidecars are INERT.
     They trigger nothing; excluding them is hygiene, not the fix.
  2. The single load-bearing key is `tokenizer_class`. An earlier revision of this module
     blamed the sidecars and copied `tokenizer_config.json` verbatim -- so it built a
     student overlay that passed its own build and then failed its own verify.
  3. "Absent" is not a safe value for that key, only an unspecified one. Whether it
     resolves correctly depends on a file the overlay does not contain, so the overlay
     states the value positively instead of relying on that.

Earlier exploratory notes quantified the cost of getting this wrong at 26.1 vs 3.29
PPL/token. That number was measured under an unrecorded, older transformers, so it is
cited here as the stakes rather than as a reproduction of the mechanism above.

PINNING THE CLASS IS NOT ALWAYS THE FIX. It removes the override; it does not decide which
of the two candidate rules the model was actually trained with. Everything above silently
assumes tokenizer.json is authoritative and the imposed ByteLevel is the intruder. On
2026-09-22 a teacher was measured where the opposite holds, and the pin turned a working
directory into a broken one:

  granite-5.0-20b-sft   tokenizer.json pre_tokenizer =
                        Sequence[Split(cl100k-ish regex), ByteLevel(use_regex=False)]
                        -> and that Sequence is VESTIGIAL. The model's own likelihood puts
                           the imposed plain ByteLevel ahead by 17.0% of total NLL on a raw
                           render and 19.8% on a chat render, over 512 corpus documents
                           (measured directly). Only 3.33% of token boundaries differ at
                           all, so ~272,666 extra nats land on ~22,400 tokens
                           -- order 12 nats each against a ~2.5 nat corpus average. That
                           concentration is what makes it a segmentation finding and not
                           noise.

So for the 4.1/4.2 family the pin is right, and for that 5.0 checkpoint the pin alone
INTRODUCED a segmentation the model had never seen -- which cost a full 8-GPU arm, killed on
every rank in `utils.py:verify_tokenizer_consistency` at 0/24 steps. Three rules follow:

  a. Class identity is the mechanism, not fast-vs-slow. The GPT-2 class rebuilds its backend
     from vocab+merges and installs plain ByteLevel(use_regex=True) while still reporting
     is_fast=True (a direct measurement loaded the published 5.0 directory untouched and
     saw exactly that). "Pin it fast" reads like a general remedy and is not one.
  b. Rank candidates on TOTAL NLL, never PPL/token -- two pre-split rules emit different
     token counts, so a per-token mean is not comparable across them. The 26.1-vs-3.29 band
     cited above does NOT transfer: it was a slow class rebuilding the MERGES, an 8x spread.
     A pre-split-only difference cannot produce that, so the absolute number says nothing and
     only the margin plus its concentration can be read.
  c. When the measurement says the stored rule is the vestigial one, the remedy is the pin
     PLUS transplanting the trained pre_tokenizer in from a directory that carries it --
     a companion tool's `--pre-tokenizer-from` mode, which refuses unless the
     two sides share model.vocab and model.merges byte-for-byte. Verify below cannot catch
     this case on its own: it compares the resolved backend against the rule stored on disk,
     so a directory whose stored rule is the vestigial one passes a self-consistent check.

WHAT THE OVERLAY THEREFORE DOES: copy the small set of files a fast tokenizer legitimately
needs; drop the sidecars anyway (inert here, but other transformers releases do consult
them, and an overlay whose contents are known exactly is the point); and rewrite
`tokenizer_config.json` to pin `tokenizer_class` to `PreTrainedTokenizerFast`, so the class
lookup cannot silently replace the pre_tokenizer stored in tokenizer.json. Whether that
stored rule is the trained one is a separate question this module does not answer -- see
PINNING THE CLASS IS NOT ALWAYS THE FIX above. Every rewrite is recorded in the manifest with
both the old and the new value.

WHY THE CHECK IS A ROUND-TRIP AND NOT `is_fast`. `is_fast` is necessary but not
sufficient: it answers "did we get a fast class", not "did we get the RIGHT fast
tokenizer" -- direct measurement shows every variant, correct and broken alike, reporting
is_fast=True.

For the avoidance of a mistake an earlier revision of this docstring invited: the trainer's
own `utils.py:verify_fast_tokenizer` is NOT merely a class check. `is_fast` is only its
first and cheapest precondition; it goes on to compare the live backend's pre_tokenizer,
post_processor, normalizer and decoder reprs against a reference built from tokenizer.json,
and then to run encoding probes. That is the same property this module checks, arrived at
independently -- and its footer already names `tokenizer_class` in tokenizer_config.json as
a remedy, so it was closer to the mechanism than the plan doc's summary of it was. It
should not be replaced on the strength of a claim that it "only tests the class".

This module checks the property at overlay-BUILD time rather than at model-load time, which
is the difference that matters: it can refuse to publish a bad artifact. It encodes probe
text with BOTH
`AutoTokenizer.from_pretrained(overlay)` and `tokenizers.Tokenizer.from_file(
overlay/tokenizer.json)` -- the raw backend, which cannot be influenced by sidecars or by
tokenizer_config -- and asserts the id sequences are identical. That compares the
tokenizer transformers hands the trainer against ground truth, which is the property the
PPL number actually depends on.

    python -m gb_steps_post_training.distillation.build_overlay \
        --source <model dir> --out <overlay dir> --verify
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from gb_steps_post_training.distillation import tokenizer_identity

# Files an overlay MAY contain. Like distill-hf-export's, this is an explicit KEEP list:
# a DROP list would silently admit whatever a future transformers release starts writing,
# and the whole point of an overlay is that its contents are known exactly.
OVERLAY_KEEP = (
    "tokenizer.json",          # the fast tokenizer. The reason the overlay exists.
    "tokenizer_config.json",   # special tokens, model_max_length, chat template pointer.
    "special_tokens_map.json", # redundant with tokenizer.json for a fast tokenizer, but
                               # harmless and some tooling still reads it.
    "chat_template.jinja",     # transformers>=4.43 writes the template here.
    "added_tokens.json",       # ditto: redundant-but-harmless for a fast tokenizer.
    "generation_config.json",  # eos/pad ids -- read by vLLM when serving from a dir.
)

# Legacy vocabulary sidecars. Measured to be INERT under transformers 5.8:
# with only these next to tokenizer.json, the resolved pre_tokenizer is still the trained
# one. They are excluded anyway -- other transformers releases do consult them, and the
# value of an overlay is that its contents are known exactly -- but do not mistake their
# exclusion for the protection. See CONFIG_KEYS_FORCED for the key that actually bites.
SIDECARS = ("vocab.json", "merges.txt", "vocab.txt", "tokenizer.model", "spiece.model")

# Keys FORCED to a fixed value in the overlay's tokenizer_config.json.
#
# `tokenizer_class` is THE mechanism this module exists to defeat: transformers builds the
# named class, and that class imposes its own pre_tokenizer over the one in tokenizer.json.
#
# We pin it to PreTrainedTokenizerFast rather than deleting it, and the difference is not
# cosmetic. Deleting was this module's first fix and it is only conditionally correct --
# measured on granite-4.1-3b-base, varying tokenizer_config.json against the
# presence of a config.json declaring `model_type: granite`:
#
#   tokenizer_config.json      config.json    resolved class       pre_tokenizer   ids
#   absent                     absent         TokenizersBackend    Sequence        match
#   absent                     PRESENT        GPT2Tokenizer        ByteLevel       DIVERGE
#   tokenizer_class stripped    absent        TokenizersBackend    Sequence        match
#   tokenizer_class stripped    PRESENT       GPT2Tokenizer        ByteLevel       DIVERGE
#   tokenizer_class=GPT2         either       GPT2Tokenizer        ByteLevel       DIVERGE
#   tokenizer_class=Fast         either       TokenizersBackend    Sequence        match
#
# With a config.json present, transformers falls back to
# TOKENIZER_MAPPING_NAMES["granite"] -> "GPT2Tokenizer" and the override returns even
# though the key is gone -- and it returns even with NO tokenizer_config.json at all.
# So "strip the key" is a remedy that silently depends on the overlay never containing a
# config.json. That invariant does hold here (config.json is deliberately absent from
# OVERLAY_KEEP), but it is invisible at the point of use and one KEEP-list edit away from
# being false. Pinning the value is correct in all four permutations and needs no
# invariant, so it is what we do. It also matches retag_student.py and the canonical
# fixture convention in docs/tokenizer_mismatch.md.
CONFIG_KEYS_FORCED = {"tokenizer_class": "PreTrainedTokenizerFast"}

# Probe strings for the round-trip. Chosen to hit the cases where a wrong pre_tokenizer
# actually diverges rather than to look thorough: leading/repeated whitespace and
# punctuation-adjacency are precisely where GPT-2's default ByteLevel and Granite's Split
# regex disagree, and the ChatML control tokens are the ones that must survive as SINGLE
# ids rather than being spelled out.
PROBES = (
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
    "  leading and   repeated   whitespace  ",
    "punctuation,adjacency;matters:here!",
    "def f(x):\n    return x ** 2  # tabs\tand newlines\n",
    "Ünïcödé and emoji 🙂 mixed with ASCII",
    "<|im_start|><|im_end|><|end_of_text|>",
)


class OverlayError(RuntimeError):
    pass


def classify(source: Path) -> dict[str, list[str]]:
    """Split the source directory into overlay / sidecar / other."""
    if not (source / "tokenizer.json").is_file():
        raise OverlayError(
            f"{source} has no tokenizer.json, so there is no fast tokenizer to overlay. "
            "This is the one case an overlay cannot repair: there is no trained "
            "pre_tokenizer on disk to preserve. The fix is to obtain a model directory "
            "that ships tokenizer.json, not to build an overlay out of sidecars."
        )
    out: dict[str, list[str]] = {"overlay": [], "sidecar": [], "other": []}
    for p in sorted(source.iterdir()):
        if p.name in OVERLAY_KEEP and p.is_file():
            out["overlay"].append(p.name)
        elif p.name in SIDECARS:
            out["sidecar"].append(p.name)
        else:
            out["other"].append(p.name)
    return out


def build(source: Path, dest: Path, *, copy_mode: str = "copy") -> dict:
    """Create `dest` as an overlay of `source`. Returns a manifest."""
    if copy_mode not in ("copy", "hardlink"):
        raise OverlayError(f"copy_mode must be 'copy' or 'hardlink', got {copy_mode!r}")
    parts = classify(source)
    dest.mkdir(parents=True, exist_ok=True)

    forced: dict[str, dict[str, object]] = {}

    for name in parts["overlay"]:
        src, dst = source / name, dest / name
        if dst.exists():
            dst.unlink()
        if name == "tokenizer_config.json":
            # Rewritten, never copied: this is the file that carries `tokenizer_class`,
            # and copying it verbatim is exactly the bug that made the first version of
            # this module emit a broken student overlay. Rewriting also means a hardlink
            # is impossible here -- correctly so, since the overlay's config is genuinely
            # a different file from the source's.
            cfg = json.loads(src.read_text())
            changed: dict[str, object] = {}
            for k, v in CONFIG_KEYS_FORCED.items():
                if cfg.get(k) != v:
                    changed[k] = {"was": cfg.get(k, None), "now": v}
                    cfg[k] = v
            dst.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
            if changed:
                forced[name] = changed
            continue
        if copy_mode == "hardlink":
            try:
                dst.hardlink_to(src)
                continue
            except OSError:
                # Different filesystem, or a filesystem without hardlinks. Copying is
                # always correct, so degrade rather than fail -- but say so, because a
                # silent fallback would make a 'hardlink' manifest entry a lie.
                print(f"  note: hardlink failed for {name}, copied instead", file=sys.stderr)
        shutil.copy2(src, dst)

    return {
        "source": str(source),
        "dest": str(dest),
        "copy_mode": copy_mode,
        "overlay_files": parts["overlay"],
        # Recorded, not merely omitted: knowing WHICH sidecars were present is what tells
        # the next reader whether this source was actually affected by the trap.
        "sidecars_excluded": parts["sidecar"],
        # The load-bearing record. `sidecars_excluded` documents a precaution;
        # this documents the actual repair, per file and per key.
        "config_keys_forced": forced,
        "source_files_not_copied": parts["other"],
    }


def _backend_ids(tokenizer_json: Path, text: str) -> list[int]:
    """Ground-truth ids, straight from the backend, bypassing transformers entirely."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_json)).encode(text, add_special_tokens=False).ids


def verify(dest: Path, *, require_chatml: bool = True) -> list[str]:
    """
    Assert the overlay yields the tokenizer it is supposed to. Returns report lines.

    Raises OverlayError on any disagreement -- an overlay that loads but mis-segments is
    strictly worse than a missing one, because it fails silently and expensively.

    `require_chatml` asserts that `<|im_start|>` / `<|im_end|>` are SINGLE ids. That holds
    for the teacher and for the retagged student, and is false BY DESIGN for a pre-retag
    base student: granite-4.1-3b-base has no ChatML control tokens in its vocabulary at
    all (measured directly -- `<|im_start|>` -> [27, 91, 318, 5011, 91, 29] straight
    from the backend), which is the reason retag_student.py exists. Pass False when
    overlaying a base model; leaving it True there reports a defect that is not one.
    """
    from transformers import AutoTokenizer

    lines: list[str] = []

    for name in SIDECARS:
        if (dest / name).exists():
            raise OverlayError(
                f"{dest} contains {name}, which build() excludes. Measured to be inert "
                "under transformers 5.8, so this is a hygiene failure rather than a "
                "mis-segmentation one -- but it means the overlay was not produced by "
                "build(), and its contents are therefore not known."
            )

    tok = AutoTokenizer.from_pretrained(str(dest), local_files_only=True)
    lines.append(f"loaded: {type(tok).__name__}, vocab={len(tok)}")

    # transformers 5.x renamed PreTrainedTokenizerFast -> TokenizersBackend, so the class
    # NAME is not a stable predicate across versions. `is_fast` is, and the round-trip
    # below is what actually matters regardless.
    if not getattr(tok, "is_fast", False):
        raise OverlayError(
            f"{dest} loaded as {type(tok).__name__}, which is not a fast tokenizer, so "
            "it cannot be using tokenizer.json at all. The pre_tokenizer check below "
            "would be meaningless."
        )

    # THE mechanism check, and the one that names the cause rather than the symptom.
    # transformers only reaches a different pre_tokenizer by being told to build a
    # different class, so comparing the resolved pre_tokenizer against the one on disk
    # catches the fault at its source -- including on text no probe happens to cover.
    on_disk = json.loads((dest / "tokenizer.json").read_text()).get("pre_tokenizer")
    resolved = json.loads(tok.backend_tokenizer.to_str()).get("pre_tokenizer")
    if resolved != on_disk:
        raise OverlayError(
            "the pre_tokenizer transformers resolved is NOT the one in tokenizer.json.\n"
            f"  on disk : {json.dumps(on_disk)[:300]}\n"
            f"  resolved: {json.dumps(resolved)[:300]}\n"
            "Almost always `tokenizer_class` in tokenizer_config.json naming a class "
            "that imposes its own pre_tokenizer, or -- if a config.json is present -- the "
            "`model_type` fallback through TOKENIZER_MAPPING_NAMES doing the same even "
            "with that key absent, confirmed by direct measurement. Pin it: see CONFIG_KEYS_FORCED. "
            "Text is being segmented with a pre_tokenizer this "
            "model was never trained with."
        )
    lines.append(f"pre_tokenizer: matches tokenizer.json ({(on_disk or {}).get('type', 'none')})")

    mismatches = []
    for probe in PROBES:
        got = tok(probe, add_special_tokens=False)["input_ids"]
        want = _backend_ids(dest / "tokenizer.json", probe)
        if got != want:
            mismatches.append((probe, want, got))
    if mismatches:
        probe, want, got = mismatches[0]
        raise OverlayError(
            f"{len(mismatches)}/{len(PROBES)} probes disagree between "
            f"AutoTokenizer(overlay) and the raw backend. First: {probe!r}\n"
            f"  backend (ground truth): {want}\n"
            f"  transformers returned : {got}\n"
            "The tokenizer transformers hands the trainer is not the one in "
            "tokenizer.json. Training on this would silently mis-segment every sample."
        )
    lines.append(f"round-trip: {len(PROBES)}/{len(PROBES)} probes match the raw backend")

    # Control tokens must be SINGLE ids -- for a model that is supposed to speak ChatML.
    # If <|im_end|> spells out into several pieces then EOS is unreachable, which is the
    # same class of failure as the retag-v1 blank-EOS row: the model trains but cannot end
    # a turn. See the docstring for why this is opt-out rather than unconditional.
    if not require_chatml:
        lines.append("control tokens: not required (base model overlay)")
        return lines

    for marker in ("<|im_start|>", "<|im_end|>"):
        ids = tok(marker, add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise OverlayError(
                f"{marker!r} encodes to {len(ids)} tokens ({ids}), not 1. It is not a "
                "single special token in this tokenizer, so ChatML turn boundaries -- and "
                "therefore EOS -- are not representable."
            )
        lines.append(f"control token: {marker} -> id {ids[0]}")

    return lines


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="model dir whose tokenizer to overlay")
    p.add_argument("--out", required=True, help="destination overlay dir")
    p.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy",
                   help="hardlink saves space when source and dest share a filesystem")
    p.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True,
                   help="load the overlay and assert it round-trips (needs transformers)")
    # BooleanOptionalAction so a step-template can pass this flag unconditionally -- see
    # steps/distill-hf-export/src/export_hf_model.py for why that matters for jinja.
    p.add_argument("--require-chatml", action=argparse.BooleanOptionalAction, default=True,
                   help="assert <|im_start|>/<|im_end|> are single ids. Pass "
                        "--no-require-chatml when overlaying a PRE-RETAG base model, "
                        "whose vocabulary legitimately lacks them.")
    p.add_argument("--manifest-name", default="overlay_manifest.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source, dest = Path(args.source), Path(args.out)
    try:
        manifest = build(source, dest, copy_mode=args.copy_mode)
        print(f"overlay {source} -> {dest}")
        print(f"  copied  : {manifest['overlay_files']}")
        print(f"  excluded: {manifest['sidecars_excluded'] or 'no sidecars present'}")
        print(f"  forced:   {manifest['config_keys_forced'] or 'config keys already correct'}")
        if args.verify:
            for line in verify(dest, require_chatml=args.require_chatml):
                print(f"  verified: {line}")
            manifest["verified"] = True
        else:
            manifest["verified"] = False
        (dest / args.manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")

        # Name the tokenizer this overlay carries, so downstream steps can compare
        # identities instead of hoping. The identity is the SOURCE directory's basename
        # rather than anything about the overlay itself, because an overlay copies
        # tokenizer.json byte-for-byte -- the teacher's overlay and a student retagged
        # onto that teacher hold the SAME tokenizer and must therefore compare EQUAL.
        # Written in main() and not in build() on purpose: build() is the pure copy whose
        # output file set is asserted exactly, and metadata is the CLI's business.
        tokenizer_identity.write(
            dest, tokenizer_identity.derive_name(source),
            produced_by="distill-tokenizer-align/build_overlay",
            source=str(source.resolve()),
        )
    except OverlayError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    # No LLMB_ARTIFACT_ID line here, deliberately. This module is a helper invoked twice
    # by one step (run-align.sh, for the teacher and student overlays), not a step of its
    # own, and it cannot know which of the step's declared outputs it is producing. It
    # used to print a hardcoded `LLMB_ARTIFACT_ID:overlay`, which meant a single run
    # emitted that same undeclared id TWICE with different paths -- caught by executing
    # the rendered step command, confirmed by direct measurement. The step's launcher owns artifact
    # emission, because only the launcher knows the artifact names it declared.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
