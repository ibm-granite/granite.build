#!/usr/bin/env python3
"""Record WHICH ROWS a training run actually had, as opposed to which rows prep emitted.

WHY THIS IS A SEPARATE QUESTION FROM THE CORPUS MANIFEST.

`distill-corpus-prep` writes `corpus_manifest.json` plus a per-row `corpus_rows.jsonl`, so
the corpus can say exactly which source records it kept, which it dropped, and why. That is
necessary and it is not sufficient, because the TRAINER filters again -- and it filters
differently depending on the arm:

    custom_gold_trainer.py:1824
        if args.max_length is None or args.max_completion_length is None or args.lmbda == 0.0:
            dataset = dataset.filter(lambda x: x.get("prompts") is not None)
        else:
            dataset = dataset.filter(lambda x: x.get("prompts") is not None
                                     and len(x["prompts"]) < args.max_length - args.max_completion_length)

So the ON-POLICY arm additionally discards every row whose rendered prompt reaches
`max_length - max_completion_length` (12,288 tokens for this deliverable), and the
OFF-POLICY arm keeps them. "Which data point was used during training" is therefore a
per-stage fact, and no amount of care at prep time answers it for both stages at once.

WHY NOT PREDICT THE FILTER FROM PREP INSTEAD.

Because predicting it means re-implementing the trainer's prompt render somewhere else --
`apply_chat_template(messages[:-1], add_generation_prompt=True, enable_thinking=False,
documents=..., tools=...)`, with the trainer's tokenizer copy and the trainer's template --
and then keeping that second implementation in step with the first forever. A provenance
record that has quietly drifted is worse than no record, because it still reads as
authoritative. This module instead asks the dataset the trainer actually built.

HOW THE ROWS ARE IDENTIFIED.

The source corpus has no id field, so prep manufactures one (`prep_corpus.row_id`: blake2b-128
over canonical JSON) and, with `--emit-row-id`, writes it INTO each emitted row. The column is
inert for rendering -- the trainer reads only messages/tools/documents/chat_template_kwargs --
and survives the tokenize map, which sets no `remove_columns`. So the id rides along and the
post-filter dataset can be asked for it directly.

WHAT "CONSUMED" HONESTLY MEANS HERE, AND WHAT IT DOES NOT.

This runs after the dataset is built and before `train()`, so what it records is the set of
rows the run HAD AVAILABLE for a full epoch -- every row the trainer would draw from. Two
things make that an upper bound on what a given run actually saw, and both are reported as
numbers rather than left for a reader to discover:

  * `max_steps` short of an epoch. The deliverable's own probes run 24 steps against a
    2,000-row corpus. `rows_scheduled` below states the arithmetic.
  * Preemption. On `preemptable` a requeue RESTARTS the job, so a preempted run's true
    consumption is the union over attempts, which no single attempt can know.

Recording the available set exactly and naming the gap is the honest version. Claiming a
per-step consumption log this does not have would not be.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Must agree with prep_corpus.ROW_ID_FIELD. Not imported from it: prep is a step with its own
# dependency closure and is not on the trainer's path, so a hard import would couple the
# training job to the prep step's installability. The name is asserted in the tests instead.
ROW_ID_FIELD = "row_id"
CONSUMED_NAME = "train_rows_available.jsonl"
MANIFEST_NAME = "train_manifest.json"


class ManifestDrift(Exception):
    """prep and the trainer disagree about the corpus. Raised BEFORE training starts."""


def _prep_manifest(dataset_path: str | None) -> tuple[dict | None, str | None]:
    """Prep's manifest, if the dataset came from a prepped corpus.

    Returns (manifest, path). Both None when the dataset is a raw hand-cut file, which is the
    case for the smoke and probe slices and NOT for the deliverable: measured 2026-08-27 across
    every dataset path any gold config names, the prepped corpus
    (`prepped/en-sft-4.1-0.2-16K-v2/merged/train.jsonl`, used by the deliverable and warmup
    geometries) carries a `corpus_manifest.json` beside it, and the three `work/{smoke,probe}/`
    slices do not.

    Which way round that is matters more than it looks. On a smoke run both-None is ordinary. On
    the deliverable it would mean the manifest lost prep's record of WHICH rows were used -- the
    thing the manifest exists for -- so it is a finding there, not a default. An earlier version
    of this docstring said absent lineage was "the expected case" for every gold config; that
    was true when only raw slices existed and became inverted, silently, the day the prepped
    corpus landed.

    A manifest that exists but does not parse returns (None, path) rather than (None, None), so
    an unreadable file cannot pass as an absent one.
    """
    if not dataset_path:
        return None, None
    p = Path(dataset_path)
    for cand in (p.parent / "corpus_manifest.json", p / "corpus_manifest.json"):
        if cand.is_file():
            try:
                return json.loads(cand.read_text()), str(cand)
            except (OSError, json.JSONDecodeError):
                return None, str(cand)
    return None, None


def rows_scheduled(*, max_steps: int, num_train_epochs: float, n_available: int,
                   world_size: int, per_device_batch: int, grad_accum: int) -> dict[str, Any]:
    """How many row-visits the schedule asks for, against how many rows exist.

    Kept as arithmetic on named inputs rather than read off the trainer, so it is testable
    and so the reader can see WHICH numbers it is a function of. `visits` can exceed
    `n_available` (more than one epoch) or fall short of it (max_steps short of an epoch);
    both are ordinary, and the ratio is what says which.
    """
    rows_per_step = max(1, world_size) * max(1, per_device_batch) * max(1, grad_accum)
    if max_steps and max_steps > 0:
        visits = max_steps * rows_per_step
        basis = f"max_steps {max_steps} x {rows_per_step} rows/step"
    else:
        visits = int(round(num_train_epochs * n_available))
        basis = f"num_train_epochs {num_train_epochs} x {n_available} rows"
    return {
        "rows_per_step": rows_per_step,
        "row_visits_scheduled": visits,
        "basis": basis,
        # Below 1.0 the run does not finish an epoch, so the available set OVERSTATES what was
        # seen and by roughly this factor. Above 1.0 rows repeat. Either way the number is
        # stated rather than implied.
        "epoch_fraction_scheduled": round(visits / n_available, 4) if n_available else None,
    }


def collect_ids(dataset) -> tuple[list[str] | None, str]:
    """Every row id in the trainer's post-filter dataset, or None with the reason why not.

    `dataset.column_names` is checked before touching the column so that a corpus prepped
    without --emit-row-id degrades to a named explanation instead of a KeyError inside a
    training job that has already loaded a 30B teacher.
    """
    if dataset is None:
        return None, "no train_dataset on the trainer"
    cols = getattr(dataset, "column_names", None)
    if cols is None:
        return None, f"dataset of type {type(dataset).__name__} exposes no column_names"
    if ROW_ID_FIELD not in cols:
        return None, (f"the corpus carries no {ROW_ID_FIELD!r} column, so rows have no "
                      "identity to report -- re-run distill-corpus-prep with --emit-row-id")
    ids = list(dataset[ROW_ID_FIELD])
    if any(not isinstance(i, str) or not i for i in ids):
        return None, f"the {ROW_ID_FIELD!r} column holds non-string or empty values"
    return ids, ""


def build(*, dataset, output_dir: str, dataset_path: str | None, lmbda: float,
          max_length: int | None, max_completion_length: int | None, max_steps: int,
          num_train_epochs: float, world_size: int, per_device_batch: int,
          grad_accum: int) -> dict[str, Any]:
    """Write the manifest (and the id list, when there is one). Returns the manifest.

    Raises ManifestDrift only for a disagreement that means the trainer is NOT training on the
    corpus prep described -- see the checks at the end. A missing id column is not drift, it is
    a corpus built before the stamp existed, and it warns.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids, why_not = collect_ids(dataset)
    n_available = len(ids) if ids is not None else (len(dataset) if dataset is not None else 0)
    prep, prep_path = _prep_manifest(dataset_path)

    # The arm, spelled out, because it is the reason the two counts can differ at all.
    onpolicy = float(lmbda) != 0.0
    prompt_budget = (max_length - max_completion_length
                     if onpolicy and max_length and max_completion_length else None)

    manifest: dict[str, Any] = {
        "dataset_path": dataset_path,
        "arm": "onpolicy" if onpolicy else "offpolicy",
        "lmbda": lmbda,
        "rows_available": n_available,
        "id_field": ROW_ID_FIELD if ids is not None else None,
        "ids_unavailable_reason": why_not or None,
        "trainer_row_filter": {
            # Quoted rather than paraphrased: the filter is upstream code we do not own, and a
            # paraphrase is what goes stale when it changes.
            "site": "custom_gold_trainer.py:1824",
            "drops_prompts_at_or_above": prompt_budget,
            "note": ("off-policy keeps long prompts; on-policy drops them. The same corpus "
                     "therefore yields different training sets in the two stages."),
        },
        "schedule": rows_scheduled(max_steps=max_steps, num_train_epochs=num_train_epochs,
                                   n_available=n_available, world_size=world_size,
                                   per_device_batch=per_device_batch, grad_accum=grad_accum),
        "consumption_caveat": (
            "rows_available is every row the run could draw from for a full epoch. It is an "
            "UPPER BOUND on what this attempt saw: see schedule.epoch_fraction_scheduled, and "
            "note that preemption on this queue restarts rather than resumes, so a preempted "
            "run's true consumption is the union over attempts."),
    }

    if prep is not None:
        expected = (prep.get("rows") or {}).get("training_rows")
        manifest["prep"] = {
            "manifest": prep_path,
            "training_rows": expected,
            "row_id_scheme": (prep.get("rows") or {}).get("row_id_scheme"),
            "documents_policy": (prep.get("policies") or {}).get("documents_policy"),
        }
        if isinstance(expected, int) and expected > 0:
            dropped = expected - n_available
            manifest["prep"]["dropped_by_trainer"] = dropped
            # A count ABOVE prep's train split is not a filter result, it is a different
            # dataset -- the one mismatch that means the run is not training on what prep
            # described, so it stops the job while stopping is still cheap.
            if n_available > expected:
                raise ManifestDrift(
                    f"the trainer's dataset has {n_available} rows but prep's manifest at "
                    f"{prep_path} reports {expected} training rows. Filtering can only "
                    "REMOVE rows, so this is not a filter result -- output_dir or "
                    "dataset_name points at a corpus other than the one prep built.")
            # Off-policy filters on `prompts is not None` alone, and every render path in the
            # trainer either sets `prompts` or raises, so the count must match exactly. A gap
            # here means the trainer and prep disagree about renderability -- a template or
            # tokenizer difference between the two steps, which is the drift this catches.
            if not onpolicy and dropped:
                raise ManifestDrift(
                    f"off-policy, but the trainer kept {n_available} of prep's {expected} "
                    f"training rows ({dropped} fewer). The off-policy filter only drops rows "
                    "whose tokenization produced no prompt, and every render path in "
                    "custom_gold_trainer.py raises rather than returning None -- so prep and "
                    "the trainer are not rendering this corpus the same way. Compare their "
                    "chat templates and tokenizer directories before training on it.")
    else:
        manifest["prep"] = {
            "manifest": None,
            "note": ("the dataset is not a prepped corpus (no corpus_manifest.json beside "
                     "it), so there is nothing to cross-check and the `documents` policy and "
                     "row ids of this run are unknown."),
        }

    if ids is not None:
        cpath = out / CONSUMED_NAME
        # One id per line rather than a JSON array: 811k ids is ~27 MB and a reader wants to
        # grep it or stream it, not parse it whole.
        cpath.write_text("".join(f"{i}\n" for i in ids))
        manifest["rows_file"] = {"path": str(cpath.resolve()), "entries": len(ids)}
        uniq = len(set(ids))
        if uniq != len(ids):
            # Duplicates are not fatal -- a corpus may legitimately contain the same
            # conversation twice -- but they mean the id is not a key, so anyone joining on it
            # has to know. Counted rather than assumed away.
            manifest["rows_file"]["distinct"] = uniq

    (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def build_from_trainer(trainer, *, dataset_path: str | None) -> dict[str, Any] | None:
    """Rank-0-only wrapper that reads what it needs off the trainer. Returns None elsewhere.

    Every rank builds the identical dataset, so having all of them write the same file would
    only race. `is_main_process` is asked of the accelerator when there is one and falls back
    to RANK, so this stays correct under a plain single-process run too.
    """
    acc = getattr(trainer, "accelerator", None)
    if acc is not None:
        if not acc.is_main_process:
            return None
        world = getattr(acc, "num_processes", 1) or 1
    else:
        if os.environ.get("RANK", "0") != "0":
            return None
        world = int(os.environ.get("WORLD_SIZE", "1") or 1)

    a = trainer.args
    return build(
        dataset=getattr(trainer, "train_dataset", None),
        output_dir=a.output_dir,
        dataset_path=dataset_path,
        lmbda=getattr(a, "lmbda", 0.0),
        max_length=getattr(a, "max_length", None),
        max_completion_length=getattr(a, "max_completion_length", None),
        max_steps=getattr(a, "max_steps", 0) or 0,
        num_train_epochs=getattr(a, "num_train_epochs", 1.0) or 1.0,
        world_size=world,
        per_device_batch=getattr(a, "per_device_train_batch_size", 1) or 1,
        grad_accum=getattr(a, "gradient_accumulation_steps", 1) or 1,
    )
