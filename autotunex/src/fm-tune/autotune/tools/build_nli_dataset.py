# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build SFT datasets that teach a model to classify an NLI relationship.

Every record becomes one SFT example whose prompt asks the model to judge the
relationship between a premise (text A) and a hypothesis (text B) and whose target
is the JSON ``{"label": "<entailment|contradiction|neutral>"}`` — the same shape the
eval pipeline parses (``parse_nli_output`` / ``_reference_label``). The prompt block
is imported from ``autotune.tools.chat_utils`` so the training and eval prompts can
never drift. Two output shapes (matching ``build_atomizer_dataset.py``):

- ``formatted``: ``input`` is a string rendered through the model's chat template;
  ``output`` is the label JSON string.
- ``chat``: ``input`` is the raw list of chat messages; ``output`` is the label JSON
  string.

The tool has three modes.

**Mixed build** (``--eli5-train/--eli5-val/--eli5-test``): mixes the raw ELI5 NLI
data with MNLI and SNLI into train and val, and writes the held-out test sets::

    python -m autotune.tools.build_nli_dataset \\
        --eli5-train /path/to/eli5_nli_raw_train.json \\
        --eli5-val /path/to/eli5_nli_raw_val.json \\
        --eli5-test /path/to/eli5_nli_raw_test.json \\
        --output-dir /path/to/out \\
        --model ibm-granite/granite-4.2-8b \\
        --train-mix eli5=180000,mnli=60000,snli=60000 \\
        --val-mix eli5=6000,mnli=3000,snli=3000

============  ==========================  ==========================  ===============================
source        train                       val                         test
============  ==========================  ==========================  ===============================
``eli5``      ``--eli5-train``            ``--eli5-val``              ``--eli5-test`` (balanced head)
``mnli``      ``train``                   ``validation_mismatched``   ``validation_matched``
``snli``      ``train``                   ``validation``              ``test``
``anli``      --                          --                          ``test_r1``, ``test_r2``, ``test_r3``
============  ==========================  ==========================  ===============================

Train and val take a fixed record count per source (``--train-mix``/``--val-mix``),
split evenly across the three labels, so the mix ratio is exactly the counts given.
Pairs are deduped on the stripped ``(premise, hypothesis)``: nothing in any test set
can appear in val or train, nothing in val can appear in train, and no pair appears
twice within a split (ELI5 may already hold an SNLI-train sample).

Test is not mixed. The ELI5 test file is written in full as ``{name}_test``, reordered
so its first ``--test-balanced-head`` rows (default 3000) carry an equal share of each
label — what an eval with ``limit: 3000`` sees — with the remaining rows following in
file order. The standard benchmarks are written separately, in full and in source
order, as ``snli_test``, ``mnli_validation_matched`` and ``anli_test_r{1,2,3}``. Every
test file renders the same prompt through the same template with thinking off, so a
base model and its LoRA adapters are scored on identical inputs.

Outputs, into ``--output-dir`` with prefix ``--name`` (default ``nli_mix``)::

    nli_mix_{split}_raw.json         [{premise, hypothesis, label, source}, ...]
    nli_mix_{split}_formatted.jsonl  {input: <chat-templated str>, output: '{"label": ...}'}
    nli_mix_{split}_chat.jsonl       {input: [messages], output: '{"label": ...}'}
    {benchmark}_{formatted,chat}.jsonl
    nli_mix_manifest.json            settings plus per-split/source/label counts

The per-split files are row-aligned, so ``source`` in the raw file labels each
formatted/chat row. Existing outputs are refused unless ``--overwrite`` is passed.

**Benchmarks only** (``--benchmarks``): rebuilds just the held-out Hugging Face
benchmarks, with ``nli_benchmarks_manifest.json``::

    python -m autotune.tools.build_nli_dataset \\
        --benchmarks snli anli mnli \\
        --output-dir /path/to/benchmarks \\
        --model ibm-granite/granite-4.2-8b

The default ``--benchmark-split auto`` selects ``stanfordnlp/snli`` test,
``facebook/anli`` test_r1/test_r2/test_r3, and ``nyu-mll/multi_nli``
validation_matched (MNLI's test labels are unavailable). ``validation`` instead
selects SNLI validation and ANLI dev_r1/dev_r2/dev_r3. ``--name-tag`` also tags
benchmark files and the manifest. Hub splits are streamed and normalized once in
memory so both output formats share the same rows. Requires the ``datasets``
dependency from the project's ``core`` extra.

**Single file** (``--input``): converts one raw JSON list of ``{premise, hypothesis,
label}`` records (the format emitted by ``collect_nli_dataset.py``) as-is. May
accompany ``--benchmarks``. ``raw_nli.json`` yields ``raw_nli_formatted.jsonl`` and
``raw_nli_chat.jsonl``.

``--format`` defaults to ``both``. ``--model`` is required when rendering the chat
template (``formatted``/``both``) or whenever the output-length gate is active (it
tokenizes the target). The output schema matches ``docs/dataset-sft.md``.

Records whose output sequence (the label JSON) tokenizes to at least
``--max-output-length`` tokens (default 1024) are dropped, matching the length gate
in ``build_atomizer_dataset.py``; ``0`` disables it. The label target is tiny, so the
gate is effectively a no-op here, but it is kept for parity with the other builders.

Granite 4.2 moved to ChatML and defaults to ``enable_thinking=True``, which would end
the prompt inside an open ``<think>`` block and train the label JSON as reasoning
content. Thinking is therefore disabled unless ``--thinking`` is passed; see
``autotune/tools/chat_utils.py``.

For the ``formatted`` pass the tool also prints token-length stats — count/min/max/
mean and p50/p90/p95/p99 — for the input, the output, and the two combined, to help
size ``max_seq_length``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from autotune.tools.chat_utils import (
    IdentityTokenizer,
    get_nli_messages,
    load_tokenizer,
    num_tokens,
    output_name_for,
    print_token_stats,
    render_chat_input,
    resolve_formats,
    summarize,
)

# Canonical NLI labels accepted in the target (must match the eval-side parser).
_CANONICAL_LABELS = ("entailment", "contradiction", "neutral")

# Default seed for mix sampling, so a rebuild reproduces the same dataset.
DEFAULT_SEED = 1234

# HF integer labels have a different order from _CANONICAL_LABELS.
_HF_LABEL_NAMES = ("entailment", "neutral", "contradiction")
_BENCHMARK_SOURCES = {
    "snli": ("stanfordnlp/snli", "plain_text"),
    "anli": ("facebook/anli", "plain_text"),
    "mnli": ("nyu-mll/multi_nli", None),
}

# Mixed build: the sources train/val draw from, and the HF split behind each.
MIX_SOURCES = ("eli5", "mnli", "snli")
_MIX_HF_SPLITS = {
    "mnli": {"train": "train", "val": "validation_mismatched"},
    "snli": {"train": "train", "val": "validation"},
}
DEFAULT_TRAIN_MIX = "eli5=180000,mnli=60000,snli=60000"
DEFAULT_VAL_MIX = "eli5=6000,mnli=3000,snli=3000"
DEFAULT_TEST_BALANCED_HEAD = 3000

Record = Dict[str, str]
PairKey = Tuple[str, str]


@dataclass(frozen=True)
class BenchmarkSplit:
    benchmark: str
    dataset: str
    config: Optional[str]
    split: str
    stem: str


def resolve_benchmark_splits(benchmarks: List[str], selection: str = "auto") -> List[BenchmarkSplit]:
    """Resolve held-out splits explicitly; never substitute an unlabeled test set."""
    if selection not in ("auto", "test", "validation"):
        raise ValueError(f"unknown benchmark split selection: {selection}")
    if "mnli" in benchmarks and selection == "test":
        raise ValueError(
            "MNLI test has no gold labels; use --benchmark-split auto or validation "
            "to build its matched validation benchmark."
        )
    result = []
    for name in dict.fromkeys(benchmarks):
        dataset, config = _BENCHMARK_SOURCES[name]
        if name == "mnli":
            splits = ["validation_matched"]
        elif name == "anli":
            prefix = "dev" if selection == "validation" else "test"
            splits = [f"{prefix}_r{round_id}" for round_id in (1, 2, 3)]
        else:
            splits = ["validation" if selection == "validation" else "test"]
        for split in splits:
            result.append(BenchmarkSplit(name, dataset, config, split, f"{name}_{split}"))
    return result


def _load_benchmark_split(spec: BenchmarkSplit):
    # Lazy import keeps local chat-only conversion usable without the HF stack.
    from datasets import load_dataset

    # Streaming avoids preparing/downloading the unrequested splits. The caller
    # materializes just this split once for both formats.
    return load_dataset(spec.dataset, name=spec.config, split=spec.split, streaming=True)


def normalize_benchmark_records(
    rows: Iterable[Dict[str, Any]], benchmark: str, label_feature=None
) -> tuple[List[Record], Dict[str, Any]]:
    """Normalize labels and validate text, accounting for every rejected row.

    Text is kept verbatim (ANLI premises can end in whitespace): eval files must match
    the source byte for byte. ``pair_key`` strips, so dedupe still sees through it.
    """
    label_names = getattr(label_feature, "names", None)
    if label_names is None:
        label_names = _HF_LABEL_NAMES
    if len(label_names) != 3 or set(label_names) != set(_CANONICAL_LABELS):
        raise ValueError(f"{benchmark}: unsupported label metadata: {label_names!r}")
    records = []
    counts = {"source_rows": 0, "skipped_unlabeled": 0, "skipped_malformed": 0}
    for row in rows:
        counts["source_rows"] += 1
        if not isinstance(row, dict):
            counts["skipped_malformed"] += 1
            continue
        label = row.get("label")
        if label == -1 or label == "unlabeled":
            counts["skipped_unlabeled"] += 1
            continue
        if isinstance(label, Integral) and not isinstance(label, bool) and 0 <= label < len(label_names):
            label = label_names[label]
        canonical = _canonical_label({"label": label})
        premise, hypothesis = row.get("premise"), row.get("hypothesis")
        if (
            canonical is None
            or not isinstance(premise, str)
            or not premise.strip()
            or not isinstance(hypothesis, str)
            or not hypothesis.strip()
        ):
            counts["skipped_malformed"] += 1
            continue
        records.append({"premise": premise, "hypothesis": hypothesis, "label": canonical})
    return records, {
        **counts,
        "valid_rows": len(records),
        "label_counts": dict(collections.Counter(row["label"] for row in records)),
    }


def load_hf_records(spec: BenchmarkSplit) -> tuple[List[Record], Dict[str, Any]]:
    """Load and normalize one HF split; RuntimeError names the split on failure."""
    print(f"[nli/hf] loading {spec.dataset} ({spec.split})")
    try:
        dataset = _load_benchmark_split(spec)
        feature = (getattr(dataset, "features", None) or {}).get("label")
        records, counts = normalize_benchmark_records(dataset, spec.benchmark, feature)
    except Exception as exc:
        raise RuntimeError(f"could not load {spec.dataset} ({spec.split}): {exc}") from exc
    if not records:
        raise ValueError(f"{spec.dataset} ({spec.split}): no valid labeled rows; counts: {counts}")
    print(
        f"[nli/hf] {spec.stem}: {counts['valid_rows']}/{counts['source_rows']} valid rows "
        f"(skipped {counts['skipped_unlabeled']} unlabeled, {counts['skipped_malformed']} malformed); "
        f"labels: {counts['label_counts']}"
    )
    return records, counts


def load_raw_records(path: str, source: str) -> tuple[List[Record], Dict[str, Any]]:
    """Load and normalize a local raw JSON list of ``{premise, hypothesis, label}``."""
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a top-level JSON list")
    records, counts = normalize_benchmark_records(rows, source)
    print(
        f"[nli/raw] {path}: {counts['valid_rows']}/{counts['source_rows']} valid rows "
        f"(skipped {counts['skipped_unlabeled'] + counts['skipped_malformed']}); labels: {counts['label_counts']}"
    )
    return records, counts


def _format_label(label: Any) -> Optional[str]:
    """Render the target JSON for a canonical label, or None if not canonical."""
    if not isinstance(label, str):
        return None
    canonical = label.strip().lower()
    if canonical not in _CANONICAL_LABELS:
        return None
    return json.dumps({"label": canonical})


def _make_record(
    dp: Dict[str, Any],
    tokenizer,
    fmt: str,
    thinking: bool = False,
) -> Optional[Dict[str, Any]]:
    """Convert one raw record to an SFT example, or None if malformed/empty."""
    premise = dp.get("premise")
    hypothesis = dp.get("hypothesis")
    label = dp.get("label")
    if not isinstance(premise, str) or not premise.strip():
        return None
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        return None

    output_seq = _format_label(label)
    if output_seq is None:
        return None

    messages = get_nli_messages(premise, hypothesis)
    if fmt == "formatted":
        return {
            "input": render_chat_input(messages, tokenizer, thinking=thinking),
            "output": output_seq,
        }
    return {"input": messages, "output": output_seq}


def _canonical_label(dp: Any) -> Optional[str]:
    """The record's canonical label, or None if malformed."""
    if not isinstance(dp, dict):
        return None
    label = dp.get("label")
    if not isinstance(label, str):
        return None
    canonical = label.strip().lower()
    return canonical if canonical in _CANONICAL_LABELS else None


def convert_split(
    input_path: str,
    output_path: str,
    fmt: str,
    tokenizer,
    max_output_length: int = 1024,
    thinking: bool = False,
) -> None:
    """Convert a single input JSON file to the target JSONL."""
    with open(input_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{input_path}: expected a top-level JSON list")

    convert_records(records, output_path, fmt, tokenizer, max_output_length=max_output_length, thinking=thinking)


def convert_records(
    records: List[Any],
    output_path: str,
    fmt: str,
    tokenizer,
    max_output_length: int = 1024,
    thinking: bool = False,
    require_nonempty: bool = False,
) -> Dict[str, Any]:
    """Shared writer for every mode; return counts for the manifest.

    Records whose output sequence tokenizes to ``max_output_length`` tokens or more
    are dropped (gate disabled when ``max_output_length <= 0``). For ``formatted``
    a real tokenizer is available, so token-length stats are collected over the
    rendered input, the output, and their sum and printed after the file is written.

    Publish the file only after conversion succeeds, preserving a previous build if
    rendering fails or a benchmark loses all its rows to the length gate.
    """
    collect_stats = fmt == "formatted"
    # The chat path may carry the null-object tokenizer, which cannot count tokens.
    gate_on = max_output_length > 0 and not isinstance(tokenizer, IdentityTokenizer)
    input_lengths: List[int] = []
    output_lengths: List[int] = []
    combined_lengths: List[int] = []

    written = 0
    skipped_malformed = 0
    skipped_length = 0
    label_counts: collections.Counter = collections.Counter()
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=os.path.dirname(os.path.abspath(output_path)), delete=False
        ) as out:
            temporary_path = out.name
            for dp in records:
                record = _make_record(dp, tokenizer, fmt, thinking) if isinstance(dp, dict) else None
                if record is None:
                    skipped_malformed += 1
                    continue

                # Count output tokens once; reused for both the gate and the stats.
                n_out = num_tokens(tokenizer, record["output"]) if (gate_on or collect_stats) else 0
                if gate_on and n_out >= max_output_length:
                    skipped_length += 1
                    continue

                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                label_counts[_canonical_label(dp)] += 1

                if collect_stats:
                    n_in = num_tokens(tokenizer, record["input"])
                    input_lengths.append(n_in)
                    output_lengths.append(n_out)
                    combined_lengths.append(n_in + n_out)
        if require_nonempty and not written:
            raise ValueError(f"{output_path}: no rows remain; check labels and --max-output-length")
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)

    print(
        f"[nli/{fmt}] wrote {written} records to {output_path} "
        f"(skipped {skipped_malformed} malformed, {skipped_length} over-length)"
    )

    if collect_stats:
        print_token_stats(
            [
                summarize("input", input_lengths),
                summarize("output", output_lengths),
                summarize("input+output", combined_lengths),
            ],
            f"nli/{fmt}",
        )

    return {
        "written": written,
        "skipped_malformed": skipped_malformed,
        "skipped_length": skipped_length,
        "label_counts": dict(label_counts),
    }


def write_benchmarks(
    loaded: List[Tuple[BenchmarkSplit, List[Record], Dict[str, Any]]],
    output_dir: str,
    formats: List[str],
    tokenizer,
    *,
    max_output_length: int = 1024,
    thinking: bool = False,
    name_tag: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Write each loaded held-out split; return its manifest entries."""
    name_parts = (name_tag,) if name_tag else ()
    entries = []
    for spec, records, counts in loaded:
        entry = {
            "benchmark": spec.benchmark,
            "dataset": spec.dataset,
            "config": spec.config,
            "split": spec.split,
            **counts,
            "outputs": [],
        }
        for fmt in formats:
            filename = output_name_for(spec.stem, fmt, *name_parts)
            stats = convert_records(
                records,
                os.path.join(output_dir, filename),
                fmt,
                tokenizer,
                max_output_length=max_output_length,
                thinking=thinking,
                require_nonempty=True,
            )
            entry["outputs"].append({"file": filename, "format": fmt, **stats})
        entries.append(entry)
    return entries


def _write_json(path: str, payload: Any, indent: Optional[int] = 2) -> None:
    with open(path, "w", encoding="utf-8") as out:
        json.dump(payload, out, indent=indent, ensure_ascii=False)
        out.write("\n")


def build_benchmarks(
    specs: List[BenchmarkSplit],
    output_dir: str,
    formats: List[str],
    tokenizer,
    *,
    model: Optional[str],
    split_selection: str,
    max_output_length: int = 1024,
    thinking: bool = False,
    name_tag: Optional[str] = None,
) -> None:
    """Build each held-out split independently and write a manifest for this run."""
    loaded = [(spec, *load_hf_records(spec)) for spec in specs]
    entries = write_benchmarks(
        loaded,
        output_dir,
        formats,
        tokenizer,
        max_output_length=max_output_length,
        thinking=thinking,
        name_tag=name_tag,
    )
    manifest = {
        "schema_version": 1,
        "settings": {
            "model": model,
            "formats": formats,
            "benchmark_split": split_selection,
            "thinking": thinking,
            "max_output_length": max_output_length,
            "name_tag": name_tag,
        },
        "benchmarks": entries,
    }
    manifest_name = "nli_benchmarks_manifest" + (f"_{name_tag}" if name_tag else "") + ".json"
    manifest_path = os.path.join(output_dir, manifest_name)
    _write_json(manifest_path, manifest)
    print(f"[nli/benchmark] wrote manifest to {manifest_path}")


# =============================================================================
# Mixed build
# =============================================================================


def parse_mix(spec: str) -> Dict[str, int]:
    """Parse ``source=count,...`` into ``{source: count}``; ValueError on a bad spec."""
    mix: Dict[str, int] = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        source, sep, raw = part.partition("=")
        source = source.strip()
        if not sep or source not in MIX_SOURCES:
            raise ValueError(f"bad mix entry {part!r}: expected SOURCE=COUNT, SOURCE in {MIX_SOURCES}")
        if source in mix:
            raise ValueError(f"bad mix entry {part!r}: {source} given twice")
        try:
            count = int(raw)
        except ValueError:
            raise ValueError(f"bad mix entry {part!r}: {raw!r} is not an integer") from None
        if count < 0 or count % len(_CANONICAL_LABELS):
            # An even per-label split is what keeps each source balanced.
            raise ValueError(f"bad mix entry {part!r}: count must be >= 0 and divisible by {len(_CANONICAL_LABELS)}")
        mix[source] = count
    return mix


def pair_key(rec: Record) -> PairKey:
    return rec["premise"].strip(), rec["hypothesis"].strip()


def sample_balanced(
    pool: List[Record], count: int, exclude: Set[PairKey], seed: int, source: str
) -> Tuple[List[Record], int]:
    """Draw ``count`` records, ``count / 3`` per label, skipping pairs in ``exclude``.

    Duplicate pairs within ``pool`` are collapsed first (first occurrence wins). Seeded
    per ``(source, label)`` so changing one source's count does not reshuffle another.
    Raises ValueError if a label cannot supply its share — silently taking fewer would
    change the mix. Returns ``(sampled, n_excluded)``.
    """
    per_label = count // len(_CANONICAL_LABELS)
    by_label: Dict[str, List[Record]] = collections.defaultdict(list)
    seen: Set[PairKey] = set()
    excluded = 0
    for rec in pool:
        key = pair_key(rec)
        if key in exclude:
            excluded += 1
            continue
        if key in seen:
            continue
        seen.add(key)
        by_label[rec["label"]].append(rec)

    sampled: List[Record] = []
    for label in _CANONICAL_LABELS:
        candidates = by_label[label]
        if len(candidates) < per_label:
            raise ValueError(f"{source}: only {len(candidates)} usable {label!r} records, {per_label} requested")
        sampled.extend(random.Random(f"{seed}:{source}:{label}").sample(candidates, per_label))
    return sampled, excluded


def balance_head(records: List[Record], head: int, seed: int) -> List[Record]:
    """Reorder so the first ``head`` rows hold ``head / 3`` of each label.

    The head is a seeded sample per label, shuffled together; the rest keep their file
    order. Raises ValueError if a label cannot fill its share.
    """
    if head <= 0:
        return list(records)
    per_label = head // len(_CANONICAL_LABELS)
    by_label: Dict[str, List[int]] = collections.defaultdict(list)
    for idx, rec in enumerate(records):
        by_label[rec["label"]].append(idx)
    chosen: List[int] = []
    for label in _CANONICAL_LABELS:
        if len(by_label[label]) < per_label:
            raise ValueError(
                f"test: only {len(by_label[label])} {label!r} records, {per_label} needed for a balanced head of {head}"
            )
        chosen.extend(random.Random(f"{seed}:test-head:{label}").sample(by_label[label], per_label))
    random.Random(f"{seed}:test-head").shuffle(chosen)
    picked = set(chosen)
    return [records[i] for i in chosen] + [rec for i, rec in enumerate(records) if i not in picked]


def _source_label_counts(records: Iterable[Record]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for rec in records:
        out.setdefault(rec["source"], dict.fromkeys(_CANONICAL_LABELS, 0))[rec["label"]] += 1
    return out


def _gated_labels(tokenizer, max_output_length: int) -> Set[str]:
    """Labels whose target the length gate would drop.

    Applied before sampling, so ``convert_records`` never drops a mixed row and the raw,
    formatted and chat files stay row-aligned.
    """
    if max_output_length <= 0 or isinstance(tokenizer, IdentityTokenizer):
        return set()
    return {
        label for label in _CANONICAL_LABELS if num_tokens(tokenizer, json.dumps({"label": label})) >= max_output_length
    }


def mix_output_paths(output_dir: str, name: str, formats: List[str], specs: List[BenchmarkSplit]) -> List[str]:
    """Every file a mixed build writes, so existing ones can be refused up front."""
    names = [f"{name}_manifest.json"]
    for split in ("train", "val", "test"):
        names.append(f"{name}_{split}_raw.json")
        names.extend(output_name_for(f"{name}_{split}", fmt) for fmt in formats)
    for spec in specs:
        names.extend(output_name_for(spec.stem, fmt) for fmt in formats)
    return [os.path.join(output_dir, n) for n in names]


def build_mix(
    eli5_paths: Dict[str, str],
    output_dir: str,
    formats: List[str],
    tokenizer,
    *,
    model: Optional[str],
    name: str,
    mixes: Dict[str, Dict[str, int]],
    test_balanced_head: int,
    seed: int = DEFAULT_SEED,
    max_output_length: int = 1024,
    thinking: bool = False,
) -> Dict[str, Any]:
    """Build the mixed train/val, the balanced ELI5 test, and the benchmark tests."""
    gated = _gated_labels(tokenizer, max_output_length)
    if gated:
        raise ValueError(f"--max-output-length {max_output_length} would drop every {sorted(gated)} target")

    # Test first: it is fixed, and its pairs are what train and val must avoid.
    print("[nli/mix] test")
    eli5_test, eli5_test_counts = load_raw_records(eli5_paths["test"], "eli5")
    test = balance_head([{**rec, "source": "eli5"} for rec in eli5_test], test_balanced_head, seed)
    benchmark_specs = resolve_benchmark_splits(["snli", "mnli", "anli"])
    benchmarks = [(spec, *load_hf_records(spec)) for spec in benchmark_specs]
    taken: Set[PairKey] = {pair_key(r) for r in test}
    for _, records, _ in benchmarks:
        taken.update(pair_key(r) for r in records)

    def load_pool(source: str, split: str) -> List[Record]:
        if source == "eli5":
            records, _ = load_raw_records(eli5_paths[split], "eli5")
        else:
            dataset, config = _BENCHMARK_SOURCES[source]
            hf_split = _MIX_HF_SPLITS[source][split]
            spec = BenchmarkSplit(source, dataset, config, hf_split, f"{source}_{hf_split}")
            records, _ = load_hf_records(spec)
        return [{**rec, "source": source} for rec in records]

    splits: Dict[str, List[Record]] = {"test": test}
    excluded: Dict[str, Dict[str, int]] = {}
    for split in ("val", "train"):
        print(f"[nli/mix] {split} {mixes[split]}")
        records: List[Record] = []
        excluded[split] = {}
        for source, count in mixes[split].items():
            if not count:
                continue
            sampled, n_excl = sample_balanced(load_pool(source, split), count, taken, seed, source)
            # Sources are sampled in turn, so a later source also skips an earlier one's pairs.
            taken.update(pair_key(r) for r in sampled)
            excluded[split][source] = n_excl
            records.extend(sampled)
        random.Random(f"{seed}:{split}").shuffle(records)
        splits[split] = records

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "settings": {
            "model": model,
            "formats": formats,
            "thinking": thinking,
            "max_output_length": max_output_length,
            "seed": seed,
            "name": name,
            "eli5": eli5_paths,
            "train_mix": mixes["train"],
            "val_mix": mixes["val"],
            "test_balanced_head": test_balanced_head,
        },
        "splits": {},
        "benchmarks": [],
    }
    for split in ("train", "val", "test"):
        print(f"[nli/mix] writing {split}")
        stem = f"{name}_{split}"
        _write_json(os.path.join(output_dir, f"{stem}_raw.json"), splits[split], indent=1)
        outputs = []
        for fmt in formats:
            filename = output_name_for(stem, fmt)
            stats = convert_records(
                splits[split],
                os.path.join(output_dir, filename),
                fmt,
                tokenizer,
                max_output_length=max_output_length,
                thinking=thinking,
                require_nonempty=True,
            )
            if stats["written"] != len(splits[split]):
                raise RuntimeError(f"{filename}: wrote {stats['written']} of {len(splits[split])} rows")
            outputs.append({"file": filename, "format": fmt, **stats})
        entry: Dict[str, Any] = {
            "total": len(splits[split]),
            "counts": _source_label_counts(splits[split]),
            "excluded_overlap": excluded.get(split, {}),
            "outputs": outputs,
        }
        if split == "test":
            entry["source_rows"] = eli5_test_counts
            entry["head"] = {
                "rows": test_balanced_head,
                "label_counts": dict(collections.Counter(r["label"] for r in test[:test_balanced_head])),
            }
        manifest["splits"][split] = entry

    print("[nli/mix] writing benchmark tests")
    manifest["benchmarks"] = write_benchmarks(
        benchmarks, output_dir, formats, tokenizer, max_output_length=max_output_length, thinking=thinking
    )
    manifest_path = os.path.join(output_dir, f"{name}_manifest.json")
    _write_json(manifest_path, manifest)
    print(f"[nli/mix] manifest -> {manifest_path}")
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build_nli_dataset",
        description="Build an SFT dataset for classifying NLI (entailment/contradiction/neutral).",
    )
    mix = p.add_argument_group("mixed build (ELI5 + MNLI + SNLI)")
    for split in ("train", "val", "test"):
        mix.add_argument(
            f"--eli5-{split}",
            help=f"Raw ELI5 NLI {split} JSON (top-level list of {{premise, hypothesis, label}}). "
            "All three --eli5-* files select the mixed build.",
        )
    mix.add_argument(
        "--train-mix",
        default=DEFAULT_TRAIN_MIX,
        help=f"Records per source in train, each divisible by 3 (default: {DEFAULT_TRAIN_MIX}).",
    )
    mix.add_argument(
        "--val-mix",
        default=DEFAULT_VAL_MIX,
        help=f"Records per source in val, each divisible by 3 (default: {DEFAULT_VAL_MIX}).",
    )
    mix.add_argument(
        "--test-balanced-head",
        type=int,
        default=DEFAULT_TEST_BALANCED_HEAD,
        help="Put this many label-balanced ELI5 rows first in the test split, divisible by 3 "
        f"(default: {DEFAULT_TEST_BALANCED_HEAD}; 0 keeps file order).",
    )
    mix.add_argument("--name", default="nli_mix", help="Output file prefix (default: nli_mix).")
    mix.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Sampling seed (default: {DEFAULT_SEED}).")
    mix.add_argument("--overwrite", action="store_true", help="Replace existing output files.")

    p.add_argument(
        "--input",
        help="Single-file mode: input JSON file (top-level list of {premise, hypothesis, label} "
        "records, e.g. the output of collect_nli_dataset.py), converted as-is.",
    )
    p.add_argument(
        "--benchmarks",
        nargs="+",
        choices=list(_BENCHMARK_SOURCES),
        help="Build held-out Hugging Face benchmarks: snli (stanfordnlp/snli), anli "
        "(facebook/anli, rounds 1-3), mnli (nyu-mll/multi_nli, matched). May accompany --input.",
    )
    p.add_argument(
        "--benchmark-split",
        choices=["auto", "test", "validation"],
        default="auto",
        help="Benchmark split selection. auto (default): SNLI test, ANLI test_r1/r2/r3, "
        "MNLI validation_matched. validation selects SNLI validation and ANLI dev_r1/r2/r3. "
        "Explicit test is invalid for MNLI because its test split is unlabeled.",
    )
    p.add_argument("--output-dir", required=True, help="Directory to write the output into.")
    p.add_argument(
        "--format",
        dest="fmt",
        default="both",
        choices=["formatted", "chat", "both"],
        help="'formatted' renders input via the chat template; 'chat' keeps the raw message list; "
        "'both' writes one file of each (default: both).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Model name or path for the tokenizer / chat template. Required for --format "
        "formatted/both or when --max-output-length > 0.",
    )
    p.add_argument(
        "--max-output-length",
        type=int,
        default=1024,
        help="Drop records whose output (label JSON) tokenizes to this many tokens or more. "
        "Set to 0 to disable the gate (default: 1024).",
    )
    p.add_argument(
        "--thinking",
        action="store_true",
        help="Enable the template's thinking mode (Granite 4.2). Off by default so the "
        "label JSON is not trained inside a <think> block.",
    )
    p.add_argument(
        "--name-tag",
        default=None,
        help="Extra tag inserted into --input/--benchmarks output filenames, e.g. --name-tag v2 "
        "turns nli_raw_formatted.jsonl into nli_raw_v2_formatted.jsonl. Also tags the "
        "benchmark manifest. Use --name for the mixed build.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    eli5_paths = {split: getattr(args, f"eli5_{split}") for split in ("train", "val", "test")}
    mixed = any(eli5_paths.values())
    if mixed:
        if not all(eli5_paths.values()):
            parser.error("the mixed build needs all of --eli5-train, --eli5-val and --eli5-test")
        if args.input or args.benchmarks or args.name_tag:
            parser.error("--input, --benchmarks and --name-tag do not apply to the mixed build")
        if args.test_balanced_head < 0 or args.test_balanced_head % len(_CANONICAL_LABELS):
            parser.error("--test-balanced-head must be >= 0 and divisible by 3")
        try:
            mixes = {"train": parse_mix(args.train_mix), "val": parse_mix(args.val_mix)}
        except ValueError as exc:
            parser.error(str(exc))
        missing = [path for path in eli5_paths.values() if not os.path.exists(path)]
        if missing:
            print(f"[error] not found: {', '.join(missing)}", file=sys.stderr)
            return 1
        benchmark_specs = resolve_benchmark_splits(["snli", "mnli", "anli"])
    else:
        if not args.input and not args.benchmarks:
            parser.error("one of --eli5-train/--eli5-val/--eli5-test, --input or --benchmarks is required")
        if args.benchmark_split != "auto" and not args.benchmarks:
            parser.error("--benchmark-split requires --benchmarks")
        try:
            benchmark_specs = resolve_benchmark_splits(args.benchmarks or [], args.benchmark_split)
        except ValueError as exc:
            parser.error(str(exc))
        if args.input and not os.path.exists(args.input):
            print(f"[error] {args.input} not found", file=sys.stderr)
            return 1
        if args.input and any(
            output_name_for(args.input, "chat") == output_name_for(spec.stem, "chat") for spec in benchmark_specs
        ):
            parser.error("--input filename collides with a benchmark output; rename the local input file")

    formats = resolve_formats(args.fmt)
    if mixed and not args.overwrite:
        outputs = mix_output_paths(args.output_dir, args.name, formats, benchmark_specs)
        existing = [p for p in outputs if os.path.exists(p)]
        if existing:
            print(
                f"[error] {len(existing)} output file(s) already exist in {args.output_dir} "
                f"(e.g. {os.path.basename(existing[0])}); pass --overwrite or a different --output-dir/--name",
                file=sys.stderr,
            )
            return 1
    os.makedirs(args.output_dir, exist_ok=True)

    # A real tokenizer is needed to render the chat template and to count output
    # tokens for the length gate.
    needs_tokenizer = "formatted" in formats or args.max_output_length > 0
    tokenizer = load_tokenizer(
        args.model,
        needs_tokenizer,
        "for --format formatted/both or when --max-output-length > 0 "
        "(used for chat template rendering and/or output-length gating)",
    )

    if mixed:
        try:
            build_mix(
                eli5_paths,
                args.output_dir,
                formats,
                tokenizer,
                model=args.model,
                name=args.name,
                mixes=mixes,
                test_balanced_head=args.test_balanced_head,
                seed=args.seed,
                max_output_length=args.max_output_length,
                thinking=args.thinking,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
        return 0

    name_parts = (args.name_tag,) if args.name_tag else ()
    if args.input:
        for fmt in formats:
            convert_split(
                input_path=args.input,
                output_path=os.path.join(args.output_dir, output_name_for(args.input, fmt, *name_parts)),
                fmt=fmt,
                tokenizer=tokenizer,
                max_output_length=args.max_output_length,
                thinking=args.thinking,
            )
    if benchmark_specs:
        try:
            build_benchmarks(
                benchmark_specs,
                args.output_dir,
                formats,
                tokenizer,
                model=args.model,
                split_selection=args.benchmark_split,
                max_output_length=args.max_output_length,
                thinking=args.thinking,
                name_tag=args.name_tag,
            )
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
