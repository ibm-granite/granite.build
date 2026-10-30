"""OfficeQA (Databricks), metric "accuracy@0%".

Upstream (github.com/databricks/OfficeQA, :data:`UPSTREAM_COMMIT`) asks
quantitative questions about U.S. Treasury Bulletins -- numbers, percentages,
dates and short lists buried in decades of financial tables -- and scores the
model's single-value answer with a fuzzy matcher (:data:`REWARD_SHA256` port
below) at numeric tolerances. Two question sets ship on the Hub
(:data:`DATASET` at :data:`REVISION`): ``officeqa_pro.csv``
(:data:`PRO_N` questions) and ``officeqa_full.csv`` (:data:`FULL_N`).

This family implements ``oracle_parsed`` (the benchmark's headline mode, the
one upstream's own runs and the benchmark's reference harness report): each question's gold
pages -- parsed from the ``source_docs`` URLs (``page=N``) zipped with
``source_files`` -- are extracted from the parsed JSON corpus
(``treasury_bulletins_parsed/jsons/*.json``, :data:`N_JSONS` documents) and
handed to the model as context, exactly as the reference harness's
index builder extracts them: elements grouped by their ``bbox[0].page_id``, tables rendered
to Markdown by ``pandas.read_html``. The model answers in
``<FINAL_ANSWER>`` tags (upstream's system prompt, :data:`SYSTEM_PROMPT`,
kept verbatim); the headline is the mean over questions of the reward at
tolerance 0 (``accuracy@0%``), with 0.1% / 1% / 5% in ``details.accuracy``.

LICENSE: the code and the parsed-corpus scripts are Apache-2.0, so the reward
function and the page extraction are ported into this file with attribution.
The question CSVs and the corpus are CC-BY-SA-4.0 behind HF gating: nothing is
vendored, baked into an image, mirrored or pushed. The data is fetched at run
start -- before any question is processed -- from the pinned revision and
verified (git-blob sha1 pins on the CSVs plus a sha256 over the corpus tree's
``path blob-sha1`` lines, and the question counts); a fetch that does not
match refuses to run. ``HF_TOKEN`` must belong to an account that has clicked
through the dataset's gate, which asks that the answer keys not be used to
train models evaluated on OfficeQA or to inflate benchmark scores. Point
``GRANITE_EVALS_OFFICEQA_DIR`` at an existing dataset checkout, or pass
``--dataset <dir>``, to use one fetched by other means.

What runs, all pinned:

- Data: :data:`DATASET` at :data:`REVISION` -- the two CSVs and
  ``treasury_bulletins_parsed/jsons/*.json`` only (the ~4 GB of PDFs and the
  convenience zips are excluded by ``allow_patterns``), verified against
  :data:`PRO_CSV_SHA1` / :data:`FULL_CSV_SHA1` (git-blob sha1), the question
  counts and :data:`JSONS_SHA256`.
- Model: the served checkpoint over an OpenAI-compatible ``/chat/completions``
  (core ``httpx``), with upstream's payload (``model``, ``messages``,
  ``max_tokens``, ``temperature``) and system prompt.
- Score: the reward.py port below (stdlib ``re`` only), at
  :data:`TOLERANCES`.

Deviations from the reference harness (also in ``details.deviations``):

- The gold-page context is extracted per question, on demand, straight from
  the parsed JSONs (the same extraction as the reference harness's index builder)
  instead of through a prebuilt BM25-index directory; the index the reference
  harness's oracle mode passes around is vestigial and is dropped.
- A synchronous httpx client with bounded retries (timeout / 5xx / 429)
  instead of the reference harness's asyncio semaphore fan-out; same endpoint, same payload.
- ``answers=gold`` (no model): scores each question's gold answer through
  the reward port with the context still built -- a model-free check of the
  fetch, the pins, the page extraction and the scorer. Upstream has no
  equivalent; a gold answer the scorer rejects is a finding about the scorer
  or the CSV, not a baseline.
- ``max_context_chars`` (default 0 = off) truncates an over-long context;
  upstream has none, and a truncated question is flagged in its record.
- An infrastructure failure (fetch, endpoint) is not persisted and is
  retried on resume (registry.failure_policy), where upstream records the
  question as failed.
- Only ``oracle_parsed`` is implemented; ``oracle_full_doc``, retrieval
  (bm25 / vector / hybrid), agent and PDF modes are refused with an error
  naming what landed instead.

Layout: ``<output_dir>/repeat-<k>/<uid>.json`` holds one record per question
(generation, extracted answer, per-tolerance scores), skipped on restart.
Splittable: ``--phase generate`` needs the model and builds the contexts,
``--phase score`` only re-scores saved generations on CPU (it still fetches
the CSVs -- questions and gold answers -- but not the corpus).

Options: ``mode=oracle_parsed`` (the only value), ``answers=model|gold``,
``questions`` (regex on uid), ``temperature=0.0``, ``max_tokens=4096``,
``max_context_chars=0``, ``timeout=120``, ``max_failed_frac=0.05``.
"""

from __future__ import annotations

import concurrent.futures
import csv
import functools
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from granite_evals.registry import Benchmark, failure_policy, pass_at_k, register

log = logging.getLogger(__name__)

# -- upstream pins -------------------------------------------------------------
#
# The Hub repo is gated (CC-BY-SA-4.0 data; click-through). The enforced pins:
# the revision, a git-blob sha1 per CSV (the benchmark itself, so a swapped
# file can never silently change the score), the question counts, and a sha256
# over the corpus tree's sorted "path blob-sha1" lines (a content pin: any
# added, removed or edited json breaks it). This token cannot download the
# gated files, so the pins were taken from the Hub's public tree metadata,
# where a non-LFS file's oid IS its git-blob sha1.

DATASET = "databricks/OfficeQA"
REVISION = "763a8366abf2a3605c381d53586d844dc60fa756"

UPSTREAM_REPO = "databricks/OfficeQA"
UPSTREAM_COMMIT = "7b9a3c154ef9fb40215bb67934afc43e6799de16"
UPSTREAM_FILES = {
    "reward.py": "0d91698c87df6d889339aac36f63ae0966607f169890b0bf8b472b26bfe8138f",
    "NOTICE": "f9f8333a26c3ccef086df6984b33ca7a369734d794486581b74c528b9f99050d",
}

PRO_CSV = "officeqa_pro.csv"
PRO_CSV_SHA1 = "8dc109a1611b29f79685b51e98f7de9cec463466"
PRO_N = 133
FULL_CSV = "officeqa_full.csv"
FULL_CSV_SHA1 = "cc59c600c6bbdf04e5bcf31f43dd58f0c7b3978c"
FULL_N = 246

JSONS_DIR = "treasury_bulletins_parsed/jsons"
N_JSONS = 697
JSONS_SHA256 = "49d7b00c2392efa355850b1d58c2e5987e4e0e21abbeddea036bab5766c8c842"
"""sha256 over the 697 sorted ``"<path> <git-blob sha1>"`` lines (LF, trailing
newline) of ``treasury_bulletins_parsed/jsons/*.json`` at :data:`REVISION`."""

# What the adapter downloads: the CSVs, the parsed corpus, and the repo's own
# license files (tiny; they travel with the data). The PDFs (~4 GB) and the
# convenience zips (LFS) are deliberately excluded.
ALLOW_PATTERNS = (
    PRO_CSV,
    FULL_CSV,
    f"{JSONS_DIR}/*.json",
    "LICENSE-APACHE",
    "LICENSE-CC-BY-SA",
    "NOTICE",
)
CSV_ONLY_PATTERNS = (PRO_CSV, FULL_CSV, "LICENSE-APACHE", "LICENSE-CC-BY-SA", "NOTICE")

CACHE_ENV = "GRANITE_EVALS_OFFICEQA_DIR"
"""An existing dataset checkout to use instead of the fetched pin (trusted)."""

# The oracle_parsed SYSTEM_PROMPT, verbatim from the benchmark's reference
# the ported benchmark logic, kept so scores stay comparable with upstream runs.
SYSTEM_PROMPT = """\
You are answering questions about U.S. Treasury Bulletin documents. \
You will be given relevant document excerpts as context.

Read the context carefully and answer the question precisely. \
Your answer must be a single value — a number, percentage, date, or short text. \
Do NOT include explanations, units unless they are part of the answer, or extra text.

Wrap your final answer in XML tags like this:
<FINAL_ANSWER>your answer here</FINAL_ANSWER>

Examples of valid answers:
<FINAL_ANSWER>2,602</FINAL_ANSWER>
<FINAL_ANSWER>1608.80%</FINAL_ANSWER>
<FINAL_ANSWER>[0.096, -184.143]</FINAL_ANSWER>
<FINAL_ANSWER>36080 million</FINAL_ANSWER>
"""

# The reference harness's TOLERANCES: the headline is the first (exact) match;
# the rest land in details.accuracy.
TOLERANCES = (0.0, 0.001, 0.01, 0.05)

GENERATED = "generated"
"""Status of a record saved by --phase generate, before it is scored."""

REQUEST_TIMEOUT_S = 120
MAX_TOKENS = 4096
TEMPERATURE = 0.0

DEVIATIONS = (
    "gold-page context extracted per question from the parsed JSONs (the reference "
    "harness's index-builder extraction) instead of a prebuilt BM25-index directory; "
    "the index oracle mode passes around is vestigial and dropped",
    "synchronous httpx client with bounded timeout/5xx/429 retries instead of "
    "the reference harness's asyncio semaphore fan-out; same endpoint and payload",
    "answers=gold scores the gold answers through the reward port with the "
    "context still built (a model-free pipeline check upstream does not have)",
    "max_context_chars (default off) truncates an over-long context; upstream "
    "has none, and a truncated question is flagged in its record",
    "an infrastructure failure is retried on resume (failure_policy), not "
    "recorded as a failed question",
    "only oracle_parsed is implemented; the other upstream modes are refused "
    "with an error",
)


# ============================================================================ #
# reward.py, ported verbatim from databricks/OfficeQA@UPSTREAM_COMMIT          #
# (Apache-2.0; NOTICE at that commit): the benchmark's scorer. Only its        #
# module docstring and ``import re`` (provided above) are dropped. The         #
# parity test (tests/test_officeqa.py, OFFICEQA_UPSTREAM) holds this port      #
# against the pinned upstream file.                                            #
# ============================================================================ #


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent parsing.

    - Convert Unicode minus (−, \u2212) to ASCII hyphen (-)
    - Remove extra whitespace

    Raises:
        ValueError: If text is None or empty
    """
    if not text:
        raise ValueError("Cannot normalize empty or None text")

    normalized = text.replace("\u2212", "-")
    return re.sub(r"\s+", " ", normalized).strip()


_CURRENCY_SYMBOLS = r"$£€¥₹¢₩₽"
_NUMBER_BODY = r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?"
_VALID_THOUSANDS_RE = r"(?<![\d.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?!\d)"
_LIST_NUMBER_RE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")


def _normalize_numeric_formatting(text: str) -> str:
    def _accounting_repl(match: re.Match) -> str:
        currency = match.group(1)
        number = match.group(2)
        if currency is None and is_likely_year(float(number.replace(",", ""))):
            return match.group(0)
        return f"-{number}"

    text = re.sub(
        rf"\(\s*([{_CURRENCY_SYMBOLS}])?\s*({_NUMBER_BODY})\s*\)",
        _accounting_repl,
        text,
    )
    return re.sub(rf"[{_CURRENCY_SYMBOLS}]", "", text)


def extract_numbers_with_context(text: str) -> list[tuple[float, str, bool, bool]]:
    """
    Extract numbers with surrounding context for unit detection.

    Returns list of tuples: (number_value, context_string, has_percent, is_negative)

    Raises:
        ValueError: If text parsing fails unexpectedly
    """
    if not text:
        raise ValueError("Cannot extract numbers from empty text")

    # Normalize text first
    text = normalize_text(text)
    text = _normalize_numeric_formatting(text)

    # Remove commas only from thousands-separated numbers (e.g. 1,000,000 -> 1000000)
    text_no_commas = re.sub(
        _VALID_THOUSANDS_RE,
        lambda m: m.group().replace(",", ""),
        text,
    )

    numbers_with_context = []

    # Match negative numbers, decimals, and percentages
    # Pattern: optional minus, digits, optional decimal part, optional %
    pattern = r"-?\d+\.?\d*%?"

    for match in re.finditer(pattern, text_no_commas):
        matched_text = match.group()
        if not matched_text or matched_text == "-":
            continue

        # Check for percentage
        has_percent = matched_text.endswith("%")
        num_text = matched_text.rstrip("%")

        # Check for negative
        is_negative = num_text.startswith("-")

        try:
            num = float(num_text)
        except ValueError as e:
            raise ValueError(
                f"Failed to parse number from '{matched_text}': {e}"
            ) from e

        # Get context (20 chars before and after)
        start = max(0, match.start() - 20)
        end = min(len(text_no_commas), match.end() + 20)
        context = text_no_commas[start:end].lower()

        numbers_with_context.append((num, context, has_percent, is_negative))

    return numbers_with_context


def _single_bracketed_list_body(text: str) -> str | None:
    """Return the body of one bracketed list if surrounding text is label-like."""
    stripped = normalize_text(text)
    matches = list(re.finditer(r"\[([^\[\]]+)\]", stripped))
    if len(matches) != 1:
        return None

    match = matches[0]
    outside = (stripped[: match.start()] + stripped[match.end() :]).strip()
    if re.search(r"\d", outside):
        return None
    return match.group(1)


def _numeric_list_body(text: str) -> str:
    return _single_bracketed_list_body(text) or normalize_text(text)


def _parse_numeric_list_item(raw_item: str) -> float | None:
    item = raw_item.strip()
    if not item:
        return None
    if "," in item and re.search(r"\s,|,\s", item):
        return None

    item = _normalize_numeric_formatting(normalize_text(item))
    if _LIST_NUMBER_RE.fullmatch(item) is None:
        return None

    return float(item.rstrip("%").replace(",", ""))


def _comma_chunk_spans(body: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(",", body):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(body)))
    return spans


def _segment_numeric_list_values(text: str, expected_len: int) -> list[list[float]]:
    """
    Parse comma-delimited numeric-list candidates with exactly expected_len items.

    This intentionally runs only in the gold-list path. It lets a prediction
    like ``[8,152,260]`` match a two-item gold list ``[8, 152260]`` without
    changing global scalar parsing of the valid thousands number ``8,152,260``.
    """
    if expected_len < 2:
        return []

    body = _numeric_list_body(text)
    spans = _comma_chunk_spans(body)
    if len(spans) < expected_len:
        return []

    memo: dict[tuple[int, int], list[list[float]]] = {}

    def dfs(chunk_index: int, remaining: int) -> list[list[float]]:
        key = (chunk_index, remaining)
        if key in memo:
            return memo[key]

        if remaining == 0:
            return [[]] if chunk_index == len(spans) else []

        results: list[list[float]] = []
        max_end = len(spans) - remaining + 1
        for end_index in range(chunk_index + 1, max_end + 1):
            start = spans[chunk_index][0]
            end = spans[end_index - 1][1]
            value = _parse_numeric_list_item(body[start:end])
            if value is None:
                continue
            for tail in dfs(end_index, remaining - 1):
                results.append([value, *tail])

        memo[key] = results
        return results

    deduped: list[list[float]] = []
    seen: set[tuple[float, ...]] = set()
    for values in dfs(0, expected_len):
        key = tuple(values)
        if key not in seen:
            deduped.append(values)
            seen.add(key)
    return deduped


def _ground_truth_bracketed_numeric_list(
    ground_truth: str,
    gt_numbers_with_context: list[tuple[float, str, bool, bool]],
) -> list[tuple[float, str]] | None:
    body = _single_bracketed_list_body(ground_truth)
    if body is None:
        return None

    gt_numbers = [(num, ctx) for num, ctx, _, _ in gt_numbers_with_context]
    if len(gt_numbers) < 2:
        return None

    candidate_values = _segment_numeric_list_values(body, len(gt_numbers))
    gt_values = [num for num, _ in gt_numbers]
    if not any(candidate == gt_values for candidate in candidate_values):
        return None

    return gt_numbers


def _ordered_numbers_match(
    gt_numbers: list[tuple[float, str]],
    pred_numbers: list[tuple[float, str]],
    tolerance: float,
) -> bool:
    if len(gt_numbers) != len(pred_numbers):
        return False

    for (gt_val, gt_context), (pred_val, pred_context) in zip(gt_numbers, pred_numbers):
        gt_base, gt_unit = normalize_number_with_units(gt_val, gt_context)
        pred_base, pred_unit = normalize_number_with_units(pred_val, pred_context)
        if not units_compatible(gt_unit, pred_unit):
            return False
        if gt_base == 0:
            if pred_base != 0:
                return False
            continue
        if abs(gt_base - pred_base) / abs(gt_base) > tolerance:
            return False

    return True


def _match_bracketed_numeric_list(
    ground_truth: str,
    predicted: str,
    gt_numbers_with_context: list[tuple[float, str, bool, bool]],
    pred_numbers_with_context: list[tuple[float, str, bool, bool]],
    tolerance: float,
) -> tuple[bool, str] | None:
    gt_numbers = _ground_truth_bracketed_numeric_list(
        ground_truth, gt_numbers_with_context
    )
    if gt_numbers is None:
        return None

    segmented_candidates = _segment_numeric_list_values(predicted, len(gt_numbers))
    for values in segmented_candidates:
        pred_candidate = [(value, "") for value in values]
        if _ordered_numbers_match(gt_numbers, pred_candidate, tolerance):
            return (
                True,
                f"Ordered list match: segmented prediction into {len(values)} numbers",
            )

    pred_numbers = [(num, ctx) for num, ctx, _, _ in pred_numbers_with_context]
    pred_numbers = _filter_context_years_for_direct_answer(
        gt_numbers, pred_numbers, ground_truth
    )
    if _ordered_numbers_match(gt_numbers, pred_numbers, tolerance):
        return (
            True,
            f"Ordered list match: All {len(gt_numbers)} numbers matched in order",
        )

    expected = [num for num, _ in gt_numbers]
    observed = [[num for num, _ in pred_numbers]]
    observed.extend(segmented_candidates)
    return (
        False,
        f"Ordered list mismatch: expected {expected}, observed candidates {observed[:5]}",
    )


def detect_unit_in_context(context: str) -> tuple[str | None, float]:
    """
    Detect unit words in context and return multiplier.

    Returns: (unit_name, multiplier)
        unit_name: 'trillion', 'billion', 'million', 'thousand', or None
        multiplier: corresponding multiplier or 1.0

    Supports plural forms: millions, billions, trillions, thousands
    """
    context_lower = context.lower()

    # Check for trillion/trillions
    if re.search(r"\btrillions?\b", context_lower):
        return ("trillion", 1e12)

    # Check for billion/billions
    if re.search(r"\bbillions?\b", context_lower) or re.search(r"\bb\b", context_lower):
        return ("billion", 1e9)

    # Check for million/millions
    if re.search(r"\bmillions?\b", context_lower) or re.search(r"\bm\b", context_lower):
        return ("million", 1e6)

    # Check for thousand/thousands
    if re.search(r"\bthousands?\b", context_lower) or re.search(
        r"\bk\b", context_lower
    ):
        return ("thousand", 1e3)

    return (None, 1.0)


def normalize_number_with_units(
    number: float, context: str
) -> tuple[float, str | None]:
    """
    Normalize a number based on its unit context.

    If the context contains "543 million", return (543, 'million')
    This represents the BASE number before unit application.

    Returns: (base_number, unit_name)

    Raises:
        ValueError: If normalization fails
    """
    try:
        unit_name, _ = detect_unit_in_context(context)

        # The number in text is already the BASE number
        # e.g., "543 million" means base=543, unit=million
        # We do NOT multiply here - that's for comparison later

        return (number, unit_name)
    except Exception as e:
        raise ValueError(
            f"Failed to normalize number {number} with context '{context}': {e}"
        ) from e


def units_compatible(gt_unit: str | None, pred_unit: str | None) -> bool:
    return gt_unit is None or pred_unit is None or gt_unit == pred_unit


def is_likely_year(num: float) -> bool:
    """
    Check if a number is likely a year (1900-2100 range).

    Note: This is used to filter out incidental year references in predictions
    (e.g., "reported in 2023") when the ground truth is clearly not a year.
    However, we DO NOT filter when:
    - Ground truth itself is a year-like number (could be a dollar value like "2003")
    - Ground truth contains text (e.g., "March 1977" - the year should match)
    """
    return 1900 <= num <= 2100 and num == int(num)


def has_significant_text(text: str) -> tuple[bool, str]:
    """
    Check if text has significant non-numeric content beyond unit words.

    Returns: (has_significant_text, cleaned_text)
        has_significant_text: True if there's meaningful text beyond numbers/units
        cleaned_text: The text with numbers and common unit words removed

    Examples:
        "March 1977" → (True, "march")
        "543 million" → (False, "")
        "April 15, 2020" → (True, "april")
        "1234" → (False, "")
    """
    if not text:
        return False, ""

    # Normalize and lowercase
    cleaned = normalize_text(text).lower()

    # Remove numbers (including decimals, percentages, commas)
    cleaned = re.sub(r"-?\d+\.?\d*%?", "", cleaned)
    cleaned = re.sub(r"[,]", "", cleaned)

    # Remove common unit words
    unit_words = [
        "trillion",
        "trillions",
        "billion",
        "billions",
        "million",
        "millions",
        "thousand",
        "thousands",
        "hundred",
        "hundreds",
        "percent",
        "percentage",
        "%",
    ]
    for unit in unit_words:
        cleaned = re.sub(r"\b" + unit + r"\b", "", cleaned)

    # Remove extra whitespace and punctuation
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Check if there's meaningful text left (at least 2 characters)
    has_text = len(cleaned) >= 2

    return has_text, cleaned


def check_text_overlap(gt_text: str, pred_text: str) -> tuple[bool, str]:
    """
    Check if key text elements overlap between ground truth and prediction.

    For hybrid answers like "March 1977", we need to ensure that both:
    1. The numeric part matches (checked elsewhere)
    2. The text part (e.g., "March") appears in both

    Returns: (matches, rationale)

    Examples:
        "March 1977" vs "March 1977" → True
        "March 1977" vs "April 1977" → False (different months)
        "March 1977" vs "1977" → False (missing month in prediction)
    """
    if not gt_text or not pred_text:
        return False, "Empty text in comparison"

    # Get significant text from both
    gt_has_text, gt_cleaned = has_significant_text(gt_text)
    pred_has_text, pred_cleaned = has_significant_text(pred_text)

    if not gt_has_text:
        # GT is purely numeric (e.g., "543 million"), no text check needed
        return True, "GT is purely numeric, text check not required"

    if not pred_has_text:
        # GT has text but prediction doesn't (e.g., GT="March 1977", Pred="1977")
        return False, f"GT has text '{gt_cleaned}' but prediction is purely numeric"

    # Both have text - check if GT text appears in prediction
    # We check substring match because prediction might have more context
    if gt_cleaned in pred_cleaned:
        return True, f"Text overlap: '{gt_cleaned}' found in prediction"

    # Also check reverse (in case GT is longer)
    if pred_cleaned in gt_cleaned:
        return True, f"Text overlap: prediction text '{pred_cleaned}' matches GT"

    # No overlap
    return False, f"Text mismatch: GT='{gt_cleaned}', Pred='{pred_cleaned}'"


def extract_final_answer_from_xml(text: str) -> tuple[str, str | None]:
    if not text:
        return "", None

    matches = list(
        re.finditer(
            r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>",
            text,
            re.DOTALL | re.IGNORECASE,
        )
    )

    if matches:
        final_answer_match = matches[-1]
        final_answer = final_answer_match.group(1).strip()
        reasoning_before = text[: final_answer_match.start()].strip()
        return final_answer, reasoning_before if reasoning_before else None

    return text, None


def extract_final_answer(text: str) -> str:
    final_answer, _ = extract_final_answer_from_xml(text)
    return final_answer


def fuzzy_match_answer(
    ground_truth: str, predicted: str, tolerance: float = 0.00
) -> tuple[bool, str]:
    """
    Fuzzy match predicted answer against ground truth with robust handling.

    Args:
        ground_truth: The expected answer
        predicted: The model's predicted answer
        tolerance: Numerical tolerance (default 5%)

    Returns:
        (is_correct, rationale)

    Raises:
        ValueError: On any parsing or validation error
    """
    if not ground_truth:
        raise ValueError("Ground truth cannot be empty")
    if not predicted:
        return False, "Predicted answer is empty - marked as incorrect"
    if not 0 <= tolerance <= 1:
        raise ValueError(f"Tolerance must be between 0 and 1, got {tolerance}")

    if "unable to determine" in predicted.lower():
        return False, "Answer contains 'Unable to determine' - marked as incorrect"

    try:
        # Extract numbers with context
        gt_numbers_with_context = extract_numbers_with_context(ground_truth)
        pred_numbers_with_context = extract_numbers_with_context(predicted)
    except Exception as e:
        raise ValueError(f"Failed to extract numbers: {e}") from e

    # Get just the numbers for empty checks
    gt_numbers = [(num, ctx) for num, ctx, _, _ in gt_numbers_with_context]
    pred_numbers = [(num, ctx) for num, ctx, _, _ in pred_numbers_with_context]

    # Case 1: Both have numbers
    if gt_numbers and pred_numbers:
        ordered_list_match = _match_bracketed_numeric_list(
            ground_truth,
            predicted,
            gt_numbers_with_context,
            pred_numbers_with_context,
            tolerance,
        )
        if ordered_list_match is not None:
            return ordered_list_match

        # Check if GT has multiple numbers (list-like answer)
        if len(gt_numbers) > 1:
            # Multi-number answer: ALL GT numbers must appear in prediction
            # Filter out years from predicted numbers first
            pred_non_years = [
                (n, c)
                for n, c in pred_numbers
                if not is_likely_year(n)
                or any(is_likely_year(g) for g, _ in gt_numbers)
            ]

            matched_gt = []
            unmatched_gt = []

            for gt_val, gt_context in gt_numbers:
                try:
                    gt_base, gt_unit = normalize_number_with_units(gt_val, gt_context)
                except Exception as e:
                    raise ValueError(
                        f"Failed to normalize GT number {gt_val}: {e}"
                    ) from e

                # Try to find this GT number in predictions
                found_match = False
                for pred_val, pred_context in pred_non_years:
                    try:
                        pred_base, pred_unit = normalize_number_with_units(
                            pred_val, pred_context
                        )
                    except Exception as e:
                        raise ValueError(
                            f"Failed to normalize prediction number {pred_val}: {e}"
                        ) from e

                    if not units_compatible(gt_unit, pred_unit):
                        continue

                    # Compare base numbers
                    if gt_base == 0:
                        if pred_base == 0:
                            # For multi-number lists, also check text overlap
                            text_matches, _ = check_text_overlap(
                                ground_truth, predicted
                            )
                            if text_matches:
                                found_match = True
                                break
                    else:
                        diff_pct = abs(gt_base - pred_base) / abs(gt_base)
                        if diff_pct <= tolerance:
                            # For multi-number lists, also check text overlap
                            text_matches, _ = check_text_overlap(
                                ground_truth, predicted
                            )
                            if text_matches:
                                found_match = True
                                break

                if found_match:
                    matched_gt.append(gt_val)
                else:
                    unmatched_gt.append(gt_val)

            # All GT numbers must be matched
            if len(matched_gt) == len(gt_numbers):
                return (
                    True,
                    f"List match: All {len(gt_numbers)} numbers found in prediction",
                )
            else:
                return (
                    False,
                    f"List mismatch: Found {len(matched_gt)}/{len(gt_numbers)} numbers. Missing: {unmatched_gt}",
                )

        else:
            # Single number answer
            gt_val, gt_context = gt_numbers[0]

            try:
                gt_base, gt_unit = normalize_number_with_units(gt_val, gt_context)
            except Exception as e:
                raise ValueError(f"Failed to normalize GT number: {e}") from e

            # Determine if we should filter year-like numbers from predictions
            # We should NOT filter if:
            # 1. Ground truth itself is a year-like number (e.g., "2003", "1977")
            # 2. Ground truth has text content (e.g., "March 1977" - the 1977 should match)
            gt_has_text, _ = has_significant_text(ground_truth)
            should_filter_years = not (is_likely_year(gt_val) or gt_has_text)

            # Try to find the expected number in the predicted response
            best_match = None
            best_diff = float("inf")
            best_pred_info = None
            unit_mismatches = []

            for pred_val, pred_context in pred_numbers:
                # Skip likely year numbers only if GT is clearly not a year-related answer
                if should_filter_years and is_likely_year(pred_val):
                    continue

                try:
                    pred_base, pred_unit = normalize_number_with_units(
                        pred_val, pred_context
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to normalize prediction number: {e}"
                    ) from e

                if not units_compatible(gt_unit, pred_unit):
                    unit_mismatches.append((pred_base, pred_unit))
                    continue

                # Compare base numbers (after unit normalization)
                # e.g., GT "543 million" → base=543, pred "543" → base=543 ✅
                # e.g., GT "543 million" → base=543, pred "543000000" → base=543000000 ❌

                if gt_base == 0:
                    if pred_base == 0:
                        # Also check text overlap for zero values
                        text_matches, text_rationale = check_text_overlap(
                            ground_truth, predicted
                        )
                        if text_matches:
                            return (
                                True,
                                f"Exact match: Found 0 in response. {text_rationale}",
                            )
                    continue

                diff_pct = abs(gt_base - pred_base) / abs(gt_base)

                if diff_pct < best_diff:
                    best_diff = diff_pct
                    best_match = pred_base
                    best_pred_info = (pred_base, pred_unit)

                if diff_pct <= tolerance:
                    # Numeric match found - but check if GT has significant text too
                    text_matches, text_rationale = check_text_overlap(
                        ground_truth, predicted
                    )
                    if not text_matches:
                        # Numbers match but text doesn't (e.g., "March 1977" vs "April 1977")
                        continue  # Try other numbers in prediction

                    return (
                        True,
                        f"Numerical match: GT={gt_base} ({gt_unit or 'no unit'}), Pred={pred_base} ({pred_unit or 'no unit'}), Diff={diff_pct*100:.2f}%. {text_rationale}",
                    )

            # No match found
            if best_match is not None:
                return (
                    False,
                    f"No match: GT={gt_base} ({gt_unit or 'no unit'}), Closest={best_pred_info[0]} ({best_pred_info[1] or 'no unit'}), Diff={best_diff*100:.2f}%",
                )
            if unit_mismatches:
                pred_units = [unit or "no unit" for _, unit in unit_mismatches[:5]]
                return (
                    False,
                    f"No match: explicit unit mismatch. GT unit={gt_unit or 'no unit'}, Pred units={pred_units}",
                )
            return (
                False,
                f"No valid numbers found in prediction (filtered out years: {[n for n, _ in pred_numbers[:5]]})",
            )

    # Case 2: Text-based comparison (case-insensitive, strip whitespace and quotes)
    # For dates and text, must be EXACT match (case-insensitive)
    gt_clean = ground_truth.strip().lower().strip('"').strip("'")
    pred_clean = predicted.strip().lower().strip('"').strip("'")

    # Strip parenthetical content like (OASI), (FY), etc. to handle abbreviations
    # e.g., "Federal Old-Age and Survivors Insurance (OASI) Trust Fund"
    #    -> "Federal Old-Age and Survivors Insurance Trust Fund"
    gt_clean = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", gt_clean)).strip()
    pred_clean = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", "", pred_clean)).strip()

    # Check if ground truth appears in prediction
    if gt_clean in pred_clean:
        return True, f"Text match: '{ground_truth}' found in prediction"

    # Check exact match
    if gt_clean == pred_clean:
        return True, "Exact text match"

    # No match
    return (
        False,
        f"No match found. GT: '{ground_truth[:100]}', Pred: '{predicted[:100]}'",
    )


def _normalize_direct_text_answer(text: str) -> str:
    cleaned = text.strip().lower().strip('"').strip("'")
    cleaned = re.sub(r"\([^)]*\)", "", cleaned).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def _filter_context_years_for_direct_answer(
    gt_numbers: list[tuple[float, str]],
    pred_numbers: list[tuple[float, str]],
    ground_truth: str,
) -> list[tuple[float, str]]:
    if not gt_numbers:
        return pred_numbers

    gt_has_text, _ = has_significant_text(ground_truth)
    if len(gt_numbers) == 1:
        gt_val, _ = gt_numbers[0]
        should_filter_years = not (is_likely_year(gt_val) or gt_has_text)
    else:
        should_filter_years = not any(
            is_likely_year(gt_val) for gt_val, _ in gt_numbers
        )

    if not should_filter_years:
        return pred_numbers

    return [
        (pred_val, pred_context)
        for pred_val, pred_context in pred_numbers
        if not is_likely_year(pred_val)
    ]


_INLINE_MARKUP_RE = re.compile(r"<[^>]*>")


def _is_direct_answer_only(ground_truth: str, predicted: str) -> tuple[bool, str]:
    """
    Guard that prediction is a single, direct answer of the same shape as GT.

    Policy:
        - Empty prediction -> fail
        - Multiple non-empty lines -> fail
        - Predictions longer than 250 chars -> fail
        - Numeric GT -> prediction must contain the same count of numeric values
        - Purely numeric GT -> prediction must not contain prose after removing
          numbers and unit words
        - Text/date GT (no numbers) -> normalized prediction must equal normalized GT

    Returns:
        (ok, rationale)
    """
    predicted = predicted.strip()
    if not predicted:
        return False, "Predicted answer is empty"

    nonempty_lines = [line for line in predicted.splitlines() if line.strip()]
    if len(nonempty_lines) > 1:
        return False, "Predicted answer spans multiple non-empty lines"

    if len(predicted) > 250:
        return False, "Predicted answer is too long to be a direct answer"

    gt_numbers_with_context = extract_numbers_with_context(ground_truth)
    pred_numbers_with_context = extract_numbers_with_context(predicted)

    gt_numbers = [(num, ctx) for num, ctx, _, _ in gt_numbers_with_context]
    pred_numbers = [(num, ctx) for num, ctx, _, _ in pred_numbers_with_context]

    if gt_numbers:
        pred_numbers = _filter_context_years_for_direct_answer(
            gt_numbers, pred_numbers, ground_truth
        )
        if len(pred_numbers) != len(gt_numbers):
            gt_list_numbers = _ground_truth_bracketed_numeric_list(
                ground_truth, gt_numbers_with_context
            )
            list_arity_ok = gt_list_numbers is not None and bool(
                _segment_numeric_list_values(predicted, len(gt_list_numbers))
            )
            if not list_arity_ok:
                return (
                    False,
                    "Predicted answer must contain exactly the expected answer numbers",
                )

        if _INLINE_MARKUP_RE.search(predicted):
            return False, "Predicted answer contains markup outside the answer value"

        return True, "Direct answer only"

    if pred_numbers:
        return False, "Prediction contains numbers but ground truth is text-only"

    if _normalize_direct_text_answer(ground_truth) != _normalize_direct_text_answer(
        predicted
    ):
        return False, "Text answer must match the expected answer text exactly"

    return True, "Direct answer only"


def score_answer(ground_truth: str, predicted: str, tolerance: float = 0.00) -> float:
    """
    Score the answer using robust fuzzy matching.
    """
    try:
        predicted, _ = extract_final_answer_from_xml(predicted)
    except Exception:
        return 0.0

    try:
        ok, _ = _is_direct_answer_only(ground_truth, predicted)
        if not ok:
            return 0.0

        is_correct, _ = fuzzy_match_answer(ground_truth, predicted, tolerance)
    except Exception:
        return 0.0

    return 1.0 if is_correct else 0.0


# -- scoring over the tolerance ladder (score at every tolerance) ----


def tolerance_labels() -> tuple[str, ...]:
    """``("0%", "0.1%", "1%", "5%")`` -- labels for :data:`TOLERANCES`."""
    return tuple(f"{t * 100:.1f}%".replace(".0%", "%") for t in TOLERANCES)


def score_all_tolerances(ground_truth: str, predicted: str) -> dict[str, float]:
    """The reward at every tolerance, keyed by its label."""
    return {
        label: score_answer(ground_truth, predicted, t)
        for label, t in zip(tolerance_labels(), TOLERANCES)
    }


def zero_scores() -> dict[str, float]:
    return {label: 0.0 for label in tolerance_labels()}


# -- pins: git-blob sha1s over the fetched tree ---------------------------------


def git_blob_sha1(data: bytes) -> str:
    """The git blob oid of ``data`` -- what the Hub's tree API reports for a
    non-LFS file, so a downloaded file can be checked against the pin without
    git."""
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


def _safe_uid(uid: str) -> str:
    """A filename-safe stand-in for a question uid (the record keeps the raw
    one; uids are expected to already be safe, this only guards the path)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", uid)


class Corpus:
    """The pinned :data:`DATASET` checkout, fetched and verified at run start.

    ``ensure(need_pages=True)`` fetches the CSVs and the parsed JSON corpus
    (the generate phase, which builds gold-page contexts); ``need_pages=False``
    fetches only the CSVs (the score phase re-scores saved generations and
    needs just the questions and gold answers). A fetch that does not match
    the pins refuses to run. ``download`` is injectable for the tests; a
    ``root`` (a local checkout) is trusted as-is.
    """

    def __init__(self, root: Path | None = None, *, download=None):
        self.root = root
        self._download = download or self._snapshot

    def _snapshot(self, allow_patterns) -> Path:
        from huggingface_hub import snapshot_download

        try:
            return Path(
                snapshot_download(
                    repo_id=DATASET,
                    repo_type="dataset",
                    revision=REVISION,
                    allow_patterns=list(allow_patterns),
                    token=os.environ.get("HF_TOKEN"),
                )
            )
        except Exception as e:  # noqa: BLE001 - turned into the gating message
            text = str(e).lower()
            if any(k in text for k in ("401", "403", "gated", "restricted")):
                raise SystemExit(
                    f"{DATASET} is gated: HF_TOKEN must be set to a token of an "
                    "account that has accepted the dataset's terms (the gate asks "
                    "that the answer keys not be used to train models evaluated on "
                    f"OfficeQA). Original error: {e}"
                ) from e
            raise

    def ensure(self, *, need_pages: bool) -> "Corpus":
        """Fetch (HF cache) and verify the pinned snapshot; returns self."""
        self.root = self._download(ALLOW_PATTERNS if need_pages else CSV_ONLY_PATTERNS)
        self.verify(self.root, need_pages=need_pages)
        return self

    def verify(self, root: Path, *, need_pages: bool) -> None:
        """Refuse a fetch that does not match the pins: the CSVs' git-blob
        sha1s and question counts always; the corpus digest when the corpus
        was fetched."""
        for name, sha1, n in (
            (PRO_CSV, PRO_CSV_SHA1, PRO_N),
            (FULL_CSV, FULL_CSV_SHA1, FULL_N),
        ):
            path = root / name
            if not path.is_file():
                raise RuntimeError(
                    f"{DATASET}@{REVISION}: {name} is missing from the fetch; "
                    "refusing to run"
                )
            got = git_blob_sha1(path.read_bytes())
            if got != sha1:
                raise RuntimeError(
                    f"{DATASET}@{REVISION}: {name} has blob sha1 {got}, expected "
                    f"{sha1}; refusing to run"
                )
            with open(path, newline="", encoding="utf-8") as f:
                rows = sum(1 for _ in csv.reader(f)) - 1
            if rows != n:
                raise RuntimeError(
                    f"{DATASET}@{REVISION}: {name} holds {rows} questions, "
                    f"expected {n}; refusing to run"
                )
        if not need_pages:
            return
        jsons = root / JSONS_DIR
        files = (
            sorted(p for p in jsons.glob("*.json") if p.is_file())
            if jsons.is_dir()
            else []
        )
        if len(files) != N_JSONS:
            raise RuntimeError(
                f"{DATASET}@{REVISION}: {JSONS_DIR} holds {len(files)} documents, "
                f"expected {N_JSONS}; refusing to run"
            )
        lines = sorted(
            f"{JSONS_DIR}/{p.name} {git_blob_sha1(p.read_bytes())}" for p in files
        )
        digest = hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()
        if digest != JSONS_SHA256:
            raise RuntimeError(
                f"{DATASET}@{REVISION}: {JSONS_DIR} digest {digest}, expected "
                f"{JSONS_SHA256}; refusing to run"
            )

    # -- gold pages ------------------------------------------------------------

    def page(self, txt_name: str, page_id: int) -> str:
        """One gold page's text: ``<stem>.json`` of the CSV's ``.txt`` name,
        extracted exactly as the reference harness's index builder does."""
        json_path = self.root / JSONS_DIR / (Path(txt_name).stem + ".json")
        if not json_path.is_file():
            raise FileNotFoundError(
                f"{txt_name}: no parsed document {JSONS_DIR}/{json_path.name} "
                f"under {self.root}"
            )
        pages = _pages_cached(str(json_path))
        if page_id not in pages:
            available = sorted(pages)[:10]
            raise FileNotFoundError(
                f"Page {page_id} not found in {txt_name}. Available pages: "
                f"{available}..."
            )
        return pages[page_id]


@functools.lru_cache(maxsize=8)
def _pages_cached(json_path: str) -> dict[int, str]:
    return extract_pages_from_json(Path(json_path))


# -- page extraction: build_bm25_index.py's, ported (Apache-2.0) ----------------


def _pandas():
    try:
        import pandas as pd
    except ImportError as e:
        raise RuntimeError(
            "officeqa: rendering the corpus tables needs pandas (and lxml for "
            "read_html); install the family extra (pip install "
            "'granite-evals[officeqa]') or run the family image"
        ) from e
    return pd


def _flatten_columns(columns) -> list[str]:
    pd = _pandas()
    if isinstance(columns, pd.MultiIndex):
        names = []
        for tup in columns:
            parts = [
                str(x).strip() for x in tup if x is not None and str(x).strip() != ""
            ]
            names.append(" > ".join(parts) if parts else "")
        return names
    return [str(c).strip() for c in list(columns)]


def _sanitize_md_cell(v) -> str:
    s = "" if v is None else str(v)
    s = s.replace("\n", " ").replace("|", "\\|")
    return s.strip()


def _dataframe_to_markdown(df) -> str:
    headers = _flatten_columns(df.columns)
    seen = {}
    for i, h in enumerate(headers):
        name = h if h else f"col_{i+1}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        headers[i] = name

    rows = []
    for _, row in df.iterrows():
        rows.append([_sanitize_md_cell(row.get(col)) for col in df.columns])

    line_header = "| " + " | ".join(_sanitize_md_cell(h) for h in headers) + " |"
    line_sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line_header, line_sep] + body_lines)


def _html_tables_to_markdown(html_str: str) -> list[str]:
    import io

    pd = _pandas()
    try:
        dfs = pd.read_html(io.StringIO(html_str))
    except Exception:  # upstream's fallback: the raw HTML in a code block
        return ["```html\n" + html_str.strip() + "\n```"]
    return [_dataframe_to_markdown(df) for df in dfs]


def extract_pages_from_json(json_path: Path) -> dict[int, str]:
    """``{page_id: text}`` of one parsed document: elements with content,
    grouped by their ``bbox[0].page_id`` (None drops the element, as the reference
    harness's index builder does), tables rendered to Markdown."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    elements = data.get("document", {}).get("elements", [])
    pages: dict[int, list[str]] = {}

    for el in elements:
        content = el.get("content")
        if not isinstance(content, str) or not content.strip():
            continue

        page_id = None
        if "bbox" in el and el["bbox"]:
            page_id = el["bbox"][0].get("page_id")
        if page_id is None:
            continue

        if page_id not in pages:
            pages[page_id] = []

        if "<table" in content.lower():
            for t in _html_tables_to_markdown(content):
                pages[page_id].append(t)
                pages[page_id].append("")
        else:
            text = content.strip()
            text = re.sub(r"\r\n?", "\n", text)
            for line in text.split("\n"):
                pages[page_id].append(line)
            pages[page_id].append("")

    return {page_id: "\n".join(lines) for page_id, lines in pages.items()}


# -- questions -----------------------------------------------------------------


QUESTION_COLUMNS = (
    "uid",
    "question",
    "answer",
    "source_docs",
    "source_files",
    "difficulty",
)


class Question:
    """One CSV row (upstream's CSV field set)."""

    __slots__ = QUESTION_COLUMNS

    def __init__(self, row: dict):
        for col in QUESTION_COLUMNS:
            value = row.get(col)
            if not isinstance(value, str):
                raise RuntimeError(f"question row without a {col!r} column: {row}")
            setattr(self, col, value)


def load_questions(csv_path: Path) -> list[Question]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in QUESTION_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError(
                f"{csv_path}: missing column(s) {', '.join(missing)}; expected "
                f"{', '.join(QUESTION_COLUMNS)}"
            )
        return [Question(row) for row in reader]


def parse_gold_pages(question: Question) -> list[tuple[str, int]]:
    """``(filename, page_id)`` pairs from ``source_docs`` (``page=N`` in each
    URL) zipped with ``source_files`` -- the reference harness's parsing, verbatim."""
    urls = [
        u.strip()
        for u in question.source_docs.replace("\r\n", "\n").split("\n")
        if u.strip()
    ]
    files = [
        f.strip()
        for f in question.source_files.replace("\r\n", "\n").split("\n")
        if f.strip()
    ]
    if not files:
        return []

    results = []
    for i, url in enumerate(urls):
        m = re.search(r"page=(\d+)", url)
        page_id = int(m.group(1)) if m else None
        fname = files[i] if i < len(files) else files[-1]
        if page_id is not None:
            results.append((fname, page_id))
    return results


def oracle_context(
    question: Question, corpus: Corpus
) -> tuple[str, list[tuple[str, int]]]:
    """The gold pages' text, joined as the reference harness's oracle mode joins them."""
    gold_pages = parse_gold_pages(question)
    if not gold_pages:
        raise ValueError(f"No gold pages found for {question.uid}")
    parts = [corpus.page(fname, page_id) for fname, page_id in gold_pages]
    return "\n\n---\n\n".join(parts), gold_pages


def build_messages(question: Question, context: str) -> list[dict]:
    """The reference harness's oracle_parsed messages, verbatim shape."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"## Document Context\n\n{context}\n\n## Question\n\n{question.question}",
        },
    ]


# -- the model endpoint (synchronous call, with bounded retries) ---------------


MAX_ATTEMPTS = 5
RETRY_WAIT_S = 2.0


class ModelClient:
    """POST /chat/completions with upstream's payload (``model``, ``messages``,
    ``max_tokens``, ``temperature``); retries timeouts, 429 and 5xx with
    backoff, and raises on anything else."""

    def __init__(self, base_url: str, model: str, *, timeout: float, transport=None):
        import httpx

        url = base_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"
        self.url, self.model = url, model
        self._http = httpx.Client(transport=transport, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def complete(
        self, messages: list[dict], *, max_tokens: int, temperature: float
    ) -> tuple[str, str | None, dict]:
        """``(text, finish_reason, usage)`` of one completion."""
        import httpx

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        attempts = 0
        while True:
            try:
                r = self._http.post(
                    self.url, json=payload, headers={"Authorization": "Bearer EMPTY"}
                )
            except httpx.TimeoutException:
                attempts += 1
                if attempts < MAX_ATTEMPTS:
                    time.sleep(RETRY_WAIT_S * 2 ** (attempts - 1))
                    continue
                raise RuntimeError("model call timed out") from None
            except httpx.HTTPError as e:
                raise RuntimeError(f"model endpoint unreachable: {e}") from e
            if r.status_code == 200:
                body = r.json()
                choice = body["choices"][0]
                return (
                    choice["message"]["content"] or "",
                    choice.get("finish_reason"),
                    body.get("usage") or {},
                )
            if r.status_code in (429, 500, 502, 503):
                attempts += 1
                if attempts < MAX_ATTEMPTS:
                    time.sleep(RETRY_WAIT_S * 2 ** (attempts - 1))
                    continue
                raise RuntimeError(f"model endpoint HTTP {r.status_code}")
            raise RuntimeError(f"model endpoint HTTP {r.status_code}: {r.text[:500]}")


# -- the benchmarks --------------------------------------------------------------


class _OfficeQA(Benchmark):
    """The port above; see the module docstring for pins and deviations."""

    csv_name: str
    csv_sha1: str
    n_questions: int

    extra = "officeqa"
    harness_packages = ("pandas", "lxml")
    dataset = DATASET
    dataset_revision = REVISION
    splittable = True

    # -- options -----------------------------------------------------------

    def opt(self, key: str, default: Any) -> Any:
        """A ``--option key=value`` (always a string on the CLI), cast like
        ``default``; bools parse ``false``/``0``/``no``/``off`` as false."""
        value = self.config.options.get(key)
        if value is None:
            return default
        if isinstance(default, bool):
            return str(value).strip().lower() not in ("", "0", "false", "no", "off")
        return type(default)(value)

    @property
    def mode(self) -> str:
        """``--option mode=...``: ``oracle_parsed`` (the benchmark's headline
        mode) is the only one this family ships."""
        mode = self.opt("mode", "oracle_parsed")
        if mode != "oracle_parsed":
            raise SystemExit(
                f"{self.id}: mode={mode!r} is not implemented; this family ships "
                "oracle_parsed only (upstream's oracle_full_doc, bm25/vector/agent "
                "and PDF modes are not ported)"
            )
        return mode

    @property
    def gold(self) -> bool:
        """``--option answers=gold``: no model. Each question's context is
        still built and its gold answer scored through the reward port -- a
        model-free check of the fetch, the pins, the page extraction and the
        scorer."""
        return self.opt("answers", "model") == "gold"

    def needs_server(self) -> bool:
        return not self.gold

    # -- entry point -------------------------------------------------------

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        self.mode  # validated before anything is fetched
        source, revision = self.dataset_source()
        corpus, pinned = self._corpus_root()
        questions = self._select(corpus.root)
        model = self.generating and not self.gold  # this run asks the model
        client = (
            ModelClient(
                base_url,
                served_model_name,
                timeout=self.opt("timeout", REQUEST_TIMEOUT_S),
            )
            if model
            else None
        )
        log.info(
            "%s: %d questions x %d repeats%s%s",
            self.id,
            len(questions),
            self.repeats,
            " (gold answers)" if self.gold else "",
            "" if self.phase == "all" else f" ({self.phase} phase)",
        )

        per_repeat = []
        all_records: list[list[dict]] = []
        by_question: dict[str, list[float]] = {}  # 0%-tolerance scores, for pass@k
        try:
            for k in range(self.repeats):
                rdir = self.config.output_dir / f"repeat-{k}"
                rdir.mkdir(parents=True, exist_ok=True)

                def one(q: Question) -> dict:
                    return self._question(q, rdir, corpus, client, k)

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.config.workers
                ) as pool:
                    recs = list(pool.map(one, questions))
                all_records.append(recs)
                scored = [r for r in recs if not r["status"].startswith("error:")]
                if self.scoring:  # generated-only records carry no scores yet
                    for r in scored:
                        by_question.setdefault(r["uid"], []).append(
                            float(r["scores"]["0%"])
                        )
                entry = {
                    "repeat": k,
                    "n": len(scored),
                    "statuses": _count(r["status"] for r in recs),
                    "failed": len(recs) - len(scored),
                }
                if self.scoring:
                    entry["accuracy"] = _mean_scores(scored)
                    log.info(
                        "%s repeat %d: %.4f at 0%% tolerance, %d/%d scored, %d failed",
                        self.id,
                        k,
                        entry["accuracy"]["0%"],
                        len(scored),
                        len(recs),
                        entry["failed"],
                    )
                else:
                    log.info(
                        "%s repeat %d: %d generated, %d failed",
                        self.id,
                        k,
                        entry["n"],
                        entry["failed"],
                    )
                per_repeat.append(entry)
        finally:
            if client is not None:
                client.close()

        failed = sum(r["failed"] for r in per_repeat)
        policy = failure_policy(
            self.id,
            "questions",
            failed,
            len(questions) * self.repeats,
            self.opt("max_failed_frac", 0.05),
        )
        out: dict[str, Any] = {
            "dataset": source,
            "dataset_revision": revision,
            "mode": "oracle_parsed",
            "answers": "gold" if self.gold else "model",
            "questions": len(questions),
            "sampling": (
                {}
                if self.gold
                else {
                    "temperature": self.opt("temperature", TEMPERATURE),
                    "max_tokens": self.opt("max_tokens", MAX_TOKENS),
                }
            ),
            "timeout_s": self.opt("timeout", REQUEST_TIMEOUT_S),
            "corpus": {
                "root": str(corpus.root),
                "csv": self.csv_name,
                "jsons_sha256": JSONS_SHA256 if pinned and self.generating else None,
            },
            "deviations": list(DEVIATIONS),
        }
        if not self.scoring:
            # Generated: a record in every repeat (failed ones are retried by rerunning generate).
            return {
                "n": min(r["n"] for r in per_repeat),
                **policy,
                "per_repeat": [
                    {"repeat": r["repeat"], "generated": r["n"], "failed": r["failed"]}
                    for r in per_repeat
                ],
                **out,
            }
        if any(r["n"] == 0 for r in per_repeat):
            raise SystemExit(
                f"{self.id}: a repeat has no scored question; rerun to retry the failed ones"
            )
        labels = tolerance_labels()
        accuracy = {
            label: sum(r["accuracy"][label] for r in per_repeat) / len(per_repeat)
            for label in labels
        }
        by_difficulty: dict[str, dict[str, Any]] = {}
        for r in (r for recs in all_records for r in recs):
            if r["status"].startswith("error:"):
                continue
            bucket = by_difficulty.setdefault(
                r["difficulty"], {"n": 0, "accuracy@0%": 0.0}
            )
            bucket["n"] += 1
            bucket["accuracy@0%"] += float(r["scores"]["0%"])
        for bucket in by_difficulty.values():
            bucket["accuracy@0%"] /= bucket["n"]
        return {
            "value": accuracy["0%"],
            "n": min(
                r["n"] for r in per_repeat
            ),  # fewest scored in a repeat: failed questions are left out
            **policy,
            "accuracy": accuracy,
            "accuracy_by_difficulty": by_difficulty,
            "pass_at_k": pass_at_k(
                by_question.values(),
                self.repeats,
                "correct at 0% tolerance in any scored repeat",
            ),
            "per_repeat": per_repeat,
            **out,
        }

    def _corpus_root(self) -> tuple[Corpus, bool]:
        """``(corpus, pinned)``: a local ``--dataset`` dir or
        ``GRANITE_EVALS_OFFICEQA_DIR`` is trusted as-is (``pinned=False``);
        otherwise the fetched snapshot, verified against the pins
        (``pinned=True``). The score phase fetches only the CSVs; anything
        that builds contexts fetches (and verifies) the corpus too."""
        if self.config.dataset:
            root = Path(self.config.dataset)
            if not root.is_dir():
                raise SystemExit(f"{self.id}: --dataset {root}: not a directory")
            return Corpus(root), False
        if env_dir := os.environ.get(CACHE_ENV, "").strip():
            root = Path(env_dir)
            if not root.is_dir():
                raise SystemExit(f"{self.id}: {CACHE_ENV}={root}: not a directory")
            return Corpus(root), False
        return Corpus().ensure(need_pages=self.generating), True

    def _select(self, root: Path) -> list[Question]:
        questions = sorted(load_questions(root / self.csv_name), key=lambda q: q.uid)
        if pattern := self.opt("questions", ""):
            questions = [q for q in questions if re.search(pattern, q.uid)]
        return questions[: self.config.limit] if self.config.limit else questions

    # -- one question ------------------------------------------------------

    def _question(
        self,
        q: Question,
        rdir: Path,
        corpus: Corpus,
        client: ModelClient | None,
        k: int,
    ) -> dict[str, Any]:
        path = rdir / f"{_safe_uid(q.uid)}.json"
        rec = json.loads(path.read_text()) if path.exists() else None
        if rec is not None and (rec["status"] != GENERATED or not self.scoring):
            return rec  # scored, or generated and this run does not score
        base = {
            "uid": q.uid,
            "difficulty": q.difficulty,
            "question": q.question,
            "answer": q.answer,
        }
        if rec is None and not self.generating:
            # Unresolved under --phase all; never generated here.
            log.warning("%s", self.not_generated(f"{q.uid} repeat {k}"))
            return {**base, "scores": zero_scores(), "status": "error:not_generated"}
        try:
            if rec is not None:
                pass  # generated by an earlier --phase generate: score it
            else:
                context, pages = oracle_context(q, corpus)
                context_chars = len(context)
                truncated = False
                limit = self.opt("max_context_chars", 0)
                if limit and context_chars > limit:
                    context = context[:limit]
                    truncated = True
                if self.gold:
                    rec = {
                        **base,
                        "answers": "gold",
                        "pages": pages,
                        "context_chars": context_chars,
                        "context_truncated": truncated,
                        "generation": q.answer,  # the gold answer is the prediction
                    }
                else:
                    t0 = time.time()
                    text, finish, usage = client.complete(
                        build_messages(q, context),
                        max_tokens=self.opt("max_tokens", MAX_TOKENS),
                        temperature=self.opt("temperature", TEMPERATURE),
                    )
                    rec = {
                        **base,
                        "pages": pages,
                        "context_chars": context_chars,
                        "context_truncated": truncated,
                        "generation": text,
                        "finish_reason": finish,
                        "usage": usage,
                        "gen_s": round(time.time() - t0, 1),
                    }
            if self.scoring:
                rec["pred"] = extract_final_answer(rec["generation"])
                rec["scores"] = score_all_tolerances(q.answer, rec["generation"])
                rec["status"] = "graded"
            else:
                rec["status"] = GENERATED
        except Exception as e:  # one broken question must not sink the run
            log.exception("%s: question %s failed", self.id, q.uid)
            # Not persisted: an infrastructure error is retried on resume.
            return {
                **base,
                "scores": zero_scores(),
                "status": f"error:{type(e).__name__}",
            }
        path.write_text(json.dumps(rec, indent=2))
        log.info("%s repeat %d: question %s %s", self.id, k, q.uid, rec["status"])
        return rec


def _mean_scores(records: list[dict]) -> dict[str, float]:
    """Mean reward per tolerance label over the scored records of one repeat."""
    labels = tolerance_labels()
    if not records:
        return {label: 0.0 for label in labels}
    return {
        label: sum(float(r["scores"][label]) for r in records) / len(records)
        for label in labels
    }


@register
class OfficeQAPro(_OfficeQA):
    """officeqa_pro.csv: the :data:`PRO_N`-question Pro set."""

    id = "officeqa-pro"
    metric = "accuracy@0%"
    csv_name, csv_sha1, n_questions = PRO_CSV, PRO_CSV_SHA1, PRO_N


@register
class OfficeQAFull(_OfficeQA):
    """officeqa_full.csv: the :data:`FULL_N`-question Full set."""

    id = "officeqa-full"
    metric = "accuracy@0%"
    csv_name, csv_sha1, n_questions = FULL_CSV, FULL_CSV_SHA1, FULL_N
