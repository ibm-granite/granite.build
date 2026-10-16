# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Pure parsing-strategy helpers: apply a strategy, validate a JS-regex.

No HTTP and no DB. The single source of validation truth, shared by
:class:`~autotunex.services.dataset_intelligence.DatasetIntelligenceService`'s
generation loop and the ``validate-strategy`` endpoint, so the two cannot drift
(the 2025 route lacked the regex check the service had).
"""

from __future__ import annotations

import json
import re
from typing import Any

from autotunex.core.exceptions import DomainValidationError
from autotunex.models.dataset_intelligence import ParsingStrategy

# (regex-that-detects-the-construct, human explanation) pairs. Each names a
# Python-only construct with no portable JS regex equivalent.
_JS_INCOMPATIBLE: tuple[tuple[str, str], ...] = (
    (r"\(\?P<", "named groups must use (?<name>...), not Python's (?P<name>...)"),
    (r"\(\?P=", "back-references must use \\k<name>, not Python's (?P=name)"),
    (r"\(\?[aiLmsux]+\)", "inline global flags like (?i) are not portable to JS"),
    (r"\(\?[aiLmsux]+:", "scoped inline flags like (?i:...) are not supported in JS"),
    (r"\\A", "\\A is not supported in JS; use ^"),
    (r"\\Z", "\\Z is not supported in JS; use $"),
    (r"\(\?\(", "conditional patterns (?(id)yes|no) are not supported in JS"),
)


def validate_js_regex(pattern: str) -> None:
    """Reject Python-only regex constructs a browser (JS) engine cannot run.

    The generated ``input_pattern``/``output_pattern`` are meant to run
    client-side, so anything with no JS equivalent is rejected here.

    Raises:
        DomainValidationError: the pattern will not compile, or uses a
            Python-only construct.
    """
    try:
        re.compile(pattern)
    except re.error as exc:
        raise DomainValidationError(f"Pattern is not a valid regular expression: {exc}.") from exc
    for detector, message in _JS_INCOMPATIBLE:
        if re.search(detector, pattern):
            raise DomainValidationError(f"Pattern uses a construct unavailable in JS: {message}.")


def apply_parsing_strategy(
    sample: list[dict[str, Any]] | str, strategy: ParsingStrategy
) -> tuple[list[dict[str, Any]], list[str]]:
    """Apply ``strategy`` to ``sample``; return ``(pairs, errors)``.

    Never raises for bad data: non-dict rows, missing fields, and no-match
    regexes become error strings so the caller sees partial results plus
    reasons. ``transformation`` is accepted by the schema but unsupported here.
    """
    if strategy.type == "direct_mapping":
        return _apply_direct_mapping(sample, strategy)
    if strategy.type == "regex":
        return _apply_regex(sample, strategy)
    return [], [f"Strategy type {strategy.type!r} is not supported for application."]


def _apply_direct_mapping(
    sample: list[dict[str, Any]] | str, strategy: ParsingStrategy
) -> tuple[list[dict[str, Any]], list[str]]:
    """Pull ``input_field``/``output_field`` from each dict row."""
    if strategy.input_field is None or strategy.output_field is None:
        return [], ["direct_mapping requires both input_field and output_field."]
    if isinstance(sample, str):
        return [], ["direct_mapping requires a list of rows, not raw text."]
    pairs: list[dict[str, Any]] = []
    errors: list[str] = []
    # ``sample`` is typed as rows, but this pure helper is defensive: a client
    # can hand it arbitrary JSON, so the element type is widened to run the guard.
    rows: list[Any] = sample
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"Row {index} is not an object; skipped.")
            continue
        if strategy.input_field not in row or strategy.output_field not in row:
            errors.append(f"Row {index} is missing input_field or output_field; skipped.")
            continue
        pairs.append({"input": row[strategy.input_field], "output": row[strategy.output_field]})
    return pairs, errors


def _apply_regex(
    sample: list[dict[str, Any]] | str, strategy: ParsingStrategy
) -> tuple[list[dict[str, Any]], list[str]]:
    """Run ``input_pattern``/``output_pattern`` over the text and pair matches."""
    if strategy.input_pattern is None or strategy.output_pattern is None:
        return [], ["regex strategy requires both input_pattern and output_pattern."]
    try:
        input_re = re.compile(strategy.input_pattern)
        output_re = re.compile(strategy.output_pattern)
    except re.error as exc:
        return [], [f"Invalid regex pattern: {exc}."]
    text = sample if isinstance(sample, str) else "\n".join(json.dumps(row) for row in sample)
    inputs = input_re.findall(text)
    outputs = output_re.findall(text)
    pairs = [
        {"input": value_in, "output": value_out}
        for value_in, value_out in zip(inputs, outputs, strict=False)
    ]
    errors: list[str] = []
    if not pairs:
        errors.append("The regex patterns matched no input/output pairs in the sample.")
    return pairs, errors
