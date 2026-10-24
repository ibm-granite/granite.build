"""Recover `ibm-granite/granite-4.2-30b-fp8` tool calls whose `</tool_call>`
closing tag never got generated.

`Granite4FCHandler._extract_tool_calls` (bfcl_eval's own, unmodified) requires
a literal `<tool_call>\\n...\\n</tool_call>` match. In practice this
checkpoint frequently stops (or opens a second `<tool_call>`, or drifts into
unrelated template tokens) right after the JSON payload's closing brace,
without ever emitting the closing tag -- observed in ~88% of otherwise
correctly-targeted calls in the original full run. This recovers those by
inserting the missing tag right after the first balanced JSON object
following each `<tool_call>` occurrence, then handing off to the untouched
strict regex.

Second, related fix: back-to-back `<tool_call>` tags with no real close in
between (the model opening a second call instead of closing the first) can
make the repair above insert a `</tool_call>` after a balanced JSON object
that isn't actually a `{"name": ..., "arguments": ...}` call -- e.g. a
fragment like `{"arguments": {...}}` with no `name` key, left over from a
previous, differently-shaped object. bfcl_eval's own `_extract_tool_calls`
only checks that each match is valid JSON, not that it has the right shape,
so a malformed-but-parseable dict like that flows straight into chat history
and crashes `Granite4FCHandler._format_prompt` on the *next* turn with
`KeyError: 'name'` (or `'arguments'`) -- a multi-turn/memory-category crash,
not merely a missed call. `_extract_tool_calls_with_repair` below filters the
original method's output to well-formed `{"name": <str>, "arguments": ...}`
dicts only, silently dropping anything else -- the same tolerance the
original method already applies to JSON that fails to parse at all.
"""

from __future__ import annotations

import re

_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"


def find_balanced_json_end(text: str, start: int) -> int | None:
    """Return the index just past the balanced `{...}` beginning at
    `text[start]` (which must be `'{'`), honoring string literals/escapes so
    a `}` inside a quoted argument value doesn't end the match early.
    Returns None if the braces never balance before the string ends.
    """
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def repair_unclosed_tool_calls(text: str) -> str:
    """Insert a missing `</tool_call>` after the first balanced JSON object
    following each `<tool_call>` tag that isn't already properly closed.

    Tags already followed by a valid close, or followed by nothing
    JSON-parseable at all, are left exactly as-is -- this only fills in the
    one specific gap the model leaves.
    """
    out: list[str] = []
    i = 0
    while True:
        idx = text.find(_TOOL_CALL_OPEN, i)
        if idx == -1:
            out.append(text[i:])
            break
        out.append(text[i:idx])
        seg_start = idx + len(_TOOL_CALL_OPEN)

        brace_idx = text.find("{", seg_start)
        if brace_idx == -1:
            out.append(text[idx:seg_start])
            i = seg_start
            continue

        end = find_balanced_json_end(text, brace_idx)
        if end is None:
            out.append(text[idx:seg_start])
            i = seg_start
            continue

        json_text = text[brace_idx:end]
        already_closed = (
            re.match(r"\s*" + re.escape(_TOOL_CALL_CLOSE), text[end : end + 32])
            is not None
        )
        out.append(
            _TOOL_CALL_OPEN
            + "\n"
            + json_text
            + ("" if already_closed else "\n" + _TOOL_CALL_CLOSE)
        )
        i = end

    return "".join(out)


def is_well_formed_tool_call(obj) -> bool:
    """True iff `obj` is shaped like a real tool call: a dict with a
    non-empty string `name` and an `arguments` key present (its value may be
    a dict or a JSON-encoded string -- both are handled downstream in
    `_format_prompt`/`decode_ast`, so it isn't validated further here).
    """
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("name"), str)
        and obj.get("name") != ""
        and "arguments" in obj
    )


def patch_granite_handler(handler_cls) -> None:
    """Wrap `handler_cls._extract_tool_calls` (a staticmethod) so it repairs
    unclosed `<tool_call>` blocks before delegating to the original, unmodified
    extraction regex, then drops any result that parsed as JSON but isn't
    actually a well-formed `{"name": ..., "arguments": ...}` call. Covers
    `_parse_query_response_prompting`, `decode_ast`, and `decode_execute`,
    which all funnel through this one method.
    """
    original_extract_tool_calls = handler_cls._extract_tool_calls

    def _extract_tool_calls_with_repair(input_string):
        extracted = original_extract_tool_calls(
            repair_unclosed_tool_calls(input_string)
        )
        return [call for call in extracted if is_well_formed_tool_call(call)]

    handler_cls._extract_tool_calls = staticmethod(_extract_tool_calls_with_repair)
