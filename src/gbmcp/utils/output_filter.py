import re
from dataclasses import dataclass

from fastmcp.utilities.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GrepOptions:
    """Parsed grep flags and pattern."""

    pattern: str = ""
    before: int = 0
    after: int = 0
    ignore_case: bool = False
    invert_match: bool = False
    word_regexp: bool = False
    line_regexp: bool = False
    fixed_strings: bool = False
    count: bool = False
    line_number: bool = False
    only_matching: bool = False
    max_count: int | None = None


def _parse_grep_args(grep_str: str) -> GrepOptions:
    """Parse a grep-style string into a GrepOptions object.

    Supports flags before the pattern:
      -Cn / --context=N      context lines before+after
      -An / --after-context=N  lines after match
      -Bn / --before-context=N lines before match
      -i  / --ignore-case    case-insensitive matching
      -v  / --invert-match   select non-matching lines
      -w  / --word-regexp    match whole words only
      -x  / --line-regexp    match whole lines only
      -F  / --fixed-strings  treat pattern as literal string
      -E  / --extended-regexp  accepted silently (no-op)
      -c  / --count          return count of matching lines
      -n  / --line-number    prefix lines with line numbers
      -o  / --only-matching  print only matched parts
      -mN / --max-count=N    stop after N matches

    Example: "-C3 -i ERROR" or "-v -F literal.string" or "--invert-match pattern".
    """
    opts = GrepOptions()
    remaining = grep_str.strip()

    long_flag_re = re.compile(r"^(--[a-z][a-z-]*(?:=\d+)?)\s*")
    short_flag_re = re.compile(r"^(-[CABivwxFEcnom]\d*)\s*")

    _long_flag_map = {
        "--context": "C",
        "--after-context": "A",
        "--before-context": "B",
        "--ignore-case": "i",
        "--invert-match": "v",
        "--word-regexp": "w",
        "--line-regexp": "x",
        "--fixed-strings": "F",
        "--extended-regexp": "E",
        "--count": "c",
        "--line-number": "n",
        "--only-matching": "o",
        "--max-count": "m",
    }

    while remaining:
        m = long_flag_re.match(remaining)
        if m:
            token = m.group(1)
            remaining = remaining[m.end() :]
            if "=" in token:
                flag_name, _, val_str = token.partition("=")
                n = int(val_str)
            else:
                flag_name = token
                n = None
            short = _long_flag_map.get(flag_name)
            if short is None:
                # Unknown long flag — treat rest as pattern
                remaining = token + (" " + remaining if remaining else "")
                break
            _apply_flag(opts, short, n)
            continue

        m = short_flag_re.match(remaining)
        if m:
            token = m.group(1)
            remaining = remaining[m.end() :]
            fl = token.lstrip("-")
            letter = fl[0]  # preserve case: C vs c, A vs a, B vs b
            num_str = fl[1:]
            n = int(num_str) if num_str else None
            _apply_flag(opts, letter, n)
            continue

        break

    opts.pattern = remaining.strip()
    return opts


def _apply_flag(opts: GrepOptions, letter: str, n: int | None) -> None:
    """Apply a single parsed flag character (case-sensitive) to opts.

    Uppercase C/A/B are the context flags (with numeric args).
    Lowercase letters are the boolean/count flags.
    """
    match letter:
        case "C":
            v = n or 0
            opts.before = v
            opts.after = v
        case "A":
            opts.after = n or 0
        case "B":
            opts.before = n or 0
        case "i":
            opts.ignore_case = True
        case "v":
            opts.invert_match = True
        case "w":
            opts.word_regexp = True
        case "x":
            opts.line_regexp = True
        case "F":
            opts.fixed_strings = True
        case "E":
            pass  # no-op: Python re is already ERE-like
        case "c":
            opts.count = True
        case "n":
            opts.line_number = True
        case "o":
            opts.only_matching = True
        case "m":
            opts.max_count = n if n is not None else None


def _apply_grep(lines: list[str], opts: GrepOptions) -> list[str]:
    """Return lines matching opts.pattern with context. Groups separated by '--'.

    Supports: invert_match, word_regexp, line_regexp, fixed_strings,
    count, line_number, only_matching, max_count.
    """
    # Build effective regex pattern
    pattern = opts.pattern
    if opts.fixed_strings:
        pattern = re.escape(pattern)
    if opts.word_regexp:
        pattern = r"\b" + pattern + r"\b"
    if opts.line_regexp:
        pattern = r"^(?:" + pattern + r")$"

    try:
        flags = re.IGNORECASE if opts.ignore_case else 0
        regex = re.compile(pattern, flags)
    except re.error as e:
        return [f"grep error: invalid pattern '{opts.pattern}': {e}"]

    if not lines:
        return []

    # Find all matching line indices
    match_indices: list[int] = []
    for i, line in enumerate(lines):
        hit = bool(regex.search(line))
        if opts.invert_match:
            hit = not hit
        if hit:
            if opts.max_count is not None and len(match_indices) >= opts.max_count:
                break
            match_indices.append(i)

    if not match_indices:
        return []

    # -c: return count of matching lines (or occurrences for -o -c)
    if opts.count:
        if opts.only_matching and not opts.invert_match:
            total = sum(len(regex.findall(lines[i])) for i in match_indices)
            return [str(total)]
        return [str(len(match_indices))]

    n = len(lines)

    # Build ranges [start, end) for each match with context
    ranges: list[tuple[int, int]] = []
    for idx in match_indices:
        start = max(0, idx - opts.before)
        end = min(n, idx + opts.after + 1)
        ranges.append((start, end))

    # Merge overlapping/adjacent ranges
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    match_index_set = set(match_indices)

    # Collect output lines with "--" separators between non-adjacent groups
    result: list[str] = []
    for i, (start, end) in enumerate(merged):
        if i > 0:
            result.append("--")
        for line_idx in range(start, end):
            line = lines[line_idx]
            orig_num = line_idx + 1  # 1-based

            is_match = line_idx in match_index_set

            if opts.only_matching and is_match and not opts.invert_match:
                for match_obj in regex.finditer(line):
                    entry = match_obj.group(0)
                    if opts.line_number:
                        entry = f"{orig_num}:{entry}"
                    result.append(entry)
            else:
                if opts.line_number:
                    sep = ":" if is_match else "-"
                    line = f"{orig_num}{sep}{line}"
                result.append(line)

    return result


def apply_output_filters(
    output: str,
    *,
    tool_name: str = "",
    grep: str | None = None,
    wc: bool | None = None,
    head: int | None = None,
    tail: int | None = None,
) -> str:
    """Apply server-side output filters to a tool's string output.

    Processing order:
    1. If wc=True, return line/character count of the raw output (ignores grep/head/tail).
    2. Apply grep (filters lines by regex with optional context).
    3. Apply head or tail (mutually exclusive slicing).

    Args:
        output: The raw tool output string.
        tool_name: Tool name used in wc output label.
        grep: Grep-style filter string. Supports flags before the pattern:
            -Cn (n context lines), -An (n lines after), -Bn (n lines before),
            -i (case-insensitive), -v (invert match), -F (fixed strings),
            -w (word regexp), -x (line regexp), -c (count), -n (line numbers),
            -o (only matching), -mN (max N matches).
            Long forms also accepted: --invert-match, --fixed-strings, etc.
            Example: "-C2 -i error" or "-v -F literal.text".
        wc: If True, return only line and character count instead of full output.
        head: Return only the first N lines. Mutually exclusive with tail.
        tail: Return only the last N lines. Mutually exclusive with head.

    Returns:
        Filtered output string, or a count string if wc=True.
    """
    if head is not None and tail is not None:
        raise ValueError("head and tail are mutually exclusive")
    if head is not None and head <= 0:
        raise ValueError(f"head must be a positive integer, got {head}")
    if tail is not None and tail <= 0:
        raise ValueError(f"tail must be a positive integer, got {tail}")

    if wc:
        lines = output.split("\n")
        label = f"{tool_name} output" if tool_name else "output"
        logger.debug(
            f"apply_output_filters wc: {len(lines)} lines, {len(output)} chars"
        )
        return f"{label}: {len(lines)} lines ({len(output)} characters)"

    lines = output.split("\n")

    if grep is not None:
        opts = _parse_grep_args(grep)
        if not opts.pattern:
            logger.debug(
                "apply_output_filters: empty grep pattern, returning full output"
            )
        else:
            lines = _apply_grep(lines, opts)
            logger.debug(
                f"apply_output_filters grep='{grep}': {len(lines)} lines after filtering"
            )

    if head is not None:
        lines = lines[:head]
        logger.debug(f"apply_output_filters head={head}: {len(lines)} lines")
    elif tail is not None:
        lines = lines[-tail:]
        logger.debug(f"apply_output_filters tail={tail}: {len(lines)} lines")

    return "\n".join(lines)
