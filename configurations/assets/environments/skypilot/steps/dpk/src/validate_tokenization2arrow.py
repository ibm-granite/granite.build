#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Validate the output of the DPK ``tokenization2arrow`` transform.

Shipped to the worker by the ``validate`` target of the sibling ``build.yaml``
via SkyPilot ``file_mounts``, and run against the artifact bound from the
``tokenize`` target.

``tokenization2arrow`` writes, per input Parquet file:

* ``<name>.arrow`` — a single ``tokens: uint32`` column holding the
  concatenated token IDs of every document in that file.
* ``meta/<name>.docs`` — one summary line:
  ``<file>, documents: <n>, tokens: <n>``
* ``meta/<name>.docs.ids`` — one ``<document_id>, <token_count>`` line per
  document.

The token stream and its metadata are written separately, so the useful check is
that they agree: the sum of the per-document counts in ``.docs.ids`` must equal
the number of rows in the ``.arrow`` file, and both must match the ``.docs``
summary. That catches truncated, duplicated, or mis-ordered writes, which a
"file exists and is non-empty" check would not.

Consistency checks (always run), in order:

1. ``metadata.json`` exists and reports at least one result file.
2. At least one ``.arrow`` file was produced.
3. Every ``.arrow`` file is readable as Arrow IPC and carries a ``tokens`` column.
4. No ``.arrow`` file is empty.
5. Each ``.arrow`` file has both ``meta/`` sidecars.
6. Per file: ``sum(.docs.ids counts) == .arrow row count == .docs summary total``,
   and the ``.docs`` document count equals the number of ``.docs.ids`` lines.
7. Document IDs are unique within a file.
8. Build-wide totals agree with ``metadata.json``'s ``num_tokens``.

Completeness checks (only with ``--input``), which answer a different question —
*did every source file produce output?* rather than *is the output consistent?*:

9. Every non-empty source ``.parquet`` has an ``.arrow`` counterpart.
10. …and both ``meta/`` sidecars (``.docs``, ``.docs.ids``).

The two axes are complementary: consistency alone passes a run that silently
dropped whole files, and completeness alone passes a truncated ``.arrow`` whose
counterpart merely exists. Emptiness is decided by the Parquet **row count** (an
empty table produces no output, by design) rather than by a file-size threshold,
so no magic byte constants are involved.

This script is deliberately **read-only**: it reports and exits non-zero, and
never deletes or rewrites anything. Pipeline recovery (pruning partial output
before a re-run) is a separate, deliberate operation.

Writes a JSON summary to ``<report_dir>/validation.json`` and exits non-zero if
any check failed, which fails the build target.

Usage:
    python validate_tokenization2arrow.py <tokenize_output_dir> <report_dir>
                              [--input <source_parquet_dir>]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq

# "pq01.parquet, documents: 3, tokens: 45"
_DOCS_SUMMARY_RE = re.compile(
    r"^(?P<file>.*?),\s*documents:\s*(?P<documents>\d+),\s*tokens:\s*(?P<tokens>\d+)\s*$"
)


def _read_arrow(path: pathlib.Path) -> pa.Table:
    """Read an Arrow IPC file, trying the file format then the stream format.

    The fallback catches ``Exception`` rather than only ``ArrowInvalid``: which
    error ``open_file`` raises on a stream-format file is a pyarrow implementation
    detail (``ArrowInvalid`` for a bad magic number, but ``OSError`` /
    ``ArrowIOError`` when it reads a footer that isn't there), and catching only one
    of them reported "unreadable arrow file" for output pyarrow could in fact read —
    failing a healthy target. Nothing is swallowed: if the stream attempt also
    fails, that exception propagates to the caller, which records it per file.
    """
    with pa.memory_map(str(path), "rb") as source:
        try:
            return pa.ipc.open_file(source).read_all()
        except Exception:  # noqa: BLE001 - retry as a stream; see the docstring
            source.seek(0)
            return pa.ipc.open_stream(source).read_all()


def _read_text(path: pathlib.Path) -> tuple[str | None, str | None]:
    """Read a text file, returning (text, error-or-None) instead of raising.

    These files are DPK's output, not this step's, so their ENCODING is an
    assumption like their types were. A single invalid UTF-8 byte in a sidecar or in
    metadata.json raised UnicodeDecodeError out of validate() and main() — a
    traceback with no validation.json written, which is the same failure the
    metadata.json type guards were added to prevent, one layer earlier. Decode errors
    are surfaced as ordinary findings so the report is still produced.

    Read as bytes then decoded explicitly: read_text() would apply the platform's
    locale encoding, so the same corrupt file could pass on one node and fail on
    another.
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, f"{path.name}: unreadable: {exc}"
    try:
        return raw.decode("utf-8"), None
    except UnicodeDecodeError as exc:
        return None, f"{path.name}: not valid UTF-8: {exc}"


def _parse_docs_ids(path: pathlib.Path) -> tuple[list[str], list[int], list[str]]:
    """Parse ``meta/<name>.docs.ids`` into (doc_ids, token_counts, errors)."""
    doc_ids: list[str] = []
    counts: list[int] = []
    errors: list[str] = []
    text, err = _read_text(path)
    if err is not None:
        return doc_ids, counts, [err]
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        doc_id, _, count = line.rpartition(",")
        if not doc_id:
            errors.append(f"{path.name}:{lineno}: malformed line {raw!r}")
            continue
        try:
            counts.append(int(count.strip()))
        except ValueError:
            errors.append(f"{path.name}:{lineno}: non-integer token count {count!r}")
            continue
        doc_ids.append(doc_id.strip())
    return doc_ids, counts, errors


def _parse_docs_summary(path: pathlib.Path) -> tuple[int | None, int | None, list[str]]:
    """Parse ``meta/<name>.docs`` into (documents, tokens, errors)."""
    text, err = _read_text(path)
    if err is not None:
        return None, None, [err]
    text = text.strip()
    match = _DOCS_SUMMARY_RE.match(text)
    if not match:
        return None, None, [f"{path.name}: unparseable summary line {text!r}"]
    return int(match.group("documents")), int(match.group("tokens")), []


def _as_int(stats: dict, key: str) -> tuple[int | None, str | None]:
    """Read ``stats[key]`` as an int, returning (value, error-or-None).

    ``metadata.json`` is DPK's output, not this step's, so its leaf types are an
    assumption rather than a guarantee — and an assumption that, when wrong, took
    the whole run down with a TypeError before any report was written. A missing
    key is not an error (returns ``(None, None)``): both callers treat absent as
    "nothing to check". A present-but-not-a-number one is reported like any other
    finding.

    ``bool`` is rejected explicitly: it is a subclass of ``int``, so ``True``
    would otherwise sail through as ``1``.
    """
    if key not in stats:
        return None, None
    value = stats[key]
    if isinstance(value, bool) or not isinstance(value, int):
        return None, (
            f"metadata.json '{key}' is {type(value).__name__}, expected an integer"
        )
    return value, None


def _parquet_is_empty(path: pathlib.Path) -> bool:
    """True if a source Parquet holds no rows.

    The transform skips empty tables (reporting them as ``skipped empty tables``),
    so an empty input legitimately produces no ``.arrow``/``meta`` output and must
    not be counted as missing. Row count is read from the Parquet footer — cheap,
    and exact where a file-size threshold would only be a guess.
    """
    try:
        return pq.ParquetFile(str(path)).metadata.num_rows == 0
    except Exception:  # noqa: BLE001 - an unreadable input is not "empty"
        return False


def check_completeness(
    src: pathlib.Path, input_dir: pathlib.Path
) -> tuple[dict, list[str]]:
    """Assert every non-empty source Parquet produced its three output files.

    Answers "did anything get dropped?", which the consistency checks cannot see:
    they only inspect the output that exists, so a run that silently skipped whole
    files still passes them.

    Output paths mirror the input tree with the ``.parquet`` suffix replaced, so
    only the FINAL suffix is swapped — ``pq03.snappy.parquet`` pairs with
    ``pq03.snappy.arrow`` / ``pq03.snappy.docs``. (Stripping every suffix would
    mis-derive ``pq03.docs`` and report false failures.)

    :param src: the transform's output directory.
    :param input_dir: the source Parquet directory.
    :returns: (summary, errors)
    """
    errors: list[str] = []
    summary: dict = {"input": str(input_dir)}

    sources = sorted(input_dir.rglob("*.parquet"))
    if not sources:
        errors.append(f"no .parquet files found under {input_dir}")

    empty: list[str] = []
    missing_arrow: list[str] = []
    missing_docs: list[str] = []
    missing_docs_ids: list[str] = []

    for path in sources:
        rel = path.relative_to(input_dir)
        if _parquet_is_empty(path):
            empty.append(str(rel))
            continue

        stem = rel.name[: -len(".parquet")]
        if not (src / rel.parent / f"{stem}.arrow").is_file():
            missing_arrow.append(str(rel))
        if not (src / "meta" / rel.parent / f"{stem}.docs").is_file():
            missing_docs.append(str(rel))
        if not (src / "meta" / rel.parent / f"{stem}.docs.ids").is_file():
            missing_docs_ids.append(str(rel))

    summary["source_parquet"] = len(sources)
    summary["empty_parquet"] = len(empty)
    summary["empty_parquet_files"] = empty
    summary["expected_outputs"] = len(sources) - len(empty)
    summary["missing_arrow"] = missing_arrow
    summary["missing_docs"] = missing_docs
    summary["missing_docs_ids"] = missing_docs_ids

    for label, missing in (
        (".arrow", missing_arrow),
        ("meta/.docs", missing_docs),
        ("meta/.docs.ids", missing_docs_ids),
    ):
        if missing:
            errors.append(
                f"{len(missing)} of {summary['expected_outputs']} non-empty source "
                f"file(s) produced no {label}: {missing[:5]}"
                + (" ..." if len(missing) > 5 else "")
            )

    return summary, errors


def validate(src: pathlib.Path) -> tuple[dict, list[str]]:
    """Run every check under ``src``, returning (summary, errors)."""
    errors: list[str] = []
    # Errors that mean "the transform produced no output". Collected separately
    # because that is CORRECT for an all-empty corpus, and validate() cannot tell:
    # it never sees the input. main() decides, then folds these in or withdraws them.
    # Identity, not string matching — keying the withdrawal on the message text meant
    # a reworded error silently stopped being withdrawn.
    no_output: list[str] = []
    summary: dict = {"source": str(src)}

    meta_path = src / "metadata.json"
    declared_tokens: int | None = None
    if not meta_path.is_file():
        errors.append(f"missing metadata.json under {src}")
    else:
        stats: dict = {}
        # The DECODE happens before the JSON parse and has its own failure mode:
        # UnicodeDecodeError is not a JSONDecodeError, so invalid UTF-8 escaped this
        # guard entirely and took main() down before any report was written.
        meta_text, decode_err = _read_text(meta_path)
        parsed = False
        loaded = None
        if decode_err is not None:
            errors.append(decode_err)
        else:
            try:
                loaded = json.loads(meta_text)
                parsed = True
            except json.JSONDecodeError as exc:
                errors.append(f"metadata.json is not valid JSON: {exc}")
        # `parsed` rather than `loaded is not None`, so a literal `null` still reaches
        # the type check below and is reported as "is NoneType, expected an object".
        if parsed:
            # Valid JSON is not necessarily an OBJECT: `[]`, `null`, `"x"` and `3`
            # all parse. Calling .get() on those raises AttributeError, which would
            # escape main() — no validation.json written, a traceback instead of the
            # report this script promises. Report it as a failure like any other.
            if isinstance(loaded, dict):
                got = loaded.get("job_output_stats", {})
                if isinstance(got, dict):
                    stats = got
                else:
                    errors.append(
                        f"metadata.json 'job_output_stats' is "
                        f"{type(got).__name__}, expected an object"
                    )
            else:
                errors.append(
                    f"metadata.json is {type(loaded).__name__}, expected an object"
                )
        summary["stats"] = stats
        if stats:
            # The leaf values need the same type check the containers above got:
            # `stats["result_files"]` of `null` or `"5"` is valid JSON, and
            # comparing it to an int raises TypeError — which escaped main() and
            # killed the run BEFORE validation.json was written, the exact failure
            # the container guards exist to prevent.
            result_files, err = _as_int(stats, "result_files")
            if err:
                errors.append(err)
            elif result_files is not None and result_files < 1:
                # TAGGED as a no-output error (see summary["no_output_errors"]): an
                # all-empty corpus legitimately produces zero result files, and DPK
                # publishes exactly "result_files": 0 in that case
                # (transform_file_processor.py's `case 0`). Only main(), which can see
                # the input, can tell that apart from a real failure.
                no_output.append(f"transform produced no result files: {stats}")
            # num_tokens is compared with `!=`, which never raises — so a string
            # "85" would not crash, it would silently report a mismatch against
            # the int 85 and fail a perfectly good run. A wrong diagnosis is worse
            # than none, so it is coerced through the same helper.
            declared_tokens, err = _as_int(stats, "num_tokens")
            if err:
                errors.append(err)

    # Exclude the meta/ sidecar tree, testing the path RELATIVE to src: matching
    # "/meta/" against the absolute path also excludes every file when src itself
    # sits under a directory named meta (a plausible output_path like
    # /shared/meta/tokens), which reported "no .arrow files found" for output that
    # was in fact fine.
    # Only the FIRST component is tested, not any component. The sidecar tree DPK
    # writes is exactly src/meta/..., so a source subdirectory named meta (giving
    # out/meta/x.arrow, since DPK mirrors the input tree) is real output — and
    # excluding it made those files invisible to every consistency check while
    # check_completeness, which applies no such filter, still found them and passed.
    # A truncated arrow file there would have gone unvalidated on a green target.
    arrow_files = sorted(
        p for p in src.rglob("*.arrow") if p.relative_to(src).parts[:1] != ("meta",)
    )
    summary["arrow_files"] = len(arrow_files)
    if not arrow_files:
        # Also tagged: same cause, same caveat.
        no_output.append(f"no .arrow files found under {src}")

    total_tokens = 0
    total_documents = 0
    per_file: list[dict] = []

    for path in arrow_files:
        name = path.relative_to(src)
        entry: dict = {"file": str(name)}

        try:
            table = _read_arrow(path)
        except Exception as exc:  # noqa: BLE001 - report any read failure
            errors.append(f"{name}: unreadable arrow file: {exc}")
            per_file.append({**entry, "error": "unreadable"})
            continue

        if "tokens" not in table.column_names:
            errors.append(
                f"{name}: missing 'tokens' column (found {table.column_names})"
            )
            per_file.append({**entry, "error": "missing tokens column"})
            continue

        arrow_rows = table.num_rows
        entry["arrow_tokens"] = arrow_rows
        if arrow_rows == 0:
            errors.append(f"{name}: contains zero tokens")

        # meta/ sidecars mirror the output tree with only the final ".arrow"
        # suffix replaced — "pq03.snappy.arrow" pairs with "pq03.snappy.docs",
        # so strip exactly that suffix rather than using with_suffix(), which
        # would also eat the ".snappy" part of a multi-suffix stem.
        stem = name.name[: -len(".arrow")]
        meta_dir = src / "meta" / name.parent
        ids_path = meta_dir / f"{stem}.docs.ids"
        docs_path = meta_dir / f"{stem}.docs"

        if not ids_path.is_file():
            errors.append(f"{name}: missing sidecar {ids_path.relative_to(src)}")
        if not docs_path.is_file():
            errors.append(f"{name}: missing sidecar {docs_path.relative_to(src)}")

        if ids_path.is_file():
            doc_ids, counts, parse_errors = _parse_docs_ids(ids_path)
            errors.extend(parse_errors)
            entry["documents"] = len(doc_ids)
            entry["docs_ids_tokens"] = sum(counts)

            if sum(counts) != arrow_rows:
                errors.append(
                    f"{name}: token count mismatch — .docs.ids sums to "
                    f"{sum(counts)} but the arrow file holds {arrow_rows} tokens"
                )
            # Counter, not `doc_ids.count(d)` per element: that was O(n^2) and this
            # runs once per .arrow file. Measured on the old form: 0.9s at 10k
            # document ids, 7s at 30k — and real corpora exceed that per file, which
            # made a cheap consistency check the dominant cost of the target.
            duplicates = {d for d, n in Counter(doc_ids).items() if n > 1}
            if duplicates:
                errors.append(
                    f"{name}: duplicate document ids in .docs.ids: {sorted(duplicates)}"
                )
            if any(c <= 0 for c in counts):
                errors.append(f"{name}: non-positive token count in .docs.ids")

            total_documents += len(doc_ids)

            if docs_path.is_file():
                docs_n, docs_tokens, parse_errors = _parse_docs_summary(docs_path)
                errors.extend(parse_errors)
                if docs_tokens is not None and docs_tokens != arrow_rows:
                    errors.append(
                        f"{name}: .docs summary claims {docs_tokens} tokens but "
                        f"the arrow file holds {arrow_rows}"
                    )
                if docs_n is not None and docs_n != len(doc_ids):
                    errors.append(
                        f"{name}: .docs summary claims {docs_n} documents but "
                        f".docs.ids lists {len(doc_ids)}"
                    )

        total_tokens += arrow_rows
        per_file.append(entry)

    summary["per_file"] = per_file
    summary["total_documents"] = total_documents
    summary["total_tokens"] = total_tokens

    summary["no_output_errors"] = no_output

    if declared_tokens is not None and declared_tokens != total_tokens:
        errors.append(
            f"metadata.json reports num_tokens={declared_tokens} but the arrow "
            f"files hold {total_tokens} tokens in total"
        )

    # Default is STRICT: no output is a failure unless a caller proves otherwise.
    # Without --input there is no way to prove it, so validate() alone behaves exactly
    # as before.
    return summary, errors + no_output


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(argv[0]).name if argv else "validate_tokenization2arrow.py",
        description="Validate DPK tokenization2arrow output (read-only).",
    )
    parser.add_argument("output_dir", help="the transform's output directory")
    parser.add_argument("report_dir", help="where validation.json is written")
    parser.add_argument(
        "--input",
        dest="input_dir",
        default=None,
        help="source Parquet directory; enables the completeness checks "
        "(every non-empty source file produced .arrow + both meta sidecars)",
    )
    # argparse exits 2 on a usage error, matching the previous contract.
    args = parser.parse_args(argv[1:])

    src = pathlib.Path(args.output_dir)
    report_dir = pathlib.Path(args.report_dir)

    if not src.is_dir():
        print(f"VALIDATION FAILED: input dir does not exist: {src}", file=sys.stderr)
        return 1

    summary, errors = validate(src)

    if args.input_dir is not None:
        input_dir = pathlib.Path(args.input_dir)
        if not input_dir.is_dir():
            print(
                f"VALIDATION FAILED: --input dir does not exist: {input_dir}",
                file=sys.stderr,
            )
            return 1
        completeness, completeness_errors = check_completeness(src, input_dir)
        summary["completeness"] = completeness
        errors = errors + completeness_errors

        # An ALL-EMPTY corpus legitimately produces no output at all: the transform
        # reports "skipped empty tables" and writes nothing, and DPK publishes
        # "result_files": 0 for it. validate() cannot know that — it never sees the
        # input — so it flagged both, failing a target that behaved correctly. Only
        # here, with both results in hand, is the distinction visible: sources exist,
        # every one is empty, so zero outputs were expected.
        #
        # ALL the no-output errors are withdrawn together, by identity. Withdrawing
        # only "no .arrow files found" by string match left "transform produced no
        # result files" in place, so the target still failed — and the test missed it
        # because its hand-written metadata.json omitted result_files, a value real
        # DPK always writes.
        #
        # Still narrow: only when check_completeness found at least one source file
        # and expects no output, so a genuinely dropped file keeps failing
        # (expected_outputs > 0 withdraws nothing). An empty input DIRECTORY is
        # untouched — "no .parquet files found" is a real fault and is not tagged.
        no_output = summary.get("no_output_errors", [])
        if (
            no_output
            and completeness.get("source_parquet", 0) > 0
            and completeness.get("expected_outputs", 0) == 0
        ):
            errors = [e for e in errors if e not in no_output]
            summary["note"] = (
                "every source file was empty, so no output was expected; "
                f"{len(no_output)} no-output error(s) were not counted as failures"
            )

    summary["errors"] = errors

    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "validation.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

    if errors:
        print(f"VALIDATION FAILED with {len(errors)} error(s)")
        for err in errors:
            print(f"  - {err}")
        return 1

    passed = (
        f"VALIDATION PASSED: {summary['arrow_files']} arrow file(s), "
        f"{summary['total_documents']} documents, "
        f"{summary['total_tokens']} tokens"
    )
    if "completeness" in summary:
        c = summary["completeness"]
        passed += (
            f"; completeness: {c['expected_outputs']} of {c['source_parquet']} "
            f"source file(s) expected output ({c['empty_parquet']} empty), all present"
        )
    print(passed)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
