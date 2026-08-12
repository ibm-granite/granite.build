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

Checks, in order:

1. ``metadata.json`` exists and reports at least one result file.
2. At least one ``.arrow`` file was produced.
3. Every ``.arrow`` file is readable as Arrow IPC and carries a ``tokens`` column.
4. No ``.arrow`` file is empty.
5. Each ``.arrow`` file has both ``meta/`` sidecars.
6. Per file: ``sum(.docs.ids counts) == .arrow row count == .docs summary total``,
   and the ``.docs`` document count equals the number of ``.docs.ids`` lines.
7. Document IDs are unique within a file.
8. Build-wide totals agree with ``metadata.json``'s ``num_tokens``.

Writes a JSON summary to ``<report_dir>/validation.json`` and exits non-zero if
any check failed, which fails the build target.

Usage:
    python validate_tokens.py <tokenize_output_dir> <report_dir>
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

import pyarrow as pa

# "pq01.parquet, documents: 3, tokens: 45"
_DOCS_SUMMARY_RE = re.compile(
    r"^(?P<file>.*?),\s*documents:\s*(?P<documents>\d+),\s*tokens:\s*(?P<tokens>\d+)\s*$"
)


def _read_arrow(path: pathlib.Path) -> pa.Table:
    """Read an Arrow IPC file, trying the file format then the stream format."""
    with pa.memory_map(str(path), "rb") as source:
        try:
            return pa.ipc.open_file(source).read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            return pa.ipc.open_stream(source).read_all()


def _parse_docs_ids(path: pathlib.Path) -> tuple[list[str], list[int], list[str]]:
    """Parse ``meta/<name>.docs.ids`` into (doc_ids, token_counts, errors)."""
    doc_ids: list[str] = []
    counts: list[int] = []
    errors: list[str] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
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
    text = path.read_text().strip()
    match = _DOCS_SUMMARY_RE.match(text)
    if not match:
        return None, None, [f"{path.name}: unparseable summary line {text!r}"]
    return int(match.group("documents")), int(match.group("tokens")), []


def validate(src: pathlib.Path) -> tuple[dict, list[str]]:
    """Run every check under ``src``, returning (summary, errors)."""
    errors: list[str] = []
    summary: dict = {"source": str(src)}

    meta_path = src / "metadata.json"
    declared_tokens: int | None = None
    if not meta_path.is_file():
        errors.append(f"missing metadata.json under {src}")
    else:
        try:
            stats = json.loads(meta_path.read_text()).get("job_output_stats", {})
        except json.JSONDecodeError as exc:
            stats = {}
            errors.append(f"metadata.json is not valid JSON: {exc}")
        summary["stats"] = stats
        if stats:
            if stats.get("result_files", 0) < 1:
                errors.append(f"transform produced no result files: {stats}")
            declared_tokens = stats.get("num_tokens")

    arrow_files = sorted(p for p in src.rglob("*.arrow") if "/meta/" not in str(p))
    summary["arrow_files"] = len(arrow_files)
    if not arrow_files:
        errors.append(f"no .arrow files found under {src}")

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
            duplicates = {d for d in doc_ids if doc_ids.count(d) > 1}
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

    if declared_tokens is not None and declared_tokens != total_tokens:
        errors.append(
            f"metadata.json reports num_tokens={declared_tokens} but the arrow "
            f"files hold {total_tokens} tokens in total"
        )

    return summary, errors


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <tokenize_output_dir> <report_dir>", file=sys.stderr)
        return 2

    src = pathlib.Path(argv[1])
    report_dir = pathlib.Path(argv[2])

    if not src.is_dir():
        print(f"VALIDATION FAILED: input dir does not exist: {src}", file=sys.stderr)
        return 1

    summary, errors = validate(src)
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

    print(
        f"VALIDATION PASSED: {summary['arrow_files']} arrow file(s), "
        f"{summary['total_documents']} documents, "
        f"{summary['total_tokens']} tokens"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
