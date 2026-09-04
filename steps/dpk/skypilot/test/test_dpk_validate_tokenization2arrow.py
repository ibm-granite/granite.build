#!/usr/bin/env python3

# Copyright LLM.build Authors
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

"""Unit tests for the `dpk` step's bundled output validator.

The validator (``steps/dpk/skypilot/src/validate_tokenization2arrow.py``) is shipped to a
SkyPilot worker via the step's ``file_mounts: {src: src}`` and is not part of the
gbserver package, so it is loaded here by path via importlib.

This is a **cluster-agnostic** unit test, so it lives at the root of the step's
``test/`` dir rather than in a per-cluster subdir — which means it runs in Mode 1
only (``make -C steps/dpk/skypilot test``) and is not copied to ``test/steps/``
by ``make publish-step``. Same placement as ``eval/skypilot/test/test_eval.py``.

These tests build synthetic ``tokenization2arrow`` output trees, so they cover
the corruption cases that matter without needing DPK, a tokenizer download, or a
cluster. Two independent axes are checked, because each misses what the other
catches:

* **consistency** — the Arrow token stream agrees with its ``meta/`` sidecars.
  A validator that only ever passes is worthless, so every case below asserts a
  *failure* for a way a truncated, duplicated, or mis-ordered write shows up.
* **completeness** (``--input``) — every non-empty source Parquet actually
  produced output. Consistency alone passes a run that silently dropped whole
  files; see ``test_consistency_alone_misses_a_dropped_file``.
"""

import importlib.util
import json
import pathlib

import pytest

pa = pytest.importorskip(
    "pyarrow", reason="pyarrow is required to write Arrow fixtures"
)

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _STEP_DIR / "src" / "validate_tokenization2arrow.py"


def _load_validator():
    """Import the standalone validator script by path."""
    spec = importlib.util.spec_from_file_location(
        "validate_tokenization2arrow", _SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


# --------------------------------------------------------------------------- #
# Fixture builders — synthesize a tokenization2arrow output tree
# --------------------------------------------------------------------------- #


def _write_arrow(path: pathlib.Path, token_count: int) -> None:
    """Write an .arrow file holding ``token_count`` uint32 tokens."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"tokens": pa.array(range(token_count), type=pa.uint32())})
    with pa.OSFile(str(path), "wb") as sink:
        writer = pa.ipc.new_file(sink, table.schema)
        writer.write_table(table)
        writer.close()


def _write_sidecars(
    root: pathlib.Path,
    rel_stem: str,
    docs: list[tuple[str, int]],
    *,
    summary_docs: int | None = None,
    summary_tokens: int | None = None,
) -> None:
    """Write the meta/<stem>.docs and meta/<stem>.docs.ids sidecars."""
    meta = root / "meta" / rel_stem
    meta.parent.mkdir(parents=True, exist_ok=True)
    total = sum(c for _, c in docs)
    n_docs = summary_docs if summary_docs is not None else len(docs)
    n_tokens = summary_tokens if summary_tokens is not None else total
    meta.with_name(meta.name + ".docs").write_text(
        f"{pathlib.Path(rel_stem).name}.parquet, documents: {n_docs}, tokens: {n_tokens}\n"
    )
    meta.with_name(meta.name + ".docs.ids").write_text(
        "".join(f"{doc_id}, {count}\n" for doc_id, count in docs)
    )


def _build_tree(
    root: pathlib.Path,
    files: dict[str, list[tuple[str, int]]],
    *,
    num_tokens: int | None = None,
) -> pathlib.Path:
    """Build a well-formed output tree.

    :param files: maps a relative stem (no ``.arrow``) to its document list of
        ``(document_id, token_count)`` pairs.
    """
    root.mkdir(parents=True, exist_ok=True)
    total = 0
    for stem, docs in files.items():
        count = sum(c for _, c in docs)
        _write_arrow(root / f"{stem}.arrow", count)
        _write_sidecars(root, stem, docs)
        total += count
    stats = {
        "result_files": len(files),
        "num_tokens": total if num_tokens is None else num_tokens,
    }
    (root / "metadata.json").write_text(json.dumps({"job_output_stats": stats}))
    return root


@pytest.fixture
def good_tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """A valid two-file output tree, one with a multi-suffix stem."""
    return _build_tree(
        tmp_path / "out",
        {
            "lang=en/pq01": [("d01", 12), ("d02", 16), ("d03", 17)],
            # multi-suffix stem: sidecars are pq03.snappy.docs, NOT pq03.docs
            "lang=en/dataset=cyber/pq03.snappy": [("d10", 12)],
        },
    )


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


class TestValidatorAcceptsGoodOutput:
    def test_valid_tree_has_no_errors(self, good_tree):
        summary, errors = validator.validate(good_tree)
        assert errors == []
        assert summary["arrow_files"] == 2
        assert summary["total_documents"] == 4
        assert summary["total_tokens"] == 57

    def test_multi_suffix_stem_resolves_its_sidecars(self, good_tree):
        """pq03.snappy.arrow pairs with pq03.snappy.docs, not pq03.docs.

        Regression guard: ``Path.with_suffix('')`` would strip ``.snappy`` too and
        report both sidecars missing.
        """
        _, errors = validator.validate(good_tree)
        assert not [e for e in errors if "missing sidecar" in e]

    def test_main_exits_zero_and_writes_report(self, good_tree, tmp_path, capsys):
        report_dir = tmp_path / "report"
        rc = validator.main(
            ["validate_tokenization2arrow.py", str(good_tree), str(report_dir)]
        )
        assert rc == 0
        assert "VALIDATION PASSED" in capsys.readouterr().out

        written = json.loads((report_dir / "validation.json").read_text())
        assert written["errors"] == []
        assert written["total_tokens"] == 57


# --------------------------------------------------------------------------- #
# Corruption cases — each must be caught
# --------------------------------------------------------------------------- #


class TestValidatorRejectsCorruptOutput:
    def test_truncated_token_stream(self, good_tree):
        """Arrow holds fewer tokens than the sidecars claim."""
        _write_arrow(good_tree / "lang=en/pq01.arrow", 10)  # was 45
        _, errors = validator.validate(good_tree)
        assert any("token count mismatch" in e for e in errors)

    def test_missing_docs_ids_sidecar(self, good_tree):
        (good_tree / "meta/lang=en/pq01.docs.ids").unlink()
        _, errors = validator.validate(good_tree)
        assert any("missing sidecar" in e and "docs.ids" in e for e in errors)

    def test_missing_docs_summary_sidecar(self, good_tree):
        (good_tree / "meta/lang=en/pq01.docs").unlink()
        _, errors = validator.validate(good_tree)
        assert any(
            "missing sidecar" in e and e.rstrip().endswith(".docs") for e in errors
        )

    def test_duplicate_document_ids(self, good_tree):
        _write_sidecars(good_tree, "lang=en/pq01", [("d01", 20), ("d01", 25)])
        _, errors = validator.validate(good_tree)
        assert any("duplicate document ids" in e for e in errors)

    def test_duplicate_detection_scales_to_a_realistic_document_count(self, tmp_path):
        """Regression guard: the duplicate scan must not be O(n^2).

        It was `{d for d in doc_ids if doc_ids.count(d) > 1}` — one linear scan per
        element — which measured 0.9s at 10k document ids and 7s at 30k, PER .arrow
        file. Real corpora exceed that per file, so a cheap consistency check became
        the dominant cost of the target. Counter makes it linear.

        20k documents is well within what DPK writes per file and completes in
        milliseconds now; the generous ceiling keeps this from flaking on slow CI
        while still failing outright if the quadratic form returns (which took
        seconds at this size).
        """
        import time

        n = 20_000
        docs = [(f"d{i:06d}", 2) for i in range(n)]
        out = _build_tree(tmp_path / "big", {"pq01": docs})

        start = time.monotonic()
        _, errors = validator.validate(out)
        elapsed = time.monotonic() - start

        assert not any("duplicate document ids" in e for e in errors)
        assert elapsed < 5.0, f"validate() took {elapsed:.1f}s for {n} documents"

    def test_docs_summary_token_disagreement(self, good_tree):
        """The .docs summary total contradicts the arrow row count."""
        _write_sidecars(
            good_tree,
            "lang=en/pq01",
            [("d01", 12), ("d02", 16), ("d03", 17)],
            summary_tokens=999,
        )
        _, errors = validator.validate(good_tree)
        assert any(".docs summary claims 999 tokens" in e for e in errors)

    def test_docs_summary_document_count_disagreement(self, good_tree):
        _write_sidecars(
            good_tree,
            "lang=en/pq01",
            [("d01", 12), ("d02", 16), ("d03", 17)],
            summary_docs=99,
        )
        _, errors = validator.validate(good_tree)
        assert any(".docs summary claims 99 documents" in e for e in errors)

    def test_corrupt_arrow_file(self, good_tree):
        (good_tree / "lang=en/pq01.arrow").write_text("this is not arrow data")
        _, errors = validator.validate(good_tree)
        assert any("unreadable arrow file" in e for e in errors)

    def test_empty_arrow_file(self, good_tree):
        _write_arrow(good_tree / "lang=en/pq01.arrow", 0)
        _write_sidecars(good_tree, "lang=en/pq01", [])
        _, errors = validator.validate(good_tree)
        assert any("zero tokens" in e for e in errors)

    def test_arrow_without_tokens_column(self, good_tree):
        path = good_tree / "lang=en/pq01.arrow"
        table = pa.table({"wrong_column": pa.array([1, 2, 3], type=pa.uint32())})
        with pa.OSFile(str(path), "wb") as sink:
            writer = pa.ipc.new_file(sink, table.schema)
            writer.write_table(table)
            writer.close()
        _, errors = validator.validate(good_tree)
        assert any("missing 'tokens' column" in e for e in errors)

    def test_missing_metadata_json(self, good_tree):
        (good_tree / "metadata.json").unlink()
        _, errors = validator.validate(good_tree)
        assert any("missing metadata.json" in e for e in errors)

    def test_malformed_metadata_json(self, good_tree):
        (good_tree / "metadata.json").write_text("{not json")
        _, errors = validator.validate(good_tree)
        assert any("not valid JSON" in e for e in errors)

    @pytest.mark.parametrize("content", ["[]", "null", '"a string"', "3"])
    def test_metadata_json_that_is_valid_json_but_not_an_object(
        self, good_tree, content
    ):
        """Regression: these used to raise AttributeError out of validate().

        The above test covers *malformed* JSON. These are WELL-FORMED JSON that
        simply is not an object, so `.get()` did not exist on it — the exception
        escaped main(), no validation.json was written, and the operator got a
        traceback instead of the report this script exists to produce.
        """
        (good_tree / "metadata.json").write_text(content)
        _, errors = validator.validate(good_tree)
        assert any("expected an object" in e for e in errors)

    def test_metadata_json_with_non_object_stats(self, good_tree):
        """Same hazard one level down: job_output_stats itself being a list."""
        (good_tree / "metadata.json").write_text(json.dumps({"job_output_stats": []}))
        _, errors = validator.validate(good_tree)
        assert any("'job_output_stats' is list" in e for e in errors)

    def test_non_object_metadata_still_writes_a_report(self, good_tree, tmp_path):
        """The point of the fix: report the failure, don't crash out of main()."""
        (good_tree / "metadata.json").write_text("[]")
        report_dir = tmp_path / "rpt"
        rc = validator.main(
            ["validate_tokenization2arrow.py", str(good_tree), str(report_dir)]
        )
        assert rc == 1
        assert (report_dir / "validation.json").is_file()

    @pytest.mark.parametrize("bad", [None, "5", 5.5, [], {}, True])
    def test_non_integer_result_files_is_reported_not_raised(self, good_tree, bad):
        """Regression: `stats["result_files"] < 1` raised TypeError on these.

        The container-level isinstance guards above stopped at the dict; the LEAF
        was still assumed to be an int. Comparing None or "5" to an int raises,
        and validate() has no handler — so this crashed out of main() before any
        report was written, which is precisely what those guards were added to
        prevent. metadata.json is DPK's output, not this step's, so its leaf types
        are an assumption rather than a guarantee.

        `True` is in the list because bool subclasses int: it would otherwise pass
        the type check and be silently read as result_files=1.
        """
        (good_tree / "metadata.json").write_text(
            json.dumps({"job_output_stats": {"result_files": bad, "num_tokens": 10}})
        )
        _, errors = validator.validate(good_tree)
        assert any("'result_files' is" in e for e in errors)

    @pytest.mark.parametrize("bad", ["10", None, 10.0, True])
    def test_non_integer_num_tokens_is_reported_not_mis_diagnosed(self, good_tree, bad):
        """num_tokens is compared with `!=`, which never raises — it MIS-REPORTS.

        A string "10" is != the int 10, so a perfectly good run was failed with a
        bogus "num_tokens mismatch". A wrong diagnosis is worse than no check,
        hence the same coercion as result_files rather than just crash-proofing.
        """
        (good_tree / "metadata.json").write_text(
            json.dumps({"job_output_stats": {"result_files": 1, "num_tokens": bad}})
        )
        _, errors = validator.validate(good_tree)
        assert any("'num_tokens' is" in e for e in errors)
        assert not any("token count mismatch" in e for e in errors)
        assert not any("hold" in e and "tokens in total" in e for e in errors)

    def test_non_integer_stats_still_writes_a_report(self, good_tree, tmp_path):
        """The guarantee: a report is written even when metadata.json is junk."""
        (good_tree / "metadata.json").write_text(
            json.dumps({"job_output_stats": {"result_files": None}})
        )
        report_dir = tmp_path / "rpt"
        rc = validator.main(
            ["validate_tokenization2arrow.py", str(good_tree), str(report_dir)]
        )
        assert rc == 1
        assert (report_dir / "validation.json").is_file()

    def test_absent_stats_keys_are_not_errors(self, good_tree):
        """Missing != malformed. Absent keys mean "nothing to check", not a fault.

        Guards the coercion against over-correcting into a false failure on
        metadata.json that simply does not carry these keys.
        """
        (good_tree / "metadata.json").write_text(json.dumps({"job_output_stats": {}}))
        _, errors = validator.validate(good_tree)
        assert not any("result_files" in e for e in errors)
        assert not any("num_tokens" in e for e in errors)

    def test_metadata_token_total_disagreement(self, tmp_path):
        """metadata.json's num_tokens contradicts the actual arrow totals."""
        tree = _build_tree(
            tmp_path / "out",
            {"lang=en/pq01": [("d01", 10)]},
            num_tokens=9999,
        )
        _, errors = validator.validate(tree)
        assert any("reports num_tokens=9999" in e for e in errors)

    def test_output_dir_under_a_meta_parent_still_finds_its_arrow_files(self, tmp_path):
        """Regression: the meta/ exclusion used to match the ABSOLUTE path.

        `if "/meta/" not in str(p)` also excluded every file when the output
        directory itself sat under a directory named meta — a plausible
        `output_path` such as /shared/meta/tokens. The result was "no .arrow files
        found" for output that was perfectly fine: a wrong diagnosis, which is worse
        than no check. Now tested relative to the output dir.
        """
        out = tmp_path / "meta" / "tokens"
        out.mkdir(parents=True)
        (out / "metadata.json").write_text(
            json.dumps({"job_output_stats": {"result_files": 1, "num_tokens": 6}})
        )
        _write_arrow(out / "pq01.arrow", 6)
        _write_sidecars(out, "pq01", [("d1", 3), ("d2", 3)])
        summary, errors = validator.validate(out)
        assert summary["arrow_files"] == 1
        assert not any("no .arrow files found" in e for e in errors)

    def test_sidecar_arrow_files_are_still_excluded(self, good_tree):
        """The exclusion must keep working: an .arrow inside meta/ is not data."""
        _write_arrow(good_tree / "meta" / "decoy.arrow", 99)
        summary, _ = validator.validate(good_tree)
        # good_tree has 2 real .arrow files; the decoy under meta/ must not count.
        assert summary["arrow_files"] == 2, "meta/ sidecar counted as a data file"

    def test_no_arrow_files(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        (empty / "metadata.json").write_text(
            json.dumps({"job_output_stats": {"result_files": 0, "num_tokens": 0}})
        )
        _, errors = validator.validate(empty)
        assert any("no .arrow files found" in e for e in errors)
        assert any("no result files" in e for e in errors)

    def test_malformed_docs_ids_line(self, good_tree):
        (good_tree / "meta/lang=en/pq01.docs.ids").write_text("d01, notanumber\n")
        _, errors = validator.validate(good_tree)
        assert any("non-integer token count" in e for e in errors)

    def test_non_positive_token_count(self, good_tree):
        _write_sidecars(good_tree, "lang=en/pq01", [("d01", 0), ("d02", 45)])
        _, errors = validator.validate(good_tree)
        assert any("non-positive token count" in e for e in errors)

    @pytest.mark.parametrize("corrupt", ["truncate", "unlink_meta", "bad_arrow"])
    def test_main_exits_nonzero_on_any_corruption(self, good_tree, tmp_path, corrupt):
        if corrupt == "truncate":
            _write_arrow(good_tree / "lang=en/pq01.arrow", 3)
        elif corrupt == "unlink_meta":
            (good_tree / "meta/lang=en/pq01.docs.ids").unlink()
        else:
            (good_tree / "lang=en/pq01.arrow").write_text("garbage")

        rc = validator.main(
            ["validate_tokenization2arrow.py", str(good_tree), str(tmp_path / "rpt")]
        )
        assert rc == 1


class TestValidatorCli:
    def test_missing_input_dir_exits_one(self, tmp_path):
        rc = validator.main(
            [
                "validate_tokenization2arrow.py",
                str(tmp_path / "nope"),
                str(tmp_path / "rpt"),
            ]
        )
        assert rc == 1

    def test_wrong_arg_count_exits_two(self):
        """argparse raises SystemExit(2) on a usage error rather than returning.

        The process exit code is still 2, so the CLI contract is unchanged; only
        the in-process call convention differs from the hand-rolled arg check
        this replaced.
        """
        with pytest.raises(SystemExit) as excinfo:
            validator.main(["validate_tokenization2arrow.py"])
        assert excinfo.value.code == 2


# --------------------------------------------------------------------------- #
# Completeness (--input) — a different axis from consistency
# --------------------------------------------------------------------------- #


def _write_parquet(path: pathlib.Path, rows: int) -> None:
    """Write a source Parquet with ``rows`` documents (0 => a legitimately empty input)."""
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "document_id": pa.array([f"d{i}" for i in range(rows)], type=pa.string()),
            "contents": pa.array(["text"] * rows, type=pa.string()),
        }
    )
    pq.write_table(table, str(path))


@pytest.fixture
def input_tree(tmp_path: pathlib.Path) -> pathlib.Path:
    """Source Parquet mirroring good_tree: 2 non-empty + 1 deliberately empty."""
    src = tmp_path / "in"
    _write_parquet(src / "lang=en/pq01.parquet", 3)
    _write_parquet(src / "lang=en/dataset=cyber/pq03.snappy.parquet", 1)
    _write_parquet(src / "lang=en/dataset=empty/empty01.parquet", 0)
    return src


class TestCompleteness:
    """Every non-empty source file must have produced its three output files.

    Consistency checks only inspect the output that exists, so a run that dropped
    whole files can still pass them — see
    ``test_consistency_alone_misses_a_dropped_file``, which is the reason this axis
    exists at all.
    """

    def test_complete_output_has_no_errors(self, good_tree, input_tree):
        summary, errors = validator.check_completeness(good_tree, input_tree)
        assert errors == []
        assert summary["source_parquet"] == 3
        assert summary["expected_outputs"] == 2

    def test_empty_source_is_not_counted_missing(self, good_tree, input_tree):
        """An empty Parquet produces no output by design (row count, not file size)."""
        summary, errors = validator.check_completeness(good_tree, input_tree)
        assert summary["empty_parquet"] == 1
        assert "lang=en/dataset=empty/empty01.parquet" in summary["empty_parquet_files"]
        assert errors == []

    def test_missing_arrow_is_reported_with_source_name(self, good_tree, input_tree):
        (good_tree / "lang=en/pq01.arrow").unlink()
        summary, errors = validator.check_completeness(good_tree, input_tree)
        assert summary["missing_arrow"] == ["lang=en/pq01.parquet"]
        assert any("produced no .arrow" in e for e in errors)

    def test_missing_sidecars_are_reported(self, good_tree, input_tree):
        (good_tree / "meta/lang=en/pq01.docs").unlink()
        (good_tree / "meta/lang=en/pq01.docs.ids").unlink()
        _, errors = validator.check_completeness(good_tree, input_tree)
        assert any("produced no meta/.docs" in e for e in errors)
        assert any("produced no meta/.docs.ids" in e for e in errors)

    def test_multi_suffix_source_stem_resolves(self, good_tree, input_tree):
        """pq03.snappy.parquet pairs with pq03.snappy.arrow, not pq03.arrow.

        Only the final suffix is swapped; stripping every suffix would look for
        ``pq03.arrow`` and report a false failure.
        """
        _, errors = validator.check_completeness(good_tree, input_tree)
        assert not [e for e in errors if "pq03" in e]

        (good_tree / "lang=en/dataset=cyber/pq03.snappy.arrow").unlink()
        summary, errors = validator.check_completeness(good_tree, input_tree)
        assert summary["missing_arrow"] == ["lang=en/dataset=cyber/pq03.snappy.parquet"]

    def test_no_source_parquet_is_an_error(self, good_tree, tmp_path):
        empty_in = tmp_path / "no-inputs"
        empty_in.mkdir()
        _, errors = validator.check_completeness(good_tree, empty_in)
        assert any("no .parquet files found" in e for e in errors)

    def test_consistency_alone_misses_a_dropped_file(self, good_tree, input_tree):
        """The case that justifies this axis.

        Drop one source file's entire output AND make metadata.json agree with
        what remains — which is what a partial or restarted run looks like.
        Consistency passes; completeness catches it.
        """
        (good_tree / "lang=en/pq01.arrow").unlink()
        (good_tree / "meta/lang=en/pq01.docs").unlink()
        (good_tree / "meta/lang=en/pq01.docs.ids").unlink()
        meta = json.loads((good_tree / "metadata.json").read_text())
        meta["job_output_stats"].update({"num_tokens": 12, "result_files": 1})
        (good_tree / "metadata.json").write_text(json.dumps(meta))

        _, consistency_errors = validator.validate(good_tree)
        assert consistency_errors == [], "consistency should be blind to this"

        _, completeness_errors = validator.check_completeness(good_tree, input_tree)
        assert completeness_errors, "completeness must catch the dropped file"


class TestAllEmptyInputIsNotAFailure:
    """An all-empty corpus produces no output, and that is CORRECT.

    The transform skips empty tables ("skipped empty tables"), writing nothing.
    validate() cannot know that — it never sees the input — so it appended "no
    .arrow files found" and failed a target that behaved correctly. Only main(),
    holding both results, can tell "nothing was produced" from "nothing was
    expected".
    """

    @pytest.fixture
    def empty_only_input(self, tmp_path):
        src = tmp_path / "in_empty"
        _write_parquet(src / "e1.parquet", 0)
        _write_parquet(src / "e2.parquet", 0)
        return src

    def test_all_empty_sources_pass_with_input(
        self, tmp_path, empty_only_input, capsys
    ):
        out = tmp_path / "out"
        out.mkdir()
        (out / "metadata.json").write_text(
            json.dumps({"job_output_stats": {"skipped empty tables": 2}})
        )
        rc = validator.main(
            [
                "validate_tokenization2arrow.py",
                str(out),
                str(tmp_path / "rpt"),
                "--input",
                str(empty_only_input),
            ]
        )
        assert rc == 0, capsys.readouterr().out
        report = json.loads((tmp_path / "rpt" / "validation.json").read_text())
        assert report["errors"] == []
        assert report["completeness"]["expected_outputs"] == 0
        assert report["completeness"]["source_parquet"] == 2
        assert "note" in report

    def test_a_dropped_file_still_fails(self, tmp_path, input_tree):
        """The narrow part of the fix: only withdraw when NOTHING was expected.

        input_tree has 2 non-empty sources, so expected_outputs == 2 and an empty
        output directory is a genuine dropped-output failure that must still fail.
        """
        out = tmp_path / "out2"
        out.mkdir()
        (out / "metadata.json").write_text(
            json.dumps({"job_output_stats": {"result_files": 0}})
        )
        rc = validator.main(
            [
                "validate_tokenization2arrow.py",
                str(out),
                str(tmp_path / "rpt2"),
                "--input",
                str(input_tree),
            ]
        )
        assert rc == 1
        report = json.loads((tmp_path / "rpt2" / "validation.json").read_text())
        assert any("no .arrow files found" in e for e in report["errors"])

    def test_without_input_the_error_still_stands(self, tmp_path):
        """No --input means no way to know output was not expected: stay strict."""
        out = tmp_path / "out3"
        out.mkdir()
        (out / "metadata.json").write_text(json.dumps({"job_output_stats": {}}))
        _, errors = validator.validate(out)
        assert any("no .arrow files found" in e for e in errors)

    def test_empty_input_directory_is_still_an_error(self, tmp_path):
        """An input dir with NO parquet at all is a real fault, not an empty corpus."""
        src = tmp_path / "in_none"
        src.mkdir()
        out = tmp_path / "out4"
        out.mkdir()
        (out / "metadata.json").write_text(json.dumps({"job_output_stats": {}}))
        rc = validator.main(
            [
                "validate_tokenization2arrow.py",
                str(out),
                str(tmp_path / "rpt4"),
                "--input",
                str(src),
            ]
        )
        assert rc == 1
        report = json.loads((tmp_path / "rpt4" / "validation.json").read_text())
        assert any("no .parquet files found" in e for e in report["errors"])


class TestArrowReaderFallback:
    """`open_file` failing on a stream-format file must fall through to a retry.

    Which exception it raises is a pyarrow implementation detail — ArrowInvalid for
    a bad magic number, but OSError/ArrowIOError when it reads a footer that is not
    there. Catching only ArrowInvalid reported "unreadable arrow file" for output
    pyarrow could actually read, failing a healthy target.
    """

    def test_stream_format_file_is_read(self, tmp_path):
        path = tmp_path / "stream.arrow"
        table = pa.table({"tokens": pa.array(range(5), type=pa.uint32())})
        with pa.OSFile(str(path), "wb") as sink:
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
        assert validator._read_arrow(path).num_rows == 5

    def test_a_genuinely_corrupt_file_still_raises(self, tmp_path):
        """The wider except must not swallow real corruption."""
        path = tmp_path / "corrupt.arrow"
        path.write_bytes(b"not an arrow file at all")
        with pytest.raises(Exception):
            validator._read_arrow(path)

    def test_a_corrupt_file_is_reported_per_file_not_fatal(self, good_tree):
        """And the caller still turns it into an ordinary finding."""
        (good_tree / "lang=en/pq01.arrow").write_bytes(b"garbage")
        _, errors = validator.validate(good_tree)
        assert any("unreadable arrow file" in e for e in errors)


class TestCompletenessCli:
    def test_input_flag_is_optional(self, good_tree, tmp_path, capsys):
        """Omitting --input keeps the original behaviour (backward compatible)."""
        rc = validator.main(
            ["validate_tokenization2arrow.py", str(good_tree), str(tmp_path / "rpt")]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "VALIDATION PASSED" in out
        assert "completeness" not in out

    def test_input_flag_adds_completeness_to_report(
        self, good_tree, input_tree, tmp_path, capsys
    ):
        report_dir = tmp_path / "rpt"
        rc = validator.main(
            [
                "validate_tokenization2arrow.py",
                str(good_tree),
                str(report_dir),
                "--input",
                str(input_tree),
            ]
        )
        assert rc == 0
        assert "completeness:" in capsys.readouterr().out
        written = json.loads((report_dir / "validation.json").read_text())
        assert written["completeness"]["expected_outputs"] == 2

    def test_missing_input_dir_exits_one(self, good_tree, tmp_path):
        rc = validator.main(
            [
                "validate_tokenization2arrow.py",
                str(good_tree),
                str(tmp_path / "rpt"),
                "--input",
                str(tmp_path / "nonexistent"),
            ]
        )
        assert rc == 1

    def test_completeness_failure_exits_one(self, good_tree, input_tree, tmp_path):
        (good_tree / "lang=en/pq01.arrow").unlink()
        meta = json.loads((good_tree / "metadata.json").read_text())
        meta["job_output_stats"].update({"num_tokens": 12, "result_files": 1})
        (good_tree / "metadata.json").write_text(json.dumps(meta))
        rc = validator.main(
            [
                "validate_tokenization2arrow.py",
                str(good_tree),
                str(tmp_path / "rpt"),
                "--input",
                str(input_tree),
            ]
        )
        assert rc == 1
