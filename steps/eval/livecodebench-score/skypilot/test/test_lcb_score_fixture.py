# SPDX-License-Identifier: Apache-2.0
"""The committed fixture and the slurm-cpu build.yaml match ``make_fixture.py``.

The build test's build.yaml embeds the fixture as one base64 string, so editing a
fixture file by hand, or editing make_fixture.py without re-running it, would leave
the cluster test scoring something else. These tests fail until it is re-run.
"""

import base64
import gzip
import importlib.util
import io
import json
import pickle
import tarfile
import zlib
from pathlib import Path

_DATA = Path(__file__).resolve().parent.parent / "test-data"
_spec = importlib.util.spec_from_file_location(
    "make_fixture", _DATA / "fixture" / "make_fixture.py"
)
mf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mf)


def test_committed_fixture_files_are_current():
    for name, text in mf.fixture_files().items():
        assert (
            _DATA / "fixture" / name
        ).read_text() == text, f"{name} is stale: re-run make_fixture.py"


def test_build_yaml_is_current():
    expected = mf.build_yaml(mf.fixture_tarball_b64(mf.fixture_files()))
    assert (
        _DATA / "slurm-cpu" / "build.yaml"
    ).read_text() == expected, "re-run make_fixture.py"


def test_tarball_round_trips():
    files = mf.fixture_files()
    raw = gzip.decompress(base64.b64decode(mf.fixture_tarball_b64(files)))
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        got = {m.name: tar.extractfile(m).read().decode() for m in tar.getmembers()}
    assert got == files


def test_tarball_is_reproducible():
    files = mf.fixture_files()
    assert mf.fixture_tarball_b64(files) == mf.fixture_tarball_b64(files)


def test_private_tests_decode_the_way_lcb_does():
    rows = [
        json.loads(line)
        for line in (_DATA / "fixture" / "code_generation_lite" / "test.jsonl")
        .read_text()
        .splitlines()
    ]
    for r in rows:
        tests = json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(r["private_test_cases"])))
        )
        assert tests and all({"input", "output", "testtype"} <= set(t) for t in tests)


def test_expected_score_follows_from_the_samples():
    """The planted outcome per sample, averaged as LiveCodeBench does."""
    outcome = {
        "fx_sum": [1, 1],
        "fx_max": [1, 0],
        "fx_rev": [0, 0],
        "fx_square": [1, 0],
    }
    per = {q: sum(v) / len(v) for q, v in outcome.items()}
    diff = {p["question_id"]: p["difficulty"] for p in mf.PROBLEMS}
    by = {d: [per[q] for q in per if diff[q] == d] for d in set(diff.values())}
    assert mf.EXPECTED == {
        "pass@1": sum(per.values()) / len(per),
        "by_difficulty": {d: sum(v) / len(v) for d, v in by.items()},
        "problems": len(per),
    }


def test_fixture_is_original_content():
    """No problem id or title from the real dataset (all ids start fx_)."""
    assert all(p["question_id"].startswith("fx_") for p in mf.PROBLEMS)
