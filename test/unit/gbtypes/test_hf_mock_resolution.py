"""Regression tests for how GBTEST_MOCK_HF is resolved at session start.

These run pytest in a subprocess because the behavior under test lives in
``pytest_sessionstart`` (test/conftest.py) — it has already run by the time an
in-process test body executes, so it cannot be re-exercised from inside the
current session.

The case that motivated these: `make .test` runs
``export GBTEST_MOCK_HF=${GBTEST_MOCK_HF}`` unconditionally, so when no caller
sets the variable the child sees it *present but empty*. Resolving that as
"false" silently un-mocked HuggingFace for the whole of CI (PR #314 review).
"""

import os
import shutil
import subprocess
import sys
import textwrap

import pytest

# Body of the throwaway test the subprocess runs: print the resolved state so the
# parent can assert on it.
_PROBE = textwrap.dedent("""
    from gbcommon.types.testing import is_hf_mocked


    def test_probe():
        print(f"RESOLVED:{is_hf_mocked()}")
    """)


def _resolved_is_hf_mocked(tmp_path, env_overrides: dict[str, str | None]) -> bool:
    """Run a probe test in a subprocess and return its is_hf_mocked() result.

    The probe must live inside the repo's test tree, not in tmp_path: the
    resolution under test happens in test/conftest.py, which only applies to
    files collected beneath it.

    Args:
        tmp_path: unused for the probe location; kept for a unique file name.
        env_overrides: env vars to set; a None value removes the variable.
    """
    probe_dir = _repo_root() / "test" / "unit" / "gbtypes" / "_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)
    (probe_dir / "__init__.py").write_text("")
    probe = probe_dir / f"test_probe_{abs(hash(tmp_path.name)) % 10**8}.py"
    probe.write_text(_PROBE)

    env = dict(os.environ)
    env["GB_ENVIRONMENT"] = "STANDALONE"
    for key, value in env_overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(probe),
            "-q",
            "-s",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            "--no-cov",
        ],
        capture_output=True,
        text=True,
        cwd=str(_repo_root()),
        env=env,
        check=False,
    )
    try:
        # The marker may be prefixed by pytest's own progress output (the test
        # node id), so search anywhere in the line rather than only at the start.
        markers = [
            line.split("RESOLVED:", 1)[1]
            for line in result.stdout.splitlines()
            if "RESOLVED:" in line
        ]
        assert (
            markers
        ), f"probe did not report a result.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        return markers[0].strip().split()[0] == "True"
    finally:
        probe.unlink(missing_ok=True)
        # Leave no scratch behind: drop the package dir once the last probe in it
        # is gone, so a test run doesn't litter the source tree with untracked
        # files. shutil.rmtree covers the __pycache__ the subprocess created.
        if not any(probe_dir.glob("test_probe_*.py")):
            shutil.rmtree(probe_dir, ignore_errors=True)


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "mock_hf,expected,because",
    [
        # The `make .test` shape: exported but empty. Must NOT read as an opt-out.
        ("", True, "blank is 'no choice made', so the mock-mode default applies"),
        (None, True, "unset in mock mode defaults to mocked"),
        ("true", True, "explicit true"),
        ("1", True, "repo-standard boolean parsing accepts 1"),
        ("yes", True, "repo-standard boolean parsing accepts yes"),
        ("false", False, "explicit false opts out of the mock-mode default"),
        ("no", False, "repo-standard boolean parsing accepts no"),
    ],
)
def test_mock_mode_hf_resolution(tmp_path, mock_hf, expected, because):
    """GBTEST_MODE=mock mocks HF by default; an explicit non-blank value wins."""
    got = _resolved_is_hf_mocked(
        tmp_path,
        {"GBTEST_MODE": "mock", "GBTEST_MOCK_HF": mock_hf, "GBTEST_LIVE_HF": None},
    )
    assert got is expected, f"GBTEST_MOCK_HF={mock_hf!r}: {because}"


def test_live_mode_honors_explicit_mock_hf(tmp_path):
    """GBTEST_MOCK_HF is an independent axis: it mocks HF even under live mode."""
    got = _resolved_is_hf_mocked(
        tmp_path,
        {"GBTEST_MODE": "live", "GBTEST_MOCK_HF": "true", "GBTEST_LIVE_HF": None},
    )
    assert got is True


def test_live_hf_opt_in_lifts_the_mock(tmp_path):
    """A whole-run GBTEST_LIVE_HF=true opt-in beats the mock-mode default."""
    got = _resolved_is_hf_mocked(
        tmp_path,
        {"GBTEST_MODE": "mock", "GBTEST_MOCK_HF": None, "GBTEST_LIVE_HF": "true"},
    )
    assert got is False
