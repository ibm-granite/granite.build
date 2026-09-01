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

"""Behaviour tests for the bundled ``src/dpk_run.sh``.

These exist because the shell moved out of ``step-template.yaml`` into a real
file: the script can now be *executed* rather than only rendered and pattern
matched. Each test runs the actual script under ``bash`` with a stub ``python`` on
``PATH``, so the assertions are about what the transform and the monitor really
receive:

* the ``--data_local_config`` python literal the DPK launcher is invoked with;
* the output directory being created and **absolutized** before it is announced
  (the server consumes the artifact path off-node, so a relative one is useless);
* transform flags reaching ``python`` untouched, including python-literal values
  containing single quotes;
* the artifact marker being its own command, at the start of a line.

Cluster-agnostic, so this sits at the root of the step's ``test/`` dir (Mode 1
only) and is not copied by ``make publish-step``.
"""

import pathlib
import re
import shutil
import subprocess

import pytest
import yaml

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _STEP_DIR / "src" / "dpk_run.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason="bash not available"
)


@pytest.fixture
def run_script(tmp_path):
    """Run dpk_run.sh in a scratch cwd with a stub `python` that echoes its argv."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    stub = stub_dir / "python"
    stub.write_text('#!/usr/bin/env bash\nfor a in "$@"; do echo "PYARG:$a"; done\n')
    stub.chmod(0o755)

    def _run(*args: str, cwd: pathlib.Path | None = None):
        env = {"PATH": f"{stub_dir}:/usr/bin:/bin", "HOME": str(tmp_path)}
        return subprocess.run(
            ["bash", str(_SCRIPT), *args],
            capture_output=True,
            text=True,
            cwd=str(cwd or tmp_path),
            env=env,
        )

    _run.tmp_path = tmp_path  # type: ignore[attr-defined]
    return _run


def _pyargs(stdout: str) -> list[str]:
    return [l[len("PYARG:") :] for l in stdout.splitlines() if l.startswith("PYARG:")]


def _marker(stdout: str) -> str | None:
    return next(
        (l for l in stdout.splitlines() if l.startswith("GB_ARTIFACT_ID:")), None
    )


_BASE = ("--module", "dpk_x.runtime", "--input-path", "/staged/docs")


class TestScriptIsValidShell:
    def test_parses_under_bash(self):
        assert subprocess.run(["bash", "-n", str(_SCRIPT)]).returncode == 0

    @pytest.mark.skipif(
        shutil.which("shellcheck") is None, reason="shellcheck not installed"
    )
    def test_shellcheck_is_clean(self):
        """Now possible at all, because the shell is a file rather than YAML."""
        proc = subprocess.run(
            ["shellcheck", str(_SCRIPT)], capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stdout


class TestRequiredOptions:
    @pytest.mark.parametrize(
        "missing,args",
        [
            (
                "--module",
                ("--input-path", "/i", "--output-path", "o", "--artifact-id", "a"),
            ),
            (
                "--input-path",
                ("--module", "m", "--output-path", "o", "--artifact-id", "a"),
            ),
            (
                "--output-path",
                ("--module", "m", "--input-path", "/i", "--artifact-id", "a"),
            ),
            (
                "--artifact-id",
                ("--module", "m", "--input-path", "/i", "--output-path", "o"),
            ),
        ],
    )
    def test_missing_required_option_fails(self, run_script, missing, args):
        """Fail loudly rather than invoking python with an empty value."""
        proc = run_script(*args)
        assert proc.returncode != 0
        assert missing in proc.stderr
        assert not _pyargs(proc.stdout)


class TestDataLocalConfig:
    def test_input_and_output_are_passed_as_one_python_literal(self, run_script):
        proc = run_script(*_BASE, "--output-path", "out", "--artifact-id", "tokens")
        assert proc.returncode == 0, proc.stderr
        args = _pyargs(proc.stdout)
        assert args[:2] == ["-m", "dpk_x.runtime"]
        assert args[2] == "--data_local_config"
        out_abs = str((run_script.tmp_path / "out").resolve())
        assert args[3] == (
            "{'input_folder': '/staged/docs', 'output_folder': '" + out_abs + "'}"
        )

    def test_module_override_is_honoured(self, run_script):
        proc = run_script(
            "--module",
            "dpk_x.ray.runtime",
            "--input-path",
            "/i",
            "--output-path",
            "o",
            "--artifact-id",
            "a",
        )
        assert _pyargs(proc.stdout)[:2] == ["-m", "dpk_x.ray.runtime"]


class TestOutputPathHandling:
    def test_relative_output_is_created_and_absolutized(self, run_script):
        """The step's default is the relative ./output, so this is the hot path."""
        proc = run_script(*_BASE, "--output-path", "./output", "--artifact-id", "a")
        assert proc.returncode == 0, proc.stderr
        created = run_script.tmp_path / "output"
        assert created.is_dir()
        assert (
            _marker(proc.stdout)
            == f"GB_ARTIFACT_ID:a GB_ARTIFACT_PATH:{created.resolve()}"
        )

    def test_nested_relative_output_is_created(self, run_script):
        proc = run_script(*_BASE, "--output-path", "a/b/c", "--artifact-id", "a")
        assert proc.returncode == 0, proc.stderr
        assert (run_script.tmp_path / "a/b/c").is_dir()

    def test_absolute_output_is_used_as_given(self, run_script, tmp_path):
        target = tmp_path / "explicit"
        proc = run_script(*_BASE, "--output-path", str(target), "--artifact-id", "a")
        assert proc.returncode == 0, proc.stderr
        assert _marker(proc.stdout).endswith(str(target.resolve()))

    def test_existing_output_dir_is_not_an_error(self, run_script):
        """mkdir -p, so a re-run or a pre-staged dir is fine."""
        (run_script.tmp_path / "out").mkdir()
        proc = run_script(*_BASE, "--output-path", "out", "--artifact-id", "a")
        assert proc.returncode == 0, proc.stderr

    def test_marker_path_is_always_absolute(self, run_script):
        proc = run_script(*_BASE, "--output-path", "rel", "--artifact-id", "a")
        path = _marker(proc.stdout).split("GB_ARTIFACT_PATH:")[1]
        assert pathlib.Path(path).is_absolute()

    def test_script_cwd_is_unchanged_by_the_absolutize(self, run_script):
        """The `cd` runs in a subshell, so a later relative path still resolves."""
        proc = run_script(*_BASE, "--output-path", "out", "--artifact-id", "a")
        assert proc.returncode == 0, proc.stderr
        # python was found via PATH and the config's output_folder is absolute;
        # if the cd had leaked, the marker would be relative to ./out instead.
        assert "/out/out" not in _marker(proc.stdout)


class TestFlagPassthrough:
    def test_flags_after_separator_reach_python_untouched(self, run_script):
        proc = run_script(
            *_BASE,
            "--output-path",
            "o",
            "--artifact-id",
            "a",
            "--",
            "--tkn_tokenizer",
            "hf-internal-testing/llama-tokenizer",
        )
        args = _pyargs(proc.stdout)
        assert args[-2:] == ["--tkn_tokenizer", "hf-internal-testing/llama-tokenizer"]

    def test_python_literal_value_with_quotes_survives(self, run_script):
        """The pii_redactor case: ast.literal_eval needs the inner quotes intact."""
        value = "['PERSON','EMAIL_ADDRESS']"
        proc = run_script(
            *_BASE,
            "--output-path",
            "o",
            "--artifact-id",
            "a",
            "--",
            "--pii_redactor_entities",
            value,
        )
        assert _pyargs(proc.stdout)[-1] == value

    def test_value_with_spaces_stays_one_argument(self, run_script):
        proc = run_script(
            *_BASE,
            "--output-path",
            "o",
            "--artifact-id",
            "a",
            "--",
            "--flag",
            "two words",
        )
        assert _pyargs(proc.stdout)[-1] == "two words"

    def test_zero_is_forwarded(self, run_script):
        proc = run_script(
            *_BASE,
            "--output-path",
            "o",
            "--artifact-id",
            "a",
            "--",
            "--tkn_chunk_size",
            "0",
        )
        assert _pyargs(proc.stdout)[-2:] == ["--tkn_chunk_size", "0"]

    def test_no_flags_is_valid(self, run_script):
        proc = run_script(*_BASE, "--output-path", "o", "--artifact-id", "a", "--")
        assert proc.returncode == 0, proc.stderr
        assert _pyargs(proc.stdout)[-1].startswith("{'input_folder'")

    def test_separator_is_optional_when_flags_come_last(self, run_script):
        proc = run_script(
            *_BASE,
            "--output-path",
            "o",
            "--artifact-id",
            "a",
            "--tkn_text_lang",
            "en",
        )
        assert _pyargs(proc.stdout)[-2:] == ["--tkn_text_lang", "en"]

    def test_a_flag_named_like_an_option_is_shielded_by_the_separator(self, run_script):
        """Why the template always emits `--`: a transform flag could collide."""
        proc = run_script(
            *_BASE,
            "--output-path",
            "real",
            "--artifact-id",
            "a",
            "--",
            "--output-path",
            "decoy",
        )
        assert proc.returncode == 0, proc.stderr
        # The decoy went to python, not to the script's own parsing.
        assert _pyargs(proc.stdout)[-2:] == ["--output-path", "decoy"]
        assert (run_script.tmp_path / "real").is_dir()
        assert not (run_script.tmp_path / "decoy").exists()


class TestFailurePropagation:
    def test_transform_failure_fails_the_script(self, run_script, tmp_path):
        """set -e: a non-zero transform must fail the step, not emit a marker."""
        failing = tmp_path / "bin" / "python"
        failing.write_text("#!/usr/bin/env bash\nexit 3\n")
        failing.chmod(0o755)
        proc = run_script(*_BASE, "--output-path", "o", "--artifact-id", "a")
        assert proc.returncode == 3
        assert _marker(proc.stdout) is None, "marker emitted despite a failed transform"


class TestArtifactMarker:
    def test_marker_starts_at_the_beginning_of_a_line(self, run_script):
        """The skypilot monitor's regex needs it unindented."""
        proc = run_script(*_BASE, "--output-path", "o", "--artifact-id", "tokens")
        line = _marker(proc.stdout)
        assert line is not None and line.startswith("GB_ARTIFACT_ID:tokens ")

    def test_exactly_one_marker_is_emitted(self, run_script):
        proc = run_script(*_BASE, "--output-path", "o", "--artifact-id", "a")
        markers = [
            l for l in proc.stdout.splitlines() if l.startswith("GB_ARTIFACT_ID:")
        ]
        assert len(markers) == 1

    def test_marker_uses_the_gb_prefix_not_the_legacy_llmb_one(self, run_script):
        """Pin the standardized prefix (#329 moved the repo to GB_).

        The monitor accepts either — its regex is ``(?:GB_|LLMB_)ARTIFACT_ID:`` —
        so a regression to LLMB_ would still WORK and therefore would not be
        caught by any build test. Assert the prefix directly.
        """
        proc = run_script(*_BASE, "--output-path", "o", "--artifact-id", "a")
        assert "LLMB_" not in proc.stdout
        assert "GB_ARTIFACT_ID:" in proc.stdout
        assert "GB_ARTIFACT_PATH:" in proc.stdout

    def test_the_emitted_marker_matches_the_shipped_monitor_regex(self, run_script):
        """End-to-end on the contract: the server must actually parse what we emit.

        Reads the regex out of the shipped skypilot monitor rather than restating
        it, so this fails if either side drifts. Also checks the form the monitor
        really sees on a cluster, where SkyPilot prefixes each stdout line with
        ``(cluster, pid=N)``.
        """
        # _STEP_DIR is steps/dpk/skypilot, so the repo root is 3 levels up.
        monitor = (
            _STEP_DIR.parents[2]
            / "src/gbserver/builtins/monitors/skypilot/monitor.yaml"
        )
        if not monitor.is_file():  # pragma: no cover - repo layout guard
            pytest.skip("shipped monitor not found from the step dir")
        spec = yaml.safe_load(monitor.read_text())
        cfg = next(
            e
            for e in spec["config"]["event_configs"]
            if "ARTIFACT_PATH" in e.get("line_regex", "")
        )
        proc = run_script(*_BASE, "--output-path", "o", "--artifact-id", "tokens")
        line = _marker(proc.stdout)
        assert re.search(cfg["line_regex"], line), f"monitor would not match {line!r}"
        assert re.search(cfg["line_regex"], f"(gb-abc123, pid=42) {line}")
        # And the captured fields are the ones we meant to publish.
        by_name = {f["field_name"]: f for f in cfg["event_fields"]}
        got_id = re.search(by_name["binding_id"]["field_regex"], line)
        got_path = re.search(by_name["path"]["field_regex"], line)
        assert got_id and got_id.group(0) == "tokens"
        assert got_path and got_path.group(0).endswith("/o")
