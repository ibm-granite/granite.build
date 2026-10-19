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

"""Unit tests for src/nemo_rl_setup.sh.

Setup runs once per cluster: it sanity-checks that the NGC image actually
bundles the NeMo-RL source (Approach A's load-bearing assumption) and creates
the output directories. It must fail LOUDLY when the image has no examples/,
rather than deferring to a confusing run-phase failure.
"""

import pathlib
import shutil
import subprocess

import pytest

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _STEP_DIR / "src" / "nemo_rl_setup.sh"


def _run(tmp_path, root: pathlib.Path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    proc = subprocess.run(
        ["bash", str(_SCRIPT), "--nemo-rl-root", str(root)],
        capture_output=True,
        text=True,
        cwd=str(workdir),
        env=env,
    )
    return proc, workdir


class TestSetup:
    def test_creates_output_dirs_when_image_has_examples(self, tmp_path):
        root = tmp_path / "nemo-rl"
        (root / "examples").mkdir(parents=True)
        proc, workdir = _run(tmp_path, root)
        assert proc.returncode == 0, proc.stderr
        assert (workdir / "results").is_dir()
        assert (workdir / "logs").is_dir()

    def test_does_not_precreate_hf_output(self, tmp_path):
        # convert_dcp_to_hf.py (run phase) raises FileExistsError if --hf-ckpt-path
        # already exists, so setup must NOT pre-create hf_output — the converter
        # creates it itself. Pre-creating it here is what broke the real e2e.
        root = tmp_path / "nemo-rl"
        (root / "examples").mkdir(parents=True)
        proc, workdir = _run(tmp_path, root)
        assert proc.returncode == 0, proc.stderr
        assert not (workdir / "hf_output").exists()

    def test_fails_loudly_when_image_lacks_examples(self, tmp_path):
        root = tmp_path / "nemo-rl"
        root.mkdir()
        proc, _ = _run(tmp_path, root)
        assert proc.returncode != 0
        reasons = [ln for ln in proc.stderr.splitlines() if ln.startswith("nemo-rl:")]
        assert any("examples" in r for r in reasons)


class TestScriptIsValidShell:
    def test_bash_parses_the_script(self):
        assert subprocess.run(["bash", "-n", str(_SCRIPT)]).returncode == 0

    def test_shellcheck_is_clean(self):
        if shutil.which("shellcheck") is None:
            pytest.skip("shellcheck not installed")
        assert subprocess.run(["shellcheck", str(_SCRIPT)]).returncode == 0
