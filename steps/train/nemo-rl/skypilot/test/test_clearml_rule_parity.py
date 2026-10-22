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

"""The bash and Python ClearML enablement rules must agree.

nemo_rl_run.sh decides whether to `uv run --with clearml` (install); then
clearml_launch.clearml_enabled decides whether to `import clearml` (use). If they
disagree, the run either imports a package that was never installed or installs
one it never uses. Both are run here over the same inputs.
"""

import itertools
import pathlib
import stat
import subprocess

import clearml_launch
import pytest

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_SCRIPT = _STEP_DIR / "src" / "nemo_rl_run.sh"
_KEYS = ("CLEARML_API_HOST", "CLEARML_API_ACCESS_KEY", "CLEARML_API_SECRET_KEY")


def _bash_installs_clearml(tmp_path, mode, env):
    root = tmp_path / "nemo-rl"
    (root / "examples" / "configs").mkdir(parents=True)
    (root / "examples" / "run_sft.py").write_text("")
    (root / "examples" / "configs" / "sft.yaml").write_text("policy: {}\n")
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "src" / "nemo_rl_run.sh").write_text(_SCRIPT.read_text())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "uv.log"
    # Record the training argv, then fail it: only the install decision matters.
    uv = bin_dir / "uv"
    uv.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$UVLOG"\nexit 7\n')
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    proc = subprocess.run(
        [
            "bash",
            "src/nemo_rl_run.sh",
            "--algorithm",
            "sft",
            "--config-name",
            "sft.yaml",
            "--nemo-rl-root",
            str(root),
            "--base-model",
            "/b",
            "--artifact-id",
            "model",
            "--output-path",
            "hf_output",
            "--clearml-mode",
            mode,
        ],
        cwd=work,
        capture_output=True,
        text=True,
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "UVLOG": str(log),
            **env,
        },
    )
    assert proc.returncode == 7, proc.stderr
    return "--with\nclearml\n" in log.read_text()


_ENVS = [
    {k: v for k, v in zip(_KEYS, vals) if v is not None}
    for vals in itertools.product(("x", "", None), repeat=3)
]


@pytest.mark.parametrize("mode", ["auto", "on", "off"])
@pytest.mark.parametrize(
    "env",
    _ENVS,
    ids=lambda e: ",".join(f"{k[11:]}={v!r}" for k, v in e.items()) or "none",
)
def test_install_and_import_decisions_agree(tmp_path, mode, env):
    assert _bash_installs_clearml(
        tmp_path, mode, env
    ) is clearml_launch.clearml_enabled(mode, env)
