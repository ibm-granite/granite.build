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

"""Unit tests for src/openinstruct_sft_setup.sh: the image sanity check.

The real image venv isn't available here, so OPEN_INSTRUCT_PYTHON points at a
stub that records its argv and exits as told.
"""

import pathlib
import stat
import subprocess

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[1] / "src" / "openinstruct_sft_setup.sh"
)


def _stub(path: pathlib.Path, body: str) -> pathlib.Path:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _run(tmp_path, py_body, with_home=True):
    home = tmp_path / "oi"
    if with_home:
        (home / "open_instruct").mkdir(parents=True)
        (home / "open_instruct" / "finetune.py").write_text("")
    py = _stub(tmp_path / "py", py_body)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir / "nvidia-smi", 'echo "NVIDIA A10G, 23028 MiB"\n')
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "OPEN_INSTRUCT_HOME": str(home),
        "OPEN_INSTRUCT_PYTHON": str(py),
    }
    return subprocess.run(
        ["bash", str(_SCRIPT)], capture_output=True, text=True, env=env, check=False
    )


def test_ok_image_passes(tmp_path):
    proc = _run(tmp_path, 'echo "open_instruct ok"\n')
    assert proc.returncode == 0, proc.stderr
    assert "openinstruct-sft: open_instruct ok" in proc.stdout
    assert "NVIDIA A10G" in proc.stdout


def test_missing_finetune_py_fails(tmp_path):
    proc = _run(tmp_path, "true\n", with_home=False)
    assert proc.returncode == 1
    assert "finetune.py not found" in proc.stderr


def test_import_failure_fails(tmp_path):
    proc = _run(tmp_path, 'echo "ModuleNotFoundError: accelerate" >&2; exit 1\n')
    assert proc.returncode == 1
    assert "cannot import" in proc.stderr
