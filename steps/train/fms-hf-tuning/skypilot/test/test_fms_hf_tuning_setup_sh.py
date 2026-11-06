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

"""Unit tests for src/fms_hf_tuning_setup.sh: the image sanity check.

The image venv isn't available here, so FMS_HF_TUNING_PYTHON points at a stub.
"""

import pathlib
import stat
import subprocess

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "src" / "fms_hf_tuning_setup.sh"
_FILES = (
    "fms_hf_tuning_entry.py",
    "fms_hf_tuning_job.py",
    "fms_hf_tuning_run.sh",
    "accelerate/single_gpu.yaml",
    "accelerate/ddp.yaml",
    "accelerate/fsdp.yaml",
)


def _stub(path: pathlib.Path, body: str) -> pathlib.Path:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _home(tmp_path, skip=None):
    home = tmp_path / "home"
    (home / "accelerate").mkdir(parents=True)
    for f in _FILES:
        if f != skip:
            (home / f).write_text("")
    return home


def _run(tmp_path, py_body, skip=None, nvidia=True):
    home = _home(tmp_path, skip)
    py = _stub(tmp_path / "py", py_body)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    if nvidia:
        _stub(bin_dir / "nvidia-smi", 'echo "NVIDIA A10G, 23028 MiB"\n')
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "FMS_HF_TUNING_HOME": str(home),
        "FMS_HF_TUNING_PYTHON": str(py),
    }
    return subprocess.run(
        ["bash", str(_SCRIPT)], capture_output=True, text=True, env=env, check=False
    )


def test_ok_image_passes(tmp_path):
    proc = _run(tmp_path, 'echo "fms-hf-tuning 3.2.2 transformers 4.55.4"\n')
    assert proc.returncode == 0, proc.stderr
    assert "fms-hf-tuning: fms-hf-tuning 3.2.2 transformers 4.55.4" in proc.stdout
    assert "NVIDIA A10G" in proc.stdout


def test_imports_the_whole_stack(tmp_path):
    proc = _run(tmp_path, 'printf "%s\\n" "$@" > "$HOME/argv"; echo ok\n')
    assert proc.returncode == 0, proc.stderr
    code = (tmp_path / "argv").read_text()
    for mod in (
        "tuning",
        "flash_attn",
        "transformers",
        "trl",
        "peft",
        "accelerate",
        "datasets",
        "clearml",
        "torch",
    ):
        assert mod in code, mod


@pytest.mark.parametrize("missing", _FILES)
def test_missing_file_fails(tmp_path, missing):
    proc = _run(tmp_path, "true\n", skip=missing)
    assert proc.returncode == 1
    assert f"{missing} not found" in proc.stderr


def test_import_failure_fails(tmp_path):
    proc = _run(tmp_path, 'echo "ModuleNotFoundError: flash_attn" >&2; exit 1\n')
    assert proc.returncode == 1
    assert "cannot import" in proc.stderr
    assert "flash_attn" in proc.stderr


def test_no_nvidia_smi_only_warns(tmp_path):
    proc = _run(tmp_path, "echo ok\n", nvidia=False)
    assert proc.returncode == 0
    assert "WARNING nvidia-smi not available" in proc.stdout
