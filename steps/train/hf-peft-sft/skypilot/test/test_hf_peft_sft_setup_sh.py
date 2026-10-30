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

"""Unit tests for src/hf_peft_sft_setup.sh: the image sanity check.

The image venv isn't available here, so HF_PEFT_SFT_PYTHON points at a stub.
"""

import pathlib
import stat
import subprocess

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "src" / "hf_peft_sft_setup.sh"


def _stub(path: pathlib.Path, body: str) -> pathlib.Path:
    path.write_text("#!/usr/bin/env bash\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def _run(tmp_path, py_body, with_train=True):
    home = tmp_path / "home"
    home.mkdir()
    if with_train:
        (home / "hf_peft_sft_train.py").write_text("")
    py = _stub(tmp_path / "py", py_body)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _stub(bin_dir / "nvidia-smi", 'echo "NVIDIA A10G, 23028 MiB"\n')
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path),
        "HF_PEFT_SFT_HOME": str(home),
        "HF_PEFT_SFT_PYTHON": str(py),
    }
    return subprocess.run(
        ["bash", str(_SCRIPT)], capture_output=True, text=True, env=env, check=False
    )


def test_ok_image_passes(tmp_path):
    proc = _run(tmp_path, 'echo "peft 0.21.2 trl 1.17.1"\n')
    assert proc.returncode == 0, proc.stderr
    assert "hf-peft-sft: peft 0.21.2 trl 1.17.1" in proc.stdout
    assert "NVIDIA A10G" in proc.stdout


def test_imports_the_whole_stack(tmp_path):
    proc = _run(tmp_path, 'printf "%s\\n" "$@" > "$HOME/argv"; echo ok\n')
    assert proc.returncode == 0, proc.stderr
    code = (tmp_path / "argv").read_text()
    for mod in (
        "peft",
        "trl",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "deepspeed",
        "datasets",
        "clearml",
        "torch",
    ):
        assert mod in code, mod


def test_missing_train_script_fails(tmp_path):
    proc = _run(tmp_path, "true\n", with_train=False)
    assert proc.returncode == 1
    assert "hf_peft_sft_train.py not found" in proc.stderr


def test_import_failure_fails(tmp_path):
    proc = _run(tmp_path, 'echo "ModuleNotFoundError: bitsandbytes" >&2; exit 1\n')
    assert proc.returncode == 1
    assert "cannot import" in proc.stderr
    assert "bitsandbytes" in proc.stderr


def test_no_nvidia_smi_only_warns(tmp_path):
    (tmp_path / "bin").mkdir()
    home = tmp_path / "home"
    home.mkdir()
    (home / "hf_peft_sft_train.py").write_text("")
    py = _stub(tmp_path / "py", "echo ok\n")
    proc = subprocess.run(
        ["bash", str(_SCRIPT)],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "HF_PEFT_SFT_HOME": str(home),
            "HF_PEFT_SFT_PYTHON": str(py),
        },
        check=False,
    )
    assert proc.returncode == 0
    assert "WARNING nvidia-smi not available" in proc.stdout
