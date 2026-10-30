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

"""Unit tests for src/hf_peft_sft_guard.sh: the staged-binding-path check."""

import pathlib
import subprocess

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "src" / "hf_peft_sft_guard.sh"


def _run(base_model: str, dataset: str = "") -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(_SCRIPT), "--base-model", base_model, "--dataset", dataset],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
        check=False,
    )


def _model(tmp_path: pathlib.Path) -> pathlib.Path:
    d = tmp_path / "base"
    d.mkdir()
    (d / "config.json").write_text("{}")
    return d


def test_valid_model_passes_silently(tmp_path):
    proc = _run(str(_model(tmp_path)))
    assert proc.returncode == 0, proc.stderr
    assert proc.stderr == ""


def test_valid_model_and_dataset_dir_pass(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    assert _run(str(_model(tmp_path)), str(ds)).returncode == 0


def test_dataset_file_passes(tmp_path):
    f = tmp_path / "train.jsonl"
    f.write_text("{}\n")
    assert _run(str(_model(tmp_path)), str(f)).returncode == 0


def test_empty_base_model_refused():
    proc = _run("")
    assert proc.returncode == 1
    assert "hf-peft-sft: ERROR no base model" in proc.stderr


def test_unrendered_jinja_refused():
    proc = _run("{{ bindings.base_modle.binding.path }}")
    assert proc.returncode == 1
    assert "unrendered" in proc.stderr


def test_unrendered_dataset_refused(tmp_path):
    proc = _run(str(_model(tmp_path)), "/staged/{%- if x %}ds")
    assert proc.returncode == 1
    assert "dataset still contains an unrendered" in proc.stderr


def test_missing_model_dir_refused(tmp_path):
    proc = _run(str(tmp_path / "nope"))
    assert proc.returncode == 1
    assert "is not a directory" in proc.stderr


def test_model_without_config_json_refused(tmp_path):
    d = tmp_path / "base"
    d.mkdir()
    proc = _run(str(d))
    assert proc.returncode == 1
    assert "config.json" in proc.stderr


def test_missing_dataset_refused(tmp_path):
    proc = _run(str(_model(tmp_path)), str(tmp_path / "nods"))
    assert proc.returncode == 1
    assert "dataset" in proc.stderr


def test_unknown_option_refused():
    proc = subprocess.run(
        ["bash", str(_SCRIPT), "--bogus", "x"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 1
    assert "unknown option" in proc.stderr
