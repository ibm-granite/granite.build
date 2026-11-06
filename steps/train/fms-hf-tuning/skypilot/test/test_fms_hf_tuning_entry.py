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

"""Tests for src/fms_hf_tuning_entry.py.

step_line and install() need nothing. The upstream checks (train()'s signature,
a real 2-step CPU run through the wrapper on a tiny random Llama) are skipped
unless `tuning` imports; run them in the CPU venv (README.md, "Tests").
"""

import inspect
import json
import os
import pathlib
import re
import subprocess
import sys
import types

import fms_hf_tuning_entry as entry
import pytest
import yaml

_STEP = pathlib.Path(__file__).resolve().parents[1]
_ENTRY = _STEP / "src" / "fms_hf_tuning_entry.py"
_MODEL = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"


def test_step_line():
    assert entry.step_line(3, 10, {"loss": 0.70123}) == "Step: 3/10 loss=0.7012"
    assert entry.step_line(3, 0, {"loss": 1}) == "Step: 3 loss=1.0000"
    assert entry.step_line(3, -1, {}) == "Step: 3"
    assert entry.step_line(3, 10, {"loss": "x"}) == "Step: 3/10"


def test_step_line_matches_the_monitor_rule():
    rule = yaml.safe_load((_STEP / "step-template.yaml").read_text())[
        "environment_configs"
    ]["Skypilot"]["monitors"]["skypilot_monitor"]["config"]["extra_event_configs"][0]
    line = entry.step_line(3, 10, {"loss": 0.7})
    # tqdm's \r-redrawn bar can share the line; search, as gbserver does.
    assert re.search(rule["line_regex"], "  0%|  | 0/10 [00:00<?]" + line)[0] == line


def test_install_appends_the_callback(monkeypatch):
    seen = {}

    def train(model_args, additional_callbacks=None, **kwargs):
        seen["callbacks"] = additional_callbacks
        seen["kwargs"] = kwargs
        return "trainer"

    sft = types.SimpleNamespace(train=train)
    marker = object()
    monkeypatch.setattr(entry, "progress_callback", lambda: marker)
    entry.install(sft)
    assert sft.train("m", additional_callbacks=None, exp_metadata={"a": 1}) == "trainer"
    assert seen["callbacks"] == [marker] and seen["kwargs"] == {
        "exp_metadata": {"a": 1}
    }
    existing = object()
    sft.train("m", additional_callbacks=[existing])
    assert seen["callbacks"] == [existing, marker]
    assert sft.train.__name__ == "train"


def test_callback_prints_only_on_rank_zero(capsys):
    pytest.importorskip("transformers")
    cb = entry.progress_callback()
    state = types.SimpleNamespace(
        is_world_process_zero=True, global_step=2, max_steps=4
    )
    cb.on_log(None, state, None, logs={"loss": 1.5})
    cb.on_log(None, state, None, logs={"train_runtime": 1.0})
    state.is_world_process_zero = False
    cb.on_log(None, state, None, logs={"loss": 1.5})
    assert capsys.readouterr().out == "Step: 2/4 loss=1.5000\n"


def test_upstream_train_still_takes_additional_callbacks():
    pytest.importorskip("tuning")
    import tuning.sft_trainer as sft  # pylint: disable=import-outside-toplevel

    params = inspect.signature(sft.train).parameters
    assert "additional_callbacks" in params
    # main() must look train up at call time (module global), not bind it early.
    assert "train(" in inspect.getsource(sft.main)


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    pytest.importorskip("tuning")
    from huggingface_hub import (  # pylint: disable=import-outside-toplevel
        snapshot_download,
    )

    return snapshot_download(_MODEL, cache_dir=str(tmp_path_factory.mktemp("hf")))


@pytest.mark.parametrize("peft", ["none", "lora"])
def test_cpu_run_prints_step_lines_and_saves(tiny_model, tmp_path, peft):
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        }
        for i in range(8)
    ]
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "out"
    job = {
        "model_name_or_path": tiny_model,
        "training_data_path": str(data),
        "output_dir": str(tmp_path / "scratch"),
        "save_model_dir": str(out),
        "save_strategy": "no",
        "logging_strategy": "steps",
        "logging_steps": 1,
        "max_steps": 2,
        "per_device_train_batch_size": 2,
        "use_flash_attn": False,
        "torch_dtype": "float32",
        "use_cpu": True,
        "max_seq_length": 128,
        "dataset_conversation_field": "messages",
        "instruction_template": "<|start_header_id|>user<|end_header_id|>",
        "response_template": "<|start_header_id|>assistant<|end_header_id|>",
    }
    if peft == "lora":
        job.update(peft_method="lora", r=4, lora_alpha=8, target_modules=["all-linear"])
    (tmp_path / "job.json").write_text(json.dumps(job))
    env = dict(
        os.environ,
        SFT_TRAINER_CONFIG_JSON_PATH=str(tmp_path / "job.json"),
        TERMINATION_LOG_FILE=str(tmp_path / "err.log"),
    )
    proc = subprocess.run(
        [sys.executable, str(_ENTRY)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )
    assert proc.returncode == 0, (
        (tmp_path / "err.log").read_text()
        if (tmp_path / "err.log").exists()
        else proc.stderr[-2000:]
    )
    steps = re.findall(r"Step: (\d+)/2 loss=", proc.stdout)
    assert steps == ["1", "2"], proc.stdout[-2000:]
    if peft == "lora":
        assert (out / "adapter_config.json").is_file()
        assert (out / "adapter_model.safetensors").is_file()
    else:
        assert (out / "config.json").is_file() and (out / "model.safetensors").is_file()
