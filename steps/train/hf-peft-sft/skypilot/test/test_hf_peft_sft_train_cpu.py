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

"""CPU smoke test: the real training script on a tiny model, 2 steps.

Skipped unless peft and trl import (the repo venv has neither). Run it in a
separate venv built from the image lock (README.md, "Tests"). It catches TRL /
peft API drift before any GPU time. qlora needs CUDA, so it is e2e-only.
"""

import json
import pathlib
import subprocess
import sys

import pytest

pytest.importorskip("peft")
pytest.importorskip("trl")

_TRAIN = pathlib.Path(__file__).resolve().parents[1] / "src" / "hf_peft_sft_train.py"
_MODEL = "trl-internal-testing/tiny-LlamaForCausalLM-3.2"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    from huggingface_hub import snapshot_download

    return snapshot_download(_MODEL, cache_dir=str(tmp_path_factory.mktemp("hf")))


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ds")
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": f"a{i}"},
            ]
        }
        for i in range(8)
    ]
    (d / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def _train(model_dir, dataset_dir, out, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(_TRAIN),
            "--model_name_or_path",
            model_dir,
            "--output_dir",
            str(out),
            "--dataset_path",
            str(dataset_dir),
            "--mixed_precision",
            "no",
            "--bf16",
            "false",
            "--use_cpu",
            "true",
            "--max_steps",
            "2",
            "--logging_steps",
            "1",
            "--per_device_train_batch_size",
            "2",
            "--max_length",
            "128",
            "--report_to",
            "none",
            "--save_strategy",
            "no",
            "--gradient_checkpointing",
            "false",
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("method", ["lora", "alora"])
def test_trains_and_saves_an_adapter(tmp_path, model_dir, dataset_dir, method):
    extra = ["--method", method, "--base_model_id", "org/tiny"]
    if method == "alora":
        extra += [
            "--alora_invocation_string",
            "<|start_header_id|>assistant<|end_header_id|>",
        ]
    proc = _train(model_dir, dataset_dir, tmp_path / "out", *extra)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "Step: 2/2" in proc.stdout
    out = tmp_path / "out"
    cfg = json.loads((out / "adapter_config.json").read_text())
    assert (out / "adapter_model.safetensors").is_file()
    assert (out / "tokenizer_config.json").is_file()
    assert cfg["base_model_name_or_path"] == "org/tiny"
    assert bool(cfg.get("alora_invocation_tokens")) == (method == "alora")


def test_wrong_invocation_fails_before_training(tmp_path, model_dir, dataset_dir):
    proc = _train(
        model_dir,
        dataset_dir,
        tmp_path / "out",
        "--method",
        "alora",
        "--alora_invocation_string",
        "<|im_start|>assistant",
    )
    assert proc.returncode == 1
    assert "hf-peft-sft: ERROR" in proc.stderr
    assert "never appears" in proc.stderr
    assert not (tmp_path / "out" / "adapter_config.json").exists()
