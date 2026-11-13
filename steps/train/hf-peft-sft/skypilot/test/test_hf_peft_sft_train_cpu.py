# Copyright Granite.Build Authors
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


_ROWS = [
    {
        "messages": [
            {"role": "user", "content": f"q{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]
    }
    for i in range(8)
]


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    """A hub-style dataset dir: a train.jsonl that load_dataset resolves."""
    d = tmp_path_factory.mktemp("ds")
    (d / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _ROWS))
    return d


@pytest.fixture(scope="module")
def saved_dataset_dir(tmp_path_factory):
    """A single-Dataset save_to_disk dir (state.json + data-*.arrow)."""
    from datasets import Dataset

    d = tmp_path_factory.mktemp("saved") / "ds"
    Dataset.from_list(_ROWS).save_to_disk(str(d))
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


def test_trains_on_a_save_to_disk_dataset_with_a_sliced_split(
    tmp_path, model_dir, saved_dataset_dir
):
    proc = _train(
        model_dir, saved_dataset_dir, tmp_path / "out", "--dataset_split", "train[:6]"
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "6 training rows, format messages" in proc.stdout
    assert "Step: 2/2" in proc.stdout
    assert (tmp_path / "out" / "adapter_model.safetensors").is_file()


def test_template_full_flags_match_the_parser():
    """The template's full_flags list is exactly the non-owned trainer flags that
    are a proper prefix of an owned one; the abbreviation check exempts them."""
    import re

    import hf_peft_sft_train as train
    from transformers import HfArgumentParser
    from trl import SFTConfig

    text = (_TRAIN.parents[1] / "step-template.yaml").read_text()

    def names(var):
        found = re.findall(r"set " + var + r" = \[(.*?)\]", text)[0]
        return set(re.findall(r'"([a-z_0-9]+)"', found))

    owned = names("owned")
    parser = HfArgumentParser((train.StepArgs, SFTConfig))
    flags = {
        s[2:].replace("-", "_")
        for a in parser._actions
        for s in a.option_strings
        if s.startswith("--")
    }
    expected = {
        f for f in flags - owned if any(o != f and o.startswith(f) for o in owned)
    }
    assert expected == names("full_flags")


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
