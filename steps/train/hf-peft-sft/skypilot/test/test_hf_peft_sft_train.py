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

"""Unit tests for the pure helpers in src/hf_peft_sft_train.py.

The module imports with the standard library only (torch/peft/trl are imported
inside main()), so these run in the repo venv. PYTHONPATH=src (make test) puts
it on the path.
"""

import dataclasses
import re

import hf_peft_sft_train as t
import pytest


class TestFormat:
    @pytest.mark.parametrize(
        "cols,fmt",
        [
            (["messages"], "messages"),
            (["source", "messages", "num_turns"], "messages"),
            (["prompt", "completion"], "prompt_completion"),
            (["text"], "text"),
            (["id", "prompt", "messages"], "messages"),
        ],
    )
    def test_detects(self, cols, fmt):
        assert t.detect_format(cols) == fmt

    @pytest.mark.parametrize("cols", [["input", "output"], ["prompt"], []])
    def test_unknown_refused(self, cols):
        with pytest.raises(t.StepError, match="messages.*prompt.*completion.*text"):
            t.detect_format(cols)


class TestInvocation:
    def test_subsequence(self):
        assert t.contains_subsequence([5, 1, 2, 3, 9], [1, 2, 3])
        assert t.contains_subsequence([1, 2], [1, 2])
        assert not t.contains_subsequence([1, 2], [1, 2, 3])
        assert not t.contains_subsequence([1, 3, 2], [1, 2])

    def test_all_rows_hit(self):
        assert t.check_invocation([[7, 1, 2], [1, 2, 0]], [1, 2]) == []

    def test_some_rows_miss_returns_them(self):
        assert t.check_invocation([[1, 2], [3, 4], [9, 1, 2]], [1, 2]) == [1]

    def test_every_row_misses_is_an_error(self):
        with pytest.raises(t.StepError, match="never appears"):
            t.check_invocation([[3, 4], [5]], [1, 2])

    def test_empty_tokens_is_an_error(self):
        with pytest.raises(t.StepError, match="tokenizes to nothing"):
            t.check_invocation([[1]], [])


class TestTargetModules:
    def test_all_linear(self):
        assert t.parse_target_modules(["all-linear"]) == "all-linear"

    def test_list(self):
        assert t.parse_target_modules(["q_proj", "v_proj"]) == ["q_proj", "v_proj"]

    def test_empty_refused(self):
        with pytest.raises(t.StepError):
            t.parse_target_modules([])


class TestReentrant:
    @pytest.mark.parametrize(
        "method,dist,ws,expected",
        [
            ("qlora", "ddp", 1, True),
            ("qlora", "ddp", 2, False),
            ("qlora", "fsdp", 2, True),
            ("qlora", "deepspeed", 2, True),
            ("lora", "ddp", 1, False),
            ("lora", "fsdp", 4, False),
            ("alora", "ddp", 2, False),
            ("lora", "deepspeed", 2, True),
            ("alora", "deepspeed", 4, True),
        ],
    )
    def test_rule(self, method, dist, ws, expected):
        assert t.use_reentrant(method, dist, ws) is expected


class TestModelKwargs:
    def test_lora_ddp(self):
        assert t.model_kwargs("lora", "ddp", "bf16", 0) == {
            "dtype": "bfloat16",
            "attn_implementation": "sdpa",
        }

    def test_lora_no_precision_is_fp32(self):
        assert t.model_kwargs("alora", "fsdp", "no", 0)["dtype"] == "float32"

    def test_qlora_ddp_pins_the_device(self):
        kw = t.model_kwargs("qlora", "ddp", "bf16", 3)
        assert kw["device_map"] == {"": 3}
        assert kw["quantization"] == t.bnb_kwargs("bf16", sharded=False)
        assert kw["dtype"] == "bfloat16"

    @pytest.mark.parametrize("dist", ["fsdp", "deepspeed"])
    def test_qlora_sharded_stores_in_bf16_and_lets_the_launcher_place(self, dist):
        kw = t.model_kwargs("qlora", dist, "bf16", 1)
        assert "device_map" not in kw
        assert kw["quantization"]["bnb_4bit_quant_storage"] == "bfloat16"
        assert kw["dtype"] == "bfloat16"

    def test_bnb_defaults(self):
        assert t.bnb_kwargs("bf16", sharded=False) == {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16",
            "bnb_4bit_quant_storage": "uint8",
        }

    def test_bnb_no_precision_computes_in_fp32(self):
        assert t.bnb_kwargs("no", sharded=False)["bnb_4bit_compute_dtype"] == "float32"


class TestLoraKwargs:
    def test_plain(self):
        assert t.lora_kwargs(16, 32, 0.05, "all-linear", None, None) == {
            "task_type": "CAUSAL_LM",
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "bias": "none",
            "target_modules": "all-linear",
        }

    def test_alora_and_modules_to_save(self):
        kw = t.lora_kwargs(8, 16, 0.0, ["q_proj"], ["lm_head"], [1, 2])
        assert kw["alora_invocation_tokens"] == [1, 2]
        assert kw["modules_to_save"] == ["lm_head"]


def test_step_line():
    assert t.step_line(3, 10, {"loss": 0.70123, "lr": 1e-4}) == "Step: 3/10 loss=0.7012"
    assert t.step_line(1, -1, {}) == "Step: 1"
    assert re.match(r"Step:\s\d+", t.step_line(2, 5, {"loss": 1.0}))


def test_rewrite_base_model():
    cfg = {"base_model_name_or_path": "/proj/cache/x", "r": 16}
    assert t.rewrite_base_model(cfg, "ibm-granite/g") == {
        "base_model_name_or_path": "ibm-granite/g",
        "r": 16,
    }
    assert t.rewrite_base_model(cfg, "") == cfg


def test_step_arg_fields_match_the_dataclass():
    assert tuple(f.name for f in dataclasses.fields(t.StepArgs)) == t.STEP_ARG_FIELDS
    assert len(t.STEP_ARG_FIELDS) == 13


def _safetensors(path, tensors):
    """Write a minimal safetensors file: 8-byte header length + JSON header."""
    import json
    import struct

    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name in tensors:
        header[name] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [offset, offset + 4],
        }
        offset += 4
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


class TestSavedAdapter:
    def test_counts_tensors_from_the_header(self, tmp_path):
        p = tmp_path / "adapter_model.safetensors"
        _safetensors(
            p,
            [
                "base_model.model.q_proj.lora_A.weight",
                "base_model.model.q_proj.lora_B.weight",
            ],
        )
        assert t.saved_tensor_count(str(p)) == 2

    def test_empty_adapter_is_an_error(self, tmp_path):
        # The 40-byte file FSDP produced: a header with only __metadata__.
        p = tmp_path / "adapter_model.safetensors"
        _safetensors(p, [])
        with pytest.raises(t.StepError, match="no weights"):
            t.check_saved_adapter(str(p))

    def test_non_empty_adapter_passes(self, tmp_path):
        p = tmp_path / "adapter_model.safetensors"
        _safetensors(p, ["base_model.model.q_proj.lora_A.weight"])
        t.check_saved_adapter(str(p))

    @pytest.mark.parametrize("content", [b"", b"\x01\x00", b"\xff" * 8 + b"x"])
    def test_unreadable_file_is_an_error(self, tmp_path, content):
        p = tmp_path / "adapter_model.safetensors"
        p.write_bytes(content)
        with pytest.raises(t.StepError, match="not a readable safetensors"):
            t.check_saved_adapter(str(p))
