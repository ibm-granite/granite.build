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

"""Unit tests for src/fms_hf_tuning_job.py that need no fms-hf-tuning.

coerce_args runs against a small argparse stand-in built with the same action
kinds fms-hf-tuning's HfArgumentParser produces (typed, nargs="+", bool with
nargs="?", choices); test_fms_hf_tuning_job_upstream.py repeats the key cases
against the real parser.
"""

import argparse
import base64
import json
import os
import struct

import fms_hf_tuning_job as job
import pytest


def _str2bool(v):
    if str(v).lower() in ("yes", "true", "t", "y", "1"):
        return True
    if str(v).lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Truthy value expected: got {v}")


@pytest.fixture
def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--max_steps", type=int)
    p.add_argument("--packing", type=_str2bool, nargs="?", const=True)
    p.add_argument("--target_modules", type=str, nargs="+")
    p.add_argument("--lr_scheduler_type", type=str, choices=["linear", "cosine"])
    p.add_argument("--response_template", type=str)
    p.add_argument("--gradient_checkpointing_kwargs", type=str)
    # A default-True bool, as HfArgumentParser adds it: the typed action, then a
    # `--no_<field>` store_false twin with the same dest and no type.
    p.add_argument(
        "--use_flash_attn", type=_str2bool, nargs="?", const=True, default=True
    )
    p.add_argument("--no_use_flash_attn", action="store_false", dest="use_flash_attn")
    return p


class TestCoerce:
    def test_converts_like_the_command_line(self, parser):
        out = job.coerce_args(
            parser,
            {
                "learning_rate": "1e-5",
                "max_steps": "15",
                "packing": "true",
                "target_modules": "all-linear",
                "lr_scheduler_type": "cosine",
            },
        )
        assert out == {
            "learning_rate": 1e-5,
            "max_steps": 15,
            "packing": True,
            "target_modules": ["all-linear"],
            "lr_scheduler_type": "cosine",
        }

    def test_native_yaml_values(self, parser):
        out = job.coerce_args(
            parser,
            {
                "learning_rate": 2e-4,
                "max_steps": 3,
                "packing": False,
                "target_modules": ["q", "v"],
            },
        )
        assert out == {
            "learning_rate": 2e-4,
            "max_steps": 3,
            "packing": False,
            "target_modules": ["q", "v"],
        }

    def test_mapping_and_null_pass_through(self, parser):
        out = job.coerce_args(
            parser,
            {
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
                "max_steps": None,
            },
        )
        assert out == {
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "max_steps": None,
        }

    def test_default_true_bool_is_converted(self, parser):
        assert job.coerce_args(parser, {"use_flash_attn": "false"}) == {
            "use_flash_attn": False
        }
        assert job.coerce_args(parser, {"use_flash_attn": False}) == {
            "use_flash_attn": False
        }

    def test_text_is_kept_byte_for_byte(self, parser):
        text = '\n### Answer: it\'s "q" {{ x }} ü'
        assert job.coerce_args(parser, {"response_template": text}) == {
            "response_template": text
        }

    @pytest.mark.parametrize(
        "args,needle",
        [
            ({"lerning_rate": 1}, "args.lerning_rate is not an fms-hf-tuning argument"),
            ({"help": 1}, "args.help is not an fms-hf-tuning argument"),
            (
                {"learning_rate": "abc"},
                "args.learning_rate: 'abc' is not a valid value",
            ),
            ({"max_steps": 1.5}, "args.max_steps: 1.5 is not a valid value"),
            ({"packing": "maybe"}, "args.packing: 'maybe' is not a valid value"),
            (
                {"use_flash_attn": "maybe"},
                "args.use_flash_attn: 'maybe' is not a valid value",
            ),
            ({"lr_scheduler_type": "nope"}, "is not one of linear, cosine"),
            ({"max_steps": [1, 2]}, "args.max_steps takes one value, got a list"),
            ({"target_modules": []}, "args.target_modules is an empty list"),
        ],
    )
    def test_refusals(self, parser, args, needle):
        with pytest.raises(job.StepError, match=None) as exc:
            job.coerce_args(parser, args)
        assert needle in str(exc.value)


def _merge(args=None, **over):
    kw = dict(
        base_model="/m",
        output_dir="/scratch",
        save_model_dir="/out.partial",
        peft_method="none",
        data_path="/ds",
        data_config_path="",
        clearml=False,
        clearml_project="",
        clearml_task="",
    )
    kw.update(over)
    return job.merge_job(args or {}, **kw)


class TestMerge:
    def test_owned_keys_and_defaults(self):
        j = _merge({"learning_rate": 1e-5})
        assert j == {
            "save_strategy": "no",
            "logging_strategy": "steps",
            "logging_steps": 10,
            "learning_rate": 1e-5,
            "model_name_or_path": "/m",
            "output_dir": "/scratch",
            "save_model_dir": "/out.partial",
            "report_to": "none",
            "training_data_path": "/ds",
        }

    def test_user_overrides_step_defaults(self):
        j = _merge({"save_strategy": "steps", "save_steps": 10, "logging_steps": 1})
        assert j["save_strategy"] == "steps" and j["logging_steps"] == 1

    def test_peft_method_only_for_adapters(self):
        assert "peft_method" not in _merge()
        assert _merge(peft_method="lora")["peft_method"] == "lora"

    def test_data_config_replaces_training_data_path(self):
        j = _merge(data_path="", data_config_path="/w/dc.json")
        assert j["data_config_path"] == "/w/dc.json" and "training_data_path" not in j

    def test_clearml(self):
        j = _merge(peft_method="alora", clearml=True)
        assert j["trackers"] == ["clearml"]
        assert (
            j["clearml_project"] == "fms-hf-tuning" and j["clearml_task"] == "sft-alora"
        )
        j = _merge(clearml=True, clearml_project="p", clearml_task="t")
        assert (j["clearml_project"], j["clearml_task"]) == ("p", "t")
        assert "trackers" not in _merge()

    @pytest.mark.parametrize("key", sorted(job.OWNED))
    def test_owned_key_refused(self, key):
        with pytest.raises(job.StepError, match="is set by the step"):
            _merge({key: "x"})


def _safetensors(path, n):
    header = {"__metadata__": {"format": "pt"}}
    for i in range(n):
        header[f"t{i}"] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4 * i, 4 * i + 4],
        }
    raw = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)) + raw + b"\0" * (4 * n))


class TestCheckOutput:
    def test_adapter(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text("{}")
        _safetensors(tmp_path / "adapter_model.safetensors", 3)
        assert job.check_output(str(tmp_path), "lora") == 3

    def test_empty_adapter_refused(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text("{}")
        _safetensors(tmp_path / "adapter_model.safetensors", 0)
        with pytest.raises(job.StepError, match="no weights"):
            job.check_output(str(tmp_path), "alora")

    @pytest.mark.parametrize(
        "missing", ["adapter_config.json", "adapter_model.safetensors"]
    )
    def test_adapter_file_missing(self, tmp_path, missing):
        (tmp_path / "adapter_config.json").write_text("{}")
        _safetensors(tmp_path / "adapter_model.safetensors", 1)
        os.remove(tmp_path / missing)
        with pytest.raises(job.StepError, match=f"has no {missing}"):
            job.check_output(str(tmp_path), "lora")

    def test_full_model_single_file(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        _safetensors(tmp_path / "model.safetensors", 20)
        assert job.check_output(str(tmp_path), "none") == 20

    def test_full_model_sharded(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        _safetensors(tmp_path / "model-00001-of-00002.safetensors", 2)
        _safetensors(tmp_path / "model-00002-of-00002.safetensors", 1)
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "a": "model-00001-of-00002.safetensors",
                        "b": "model-00001-of-00002.safetensors",
                        "c": "model-00002-of-00002.safetensors",
                    }
                }
            )
        )
        assert job.check_output(str(tmp_path), "none") == 3

    def test_full_model_missing_shard(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors"}})
        )
        with pytest.raises(job.StepError, match="which is missing"):
            job.check_output(str(tmp_path), "none")

    def test_full_model_without_config_or_weights(self, tmp_path):
        with pytest.raises(job.StepError, match="has no config.json"):
            job.check_output(str(tmp_path), "none")
        (tmp_path / "config.json").write_text("{}")
        with pytest.raises(job.StepError, match="no \\*.safetensors weights"):
            job.check_output(str(tmp_path), "none")

    def test_full_model_with_empty_weights(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        _safetensors(tmp_path / "model.safetensors", 0)
        with pytest.raises(job.StepError, match="no weights"):
            job.check_output(str(tmp_path), "none")

    @pytest.mark.parametrize("text", ["{not json", "[]", '{"weight_map": []}'])
    def test_bad_index_refused(self, tmp_path, text):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors.index.json").write_text(text)
        with pytest.raises(job.StepError, match="model.safetensors.index.json"):
            job.check_output(str(tmp_path), "none")

    def test_unreadable_safetensors(self, tmp_path):
        (tmp_path / "adapter_config.json").write_text("{}")
        (tmp_path / "adapter_model.safetensors").write_bytes(b"\xff" * 4)
        with pytest.raises(job.StepError, match="not a readable safetensors file"):
            job.check_output(str(tmp_path), "lora")


def test_rewrite_base_model(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "/staged/x", "r": 8})
    )
    job.rewrite_base_model(str(tmp_path), "org/model")
    cfg = json.loads((tmp_path / "adapter_config.json").read_text())
    assert cfg == {"base_model_name_or_path": "org/model", "r": 8}


@pytest.mark.parametrize("text", ["{not json", "[]"])
def test_rewrite_base_model_bad_config_refused(tmp_path, text):
    (tmp_path / "adapter_config.json").write_text(text)
    with pytest.raises(job.StepError, match="adapter_config.json"):
        job.rewrite_base_model(str(tmp_path), "org/model")


def test_decode_b64_json():
    assert job.decode_b64_json(base64.b64encode(b'{"a": 1}').decode(), "x") == {"a": 1}
    with pytest.raises(job.StepError, match="x is not base64-encoded JSON"):
        job.decode_b64_json("not base64!", "x")


class TestCli:
    def test_check_output_prints_ok_and_rewrites(self, tmp_path, capsys):
        (tmp_path / "adapter_config.json").write_text(
            '{"base_model_name_or_path": "/s"}'
        )
        _safetensors(tmp_path / "adapter_model.safetensors", 2)
        job.main(["check-output", str(tmp_path), "lora", "--base-model-id", "org/m"])
        out = capsys.readouterr().out.splitlines()
        assert out[0] == "fms-hf-tuning-output: ok 2"
        assert out[1].startswith("fms-hf-tuning-versions: ")
        assert (
            json.loads((tmp_path / "adapter_config.json").read_text())[
                "base_model_name_or_path"
            ]
            == "org/m"
        )

    def test_check_output_error_exits_1(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc:
            job.main(["check-output", str(tmp_path), "none"])
        assert exc.value.code == 1
        assert capsys.readouterr().err.startswith("fms-hf-tuning: ERROR ")

    def test_build_unwritable_job_out_exits_1(self, tmp_path, capsys, monkeypatch):
        # Skip the upstream parse (the repo venv has no fms-hf-tuning); the
        # failure under test is the job JSON write.
        monkeypatch.setattr(job, "strict_check", lambda j: j)
        e = base64.b64encode(b"{}").decode()
        with pytest.raises(SystemExit) as exc:
            job.main(
                [
                    "build",
                    "--args-b64",
                    e,
                    "--data-config-b64",
                    e,
                    "--data-path",
                    "/ds",
                    "--base-model",
                    "/m",
                    "--peft-method",
                    "none",
                    "--output-dir",
                    "/o",
                    "--save-model-dir",
                    "/s",
                    "--clearml",
                    "off",
                    "--clearml-project",
                    "",
                    "--clearml-task",
                    "",
                    "--job-out",
                    str(tmp_path / "missing" / "j.json"),
                    "--data-config-out",
                    str(tmp_path / "dc.json"),
                ]
            )
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert err.startswith("fms-hf-tuning: ERROR ") and "j.json" in err

    def test_build_unwritable_data_config_out_exits_1(self, tmp_path, capsys):
        e = base64.b64encode(b"{}").decode()
        dc = base64.b64encode(b'{"datasets": []}').decode()
        with pytest.raises(SystemExit) as exc:
            job.main(
                [
                    "build",
                    "--args-b64",
                    e,
                    "--data-config-b64",
                    dc,
                    "--data-path",
                    "",
                    "--base-model",
                    "/m",
                    "--peft-method",
                    "none",
                    "--output-dir",
                    "/o",
                    "--save-model-dir",
                    "/s",
                    "--clearml",
                    "off",
                    "--clearml-project",
                    "",
                    "--clearml-task",
                    "",
                    "--job-out",
                    str(tmp_path / "j.json"),
                    "--data-config-out",
                    str(tmp_path / "missing" / "dc.json"),
                ]
            )
        assert exc.value.code == 1
        err = capsys.readouterr().err
        assert err.startswith("fms-hf-tuning: ERROR ") and "dc.json" in err

    def test_build_without_any_data_exits_1(self, tmp_path, capsys):
        e = base64.b64encode(b"{}").decode()
        with pytest.raises(SystemExit) as exc:
            job.main(
                [
                    "build",
                    "--args-b64",
                    e,
                    "--data-config-b64",
                    e,
                    "--data-path",
                    "",
                    "--base-model",
                    "/m",
                    "--peft-method",
                    "none",
                    "--output-dir",
                    "/o",
                    "--save-model-dir",
                    "/s",
                    "--clearml",
                    "off",
                    "--clearml-project",
                    "",
                    "--clearml-task",
                    "",
                    "--job-out",
                    str(tmp_path / "j.json"),
                    "--data-config-out",
                    str(tmp_path / "dc.json"),
                ]
            )
        assert exc.value.code == 1
        assert "no training data path and no data_config" in capsys.readouterr().err
