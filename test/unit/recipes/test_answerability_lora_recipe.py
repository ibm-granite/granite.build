#!/usr/bin/env python3

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

"""Unit tests for the granite-switch answerability-lora recipe.

The recipe is rendered with its parameters and then through hf-peft-sft's own
expanded step-template blocks with the strict renderer, so a config error that
would fail the build at step creation fails here instead. The prepare target's command is
rendered the way the command step renders it and run on a small parquet with
the schema of the HF source (pre-tokenized columns next to prompt/completion).
"""

import importlib.util
import pathlib
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters
from gbserver.utils.template import fill_objtemplate, fill_template

_REPO = pathlib.Path(__file__).resolve().parents[3]
_RECIPE = (
    _REPO
    / "recipes"
    / "granite-switch"
    / "skypilot"
    / "slurm-bluevela"
    / "answerability-lora"
)
_STEP = _REPO / "steps" / "train" / "hf-peft-sft" / "skypilot"


def _expanded_template() -> dict:
    """The template as `make space` renders it: gb-include lines expanded."""
    script = _REPO / "steps" / "scripts" / "render_step_template.py"
    spec = importlib.util.spec_from_file_location("render_step_template", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return yaml.safe_load(module.expand(_STEP / "step-template.yaml"))


_TEMPLATE = _expanded_template()
_LAUNCHER = _TEMPLATE["environment_configs"]["Skypilot"]["launchers"]["hf_peft_sft"][
    "config"
]
_SCRIPTS = ("hf_peft_sft_guard.sh", "hf_peft_sft_setup.sh", "hf_peft_sft_run.sh")


def _build(tmp_path, *params, overrides=None):
    """The recipe rendered with parameters.yaml, --param style *params*, and
    *overrides* merged into the file values (for values --param can't carry)."""
    file_params = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    text = apply_parameters(
        (_RECIPE / "build.yaml").read_text(),
        list(params),
        {**file_params, **(overrides or {})},
        str(tmp_path),
    )
    return yaml.safe_load(text)["granite.build"]


def _target(tmp_path, *params):
    return _build(tmp_path, *params)["targets"]["train"]


def _prepare(tmp_path, *params):
    return _build(tmp_path, *params)["targets"]["prepare"]


def _step_config(target):
    (step,) = target["steps"]
    assert step["step_uri"] == "space://steps/train/hf-peft-sft"
    return step["config"]


def _run_argv(target):
    """The trainer argv the step's run block hands hf_peft_sft_run.sh.

    The step config first goes through fill_objtemplate, as gbserver's target
    step run does: it renders every string value as a template, which is
    where a value's trailing newline is lost.
    """
    config = fill_objtemplate(_step_config(target), {}, strict=False)
    cfg = {**_TEMPLATE["config"]["hf_peft_sft_config"], **config["hf_peft_sft_config"]}
    cfg["image_id"] = "docker:quay.io/x/img:t"
    ctx = {
        "config": {**config, "hf_peft_sft_config": cfg},
        "bindings": {
            "base_model": {"binding": {"path": "/staged/base"}},
            "dataset": {"binding": {"path": "/staged/ds"}},
        },
    }
    for block in ("setup", "run"):
        text = fill_template(_LAUNCHER[block], ctx, strict=True)
    for s in _SCRIPTS:
        # NUL-separated, so an argument with a newline (the aLoRA invocation)
        # comes back whole.
        stub = (
            "f() { printf 'ARG:%s\\0' \"$@\"; }; f"
            if s == "hf_peft_sft_run.sh"
            else "true"
        )
        text = text.replace(f"bash /opt/hf-peft-sft/{s}", stub)
    proc = subprocess.run(
        ["bash", "-c", text], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return [a[4:] for a in proc.stdout.split("\0") if a.startswith("ARG:")]


def _flags(argv):
    """`--key value` pairs after the `--` that ends the run script's own options."""
    rest = argv[argv.index("--") + 1 :]
    return dict(zip(rest[::2], rest[1::2]))


def test_environment_is_bluevela_slurm(tmp_path):
    target = _target(tmp_path)
    assert target["environment_uri"] == "space://environments/skypilot/slurm/bluevela"
    resources = _step_config(target)["launcher_config"]["resources"]
    assert resources["accelerators"].startswith("H200:")
    assert isinstance(resources["memory"], str)


def test_regular_lora_with_the_prepared_dataset(tmp_path):
    target = _target(tmp_path)
    cfg = _step_config(target)["hf_peft_sft_config"]
    assert cfg["method"] == "lora"
    assert "dataset_name" not in cfg
    assert target["inputs"]["dataset"] == {"binding": "prepare.dataset"}


def test_prepare_pulls_only_the_parquet_dir_from_hf(tmp_path):
    from gbcommon.uri.hf import HfType, HfURI

    source = _prepare(tmp_path)["inputs"]["source"]
    assert source["type"] == "dataset"
    uri = HfURI.parse(source["uri"])
    assert uri.get_hf_type() == HfType.DATASET
    assert f"{uri.get_owner()}/{uri.get_repo()}" == "ibm-research/touma"
    assert uri.get_path_in_repo() == "training_data/answerability/parquet_granite4.2_8b"


def test_prepare_writes_to_shared_storage_on_the_cpu_partition(tmp_path):
    prepare = _prepare(tmp_path, "RUN_ID=abc")
    out = prepare["outputs"]["dataset"]
    assert out["uri"].startswith("env:///proj/data-eng/llmb-read-write/")
    assert out["uri"].endswith("/abc")
    (step,) = prepare["steps"]
    assert step["step_uri"] == "space://steps/command"
    assert step["config"]["zone"] == "cpu"
    assert step["config"]["launcher_config"]["sbatch_options"] == {"constraint": None}
    resources = step["config"]["launcher_config"]["resources"]
    assert "accelerators" not in resources
    # cpu's MaxMemPerCPU is 8 GiB: more makes SLURM raise cpus-per-task past
    # SkyPilot's srun --cpus, and srun aborts.
    assert int(resources["memory"]) <= 8 * int(resources["cpus"])


def test_prepare_runs_in_the_image_the_train_runs_in(tmp_path):
    """prepare borrows the hf-peft-sft image for its venv, counts tokens with
    its transformers to drop the rows train would truncate, and shares the
    node's image cache with train. So PREP_IMAGE must follow the image of the
    published step (configurations/), which train runs unless it overrides it."""
    published = (
        _REPO / "configurations/assets/environments/skypilot/steps/train/hf-peft-sft"
    )
    step = yaml.safe_load((published / "step.yaml").read_text())
    build = _build(tmp_path)
    image = build["targets"]["prepare"]["steps"][0]["config"]["command_config"]["image"]
    train = _step_config(build["targets"]["train"])["hf_peft_sft_config"]
    train_image = train.get(
        "image_id", step["config"]["hf_peft_sft_config"]["image_id"]
    )
    assert image == train_image.removeprefix("docker:")


def test_output_path_matches_the_output_uri(tmp_path):
    target = _target(tmp_path, "RUN_ID=abc")
    cfg = _step_config(target)["hf_peft_sft_config"]
    assert target["outputs"][cfg["output"]]["type"] == "model"
    assert target["outputs"][cfg["output"]]["uri"] == "env://" + cfg["output_path"]
    assert cfg["output_path"].endswith("/abc")


def test_hyperparameters_mirror_the_source_config(tmp_path):
    """The answerability LoRA recipe: r16/a32, lr 5e-5 cosine, batch 32."""
    argv = _run_argv(_target(tmp_path))
    flags = _flags(argv)
    assert argv[argv.index("--method") + 1] == "lora"
    assert {k: flags[k] for k in ("--lora_r", "--lora_alpha", "--lora_dropout")} == {
        "--lora_r": "16",
        "--lora_alpha": "32",
        "--lora_dropout": "0.05",
    }
    assert flags["--lora_target_modules"] == "all-linear"
    expected = {
        "--max_length": "8192",
        "--per_device_train_batch_size": "1",
        "--gradient_accumulation_steps": "8",
        "--learning_rate": "5e-05",
        "--lr_scheduler_type": "cosine",
        "--warmup_steps": "0.05",
        "--weight_decay": "0.01",
        "--num_train_epochs": "3",
        "--seed": "0",
    }
    assert {k: flags[k] for k in expected} == expected
    assert flags["--dataset_path"] == "/staged/ds"


def test_clearml_off_by_default_declares_no_secrets(tmp_path):
    target = _target(tmp_path)
    argv = _run_argv(target)
    assert argv[argv.index("--clearml-mode") + 1] == "off"
    assert "skypilot" not in _step_config(target)


def test_clearml_on_declares_the_three_secrets(tmp_path):
    target = _target(tmp_path, "CLEARML=true")
    argv = _run_argv(target)
    assert argv[argv.index("--clearml-mode") + 1] == "on"
    declared = _step_config(target)["skypilot"]["secrets"][
        "secret_names_to_use_as_env_variable"
    ]
    assert {d["env_name"] for d in declared} == {
        "CLEARML_API_HOST",
        "CLEARML_API_ACCESS_KEY",
        "CLEARML_API_SECRET_KEY",
    }


def test_smoke_overrides_render(tmp_path):
    target = _target(
        tmp_path, "DATASET_SPLIT=train[:64]", "MAX_STEPS=4", "ACCELERATORS=H200:1"
    )
    flags = _flags(_run_argv(target))
    assert flags["--dataset_split"] == "train[:64]"
    assert flags["--max_steps"] == "4"


# ── prepare ──────────────────────────────────────────────────────────────────


# A stand-in for transformers' tokenizer: one token per whitespace-separated
# word of each message's content, plus 4 per message for the template's role
# markers. Enough to test which rows prepare drops; the real tokenizer is the
# base model's, staged by the hfpull.
_FAKE_TRANSFORMERS = """
class _Tok:
    # Returns what transformers 5 returns: a mapping, whose len() is its number
    # of keys, not of tokens.
    def apply_chat_template(self, messages, tokenize=True):
        ids = [0] * sum(4 + len(m["content"].split()) for m in messages)
        return dict(input_ids=ids, attention_mask=[1] * len(ids))


class AutoTokenizer:
    @staticmethod
    def from_pretrained(path):
        assert path.endswith("/model"), path
        return _Tok()
"""


def _prepare_script(tmp_path, src, out_root, *params):
    """prepare's command, rendered as the command step renders it.

    The hf-peft-sft image's venv python is swapped for this interpreter, with a
    stub transformers first on its path. The script runs with -I, which ignores
    PYTHONPATH, so a -c launcher inserts the stub dir and execs the heredoc.
    """
    import sys

    stub = tmp_path / "stub"
    stub.mkdir(exist_ok=True)
    (stub / "transformers.py").write_text(_FAKE_TRANSFORMERS)
    step = _prepare(tmp_path, f"DATA_ROOT={out_root}", *params)["steps"][0]
    command = fill_template(
        step["config"]["command_config"]["command"],
        {
            "bindings": {
                "source": {"binding": {"path": str(src)}},
                "base_model": {"binding": {"path": str(tmp_path / "model")}},
            }
        },
        strict=True,
    )
    launcher = (
        "import sys; "
        f"sys.path.insert(0, {str(stub)!r}); "
        "sys.argv = sys.argv[1:]; "
        "exec(compile(sys.stdin.read(), 'prepare', 'exec'), dict(__name__='__main__'))"
    )
    return command.replace('"$PY" -I -', f'"{sys.executable}" -I -c "{launcher}" _')


def _source_table(n, bad=(), long=()):
    """The HF source's columns: pre-tokenized ids next to prompt/completion.

    Rows in *bad* end their prompt with an `agent` turn, as 6 rows of the real
    train split do. Rows in *long* carry a 9,000-word document, past the
    default MAX_LENGTH of 8192 with the stub tokenizer.
    """

    def doc(i):
        return " ".join(["word"] * 9000) if i in long else "doc"

    prompts = [
        [
            {
                "role": "user",
                "content": f"You have access to the following documents: {doc(i)}",
            },
            {"role": "agent" if i in bad else "user", "content": f"q{i}?"},
        ]
        for i in range(n)
    ]
    completions = [
        [
            {
                "role": "assistant",
                "content": '"answerable"' if i % 2 else '"unanswerable"',
            }
        ]
        for i in range(n)
    ]
    ids = pa.array([[1, 2, 3]] * n, type=pa.list_(pa.int32()))
    return pa.table(
        {
            "input_ids": ids,
            "labels": ids,
            "attention_mask": ids,
            "prompt": pa.array(prompts),
            "completion": pa.array(completions),
        }
    )


def _write_src(tmp_path, train_rows=6, bad=(), long=()):
    src = tmp_path / "src"
    src.mkdir()
    pq.write_table(_source_table(train_rows, bad, long), src / "train.parquet")
    pq.write_table(_source_table(2), src / "test.parquet")
    return src


def _run_prepare(tmp_path, src, *params):
    script = _prepare_script(tmp_path, src, tmp_path / "data", "RUN_ID=r1", *params)
    return subprocess.run(
        ["bash", "-c", script], capture_output=True, text=True, check=False
    )


def test_prepare_keeps_only_prompt_and_completion(tmp_path):
    proc = _run_prepare(tmp_path, _write_src(tmp_path))
    assert proc.returncode == 0, proc.stderr
    out = tmp_path / "data" / "r1"
    for split, rows in (("train", 6), ("test", 2)):
        table = pq.read_table(out / "data" / f"{split}.parquet")
        # No input_ids: TRL would take the data as pre-tokenized and skip its
        # chat-template tokenization and completion-only masking.
        assert table.column_names == ["prompt", "completion"]
        assert table.num_rows == rows
    readme = (out / "README.md").read_text()
    assert "source_sha256" in readme
    assert "hf:///datasets/ibm-research/touma/main/" in readme
    assert f"GB_ARTIFACT_ID:dataset GB_ARTIFACT_PATH:{out}" in proc.stdout


def test_prepare_output_matches_the_trl_prompt_completion_format(tmp_path):
    hf_train = pytest.importorskip("hf_peft_sft_train", reason="needs step src on path")
    assert _run_prepare(tmp_path, _write_src(tmp_path)).returncode == 0
    cols = pq.read_table(
        tmp_path / "data" / "r1" / "data" / "train.parquet"
    ).column_names
    assert hf_train.detect_format(cols) == "prompt_completion"


def test_prepare_fails_without_the_parquet(tmp_path):
    (tmp_path / "src").mkdir()
    proc = _run_prepare(tmp_path, tmp_path / "src")
    assert proc.returncode != 0
    assert "train.parquet not found" in proc.stderr
    assert "GB_ARTIFACT_ID" not in proc.stdout


def test_prepare_drops_and_records_a_prompt_not_ending_in_a_user_turn(tmp_path):
    proc = _run_prepare(tmp_path, _write_src(tmp_path, train_rows=200, bad={7}))
    assert proc.returncode == 0, proc.stderr
    assert "train: dropping 1 row(s)" in proc.stdout
    out = tmp_path / "data" / "r1"
    table = pq.read_table(out / "data" / "train.parquet")
    assert table.num_rows == 199
    assert "q7?" not in str(table.column("prompt").to_pylist())
    readme = (out / "README.md").read_text()
    assert '"row": 7' in readme and '"last_prompt_role": "agent"' in readme


def test_prepare_fails_when_too_many_rows_are_bad(tmp_path):
    proc = _run_prepare(tmp_path, _write_src(tmp_path, train_rows=50, bad={1, 2}))
    assert proc.returncode != 0
    assert "2 of 50 rows" in proc.stderr
    assert "GB_ARTIFACT_ID" not in proc.stdout


def test_prepare_binds_the_base_model_for_its_tokenizer(tmp_path):
    inputs = _prepare(tmp_path)["inputs"]
    assert inputs["base_model"] == _target(tmp_path)["inputs"]["base_model"]


def test_prepare_drops_rows_longer_than_max_length(tmp_path):
    """TRL keeps a row's start, so the label of a row past max_length is cut
    off and TRL drops the row silently; prepare drops and records it instead."""
    src = _write_src(tmp_path, train_rows=200, bad={3}, long={5, 17})
    proc = _run_prepare(tmp_path, src)
    assert proc.returncode == 0, proc.stderr
    assert "train: dropping 2 row(s) longer than 8192 tokens" in proc.stdout
    out = tmp_path / "data" / "r1"
    table = pq.read_table(out / "data" / "train.parquet")
    assert table.num_rows == 197
    prompts = str(table.column("prompt").to_pylist())
    for gone in ("q3?", "q5?", "q17?"):
        assert gone not in prompts
    readme = (out / "README.md").read_text()
    # Numbered by source row, even after the earlier drop of row 3.
    assert '"row": 5,' in readme and '"row": 17,' in readme
    assert '"max_length": 8192' in readme


def test_prepare_fails_when_max_length_drops_too_many_rows(tmp_path):
    proc = _run_prepare(tmp_path, _write_src(tmp_path, train_rows=10, long={1, 2}))
    assert proc.returncode != 0
    assert "2 of 10 rows are longer than MAX_LENGTH=8192" in proc.stderr
    assert "GB_ARTIFACT_ID" not in proc.stdout


def test_train_max_length_follows_the_parameter(tmp_path):
    flags = _flags(_run_argv(_target(tmp_path, "MAX_LENGTH=4096")))
    assert flags["--max_length"] == "4096"


# ── aLoRA ────────────────────────────────────────────────────────────────────


def test_lora_by_default_with_gradient_checkpointing(tmp_path):
    target = _target(tmp_path)
    cfg = _step_config(target)["hf_peft_sft_config"]
    assert cfg["method"] == "lora"
    assert "alora_invocation_string" not in cfg
    argv = _run_argv(target)
    assert "--alora_invocation_string" not in argv
    assert _flags(argv)["--gradient_checkpointing"] == "true"


def test_alora_invocation_reaches_the_trainer(tmp_path):
    target = _target(tmp_path, "METHOD=alora")
    cfg = _step_config(target)["hf_peft_sft_config"]
    assert cfg["method"] == "alora"
    assert cfg["alora_invocation_string"] == "<|im_start|>assistant"
    argv = _run_argv(target)
    assert argv[argv.index("--method") + 1] == "alora"
    assert argv[argv.index("--alora_invocation_string") + 1] == "<|im_start|>assistant"


def test_a_trailing_newline_in_the_invocation_is_dropped(tmp_path):
    """Documents the gbserver behaviour the default works around (README,
    "aLoRA"): the build config keeps the newline, the trainer does not get it.
    If this starts failing, gbserver keeps trailing newlines and the default
    can follow the Granite 4.2 QR/QC adapters' "<|im_start|>assistant\\n"."""
    # From the parameters file, as the smoke run had it: a --param value loses
    # its trailing newline even earlier, when the CLI parses it.
    newline = {"ALORA_INVOCATION_STRING": "<|im_start|>assistant\n"}
    target = _build(tmp_path, "METHOD=alora", overrides=newline)["targets"]["train"]
    cfg = _step_config(target)["hf_peft_sft_config"]
    assert cfg["alora_invocation_string"] == "<|im_start|>assistant\n"
    argv = _run_argv(target)
    assert argv[argv.index("--alora_invocation_string") + 1] == "<|im_start|>assistant"


def test_alora_turns_gradient_checkpointing_off(tmp_path):
    """peft #2826: recomputation loses the aLoRA offsets and zeroes gradients."""
    flags = _flags(_run_argv(_target(tmp_path, "METHOD=alora")))
    assert flags["--gradient_checkpointing"] == "false"


def test_alora_names_the_build_and_task_after_the_method(tmp_path):
    build = _build(tmp_path, "METHOD=alora", "RUN_ID=a1")
    assert build["name"] == "answerability-alora-a1"
    cfg = _step_config(build["targets"]["train"])["hf_peft_sft_config"]
    assert cfg["clearml_task_name"] == "answerability-alora-a1"


def test_alora_hyperparameters_are_parameters(tmp_path):
    params = ("METHOD=alora", "LORA_R=32", "LORA_ALPHA=64", "LEARNING_RATE=1e-05")
    flags = _flags(_run_argv(_target(tmp_path, *params)))
    got = (flags["--lora_r"], flags["--lora_alpha"], flags["--learning_rate"])
    assert got == ("32", "64", "1e-05")


def test_adapter_dir_follows_the_method(tmp_path):
    """Same RUN_ID, different METHOD: the two adapters never share a dir."""
    root = "/proj/data-eng/llmb-read-write/touma/adapters"
    for method in ("lora", "alora"):
        target = _target(tmp_path, f"METHOD={method}", "RUN_ID=r1")
        cfg = _step_config(target)["hf_peft_sft_config"]
        assert cfg["output_path"] == f"{root}/answerability-{method}/r1"
        assert (
            target["outputs"]["adapter"]["uri"]
            == f"env://{root}/answerability-{method}/r1"
        )


def test_lora_is_the_default_method(tmp_path):
    target = _target(tmp_path, "RUN_ID=r1")
    cfg = _step_config(target)["hf_peft_sft_config"]
    assert cfg["method"] == "lora"
    # Unchanged from before METHOD existed, e.g. it1-r1's adapter.
    assert cfg["output_path"].endswith("/adapters/answerability-lora/r1")
