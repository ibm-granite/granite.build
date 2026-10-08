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

"""Unit tests for the gold-smoke distillation recipe (issue #60).

The recipe is a build.yaml + parameters.yaml pair, so the thing that can break it
is a parameter that exists in one file and not the other — which on a live run
surfaces only after LSF has queued a 2-node H100 allocation. Rendering catches one
direction on its own: apply_parameters uses StrictUndefined, so a marker with no
declared value raises. The opposite direction — a parameter declared and never used
— raises nothing, and gets more likely once the gold-offpolicy-* recipes share this
build.yaml across several parameters.yaml files, so it is asserted explicitly.

The rest pin what a live run would otherwise be the first to reveal: the three
lineage inputs, their agreement with the paths gold_config hands the trainer and
with the step fixture's own copies, and the value TYPES, since a float written
unquoted in parameters.yaml round-trips into build.yaml as a string.
"""

import pathlib
import re

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_RECIPE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "recipes"
    / "granite4-gold-distillation"
    / "lsf"
    / "gold-smoke"
)


@pytest.fixture
def rendered(tmp_path):
    """The recipe rendered against its own parameters.yaml.

    apply_parameters writes a parameters-applied side-effect file into the folder
    it is handed, so it gets tmp_path rather than the recipe dir.
    """
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    params = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    return yaml.safe_load(apply_parameters(contents, [], params, str(tmp_path)))


def _target(rendered):
    targets = rendered["granite.build"]["targets"]
    assert list(targets) == ["gold-smoke"], "recipe is a single-target build"
    return targets["gold-smoke"]


_FIXTURE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "steps"
    / "distill"
    / "gold"
    / "skypilot"
    / "test-data"
    / "lsf"
    / "build.yaml"
)

# apply_parameters' delimiters: variable_start_string="$${", variable_end_string="}".
_MARKER = re.compile(r"\$\$\{([A-Za-z0-9_]+)\}")


def test_parameter_sets_match_exactly():
    """Every marker is declared, and every declaration is used.

    Rendering only catches the first direction, and only by raising. An unused
    parameter is silent — it reads as configuration that does something, and the
    next reader tunes it expecting an effect. That is the direction the planned
    gold-offpolicy-2node / -4node recipes make likely, since they will drive this
    same build.yaml from their own parameters.yaml.
    """
    text = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    used = set(_MARKER.findall(text))
    declared = set(get_params_from_file(str(_RECIPE / "parameters.yaml")))

    assert used - declared == set(), "markers with no declared parameter"
    assert declared - used == set(), "parameters declared but never used"


def test_no_single_dollar_substitution_markers():
    """A `${VAR}` typo is passed through untouched, not substituted.

    Only `$${` opens a marker, so `${VAR}` survives rendering verbatim and reaches
    the cluster as a literal — a path like `${STUDENT_MODEL}` that no error
    mentions. `$$` and `{{ }}` are both legitimate here (the latter is gbserver's
    own Jinja, e.g. `env://{{ binding.path }}`), so this looks only for the exact
    single-dollar-brace form.
    """
    text = (_RECIPE / "build.yaml").read_text(encoding="utf-8")

    assert "${" not in text.replace("$${", "")


def test_three_lineage_inputs_declared(rendered):
    """The teacher, student and dataset are lineage inputs.

    Lineage inputs[] is built from the target's input_artifacts, i.e. this block —
    see WandBLineageStore.create_jobstats_for_target. A lineage-supported `type` is
    required for the artifact to appear.
    """
    inputs = _target(rendered)["inputs"]
    assert set(inputs) == {"teacher_model", "student_model", "training_dataset"}
    assert inputs["teacher_model"]["type"] == "model"
    assert inputs["student_model"]["type"] == "model"
    assert inputs["training_dataset"]["type"] == "dataset"
    for name, spec in inputs.items():
        # env:// needs an absolute path (pullasset_envstore rejects a relative
        # one), so the scheme is followed by three slashes once /proj/... lands.
        assert spec["uri"].startswith("env:///"), name


def test_inputs_agree_with_what_the_trainer_is_given(rendered):
    """The lineage record and the trainer's arguments cannot disagree.

    Both sides resolve from the same parameter, so this pins the wiring: an input
    renamed on one side only would make lineage describe a run that did not happen.
    """
    target = _target(rendered)
    inputs = target["inputs"]
    gold = target["steps"][0]["config"]["gold_config"]

    assert inputs["student_model"]["uri"] == "env://" + gold["model_name_or_path"]
    assert (
        inputs["teacher_model"]["uri"] == "env://" + gold["teacher_model_name_or_path"]
    )
    assert inputs["training_dataset"]["uri"] == "env://" + gold["dataset_name"]


def test_dataset_is_think_filtered(rendered):
    """An inline <think>...</think> in an assistant turn breaks gold's completion
    extraction, silently. The reference and smoke datasets are *_nothink.jsonl."""
    gold = _target(rendered)["steps"][0]["config"]["gold_config"]
    assert "nothink" in gold["dataset_name"]


def test_fixture_declares_the_same_three_paths(rendered):
    """The step fixture's inputs and this recipe's parameters must agree.

    steps/distill/gold/skypilot/test-data/lsf/build.yaml states in a comment that
    its inputs are "the same three the gold-smoke recipe declares". It cannot use
    $${VAR} — the build-test harness passes no parameters file — so the paths are
    literal there and nothing but this test keeps the claim honest. Without it the
    fixture can drift to a different pair while still asserting it matches, and the
    comment becomes a confident lie about what is covered.
    """
    fixture_inputs = yaml.safe_load(_FIXTURE.read_text(encoding="utf-8"))[
        "granite.build"
    ]["targets"]["gold-smoke"]["inputs"]
    params = get_params_from_file(str(_RECIPE / "parameters.yaml"))

    assert fixture_inputs["teacher_model"]["uri"] == "env://" + params["TEACHER_MODEL"]
    assert fixture_inputs["student_model"]["uri"] == "env://" + params["STUDENT_MODEL"]
    assert (
        fixture_inputs["training_dataset"]["uri"]
        == "env://" + params["TRAINING_DATASET"]
    )


def test_every_float_parameter_survives_as_a_float(rendered):
    """Guards the parameters.yaml quoting trap for EVERY float, not a named few.

    A float written unquoted in parameters.yaml is rendered by Jinja as its Python
    repr (1e-05), which PyYAML then re-reads from build.yaml as a string. Quoted
    ("1.0e-05") it is emitted verbatim and parses as a float.

    Deriving the set from parameters.yaml rather than naming keys is the point: a
    parameter added later hits the identical trap, and a test listing today's two
    float keys would not fire on it.
    """
    params = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    text = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    step_config = _target(rendered)["steps"][0]["config"]
    flat = {**step_config["gold_config"], **step_config["compute_config"]}

    # Map each build.yaml field to the parameter it substitutes, so a declared
    # float can be checked against the type it actually rendered to.
    field_to_param = {
        field: param
        for field, param in re.findall(r"^\s*([a-z_]+):\s*\"?\$\$\{(\w+)\}", text, re.M)
    }
    floats = {p for p, v in params.items() if isinstance(v, float)}
    quoted_floats = {
        p
        for p, v in params.items()
        if isinstance(v, str) and re.fullmatch(r"[-+0-9.]+([eE][-+]?\d+)?", v)
    }

    checked = 0
    for field, param in field_to_param.items():
        if param in floats | quoted_floats and field in flat:
            assert isinstance(flat[field], float), f"{field} <- {param} is not a float"
            checked += 1
    assert checked >= 2, "expected at least LEARNING_RATE and MIN_LR to be covered"


def test_integer_and_boolean_parameters_keep_their_types(rendered):
    """Ints and bools have their own trap-free path, but a stray quote in either
    parameters.yaml or build.yaml turns them into strings, which reaches the step
    as a rendered config value rather than an error."""
    target = _target(rendered)
    gold = target["steps"][0]["config"]["gold_config"]

    assert isinstance(gold["max_length"], int)
    assert isinstance(gold["save_steps"], int)
    assert isinstance(gold["gradient_checkpointing"], bool)
    assert isinstance(gold["use_liger_fused_jsd"], bool)
    assert isinstance(target["steps"][0]["config"]["compute_config"]["num_nodes"], int)


def test_smoke_run_is_off_policy_and_not_liger(rendered):
    """The two settings whose wrong value corrupts a run rather than failing it:
    liger's fused bf16 JSD overflows on granite's logits_scaling=10 (NaN loss), and
    vllm_num_servers > 0 would dedicate nodes to serving instead of training."""
    gold = _target(rendered)["steps"][0]["config"]["gold_config"]
    assert gold["use_liger_fused_jsd"] is False
    assert gold["lmbda"] == 0.0
    assert gold["vllm_num_servers"] == 0


def test_effective_batch_is_the_documented_16(rendered):
    """Locks the smoke effective batch at 16 = per_device x grad_accum x nodes x
    gpus_per_node.

    The reference run is 192 and the LR schedule was tuned against it; the smoke run
    deliberately does not hold that, trading a comparable optimization for a low
    step count. 16 is therefore a documented choice rather than a derived value, and
    this fails if any of the four factors moves without the README's arithmetic and
    the parameters.yaml comment moving with it."""
    target = _target(rendered)
    gold = target["steps"][0]["config"]["gold_config"]
    compute = target["steps"][0]["config"]["compute_config"]

    effective = (
        gold["per_device_train_batch_size"]
        * gold["gradient_accumulation_steps"]
        * compute["num_nodes"]
        * compute["num_gpus_per_node"]
    )
    assert effective == 16, "2 nodes x 8 GPUs x 1 x 1 for the smoke run"
