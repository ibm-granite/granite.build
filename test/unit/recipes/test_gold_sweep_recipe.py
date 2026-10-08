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

"""Unit tests for the gold-sweep-100 / gold-sweep-smoke distillation recipes.

The pair is one arm at two scales: gold-sweep-smoke is gold-sweep-100's shape with
only the cost shrunk, and it exists so that a wrong path or a bad render costs
minutes rather than the two hours and ~15 GPU-h the real arm costs. Every shared
assertion below therefore runs against both, so the gate cannot pass while the run
it gates would fail.

Two things here are not in test_gold_smoke_recipe.py, and both are failures that
produce a plausible-looking run rather than an error:

* the RESPONSE_TEMPLATE trailing newline. Written with double quotes it renders as
  a trailing SPACE, because YAML folds a newline inside a double-quoted scalar.
  That moves the loss-mask boundary and nothing complains.
* the corpus. Every kd-sandbox dataset is named *_nothink, and gold-smoke asserts
  that; prep's output is train.jsonl and carries no such marker, so filename is not
  available as a check here. The allowlist below is what replaces it.
"""

import pathlib
import re

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_LSF = (
    pathlib.Path(__file__).resolve().parents[3]
    / "recipes"
    / "granite4-gold-distillation"
    / "lsf"
)

_RECIPES = ("gold-sweep-100", "gold-sweep-smoke")

# apply_parameters' delimiters: variable_start_string="$${", variable_end_string="}".
_MARKER = re.compile(r"\$\$\{([A-Za-z0-9_]+)\}")

# Corpora whose think-filtering has been established, and how. gold's completion
# extraction breaks on an inline <think>...</think> in an assistant turn, silently,
# so a corpus may only be used here once somebody has checked. The *_nothink naming
# convention covers the kd-sandbox datasets; prep's output has no such marker, so it
# is listed explicitly with the evidence. Repointing TRAINING_DATASET at something
# unchecked fails this test rather than the run.
_THINK_CHECKED_CORPORA = {
    # 20,000-row head sample, zero <think> occurrences, 2026-09-13. Schema is
    # {messages, row_id}; 802,027 rows / 3.2 GB.
    "/proj/data-eng/hew/gb-steps-collection-post-training/data/distillation"
    "/prepped/en-sft-4.1-0.2-16K-v2/merged/train.jsonl",
}


def _params(recipe):
    return get_params_from_file(str(_LSF / recipe / "parameters.yaml"))


def _render(recipe, tmp_path):
    """The recipe rendered against its own parameters.yaml.

    apply_parameters writes a parameters-applied side-effect file into the folder it
    is handed, so it gets tmp_path rather than the recipe dir.
    """
    contents = (_LSF / recipe / "build.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(
        apply_parameters(contents, [], _params(recipe), str(tmp_path))
    )


def _target(rendered, recipe):
    targets = rendered["granite.build"]["targets"]
    assert list(targets) == [recipe], "recipe is a single-target build"
    return targets[recipe]


def _gold(rendered, recipe):
    return _target(rendered, recipe)["steps"][0]["config"]["gold_config"]


# ─── The pair is one build.yaml ────────────────────────────────────────────────


def test_the_two_build_files_differ_only_by_name():
    """The gate must be the same pipeline as the thing it gates.

    Both recipes carry their own build.yaml so that each directory is
    self-contained and `gb build start -f <dir>/build.yaml` picks up the sibling
    parameters.yaml with no extra flag. The cost of that is two copies, and a copy
    drifts. Normalising the names away and diffing is what makes a divergence a
    test failure instead of a discovery two hours into a run.
    """
    texts = []
    for recipe in _RECIPES:
        text = (_LSF / recipe / "build.yaml").read_text(encoding="utf-8")
        for name in _RECIPES:
            text = text.replace(name, "RECIPE")
        # The header's one-line description of what each recipe is for is allowed
        # to differ; nothing below `granite.build:` is.
        texts.append(text[text.index("granite.build:") :])

    assert texts[0] == texts[1]


def test_the_two_parameter_sets_have_the_same_keys():
    """Same keys, different values: that is what makes the two comparable, and it
    is what lets the shared build.yaml render under either."""
    assert set(_params(_RECIPES[0])) == set(_params(_RECIPES[1]))


# ─── Shared assertions, run against both recipes ───────────────────────────────


@pytest.mark.parametrize("recipe", _RECIPES)
def test_parameter_sets_match_exactly(recipe):
    """Every marker is declared, and every declaration is used.

    Rendering catches the first direction on its own, by raising under
    StrictUndefined. An unused parameter is silent — it reads as configuration that
    does something, and the next reader tunes it expecting an effect. With one
    build.yaml driven by two parameter files, that is the likely direction.
    """
    text = (_LSF / recipe / "build.yaml").read_text(encoding="utf-8")
    used = set(_MARKER.findall(text))
    declared = set(_params(recipe))

    assert used - declared == set(), "markers with no declared parameter"
    assert declared - used == set(), "parameters declared but never used"


@pytest.mark.parametrize("recipe", _RECIPES)
def test_no_single_dollar_substitution_markers(recipe):
    """A `${VAR}` typo is passed through untouched, not substituted, and reaches the
    cluster as a literal path that no error mentions."""
    text = (_LSF / recipe / "build.yaml").read_text(encoding="utf-8")

    assert "${" not in text.replace("$${", "")


@pytest.mark.parametrize("recipe", _RECIPES)
def test_three_lineage_inputs_declared(recipe, tmp_path):
    """Lineage inputs[] is built from the target's input_artifacts, not from step
    config, so a path handed only to gold_config records nothing."""
    inputs = _target(_render(recipe, tmp_path), recipe)["inputs"]

    assert set(inputs) == {"teacher_model", "student_model", "training_dataset"}
    assert inputs["teacher_model"]["type"] == "model"
    assert inputs["student_model"]["type"] == "model"
    assert inputs["training_dataset"]["type"] == "dataset"
    for name, spec in inputs.items():
        # env:// needs an absolute path; pullasset_envstore rejects a relative one.
        assert spec["uri"].startswith("env:///"), name


@pytest.mark.parametrize("recipe", _RECIPES)
def test_inputs_agree_with_what_the_trainer_is_given(recipe, tmp_path):
    """An input renamed on one side only would make lineage describe a run that did
    not happen."""
    rendered = _render(recipe, tmp_path)
    inputs = _target(rendered, recipe)["inputs"]
    gold = _gold(rendered, recipe)

    assert inputs["student_model"]["uri"] == "env://" + gold["model_name_or_path"]
    assert (
        inputs["teacher_model"]["uri"] == "env://" + gold["teacher_model_name_or_path"]
    )
    assert inputs["training_dataset"]["uri"] == "env://" + gold["dataset_name"]


@pytest.mark.parametrize("recipe", _RECIPES)
def test_corpus_think_filtering_has_been_established(recipe, tmp_path):
    """gold-smoke can assert *_nothink in the filename; these recipes cannot.

    prep's output is `train.jsonl`, so the naming convention that carries the
    guarantee for every kd-sandbox dataset is simply absent. Falling back to "no
    check" would be the wrong trade — an inline <think> breaks completion
    extraction with no error — so the guarantee moves into an explicit allowlist
    that records how each corpus was checked.
    """
    dataset = _gold(_render(recipe, tmp_path), recipe)["dataset_name"]

    assert "nothink" in dataset or dataset in _THINK_CHECKED_CORPORA, (
        f"{dataset} is neither named *_nothink nor listed in "
        "_THINK_CHECKED_CORPORA; sample it for <think> before using it"
    )


@pytest.mark.parametrize("recipe", _RECIPES)
def test_response_template_transports_its_newline_as_an_escape(recipe, tmp_path):
    """The specific silent failure this pair was most likely to ship — and did.

    The template must reach the trainer ending in a newline: that is where loss
    masking begins, and the published sweep's own repo carries a dedicated check
    for it. What it must NOT do is carry that newline as a real newline through
    build.yaml, because gbserver's config fill (fill_objtemplate -> Jinja without
    keep_trailing_newline) strips exactly one trailing newline from every config
    VALUE. Build d8470f14 rendered a real newline here, dispatched
    `--response-template '<|im_start|>assistant'`, and trained against a span one
    token off while reporting success.

    So the newline crosses the wire as a literal backslash-n, which has no trailing
    whitespace to strip, and distill-gold's renderer decodes it exactly once at the
    far end (test_distill_gold.py covers that decode).

    Getting the escape itself through two templating layers is the trap.
    Double-quoted in parameters.yaml, YAML would decode \\n to a real newline here.
    Single-quoted with a doubled backslash, the literal two-character escape
    survives parameters.yaml and build.yaml both.
    """
    template = _gold(_render(recipe, tmp_path), recipe)["response_template"]

    assert template == "<|im_start|>assistant\\n"
    assert not template.endswith(
        "\n"
    ), "a real newline here is stripped by the config fill before the step sees it"
    assert not template.endswith(" "), "rendered as a trailing space, not an escape"


@pytest.mark.parametrize("recipe", _RECIPES)
def test_run_is_step_bounded(recipe, tmp_path):
    """These recipes exist to be bounded by optimizer steps rather than by epochs,
    because their corpus is far too large to epoch through — one epoch is 4,177
    steps at effective batch 192. max_steps must render as an int: a string reaches
    the trainer as a config value and fails against a step counter partway in."""
    gold = _gold(_render(recipe, tmp_path), recipe)

    assert isinstance(gold["max_steps"], int)
    assert gold["max_steps"] > 0
    # Both keys travel together: the LR schedule is built from the epoch count even
    # when max_steps truncates the run.
    assert gold["num_train_epochs"] == pytest.approx(1.0)


@pytest.mark.parametrize("recipe", _RECIPES)
def test_run_is_off_policy_single_node(recipe, tmp_path):
    """lmbda 0.0 is the arm; on one node the renderer would refuse anything else
    (on-policy needs at least two nodes, one to serve and one to train)."""
    rendered = _render(recipe, tmp_path)
    gold = _gold(rendered, recipe)
    compute = _target(rendered, recipe)["steps"][0]["config"]["compute_config"]

    assert gold["lmbda"] == pytest.approx(0.0)
    assert gold["vllm_num_servers"] == 0
    assert compute["num_nodes"] == 1


@pytest.mark.parametrize("recipe", _RECIPES)
def test_objective_is_a_true_mixture(recipe, tmp_path):
    """beta 0.5 is the point of difference from gold-smoke, which runs 0.0.

    The trainer short-circuits beta 0.0 to forward KL and 1.0 to reverse KL, so
    those two endpoints are not mixtures at all. A recipe that drifted back to 0.0
    would still train, still report a loss, and be a different objective than the
    arm it claims to reproduce.
    """
    assert _gold(_render(recipe, tmp_path), recipe)["beta"] == pytest.approx(0.5)


@pytest.mark.parametrize("recipe", _RECIPES)
def test_every_float_parameter_survives_as_a_float(recipe, tmp_path):
    """A float written unquoted in parameters.yaml renders as its Python repr
    (1e-05), which PyYAML then re-reads from build.yaml as a string.

    The set is derived from parameters.yaml rather than named, so a float parameter
    added later hits the identical trap and is covered without editing this test.
    """
    params = _params(recipe)
    text = (_LSF / recipe / "build.yaml").read_text(encoding="utf-8")
    step_config = _render(recipe, tmp_path)["granite.build"]["targets"][recipe][
        "steps"
    ][0]["config"]
    flat = {**step_config["gold_config"], **step_config["compute_config"]}

    field_to_param = dict(re.findall(r"^\s*([a-z_]+):\s*\"?\$\$\{(\w+)\}", text, re.M))
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


@pytest.mark.parametrize("recipe", _RECIPES)
def test_integer_and_boolean_parameters_keep_their_types(recipe, tmp_path):
    """A stray quote in either file turns these into strings, which reaches the step
    as a rendered config value rather than an error."""
    rendered = _render(recipe, tmp_path)
    gold = _gold(rendered, recipe)

    for key in ("max_length", "save_steps", "max_steps", "dataset_num_proc"):
        assert isinstance(gold[key], int), key
    for key in ("gradient_checkpointing", "use_liger_fused_jsd"):
        assert isinstance(gold[key], bool), key
    assert isinstance(gold["save_strategy"], str)


# ─── Per-recipe values ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "recipe,steps,per_device,grad_accum,effective_batch",
    [
        # The published sweep's geometry exactly: 4 x 6 x 1 x 8 = 192, held equal
        # across all three of its arms, which is what made them comparable.
        ("gold-sweep-100", 100, 4, 6, 192),
        # Deliberately NOT 192. Effective batch is not meaningful at two steps, and
        # shrinking it is what keeps each step cheap; nothing compares this run's
        # loss to anything.
        ("gold-sweep-smoke", 2, 1, 1, 8),
    ],
)
def test_geometry_is_the_documented_one(
    recipe, steps, per_device, grad_accum, effective_batch, tmp_path
):
    """Effective batch = per_device x grad_accum x nodes x gpus_per_node.

    The published arm's LR schedule was tuned against 192, so a recipe claiming to
    reproduce it and running a different global batch has changed the optimization
    rather than the throughput. Asserting the product rather than the four factors
    is what catches a compensating edit to only one of them.
    """
    rendered = _render(recipe, tmp_path)
    gold = _gold(rendered, recipe)
    compute = _target(rendered, recipe)["steps"][0]["config"]["compute_config"]

    assert gold["max_steps"] == steps
    assert gold["per_device_train_batch_size"] == per_device
    assert gold["gradient_accumulation_steps"] == grad_accum
    assert (
        per_device * grad_accum * compute["num_nodes"] * compute["num_gpus_per_node"]
        == effective_batch
    )


def test_smoke_and_real_arm_share_every_behavioural_parameter(tmp_path):
    """The gate is only a gate for what it holds identical.

    Anything in this list differing between the two would mean the smoke run
    exercises a different behaviour than the run it clears — the failure mode where
    a green gate is actively misleading. The parameters NOT listed are the cost
    knobs the smoke tier is allowed to shrink, plus the two documented deviations:
    NCCL_DEBUG/NCCL_TIMEOUT_MS (a smoke run wants a fast, loud failure).
    """
    behavioural = (
        "DS_CONFIG",
        "IMAGE_ID",
        "TEACHER_MODEL",
        "STUDENT_MODEL",
        "TRAINING_DATASET",
        "NUM_TRAIN_EPOCHS",
        "LEARNING_RATE",
        "MIN_LR",
        "WARMUP_RATIO",
        "LR_SCHEDULER_TYPE",
        "GRADIENT_CHECKPOINTING",
        "SAVE_STRATEGY",
        "TEMPERATURE",
        "LMBDA",
        "BETA",
        "USE_LIGER_FUSED_JSD",
        "RESPONSE_TEMPLATE",
        "VLLM_NUM_SERVERS",
        "TOP_P",
        "USE_SAMPLED_OPD_LOSS",
        "LAST_MESSAGE_ONLY",
        "CLIP_ALPHA",
        "OPD_IMPORTANCE_SAMPLING",
        "NUM_NODES",
        "NUM_GPUS_PER_NODE",
        "ACCELERATORS",
        "MEMORY",
        "CLUSTER",
        "QUEUE",
        "DATASET_NUM_PROC",
        "NCCL_ENABLE_MONITORING",
    )
    real, smoke = _params("gold-sweep-100"), _params("gold-sweep-smoke")

    differing = {k: (real[k], smoke[k]) for k in behavioural if real[k] != smoke[k]}
    assert differing == {}, f"gate diverges from the run it gates: {differing}"
