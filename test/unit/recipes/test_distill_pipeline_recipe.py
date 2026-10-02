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

"""Unit tests for the distill-pipeline-smoke recipe.

This recipe is the first one in the repository that chains the six ported
distillation steps into one build, so most of what is checked here is
AGREEMENT BETWEEN TARGETS — the class of error that produces a run which
completes and reports a number that means something other than what it says.

The recipe also has a switch, INCLUDE_SFT, which the other recipes here do
not: with it on, an SFT pass runs between the aligned student and the GOLD
trainer and GOLD continues from the SFT checkpoint. A switch that selects
targets is rendered by the CLI's parameter layer (block delimiters <% %>),
so it is checked in both positions AND with the string values that
`--param INCLUDE_SFT=false` actually delivers: a bare Jinja truth test on
the string "false" is true, which would turn the arm on while the command
line said to turn it off.
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
    / "distill-pipeline-smoke"
)

# apply_parameters' delimiters: variable_start_string="$${", variable_end_string="}".
_MARKER = re.compile(r"\$\$\{([A-Za-z0-9_]+)\}")

# The targets the recipe declares, in the order the pipeline runs them. train-sft
# is present only when the SFT arm is switched on.
_TARGETS_WITHOUT_SFT = [
    "align",
    "corpus",
    "train-gold",
    "export",
    "eval-transfer-baseline",
    "eval-transfer",
    "eval-bfcl",
]
_TARGETS_WITH_SFT = [
    "align",
    "corpus",
    "train-sft",
    "train-gold",
    "export",
    "eval-transfer-baseline",
    "eval-transfer",
    "eval-bfcl",
]


def _params(**overrides):
    params = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    params.update(overrides)
    return params


def _render(tmp_path, **overrides):
    """The recipe rendered against its own parameters.yaml.

    apply_parameters writes a parameters-applied side-effect file into the folder
    it is handed, so it gets tmp_path rather than the recipe directory.
    """
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(
        apply_parameters(contents, [], _params(**overrides), str(tmp_path))
    )


def _targets(rendered):
    return rendered["granite.build"]["targets"]


def _config(rendered, target):
    return _targets(rendered)[target]["steps"][0]["config"]


@pytest.fixture(name="off")
def fixture_off(tmp_path):
    return _render(tmp_path, INCLUDE_SFT=False)


@pytest.fixture(name="on")
def fixture_on(tmp_path):
    return _render(tmp_path, INCLUDE_SFT=True)


# ─── The parameter surface ─────────────────────────────────────────────────────


def test_every_marker_has_a_parameter(tmp_path):
    """apply_parameters uses StrictUndefined, so a marker with no parameter fails
    the render rather than rendering empty. Asserting it here is what makes that a
    test failure rather than a submission-time one."""
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    declared = set(_params())
    referenced = set(_MARKER.findall(contents))
    assert referenced - declared == set()


def test_every_parameter_is_referenced(tmp_path):
    """The other direction: a parameter nothing reads is a value somebody will
    change expecting an effect. Both arms are rendered, so a parameter used only
    inside the SFT block counts as referenced."""
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    referenced = set(_MARKER.findall(contents))
    # INCLUDE_SFT drives a <% if %> block rather than a $${...} marker, so it is
    # not in `referenced` and is not dead either.
    unread = set(_params()) - referenced - {"INCLUDE_SFT"}
    assert unread == set()


def test_the_recipe_renders_valid_yaml_in_both_positions(off, on):
    assert list(_targets(off)) == _TARGETS_WITHOUT_SFT
    assert list(_targets(on)) == _TARGETS_WITH_SFT


# ─── The SFT switch ────────────────────────────────────────────────────────────


def test_the_off_render_carries_no_reference_to_the_absent_sft_target(off):
    """A binding naming a target that is not in the build is a validation failure,
    so the switch has to remove every reference as well as the target itself.

    Checked structurally rather than by substring: the prose comments explain why
    the GPU targets are serialised and name train-sft while doing so, and a
    comment is not a reference.
    """
    targets = _targets(off)
    assert "train-sft" not in targets
    for name, target in targets.items():
        for input_name, spec in target.get("inputs", {}).items():
            binding = spec.get("binding")
            if binding is not None:
                assert not binding.startswith("train-sft."), f"{name}.{input_name}"
        config = target["steps"][0]["config"]
        assert "sft_config" not in config, name


def test_gold_continues_from_the_sft_checkpoint_when_the_arm_is_on(on, off):
    """The whole point of the switch: with the arm on, the student GOLD starts from
    is the SFT checkpoint, not the aligned student."""
    assert _targets(on)["train-gold"]["inputs"]["student"]["binding"] == (
        "train-sft.checkpoint"
    )
    assert _targets(off)["train-gold"]["inputs"]["student"]["binding"] == (
        "align.retagged_student"
    )
    # Either way the trainer reads the path from that one binding, so the two
    # cannot disagree about which model was loaded.
    for rendered in (on, off):
        gold = _config(rendered, "train-gold")["gold_config"]
        assert gold["model_name_or_path"] == "{{ bindings.student.binding.path }}"


def test_the_sft_arm_trains_the_aligned_student_on_the_prepared_corpus(on):
    sft_target = _targets(on)["train-sft"]
    assert sft_target["inputs"]["student"]["binding"] == "align.retagged_student"
    assert sft_target["inputs"]["corpus"]["binding"] == "corpus.corpus"
    sft = _config(on, "train-sft")["sft_config"]
    assert sft["student_model_path"] == "{{ bindings.student.binding.path }}"
    assert sft["corpus_path"] == "{{ bindings.corpus.binding.path }}"
    # EMPTY is what makes this a plain-SFT pass rather than a forward-KL arm: with
    # a logits directory it would be a distillation run of a different kind, and
    # the recipe would have two treatments and no control.
    assert sft["precomputed_logits_dir"] == ""


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, _TARGETS_WITH_SFT),
        (False, _TARGETS_WITHOUT_SFT),
        # What `gb build start --param INCLUDE_SFT=...` actually delivers:
        # add_parameter does no YAML parsing, so the value arrives as a string.
        # "false" is truthy to a bare Jinja `if`, so a recipe that tested
        # truthiness would turn the arm ON here while the command line said off.
        ("true", _TARGETS_WITH_SFT),
        ("false", _TARGETS_WITHOUT_SFT),
        ("True", _TARGETS_WITH_SFT),
        ("False", _TARGETS_WITHOUT_SFT),
        ("yes", _TARGETS_WITH_SFT),
        ("no", _TARGETS_WITHOUT_SFT),
        ("1", _TARGETS_WITH_SFT),
        ("0", _TARGETS_WITHOUT_SFT),
    ],
)
def test_the_switch_reads_command_line_strings_the_way_a_human_means_them(
    tmp_path, value, expected
):
    assert list(_targets(_render(tmp_path, INCLUDE_SFT=value))) == expected


# ─── Ordering edges ────────────────────────────────────────────────────────────


def test_every_edge_the_pipeline_depends_on_is_a_binding(off, on):
    """Each edge below exists because the downstream step cannot be correct
    without the upstream artifact, and a binding is the only thing that both
    delivers the path and orders the targets."""
    for rendered in (off, on):
        targets = _targets(rendered)
        assert targets["corpus"]["inputs"]["tokenizer"]["binding"] == (
            "align.retagged_student"
        )
        assert targets["train-gold"]["inputs"]["corpus"]["binding"] == "corpus.corpus"
        assert targets["export"]["inputs"]["train_output"]["binding"] == (
            "train-gold.checkpoint"
        )
        assert targets["export"]["inputs"]["expected_tokenizer"]["binding"] == (
            "align.retagged_student"
        )
        # The t=0 read is the UNTRAINED aligned student; the post-training read is
        # the EXPORTED model. Swapping these two silently reports no transfer.
        assert targets["eval-transfer-baseline"]["inputs"]["student"]["binding"] == (
            "align.retagged_student"
        )
        assert targets["eval-transfer"]["inputs"]["student"]["binding"] == (
            "export.hf_model"
        )
        assert targets["eval-bfcl"]["inputs"]["model"]["binding"] == "export.hf_model"


def test_a_failed_target_is_retried_with_the_earlier_ones_reused(off):
    """Builds 7853d33b and b5f030cd lost three GPU targets to a single bad host
    (p2-r03-n1: 3 runs, 3 cuda_init failures; every other host in the logs, none).
    No build.yaml can steer LSF away from a host, so the recipe's only lever is to
    run the target again. target_reuse_enabled is the part that matters: without it
    a fault in eval-transfer would re-train the whole pipeline."""
    retries = off["granite.build"]["retries"]
    assert retries["max_retries"] >= 1
    assert retries["target_reuse_enabled"] is True


def test_the_gpu_targets_are_not_serialised_by_hand(off):
    """An earlier revision chained the eval targets with ordering-only bindings,
    on the theory that two concurrent single-GPU jobs on one host collided under
    mode=exclusive_process. Build b5f030cd disproved it: fully serialised, the
    third eval still failed — on p2-r03-n1 again, while the identical step had
    just succeeded on another host. The edges were removed rather than left as
    harmless belt-and-braces, because a workaround whose stated reason is false
    is worse than none: it invites the same wrong diagnosis next time.

    Concurrency remains UNTESTED on a healthy pair of hosts. If it ever does turn
    out to be a real constraint, the fix belongs in the environment or in retries,
    not in a hand-built chain that also costs the baseline its independence.
    """
    targets = _targets(off)
    assert set(targets["eval-transfer-baseline"]["inputs"]) == {
        "student",
        "corpus",
        "teacher_model",
    }
    assert set(targets["eval-transfer"]["inputs"]) == {
        "student",
        "corpus",
        "teacher_model",
    }
    assert set(targets["eval-bfcl"]["inputs"]) == {"model"}


def test_both_evals_bind_the_corpus_for_the_ordering_edge(off):
    """eval.jsonl is deliberately not a declared artifact of corpus-prep — it
    exists only when eval_fraction is above zero, and a declared-but-absent output
    is a resolver failure — so its path is composed. The binding is what still
    guarantees neither eval target can start before the split exists."""
    for target in ("eval-transfer-baseline", "eval-transfer"):
        assert _targets(off)[target]["inputs"]["corpus"]["binding"] == "corpus.corpus"


def test_every_artifact_each_step_emits_is_declared_as_an_output(off, on):
    """An undeclared output makes the buildrun resolver drop the NEWARTIFACT event
    and the target completes with no output AND no error — a silent failure that
    costs the whole pipeline. The counts here are the artifact ids each step
    echoes as GB_ARTIFACT_ID."""
    expected = {
        "align": {"retagged_student", "teacher_overlay", "student_overlay"},
        "corpus": {"corpus"},
        "train-gold": {"checkpoint"},
        "export": {"hf_model"},
        "eval-transfer-baseline": {"eval_metrics"},
        "eval-transfer": {"eval_metrics"},
        "eval-bfcl": {"bfcl_results"},
    }
    for target, outputs in expected.items():
        assert set(_targets(off)[target]["outputs"]) == outputs, target
    assert set(_targets(on)["train-sft"]["outputs"]) == {"checkpoint"}


# ─── Agreement between targets ─────────────────────────────────────────────────


def test_one_teacher_for_the_whole_pipeline(off, on):
    """The student is retagged onto the TEACHER's tokenizer, trains toward that
    teacher, and is measured against it. A different teacher in any one of those
    places still runs: it reports a divergence against a model the student never
    trained toward."""
    for rendered in (off, on):
        teacher = _config(rendered, "align")["align_config"]["teacher_model"]
        assert (
            _config(rendered, "train-gold")["gold_config"]["teacher_model_name_or_path"]
            == teacher
        )
        for target in ("eval-transfer-baseline", "eval-transfer"):
            assert _config(rendered, target)["eval_config"]["teacher_model"] == teacher


def test_one_length_budget_for_the_whole_pipeline(on):
    """prep filters in tokens, the trainers train to a budget, and the evals score
    to one. A smaller budget downstream scores a truncated completion and reports
    the result as the divergence."""
    budget = _config(on, "corpus")["corpus_config"]["max_length"]
    assert _config(on, "train-gold")["gold_config"]["max_length"] == budget
    assert _config(on, "train-sft")["sft_config"]["max_length"] == budget
    for target in ("eval-transfer-baseline", "eval-transfer"):
        assert _config(on, target)["eval_config"]["max_length"] == budget


def test_the_two_transfer_evals_differ_only_in_the_student(off):
    """Same step, same teacher, same corpus, same budget, same metrics — that is
    the only way the two numbers are comparable at all."""
    baseline = dict(_config(off, "eval-transfer-baseline")["eval_config"])
    trained = dict(_config(off, "eval-transfer")["eval_config"])
    for key in ("student_model", "output_dir"):
        baseline.pop(key)
        trained.pop(key)
    assert baseline == trained


def test_the_evals_read_the_split_the_corpus_target_wrote(off):
    """The eval corpus path is composed rather than bound, so it has to be
    composed from the SAME out_dir the corpus target was given. An absolute shared
    root is forced here, not chosen: every target run gets its own
    GB_BUILD_WORKDIR, so a relative path would point into the wrong directory."""
    out_dir = _config(off, "corpus")["corpus_config"]["out_dir"]
    assert out_dir.startswith("/"), "the shared corpus root must be absolute"
    for target in ("eval-transfer-baseline", "eval-transfer"):
        assert _config(off, target)["eval_config"]["corpus"] == f"{out_dir}/eval.jsonl"


def test_the_corpus_has_an_eval_split_to_measure(off):
    """At eval_fraction 0 the split does not exist and both eval targets fail on a
    missing file — after align, corpus and training have been paid for."""
    assert _config(off, "corpus")["corpus_config"]["eval_fraction"] > 0


def test_the_tokenizer_the_corpus_is_measured_with_is_the_one_that_trains(off):
    """Lengths and the assistant-mask check are counted in tokens, so a corpus
    prepared against a different tokenizer is filtered against the wrong
    distribution — and prep needs align's chat template to locate assistant spans
    at all."""
    assert _config(off, "corpus")["corpus_config"]["tokenizer"] == (
        "{{ bindings.tokenizer.binding.path }}"
    )


def test_the_export_asserts_the_tokenizer_it_ships(off):
    """Bound so the export checks the tokenizer it publishes is the one the run
    trained with, rather than whatever happened to land in the checkpoint dir."""
    export = _config(off, "export")["export_config"]
    assert export["expect_tokenizer_from"] == (
        "{{ bindings.expected_tokenizer.binding.path }}"
    )
    assert export["verify"] is True


def test_the_export_tolerates_the_per_rank_rng_files(off):
    """allow_unknown must be TRUE here, and that is forced rather than lax.

    HF Trainer writes one rng_state_<rank>.pth per rank for a distributed run,
    while the ported classifier's PRUNE_KNOWN carries only the single-process
    "rng_state.pth" — it was validated against a 1-GPU checkpoint. So every
    checkpoint this pipeline produces contains files the classifier cannot
    classify, and with allow_unknown false the export refuses after the training
    run has been paid for. export_manifest.json still records exactly what was
    dropped. The fix is one line upstream; until it lands, this is the setting
    that lets the pipeline complete.
    """
    assert _config(off, "export")["export_config"]["allow_unknown"] is True


# ─── Sizing, so the smoke run can actually take a step ─────────────────────────


def _accelerator_count(target_config):
    return int(
        target_config["launcher_config"]["resources"]["accelerators"].split(":")[1]
    )


def test_the_smoke_corpus_is_big_enough_for_at_least_one_optimizer_step(on):
    """Effective batch = per_device x grad_accum x nodes x gpus_per_node. Larger
    than the training split and the trainer has no full batch to form: the run
    ends having optimised nothing and reports success."""
    corpus = _config(on, "corpus")["corpus_config"]
    train_rows = corpus["max_examples"] * (1.0 - corpus["eval_fraction"])
    for target, key in (("train-gold", "gold_config"), ("train-sft", "sft_config")):
        config = _config(on, target)
        block = config[key]
        effective = (
            block["per_device_train_batch_size"]
            * block["gradient_accumulation_steps"]
            * config["compute_config"]["num_nodes"]
            * config["compute_config"]["num_gpus_per_node"]
        )
        assert (
            effective <= train_rows
        ), f"{target}: batch {effective} > {train_rows} rows"


def test_each_training_target_asks_for_the_gpus_it_says_it_has(on):
    """A mismatch between the allocation and the count the trainer is told trains
    at a different effective batch size than the config says. run-sft.sh asserts
    its own; gold's renderer does not, so this is the check."""
    for target in ("train-gold", "train-sft"):
        config = _config(on, target)
        assert config["compute_config"]["num_gpus_per_node"] == _accelerator_count(
            config
        ), target
    # The SFT step ALSO carries the count in its own workload block, and refuses
    # more than one node.
    workload = _config(on, "train-sft")["workload"]
    assert (
        workload["gpus_per_node"]
        == _config(on, "train-sft")["compute_config"]["num_gpus_per_node"]
    )
    assert workload["nodes"] == 1


def test_gold_saves_a_checkpoint_the_export_can_select(on):
    """max_steps bounds the smoke run, so save_steps has to be no larger or the run
    finishes with no checkpoint-* directory and the export has nothing to pick."""
    gold = _config(on, "train-gold")["gold_config"]
    assert gold["max_steps"] > 0
    assert gold["save_steps"] <= gold["max_steps"]


# ─── The response-template escape ──────────────────────────────────────────────


def test_the_response_template_carries_its_line_boundary_as_an_escape(on):
    r"""The trailing newline is where loss masking begins, and it travels as the
    two-character escape \n rather than as a real newline. It has to: gbserver
    fills every config string through Jinja, whose environment is built without
    keep_trailing_newline, so exactly one trailing newline is stripped from each
    value. A real newline therefore never reaches the step, and the run trains
    against a span one token off while reporting success."""
    for target, key in (("train-gold", "gold_config"), ("train-sft", "sft_config")):
        template = _config(on, target)[key]["response_template"]
        assert template.endswith("\\n"), target
        assert not template.endswith("\n"), target


def test_both_trainers_mask_on_the_same_boundary(on):
    """The control and the treatment have to differ in the OBJECTIVE only. A
    different response template is a different labelled span."""
    assert (
        _config(on, "train-sft")["sft_config"]["response_template"]
        == _config(on, "train-gold")["gold_config"]["response_template"]
    )


# ─── Results land somewhere findable ──────────────────────────────────────────


def test_every_composed_path_lands_under_one_per_build_run_directory(off):
    """The pipeline's own root, so a run's align overlay, corpus, evals and BFCL
    results can be found and cleaned up together rather than hunted for across
    per-target workdirs — and ONE DIRECTORY PER BUILD under it.

    The per-build segment is correctness, not tidiness. gbserver refuses to
    register an artifact whose URI another build in the space already registered,
    the emitting target still reports SUCCESS with an empty output list, and a
    later retry or restart then propagates no binding from it: consumers never
    become ready and the build reports SUCCESS having skipped them. Build b5f030cd
    reported SUCCESS with eval-transfer FAILED and eval-bfcl never started.
    """
    root = _params()["WORKDIR_ROOT"]
    run = _params()["RUN_NAME"]
    subdir = _params()["BUILD_SUBDIR"]
    assert (
        "run_metadata.build_id" in subdir
    ), "the per-build segment must vary by build, or artifact URIs collide"
    prefix = f"{root}/{run}/{subdir}/"
    composed = [
        _config(off, "align")["align_config"]["out_dir"],
        _config(off, "corpus")["corpus_config"]["out_dir"],
        _config(off, "export")["export_config"]["dest"],
        _config(off, "eval-transfer-baseline")["eval_config"]["output_dir"],
        _config(off, "eval-transfer")["eval_config"]["output_dir"],
        _config(off, "eval-bfcl")["bfcl_config"]["output_dir"],
    ]
    for path in composed:
        assert path.startswith(prefix), path
    # Distinct, or one target overwrites another's results.
    assert len(set(composed)) == len(composed)


def test_the_evals_and_the_corpus_agree_inside_one_build(off):
    """Both eval targets compose the split's path, and the segment that makes the
    root per-build has to be the SAME template in all three places — a literal
    build id typed into one of them, or a different template, would send the evals
    looking in another build's directory."""
    out_dir = _config(off, "corpus")["corpus_config"]["out_dir"]
    for target in ("eval-transfer-baseline", "eval-transfer"):
        assert _config(off, target)["eval_config"]["corpus"] == f"{out_dir}/eval.jsonl"
