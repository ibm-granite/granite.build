# SPDX-License-Identifier: Apache-2.0

"""
PORTED from the upstream gb-steps-collection-post-training step collection:
  path   steps/distill-sft/test/test_render_sft_config.py
  commit 70c1550a171aa8e09a9ad9047a5bf763c39e8579

Divergences from upstream:
  - upstream's sys.path inserts are removed. conftest.py resolves the step's own src/ and
    the shared distillation package (from the in-repo steps/train/distill/src).
  - the PARITY TESTS read the treatment from this repo: logit-distill's renderer
    (steps/train/distill/logit-distill/skypilot/src/render_gold_config.py) and the vendored
    sft.py. The two in-repo arms are NOT a matched pair the way upstream's were: they differ
    in several optimization defaults, in their resume model and in tracking. The parity
    tests therefore pin what IS shared (the error type, the corpus guard) and the documented
    divergences (see USAGE.md, "Using this step as a control"), so a new divergence, or a
    documented one that quietly goes away, fails here instead of going unnoticed.
"""

"""Tests for student-sft's config validation and rendering.

Two kinds of test live here, and the second kind is the reason this file is not just a copy
of logit-distill's suite with the names changed.

THE FIRST KIND covers the refusals only this step has -- the attention backend, the Liger
SWA patch that is not in this collection, and the four ways to set `precomputed_logits_dir`
and still train no KD term. Each one would otherwise fire inside `sft.py` AFTER the model is
loaded on every rank, which on this cluster means after the allocation is held.

THE SECOND KIND is about being a CONTROL. Every number the distillation arms report is a
comparison against this step, so a way for it to differ silently from them is a way the
comparison stops meaning anything. These are asserted directly against logit-distill's
renderer rather than restated: that the shared guards are literally the same objects, and
that the optimization defaults differ from the treatment's in exactly the documented keys,
so the test fails if another one appears.

Nothing here needs a GPU, CUDA, trl, or transformers.
"""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve()
# steps/train/distill, from steps/train/distill/student-sft/skypilot/test/<this file>.
_DISTILL = _HERE.parents[3]
# The treatment, for the parity tests: logit-distill's renderer and the vendored trainer
# package. A test that compares the control with the treatment has to read the treatment, not
# a transcription of it.
_GOLD_SRC = _DISTILL / "logit-distill" / "skypilot" / "src"
_SFT_PY = _DISTILL / "src" / "gb_steps_post_training" / "distillation" / "sft.py"


def _load_gold_renderer():
    """Load logit-distill's renderer from its file, without putting its src/ on sys.path:
    that directory has a check_weight_residency.py of its own, which would shadow this
    step's copy for every other test in the session."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "render_gold_config", _GOLD_SRC / "render_gold_config.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


render_gold_config = _load_gold_renderer()

from gb_steps_post_training.distillation import render_common  # noqa: E402
from render_sft_config import (  # noqa: E402
    ATTN_BY_ARCH,
    ConfigError,
    build_parser,
    render,
    validate,
)

BASE = [
    "--student-model-path",
    "/s",
    "--corpus-path",
    "/c",
    "--output-dir",
    "/out",
    "--deepspeed-config",
    "/ds.yaml",
    "--out",
    "/rendered.yaml",
]


def parse(*extra):
    return build_parser().parse_args(BASE + list(extra))


# ----------------------------------------------------------------- shapes and schedules


@pytest.mark.parametrize(
    "flag,value,match",
    [
        ("--max-length", "0", "max_length"),
        ("--per-device-train-batch-size", "0", "per_device_train_batch_size"),
        ("--gradient-accumulation-steps", "0", "gradient_accumulation_steps"),
        ("--num-train-epochs", "0", "num_train_epochs"),
        ("--save-steps", "0", "save_steps"),
    ],
)
def test_non_positive_shapes_are_refused(flag, value, match):
    with pytest.raises(ConfigError, match=match):
        validate(parse(flag, value), check_paths=False)


def test_save_steps_refusal_names_preemption():
    """The reason this one is not merely a sanity bound: a run that never checkpoints on
    preemptable capacity loses everything on the first preemption."""
    with pytest.raises(ConfigError, match="preempt"):
        validate(parse("--save-steps", "0"), check_paths=False)


def test_fractional_epochs_are_accepted():
    """num_train_epochs is a float on purpose -- the smoke configs train 0.02 of an epoch."""
    validate(parse("--num-train-epochs", "0.02"), check_paths=False)


@pytest.mark.parametrize("value", ["0", "-2", "-100"])
def test_bad_max_dataset_size_is_refused(value):
    with pytest.raises(ConfigError, match="max_dataset_size"):
        validate(parse("--max-dataset-size", value), check_paths=False)


@pytest.mark.parametrize("value", ["-1", "1", "3000"])
def test_good_max_dataset_size_is_accepted(value):
    validate(parse("--max-dataset-size", value), check_paths=False)


def test_non_numeric_learning_rate_is_refused():
    with pytest.raises(ConfigError, match="learning_rate"):
        validate(parse("--learning-rate", "1e-6mistyped"), check_paths=False)


def test_quoted_scientific_learning_rate_is_accepted():
    validate(parse("--learning-rate", "1e-6"), check_paths=False)


@pytest.mark.parametrize("gpn", ["0", "9", "-1"])
def test_gpus_per_node_bounds_come_from_render_common(gpn):
    with pytest.raises(ConfigError):
        validate(parse("--gpus-per-node", gpn), check_paths=False)


# ------------------------------------------------- the Liger SWA patch is not in this tree


def test_liger_swiglu_mlp_is_refused():
    """sft.py imports _liger_granite_swa_patch inside __main__, so without this refusal the
    ImportError lands minutes into an allocated run."""
    with pytest.raises(ConfigError, match="_liger_granite_swa_patch"):
        validate(parse("--use-liger-swiglu-mlp"), check_paths=False)


def test_liger_swiglu_mlp_off_is_accepted():
    validate(parse("--no-use-liger-swiglu-mlp"), check_paths=False)


@pytest.mark.parametrize("flag", ["--no-use-liger-swiglu-mlp", None])
def test_use_liger_swiglu_mlp_is_never_rendered(flag):
    """Emitting `false` for a key with exactly one legal value would suggest it has two."""
    args = parse(*([flag] if flag else []))
    assert "use_liger_swiglu_mlp" not in render(args)


def test_use_liger_memory_opt_is_rendered_both_ways():
    """The other Liger flag patches nothing model-internal, so it really is a choice -- and
    it must round-trip as a BOOLEAN, not the string "False" (LSF job 1138651)."""
    assert render(parse("--use-liger-memory-opt"))["use_liger_memory_opt"] is True
    assert render(parse("--no-use-liger-memory-opt"))["use_liger_memory_opt"] is False


# ----------------------------------------------------------------- KD mode coherence
#
# Setting precomputed_logits_dir is what turns this step from the control into forward-KL
# distillation. Every test below is a config that READS as one arm and would TRAIN as the
# other -- the class of failure that makes a comparison meaningless without ever failing.


def test_control_arm_renders_no_kd_keys():
    cfg = render(parse())
    for key in (
        "precomputed_logits_dir",
        "kd_top_k",
        "kd_weight",
        "ce_weight",
        "kd_temperature",
    ):
        assert key not in cfg


def test_kd_arm_renders_all_five_keys_together():
    cfg = render(
        parse(
            "--precomputed-logits-dir",
            "/logits",
            "--kd-weight",
            "0.9",
            "--ce-weight",
            "0.1",
        )
    )
    assert cfg["precomputed_logits_dir"] == "/logits"
    assert cfg["kd_top_k"] == 256
    assert cfg["kd_weight"] == 0.9
    assert cfg["ce_weight"] == 0.1
    assert cfg["kd_temperature"] == 1.0


def test_kd_weight_zero_with_a_logits_dir_is_refused():
    """The teacher's logits would be loaded, mmapped and collated on every step and then
    multiplied by zero: a run that pays for distillation and performs plain SFT."""
    with pytest.raises(ConfigError, match="plain SFT"):
        validate(
            parse(
                "--precomputed-logits-dir",
                "/logits",
                "--kd-weight",
                "0.0",
                "--ce-weight",
                "1.0",
            ),
            check_paths=False,
        )


def test_both_weights_zero_is_refused_as_a_zero_loss():
    with pytest.raises(ConfigError, match="identically zero"):
        validate(
            parse(
                "--precomputed-logits-dir",
                "/logits",
                "--kd-weight",
                "0.0",
                "--ce-weight",
                "0.0",
            ),
            check_paths=False,
        )


@pytest.mark.parametrize(
    "flag,value,match",
    [
        ("--kd-top-k", "0", "kd_top_k"),
        ("--kd-temperature", "0.0", "kd_temperature"),
        ("--kd-temperature", "-1.0", "kd_temperature"),
        ("--kd-weight", "-0.5", "must be >= 0"),
        ("--ce-weight", "-0.5", "must be >= 0"),
    ],
)
def test_incoherent_kd_values_are_refused(flag, value, match):
    with pytest.raises(ConfigError, match=match):
        validate(
            parse("--precomputed-logits-dir", "/logits", flag, value), check_paths=False
        )


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--kd-top-k", "64"),
        ("--kd-weight", "0.5"),
        ("--ce-weight", "0.5"),
        ("--kd-temperature", "2.0"),
    ],
)
def test_a_changed_kd_key_without_a_logits_dir_is_refused(flag, value):
    """The inverse failure, and the quieter one: the run trains plain CE while its config
    says top-256 KD at temperature 2, and nothing anywhere reports the discrepancy."""
    with pytest.raises(ConfigError, match="silently ignored"):
        validate(parse(flag, value), check_paths=False)


def test_default_kd_keys_without_a_logits_dir_are_accepted():
    """Left at their template defaults the kd_* keys are inert, which is what the control
    arm looks like -- the refusal above must fire on INTENT, not on presence."""
    validate(parse(), check_paths=False)


def test_the_refused_defaults_are_the_parsers_own():
    """Pins the four default values the 'changed with no logits dir' check compares against.
    If a default moves in build_parser and not in validate, that check starts refusing the
    control arm itself -- or stops refusing a real mismatch."""
    a = parse()
    assert (a.kd_top_k, a.kd_weight, a.ce_weight, a.kd_temperature) == (
        256,
        1.0,
        0.0,
        1.0,
    )


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_whitespace_logits_dir_is_the_control_arm(blank):
    """run-sft.sh passes --precomputed-logits-dir "$PRECOMPUTED_LOGITS_DIR" unconditionally,
    so an unset step key arrives as an empty or blank string rather than not arriving.
    """
    args = parse("--precomputed-logits-dir", blank)
    validate(args, check_paths=False)
    assert "precomputed_logits_dir" not in render(args)


# ----------------------------------------------------------------- rendering


def test_learning_rate_is_emitted_as_a_float():
    """Bare 1e-6 is a STRING in YAML 1.1, so a round-trip through yaml is part of the test:
    only 1.0e-6 parses back as a float, and a string learning rate reaches
    TrainingArguments as one."""
    cfg = render(parse("--learning-rate", "1e-6"))
    assert isinstance(cfg["learning_rate"], float)
    assert isinstance(yaml.safe_load(yaml.safe_dump(cfg))["learning_rate"], float)


def test_attn_implementation_is_always_emitted():
    """Unlike the gold path, where nothing reads it. verify_optimization_stack compares
    model.config._attn_implementation against one literal per architecture and aborts every
    rank when it differs -- and trl's ModelConfig defaults the key to None."""
    assert render(parse())["attn_implementation"] == "flash_attention_2"


def test_student_dtype_is_emitted_as_bfloat16():
    """trl 1.x's ModelConfig.dtype defaults to "float32" and create_model_from_path applies
    it, so a config that leaves dtype out trains the student in fp32. trl 0.26.2 loaded it
    in the checkpoint's bfloat16."""
    cfg = render(parse())
    assert cfg["dtype"] == "bfloat16"
    assert "torch_dtype" not in cfg


def test_loss_switches_are_emitted_explicitly():
    """completion_only_loss=True, assistant_only_loss=False reproduce the trl 0.26.2 labels
    (sft.py builds them from the masks with trl's rule). assistant_only_loss=True would move
    the non-KD mask to the generation markers, one token per turn more than the
    response-template scan."""
    cfg = render(parse())
    assert cfg["completion_only_loss"] is True
    assert cfg["assistant_only_loss"] is False
    kd = render(parse("--precomputed-logits-dir", "/logits"))
    assert kd["completion_only_loss"] is True
    assert kd["assistant_only_loss"] is False


def test_response_template_survives_its_trailing_newline():
    """The ChatML opener is "<|im_start|>assistant\\n". The newline is part of the literal
    that is searched for in the rendered ids; losing it makes the fallback boundary miss.
    """
    cfg = render(parse())
    assert cfg["response_template"] == "<|im_start|>assistant\n"
    assert (
        yaml.safe_load(yaml.safe_dump(cfg))["response_template"]
        == "<|im_start|>assistant\n"
    )


def test_student_and_corpus_land_on_the_keys_the_trainer_reads():
    cfg = render(parse())
    assert cfg["model_name_or_path"] == "/s"
    assert cfg["dataset_name"] == "/c"


# ----------------------------------------------------------------- tracking keys


def test_tracking_keys_are_omitted_when_empty():
    cfg = render(parse())
    for key in (
        "clearml_project",
        "clearml_run_name",
        "wandb_entity",
        "wandb_project",
        "wandb_run_name",
    ):
        assert key not in cfg


def test_tracking_keys_are_emitted_stripped_when_set():
    cfg = render(parse("--clearml-project", " distill ", "--wandb-project", "kd"))
    assert cfg["clearml_project"] == "distill"
    assert cfg["wandb_project"] == "kd"
    assert "wandb_entity" not in cfg


def test_no_run_name_is_defaulted_here():
    """tracking.auto_run_name() composes it, from the student and corpus, because it knows
    `sft-config.rendered.yaml` is a generic stem. A name invented here would be shaped
    differently from the arms this step is compared against."""
    cfg = render(parse("--clearml-project", "p"))
    assert "clearml_run_name" not in cfg


# ----------------------------------------------------------------- extra_config_yaml


def test_extra_yaml_can_add_a_key():
    cfg = render(parse("--extra-config-yaml", "bf16: true"))
    assert cfg["bf16"] is True


def test_extra_yaml_cannot_override_a_contract_key():
    with pytest.raises(ConfigError, match="collides"):
        render(parse("--extra-config-yaml", "learning_rate: 1.0e-4"))


def test_extra_yaml_cannot_override_the_attention_backend():
    """The key most tempting to set this way, and the one whose silent override aborts every
    rank after the model loads."""
    with pytest.raises(ConfigError, match="collides"):
        render(parse("--extra-config-yaml", "attn_implementation: sdpa"))


def test_extra_yaml_must_be_a_mapping():
    with pytest.raises(ConfigError, match="mapping"):
        render(parse("--extra-config-yaml", "- a\n- b"))


def test_empty_extra_yaml_is_a_no_op():
    assert render(parse("--extra-config-yaml", "   ")) == render(parse())


# ----------------------------------------------------------------- path-dependent checks


def _student(
    tmp_path, *, markers=False, model_type=None, template=True, name="student"
):
    d = tmp_path / name
    d.mkdir()
    (d / "tokenizer.json").write_text("{}")
    if template:
        body = "{% generation %}{{ x }}{% endgeneration %}" if markers else "{{ x }}"
        (d / "chat_template.jinja").write_text(body)
    if model_type is not None:
        (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    return d


def _corpus(tmp_path, identity=None):
    d = tmp_path / "corpus"
    d.mkdir()
    (d / "data.jsonl").write_text("")
    if identity is not None:
        (d / "manifest.json").write_text(json.dumps({"tokenizer_identity": identity}))
    return d


def _full_args(tmp_path, student, corpus, *extra):
    ds = tmp_path / "ds.yaml"
    ds.write_text("{}")
    return build_parser().parse_args(
        [
            "--student-model-path",
            str(student),
            "--corpus-path",
            str(corpus),
            "--output-dir",
            str(tmp_path / "out"),
            "--deepspeed-config",
            str(ds),
            "--out",
            str(tmp_path / "r.yaml"),
        ]
        + list(extra)
    )


def test_a_complete_config_validates_against_real_paths(tmp_path):
    validate(_full_args(tmp_path, _student(tmp_path, markers=True), _corpus(tmp_path)))


def test_missing_student_dir_is_refused(tmp_path):
    args = _full_args(tmp_path, tmp_path / "absent", _corpus(tmp_path))
    with pytest.raises(ConfigError, match="student_model_path"):
        validate(args)


def test_missing_deepspeed_config_is_refused(tmp_path):
    args = _full_args(tmp_path, _student(tmp_path, markers=True), _corpus(tmp_path))
    args.deepspeed_config = str(tmp_path / "absent.yaml")
    with pytest.raises(ConfigError, match="deepspeed_config"):
        validate(args)


def test_missing_chat_template_is_refused(tmp_path):
    args = _full_args(tmp_path, _student(tmp_path, template=False), _corpus(tmp_path))
    with pytest.raises(ConfigError, match="chat_template.jinja"):
        validate(args)


def test_a_logits_dir_that_does_not_exist_is_refused(tmp_path):
    """The KD arm's own input. Missing, sft.py's collator fails per-batch, deep into a run."""
    args = _full_args(
        tmp_path,
        _student(tmp_path, markers=True),
        _corpus(tmp_path),
        "--precomputed-logits-dir",
        str(tmp_path / "absent"),
    )
    with pytest.raises(ConfigError, match="precomputed_logits_dir"):
        validate(args)


def test_a_logits_dir_that_exists_is_accepted(tmp_path):
    logits = tmp_path / "logits"
    logits.mkdir()
    validate(
        _full_args(
            tmp_path,
            _student(tmp_path, markers=True),
            _corpus(tmp_path),
            "--precomputed-logits-dir",
            str(logits),
        )
    )


def test_mismatched_tokenizer_identity_is_refused(tmp_path):
    """Shared with logit-distill, and it has to be: a control trained on a corpus
    segmented by a different tokenizer is not a control."""
    s = _student(tmp_path, markers=True)
    (s / "tokenizer_identity.json").write_text(
        json.dumps({"tokenizer_identity": "granite-4.1-3b-retagged-v2"})
    )
    c = _corpus(tmp_path, identity="granite-4.2-30b")
    with pytest.raises(ConfigError, match="tokenizer-specific"):
        validate(_full_args(tmp_path, s, c))


# ------------------------------------------------- the completion boundary


def test_no_markers_and_no_response_template_is_refused(tmp_path):
    """Both mechanisms absent means the assistant mask comes out all zeros and sft.py raises
    -- after the corpus is tokenized. The refusal names the patch that installs markers.
    """
    args = _full_args(
        tmp_path,
        _student(tmp_path, markers=False),
        _corpus(tmp_path),
        "--response-template",
        "",
    )
    with pytest.raises(ConfigError, match="generation"):
        validate(args)


def test_no_markers_but_a_response_template_is_accepted(tmp_path):
    """The fallback. It CAN match a ChatML render -- job 1135379 measured 38.7% of tokens
    labelled over 3000 real rows, zero rows without labels."""
    validate(_full_args(tmp_path, _student(tmp_path, markers=False), _corpus(tmp_path)))


def test_markers_alone_are_enough(tmp_path):
    """Markers are the primary mechanism, so an empty response_template is fine with them."""
    validate(
        _full_args(
            tmp_path,
            _student(tmp_path, markers=True),
            _corpus(tmp_path),
            "--response-template",
            "",
        )
    )


# ------------------------------------------------- the attention backend


def test_default_backend_matches_a_granitemoehybrid_student(tmp_path):
    validate(
        _full_args(
            tmp_path,
            _student(tmp_path, markers=True, model_type="granitemoehybrid"),
            _corpus(tmp_path),
        )
    )


def test_a_wrong_backend_is_refused_before_the_model_loads(tmp_path):
    args = _full_args(
        tmp_path,
        _student(tmp_path, markers=True, model_type="granitemoehybrid"),
        _corpus(tmp_path),
        "--attn-implementation",
        "sdpa",
    )
    with pytest.raises(ConfigError, match="verify_optimization_stack"):
        validate(args)


def test_an_unreadable_config_json_falls_back_to_the_default_backend(tmp_path):
    """Deliberately tolerant: the input-existence checks already speak about the directory,
    and a second error message for the same cause makes the first harder to find."""
    s = _student(tmp_path, markers=True)
    (s / "config.json").write_text("{ not json")
    validate(_full_args(tmp_path, s, _corpus(tmp_path)))


def test_a_granite_swa_student_is_refused(tmp_path):
    """Its model code was classified DELETE during triage, so transformers cannot
    instantiate it at all -- and the FA3 preamble the architecture needs is gated off.
    """
    args = _full_args(
        tmp_path,
        _student(tmp_path, markers=True, model_type="granite_swa"),
        _corpus(tmp_path),
        "--attn-implementation",
        ATTN_BY_ARCH["granite_swa"],
    )
    with pytest.raises(ConfigError, match="granite_swa"):
        validate(args)


def test_the_backend_table_has_a_catch_all():
    """A model_type nobody has seen must resolve to something, not KeyError inside a
    preflight whose whole job is to produce a legible message."""
    assert "*" in ATTN_BY_ARCH
    assert ATTN_BY_ARCH["*"] == "flash_attention_2"


# ----------------------------------------------------------------- resume modes
#
# The rules are render_common's and logit-distill's suite covers their behaviour. What
# is tested here is that THIS step reaches them, and reaches them named as itself: the
# message has to say sft.py, because an operator reading "gold.py resumes on presence" about
# a control run will go looking in the wrong file.


def _out_with_checkpoint(tmp_path):
    out = tmp_path / "out"
    (out / "checkpoint-50").mkdir(parents=True)
    return out


def test_resume_defaults_to_auto():
    assert parse().resume == "auto"


def test_resume_rejects_unknown_mode():
    with pytest.raises(SystemExit):
        parse("--resume", "maybe")


def test_resume_never_refuses_an_existing_checkpoint(tmp_path):
    args = _full_args(
        tmp_path,
        _student(tmp_path, markers=True),
        _corpus(tmp_path),
        "--resume",
        "never",
    )
    args.output_dir = str(_out_with_checkpoint(tmp_path))
    with pytest.raises(ConfigError, match="sft.py"):
        validate(args)


def test_resume_never_does_not_delete_the_checkpoint(tmp_path):
    """A preflight that fixed the problem it reports would silently discard training."""
    args = _full_args(
        tmp_path,
        _student(tmp_path, markers=True),
        _corpus(tmp_path),
        "--resume",
        "never",
    )
    out = _out_with_checkpoint(tmp_path)
    args.output_dir = str(out)
    with pytest.raises(ConfigError):
        validate(args)
    assert (out / "checkpoint-50").is_dir()


def test_resume_require_refuses_a_missing_checkpoint(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    args = _full_args(
        tmp_path,
        _student(tmp_path, markers=True),
        _corpus(tmp_path),
        "--resume",
        "require",
    )
    args.output_dir = str(out)
    with pytest.raises(ConfigError):
        validate(args)


def test_resume_require_accepts_an_existing_checkpoint(tmp_path):
    args = _full_args(
        tmp_path,
        _student(tmp_path, markers=True),
        _corpus(tmp_path),
        "--resume",
        "require",
    )
    args.output_dir = str(_out_with_checkpoint(tmp_path))
    validate(args)


def test_resume_is_not_a_rendered_key():
    """It is a preflight decision, not a trainer setting -- sft.py has no such field, and
    TrlParser runs with fail_with_unknown_args, so emitting it would be a hard parse error.
    """
    assert "resume" not in render(parse("--resume", "require"))


# ------------------------------------------------- being a control: parity with the treatment
#
# These are the tests that make the README's claims checkable. Both properties were asserted
# by prose alone while this step was a scaffold.


# The guards that must be ONE implementation, not two agreeing ones. Every entry is a rule
# whose divergence would go unnoticed: a control that resumes by a slightly different rule, or
# accepts a corpus the treatment refuses, produces numbers that still look comparable.
SHARED_GUARDS = (
    "validate_resume",
    "validate_topology",
    "validate_student_against_corpus",
)

_RENDERERS = {
    "render_sft_config": _HERE.parent.parent / "src" / "render_sft_config.py",
    "render_gold_config": _GOLD_SRC / "render_gold_config.py",
}

# Which shared guards each renderer calls. logit-distill calls only the corpus guard:
#   - validate_resume: logit-distill has no resume=auto|never|require. Each run starts in a
#     fresh workdir and resumes only from an explicit resume_from_checkpoint_dir.
#   - validate_topology: logit-distill takes no gpus_per_node, and its node rules depend on
#     the on-policy layout (total_nodes against vllm_num_servers), checked in its own
#     build_config.
_CALLED_GUARDS = {
    "render_sft_config": SHARED_GUARDS,
    "render_gold_config": ("validate_student_against_corpus",),
}


def _defined_names(src):
    return {
        n.name
        for n in ast.walk(ast.parse(src))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def test_both_renderers_use_the_one_config_error():
    """`except ConfigError` must catch what the shared guards raise: this step aliases
    render_common.ConfigError, and logit-distill's renderer catches that same class from the
    corpus guard. Neither may define a same-named class of its own."""
    assert ConfigError is render_common.ConfigError
    for module, path in _RENDERERS.items():
        src = path.read_text()
        assert "ConfigError" not in _defined_names(src), f"{module} defines ConfigError"
    gold_src = _RENDERERS["render_gold_config"].read_text()
    assert "render_common.ConfigError" in gold_src


@pytest.mark.parametrize("name", SHARED_GUARDS)
def test_the_shared_guards_exist_in_one_place(name):
    assert callable(getattr(render_common, name))


@pytest.mark.parametrize("module,path", sorted(_RENDERERS.items()))
@pytest.mark.parametrize("name", SHARED_GUARDS)
def test_no_renderer_defines_a_copy_of_a_shared_guard(name, module, path):
    """Read as SOURCE, not as attributes: the failure this catches is someone answering a
    divergence by pasting a local copy of the rule into one of the two files, which no
    identity check on render_common's own attributes can see.

    A local `def validate_resume` would shadow the import, both steps would pass their own
    tests, and the two arms would differ on exactly the rule this asserts they share."""
    assert name not in _defined_names(
        path.read_text()
    ), f"{module} defines its own {name}; it must call render_common's"


@pytest.mark.parametrize(
    "module,name",
    [(m, g) for m, guards in sorted(_CALLED_GUARDS.items()) for g in guards],
)
def test_each_renderer_calls_its_shared_guards(module, name):
    src = _RENDERERS[module].read_text()
    assert f"render_common.{name}(" in src, f"{module} never calls render_common.{name}"


def test_logit_distill_still_skips_the_guards_it_has_no_input_for():
    """Pins the two guards _CALLED_GUARDS says logit-distill does not call. If it starts
    calling one, move it into its entry above, so the list keeps describing the code."""
    src = _RENDERERS["render_gold_config"].read_text()
    for name in set(SHARED_GUARDS) - set(_CALLED_GUARDS["render_gold_config"]):
        assert f"render_common.{name}(" not in src


# ---------------------------------------------------------------------------------------------
# The KD mask mechanism, pinned as SOURCE for the same reason as the shared guards above: this
# is a property of sft.py, which cannot be imported here (it pulls torch at module scope), and
# the regression it guards against is a one-word edit that no run fails on loudly.
#
# `assistant_only_loss` decides what the LOSS covers. KD needs the same mask for something
# else entirely -- to align each stored teacher row with the token it was computed for -- so
# gating the request on that flag does not disable the mask, it silently switches the MECHANISM
# that produces it: apply_chat_template's `{% generation %}` markers vs. the response-template
# scan below them. Those two disagree by exactly one token per assistant turn, because the
# generation block covers the newline after <|im_end|> and the scan stops at it. The producer
# uses the markers, so every KD run died on its first row with "515 != 514" (LSF job 1201653).
#
# Reverting to `=assistant_only_loss` would not fail these tests any other way: the column is
# still present (the fallback fills it), the collator still runs, the shapes still match. It
# fails one row deep into every KD run, and only when a template's generation block and its
# response template happen to disagree -- which is a property of the template, not of us.
# _SFT_PY (defined at the top) is the vendored trainer this step runs.


def test_kd_requests_the_assistant_mask_regardless_of_assistant_only_loss():
    # Compared with all whitespace removed: black wraps the vendored call sites over three
    # lines, and an exact-text match would then count none of them.
    src = "".join(_SFT_PY.read_text().split())
    assert "return_assistant_tokens_mask=assistant_only_loss," not in src, (
        "a call site gates the mask on assistant_only_loss again; in KD mode that switches "
        "the mask mechanism to the response-template scan and desynchronises it from the "
        "producer by one token per assistant turn"
    )
    n = src.count("return_assistant_tokens_mask=(assistant_only_lossorkd_mode)")
    assert n == 2, (
        f"expected both branches of _tokenize_one_row (prompt/completion and language "
        f"modeling) to request the mask in KD mode, found {n}"
    )


def test_the_two_tokenize_branches_are_the_only_call_sites():
    """If a third branch appears, it needs the same treatment -- and this is the test that
    says so, rather than a KD run failing on row 0 months later."""
    src = _SFT_PY.read_text()
    assert src.count("return_assistant_tokens_mask=") == 2


def test_the_count_mismatch_error_names_the_mechanism_split():
    """A changed template and a mechanism split raise the SAME error with the same numbers.
    An operator who is only told "the chat template changed" will go and diff templates that
    did not change."""
    src = _SFT_PY.read_text()
    i = src.index("KD: assistant token count from chat template")
    msg = src[i : i + 1200]
    assert "MECHANISM" in msg
    assert "response-template scan" in msg


def _parser_defaults(parser):
    return {a.dest: a.default for a in parser._actions if a.dest != "help"}


_GOLD_REQUIRED = [
    "--output",
    "/o",
    "--total-nodes",
    "1",
    "--model-name-or-path",
    "/m",
    "--teacher-model-name-or-path",
    "/t",
    "--dataset-name",
    "/d",
]


def _num(value):
    """Compare numbers as numbers: this step's parser keeps learning_rate as the string
    "1e-6", logit-distill's parses it to a float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _shared_defaults():
    """(logit-distill, student-sft) parsed defaults for the keys both renderers take."""
    gold = vars(render_gold_config._parse_args(_GOLD_REQUIRED))
    sft = vars(build_parser().parse_args(BASE))
    return {k: (_num(gold[k]), _num(sft[k])) for k in sorted(set(gold) & set(sft))}


# The defaults the two steps differ on, as (logit-distill, student-sft). Documented in
# USAGE.md ("Using this step as a control"): a recipe that runs this step as a control sets
# these to logit-distill's values. The response template differs by its trailing newline,
# which USAGE.md explains in its own section.
DOCUMENTED_DIVERGENCES = {
    "gradient_accumulation_steps": (6.0, 8.0),
    "learning_rate": (1e-05, 1e-06),
    "max_length": (16384.0, 4096.0),
    "response_template": ("<|im_start|>assistant", "<|im_start|>assistant\n"),
    "save_steps": (500.0, 100.0),
}


def test_optimization_defaults_differ_only_where_documented():
    """A control that trains at a different learning rate, batch size or sequence length
    than the treatment is not a control, and nothing about either run would fail. The two
    steps' defaults are not matched today, so this pins the documented differences: a new
    one, or a changed value, fails here until USAGE.md and the table above say so."""
    differ = {k: v for k, v in _shared_defaults().items() if v[0] != v[1]}
    assert differ == DOCUMENTED_DIVERGENCES


def test_the_keys_both_steps_take_include_the_batch_shape():
    """Guards the test above from going vacuous: if a renaming made the two parsers stop
    sharing these keys, the comparison would silently cover less."""
    assert {
        "gradient_accumulation_steps",
        "learning_rate",
        "max_length",
        "num_train_epochs",
        "per_device_train_batch_size",
    } <= set(_shared_defaults())


def test_this_steps_tracking_keys_default_to_off():
    """logit-distill renders no tracking keys, so there is no surface to compare. This step's
    five default to empty, so a run tracks nothing unless the recipe says where."""
    keys = {
        "clearml_project",
        "clearml_run_name",
        "wandb_entity",
        "wandb_project",
        "wandb_run_name",
    }
    sft = _parser_defaults(build_parser())
    assert {k: sft[k] for k in keys} == {k: "" for k in keys}


def test_no_teacher_flag_exists_on_this_step():
    """The structural half of "share the code, not the step": a control you can point at a
    teacher is a control you can silently mis-configure into the treatment. The teacher
    enters here, if at all, as a directory of PRECOMPUTED logits."""
    sft = _parser_defaults(build_parser())
    assert not [k for k in sft if "teacher" in k]
    assert "lmbda" not in sft


# ----------------------------------------------------------------- the CLI contract


def test_main_writes_the_rendered_file(tmp_path):
    import render_sft_config

    out = tmp_path / "nested" / "r.yaml"
    rc = render_sft_config.main(
        [
            "--student-model-path",
            "/s",
            "--corpus-path",
            "/c",
            "--output-dir",
            str(tmp_path / "out"),
            "--deepspeed-config",
            "/ds.yaml",
            "--out",
            str(out),
            "--no-check-paths",
        ]
    )
    assert rc == 0
    cfg = yaml.safe_load(out.read_text())
    assert cfg["model_name_or_path"] == "/s"
    assert cfg["attn_implementation"] == "flash_attention_2"


def test_main_returns_2_and_writes_nothing_on_a_refusal(tmp_path):
    """run-sft.sh propagates this with `|| exit $?`, so the exit code is the contract -- and
    a half-written config file left behind by a refused render is a file some later run picks
    up."""
    import render_sft_config

    out = tmp_path / "r.yaml"
    rc = render_sft_config.main(
        [
            "--student-model-path",
            "/s",
            "--corpus-path",
            "/c",
            "--output-dir",
            str(tmp_path / "out"),
            "--deepspeed-config",
            "/ds.yaml",
            "--out",
            str(out),
            "--no-check-paths",
            "--use-liger-swiglu-mlp",
        ]
    )
    assert rc == 2
    assert not out.exists()


# ------------------------------------------- the template's defaults are the renderer's
#
# The parity tests above compare the two RENDERERS' argparse defaults, but a recipe never
# reaches those: it gets step-template.yaml's values, which the run block passes as flags to
# run-sft.sh, which forwards them to render_sft_config.py. These tests read that chain from
# the template and the script rather than restating it, then check that the template's
# defaults parse to the renderer's own -- so the table in USAGE.md ("Using this step as a
# control"), which is pinned against the renderers above, describes what recipes run.

_STEP_DIR = _HERE.parents[1]
_STEP_TEMPLATE = _STEP_DIR / "step-template.yaml"
_RUN_SFT_SH = _STEP_DIR / "src" / "run-sft.sh"

# `--flag {{ config.<section>.<key> }}`, optionally quoted, as the run block writes them.
_CONFIG_FLAG = re.compile(
    r"""(--[a-z][a-z0-9-]*)\s+(["']?)(\{\{\s*config\.(\w+)\.(\w+)[^}]*\}\})\2(?=\s|$)"""
)
# `{% if config.<section>.<key> %}--flag{% else %}--no-flag{% endif %}`, the boolean pairs.
_FLAG_PAIR = re.compile(
    r"\{%\s*if\s+config\.(\w+)\.(\w+)\s*%\}(--[a-z][a-z0-9-]*)"
    r"\{%\s*else\s*%\}(--no-[a-z][a-z0-9-]*)\{%\s*endif\s*%\}"
)

# Renderer flags with a default that the template does not pass, and why. A flag the template
# stops passing fails test_every_defaulted_renderer_flag_is_accounted_for until it is listed.
_RENDERER_DEFAULTS_THE_TEMPLATE_DOES_NOT_PASS = {
    # Not a recipe knob: the renderer picks the backend per architecture (ATTN_BY_ARCH).
    "--attn-implementation",
    # A rendering-outside-the-runtime switch for tests; the step always checks paths.
    "--no-check-paths",
}


def _jinja(source, step):
    """Render a template fragment the way gbserver fills a step (StrictUndefined)."""
    from jinja2 import StrictUndefined  # noqa: PLC0415
    from jinja2.sandbox import SandboxedEnvironment  # noqa: PLC0415

    env = SandboxedEnvironment(undefined=StrictUndefined)
    return env.from_string(source).render(config=step["config"])


def _run_sft_command(run_block):
    """The run block's one (backslash-continued) command that invokes run-sft.sh."""
    joined = re.sub(r"\\\n", " ", run_block)
    [command] = [line for line in joined.splitlines() if "/run-sft.sh" in line]
    return command


def _response_template_as_passed(run_block, step):
    """`--response-template "$RESPONSE_TEMPLATE"`, set by the run block's own decode lines.

    Executed rather than restated: the decode is `printf '%b.'` plus a sentinel strip, and
    that is exactly the part that is easy to get wrong (see TestResponseTemplateNewline in
    test_step_template.py).
    """
    lines = [
        line.strip()
        for line in run_block.splitlines()
        if re.match(r"\s*(RT_RAW|RESPONSE_TEMPLATE)=", line)
    ]
    assert len(lines) == 3, lines
    script = "\n".join(_jinja(line, step) for line in lines)
    result = subprocess.run(
        ["bash", "-c", script + '\nprintf %s "$RESPONSE_TEMPLATE"'],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def _template_flags():
    """[(template key, argv tokens)] for each config value the run block passes run-sft.sh."""
    step = yaml.safe_load(_STEP_TEMPLATE.read_text())
    run_block = step["environment_configs"]["Skypilot"]["launchers"]["train"]["config"][
        "run"
    ]
    command = _run_sft_command(run_block)
    flags = []
    for match in _CONFIG_FLAG.finditer(command):
        flag, quote, expr, section, key = match.groups()
        value = _jinja(expr, step)
        # An unquoted value reaches the script as one word only if it is one word.
        assert quote or (value and not value.split()[1:]), (flag, value)
        flags.append((f"{section}.{key}", [flag, value]))
    for match in _FLAG_PAIR.finditer(command):
        section, key, on, off = match.groups()
        flags.append(
            (f"{section}.{key}", [on if step["config"][section][key] else off])
        )
    # Every config reference in the command is one of the forms above, so none is skipped.
    assert len(re.findall(r"\bconfig\.\w+\.\w+", command)) == len(flags), command
    # Passed under its own flag, but through a shell variable the run block decodes.
    assert '--response-template "$RESPONSE_TEMPLATE"' in command
    flags.append(
        (
            "sft_config.response_template",
            ["--response-template", _response_template_as_passed(run_block, step)],
        )
    )
    return flags


def _accepted_by_the_renderer(tokens):
    return tokens[0] in build_parser()._option_string_actions


def _renderer_argv_from_run_sft(tmp_path, argv):
    """What run-sft.sh hands render_sft_config.py, captured by a stub PYBIN."""
    captured = tmp_path / "argv"
    stub = tmp_path / "pybin"
    stub.write_text(
        f"#!/bin/sh\nshift\nprintf '%s\\0' \"$@\" > '{captured}'\nexit 97\n"
    )
    stub.chmod(0o755)
    (tmp_path / "lib").mkdir()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYBIN": str(stub),
        # Neither is read before the render; set so nothing resolves outside tmp_path.
        "SFT_SRC": str(tmp_path / "pkg" / "distillation"),
        "LIB_DIR": str(tmp_path / "lib"),
        "SFT_PREPROCESS_CACHE_ROOT": str(tmp_path / "cache"),
    }
    required = [
        "--student-model-path",
        "/s",
        "--corpus-path",
        "/c",
        "--output-dir",
        str(tmp_path / "out"),
        "--deepspeed-config",
        "/ds.yaml",
    ]
    result = subprocess.run(
        ["bash", str(_RUN_SFT_SH), *required, *argv],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 97, result.stdout + result.stderr
    return captured.read_text().split("\0")[:-1]


def test_run_sft_forwards_each_renderer_flag_the_template_passes_unchanged(tmp_path):
    """The template does not call the renderer; run-sft.sh does, from its own parsed
    variables. So the template's flag reaches the renderer only if the script forwards it
    under the same name with the same value -- checked by running the script."""
    passed = [tokens for _, tokens in _template_flags()]
    forwarded = _renderer_argv_from_run_sft(
        tmp_path, [token for tokens in passed for token in tokens]
    )
    for tokens in filter(_accepted_by_the_renderer, passed):
        i = forwarded.index(tokens[0])
        assert forwarded[i : i + len(tokens)] == tokens, tokens


def test_the_template_defaults_are_the_renderers_defaults():
    """Every config value the template passes parses, in the renderer, to the renderer's own
    default. Numbers compare as numbers: the parser keeps learning_rate as the string "1e-6".
    """
    defaults = vars(parse())
    differ = {}
    for key, tokens in _template_flags():
        if not _accepted_by_the_renderer(tokens):
            continue
        dest = build_parser()._option_string_actions[tokens[0]].dest
        got = vars(parse(*tokens))[dest]
        if _num(got) != _num(defaults[dest]):
            differ[key] = (got, defaults[dest])
    assert differ == {}


def test_every_defaulted_renderer_flag_is_accounted_for():
    """Guards the test above from going vacuous: a renderer flag with a default is either
    passed by the template or listed, with its reason, as not passed."""
    passed = {tokens[0] for _, tokens in _template_flags()}
    for action in build_parser()._actions:
        if action.required or action.dest == "help":
            continue
        assert passed & set(action.option_strings) or (
            set(action.option_strings) & _RENDERER_DEFAULTS_THE_TEMPLATE_DOES_NOT_PASS
        ), f"{action.option_strings} has a default the template neither passes nor excuses"


def test_the_template_passes_the_batch_shape_to_the_renderer():
    keys = {
        key for key, tokens in _template_flags() if _accepted_by_the_renderer(tokens)
    }
    assert {
        "sft_config.gradient_accumulation_steps",
        "sft_config.learning_rate",
        "sft_config.max_length",
        "sft_config.num_train_epochs",
        "sft_config.per_device_train_batch_size",
        "sft_config.response_template",
        "sft_config.save_steps",
    } <= keys
