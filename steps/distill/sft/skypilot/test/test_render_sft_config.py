"""
PORTED, not authored here. Upstream source of truth:
  repo   github.ibm.com/Herbert-Woisetschlaeger/gb-steps-collection-post-training
  path   steps/distill-sft/test/test_render_sft_config.py
  commit 70c1550a171aa8e09a9ad9047a5bf763c39e8579

Divergences, all about path resolution rather than behaviour:
  - upstream's sys.path inserts are removed. conftest.py resolves the step's own src/ and
    the shared distillation package (from GB_DISTILL_CODE_DIR).
  - the PARITY TESTS import distill-gold-train's renderer, which granite.build does not
    contain: this repo's steps/distill/gold ships a DIFFERENT renderer (268 lines against
    upstream's 575), because it drives kd-sandbox's trainer rather than the vendored
    gb_steps_post_training one. So those tests would be asserting a parity that does not
    exist here.
    They are kept rather than dropped, and pointed at the UPSTREAM renderer in the
    delivered checkout. What they then assert is true and worth knowing: that upstream's
    control and its treatment share their guards, their optimization defaults and their
    tracking surface. That is a property of the code this step actually runs, which is the
    point of the whole ported test tier.

Keep this a near-verbatim copy so re-syncing upstream stays a three-way merge.
"""

"""Tests for distill-sft's config validation and rendering.

Two kinds of test live here, and the second kind is the reason this file is not just a copy
of distill-gold-train's suite with the names changed.

THE FIRST KIND covers the refusals only this step has -- the attention backend, the Liger
SWA patch that is not in this collection, and the four ways to set `precomputed_logits_dir`
and still train no KD term. Each one would otherwise fire inside `sft.py` AFTER the model is
loaded on every rank, which on this cluster means after the allocation is held.

THE SECOND KIND is about being a CONTROL. Every number the distillation arms report is a
comparison against this step, so a way for it to differ silently from them is a way the
comparison stops meaning anything. Two properties are asserted directly against
distill-gold-train's renderer rather than restated: that the shared guards are literally the
same objects, and that the optimization defaults agree key for key -- which the step README
has claimed since it was a scaffold and nothing checked. `nodes` is the single intended
divergence and it is named as such, so the test fails if a SECOND one appears.

Nothing here needs a GPU, CUDA, trl, or transformers.
"""

import ast
import json
import sys
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve()
# The repo's shared distillation package: render_sft_config imports render_common from it, and
# the parity tests below import distill-gold-train's renderer, which does too. The Dockerfile
# vendors the same package onto PYTHONPATH, so this mirrors the image layout.
# distill-gold-train's renderer, for the parity tests. A test that asserts the control matches
# the treatment has to read the treatment, not a transcription of it.

# See the module docstring: distill-gold-train's renderer lives only in the delivered
# checkout, so resolve it there rather than expecting it beside this file. conftest.py has
# already validated that GB_DISTILL_CODE_DIR points at a real checkout, or skipped this
# module entirely.
import os as _os
import sys as _sys

_CHECKOUT = Path(
    _os.environ.get(
        "GB_DISTILL_CODE_DIR",
        "/proj/granite-build/g4os/gb-steps-collection-post-training",
    )
)
_GOLD_SRC = _CHECKOUT / "steps" / "distill-gold-train" / "src"
if _GOLD_SRC.is_dir() and str(_GOLD_SRC) not in _sys.path:
    _sys.path.insert(0, str(_GOLD_SRC))

import render_gold_config  # noqa: E402
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
    """Shared with distill-gold-train, and it has to be: a control trained on a corpus
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
# The rules are render_common's and distill-gold-train's suite covers their behaviour. What
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
    # Repointed: upstream's _HERE.parents[2] assumed this file sits inside the steps
    # collection. From granite.build the treatment's renderer is in the delivered checkout.
    "render_gold_config": _GOLD_SRC / "render_gold_config.py",
}


def test_both_renderers_raise_the_same_error_type():
    """`except ConfigError` in one launcher must catch what the other raises: the two are
    aliases of render_common.ConfigError, not two same-named classes."""
    assert ConfigError is render_common.ConfigError
    assert render_gold_config.ConfigError is render_common.ConfigError


@pytest.mark.parametrize("name", SHARED_GUARDS)
def test_the_shared_guards_exist_in_one_place(name):
    assert callable(getattr(render_common, name))


@pytest.mark.parametrize("module,path", sorted(_RENDERERS.items()))
@pytest.mark.parametrize("name", SHARED_GUARDS)
def test_each_renderer_calls_the_shared_guard_and_defines_no_copy(name, module, path):
    """Read as SOURCE, not as attributes: the failure this catches is someone answering a
    divergence by pasting a local copy of the rule into one of the two files, which no
    identity check on render_common's own attributes can see.

    A local `def validate_resume` would shadow the import, both steps would pass their own
    tests, and the two arms would differ on exactly the rule this asserts they share."""
    src = path.read_text()
    tree = ast.parse(src)
    defined = {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert (
        name not in defined
    ), f"{module} defines its own {name}; it must call render_common's"
    assert f"render_common.{name}(" in src, f"{module} never calls render_common.{name}"


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
# Repointed for the same reason as render_gold_config above: sft.py is the trainer this
# step runs, and it lives in the delivered checkout rather than in this repo.
_SFT_PY = _CHECKOUT / "src" / "gb_steps_post_training" / "distillation" / "sft.py"


def test_kd_requests_the_assistant_mask_regardless_of_assistant_only_loss():
    src = _SFT_PY.read_text()
    assert "return_assistant_tokens_mask=assistant_only_loss," not in src, (
        "a call site gates the mask on assistant_only_loss again; in KD mode that switches "
        "the mask mechanism to the response-template scan and desynchronises it from the "
        "producer by one token per assistant turn"
    )
    n = src.count("return_assistant_tokens_mask=(assistant_only_loss or kd_mode)")
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


# The one key the two steps are MEANT to differ on. There is no student rollout in plain SFT,
# so there is no vLLM server, no split-role allocation and no reason to hold a second node.
INTENDED_DIVERGENCES = {"nodes": (2, 1)}


def test_optimization_defaults_match_distill_gold_train():
    """The README has said "the optimization defaults, which are distill-gold-train's, key
    for key" since this step was a scaffold. A control that trains at a different learning
    rate, batch size, sequence length or seed than the treatment is not a control, and
    nothing about either run would fail -- the two just stop being comparable."""
    gold = _parser_defaults(render_gold_config.build_parser())
    sft = _parser_defaults(build_parser())
    differ = {k: (gold[k], sft[k]) for k in set(gold) & set(sft) if gold[k] != sft[k]}
    assert differ == INTENDED_DIVERGENCES


def test_the_intended_divergence_is_still_real():
    """Guards the test above from going vacuous: if `nodes` is ever made equal, the entry
    above stops describing anything and would start hiding the next divergence."""
    gold = _parser_defaults(render_gold_config.build_parser())
    sft = _parser_defaults(build_parser())
    for key, (want_gold, want_sft) in INTENDED_DIVERGENCES.items():
        assert (gold[key], sft[key]) == (want_gold, want_sft)


def test_the_tracking_surface_is_the_same_five_keys():
    """Same keys, same defaults, same tracking.py. A control whose runs land in a different
    project, or under a differently shaped name, is a control nobody can line up against the
    arms it exists to anchor."""
    keys = {
        "clearml_project",
        "clearml_run_name",
        "wandb_entity",
        "wandb_project",
        "wandb_run_name",
    }
    gold = _parser_defaults(render_gold_config.build_parser())
    sft = _parser_defaults(build_parser())
    assert keys <= set(gold) and keys <= set(sft)
    assert (
        {k: sft[k] for k in keys} == {k: gold[k] for k in keys} == {k: "" for k in keys}
    )


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
