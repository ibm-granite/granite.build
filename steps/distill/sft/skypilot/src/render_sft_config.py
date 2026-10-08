#
# PORTED, not authored here. Upstream source of truth:
#   repo   github.ibm.com/Herbert-Woisetschlaeger/gb-steps-collection-post-training
#   path   steps/distill-sft/src/render_sft_config.py
#   commit 70c1550a171aa8e09a9ad9047a5bf763c39e8579
#
# Verbatim apart from `black`/`isort` reflow where the file is Python, which CI requires
# repo-wide. Keep it that way so re-syncing upstream stays a three-way merge; behaviour
# changes belong upstream.
#
# The response template's trailing newline is decoded by the STEP TEMPLATE before this is
# called, not here: it cannot cross gbserver's config fill as a real newline, so it travels
# as a two-character escape. That is a property of granite.build's transport, so the
# compensation lives on granite.build's side rather than in this ported file.
#
"""Validate distill-sft's step config and render it as an SFT YAML file.

WHY THIS IS A MODULE AND NOT INLINE IN run-sft.sh, and why it is a near-twin of
distill-gold-train's render_gold_config.py rather than a copy of it: this step is the CONTROL
the distillation arms are measured against, so every way it can silently differ from them is a
way the comparison stops meaning anything. The rules that must be IDENTICAL on both sides
(resume, the tokenizer agreement, the topology bounds) are imported from
gb_steps_post_training.distillation.render_common and are one implementation. What is written
out below is only what is genuinely specific to plain SFT.

The output is consumed by trl's TrlParser via `sft.py --config <file>`, so the dataclasses
(CustomSFTConfig + CustomArguments) stay the single authority on the full key surface: anything
this renderer emits that they do not accept is a hard parse error there rather than a silently
ignored key -- TrlParser runs with fail_with_unknown_args=True.

THE THREE REFUSALS THAT ARE ONLY HERE, each for a failure that otherwise lands after the
allocation is held:

  1. `use_liger_swiglu_mlp: true`. Its patch module was classified DELETE during triage and is
     not in this collection, so sft.py's `from _liger_granite_swa_patch import ...` raises --
     inside __main__, i.e. after the nodes are allocated.

  2. A KD config that reads as distillation and is not. This step becomes forward-KL
     distillation the moment `precomputed_logits_dir` is set, and there are four ways to set
     that and still train no KD term at all.

  3. `attn_implementation`. sft.py's verify_optimization_stack REQUIRES flash_attention_2 on a
     non-SWA student and aborts every rank otherwise -- after the model has loaded. The
     scratchpad configs never set it, which is why this renderer emits it explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml
from gb_steps_post_training.distillation import render_common

# Aliases so this module reads like its twin and so the shared guards are visibly shared.
ConfigError = render_common.ConfigError
RESUME_MODES = render_common.RESUME_MODES

# The attention backends sft.py's verify_optimization_stack will accept, keyed by what the
# student's config.json says it is. Not a free-form string: the check compares
# `model.config._attn_implementation` against ONE literal per architecture class and aborts
# every rank when it differs, so an unrecognised value here is a run that dies after load.
ATTN_BY_ARCH = {
    "granite_swa": "flash_attention_3",  # the vendored SWA arm; unreachable in this collection
    "*": "flash_attention_2",  # everything else, including granitemoehybrid
}


def _student_model_type(student_path: Path) -> str | None:
    """`model_type` from the student's config.json, or None if it cannot be read.

    Used for two preflights that would otherwise fire after the model loads. Deliberately
    tolerant: a missing or unparseable config.json is reported by the input-existence checks in
    terms of the directory, and inventing a second error message for it here would just make the
    first one harder to find.
    """
    cfg = student_path / "config.json"
    if not cfg.is_file():
        return None
    try:
        return json.loads(cfg.read_text()).get("model_type")
    except ValueError:
        return None


def _has_generation_markers(student_path: Path) -> bool | None:
    """Does the student's chat template carry `{% generation %}`? None if unreadable.

    This is the PRIMARY mechanism for the completion boundary: apply_chat_template's
    `return_assistant_tokens_mask=True` builds the mask from these markers, and sft.py raises
    when the resulting mask is all zeros. `response_template` is the FALLBACK, a literal string
    searched for in the rendered ids.

    Only a textual test, on purpose. Whether the fallback marker actually OCCURS in a render is
    a question about a tokenizer and a corpus, which is what
    gb_steps_post_training.distillation.masking and scripts/bluevela/checks/collator-masking.py
    answer; duplicating a weaker version of that here would produce a second, disagreeing
    authority on the one question this recipe has already got wrong once.
    """
    tpl = student_path / "chat_template.jinja"
    if not tpl.is_file():
        return None
    return "{% generation %}" in tpl.read_text()


def validate(args: argparse.Namespace, *, check_paths: bool = True) -> None:
    """Raise ConfigError on any config this step must not run."""
    if args.resume not in RESUME_MODES:
        raise ConfigError(f"resume={args.resume!r} is not one of {list(RESUME_MODES)}.")

    # ---- Shapes and schedules. Cheap, and each one is a way to run a control that trains on
    # something other than what its config appears to say.
    if args.max_length < 1:
        raise ConfigError(f"max_length must be >= 1; got {args.max_length}.")
    if args.per_device_train_batch_size < 1:
        raise ConfigError(
            f"per_device_train_batch_size must be >= 1; got {args.per_device_train_batch_size}."
        )
    if args.gradient_accumulation_steps < 1:
        raise ConfigError(
            f"gradient_accumulation_steps must be >= 1; got {args.gradient_accumulation_steps}."
        )
    if args.num_train_epochs <= 0:
        raise ConfigError(f"num_train_epochs must be > 0; got {args.num_train_epochs}.")
    if args.save_steps < 1:
        raise ConfigError(
            f"save_steps must be >= 1; got {args.save_steps}. This step runs on preemptable "
            "capacity, so a run that never checkpoints loses everything on the first preemption."
        )
    # -1 means the whole corpus. 0 selects nothing and trains on an empty dataset, which fails
    # far from its cause; any other negative is a typo for -1.
    if args.max_dataset_size == 0 or args.max_dataset_size < -1:
        raise ConfigError(
            f"max_dataset_size={args.max_dataset_size} is neither -1 (the whole corpus) nor a "
            "positive row count. 0 would select an empty dataset."
        )
    try:
        float(args.learning_rate)
    except (TypeError, ValueError):
        raise ConfigError(
            f"learning_rate={args.learning_rate!r} is not a number. It is quoted in the step "
            "template so that '1e-6' survives YAML 1.1 (which parses bare 1e-6 as a STRING), "
            "and converted to a float here."
        ) from None

    render_common.validate_topology(args.gpus_per_node, args.nodes)

    # ---- The SWA liger patch is not in this collection. sft.py imports its module inside
    # __main__ (`from _liger_granite_swa_patch import apply_liger_swiglu_to_granite_swa`), so
    # without this the failure is an ImportError several minutes into an allocated run. Refused
    # rather than silently ignored: the flag exists to prevent a backward-pass OOM at >=64K
    # context, and a run that believes it is protected and is not would OOM anyway, later.
    if args.use_liger_swiglu_mlp:
        raise ConfigError(
            "use_liger_swiglu_mlp=True asks for the Liger tiled-MLP patch to GraniteSWAMLP, "
            "which is not part of this collection: _liger_granite_swa_patch.py lives in "
            "distillation.scratchpad/gold/ and was classified DELETE during triage because no "
            "model in the granite 4.1/4.2 pairing uses sliding-window attention. It is also "
            "refused by sft.py's own verify_optimization_stack on a non-SWA student. Leave it "
            "false; use use_liger_memory_opt, which patches nothing model-internal."
        )

    # ---- KD mode coherence. Setting precomputed_logits_dir turns this step from the control
    # into forward-KL distillation, and there are four ways to do that and train no KD term.
    kd_on = bool((args.precomputed_logits_dir or "").strip())
    if kd_on:
        if args.kd_top_k < 1:
            raise ConfigError(f"kd_top_k must be >= 1; got {args.kd_top_k}.")
        if args.kd_temperature <= 0.0:
            raise ConfigError(
                f"kd_temperature must be > 0; got {args.kd_temperature}. It divides the logits."
            )
        if args.kd_weight < 0.0 or args.ce_weight < 0.0:
            raise ConfigError(
                f"kd_weight and ce_weight must be >= 0; got kd_weight={args.kd_weight}, "
                f"ce_weight={args.ce_weight}."
            )
        if args.kd_weight == 0.0 and args.ce_weight == 0.0:
            raise ConfigError(
                "kd_weight and ce_weight are both 0.0, so the total loss is identically zero "
                "and the run would train nothing while reporting a clean loss of 0."
            )
        if args.kd_weight == 0.0:
            raise ConfigError(
                f"precomputed_logits_dir is set ({args.precomputed_logits_dir}) but "
                "kd_weight=0.0, so the teacher's logits would be loaded, mmapped and collated "
                "on every step and then multiplied by zero: a run that pays for distillation "
                "and performs plain SFT, under a config that reads as distillation. If a "
                "CE-only control is what you want, leave precomputed_logits_dir empty -- that "
                "IS this step's control arm."
            )
    else:
        # The kd_* keys are read only on the precomputed-logits path (sft.py's KDDataCollator
        # and the KD branch of compute_loss). Left at their template defaults they are inert,
        # which is fine; CHANGED with no logits directory they express an intent that nothing
        # will honour, and the run trains plain CE while its config says top-256 KD at
        # temperature 2. That is the exact class of failure this renderer exists to convert
        # into a refusal.
        changed = [
            f"{name}={val}"
            for name, val, dflt in (
                ("kd_top_k", args.kd_top_k, 256),
                ("kd_weight", args.kd_weight, 1.0),
                ("ce_weight", args.ce_weight, 0.0),
                ("kd_temperature", args.kd_temperature, 1.0),
            )
            if val != dflt
        ]
        if changed:
            raise ConfigError(
                f"{', '.join(changed)} was set, but precomputed_logits_dir is empty -- so this "
                "is the plain-SFT control arm and every kd_* key is silently ignored (they are "
                "read only on the precomputed-logits path). Point precomputed_logits_dir at a "
                "distill-logit-precompute output to make them take effect, or return them to "
                "their defaults to keep this a control."
            )

    if not check_paths:
        return

    # Resume enforcement. Path-dependent, so below the check_paths gate -- but FIRST within it,
    # because it is the check most likely to abort a run the operator did not mean to start.
    render_common.validate_resume(args.resume, args.output_dir, entrypoint="sft.py")

    missing = []
    if not args.student_model_path:
        missing.append("student_model_path is empty")
    elif not Path(args.student_model_path).is_dir():
        missing.append(
            f"student_model_path is not a directory: {args.student_model_path}"
        )
    if not args.corpus_path:
        missing.append("corpus_path is empty")
    elif not Path(args.corpus_path).exists():
        missing.append(f"corpus_path does not exist: {args.corpus_path}")
    if not Path(args.deepspeed_config).is_file():
        missing.append(f"deepspeed_config is not a file: {args.deepspeed_config}")
    if kd_on and not Path(args.precomputed_logits_dir).is_dir():
        missing.append(
            f"precomputed_logits_dir is not a directory: {args.precomputed_logits_dir}"
        )
    if missing:
        raise ConfigError("input validation failed:\n  - " + "\n  - ".join(missing))

    # Shared with distill-gold-train: the student must carry a chat template, and must be the
    # tokenizer the corpus was built for. See render_common for why each is an assertion.
    render_common.validate_student_against_corpus(
        args.student_model_path, args.corpus_path
    )

    student = Path(args.student_model_path)

    # ---- The completion boundary, which is the one thing in this step that is easy to get
    # silently wrong -- and it is not fully silent: sft.py raises when the assistant mask comes
    # out all zeros. What it cannot do is tell you WHICH mechanism it was relying on.
    if (
        _has_generation_markers(student) is False
        and not (args.response_template or "").strip()
    ):
        raise ConfigError(
            f"{student}/chat_template.jinja carries no {{% generation %}} markers and "
            "response_template is empty, so there is no way to locate the assistant spans: "
            "apply_chat_template's assistant mask will be all zeros and sft.py raises on it. "
            "Either install patches/granite-chat-template-generation-markers/ (which "
            "distill-tokenizer-align does) or set response_template to the literal that "
            "precedes each assistant turn."
        )

    # ---- The attention backend. verify_optimization_stack aborts every rank when this does not
    # match the architecture, and it does so AFTER the model is loaded on every rank.
    mt = _student_model_type(student)
    want = ATTN_BY_ARCH.get(mt or "*", ATTN_BY_ARCH["*"])
    if args.attn_implementation != want:
        raise ConfigError(
            f"attn_implementation={args.attn_implementation!r} but the student's config.json "
            f"says model_type={mt!r}, for which sft.py's verify_optimization_stack requires "
            f"{want!r} and aborts every rank otherwise (it compares "
            "model.config._attn_implementation against one literal per architecture). Pass "
            f"--attn-implementation {want} or leave it at its default."
        )
    if mt == "granite_swa":
        # Reachable only if someone points this step at a vendored SWA checkpoint. The model
        # code is not in this collection, so it would fail at load -- and the FA3 preamble that
        # the architecture needs is gated off by default (_swa_arm.py).
        raise ConfigError(
            "the student's config.json says model_type='granite_swa', whose model code is not "
            "part of this collection (see _swa_arm.py: granite_swa/, _fa3_preamble.py and the "
            "Liger patch were classified DELETE during triage). transformers cannot "
            "instantiate it, so this would fail at model load."
        )


def render(args: argparse.Namespace) -> dict[str, Any]:
    """Build the SFT config mapping. Assumes validate() has already passed."""
    cfg: dict[str, Any] = {
        "model_name_or_path": args.student_model_path,
        "dataset_name": args.corpus_path,
        "output_dir": args.output_dir,
        "max_length": args.max_length,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        # Emitted as a float so "1e-6" survives: YAML 1.1 parses bare 1e-6 as a STRING (only
        # 1.0e-6 is a float), and a string learning rate reaches TrainingArguments as one.
        "learning_rate": float(args.learning_rate),
        "num_train_epochs": args.num_train_epochs,
        "save_steps": args.save_steps,
        "seed": args.seed,
        "response_template": args.response_template,
        "use_liger_memory_opt": args.use_liger_memory_opt,
        "max_dataset_size": args.max_dataset_size,
        # ALWAYS emitted, unlike on the gold path where nothing checks it. sft.py's
        # verify_optimization_stack requires the architecture's literal and aborts otherwise;
        # the scratchpad configs never set this key, so relying on the ModelConfig default
        # (None -> whatever transformers picks) is how that check fires on a correct config.
        "attn_implementation": args.attn_implementation,
    }

    # use_liger_swiglu_mlp is deliberately NOT emitted: validate() refuses it outright, and
    # emitting `false` for a flag that has only one legal value would suggest it has two.

    if (args.precomputed_logits_dir or "").strip():
        # The KD arm. Emitted as a GROUP, because these five keys are only meaningful together
        # -- see validate() for the four ways to set them and train no KD term.
        cfg["precomputed_logits_dir"] = args.precomputed_logits_dir
        cfg["kd_top_k"] = args.kd_top_k
        cfg["kd_weight"] = args.kd_weight
        cfg["ce_weight"] = args.ce_weight
        cfg["kd_temperature"] = args.kd_temperature

    # ---- Experiment tracking. Emitted ONLY when non-empty: step-template.yaml renders every
    # unset string key as "", and a key present-but-empty is a different statement from a key
    # absent. tracking.py treats "" as unset too, so this is belt and braces -- but it keeps the
    # rendered yaml honest, which is the file a human reads when a run logged nowhere.
    #
    # No run name is defaulted here. Under granite.build this file is written to the fixed name
    # `sft-config.rendered.yaml`, which tracking.auto_run_name() knows is a generic stem, so it
    # composes the name from student/corpus instead -- and, because this step passes no teacher
    # hint unless precomputed_logits_dir is set, names the control `<student>_<corpus>_sft`
    # rather than as a distillation arm.
    for _key, _val in (
        ("clearml_project", args.clearml_project),
        ("clearml_run_name", args.clearml_run_name),
        ("wandb_entity", args.wandb_entity),
        ("wandb_project", args.wandb_project),
        ("wandb_run_name", args.wandb_run_name),
    ):
        if (_val or "").strip():
            cfg[_key] = _val.strip()

    extra_raw = (args.extra_config_yaml or "").strip()
    if extra_raw:
        extra = yaml.safe_load(extra_raw)
        if not isinstance(extra, dict):
            raise ConfigError(
                f"extra_config_yaml must parse to a mapping; got {type(extra).__name__}."
            )
        # Merged UNDER the contract keys: a collision is an error, not a silent winner. An
        # escape hatch that can override an explicit contract key is not an escape hatch, it is
        # a second config system with undefined precedence.
        collisions = sorted(set(extra) & set(cfg))
        if collisions:
            raise ConfigError(
                "extra_config_yaml collides with keys this step sets explicitly: "
                f"{collisions}. Change the step's own config value instead -- these keys have "
                "one owner on purpose."
            )
        cfg.update(extra)

    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student-model-path", required=True)
    p.add_argument("--corpus-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--response-template", default="<|im_start|>assistant\n")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", default="1e-6")
    p.add_argument("--num-train-epochs", type=float, default=1)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default="auto", choices=RESUME_MODES)
    p.add_argument("--deepspeed-config", required=True)
    # BooleanOptionalAction is the house rule: the step template renders a --flag/--no-flag PAIR
    # because jinja writes a YAML boolean with PYTHON casing, so `--flag {{ x }}` reaches the
    # script as the literal string "False" and silently does nothing (LSF job 1138651).
    p.add_argument(
        "--use-liger-memory-opt", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument(
        "--use-liger-swiglu-mlp", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument("--precomputed-logits-dir", default="")
    p.add_argument("--kd-top-k", type=int, default=256)
    p.add_argument("--kd-weight", type=float, default=1.0)
    p.add_argument("--ce-weight", type=float, default=0.0)
    p.add_argument("--kd-temperature", type=float, default=1.0)
    p.add_argument("--max-dataset-size", type=int, default=-1)
    # Not in the step contract, and default-only in practice: it exists as a flag so the ONE
    # legal value per architecture is visible and overridable by someone who has read
    # verify_optimization_stack, rather than hidden in this file.
    p.add_argument("--attn-implementation", default="flash_attention_2")
    p.add_argument("--extra-config-yaml", default="")
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--nodes", type=int, default=1)
    # Tracking. Default "" rather than None so an unset step config key round-trips as "unset"
    # instead of the literal string "None".
    p.add_argument("--clearml-project", default="")
    p.add_argument("--clearml-run-name", default="")
    p.add_argument("--wandb-entity", default="")
    p.add_argument("--wandb-project", default="")
    p.add_argument("--wandb-run-name", default="")
    p.add_argument("--out", required=True, help="Path to write the rendered SFT YAML.")
    p.add_argument(
        "--no-check-paths",
        action="store_true",
        help="Skip filesystem checks (for rendering outside the runtime).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate(args, check_paths=not args.no_check_paths)
        cfg = render(args)
    except ConfigError as exc:
        print(f"FATAL [distill-sft config]: {exc}", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=True))
    print(f"rendered SFT config -> {out}")
    for k in sorted(cfg):
        print(f"  {k}: {cfg[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
