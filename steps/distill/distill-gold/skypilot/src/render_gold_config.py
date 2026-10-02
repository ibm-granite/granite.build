#!/usr/bin/env python3
"""Render the flat YAML config the GOLD trainer (gb_steps_post_training.distillation.gold)
consumes.

Why this is a Python script rather than a heredoc in the step's ``run:`` block.
Jinja does work there, so this is not about capability — it is about which
mistakes stay silent. Three of the trainer's requirements corrupt a run without
producing an error, and each is handled badly by a shell heredoc:

* ``learning_rate`` and the scheduler's ``min_lr`` must be YAML floats. PyYAML
  parses a bare ``1e-05`` as a *string*, which crashes the trainer's min_lr
  handling with a str/float TypeError partway into a run. Emitting through
  ``yaml.safe_dump`` of an actual float makes that structural instead of a
  formatting convention one edit away from breaking.
* ``min_lr`` must be nested under ``lr_scheduler_kwargs``, never top level. The
  trainer parses with TRL's ``parse_args_and_config``, which rejects unknown
  top-level keys, so a flat ``min_lr`` fails the run outright — after the model
  and teacher have already loaded.
* Booleans must be lower-case YAML. ``safe_dump`` does that by construction;
  templating emits ``True`` unless every site remembers to lower-case it.
* The six on-policy keys must be emitted **only** when ``vllm_num_servers > 0``.
  That is one testable branch here, versus ``{% if %}`` nested inside a quoted
  heredoc inside a YAML literal block — and it is exactly the seam the on-policy
  phase will reopen.

* The trainer's newer keys must be ABSENT, not false, when unused. ``ce_coef``,
  ``log_student_entropy`` and the entropy-guard trio are NOT at the commit
  ``code_config`` pins by default; they need a checkout carrying
  ``patches/ce_anchor_and_entropy_guard.diff``. Emitting them unconditionally would make
  every config here unreadable by any other checkout, and TRL rejects unknown top-level
  keys outright -- which is what ``--verify-trainer-accepts-keys`` turns into a refusal
  that names the patch instead of a failure on a held allocation. Emitting them only when they are non-default also keeps
  the rendered config of every existing recipe byte-identical, which is what
  ``test_off_policy_key_set_is_exact`` asserts.

Dumping with the same library the trainer parses with also means a config that
renders is a config the trainer can read.
"""

import argparse
import dataclasses
import sys
from typing import Any, Dict

import yaml

# Fields emitted only on the on-policy (online) path. Keeping an off-policy
# config free of them is not cosmetic: their presence is what the launcher and
# trainer read to decide whether to expect a vLLM server at all.
_ONLINE_ONLY = (
    "vllm_num_servers",
    "top_p",
    "use_sampled_opd_loss",
    "last_message_only",
    "clip_alpha",
    "opd_importance_sampling",
)


# ─── Loss arms ────────────────────────────────────────────────────────────────────────
#
# CustomGOLDConfig exposes nine boolean loss switches. They are NOT orthogonal, and this
# branch already knows it: gold-onpolicy-smoke/parameters.yaml notes that "CustomGOLDConfig
# rejects liger with several on-policy options". This table is that knowledge made
# mechanical, for the two kinds of invalid combination an enum cannot express:
#
#   (a) COMBINATIONS THE DATACLASS REJECTS in __post_init__. Failing here only saves the
#       allocation -- the run would have died anyway, just later and after the models loaded.
#
#   (b) COMBINATIONS THAT SILENTLY TRAIN THE WRONG OBJECTIVE, which is worse because the run
#       SUCCEEDS. use_liger_fused_jsd selects a fused-kernel branch constructed with beta,
#       alpha, temperature and use_kl_interpolation ONLY; every other secondary switch is
#       dropped by that path without a word. So liger together with use_ce_loss,
#       use_distillm2, use_distillm2_like, use_adaptive_kld or use_reversed_distillm2_like
#       trains fused generalized JSD while the config claims another objective. Exactly one
#       of those five is caught by the dataclass; the other four pass validation.
#
# DELIBERATELY WITHOUT TRAINER LINE NUMBERS. An earlier copy of this table cited
# custom_gold_trainer.py line numbers, and they were not transferable: they were read off a
# fork whose additions shift every line below them. A precise-looking wrong citation is worse
# than none, so the branch conditions are named by their flags -- which are stable -- instead.
#
# WHAT IS DELIBERATELY NOT AN ARM, because `beta` already reaches it: forward KL is `jsd` at
# beta 0.0, reverse KL is `jsd` at beta 1.0. Arms for those would be two ways to say one
# thing, free to disagree.
#
# EACH ARM CARRIES ITS OWN REQUIREMENTS rather than having them coerced, because the second
# kind of requirement below is one nothing downstream would catch: outside its bounds the arm
# does not fail, it silently BECOMES a different arm.
LOSS_ARMS: Dict[str, Dict[str, Any]] = {
    "jsd": {
        "flags": {},
        "doc": "mixture generalized JSD; beta 0.0 = forward KL, 1.0 = reverse KL",
    },
    "kl_interpolation": {
        "flags": {"use_kl_interpolation": True},
        "doc": "convex (1-beta)*FKL + beta*RKL instead of the mixture JSD",
        # Not a dataclass rule; the arm is simply pointless outside it. The beta == 0 and
        # beta == 1 short-circuits PRECEDE the interpolation branch, so at those values the
        # flag is never read and this arm is byte-identical to `jsd`.
        "requires": {"beta_strictly_between": (0.0, 1.0)},
        "why": (
            "the interpolation branch sits after the beta == 0 and beta == 1 "
            "short-circuits, so at those values the flag is never read and this arm is "
            "identical to loss_arm='jsd'"
        ),
    },
    "adaptive_kld": {
        "flags": {"use_adaptive_kld": True},
        "doc": "adaptive KL divergence weighting inside generalized_jsd_loss",
    },
    "liger_fused_jsd": {
        "flags": {"use_liger_fused_jsd": True},
        "doc": "same JSD math via Liger's fused linear kernel; lower peak memory",
    },
    "liger_fused_kl_interpolation": {
        "flags": {"use_liger_fused_jsd": True, "use_kl_interpolation": True},
        # The ONE composition with liger that is genuinely honoured: the fused loss object
        # takes use_kl_interpolation as a constructor argument. Every other secondary flag is
        # silently dropped by that path, which is why no other liger pairing is offered.
        "doc": "fused liger kernel computing the convex FKL/RKL interpolation",
        "requires": {"beta_strictly_between": (0.0, 1.0)},
        "why": (
            "the fused skewed-JSD loss receives beta at construction and the interpolation "
            "is only meaningful strictly between the two pure divergences"
        ),
    },
    "distillm2": {
        "flags": {"use_distillm2": True},
        "doc": "DistiLLM-2, comparative: reverse KL on on-policy tokens, forward KL off",
        # Both terms are guarded by .any(), so outside 0 < lmbda < 1 this does not crash --
        # it DEGENERATES. At lmbda 0.0 every token is off-policy and the loss IS forward KL;
        # at 1.0 every token is on-policy and it IS reverse KL. Silent equivalence.
        "requires": {"lmbda_strictly_between": (0.0, 1.0)},
        "why": (
            "the comparative loss needs both policy classes present in a batch; at "
            "lmbda 0.0 it degenerates to forward KL and at 1.0 to reverse KL, either of "
            "which loss_arm='jsd' expresses directly at beta 0.0 / 1.0"
        ),
    },
    "distillm2_like": {
        "flags": {"use_distillm2_like": True},
        "doc": "DistiLLM-2 shape, non-comparative; policy class decided per microbatch",
        "requires": {"lmbda_strictly_between": (0.0, 1.0)},
        "why": (
            "the on/off-policy branch is what distinguishes this arm; outside "
            "0 < lmbda < 1 one side is unreachable and it reduces to a fixed divergence"
        ),
    },
    "reversed_distillm2_like": {
        "flags": {"use_reversed_distillm2_like": True},
        # A SIBLING branch of distillm2_like, not a modifier of it: setting both would make
        # this one dead code, which is why it is a separate arm with the flag alone.
        "doc": "distillm2_like with the on/off-policy divergence roles exchanged",
        "requires": {"lmbda_strictly_between": (0.0, 1.0)},
        "why": (
            "same reason as distillm2_like: the exchanged roles are only observable when "
            "both policy classes occur"
        ),
    },
    "sampled_opd": {
        "flags": {"use_sampled_opd_loss": True},
        "doc": "REINFORCE-style policy gradient on sampled tokens; optional truncated IS",
        # The dataclass's own rules, surfaced rather than coerced: quietly rewriting a
        # requested lmbda to satisfy a loss arm would change the experiment without saying so.
        "requires": {"lmbda_eq": 1.0, "last_message_only": True},
        "why": "CustomGOLDConfig.__post_init__ demands both",
    },
    "uld": {
        "flags": {"use_uld_loss": True},
        # NOT in the chain above at all: it compares SORTED logit distributions, so teacher
        # and student need not share a vocabulary. The arm for a genuinely cross-tokenizer
        # pair; for Granite/Granite it buys nothing over an aligned comparison.
        "doc": "Universal Logit Distillation over sorted logits; the cross-tokenizer arm",
    },
    "ce": {
        "flags": {"use_ce_loss": True},
        # An SFT control that runs INSIDE this trainer, on the same data path, collator and
        # masking as every distillation arm -- so a gap between them cannot be a data-pipeline
        # artefact. Not a replacement for distill-sft, which trains with no teacher
        # resident at all.
        "doc": "cross entropy only -- in-trainer SFT control, teacher still loaded",
        "requires": {"lmbda_eq": 0.0},
        "why": (
            "with no teacher term in the loss, on-policy generation would train the student "
            "on its own samples with no teacher signal at all, and a control that trains on "
            "its own output is not a control. Note the cost this arm pays regardless: "
            "use_ce_loss does not unload the teacher, so it holds memory it never reads"
        ),
    },
}

# Requirement kinds _validate_arm knows how to enforce. Named separately so a typo'd
# requirement key is an error rather than a requirement that silently does not apply.
_REQUIREMENT_KINDS = frozenset(
    {"lmbda_eq", "lmbda_strictly_between", "beta_strictly_between", "last_message_only"}
)

# Arm flags that ALSO have a standalone CLI flag on this renderer. An arm and its standalone
# flag disagreeing is a contradiction rather than a precedence question: see _validate_arm.
_STANDALONE_ARM_FLAGS = {
    "use_liger_fused_jsd": "--use-liger-fused-jsd",
    "use_sampled_opd_loss": "--use-sampled-opd-loss",
}


def _lr(value: float) -> float:
    """Return a learning rate as a float with two-decimal exponent precision.

    Round-tripping through ``%.2e`` keeps the rendered value identical to the
    validated reference configs (``1.00e-05``) while remaining a float, so PyYAML
    reads a number rather than a string.
    """
    return float(f"{float(value):.2e}")


def _validate_arm(args: argparse.Namespace, *, online: bool) -> Dict[str, Any]:
    """Return the loss-arm's flags, or {} when no arm was named.

    NO ARM IS THE DEFAULT, and that is a deliberate three-way distinction rather than
    defaulting to ``jsd``. Every recipe on this branch pins ``USE_LIGER_FUSED_JSD: false``
    and three of their READMEs instruct re-running with ``--param USE_LIGER_FUSED_JSD=true``
    to answer an open question. If ``jsd`` were the default arm, that documented escape hatch
    would start hard-erroring on a contradiction the operator never asked for. So:

      * no arm            -- the standalone flags are authoritative, exactly as before. The
                             rendered config is unchanged, which is what keeps
                             test_off_policy_key_set_is_exact green.
      * ``--loss-arm jsd`` -- an explicit claim about the objective, and now a contradiction
                             with ``--use-liger-fused-jsd true`` is a real disagreement to
                             refuse rather than a default arguing with an operator.
    """
    if not args.loss_arm:
        return {}
    if args.loss_arm not in LOSS_ARMS:
        raise ValueError(
            f"loss_arm={args.loss_arm!r} is not one of {sorted(LOSS_ARMS)}"
        )

    arm = LOSS_ARMS[args.loss_arm]
    flags: Dict[str, Any] = dict(arm["flags"])
    why = arm.get("why", "")
    tail = f" WHY: {why}." if why else ""

    # Per-arm requirements, driven off the table rather than written out per arm. Eleven arms
    # and six requirements written by hand is six near-identical blocks, and the failure mode
    # of six near-identical blocks is a SEVENTH arm added with a "requires" entry and no block
    # to read it: a requirement that exists, reads as enforced, and is not. The final branch
    # makes that impossible.
    for kind, bound in arm.get("requires", {}).items():
        if kind == "lmbda_eq" and args.lmbda != bound:
            raise ValueError(
                f"loss_arm={args.loss_arm!r} requires lmbda={bound}; got {args.lmbda}."
                f"{tail} Set lmbda explicitly rather than having it changed for you"
            )
        if kind == "beta_strictly_between" and not bound[0] < args.beta < bound[1]:
            raise ValueError(
                f"loss_arm={args.loss_arm!r} requires {bound[0]} < beta < {bound[1]}; "
                f"got beta={args.beta}.{tail}"
            )
        if kind == "lmbda_strictly_between" and not bound[0] < args.lmbda < bound[1]:
            raise ValueError(
                f"loss_arm={args.loss_arm!r} requires {bound[0]} < lmbda < {bound[1]}; "
                f"got lmbda={args.lmbda}.{tail}"
            )
        if kind == "last_message_only" and bool(args.last_message_only) != bool(bound):
            raise ValueError(
                f"loss_arm={args.loss_arm!r} requires last_message_only={bound}.{tail}"
            )
        if kind not in _REQUIREMENT_KINDS:
            raise ValueError(
                f"internal: loss_arm={args.loss_arm!r} declares requirement {kind!r}, "
                "which this renderer does not implement. Fix the arm table or this loop"
            )

    # A named arm and a standalone flag disagreeing is a CONTRADICTION, refused in both
    # directions -- not a precedence question resolved quietly in favour of one of them.
    # Both directions matter, and the second is the dangerous one: `--loss-arm uld
    # --use-liger-fused-jsd true` would select the fused branch and drop the ULD flag
    # without a word, producing a run that reports ULD in its config and trained JSD.
    for key, flag in _STANDALONE_ARM_FLAGS.items():
        implied = bool(flags.get(key, False))
        if bool(getattr(args, key)) != implied:
            raise ValueError(
                f"loss_arm={args.loss_arm!r} implies {key}={implied} but {flag} was given "
                f"as {bool(getattr(args, key))}. Refusing rather than silently preferring "
                f"one: either drop --loss-arm and drive the objective with {flag} directly, "
                f"or pass {flag} {str(implied).lower()}"
            )

    # An arm whose flag lives in the on-policy block, selected on an off-policy render, would
    # have that flag DROPPED at emission -- the config would claim the arm and the trainer
    # would never see it. The one arm this can happen to is sampled_opd, whose lmbda_eq 1.0
    # requirement makes it on-policy by intent while vllm_num_servers is what actually gates
    # emission here.
    if not online:
        dropped = sorted(k for k in flags if k in _ONLINE_ONLY)
        if dropped:
            raise ValueError(
                f"loss_arm={args.loss_arm!r} sets {', '.join(dropped)}, which is emitted "
                "only on the on-policy path (vllm_num_servers > 0). Rendering it off-policy "
                "would drop the flag and train a different objective than the config names; "
                "set vllm_num_servers, or choose an off-policy arm"
            )
    return flags


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Assemble the gold config mapping from parsed arguments."""
    online = args.vllm_num_servers > 0
    # An EXTERNAL server (its URL arrives from another target's mem:// binding)
    # is not part of this allocation, so none of this allocation's nodes is taken
    # away from the trainer and the node arithmetic below does not apply.
    external = bool(args.vllm_server_url)

    if external and not online:
        raise ValueError(
            "a vllm_server_url was given but vllm_num_servers is 0, so the "
            "on-policy config keys would not be emitted and the run would train "
            "off-policy while a server sat idle; set vllm_num_servers to the "
            "number of external servers"
        )
    if external and args.lmbda <= 0:
        raise ValueError(
            f"a vllm_server_url was given but lmbda is {args.lmbda}; at lmbda 0 "
            "the student never generates, so the server would be allocated and "
            "never used"
        )
    if online and not external and args.total_nodes < 2:
        raise ValueError(
            "on-policy needs at least 2 nodes (one vLLM server plus one "
            f"trainer); got total_nodes={args.total_nodes}"
        )
    if online and not external and args.vllm_num_servers >= args.total_nodes:
        raise ValueError(
            f"vllm_num_servers ({args.vllm_num_servers}) must be < total nodes "
            f"({args.total_nodes}); every node would serve and none would train"
        )

    # The CE anchor and the collapse guard. Validated here rather than left to the
    # trainer's __post_init__ because a bad value there surfaces after accelerate has
    # launched, the teacher has loaded and a 2-node allocation is already held.
    if args.ce_coef < 0:
        raise ValueError(f"ce_coef must be >= 0; got {args.ce_coef}")
    if not 0.0 <= args.entropy_guard_drop_frac < 1.0:
        raise ValueError(
            "entropy_guard_drop_frac must be in [0.0, 1.0); got "
            f"{args.entropy_guard_drop_frac}"
        )
    if args.entropy_guard_drop_frac > 0 and not args.log_student_entropy:
        raise ValueError(
            "entropy_guard_drop_frac > 0 needs log_student_entropy true: the guard reads "
            "the entropy that flag computes, and without it the guard never arms — which "
            "looks exactly like a run that never collapsed"
        )
    if args.entropy_guard_baseline_steps < 1:
        raise ValueError("entropy_guard_baseline_steps must be >= 1")
    if args.entropy_guard_patience < 1:
        raise ValueError("entropy_guard_patience must be >= 1")
    if args.entropy_guard_action not in ("stop", "warn"):
        raise ValueError(
            "entropy_guard_action must be 'stop' or 'warn'; got "
            f"{args.entropy_guard_action!r}"
        )

    # The lmbda ramp and the generation floor are on-policy-only. At lmbda 0 the student
    # never generates, so both would be silently inert rather than wrong — which is the
    # class of mistake this renderer exists to turn into an error.
    if args.lmbda_schedule not in ("constant", "linear"):
        raise ValueError(
            f"lmbda_schedule must be 'constant' or 'linear'; got {args.lmbda_schedule!r}"
        )
    if args.lmbda_schedule == "linear":
        if not 0.0 <= args.lmbda_init <= 1.0:
            raise ValueError(
                "lmbda_schedule linear needs lmbda_init in [0.0, 1.0]; got "
                f"{args.lmbda_init}"
            )
        if not online:
            raise ValueError(
                "lmbda_schedule linear ramps towards on-policy sampling, so it needs a "
                "vLLM server; set vllm_num_servers > 0 or leave the schedule constant"
            )
    if args.min_completion_length > 0 and not online:
        raise ValueError(
            "min_completion_length bounds the student's vLLM rollout, so it does nothing "
            "off-policy; set vllm_num_servers > 0 or leave it 0"
        )

    config: Dict[str, Any] = {
        "model_name_or_path": args.model_name_or_path,
        "teacher_model_name_or_path": args.teacher_model_name_or_path,
        "dataset_name": args.dataset_name,
        "num_train_epochs": float(args.num_train_epochs),
        # max_steps bounds a run by optimizer steps instead of by epochs, which is
        # what the sweep arms need: their corpus is 802,027 rows, so one epoch is
        # 4,177 steps at effective batch 192 and no amount of shrinking the data
        # substitutes for capping the steps. Emitted ONLY when > 0, for two
        # reasons: every validated epoch-bounded reference config omits the key
        # entirely, and gold-smoke's rendered config has to stay byte-identical
        # (test_off_policy_key_set_is_exact asserts the key set, not just the
        # values). When set it overrides num_train_epochs in the trainer.
        **({"max_steps": args.max_steps} if args.max_steps > 0 else {}),
        "learning_rate": _lr(args.learning_rate),
        "warmup_ratio": float(args.warmup_ratio),
        "lr_scheduler_type": args.lr_scheduler_type,
        # min_lr belongs to the SCHEDULER, not the top level. The trainer parses
        # its config with TRL's parse_args_and_config, which rejects unknown
        # top-level keys outright:
        #   ValueError: Unknown arguments from config file: ['--min_lr', ...]
        # Every validated GOLD config nests it this way.
        "lr_scheduler_kwargs": {
            "min_lr": _lr(args.min_lr),
        },
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_completion_length": args.max_completion_length,
        "max_length": args.max_length,
        "gradient_checkpointing": args.gradient_checkpointing,
        # Set explicitly by every validated reference config.
        "save_strategy": args.save_strategy,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "logging_steps": args.logging_steps,
        "dataset_num_proc": args.dataset_num_proc,
        "temperature": float(args.temperature),
        "lmbda": float(args.lmbda),
        "beta": float(args.beta),
        # granite's logits_scaling=10 overflows the fused bf16 JSD kernel and
        # yields NaN loss; the non-fused generalized_jsd_loss is stable.
        "use_liger_fused_jsd": args.use_liger_fused_jsd,
        # Required: locates the completion span for loss masking. The nothink
        # chat template carries no {% generation %} tag, so without this the
        # trainer cannot tell prompt from completion.
        "response_template": _decode_escapes(args.response_template),
        # Emitted only when asked for — see the fourth bullet in the module docstring.
        # ce_coef ADDS cross entropy to the divergence; it is not use_ce_loss, which
        # replaces it. Build df8512e0 ran the unanchored objective for 8,150 steps and
        # lost 42% of the student's entropy to it.
        **({"ce_coef": float(args.ce_coef)} if args.ce_coef > 0 else {}),
        **({"log_student_entropy": True} if args.log_student_entropy else {}),
        **(
            {
                "entropy_guard_drop_frac": float(args.entropy_guard_drop_frac),
                "entropy_guard_baseline_steps": args.entropy_guard_baseline_steps,
                "entropy_guard_patience": args.entropy_guard_patience,
                "entropy_guard_action": args.entropy_guard_action,
            }
            if args.entropy_guard_drop_frac > 0
            else {}
        ),
        # The two on-policy shaping keys, same rule. lmbda_init is meaningless unless the
        # schedule is linear, so the pair travels together.
        **(
            {
                "lmbda_schedule": args.lmbda_schedule,
                "lmbda_init": float(args.lmbda_init),
            }
            if args.lmbda_schedule == "linear"
            else {}
        ),
        **(
            {"min_completion_length": args.min_completion_length}
            if args.min_completion_length > 0
            else {}
        ),
    }

    # The arm's flags, and ONLY the ones not already emitted above. A flag that is also a
    # standalone CLI flag (use_liger_fused_jsd) or lives in the on-policy block
    # (use_sampled_opd_loss) is already in the config with the right value, because
    # _validate_arm refused any render where args and arm disagreed. So this adds exactly the
    # switches the renderer had no flag for, and adds NOTHING when no arm was named -- which
    # is what keeps every existing recipe's key set unchanged.
    config.update(
        {
            key: value
            for key, value in _validate_arm(args, online=online).items()
            if key not in _ONLINE_ONLY and key not in _STANDALONE_ARM_FLAGS
        }
    )

    if online:
        config.update(
            {
                "vllm_num_servers": args.vllm_num_servers,
                "top_p": float(args.top_p),
                "use_sampled_opd_loss": args.use_sampled_opd_loss,
                "last_message_only": args.last_message_only,
                "clip_alpha": float(args.clip_alpha),
                "opd_importance_sampling": args.opd_importance_sampling,
            }
        )
    return config


def _decode_escapes(value: str) -> str:
    r"""Decode a literal ``\n`` in a shell-supplied string into a real newline.

    Why the template travels escaped. ``response_template`` must end at a line
    boundary, and gbserver fills every config string through Jinja
    (fill_objtemplate -> SandboxedEnvironment without keep_trailing_newline), which
    strips exactly one trailing newline from each VALUE. A real newline therefore
    cannot survive the trip: build d8470f14 sent one and the step received
    ``<|im_start|>assistant``, masking loss from the wrong token with nothing to
    read. A two-character escape has no trailing whitespace to strip, so it arrives
    intact and is decoded here, once, at the far end.

    Deliberately not ``unicode_escape``: that codec round-trips through latin-1 and
    mangles any non-ASCII in a chat template. Only the escape the transport needs is
    interpreted, so every other backslash reaches the trainer as written.
    """
    return value.replace("\\n", "\n")


def check_corpus_tokenizer(args: argparse.Namespace) -> None:
    """Refuse to train a student against a corpus built for a different tokenizer.

    THE FAILURE THIS CATCHES IS SILENT AND EXPENSIVE. After retagging, two Granite
    tokenizers' ids are interchangeable, but their pre_tokenizers still differ (a Split regex
    on 4.1 vs plain ByteLevel on 4.2), so the same text segments differently. A run on the
    wrong corpus does not error: it trains on mis-segmented text at full speed with a falling
    loss. distill-pipeline-smoke chains corpus-prep straight into this step, which is exactly
    where a mismatch can be introduced by changing one parameter.

    LAZY AND LOUD, never silent. The check lives in the shared package, which this step
    delivers via code_config on every run; if PYTHONPATH was not set up to reach it (this
    script invoked directly rather than through the step's run block, for instance), asking
    for the check anyway is an error naming why -- because a validator that quietly does not
    run is worse than one that is absent: the operator believes they checked.
    """
    try:
        from gb_steps_post_training.distillation import (  # noqa: PLC0415
            render_common,
        )
    except ImportError as exc:
        raise ValueError(
            "--check-corpus-tokenizer was requested but gb_steps_post_training is not "
            f"importable ({exc}). That package reaches this container through code_config's "
            "clone, normally exported onto PYTHONPATH by the step's run block -- either run "
            "this through the step, or drop the check deliberately rather than leaving it "
            "to fail open"
        ) from exc

    # The corpus argument is the DATASET path the trainer will read. corpus_tokenizer_identity
    # resolves a file's sibling manifest, so this works with the flag the step already passes
    # and needs no second path to be kept in step with it.
    try:
        render_common.validate_student_against_corpus(
            args.model_name_or_path, args.dataset_name
        )
    except render_common.ConfigError as exc:
        raise ValueError(str(exc)) from exc


# Keys this renderer can emit that upstream trl's GOLDConfig does not define. Every one of
# them needs the DELIVERED trainer to carry the field, and the delivered trainer is whatever
# code_config's pin resolves to -- which is not guaranteed to be the tree these keys were
# written against. See patches/ce_anchor_and_entropy_guard.diff.
_TRAINER_EXTENSION_KEYS = frozenset(
    {
        "ce_coef",
        "log_student_entropy",
        "entropy_guard_drop_frac",
        "entropy_guard_baseline_steps",
        "entropy_guard_patience",
        "entropy_guard_action",
        "lmbda_schedule",
        "lmbda_init",
        "min_completion_length",
    }
)


def check_trainer_accepts(config: Dict[str, Any]) -> None:
    """Refuse to emit a key the delivered trainer's dataclass does not define.

    THE FAILURE THIS CATCHES COSTS AN ALLOCATION. TrlParser rejects unknown top-level keys
    outright, so a config naming ce_coef against a trainer without the field dies after
    accelerate has launched and the teacher has loaded on every node -- and the message names
    a YAML key, not the reason, so it reads as a recipe typo rather than a trainer that
    predates the key.

    It is also the SILENT direction that matters: min_completion_length reaches the sampler
    through run_vllm_serve's GOLD_MIN_TOKENS handling, and where the delivered code has
    neither the field nor that handling, a floor the recipe asked for simply never applies.
    The student is then free to emit an immediate EOS and the run collapses into empty
    completions the teacher scores as if they were rollouts, with nothing in any log saying
    the floor was dropped.

    Checked against dataclasses.fields rather than a hardcoded list of "patched" commits, so
    the check cannot drift from the trainer it is about: bump the pin to a tree that carries
    the fields and this goes quiet on its own.

    Off in this script's own argparse and ON in the step's config, for the same reason
    check_corpus_tokenizer is: the step delivers the package on every run, while a direct
    invocation (a unit test rendering a key in isolation) has no package to ask.
    """
    requested = sorted(_TRAINER_EXTENSION_KEYS & set(config))
    if not requested:
        return

    try:
        from gb_steps_post_training.distillation.custom_gold_config import (  # noqa: PLC0415
            CustomGOLDConfig,
        )
    except ImportError as exc:
        raise ValueError(
            f"this config emits {', '.join(requested)}, which only some trainers accept, but "
            f"gb_steps_post_training is not importable ({exc}) so that cannot be checked. "
            "The package reaches this container through code_config's clone, normally exported "
            "onto PYTHONPATH by the step's run block -- either run this through the step, or "
            "set gold_config.verify_trainer_accepts_keys false to skip the check deliberately"
        ) from exc

    known = {f.name for f in dataclasses.fields(CustomGOLDConfig)}
    missing = [key for key in requested if key not in known]
    if missing:
        raise ValueError(
            f"the delivered trainer's CustomGOLDConfig has no field(s) "
            f"{', '.join(missing)}, so emitting them would make TrlParser reject this config "
            "on every node after the allocation is already held. The trainer comes from "
            "code_config (see the step's step-template.yaml); to use these keys, point "
            "code_config.code_dir at a checkout carrying "
            "steps/distill/distill-gold/skypilot/patches/ce_anchor_and_entropy_guard.diff -- its "
            "header has the exact patch invocation and the directory remap it needs. To run "
            "without them, leave the corresponding gold_config keys at their defaults"
        )


def _bool(value: str) -> bool:
    """Parse a YAML-ish boolean from the step's shell-rendered arguments."""
    lowered = str(value).strip().lower()
    if lowered in ("true", "1", "yes", "on"):
        return True
    if lowered in ("false", "0", "no", "off", ""):
        return False
    raise argparse.ArgumentTypeError(f"not a boolean: {value!r}")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", required=True, help="Path to write the YAML to.")
    p.add_argument(
        "--total-nodes",
        type=int,
        required=True,
        help="Nodes in the allocation, used to validate the split.",
    )

    p.add_argument("--model-name-or-path", required=True, help="Student init.")
    p.add_argument("--teacher-model-name-or-path", required=True)
    p.add_argument("--dataset-name", required=True)

    p.add_argument("--num-train-epochs", type=float, default=1.0)
    # 0 => absent from the rendered config, i.e. bound the run by epochs.
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--learning-rate", type=float, default=1.0e-05)
    p.add_argument("--min-lr", type=float, default=1.0e-06)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--lr-scheduler-type", default="cosine_with_min_lr")
    p.add_argument("--per-device-train-batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=6)
    p.add_argument("--max-completion-length", type=int, default=4096)
    p.add_argument("--max-length", type=int, default=16384)
    p.add_argument("--gradient-checkpointing", type=_bool, default=True)
    p.add_argument(
        "--save-strategy",
        default="steps",
        help="Set explicitly by every validated reference config.",
    )
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=20)
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--dataset-num-proc", type=int, default=64)

    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument(
        "--lmbda",
        type=float,
        default=0.0,
        help="0 => off-policy (no vLLM); >0 => online.",
    )
    p.add_argument("--beta", type=float, default=0.0)
    p.add_argument("--use-liger-fused-jsd", type=_bool, default=False)
    p.add_argument(
        "--ce-coef",
        type=float,
        default=0.0,
        help="Additive CE anchor weight; 0 omits the key and leaves the objective pure.",
    )
    p.add_argument(
        "--log-student-entropy",
        type=_bool,
        default=False,
        help="Log mean student entropy and reverse KL every logging_steps.",
    )
    p.add_argument(
        "--entropy-guard-drop-frac",
        type=float,
        default=0.0,
        help="Stop when entropy falls this fraction below baseline; 0 disables the guard.",
    )
    p.add_argument("--entropy-guard-baseline-steps", type=int, default=20)
    p.add_argument("--entropy-guard-patience", type=int, default=3)
    p.add_argument(
        "--entropy-guard-action",
        default="stop",
        help="What a tripped guard does: stop the run at that step, or warn and carry "
        "on to max_steps. Use warn when the run's purpose is the collapse curve "
        "itself and a fixed export ladder needs every rung to exist.",
    )
    p.add_argument(
        "--lmbda-schedule",
        default="constant",
        help="constant, or linear to ramp lmbda from --lmbda-init up to --lmbda.",
    )
    p.add_argument("--lmbda-init", type=float, default=0.0)
    p.add_argument(
        "--min-completion-length",
        type=int,
        default=0,
        help="vLLM min_tokens for the student's rollout; >0 blocks the immediate-EOS "
        "mode collapse. On-policy only.",
    )
    p.add_argument(
        "--loss-arm",
        default="",
        help=(
            "distillation objective as a validated preset; one of: "
            + ", ".join(sorted(LOSS_ARMS))
            + ". EMPTY BY DEFAULT, which leaves the standalone loss flags authoritative and "
            "the rendered config unchanged. Naming an arm sets its switches and enforces its "
            "requirements, and disagreeing with a standalone flag is then an error"
        ),
    )
    p.add_argument(
        "--verify-trainer-accepts-keys",
        type=_bool,
        default=False,
        help=(
            "check every emitted key that upstream GOLDConfig does not define against the "
            "delivered CustomGOLDConfig, and refuse rather than letting TrlParser reject the "
            "config on a held allocation. Needs gb_steps_post_training importable."
        ),
    )
    p.add_argument(
        "--check-corpus-tokenizer",
        type=_bool,
        default=False,
        help=(
            "verify the student's tokenizer matches the one the corpus was built with. "
            "Requires gb_steps_post_training to be importable (delivered by code_config); "
            "errors rather than skipping if the package is absent."
        ),
    )
    p.add_argument("--response-template", default="<|im_start|>assistant")

    p.add_argument(
        "--vllm-num-servers",
        type=int,
        default=0,
        help=">0 emits the on-policy block and splits the nodes.",
    )
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--use-sampled-opd-loss", type=_bool, default=False)
    p.add_argument("--last-message-only", type=_bool, default=False)
    p.add_argument("--clip-alpha", type=float, default=0.1)
    p.add_argument("--opd-importance-sampling", type=_bool, default=False)
    # Read for VALIDATION only and never emitted. The trainer learns the server's
    # address from CLI flags on gold.py (--vllm_server_host / --vllm_server_port),
    # mirroring the one launcher that has actually run on-policy; putting them in
    # the config file instead would bet on them being accepted config-file keys,
    # and TRL's parse_args_and_config rejects unknown top-level keys outright.
    p.add_argument("--vllm-server-url", default="")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        # Before the config is built, so a tokenizer mismatch costs nothing: the alternative
        # is discovering it from a loss curve after the allocation has been held for hours.
        if args.check_corpus_tokenizer:
            check_corpus_tokenizer(args)
        config = build_config(args)
        # After build_config, because it is the EMITTED key set that matters: a key left at
        # its default is never emitted and so never needs the trainer to know it.
        if args.verify_trainer_accepts_keys:
            check_trainer_accepts(config)
    except ValueError as e:
        print(f"render_gold_config: {e}", file=sys.stderr)
        return 2
    # sort_keys=False keeps the emitted order stable and readable against the
    # reference configs; default_flow_style=False forces block style.
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
    print(
        f"render_gold_config: wrote {args.output} "
        f'({"on-policy" if args.vllm_num_servers > 0 else "off-policy"}, '
        f"{args.total_nodes} node(s))"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
