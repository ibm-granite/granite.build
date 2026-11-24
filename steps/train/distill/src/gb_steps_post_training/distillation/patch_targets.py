# SPDX-License-Identifier: Apache-2.0

"""Assert every private upstream surface this code patches or overrides still exists, and
report which workarounds are still needed.

WHY THIS EXISTS. The image pins trl, transformers, vLLM, DeepSpeed and kernels with `==`
(images/gpu/pyproject.toml) because this code reaches into their PRIVATE surface: it
monkeypatches attributes, overrides private trainer methods, and registers modules under
names transformers imports. A version bump does not break those loudly. It breaks them
QUIETLY, in two ways that need opposite responses:

  DEAD    the symbol a patch or override reaches for is gone. `setattr` on a module that no
          longer has the attribute succeeds; an override of a method the parent no longer
          calls is never called. The workaround stops running and the bug it fixed comes
          back, wearing the original traceback.
  MOOT    the symbol is there and upstream fixed the bug, so the workaround is a no-op that
          nobody will delete because nobody knows it stopped mattering.

Both are invisible from a training run that succeeds. This module makes them visible: it
probes each target and reports EXISTS/MISSING (is the workaround still applicable) alongside
NEEDED/MOOT (is the upstream behaviour it works around still there). Run it after any
dependency bump: MISSING means re-derive the workaround against the new version, MOOT means
delete it and record why.

THE vLLM 0.34 / trl 1.18 / transformers 5.22 MOVE (#676) DELETED FOUR PATCHES, recorded here
so the reasons outlive the code:

  1  trl 0.26.2 stored `_is_package_available()`'s (bool, version) tuple as
     `_vllm_ascend_available`, which is always truthy. trl 1.x's `import_utils` has its own
     `_is_package_available` returning a bool and no ascend guard at all.
  2  vLLM 0.11.0's `GraniteMoeHybridAttention` / `GraniteDecoderLayer` read
  2b `config.rope_theta`, which transformers 5.x moved into `config.rope_parameters`.
     vLLM 0.34.0 passes `config.rope_parameters` itself. A source check below guards that,
     so the patch's return would be noticed.
  3  trl 0.26.2's `vllm_serve.sanitize_logprob` returned None for a NaN logprob, which its
     response schema rejected (HTTP 500). trl 1.x serves through vLLM's own server; its
     client returns None for NaN, and the trainer masks that token out of its
     importance-sampling terms.

DELETED rows below are those guards: they fail (like MISSING) if the bug comes BACK, because
there is no longer a patch to catch it.

WHAT IT DELIBERATELY DOES NOT DO. It does not import gold.py or the trainers, and it applies
nothing. The new probes read upstream SOURCE, located with `importlib.util.find_spec`, which
executes nothing below the top-level package: neither trl's trainer modules nor transformers'
granitemoehybrid modeling module are imported -- the latter matters, because importing it
binds its kernel decorators for the rest of the process (see hub_kernel_pins.py). Only the
DeepSpeed and transformers.generation probes import their targets, as before.

    python -m gb_steps_post_training.distillation.patch_targets

Exit status: 0 if every probed target EXISTS and no deleted patch's bug is back, 1 otherwise.
A MOOT workaround is reported and does NOT fail -- a no-op is a cleanup task, not a broken run.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Probe:
    name: str
    where: str  # where the workaround lives, so the report is actionable
    exists: bool | None  # None = not probed
    needed: bool | None
    detail: str
    notes: str = ""
    deleted: bool = (
        False  # the workaround was removed; needed=True means the bug is back
    )


def _version(pkg: str) -> str:
    try:
        from importlib.metadata import version

        return version(pkg)
    except Exception:
        return "?"


def _source_path(package: str, relative: str) -> Path | None:
    """Path of `relative` inside the installed top-level `package`, without importing it.

    find_spec on a TOP-LEVEL name locates the package without executing it; a dotted name
    would import every parent package first.
    """
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        return None
    path = Path(next(iter(spec.submodule_search_locations))) / relative
    return path if path.is_file() else None


def _classes(path: Path) -> dict[str, ast.ClassDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}


def _methods(cls: ast.ClassDef) -> dict[str, ast.FunctionDef]:
    return {
        n.name: n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _self_assignments(cls: ast.ClassDef) -> set[str]:
    return {
        n.attr
        for n in ast.walk(cls)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Name)
        and n.value.id == "self"
        and isinstance(n.ctx, ast.Store)
    }


def _missing_file(name: str, where: str, package: str, relative: str) -> Probe:
    return Probe(
        name,
        where,
        False,
        None,
        f"{package}/{relative} not found -- the module moved; re-derive against "
        f"{package} {_version(package)}",
    )


# --- kept workarounds -------------------------------------------------------------------


def probe_deepspeed_defaults() -> Probe:
    """The ZeRO wrappers must exist to receive the `defaults` property.

    NEEDED when neither class defines `defaults`: transformers' `cosine_with_min_lr` reads
    `optimizer.defaults`, and a ZeRO wrapper that does not forward it raises mid-run, after
    the first scheduler step rather than at startup.
    """
    where = "gold.py / sft.py (deepspeed `defaults` property)"
    try:
        s3 = importlib.import_module("deepspeed.runtime.zero.stage3")
        s12 = importlib.import_module("deepspeed.runtime.zero.stage_1_and_2")
    except Exception as exc:
        return Probe(
            "deepspeed defaults",
            where,
            False,
            None,
            f"cannot import deepspeed zero modules: {type(exc).__name__}: {exc}",
        )
    classes = (
        (s3, "DeepSpeedZeroOptimizer_Stage3"),
        (s12, "DeepSpeedZeroOptimizer"),
    )
    missing = [cls for mod, cls in classes if getattr(mod, cls, None) is None]
    if missing:
        return Probe(
            "deepspeed defaults",
            where,
            False,
            None,
            f"missing: {', '.join(missing)} -- re-derive against this deepspeed",
        )
    needed = not any(hasattr(getattr(mod, cls), "defaults") for mod, cls in classes)
    return Probe(
        "deepspeed defaults",
        where,
        True,
        needed,
        "both ZeRO optimizer classes present; "
        + (
            "neither exposes `defaults` -> patch needed"
            if needed
            else "at least one now exposes `defaults` -> upstream may have fixed it"
        ),
    )


def probe_generation_aliases() -> Probe:
    """The mamba-ssm Hub kernel's `decode()` names classes transformers 5.x removed.

    EXISTS is about the replacement (`GenerateDecoderOnlyOutput`) still being importable --
    without it the alias patch cannot run at all. NEEDED is about the old names still being
    absent.
    """
    where = "gold.py (generation-output aliases)"
    g = importlib.import_module("transformers.generation")
    have_new = hasattr(g, "GenerateDecoderOnlyOutput")
    have_old = hasattr(g, "GreedySearchDecoderOnlyOutput")
    if not have_new:
        return Probe(
            "generation output aliases",
            where,
            False,
            None,
            "transformers.generation.GenerateDecoderOnlyOutput is GONE -- the alias "
            "patch has nothing to alias TO. Re-derive.",
        )
    return Probe(
        "generation output aliases",
        where,
        True,
        not have_old,
        "GenerateDecoderOnlyOutput present; GreedySearchDecoderOnlyOutput "
        + ("absent -> patch needed" if not have_old else "is BACK -> patch is moot"),
    )


# --- deleted workarounds, guarded ---------------------------------------------------------


def probe_vllm_rope_parameters() -> list[Probe]:
    """Patches 2/2b were deleted because vLLM reads `config.rope_parameters` itself.

    Guarded per model FILE, not per class: the patches wrapped
    `GraniteMoeHybridAttention.__init__` and `GraniteDecoderLayer.__init__`, but vLLM moves
    the rope set-up between classes (0.34 does it in `GraniteAttention` via a module
    helper). The bug is back if a module reads a top-level `config.rope_theta` -- which a
    transformers 5.x Granite config does not have -- or no longer reads
    `config.rope_parameters` at all.
    """
    probes = []
    for relative, label in (
        (
            "model_executor/models/granitemoehybrid.py",
            "vLLM rope_parameters (MoeHybrid)",
        ),
        ("model_executor/models/granite.py", "vLLM rope_parameters (dense Granite)"),
    ):
        where = "deleted from gold.py / run_vllm_serve.py (patch 2/2b)"
        path = _source_path("vllm", relative)
        if path is None:
            probes.append(_missing_file(label, where, "vllm", relative))
            continue
        source = path.read_text(encoding="utf-8")
        reads_theta = "config.rope_theta" in source
        reads_params = "config.rope_parameters" in source
        back = reads_theta or not reads_params
        probes.append(
            Probe(
                label,
                where,
                True,
                back,
                (
                    f"vllm {relative} "
                    + (
                        "reads config.rope_theta"
                        if reads_theta
                        else "never reads config.rope_parameters"
                    )
                    + " -> the deleted patch is NEEDED again"
                    if back
                    else f"vllm {relative} reads config.rope_parameters and never config.rope_theta "
                    "-> the deleted patch stays deleted"
                ),
                deleted=True,
            )
        )
    return probes


def probe_trl_bool_guards() -> Probe:
    """Patch 1 was deleted because trl 1.x's `_is_package_available` returns a bool."""
    where = "deleted from gold.py / sft.py / run_vllm_serve.py / render_gold_config.py (patch 1)"
    label = "trl package-availability guards"
    m = importlib.import_module("trl.import_utils")
    tuples = [
        name
        for name in dir(m)
        if name.startswith("_")
        and name.endswith("_available")
        and isinstance(getattr(m, name), tuple)
    ]
    return Probe(
        label,
        where,
        True,
        bool(tuples),
        (
            f"tuple-valued guards are BACK: {', '.join(tuples)} -> the deleted unwrap is "
            "NEEDED again"
            if tuples
            else "no module-level *_available guard is a tuple -> the deleted unwrap stays deleted"
        ),
        deleted=True,
    )


# --- surfaces the trl 1.x port depends on --------------------------------------------------


# CustomGOLDTrainer overrides these back to transformers.Trainer / SFTTrainer behaviour,
# because trl 1.x's GOLDTrainer replaced them with its buffered, RepeatSampler-driven
# generation (#676 D-1). NEEDED while GOLDTrainer still defines them.
_GOLD_OVERRIDDEN = (
    "_prepare_inputs",
    "get_train_dataloader",
    "_get_train_sampler",
    "get_batch_samples",
)


def probe_gold_trainer_overrides() -> Probe:
    label, where = (
        "GOLDTrainer data/generation path",
        "custom_gold_trainer.py (D-1 overrides)",
    )
    relative = "experimental/gold/gold_trainer.py"
    path = _source_path("trl", relative)
    if path is None:
        return _missing_file(label, where, "trl", relative)
    cls = _classes(path).get("GOLDTrainer")
    if cls is None:
        return Probe(label, where, False, None, f"trl {relative}: GOLDTrainer is gone")
    defined = set(_methods(cls))
    overridden = [m for m in _GOLD_OVERRIDDEN if m in defined]
    return Probe(
        label,
        where,
        True,
        bool(overridden),
        (
            f"GOLDTrainer defines {', '.join(overridden)} -> restoring Trainer behaviour is needed"
            if overridden
            else "GOLDTrainer no longer overrides any of "
            + ", ".join(_GOLD_OVERRIDDEN)
            + " -> the restoring overrides look moot"
        ),
    )


# What CustomULDLoss reads from its trl parent instead of defining itself.
_ULD_METHODS = (
    "__init__",
    "__call__",
    "_initialize_vocabulary_mapping",
    "_compute_distillation_loss",
    "_compute_jsd_loss_for_matched_tokens",
    "_get_start_and_size_answers",
)
_ULD_ATTRIBUTES = (
    "_vocab_mapping",
    "_student_matched_ids",
    "_teacher_matched_ids",
    "beta",
    "distillation_weight",
    "hybrid_matched_weight",
    "hybrid_unmatched_weight",
    "ignore_index",
    "student_temperature",
    "teacher_temperature",
    "student_tokenizer",
    "teacher_tokenizer",
    "use_extended_uld",
    "use_hybrid_loss",
)


def probe_uld_loss_surface() -> Probe:
    label, where = "ULDLoss parent surface", "custom_gold_trainer.py (CustomULDLoss)"
    relative = "experimental/gold/gold_trainer.py"
    path = _source_path("trl", relative)
    if path is None:
        return _missing_file(label, where, "trl", relative)
    cls = _classes(path).get("ULDLoss")
    if cls is None:
        return Probe(label, where, False, None, f"trl {relative}: ULDLoss is gone")
    methods = set(_methods(cls))
    assigned = _self_assignments(cls)
    missing = [m for m in _ULD_METHODS if m not in methods] + [
        a for a in _ULD_ATTRIBUTES if a not in assigned
    ]
    if missing:
        return Probe(
            label,
            where,
            False,
            None,
            f"ULDLoss no longer defines/sets: {', '.join(missing)} -- CustomULDLoss reads them",
        )
    return Probe(
        label,
        where,
        True,
        None,
        f"all {len(_ULD_METHODS)} methods and {len(_ULD_ATTRIBUTES)} attributes CustomULDLoss "
        "reads from its parent are still defined",
    )


# (class, method, required parameters) of trl's vLLM client side the trainer calls.
_VLLM_CLIENT_API = (
    (
        "vllm_client.py",
        "VLLMClient",
        "generate",
        ("prompts", "logprobs", "structured_outputs_regex", "generation_kwargs"),
    ),
    ("vllm_client.py", "VLLMClient", "init_communicator", ("device",)),
    ("vllm_client.py", "VLLMClient", "weight_update", ()),
    (
        "vllm_client.py",
        "VLLMClient",
        "update_named_params",
        ("metadata", "named_params"),
    ),
    ("vllm_client.py", "VLLMClient", "reset_prefix_cache", ()),
    ("vllm_generation.py", "VLLMGeneration", "sync_weights", ()),
)


def probe_vllm_client_api() -> Probe:
    label, where = (
        "trl vLLM client API",
        "custom_gold_trainer.py (MultiVLLMClient, rollouts, weight sync)",
    )
    problems = []
    cache: dict[str, dict[str, ast.ClassDef]] = {}
    for filename, cls_name, method, params in _VLLM_CLIENT_API:
        relative = f"generation/{filename}"
        if relative not in cache:
            path = _source_path("trl", relative)
            if path is None:
                return _missing_file(label, where, "trl", relative)
            cache[relative] = _classes(path)
        cls = cache[relative].get(cls_name)
        if cls is None:
            problems.append(f"{cls_name} gone")
            continue
        fn = _methods(cls).get(method)
        if fn is None:
            problems.append(f"{cls_name}.{method} gone")
            continue
        args = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
        lost = [p for p in params if p not in args]
        if lost:
            problems.append(f"{cls_name}.{method} lost parameter(s) {', '.join(lost)}")
    if problems:
        return Probe(label, where, False, None, "; ".join(problems))
    return Probe(
        label,
        where,
        True,
        None,
        f"{len(_VLLM_CLIENT_API)} methods present with the parameters the trainer passes",
    )


# transformers' granitemoehybrid modeling binds these at import; hub_kernel_pins registers
# the pinned hub builds under the package names first.
_KERNEL_DECORATIONS = (
    (
        'use_kernel_func_from_hub_with_fallback("causal_conv1d_fn", "causal_conv1d")',
        "causal_conv1d",
    ),
    (
        'use_kernel_func_from_hub_with_fallback("causal_conv1d_update", "causal_conv1d")',
        "causal_conv1d",
    ),
    (
        'use_kernel_func_from_hub_with_fallback("mamba_chunk_scan_combined", "mamba_ssm")',
        "mamba_ssm",
    ),
    (
        'use_kernel_func_from_hub_with_fallback("mamba_split_conv1d_scan_combined", "mamba_ssm")',
        "mamba_ssm",
    ),
    (
        'use_kernel_func_from_hub_with_fallback("selective_state_update", "mamba_ssm")',
        "mamba_ssm",
    ),
)


def probe_hub_kernel_registration() -> Probe:
    """hub_kernel_pins.prepare_hub_kernels registers `causal_conv1d` / `mamba_ssm`.

    That only helps while transformers' granitemoehybrid modeling looks the kernels up
    under those package names. Read from source: importing the modeling module would bind
    its decorators in this process.
    """
    label, where = (
        "hub-kernel registration targets",
        "hub_kernel_pins.py (prepare_hub_kernels)",
    )
    relative = "models/granitemoehybrid/modeling_granitemoehybrid.py"
    path = _source_path("transformers", relative)
    if path is None:
        return _missing_file(label, where, "transformers", relative)
    source = path.read_text(encoding="utf-8")
    absent = [
        decoration for decoration, _ in _KERNEL_DECORATIONS if decoration not in source
    ]
    if absent:
        legacy = "lazy_load_kernel(" in source
        return Probe(
            label,
            where,
            False,
            None,
            f"{len(absent)} of {len(_KERNEL_DECORATIONS)} kernel lookups changed "
            f"(first: {absent[0]})"
            + ("; the modeling code calls lazy_load_kernel again" if legacy else ""),
        )
    return Probe(
        label,
        where,
        True,
        True,
        "the modeling code binds causal_conv1d / mamba_ssm at import -> registration needed",
    )


def main(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)

    print("environment")
    for pkg in (
        "trl",
        "transformers",
        "vllm",
        "deepspeed",
        "accelerate",
        "kernels",
        "torch",
    ):
        print(f"  {pkg:<14} {_version(pkg)}")
    print(f"  {'python':<14} {sys.version.split()[0]}")
    print()

    probes = [
        probe_deepspeed_defaults(),
        probe_generation_aliases(),
        probe_trl_bool_guards(),
        *probe_vllm_rope_parameters(),
        probe_gold_trainer_overrides(),
        probe_uld_loss_surface(),
        probe_vllm_client_api(),
        probe_hub_kernel_registration(),
    ]

    def flag(v: bool | None, yes: str, no: str) -> str:
        return "  -  " if v is None else (yes if v else no)

    failed = []
    for p in probes:
        needed = flag(p.needed, "BACK  " if p.deleted else "NEEDED", " MOOT ")
        print(f"{flag(p.exists, 'EXISTS', 'MISSING')}  {needed}  {p.name}")
        print(f"                    at {p.where}")
        print(f"                    {p.detail}")
        if p.notes:
            print(f"                    note: {p.notes}")
        print()
        if p.exists is False:
            failed.append(f"{p.name} (target MISSING)")
        elif p.deleted and p.needed:
            failed.append(f"{p.name} (deleted patch's bug is BACK)")

    if failed:
        print(f"FAILED: {len(failed)} workaround target(s) broken:")
        for n in failed:
            print(f"  - {n}")
        print(
            "Do not bump past this. Re-derive the workaround, or delete it and record why."
        )
        return 1
    moot = [p.name for p in probes if p.needed is False and not p.deleted]
    print("ALL PROBED TARGETS EXIST")
    if moot:
        print(
            f"but {len(moot)} workaround(s) look MOOT in this environment -- upstream may have "
            "fixed the bug. Verify, then delete the workaround and record it:"
        )
        for n in moot:
            print(f"  - {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
