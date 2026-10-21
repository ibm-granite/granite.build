"""Assert every monkeypatch target still exists, and report which patches are still needed.

WHY THIS EXISTS. `steps/distill-gold-train/pyproject.toml` pins nine runtime dependencies
with `==`, and the argument for exactness is that seven monkeypatches in `gold.py` and
`run_vllm_serve.py` reach into PRIVATE surface -- module-level flags, dunder-init methods,
free functions inside a script module. A resolver-chosen bump does not break those loudly. It
breaks them QUIETLY, in two different ways, and the two need opposite responses:

  DEAD    the symbol the patch reaches for is gone. `setattr` on a module that no longer has
          the attribute succeeds; a `getattr(..., None)` guard skips silently. The patch
          stops running and the bug it fixed comes back, wearing the original traceback.
          Upstream is already doing this: `trl/extras/vllm_client.py`, the consumer of the
          `_vllm_ascend_available` workaround, is DELETED on trl main.
  MOOT    the symbol is there and upstream fixed the bug, so the patch is now a no-op that
          nobody will ever delete because nobody knows it stopped mattering.

Both are invisible from a training run that succeeds. This module makes them visible: it
probes each target and reports EXISTS/MISSING (is the patch still applicable) alongside
NEEDED/MOOT (is the bug still there). Run it after any dependency bump, and read the two
columns as instructions -- MISSING means re-derive the patch against the new version, MOOT
means delete the patch and record why.

WHAT IT DELIBERATELY DOES NOT DO. It does not import `gold.py`, and it applies nothing. It
reads the same third-party surface the patches read, in the same environment, and reports.
That keeps it runnable in a container as a pre-flight without the side effect of half-patching
a process that then goes on to train.

Two of the seven are not probed here and say so in their rows: the `causal_conv1d` shim
(whose precondition is a warm HF kernel cache, not a symbol -- it is self-diagnosing, it was
written because the import fails) and the `granite_swa` registration (which is gated on an arm
this pairing does not use -- see `_swa_arm.py`).

    python -m gb_steps_post_training.distillation.patch_targets
    python -m gb_steps_post_training.distillation.patch_targets --skip-vllm   # no vLLM import

Exit status: 0 if every probed target EXISTS, 1 otherwise. A MOOT patch is reported and does
NOT fail -- a no-op patch is a cleanup task, not a broken run.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass, field


@dataclass
class Probe:
    name: str
    where: str          # where the patch lives, file:line, so the report is actionable
    exists: bool | None  # None = not probed
    needed: bool | None
    detail: str
    notes: str = ""


def _version(pkg: str) -> str:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return "?"


def probe_trl_ascend() -> Probe:
    """`trl.import_utils._vllm_ascend_available` must be a module attribute to unwrap.

    NEEDED when it is a TUPLE: under transformers >= 4.57 `_is_package_available` returns
    `(bool, version)` and `import_utils.py:43` stores it without `return_version=True`, so the
    tuple is truthy and `is_vllm_ascend_available()` reports an absent package as present.
    """
    m = importlib.import_module("trl.import_utils")
    if not hasattr(m, "_vllm_ascend_available"):
        return Probe("trl ascend guard", "gold.py:26-35 / run_vllm_serve.py:12-14",
                     False, None,
                     "trl.import_utils._vllm_ascend_available is GONE -- the unwrap patches "
                     "now setattr an attribute nothing reads. Re-derive: find what "
                     "is_vllm_ascend_available() reads in this version.")
    val = getattr(m, "_vllm_ascend_available")
    is_tuple = isinstance(val, tuple)
    return Probe("trl ascend guard", "gold.py:26-35 / run_vllm_serve.py:12-14",
                 True, is_tuple,
                 f"_vllm_ascend_available = {val!r} ({'tuple -> truthy, bug present' if is_tuple else 'not a tuple'})",
                 "trl main has DELETED trl/extras/vllm_client.py, the consumer this "
                 "workaround exists for. A bump can make this MOOT and the patch would "
                 "not tell you.")


def probe_vllm_rope(module: str, cls: str, where: str, label: str) -> Probe:
    """vLLM reads `config.rope_theta`; transformers 5.x moved it into `config.rope_parameters`.

    The patch wraps `__init__`, so the target is the class and its dunder. EXISTS is about the
    class still being there under that name; NEEDED is about transformers still keeping
    rope_theta inside rope_parameters rather than as a top-level attribute.
    """
    try:
        m = importlib.import_module(module)
    except Exception as exc:
        return Probe(label, where, False, None,
                     f"cannot import {module}: {type(exc).__name__}: {exc}")
    obj = getattr(m, cls, None)
    if obj is None:
        return Probe(label, where, False, None,
                     f"{module}.{cls} is GONE -- the __init__ wrapper patches a class that "
                     "no longer exists under that name, and vLLM will read config.rope_theta "
                     "unpatched. Re-derive against this vLLM version.")
    return Probe(label, where, True, None,
                 f"{module}.{cls}.__init__ present",
                 "NEEDED is not probed: it depends on the CONFIG of the model being served "
                 "(a granite-4.x config.json under transformers 5.x has no top-level "
                 "rope_theta). The patch is a no-op when the attribute is present, so it is "
                 "safe either way -- what is not safe is the class disappearing.")


def probe_deepspeed_defaults() -> Probe:
    """The ZeRO wrappers must exist to receive the `defaults` property.

    NEEDED when neither class defines `defaults`: transformers' `cosine_with_min_lr` reads
    `optimizer.defaults`, and a ZeRO wrapper that does not forward it raises mid-run, after
    the first scheduler step rather than at startup.
    """
    try:
        s3 = importlib.import_module("deepspeed.runtime.zero.stage3")
        s12 = importlib.import_module("deepspeed.runtime.zero.stage_1_and_2")
    except Exception as exc:
        return Probe("deepspeed defaults", "gold.py:73-80", False, None,
                     f"cannot import deepspeed zero modules: {type(exc).__name__}: {exc}")
    missing = [n for n, mod, cls in (
        ("DeepSpeedZeroOptimizer_Stage3", s3, "DeepSpeedZeroOptimizer_Stage3"),
        ("DeepSpeedZeroOptimizer", s12, "DeepSpeedZeroOptimizer"),
    ) if getattr(mod, cls, None) is None]
    if missing:
        return Probe("deepspeed defaults", "gold.py:73-80", False, None,
                     f"missing: {', '.join(missing)} -- re-derive against this deepspeed")
    needed = not any(
        hasattr(getattr(mod, cls), "defaults")
        for mod, cls in ((s3, "DeepSpeedZeroOptimizer_Stage3"), (s12, "DeepSpeedZeroOptimizer"))
    )
    return Probe("deepspeed defaults", "gold.py:73-80", True, needed,
                 "both ZeRO optimizer classes present; "
                 + ("neither exposes `defaults` -> patch needed"
                    if needed else "at least one now exposes `defaults` -> upstream may have fixed it"))


def probe_generation_aliases() -> Probe:
    """The mamba-ssm Hub kernel imports names transformers 5.x removed.

    EXISTS is about the replacement (`GenerateDecoderOnlyOutput`) still being importable --
    without it the alias patch cannot run at all. NEEDED is about the old names still being
    absent.
    """
    g = importlib.import_module("transformers.generation")
    have_new = hasattr(g, "GenerateDecoderOnlyOutput")
    have_old = hasattr(g, "GreedySearchDecoderOnlyOutput")
    if not have_new:
        return Probe("generation output aliases", "gold.py:82-91", False, None,
                     "transformers.generation.GenerateDecoderOnlyOutput is GONE -- the alias "
                     "patch has nothing to alias TO. Re-derive.")
    return Probe("generation output aliases", "gold.py:82-91", True, not have_old,
                 "GenerateDecoderOnlyOutput present; GreedySearchDecoderOnlyOutput "
                 + ("absent -> patch needed" if not have_old else "is BACK -> patch is moot"))


def probe_sanitize_logprob() -> Probe:
    """`trl.scripts.vllm_serve.sanitize_logprob` is replaced wholesale by name.

    This is the most fragile of the seven: a free function in a SCRIPT module, replaced rather
    than wrapped. If it is renamed or inlined, `setattr` quietly adds an unused attribute and
    a NaN logprob comes back as a Pydantic ResponseValidationError -> HTTP 500 mid-rollout.

    IT CANNOT BE PROBED IN ISOLATION, and the first run of this module, confirmed directly,
    is what established that: a cold `import trl.scripts.vllm_serve` raises
    ModuleNotFoundError: No module named 'vllm_ascend'. That is not this target being gone --
    it is PATCH 1 not having been applied yet. trl's `is_vllm_ascend_available()` reads the
    raw tuple `(False, None)`, which is truthy, so the module's import-time ascend branch
    fires and tries to import a package that is not installed.

    So patch 1 is a hard PRECONDITION for patch 3's target even existing, and the ordering in
    run_vllm_serve.py (line 14 neutralizes the guard, line 55 imports the module) is
    load-bearing rather than incidental. Nothing in that file said so; reordering the patch
    blocks, or splitting them across modules, would resurrect this failure as a confusing
    "No module named 'vllm_ascend'" from a file that never mentions ascend. This probe
    reproduces the real ordering deliberately, and REPORTS the dependency instead of hiding
    it behind a working import.
    """
    label, where = "sanitize_logprob NaN", "run_vllm_serve.py:51-64"
    ordering = ""
    try:
        m = importlib.import_module("trl.scripts.vllm_serve")
    except ModuleNotFoundError as exc:
        if "vllm_ascend" not in str(exc):
            return Probe(label, where, False, None,
                         f"cannot import trl.scripts.vllm_serve: {type(exc).__name__}: {exc}")
        # The documented precondition. Apply patch 1 exactly as run_vllm_serve.py:14 does,
        # then retry -- and say so in the report, because a silent retry would turn a
        # load-bearing ordering into an invisible one.
        importlib.import_module("trl.import_utils")._vllm_ascend_available = False
        ordering = ("cold import raised ModuleNotFoundError('vllm_ascend') -- patch 1 is a "
                    "PRECONDITION for this target; applied it and retried (run_vllm_serve.py:14 "
                    "before :55 does the same, and that order is load-bearing). ")
        try:
            m = importlib.import_module("trl.scripts.vllm_serve")
        except Exception as exc2:
            return Probe(label, where, False, None,
                         ordering + f"still cannot import: {type(exc2).__name__}: {exc2}")
    except Exception as exc:
        return Probe(label, where, False, None,
                     f"cannot import trl.scripts.vllm_serve: {type(exc).__name__}: {exc}")
    fn = getattr(m, "sanitize_logprob", None)
    if fn is None:
        return Probe(label, where, False, None,
                     ordering + "trl.scripts.vllm_serve.sanitize_logprob is GONE -- the "
                     "replacement is a dead attribute and NaN logprobs return HTTP 500 again. "
                     "Re-derive.")
    # Whether it still mishandles NaN is checked by calling it, which is cheap and exact.
    needed = None
    detail = ordering + "sanitize_logprob present"
    try:
        class _L:
            logprob = float("nan")
        out = fn(_L())
        needed = out is None
        detail += f"; on a NaN logprob it returns {out!r} -> " + (
            "None, rejected by the list[list[float]] response schema -> patch needed"
            if needed else "a value the schema accepts -> patch may be moot")
    except Exception as exc:  # signature changed
        detail += f"; could not call it with a stub logprob ({type(exc).__name__}) -- the "
        detail += "signature changed, re-check the patch"
    return Probe(label, where, True, needed, detail)


def unprobed() -> list[Probe]:
    return [
        Probe("causal_conv1d / mamba-ssm shim", "gold.py:103-209", None, None,
              "NOT PROBED, and deliberately: its precondition is a warm HF kernel cache and a "
              "specific `kernels` layout, not a symbol. It is self-diagnosing -- it exists "
              "because the import raises, so a break is loud on the next run."),
        Probe("granite_swa Auto*/vLLM registration", "gold.py:100-101, 211-219 / "
              "run_vllm_serve.py:66-75", None, None,
              "NOT PROBED: gated on the SWA arm, which nothing in the granite 4.1/4.2 pairing "
              "uses (D4). Probing it would require the vendored package this collection does "
              "not carry. See _swa_arm.py."),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-vllm", action="store_true",
                    help="skip the two vLLM probes (importing vllm is slow and pulls CUDA)")
    a = ap.parse_args()

    print("environment")
    for pkg in ("trl", "transformers", "vllm", "deepspeed", "accelerate", "torch"):
        print(f"  {pkg:<14} {_version(pkg)}")
    print(f"  {'python':<14} {sys.version.split()[0]}")
    print()

    # ORDER IS LOAD-BEARING HERE TOO, for the same reason it is in run_vllm_serve.py:
    # probe_sanitize_logprob() has to SET _vllm_ascend_available = False to import its target
    # at all (see its docstring), so probe_trl_ascend() must run FIRST or it would read the
    # value this module just wrote and report the bug as already fixed.
    probes = [probe_trl_ascend()]
    if a.skip_vllm:
        probes.append(Probe("vLLM rope_theta (x2)", "gold.py:37-71 / run_vllm_serve.py:16-49",
                            None, None, "SKIPPED via --skip-vllm"))
    else:
        probes.append(probe_vllm_rope("vllm.model_executor.models.granitemoehybrid",
                                      "GraniteMoeHybridAttention",
                                      "gold.py:37-53 / run_vllm_serve.py:16-32",
                                      "vLLM rope_theta (MoeHybrid)"))
        probes.append(probe_vllm_rope("vllm.model_executor.models.granite",
                                      "GraniteDecoderLayer",
                                      "gold.py:55-71 / run_vllm_serve.py:34-49",
                                      "vLLM rope_theta (dense Granite)"))
    probes.append(probe_deepspeed_defaults())
    probes.append(probe_generation_aliases())
    probes.append(probe_sanitize_logprob())
    probes += unprobed()

    def flag(v: bool | None, yes: str, no: str) -> str:
        return "  -  " if v is None else (yes if v else no)

    failed = []
    for p in probes:
        print(f"{flag(p.exists, 'EXISTS', 'MISSING')}  "
              f"{flag(p.needed, 'NEEDED', ' MOOT ')}  {p.name}")
        print(f"                    at {p.where}")
        print(f"                    {p.detail}")
        if p.notes:
            print(f"                    note: {p.notes}")
        print()
        if p.exists is False:
            failed.append(p.name)

    if failed:
        print(f"FAILED: {len(failed)} patch target(s) MISSING -- a patch is now a silent "
              f"no-op and the bug it fixed is back:")
        for n in failed:
            print(f"  - {n}")
        print("Do not bump past this. Re-derive the patch, or delete it and record why.")
        return 1
    moot = [p.name for p in probes if p.needed is False]
    print("ALL PROBED TARGETS EXIST")
    if moot:
        print(f"but {len(moot)} patch(es) look MOOT in this environment -- upstream may have "
              "fixed the bug. Verify, then delete the patch and record it:")
        for n in moot:
            print(f"  - {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
