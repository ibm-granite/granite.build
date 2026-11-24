# SPDX-License-Identifier: Apache-2.0
"""The hub-kernel pins: what they pin, in what order they load, and that the loaders call them.

transformers 5.16+ binds the Mamba2 mixer functions when the granitemoehybrid modeling
module is imported, from whatever ``causal_conv1d`` / ``mamba_ssm`` package is
importable then, and falls back to slow PyTorch code (which raises on packed sequences)
otherwise. ``prepare_hub_kernels`` loads both kernels from the kernel-type Hub repos at
pinned commits (#763) and registers them under those package names first.

Two lessons from the transformers 5.8.0 era still hold and stay pinned here:

- LOAD ORDER. The model-type mamba-ssm builds imported ``causal_conv1d`` at module load;
  loading one before the legacy names existed bound ``causal_conv1d_cuda = None`` for good
  and every hybrid load died (student-sft build test, BlueVela, image 173af080,
  2026-11-20). causal-conv1d must be registered before mamba-ssm is loaded.
- REVISION AGREEMENT. Under ``HF_HUB_OFFLINE=1`` the trainer can only read the cache entry
  the preflight wrote (student-sft failed with ``OfflineModeIsEnabled`` on
  causal-conv1d/refs when the two disagreed; image 939685f2). gold-kernels.sh must fetch
  with the same ``kernel_spec`` the loader uses.

See ``hub_kernel_pins.py`` for the full reasoning. The loader tests run against injected
stand-ins for ``kernels`` and transformers, so they cover the logic in any environment;
two tests additionally read the real transformers when it is installed (the repo venv
does not ship it -- the step images do).
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
from gb_steps_post_training.distillation import hub_kernel_pins
from gb_steps_post_training.distillation.hub_kernel_pins import (
    PINNED_KERNEL_REVISIONS,
    kernel_spec,
    pinned_revision,
    prepare_hub_kernels,
)

_SRC = Path(__file__).resolve().parents[3]  # steps/train/distill/
_ENTRYPOINTS = [
    _SRC / "src" / "gb_steps_post_training" / "distillation" / "gold.py",
    _SRC / "src" / "gb_steps_post_training" / "distillation" / "sft.py",
    _SRC / "src" / "gb_steps_post_training" / "distillation" / "run_divergence.py",
    _SRC / "preflight-check" / "skypilot" / "src" / "distill_probe.py",
]
_SHELL = _SRC / "images" / "gpu" / "shell" / "gold-kernels.sh"
_MODELING = "transformers.models.granitemoehybrid.modeling_granitemoehybrid"


def test_pinned_revisions():
    # Commit SHAs in the kernel-type repos: a branch moves, a commit does not.
    # mamba-ssm: oldest v3 commit with a torch214-cxx11-cu130-x86_64-linux build.
    # causal-conv1d: oldest v1 commit with that build.
    assert (
        pinned_revision("kernels-community/mamba-ssm")
        == "20b2508ad12ae40260291539bf45183000451850"
    )
    assert (
        pinned_revision("kernels-community/causal-conv1d")
        == "adc40e4163fbac573adb44ea47b2c8d9080f13bc"
    )
    assert pinned_revision("kernels-community/flash-attn2") == "main"


@pytest.fixture()
def fake_fa_versions(monkeypatch):
    """A stand-in transformers.modeling_flash_attention_utils with its version table."""
    module = types.ModuleType("transformers.modeling_flash_attention_utils")
    module.FLASH_ATTN_KERNEL_VERSIONS = {"kernels-community/flash-attn2": 4}
    monkeypatch.setitem(
        sys.modules, "transformers.modeling_flash_attention_utils", module
    )
    return module


def test_kernel_spec_uses_the_pin(fake_fa_versions):
    for repo, sha in PINNED_KERNEL_REVISIONS.items():
        assert kernel_spec(repo) == {"revision": sha}


def test_kernel_spec_follows_transformers_for_unpinned_repos(fake_fa_versions):
    # flash-attn2 is loaded by transformers itself, at the major version it names;
    # the prefetch must fetch that, not main.
    assert kernel_spec("kernels-community/flash-attn2") == {"version": 4}
    assert kernel_spec("kernels-community/something-else") == {"revision": "main"}


def test_kernel_spec_without_transformers(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "transformers.modeling_flash_attention_utils", None
    )
    assert kernel_spec("kernels-community/flash-attn2") == {"revision": "main"}


@pytest.mark.parametrize("path", _ENTRYPOINTS, ids=lambda p: p.name)
def test_every_hybrid_model_entrypoint_calls_the_pin(path):
    """The registration only helps if it runs before the first model load.

    These four entrypoints instantiate hybrid (granitemoehybrid) models; the
    teacher-logits step loads only the dense teacher, and vLLM resolves its own
    kernels, so neither needs the call. A source check, not an import, because
    importing every entrypoint here would drag torch and trl into this suite.
    """
    src = path.read_text(encoding="utf-8")
    assert "prepare_hub_kernels()" in src, (
        f"{path.name} loads hybrid models but never prepares the hub kernels; "
        "add the prepare_hub_kernels() call (see hub_kernel_pins.py)"
    )


@pytest.fixture()
def fake_kernels(monkeypatch):
    """A stand-in ``kernels.get_kernel`` that records every load, in order.

    Returns hub-build stand-ins: causal-conv1d with the cpp_functions-style
    interface, mamba-ssm with the scan ops.
    """
    loaded = []
    iface = types.ModuleType("causal_conv1d_interface")
    iface.causal_conv1d_fwd_function = lambda *a: "fwd"
    iface.causal_conv1d_bwd_function = lambda *a: "bwd"
    cc1d = types.ModuleType("causal_conv1d_hub_abc123")
    cc1d.causal_conv1d_interface = iface
    cc1d.causal_conv1d_fn = lambda *a: "conv"
    mamba = types.ModuleType("mamba_ssm_hub_def456")
    mamba.mamba_chunk_scan_combined = lambda *a: "scan"

    def get_kernel(repo_id, **kw):
        # The load order is the regression: record what was registered at the time.
        loaded.append((repo_id, kw, "causal_conv1d" in sys.modules))
        return cc1d if repo_id.endswith("causal-conv1d") else mamba

    kernels_pkg = types.ModuleType("kernels")
    kernels_pkg.get_kernel = get_kernel
    monkeypatch.setitem(sys.modules, "kernels", kernels_pkg)
    # A stand-in torch that reports a CUDA device, so the loader runs on any test host.
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    for name in (
        "causal_conv1d",
        "causal_conv1d.causal_conv1d_interface",
        "mamba_ssm",
        _MODELING,
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)
    # Pinned repos never consult transformers; keep the real one out of the way.
    monkeypatch.setitem(
        sys.modules, "transformers.modeling_flash_attention_utils", None
    )
    yield loaded, cc1d, iface, mamba
    for name in ("causal_conv1d", "causal_conv1d.causal_conv1d_interface", "mamba_ssm"):
        sys.modules.pop(name, None)


def test_prepare_hub_kernels_registers_causal_conv1d_before_loading_mamba_ssm(
    fake_kernels,
):
    """The 173af080 regression, restated for the 5.16+ loader: causal-conv1d is loaded
    and registered first, mamba-ssm second, and nothing else is loaded."""
    loaded, _, _, _ = fake_kernels

    prepare_hub_kernels()

    assert [repo for repo, _, _ in loaded] == [
        "kernels-community/causal-conv1d",
        "kernels-community/mamba-ssm",
    ], f"unexpected load order or extra loads: {loaded}"
    # causal_conv1d was not registered when it was loaded, and was when mamba-ssm was.
    assert [registered for _, _, registered in loaded] == [False, True]


def test_prepare_hub_kernels_loads_the_pinned_revisions(fake_kernels):
    loaded, _, _, _ = fake_kernels

    prepare_hub_kernels()

    for repo, kw, _ in loaded:
        assert kw["revision"] == PINNED_KERNEL_REVISIONS[repo]
        assert "version" not in kw  # get_kernel refuses both


def test_prepare_hub_kernels_registers_the_package_names(fake_kernels):
    _, cc1d, _, mamba = fake_kernels

    prepare_hub_kernels()

    # transformers' fallback decorators import exactly these names.
    assert sys.modules["causal_conv1d"] is cc1d
    assert sys.modules["mamba_ssm"] is mamba


def test_prepare_hub_kernels_registers_the_legacy_causal_conv1d_names(fake_kernels):
    """Older modeling code and the model-type mamba-ssm builds asserted on the legacy
    causal_conv1d_cuda object; the hub build only has the new interface (measured
    2026-11-20, image 2990684b, student-sft build test)."""
    _, _, iface, _ = fake_kernels

    prepare_hub_kernels()

    assert sys.modules["causal_conv1d.causal_conv1d_interface"] is iface
    assert iface.causal_conv1d_cuda.causal_conv1d_fwd() == "fwd"
    assert iface.causal_conv1d_cuda.causal_conv1d_bwd() == "bwd"


def test_prepare_hub_kernels_is_idempotent(fake_kernels):
    loaded, _, _, _ = fake_kernels

    prepare_hub_kernels()
    prepare_hub_kernels()

    assert len(loaded) == 2


def test_prepare_hub_kernels_leaves_an_installed_package_alone(fake_kernels):
    loaded, _, _, _ = fake_kernels
    real = types.ModuleType("causal_conv1d")
    sys.modules["causal_conv1d"] = real

    prepare_hub_kernels()

    assert sys.modules["causal_conv1d"] is real
    assert [repo for repo, _, _ in loaded] == ["kernels-community/mamba-ssm"]


def test_prepare_hub_kernels_refuses_after_the_modeling_import(fake_kernels):
    """Too late is silent in transformers (the decorators already bound the PyTorch
    fallback), so it must be loud here."""
    loaded, _, _, _ = fake_kernels
    sys.modules[_MODELING] = types.ModuleType(_MODELING)
    try:
        with pytest.raises(RuntimeError, match="before prepare_hub_kernels"):
            prepare_hub_kernels()
        assert loaded == []
    finally:
        sys.modules.pop(_MODELING, None)


def test_prepare_hub_kernels_is_a_no_op_without_kernels(monkeypatch):
    """CPU-only entrypoints may run where kernels is absent; the call must not raise."""
    monkeypatch.setitem(sys.modules, "kernels", None)
    prepare_hub_kernels()  # must not raise


def test_the_real_transformers_imports_these_package_names():
    """The registration targets transformers' fallback imports by name. Read the real
    modeling source (without importing it, which would bind the decorators in this
    process) and check the names have not moved."""
    pytest.importorskip("transformers")
    spec = importlib.util.find_spec(_MODELING)
    assert spec is not None and spec.origin
    src = Path(spec.origin).read_text(encoding="utf-8")
    assert (
        'use_kernel_func_from_hub_with_fallback("causal_conv1d_fn", "causal_conv1d")'
        in src
    )
    assert (
        'use_kernel_func_from_hub_with_fallback("mamba_chunk_scan_combined", "mamba_ssm")'
        in src
    )
    # The 5.8.0-era mapping entries are gone, which is why the mapping rewrite was removed.
    from transformers.integrations import hub_kernels

    assert "mamba-ssm" not in getattr(hub_kernels, "_HUB_KERNEL_MAPPING", {})


def test_the_real_transformers_names_a_flash_attn2_version():
    """kernel_spec follows transformers for flash-attn2; that only works while
    transformers says which major version it loads."""
    pytest.importorskip("transformers")
    from transformers.modeling_flash_attention_utils import FLASH_ATTN_KERNEL_VERSIONS

    assert isinstance(
        FLASH_ATTN_KERNEL_VERSIONS.get("kernels-community/flash-attn2"), int
    )
    assert (
        hub_kernel_pins._transformers_kernel_version("kernels-community/flash-attn2")
        is not None
    )


def test_the_shell_preflight_reads_the_same_pin():
    """gold-kernels.sh probes and prefetches what the loader will resolve.

    If the shell fetched a different revision than the Python loader, the out-of-band
    prefetch would warm a cache entry the trainer never reads, and the first hybrid
    load would fail offline inside the allocation -- exactly what the preflight exists
    to prevent.
    """
    src = _SHELL.read_text(encoding="utf-8")
    assert "kernel_spec(repo)" in src
    assert 'revision="main"' not in src, (
        "gold-kernels.sh hardcodes revision=main; it must resolve through "
        "kernel_spec() so the prefetch matches the trainer's pin"
    )
    assert "kernels.utils" not in src, "kernels >= 0.14 has no kernels.utils module"


def test_prepare_hub_kernels_after_the_modeling_import_is_fine_once_registered(
    fake_kernels,
):
    """A second entrypoint module in the same process calls it again after a model load:
    nothing left to register, so nothing to refuse."""
    loaded, _, _, _ = fake_kernels
    prepare_hub_kernels()
    sys.modules[_MODELING] = types.ModuleType(_MODELING)
    try:
        prepare_hub_kernels()
        assert len(loaded) == 2
    finally:
        sys.modules.pop(_MODELING, None)


def test_prepare_hub_kernels_skips_without_cuda_unless_forced(
    fake_kernels, monkeypatch
):
    """CUDA kernels on a CPU host are useless; importing an entrypoint there (tests,
    preflights) must not fetch them. The kernel probe forces the load."""
    loaded, _, _, _ = fake_kernels
    sys.modules["torch"].cuda = types.SimpleNamespace(is_available=lambda: False)

    prepare_hub_kernels()
    assert loaded == []

    prepare_hub_kernels(force=True)
    assert [repo for repo, _, _ in loaded] == [
        "kernels-community/causal-conv1d",
        "kernels-community/mamba-ssm",
    ]
    assert all(kw["check_arch"] is False for _, kw, _ in loaded)
