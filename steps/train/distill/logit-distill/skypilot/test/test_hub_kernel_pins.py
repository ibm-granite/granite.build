# SPDX-License-Identifier: Apache-2.0
"""The hub-kernel pins: what they pin, and that the loaders call them.

mamba-ssm is pinned because transformers 5.8.0's default mapping resolves it at
``version=1`` -- the kernels-community repo's v1 branch -- whose build imports
``GreedySearchDecoderOnlyOutput``, a symbol transformers 5.x renamed. Every fresh
hybrid (granitemoehybrid) model load dies on that import mid-load. Every revision
up to and including v0.0.3 has it (measured 2026-11-20 on BlueVela, image 91cb624c,
teacher-divergence build test; an earlier reading that v0.0.3 was clean was wrong,
see c6f3a799). ``v0.0.4`` was the compatible revision under torch 2.8 / cu128. The
broken import is per build variant: the torch 2.11 / CUDA 13 image (#676) pins
7e8bf9ea, the oldest v1-branch commit with a torch211-cxx11-cu130 build, whose
build imports the transformers 5.x name.

causal-conv1d is pinned for revision AGREEMENT: transformers resolves it at the
v1 branch while the preflight's unpinned fallback is ``main``, and under
``HF_HUB_OFFLINE=1`` the trainer cannot resolve the uncached ref (measured
2026-11-20, image 939685f2: student-sft failed with ``OfflineModeIsEnabled`` on
causal-conv1d/refs seconds after the preflight downloaded the repo at ``main``).

See ``hub_kernel_pins.py`` for the full measurements.

The mapping tests run against an injected stand-in module so they cover the pin's
logic in any environment; one test additionally runs against the real transformers
when it is installed (the repo venv does not ship it -- the step images do).
"""

import sys
import types
from pathlib import Path

import pytest
from gb_steps_post_training.distillation.hub_kernel_pins import (
    PINNED_KERNEL_REVISIONS,
    pin_hub_kernels,
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


def _default_mapping() -> dict:
    """A stand-in for transformers 5.8.0's default mapping (verbatim shapes)."""
    return {
        "causal-conv1d": {"repo_id": "kernels-community/causal-conv1d", "version": 1},
        "mamba-ssm": {"repo_id": "kernels-community/mamba-ssm", "version": 1},
        "falcon_mamba-ssm": {"repo_id": "kernels-community/mamba-ssm", "version": 1},
        "sonic-moe": {
            "repo_id": "kernels-community/sonic-moe",
            "revision": "ep-support",
        },
    }


@pytest.fixture()
def fake_mapping(monkeypatch):
    """Inject a stand-in transformers.integrations.hub_kernels and return its mapping."""
    module = types.ModuleType("transformers.integrations.hub_kernels")
    module._HUB_KERNEL_MAPPING = _default_mapping()
    # The pin imports `from transformers.integrations import hub_kernels`, so the
    # stand-in needs its parent packages in sys.modules too, or that import raises
    # ImportError and the pin no-ops.
    transformers = types.ModuleType("transformers")
    integrations = types.ModuleType("transformers.integrations")
    integrations.hub_kernels = module
    transformers.integrations = integrations
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.integrations", integrations)
    monkeypatch.setitem(sys.modules, "transformers.integrations.hub_kernels", module)
    return module._HUB_KERNEL_MAPPING


def test_pin_rewrites_mamba_ssm_onto_the_pinned_revision(fake_mapping):
    assert fake_mapping["mamba-ssm"].get("version") == 1  # the broken default

    pin_hub_kernels()

    entry = fake_mapping["mamba-ssm"]
    assert entry["revision"] == PINNED_KERNEL_REVISIONS["kernels-community/mamba-ssm"]
    # version must be GONE, not just overridden: get_kernel refuses a call that
    # passes both revision and version.
    assert "version" not in entry


def test_pin_covers_the_falcon_variant_of_the_same_repo(fake_mapping):
    pin_hub_kernels()

    entry = fake_mapping["falcon_mamba-ssm"]
    assert entry["revision"] == PINNED_KERNEL_REVISIONS["kernels-community/mamba-ssm"]
    assert "version" not in entry


def test_pin_rewrites_causal_conv1d_onto_the_pinned_revision(fake_mapping):
    assert fake_mapping["causal-conv1d"].get("version") == 1  # the v1-branch default

    pin_hub_kernels()

    entry = fake_mapping["causal-conv1d"]
    assert (
        entry["revision"] == PINNED_KERNEL_REVISIONS["kernels-community/causal-conv1d"]
    )
    # version must be GONE for the same reason as mamba-ssm: get_kernel refuses
    # a call that passes both revision and version.
    assert "version" not in entry


def test_pin_leaves_unpinned_kernels_alone(fake_mapping):
    pin_hub_kernels()

    assert fake_mapping["sonic-moe"] == {
        "repo_id": "kernels-community/sonic-moe",
        "revision": "ep-support",
    }


def test_pin_is_idempotent(fake_mapping):
    pin_hub_kernels()
    first = dict(fake_mapping["mamba-ssm"])
    pin_hub_kernels()
    assert fake_mapping["mamba-ssm"] == first


def test_pin_respects_a_revision_transformers_set_itself(fake_mapping):
    fake_mapping["mamba-ssm"]["revision"] = "their-own-fix"

    pin_hub_kernels()

    # setdefault, not assignment: a transformers that pins its own revision has
    # done the work, and this pin must not fight it.
    assert fake_mapping["mamba-ssm"]["revision"] == "their-own-fix"


def test_pin_is_a_no_op_without_transformers(monkeypatch):
    """CPU-only entrypoints may run where transformers is absent; the pin must not raise."""
    monkeypatch.setitem(sys.modules, "transformers.integrations.hub_kernels", None)
    pin_hub_kernels()  # must not raise


def test_pin_against_the_real_transformers():
    """The same assertions against the installed transformers, when there is one.

    The repo venv does not ship transformers (the step images do); this runs on a
    machine that has it and skips elsewhere.
    """
    pytest.importorskip("transformers")
    from transformers.integrations import hub_kernels

    before = {k: dict(v) for k, v in hub_kernels._HUB_KERNEL_MAPPING.items()}
    try:
        pin_hub_kernels()
        entry = hub_kernels._HUB_KERNEL_MAPPING["mamba-ssm"]
        assert (
            entry.get("revision")
            == PINNED_KERNEL_REVISIONS["kernels-community/mamba-ssm"]
        )
        assert "version" not in entry
        entry = hub_kernels._HUB_KERNEL_MAPPING["causal-conv1d"]
        assert (
            entry.get("revision")
            == PINNED_KERNEL_REVISIONS["kernels-community/causal-conv1d"]
        )
        assert "version" not in entry
    finally:
        hub_kernels._HUB_KERNEL_MAPPING.clear()
        hub_kernels._HUB_KERNEL_MAPPING.update(before)


def test_pinned_revision_helper():
    # Both pinned by commit SHA: a branch or tag can move. mamba-ssm's is the oldest
    # v1-branch commit with a torch211-cxx11-cu130-x86_64-linux build.
    assert (
        pinned_revision("kernels-community/mamba-ssm")
        == "7e8bf9ea951cf17e161bd82e44ecf2d95d8bc484"
    )
    assert (
        pinned_revision("kernels-community/causal-conv1d")
        == "6e9a58272220a363dc50e59cba737abcedd7b59f"
    )
    # Unlisted repos keep their default resolution.
    assert pinned_revision("kernels-community/flash-attn2") == "main"


@pytest.mark.parametrize("path", _ENTRYPOINTS, ids=lambda p: p.name)
def test_every_hybrid_model_entrypoint_calls_the_pin(path):
    """The pin (and the legacy-name shim around it) only helps if it runs before
    the first model load.

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
def fake_lazy_loader(fake_mapping, monkeypatch):
    """A stand-in lazy_load_kernel on the stand-in hub_kernels module.

    Records which kernels prepare_hub_kernels asks for, and returns a fake hub
    causal-conv1d build exposing the new cpp_functions-style interface.
    """
    loaded = []
    iface = types.ModuleType("causal_conv1d_interface")
    iface.causal_conv1d_fwd_function = lambda *a: "fwd"
    iface.causal_conv1d_bwd_function = lambda *a: "bwd"
    cc1d = types.ModuleType("hub_causal_conv1d_abc123")
    cc1d.causal_conv1d_interface = iface

    def lazy_load_kernel(name):
        loaded.append(name)
        return cc1d if name == "causal-conv1d" else types.ModuleType(name)

    # A stand-in kernels package too: the 173af080 pre-load bypassed
    # lazy_load_kernel and went through kernels.utils.install_kernel directly.
    def install_kernel(repo_id, **kw):
        loaded.append(f"install_kernel:{repo_id}")
        return repo_id.split("/")[-1], f"/cache/{repo_id}"

    def _import_from_path(package_name, variant_path):
        loaded.append(f"import_from_path:{package_name}")
        return types.ModuleType(package_name)

    kernels_pkg = types.ModuleType("kernels")
    kernels_utils = types.ModuleType("kernels.utils")
    kernels_utils.install_kernel = install_kernel
    kernels_utils._import_from_path = _import_from_path
    kernels_pkg.utils = kernels_utils
    monkeypatch.setitem(sys.modules, "kernels", kernels_pkg)
    monkeypatch.setitem(sys.modules, "kernels.utils", kernels_utils)

    hub_kernels = sys.modules["transformers.integrations.hub_kernels"]
    hub_kernels._KERNEL_MODULE_MAPPING = {}
    monkeypatch.setattr(
        hub_kernels, "lazy_load_kernel", lazy_load_kernel, raising=False
    )
    monkeypatch.delitem(sys.modules, "causal_conv1d", raising=False)
    monkeypatch.delitem(
        sys.modules, "causal_conv1d.causal_conv1d_interface", raising=False
    )
    return loaded, cc1d, iface


def test_prepare_hub_kernels_loads_nothing_but_causal_conv1d(fake_lazy_loader):
    """The regression this pins: mamba-ssm's build imports causal_conv1d at module
    load, so anything that loads mamba-ssm before the legacy names exist binds
    causal_conv1d_cuda = None for good. On the 173af080 image a cache pre-load did
    exactly that and every hybrid load died (student-sft build test, BlueVela,
    2026-11-20). prepare_hub_kernels must load causal-conv1d and nothing else."""
    loaded, _, _ = fake_lazy_loader

    prepare_hub_kernels()

    assert loaded == ["causal-conv1d"], (
        "prepare_hub_kernels loaded more than causal-conv1d's lazy load: "
        f"{loaded}. Loading mamba-ssm before the legacy names exist breaks every "
        "hybrid load; see the NO CACHE PRE-LOAD note in hub_kernel_pins.py."
    )
    hub_kernels = sys.modules["transformers.integrations.hub_kernels"]
    assert hub_kernels._KERNEL_MODULE_MAPPING == {}


def test_prepare_hub_kernels_registers_the_legacy_causal_conv1d_names(
    fake_lazy_loader,
):
    """The modeling code asserts on the legacy causal_conv1d_cuda object; the hub
    build only has the new interface (measured 2026-11-20, image 2990684b, student-sft
    build test)."""
    _, cc1d, iface = fake_lazy_loader

    prepare_hub_kernels()

    assert sys.modules["causal_conv1d"] is cc1d
    assert sys.modules["causal_conv1d.causal_conv1d_interface"] is iface
    assert iface.causal_conv1d_cuda.causal_conv1d_fwd() == "fwd"
    assert iface.causal_conv1d_cuda.causal_conv1d_bwd() == "bwd"


def test_the_shell_preflight_reads_the_same_pin():
    """gold-kernels.sh probes and prefetches the revision the trainer will resolve.

    If the shell hardcoded a different revision than the Python pin, the out-of-band
    prefetch would warm a cache entry the trainer never reads, and the first hybrid
    load would download inside the allocation -- exactly what the preflight exists
    to prevent.
    """
    src = _SHELL.read_text(encoding="utf-8")
    assert "pinned_revision(repo)" in src
    assert 'revision="main"' not in src, (
        "gold-kernels.sh hardcodes revision=main; it must resolve through "
        "pinned_revision() so the prefetch matches the trainer's pin"
    )
