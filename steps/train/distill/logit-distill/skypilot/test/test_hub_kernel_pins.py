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
"""The mamba-ssm hub-kernel pin: what it pins, and that the loaders call it.

The pin exists because transformers 5.8.0's default mapping resolves mamba-ssm at
``version=1`` -- the kernels-community repo's v1 branch -- whose build imports
``GreedySearchDecoderOnlyOutput``, a symbol transformers 5.x renamed. Every fresh
hybrid (granitemoehybrid) model load dies on that import mid-load (measured
2026-11-13 on BlueVela, image ec82c1d9: teacher-divergence failed its build,
preflight-check's probe reported ``loaded=NONE``). ``v0.0.3`` is the newest
compatible revision. See ``hub_kernel_pins.py`` for the full measurement.

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
    assert entry["revision"] == "v0.0.3"
    assert "version" not in entry


def test_pin_leaves_unpinned_kernels_alone(fake_mapping):
    pin_hub_kernels()

    assert fake_mapping["causal-conv1d"] == {
        "repo_id": "kernels-community/causal-conv1d",
        "version": 1,
    }
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
        assert entry.get("revision") == "v0.0.3"
        assert "version" not in entry
    finally:
        hub_kernels._HUB_KERNEL_MAPPING.clear()
        hub_kernels._HUB_KERNEL_MAPPING.update(before)


def test_pinned_revision_helper():
    assert pinned_revision("kernels-community/mamba-ssm") == "v0.0.3"
    # Unlisted repos keep their default resolution.
    assert pinned_revision("kernels-community/causal-conv1d") == "main"
    assert pinned_revision("kernels-community/flash-attn2") == "main"


@pytest.mark.parametrize("path", _ENTRYPOINTS, ids=lambda p: p.name)
def test_every_hybrid_model_entrypoint_calls_the_pin(path):
    """The pin only helps if it runs before the first model load.

    These four entrypoints instantiate hybrid (granitemoehybrid) models; the
    teacher-logits step loads only the dense teacher, and vLLM resolves its own
    kernels, so neither needs the call. A source check, not an import, because
    importing every entrypoint here would drag torch and trl into this suite.
    """
    src = path.read_text(encoding="utf-8")
    assert "pin_hub_kernels()" in src, (
        f"{path.name} loads hybrid models but never pins the mamba-ssm kernel; "
        "add the pin_hub_kernels() call (see hub_kernel_pins.py)"
    )


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
