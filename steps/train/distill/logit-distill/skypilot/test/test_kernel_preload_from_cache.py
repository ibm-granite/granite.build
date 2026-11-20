# SPDX-License-Identifier: Apache-2.0
"""The kernel preamble's cache pre-load, run against a fake HF cache.

``preload_cached_kernels`` loads each hub kernel at its PINNED revision from the
local cache, so transformers' ``lazy_load_kernel`` short-circuits. These tests
build a cache in a temp dir with huggingface_hub's on-disk layout and resolve it
with the real ``huggingface_hub.snapshot_download(local_files_only=True)``, which
is what kernels' ``install_kernel`` calls.

What they pin, from the #540 review:

- a full-SHA pin resolves through ``snapshots/<sha>``. huggingface_hub writes no
  ``refs/<sha>`` file, and the earlier pre-load read exactly that file, so with
  a SHA pin it never loaded anything;
- a tag pin resolves through ``refs/<tag>``;
- a cached ``main`` is never loaded for a pinned repo;
- a cold cache for one kernel does not skip the other;
- a corrupt metadata.json still fails hard.

These replace a test that grepped hub_kernel_pins.py's source, which passed with
the broken pre-load.
"""

import types

import pytest
from gb_steps_post_training.distillation import hub_kernel_pins
from gb_steps_post_training.distillation.hub_kernel_pins import (
    PINNED_KERNEL_REVISIONS,
    preload_cached_kernels,
)

hf_hub = pytest.importorskip("huggingface_hub")

MAMBA = "kernels-community/mamba-ssm"
CONV = "kernels-community/causal-conv1d"
OTHER_SHA = "f" * 40  # what a stale refs/main would point at


def _snapshot(cache, repo_id, sha, *, ref=None, metadata="{}"):
    """Lay out one cached snapshot the way huggingface_hub does."""
    repo = cache / f"models--{repo_id.replace('/', '--')}"
    variant = repo / "snapshots" / sha / "build" / "torch28-cxx11-cu128-x86_64-linux"
    variant.mkdir(parents=True)
    (variant / "__init__.py").write_text("")
    if metadata is not None:
        (variant / "metadata.json").write_text(metadata)
    if ref is not None:
        (repo / "refs").mkdir(exist_ok=True)
        (repo / "refs" / ref).write_text(sha)
    return variant


def _run(cache, *, hub_mapping=None, module_mapping=None):
    """preload_cached_kernels with the real offline snapshot lookup."""

    def install_kernel(repo_id, *, revision, local_files_only):
        assert local_files_only, "the pre-load must never download"
        path = hf_hub.snapshot_download(
            repo_id, revision=revision, local_files_only=True, cache_dir=str(cache)
        )
        variants = sorted((cache / path / "build").iterdir())
        return repo_id.split("/")[-1], variants[0]

    def import_from_path(package_name, variant_path):
        module = types.ModuleType(package_name)
        module.loaded_from = str(variant_path)
        return module

    module_mapping = {} if module_mapping is None else module_mapping
    outcomes = preload_cached_kernels(
        install_kernel=install_kernel,
        import_from_path=import_from_path,
        hub_mapping=(
            {"causal-conv1d": {}, "mamba-ssm": {}}
            if hub_mapping is None
            else hub_mapping
        ),
        module_mapping=module_mapping,
        log=lambda _msg: None,
    )
    return outcomes, module_mapping


def test_sha_pins_resolve_without_a_refs_file(tmp_path):
    """The regression: both kernels are pinned by SHA, and a SHA download leaves no
    refs file behind. Both must still load."""
    conv = _snapshot(tmp_path, CONV, PINNED_KERNEL_REVISIONS[CONV])
    mamba = _snapshot(tmp_path, MAMBA, PINNED_KERNEL_REVISIONS[MAMBA])

    outcomes, mapping = _run(tmp_path)

    assert outcomes == {"causal-conv1d": "loaded", "mamba-ssm": "loaded"}
    assert mapping["causal-conv1d"].loaded_from == str(conv)
    assert mapping["mamba-ssm"].loaded_from == str(mamba)


def test_a_tag_pin_resolves_through_its_refs_file(tmp_path, monkeypatch):
    sha = PINNED_KERNEL_REVISIONS[MAMBA]
    monkeypatch.setitem(hub_kernel_pins.PINNED_KERNEL_REVISIONS, MAMBA, "v0.0.4")
    mamba = _snapshot(tmp_path, MAMBA, sha, ref="v0.0.4")

    outcomes, mapping = _run(tmp_path, hub_mapping={"mamba-ssm": {}})

    assert outcomes["mamba-ssm"] == "loaded"
    assert mapping["mamba-ssm"].loaded_from == str(mamba)


def test_a_cached_main_is_never_loaded_for_a_pinned_repo(tmp_path):
    """A cache warmed before the pin holds refs/main. Loading that would install the
    build the pin exists to avoid."""
    _snapshot(tmp_path, MAMBA, OTHER_SHA, ref="main")

    outcomes, mapping = _run(tmp_path, hub_mapping={"mamba-ssm": {}})

    assert outcomes["mamba-ssm"].startswith("skipped: ")
    assert "mamba-ssm" not in mapping


def test_a_cold_kernel_does_not_skip_the_next_one(tmp_path):
    """causal-conv1d is first. With one shared try, its cold cache used to skip
    mamba-ssm too."""
    mamba = _snapshot(tmp_path, MAMBA, PINNED_KERNEL_REVISIONS[MAMBA])

    outcomes, mapping = _run(tmp_path)

    assert outcomes["causal-conv1d"].startswith("skipped: ")
    assert outcomes["mamba-ssm"] == "loaded"
    assert mapping["mamba-ssm"].loaded_from == str(mamba)


def test_a_missing_metadata_json_is_not_an_error(tmp_path):
    _snapshot(tmp_path, MAMBA, PINNED_KERNEL_REVISIONS[MAMBA], metadata=None)

    outcomes, _ = _run(tmp_path, hub_mapping={"mamba-ssm": {}})

    assert outcomes["mamba-ssm"] == "loaded"


def test_a_corrupt_metadata_json_fails_hard(tmp_path):
    """A doubled-JSON write must not fall through to the racing online path."""
    _snapshot(tmp_path, MAMBA, PINNED_KERNEL_REVISIONS[MAMBA], metadata="{}{}")

    with pytest.raises(ValueError):
        _run(tmp_path, hub_mapping={"mamba-ssm": {}})


def test_unmapped_and_already_loaded_kernels_are_left_alone(tmp_path):
    loaded = types.ModuleType("mamba_ssm")

    outcomes, mapping = _run(
        tmp_path, hub_mapping={"mamba-ssm": {}}, module_mapping={"mamba-ssm": loaded}
    )

    assert outcomes == {"causal-conv1d": "not mapped", "mamba-ssm": "already loaded"}
    assert mapping["mamba-ssm"] is loaded
