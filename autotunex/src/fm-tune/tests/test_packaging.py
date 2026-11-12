"""Tests that fm-tune's base install is slim and the heavy stack lives in extras."""

import re
from pathlib import Path

import tomllib  # Python 3.11+

_PYPROJECT = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())


def test_base_dependencies_are_slim():
    assert _PYPROJECT["project"]["dependencies"] == ["pyyaml>=6.0.2"]


def _names(reqs):
    """Requirement names without versions, extras or markers ("verl[vllm]==0.9.1; ..." -> "verl")."""
    return {re.split(r"[\[=<>~!; ]", r, maxsplit=1)[0].lower() for r in reqs}


def test_core_carries_the_training_stack():
    """`core` is the lean CPU/demo install (single-device SFT + DPO/KTO); pyproject explains why it exists."""
    core = _names(_PYPROJECT["project"]["optional-dependencies"]["core"])
    for dep in (
        "torch",
        "transformers",
        "peft",
        "trl",
        "bitsandbytes",
        "accelerate",
        "hyperopt",
        "tokenizers",
        "sentencepiece",
        "tabulate",
        "tensorboardx",
        "packaging",
        "datasets",
        "ray",
    ):
        assert dep in core, dep
    # The GPU-only stack must stay out of core (it does not build on CPU / macOS).
    assert not core & {"verl", "flash-attn", "deepspeed", "vllm"}


def test_full_is_core_plus_the_gpu_stack():
    full = _PYPROJECT["project"]["optional-dependencies"]["full"]
    assert "autotune[core]" in full  # every shared pin lives in core only
    assert {"verl", "flash-attn", "deepspeed"} <= _names(full)


def test_configs_are_shipped_as_package_data():
    assert _PYPROJECT["tool"]["setuptools"]["package-data"]["autotune"] == ["configs/*.yaml"]
