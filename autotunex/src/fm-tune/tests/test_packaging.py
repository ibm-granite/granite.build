"""Tests that fm-tune's base install is slim and the heavy stack lives in extras."""

from pathlib import Path

import tomllib  # Python 3.11+

_PYPROJECT = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())


def test_base_dependencies_are_slim():
    assert _PYPROJECT["project"]["dependencies"] == ["pyyaml>=6.0.2"]


def test_core_carries_the_training_stack_and_full_adds_the_gpu_stack():
    extras = _PYPROJECT["project"]["optional-dependencies"]
    core = " ".join(extras["core"])
    for dep in (
        "torch==2.8.0",
        "transformers==4.57.6",
        "peft==0.18.0",
        "trl==0.29.0",
        "bitsandbytes==0.49.0",
        "accelerate",
        "hyperopt",
        "tokenizers",
        "sentencepiece",
        "tabulate",
        "tensorboardx",
        "packaging",
        "datasets",
        "ray[tune,default]",
    ):
        assert dep in core, dep
    full = " ".join(extras["full"])
    for dep in ("autotune[core]", "verl[vllm]", "flash-attn", "deepspeed"):
        assert dep in full, dep


def test_configs_are_shipped_as_package_data():
    assert _PYPROJECT["tool"]["setuptools"]["package-data"]["autotune"] == ["configs/*.yaml"]
