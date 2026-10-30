"""The Granite suite definitions (which benchmarks a model family is scored on)."""

from __future__ import annotations

from importlib import resources

import yaml


def names() -> list[str]:
    return sorted(p.name.removesuffix(".yaml") for p in resources.files(__name__).iterdir() if p.name.endswith(".yaml"))


def load(name: str) -> dict:
    path = resources.files(__name__) / f"{name}.yaml"
    if not path.is_file():
        raise SystemExit(f"unknown suite {name!r}; known: {', '.join(names())}")
    return yaml.safe_load(path.read_text())
