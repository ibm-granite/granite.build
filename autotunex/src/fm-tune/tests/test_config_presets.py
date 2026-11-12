"""The preset configs stay in step with ``autotune.yaml``.

Each file under ``autotune/configs/`` is a full, self-contained copy of the base
config with a few values changed, because users copy them and pass one with
``--config_file``. Nothing but this test stops a key added to ``autotune.yaml``
from being forgotten in the other seven, so that preset would silently lose the
parameter. A preset may add keys of its own, but only the ones listed in
``PRESET_EXTRAS``.
"""

from pathlib import Path
from typing import Any, Dict, Set

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "autotune" / "configs"
BASE = "autotune"

# Keys a preset has and the base does not, by design.
PRESET_EXTRAS: Dict[str, Set[str]] = {
    "autotune_mlx": {"training_config/mlx_num_layers"},
    "autotune_sr_3b": {"training_config/sr_share_moe_routing"},
    "autotune_sr_smoke_1gpu": {"training_config/sr_share_moe_routing"},
    "autotune_sr_smoke_2gpu": {"training_config/sr_share_moe_routing"},
}

# A parameter spec is a dict carrying one of these; its own fields are values, not keys to compare.
_SPEC_FIELDS = {"default", "type", "values"}


def _key_paths(node: Any, prefix: str = "") -> Set[str]:
    """Paths of every config key, stopping at a parameter spec's own fields."""
    paths: Set[str] = set()
    for key, value in node.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        paths.add(path)
        if isinstance(value, dict) and not _SPEC_FIELDS & set(value):
            paths |= _key_paths(value, path)
    return paths


def _load(name: str) -> Dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text())


PRESETS = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml") if p.stem != BASE)


def test_there_are_presets_to_compare():
    assert PRESETS, "no preset configs found next to autotune.yaml"


@pytest.mark.parametrize("preset", PRESETS)
def test_preset_has_every_base_key(preset):
    missing = _key_paths(_load(BASE)) - _key_paths(_load(preset))
    assert not missing, f"{preset}.yaml is missing keys that autotune.yaml has: {sorted(missing)}"


@pytest.mark.parametrize("preset", PRESETS)
def test_preset_adds_only_declared_keys(preset):
    extra = _key_paths(_load(preset)) - _key_paths(_load(BASE))
    undeclared = extra - PRESET_EXTRAS.get(preset, set())
    assert not undeclared, (
        f"{preset}.yaml has keys autotune.yaml lacks (declare in PRESET_EXTRAS?): {sorted(undeclared)}"
    )
