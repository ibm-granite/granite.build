"""Tests for the torch-free ``autotune.catalog`` module."""

import subprocess
import sys
import textwrap

_BLOCKER = textwrap.dedent(
    """
    import sys
    _BLOCKED = {"torch", "ray", "peft", "transformers"}
    class _Blocker:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in _BLOCKED:
                raise ImportError("blocked: " + name)
            return None
    sys.meta_path.insert(0, _Blocker())
    """
)


def _run_blocked(body: str) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter with torch/ray/peft/transformers import-blocked."""
    return subprocess.run(
        [sys.executable, "-c", _BLOCKER + textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )


def test_catalog_imports_and_works_without_heavy_deps():
    proc = _run_blocked(
        """
        import autotune.catalog as c
        cfg = c.get_autotune_config()
        dts = c.get_autotune_dataset_types()
        assert isinstance(cfg, dict) and cfg
        assert {"dataset_type_a", "dataset_type_b", "dataset_type_c", "dataset_type_d"} <= set(dts)
        print("CATALOG_OK")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert "CATALOG_OK" in proc.stdout


def test_importing_utils_without_heavy_deps_raises_the_friendly_hint():
    proc = _run_blocked("import autotune.utils")
    assert proc.returncode != 0
    combined = (proc.stderr + proc.stdout).lower()
    assert "[full]" in combined
    assert "[core]" not in combined and "[sft]" not in combined and "[rl]" not in combined
    assert "utils" in combined


def test_get_autotune_config_returns_a_nonempty_mapping():
    from autotune.catalog import get_autotune_config

    config = get_autotune_config()

    assert isinstance(config, dict)
    assert config  # configs/autotune.yaml is not empty


def test_get_autotune_dataset_types_has_the_four_known_types():
    from autotune.catalog import get_autotune_dataset_types

    types = get_autotune_dataset_types()

    assert set(types) >= {"dataset_type_a", "dataset_type_b", "dataset_type_c", "dataset_type_d"}


def test_constants_reexports_dataset_types_from_catalog():
    import autotune.catalog as catalog
    import autotune.constants as constants

    assert constants.AutotuneDatasetTypes is catalog.AutotuneDatasetTypes


def test_utils_reexports_the_catalog_functions():
    import autotune.catalog as catalog
    from autotune.utils import get_autotune_config, get_autotune_dataset_types

    assert get_autotune_config is catalog.get_autotune_config
    assert get_autotune_dataset_types is catalog.get_autotune_dataset_types
