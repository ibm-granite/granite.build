"""Tests for autotune.utils.cleanup — verifies it wipes the launcher's
transient artifact dirs (now including <output_dir>/logs/) and leaves
unrelated content alone.
"""

from pathlib import Path

from autotune.utils import cleanup


def _populate(output_dir: Path) -> None:
    """Create the dirs cleanup() targets plus an unrelated sibling."""
    (output_dir / "ray_results").mkdir(parents=True)
    (output_dir / "ray_results" / "trial.json").write_text("trial data")

    (output_dir / "train_results").mkdir(parents=True)
    (output_dir / "train_results" / "metrics.csv").write_text("loss\n0.1")

    (output_dir / "lightning_logs").mkdir(parents=True)
    (output_dir / "lightning_logs" / "events.out").write_text("events")

    # The new entry: per-run worker bring-up logs.
    run_dir = output_dir / "logs" / "ray_nodes_2026-06-03-12-00-00"
    run_dir.mkdir(parents=True)
    (run_dir / "host1.log").write_text("ibstat output")
    (run_dir / "host2.log").write_text("nvidia-smi output")

    # Untouched siblings (model artifacts, etc.).
    (output_dir / "granite-lora").mkdir(parents=True)
    (output_dir / "granite-lora" / "adapter_model.safetensors").write_text("weights")
    (output_dir / "results.json").write_text("{}")


class TestCleanup:
    def test_wipes_logs_dir(self, tmp_path):
        _populate(tmp_path)
        cleanup(str(tmp_path))
        # The dir is recreated empty (cleanup uses rmtree + mkdir).
        assert (tmp_path / "logs").is_dir()
        assert list((tmp_path / "logs").iterdir()) == []

    def test_wipes_ray_results(self, tmp_path):
        _populate(tmp_path)
        cleanup(str(tmp_path))
        assert (tmp_path / "ray_results").is_dir()
        assert list((tmp_path / "ray_results").iterdir()) == []

    def test_wipes_train_results_and_lightning_logs(self, tmp_path):
        _populate(tmp_path)
        cleanup(str(tmp_path))
        for name in ("train_results", "lightning_logs"):
            d = tmp_path / name
            assert d.is_dir()
            assert list(d.iterdir()) == []

    def test_preserves_model_artifacts(self, tmp_path):
        _populate(tmp_path)
        cleanup(str(tmp_path))
        # Model output and top-level files are NOT touched.
        assert (tmp_path / "granite-lora" / "adapter_model.safetensors").exists()
        assert (tmp_path / "results.json").exists()

    def test_idempotent_when_dirs_missing(self, tmp_path):
        # cleanup should be a no-op when the targeted dirs don't exist.
        # In particular, the new logs/ entry must not fail on fresh runs
        # where worker bring-up never created it.
        cleanup(str(tmp_path))
        # Nothing was created.
        assert not (tmp_path / "logs").exists()
        assert not (tmp_path / "ray_results").exists()
