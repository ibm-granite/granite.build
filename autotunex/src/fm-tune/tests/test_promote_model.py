"""Tests for --model_at_root: promote_model_to_root / resolve_keep_diagnostics.

Pure filesystem logic — no Ray, Torch, or GPU needed.
"""

import pytest

from autotune.utils import promote_model_to_root, resolve_keep_diagnostics

MODEL_FILES = ["config.json", "model.safetensors", "tokenizer.json"]


def _make_run(tmp_path, model_subdir="granite-grpo"):
    """Lay out an output dir the way a finished verl/multi-GPU run leaves it."""
    model_dir = tmp_path / model_subdir
    model_dir.mkdir(parents=True)
    for name in MODEL_FILES:
        (model_dir / name).write_text(name)
    (tmp_path / "final_checkpoints").mkdir()
    (tmp_path / "final_checkpoints" / "final_config.json").write_text("{}")
    (tmp_path / "outputs" / "trial1").mkdir(parents=True)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "run_trials.csv").write_text("a,b")
    (tmp_path / "logs").mkdir()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("10")
    return model_dir


def _entries(path):
    return sorted(p.name for p in path.iterdir())


class TestPromoteModelToRoot:
    def test_promotes_model_and_deletes_everything_else(self, tmp_path):
        _make_run(tmp_path)

        assert promote_model_to_root(str(tmp_path), "granite-grpo") is True

        assert _entries(tmp_path) == sorted(MODEL_FILES)
        assert (tmp_path / "config.json").read_text() == "config.json"

    def test_single_device_models_subdir_is_promoted(self, tmp_path):
        _make_run(tmp_path, model_subdir="models/smollm2-lora")

        assert promote_model_to_root(str(tmp_path), "smollm2-lora") is True

        assert _entries(tmp_path) == sorted(MODEL_FILES)

    def test_keep_diagnostics_moves_everything_else_into_diagnostics(self, tmp_path):
        _make_run(tmp_path)

        assert promote_model_to_root(str(tmp_path), "granite-grpo", keep_diagnostics=True) is True

        assert _entries(tmp_path) == sorted(MODEL_FILES + ["diagnostics"])
        assert _entries(tmp_path / "diagnostics") == [
            "final_checkpoints",
            "latest_checkpointed_iteration.txt",
            "logs",
            "outputs",
            "results",
        ]
        assert (tmp_path / "diagnostics" / "results" / "run_trials.csv").read_text() == "a,b"

    def test_missing_model_dir_touches_nothing(self, tmp_path):
        _make_run(tmp_path)
        before = _entries(tmp_path)

        assert promote_model_to_root(str(tmp_path), "other-name") is False

        assert _entries(tmp_path) == before

    def test_empty_model_dir_touches_nothing(self, tmp_path):
        (tmp_path / "granite-grpo").mkdir()
        (tmp_path / "logs").mkdir()

        assert promote_model_to_root(str(tmp_path), "granite-grpo") is False

        assert _entries(tmp_path) == ["granite-grpo", "logs"]

    def test_model_entry_named_diagnostics_is_refused(self, tmp_path):
        model_dir = _make_run(tmp_path)
        (model_dir / "diagnostics").mkdir()
        before = _entries(tmp_path)

        assert promote_model_to_root(str(tmp_path), "granite-grpo", keep_diagnostics=True) is False

        assert _entries(tmp_path) == before
        assert (model_dir / "config.json").exists()

    def test_leftover_staging_dir_is_refused(self, tmp_path):
        # An interrupted earlier promote leaves the staging dir behind;
        # shutil.move would nest the new model inside it.
        _make_run(tmp_path)
        (tmp_path / ".fmtune-model-staging").mkdir()
        (tmp_path / ".fmtune-model-staging" / "model.safetensors").write_text("stale")
        before = _entries(tmp_path)

        assert promote_model_to_root(str(tmp_path), "granite-grpo") is False

        assert _entries(tmp_path) == before

    def test_tilde_is_not_expanded(self, tmp_path, monkeypatch):
        # The drivers write to output_dir as given, so a literal "~" path
        # must resolve to the same (relative) dir here, not to $HOME.
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.chdir(tmp_path)
        run_dir = tmp_path / "~" / "run"
        _make_run(run_dir)

        assert promote_model_to_root("~/run", "granite-grpo") is True

        assert _entries(run_dir) == sorted(MODEL_FILES)


class TestResolveKeepDiagnostics:
    def test_flag_wins(self, monkeypatch):
        monkeypatch.setenv("FMTUNE_KEEP_DIAGNOSTICS", "0")
        assert resolve_keep_diagnostics(True) is True

    def test_off_when_unset(self, monkeypatch):
        monkeypatch.delenv("FMTUNE_KEEP_DIAGNOSTICS", raising=False)
        assert resolve_keep_diagnostics(False) is False

    @pytest.mark.parametrize("val", ["1", "true", "YES", " on "])
    def test_env_truthy(self, monkeypatch, val):
        monkeypatch.setenv("FMTUNE_KEEP_DIAGNOSTICS", val)
        assert resolve_keep_diagnostics(False) is True

    @pytest.mark.parametrize("val", ["", "0", "false", "no", "off"])
    def test_env_falsy(self, monkeypatch, val):
        monkeypatch.setenv("FMTUNE_KEEP_DIAGNOSTICS", val)
        assert resolve_keep_diagnostics(False) is False
