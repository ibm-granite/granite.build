"""Smoke tests for main.py — verify imports and CLI argument shape via --help.

main.py builds its argparse parser inside `if __name__ == "__main__":`, so
we exercise it via a subprocess call to `--help`. We don't call it without
--help because that triggers ray.init() and reads real files.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# These tests spawn subprocesses that import the full fm-tune stack (~10s each).
pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).resolve().parent.parent
MAIN_PY = REPO_ROOT / "main.py"


class TestMainHelp:
    def _run_help(self):
        return subprocess.run(
            [sys.executable, str(MAIN_PY), "--help"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
        )

    def test_help_exits_zero(self):
        result = self._run_help()
        assert result.returncode == 0, f"main.py --help exited {result.returncode}\nstderr: {result.stderr}"

    def test_help_lists_required_args(self):
        result = self._run_help()
        out = result.stdout + result.stderr
        for required in [
            "--config_file",
            "--train_file",
            "--validation_file",
            "--model_name_or_path",
            "--output_dir",
            "--output_model_name",
        ]:
            assert required in out, f"Required arg {required} not in --help output"

    def test_help_lists_optional_flags(self):
        result = self._run_help()
        out = result.stdout + result.stderr
        for flag in [
            "--seed",
            "--tuning_algo",
            "--rl_algo",
            "--no_autotune",
            "--backend",
            "--resume_from_checkpoint",
            "--keep_checkpoints",
            "--cleanup",
            "--save_history",
            "--data_backend",
            "--tokenizer_name_or_path",
            "--additional_special_tokens",
            "--pad_token",
            "--eos_token",
            "--bos_token",
            "--autotunex_server_url",
        ]:
            assert flag in out, f"Optional flag {flag} not in --help output"

    def test_help_omits_removed_flags(self):
        result = self._run_help()
        out = result.stdout + result.stderr
        # The HPO-restore flags were removed; only --resume_from_checkpoint remains.
        for removed in ["--env", "--logger_url", "--restore", "--restore_path"]:
            assert removed not in out, f"Removed flag {removed} still in --help output"


class TestMainModuleImportable:
    def test_main_module_imports(self, monkeypatch):
        """Importing main triggers ray import. That should still succeed."""
        # main is already imported (by other tests via collection), but ensure
        # a fresh subprocess can also import it.
        result = subprocess.run(
            [sys.executable, "-c", "import main; print('ok')"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, f"import main failed:\n{result.stderr}"
        assert "ok" in result.stdout
