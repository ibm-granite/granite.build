import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "src" / "patch_quant_config.py"

RECIPE = """\
default_stage:
  default_modifiers:
    QuantizationModifier:
      scheme: FP8_DYNAMIC
"""


def _fp8_checkpoint_without_quant_config(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "granite"}))
    (model / "recipe.yaml").write_text(RECIPE)
    (model / "model.safetensors").write_text("weights")
    return model


def test_relative_model_path_stages_links_that_resolve(tmp_path):
    # Link targets resolve against the staging dir, not cwd, so a relative
    # model_path used to leave every staged link dangling while reporting success.
    _fp8_checkpoint_without_quant_config(tmp_path)
    out = tmp_path / "out"
    out.mkdir()

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "model", "out"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    staging = tmp_path / result.stdout.strip()
    link = staging / "model.safetensors"
    assert link.is_symlink()
    assert os.path.isabs(os.readlink(link))
    assert link.read_text() == "weights"
    patched = json.loads((staging / "config.json").read_text())
    assert patched["quantization_config"]["quant_method"] == "compressed-tensors"
