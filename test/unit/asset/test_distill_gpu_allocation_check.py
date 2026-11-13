"""The GPU allocation check in the six distill GPU steps (ibm-granite/granite.build#604).

Each step's run script refuses to start when its container can access fewer GPUs than
were allocated, instead of letting extra ranks collide on a shared GPU later. The block
is extracted from each step template and run against a stand-in ``nvidia-smi``, so the
decision itself is tested, not just its text.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

_STEPS = Path(__file__).resolve().parents[3] / "steps" / "train" / "distill"

# step -> (launcher key, echo prefix)
_GPU_STEPS = {
    "logit-distill": ("gold", "logit-distill"),
    "teacher-divergence": ("eval", "teacher-divergence"),
    "student-sft": ("train", "student-sft"),
    "teacher-logits": ("precompute", "teacher-logits"),
    "student-rollout-generator": ("vllm", "student-rollout-generator"),
    "repetition-check": ("gen_smoke", "repetition-check"),
}


def _run_script(step):
    launcher, _ = _GPU_STEPS[step]
    doc = yaml.safe_load(
        (_STEPS / step / "skypilot" / "step-template.yaml").read_text()
    )
    return doc["environment_configs"]["Skypilot"]["launchers"][launcher]["config"][
        "run"
    ]


def _check_block(step):
    """The check, from its comment header through the closing ``fi``."""
    lines = _run_script(step).split("\n")
    start = next(i for i, l in enumerate(lines) if "# GPU allocation check" in l)
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "fi")
    return "\n".join(lines[start : end + 1])


def _run(tmp_path, step, *, allocated, visible, strict=False):
    """Run the check with ``visible`` GPUs listed by a stand-in nvidia-smi.

    ``visible=None`` stands for nvidia-smi missing or failing. ``strict`` runs the block
    under ``set -euo pipefail``, as gen-smoke's script does.
    """
    shim = tmp_path / "bin"
    shim.mkdir(exist_ok=True)
    smi = shim / "nvidia-smi"
    if visible is None:
        smi.write_text("#!/bin/sh\nexit 9\n")
    else:
        gpus = "".join(
            f"echo 'GPU {i}: NVIDIA H200 (UUID: GPU-{i})'\n" for i in range(visible)
        )
        smi.write_text(f"#!/bin/sh\n{gpus}")
    smi.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "CUDA_"))}
    env["PATH"] = f"{shim}{os.pathsep}{env['PATH']}"
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    if allocated is not None:
        env["SKYPILOT_NUM_GPUS_PER_NODE"] = str(allocated)
    else:
        env.pop("SKYPILOT_NUM_GPUS_PER_NODE", None)
    script = (
        ("set -euo pipefail\n" if strict else "")
        + _check_block(step)
        + "\necho REACHED\n"
    )
    return subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize("step", sorted(_GPU_STEPS))
class TestGpuAllocationCheck:
    def test_an_undercount_is_refused_with_the_issue(self, tmp_path, step):
        result = _run(tmp_path, step, allocated=2, visible=1)
        assert result.returncode == 1
        assert "REACHED" not in result.stdout
        assert "2 GPU(s) were allocated" in result.stderr
        assert "only 1" in result.stderr
        assert "granite.build#604" in result.stderr

    @pytest.mark.parametrize("allocated,visible", [(2, 2), (1, 1), (2, 8)])
    def test_a_full_allocation_passes(self, tmp_path, step, allocated, visible):
        """8 visible covers a container that mounts every host GPU device node."""
        result = _run(tmp_path, step, allocated=allocated, visible=visible)
        assert result.returncode == 0, result.stderr
        assert "REACHED" in result.stdout

    def test_a_missing_nvidia_smi_proves_nothing(self, tmp_path, step):
        result = _run(tmp_path, step, allocated=2, visible=None)
        assert result.returncode == 0, result.stderr
        assert "REACHED" in result.stdout

    def test_an_unknown_allocation_is_not_refused(self, tmp_path, step):
        result = _run(tmp_path, step, allocated=None, visible=1)
        assert result.returncode == 0, result.stderr

    def test_it_holds_under_strict_mode(self, tmp_path, step):
        assert _run(tmp_path, step, allocated=2, visible=2, strict=True).returncode == 0
        assert _run(tmp_path, step, allocated=2, visible=1, strict=True).returncode == 1
        assert (
            _run(tmp_path, step, allocated=2, visible=None, strict=True).returncode == 0
        )

    def test_the_diagnostics_are_logged(self, tmp_path, step):
        _, prefix = _GPU_STEPS[step]
        out = _run(tmp_path, step, allocated=2, visible=2).stdout
        assert re.search(
            rf"^{re.escape(prefix)}: gpu check: allocated=2 visible=2 "
            r"CUDA_VISIBLE_DEVICES='0,1' SLURM_JOB_GPUS='' SLURM_STEP_GPUS='' devices=",
            out,
            re.M,
        ), out

    def test_the_mask_is_never_rewritten(self, step):
        assert "unset CUDA_VISIBLE_DEVICES" not in _run_script(step)
        assert "export CUDA_VISIBLE_DEVICES" not in _check_block(step)
