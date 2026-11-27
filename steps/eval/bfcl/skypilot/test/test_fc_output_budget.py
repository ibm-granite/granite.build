"""The bfcl step's FC output budget: the run script serves vLLM with a max_new_tokens
default (--override-generation-config) unless handler=legacy-granite, ahead of the
step's own vllm_args, so a caller's value still wins. The script's own lines, run in bash."""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

TEMPLATE = Path(__file__).resolve().parent.parent / "step-template.yaml"
BUDGET = '--vllm-arg=--override-generation-config={"max_new_tokens":16384}'


def _vllm_args(options: str, vllm_args: str = "") -> list[str]:
    step = yaml.safe_load(TEMPLATE.read_text())
    run = step["environment_configs"]["Skypilot"]["launchers"]["granite"]["config"]["run"]
    block = re.search(r"^\s*VLLM_ARGS=\(\)\n.*?\{\{ config\.vllm_args \}\}.*?\n", run, re.S | re.M).group(0)
    script = block.replace("{{ config.options }}", options).replace("{{ config.vllm_args }}", vllm_args)
    out = subprocess.run(["bash", "-c", script + 'printf "%s\\n" "${VLLM_ARGS[@]}"'],
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


@pytest.mark.parametrize("options", ["", "categories=simple", "handler=fc temperature=0.001", "max_tokens=100"])
def test_fc_gets_the_served_default(options):
    assert _vllm_args(options) == [BUDGET]


@pytest.mark.parametrize("options", ["handler=legacy-granite", "handler=legacy-granite temperature=0.001",
                                     "temperature=0.001 handler=legacy-granite"])
def test_legacy_granite_keeps_its_serving(options):
    assert _vllm_args(options) == []


def test_a_callers_vllm_args_come_after_the_budget():
    # After it, so a caller's own --override-generation-config wins (vLLM keeps the last).
    assert _vllm_args("", "--max-num-seqs=4 --enable-prefix-caching") == [
        BUDGET, "--vllm-arg=--max-num-seqs=4", "--vllm-arg=--enable-prefix-caching"]


def test_the_budget_is_above_the_calibrations_largest_fc_output():
    # The bfcl-v3 calibration runs' largest per-request FC output: 8,697 tokens
    # (granite-4.0-micro, multi_turn_long_context_132). USAGE.md records the evidence.
    assert int(re.search(r'"max_new_tokens":(\d+)', BUDGET).group(1)) > 8697 * 1.5
