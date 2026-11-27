# SPDX-License-Identifier: Apache-2.0

"""The bfcl step's FC output budget plumbing: the run script asks
granite_evals.benchmarks.bfcl_serving (tested with the runtime) for the fc_budget record
and the vllm serve arguments, passes the record as an --option, puts the arguments ahead
of the step's own vllm_args, skips it all in the score phase, and stops on a failure.
The script's own lines run in bash, with a stub in place of the helper."""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

TEMPLATE = Path(__file__).resolve().parent.parent / "step-template.yaml"
HELPER = "python -m granite_evals.benchmarks.bfcl_serving"
BUDGET = '--override-generation-config={"max_new_tokens":16384}'


def _run(
    phase="generate",
    vllm_args="",
    helper_out="fc_budget=16384:step-default\n" + BUDGET,
    helper_rc=0,
):
    step = yaml.safe_load(TEMPLATE.read_text())
    run = step["environment_configs"]["Skypilot"]["launchers"]["granite"]["config"][
        "run"
    ]
    block = re.search(
        r"^\s*OPTS=\(\)\n.*?\{\{ config\.vllm_args \}\}.*?\n", run, re.S | re.M
    ).group(0)
    assert HELPER in block
    stub = f"printf '%s\\n' {_sq(helper_out)}; exit {helper_rc}"
    script = (
        block.replace(HELPER, f"bash -c {_sq(stub)} --")
        .replace("{{ config.options }}", "handler=fc")
        .replace("{{ config.vllm_args }}", vllm_args)
        .replace("{{ config.phase }}", phase)
        .replace("{{ config.model_path }}", "m")
        .replace("{{ config.base_url }}", "")
    )
    script += 'printf "OPT %s\\n" "${OPTS[@]}"; printf "VLLM %s\\n" "${VLLM_ARGS[@]}"'
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def _sq(s):
    return "'" + s.replace("'", "'\\''") + "'"


def _lines(proc, tag):
    return [
        line[len(tag) + 1 :]
        for line in proc.stdout.splitlines()
        if line.startswith(tag + " ")
    ]


@pytest.mark.parametrize("phase", ["generate", "all"])
def test_the_budget_and_its_record_reach_the_run(phase):
    proc = _run(phase, vllm_args="--max-num-seqs=4")
    assert proc.returncode == 0, proc.stderr
    assert _lines(proc, "OPT")[-2:] == ["--option", "fc_budget=16384:step-default"]
    # Ahead of the step's own vllm_args: a caller's --override-generation-config replaces it
    # (vLLM keeps the last one; USAGE.md).
    assert _lines(proc, "VLLM") == [
        f"--vllm-arg={BUDGET}",
        "--vllm-arg=--max-num-seqs=4",
    ]


def test_no_budget_means_only_the_record():
    proc = _run(helper_out="fc_budget=none:legacy-granite")
    assert _lines(proc, "OPT")[-1] == "fc_budget=none:legacy-granite" and _lines(
        proc, "VLLM"
    ) == [""]


def test_the_score_phase_asks_for_nothing():
    proc = _run("score", helper_rc=3)  # the helper would fail: it must not run at all
    assert proc.returncode == 0, proc.stderr
    assert "fc_budget" not in proc.stdout and _lines(proc, "VLLM") == [""]


def test_a_helper_failure_stops_the_job():
    proc = _run(helper_rc=1)
    assert proc.returncode == 1 and "bfcl_serving failed" in proc.stderr
