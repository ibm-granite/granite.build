# SPDX-License-Identifier: Apache-2.0

"""The bfcl step's FC output budget plumbing: the run script asks
granite_evals.benchmarks.bfcl_serving (tested with the runtime) to record its decision in
the output dir's fc_budget.json and for the vllm serve arguments, puts them ahead of the
step's own vllm_args, passes no --option (the score phase's options would differ from the
generation's), skips it all in the score phase, and stops on a failure. The script's own
lines run in bash (set -e, pipefail, as the template), with a stub in place of the helper
that records the arguments it was given."""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

TEMPLATE = Path(__file__).resolve().parent.parent / "step-template.yaml"
HELPER = "python -m granite_evals.benchmarks.bfcl_serving"
BUDGET = '--override-generation-config={"max_new_tokens":16384}'
CONFIG = {
    "{{ config.options }}": "handler=fc temperature=0.5",
    "{{ config.model_path }}": "/models/m",
    "{{ config.base_url }}": "",
}


def _run(tmp_path, phase="generate", vllm_args="", helper_out=BUDGET, helper_rc=0):
    step = yaml.safe_load(TEMPLATE.read_text())
    run = step["environment_configs"]["Skypilot"]["launchers"]["granite"]["config"][
        "run"
    ]
    block = re.search(
        r"^\s*OPTS=\(\)\n.*?(?=^\s*granite-evals run )", run, re.S | re.M
    ).group(0)
    assert HELPER in block
    argv = tmp_path / "helper-argv"
    stub = (
        f"printf '%s\\n' \"$@\" > {_sq(str(argv))}; "
        f"printf '%s' {_sq(helper_out)}; exit {helper_rc}"
    )
    script = block.replace(HELPER, f"bash -c {_sq(stub)} --")
    for key, value in {**CONFIG, "{{ config.vllm_args }}": vllm_args}.items():
        script = script.replace(key, value)
    script = script.replace("{{ config.phase }}", phase)
    assert "{{" not in script, script
    script = (
        "set -e -o pipefail\n"
        f"OUTPUT_DIR={_sq(str(tmp_path / 'out'))}\n"
        + script
        + 'printf "OPT %s\\n" "${OPTS[@]}"; printf "VLLM %s\\n" "${VLLM_ARGS[@]}"'
    )
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    proc.helper_argv = argv.read_text().splitlines() if argv.exists() else None
    return proc


def _sq(s):
    return "'" + s.replace("'", "'\\''") + "'"


def _lines(proc, tag):
    return [
        line[len(tag) + 1 :]
        for line in proc.stdout.splitlines()
        if line.startswith(tag + " ")
    ]


@pytest.mark.parametrize("phase", ["generate", "all"])
def test_the_budget_reaches_the_run(tmp_path, phase):
    proc = _run(tmp_path, phase, vllm_args="--max-num-seqs=4")
    assert proc.returncode == 0, proc.stderr
    # In this order: the record's file in the output dir, then what decides the budget
    # (vllm_args too: a generation config set there is recorded, not overridden).
    assert proc.helper_argv == [
        str(tmp_path / "out" / "fc_budget.json"),
        "/models/m",
        "",
        "handler=fc temperature=0.5",
        "--max-num-seqs=4",
    ]
    # Ahead of the step's own vllm_args: a caller's --override-generation-config replaces it
    # (vLLM keeps the last one; USAGE.md).
    assert _lines(proc, "VLLM") == [
        f"--vllm-arg={BUDGET}",
        "--vllm-arg=--max-num-seqs=4",
    ]
    # Only the caller's options: an added one would differ from the score phase's.
    assert _lines(proc, "OPT") == [
        "--option",
        "handler=fc",
        "--option",
        "temperature=0.5",
    ]


def test_the_helper_sees_vllm_args_as_vllm_gets_them(tmp_path):
    # Quoted JSON is one word to vLLM after bash's quote removal; so it is to the helper.
    vllm_args = "--max-num-seqs=4 --override-generation-config '{\"temperature\": 0.5}'"
    proc = _run(tmp_path, vllm_args=vllm_args, helper_out="")
    assert proc.returncode == 0, proc.stderr
    words = ["--max-num-seqs=4", "--override-generation-config", '{"temperature": 0.5}']
    assert proc.helper_argv[4:] == words
    assert _lines(proc, "VLLM") == [f"--vllm-arg={w}" for w in words]


def test_no_budget_adds_nothing(tmp_path):
    proc = _run(tmp_path, helper_out="")
    assert proc.returncode == 0, proc.stderr
    assert _lines(proc, "VLLM") == [""] and len(_lines(proc, "OPT")) == 4


def test_the_score_phase_asks_for_nothing(tmp_path):
    proc = _run(
        tmp_path, "score", helper_rc=3
    )  # the helper would fail: it must not run
    assert proc.returncode == 0, proc.stderr
    assert proc.helper_argv is None and _lines(proc, "VLLM") == [""]


def test_a_helper_failure_stops_the_job(tmp_path):
    proc = _run(tmp_path, helper_rc=1)
    assert proc.returncode == 1 and "bfcl_serving failed" in proc.stderr
