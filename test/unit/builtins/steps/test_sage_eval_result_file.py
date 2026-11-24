# SPDX-License-Identifier: Apache-2.0

"""The sage-eval steps' results-file search must find SALAD-Bench's own result (#482).

After the eval script, each step emits the first results file under the experiment's
output tree as its sage_eval_results artifact. SALAD-Bench writes
safety-eval/salad-bench/<model>/base_set-result.json. With only results*.json,
metrics*.json and metrics*.jsonl to match, a SALAD eval alone in its tree found
nothing and exited 1 after a complete run (builds f71274d5, 1131fd71); in a shared
tree it emitted a sibling eval's file instead. The search is also anchored at
/<experiment>/, so one rung's name cannot match a longer one (ckpt_25, ckpt_250).
These tests run each step's real search block in bash against such trees.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from gbserver.utils.template import fill_template

from .sage_eval_steps import one_script

REPO_ROOT = Path(__file__).resolve().parents[4]
ENVS = REPO_ROOT / "configurations/assets/environments/skypilot"

# The steps under test, and the variable each roots its search at.
STEPS = {
    "aws/steps/sage-eval": "OUTPUT_BASE_PATH",
    "steps/sage-eval": "OUTPUT_BASE_PATH",
}

EXPERIMENT = "exp-ckpt_1000"
SALAD = "safety-eval/salad-bench/export-1000/base_set-result.json"


def _search_block(step: str) -> str:
    """The step's run-script lines from the first results search to the emit's `fi`."""
    run = one_script(ENVS / step / "step.yaml", "GB_ARTIFACT_ID:sage_eval_results")
    lines = run.splitlines()
    start = next(i for i, line in enumerate(lines) if "RESULT_FILE=$(find" in line)
    emit = next(i for i, line in enumerate(lines) if 'if [ -n "$RESULT_FILE" ]' in line)
    indent = len(lines[emit]) - len(lines[emit].lstrip())
    end = next(
        i
        for i in range(emit + 1, len(lines))
        if lines[i].strip() == "fi" and len(lines[i]) - len(lines[i].lstrip()) == indent
    )
    return "\n".join(lines[start : end + 1])


# mtimes: the eval's log, and files written before or after it (the -newer fallback).
LOG_MTIME, OLD, NEW = 1000, 0, 2000


def _emitted(step: str, root: Path, tmp_path: Path) -> str | None:
    log = tmp_path / "sage-eval.log"
    log.write_text("")
    os.utime(log, (LOG_MTIME, LOG_MTIME))
    # The fallback search compares against a file touched before the eval: the
    # eval's log on LSF, a start marker on AWS. Point either at a private copy.
    block = _search_block(step)
    env = {**os.environ, STEPS[step]: str(root), "EXPERIMENT": EXPERIMENT}
    if "/tmp/sage-eval.log" in block:
        block = block.replace("/tmp/sage-eval.log", str(log))
    else:
        assert '-newer "$EVAL_START_MARKER"' in block
        env["EVAL_START_MARKER"] = str(log)
    # -e as in the step's run script. check=False: an empty tree is meant to exit 1;
    # the emitted line is what counts.
    out = subprocess.run(
        ["bash", "-ec", block], env=env, capture_output=True, text=True, check=False
    ).stdout
    m = re.search(r"GB_ARTIFACT_ID:sage_eval_results GB_ARTIFACT_PATH:(.+)", out)
    return m.group(1) if m else None


def _touch(
    root: Path, rel: str, experiment: str = EXPERIMENT, mtime: int = OLD
) -> Path:
    p = root / experiment / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}")
    # OLD by default: older than the eval's log, so only the primary search sees it.
    os.utime(p, (mtime, mtime))
    return p


@pytest.mark.parametrize("step", STEPS)
def test_a_salad_eval_alone_in_its_tree_emits_its_own_result(step, tmp_path):
    """The eval-rerun case: RERUN_SAGE_DIR holds only the SALAD eval's output."""
    root = tmp_path / "out"
    _touch(root, SALAD)
    _touch(root, "safety-eval/salad-bench/export-1000/base_set-gen.json")
    assert _emitted(step, root, tmp_path) == str(root / EXPERIMENT / SALAD)


@pytest.mark.parametrize("step", STEPS)
def test_a_longer_rung_name_is_not_this_rungs_result(step, tmp_path):
    """A failed exp-ckpt_1000 must not pass on exp-ckpt_10000's result."""
    root = tmp_path / "out"
    _touch(root, "safety-eval/attaq/results.json", experiment=EXPERIMENT + "0")
    (root / EXPERIMENT / "logs-dir").mkdir(parents=True)
    assert _emitted(step, root, tmp_path) is None


@pytest.mark.parametrize("step", STEPS)
def test_an_empty_tree_emits_nothing(step, tmp_path):
    root = tmp_path / "out"
    (root / EXPERIMENT / "logs-dir").mkdir(parents=True)
    assert _emitted(step, root, tmp_path) is None


@pytest.mark.parametrize("step", STEPS)
def test_the_fallback_emits_a_new_csv_under_the_experiment(step, tmp_path):
    """Only the fallback matches '*.csv', and only files newer than the eval's log."""
    root = tmp_path / "out"
    _touch(root, "code-eval/old.csv")
    new = _touch(root, "code-eval/summary.csv", mtime=NEW)
    assert _emitted(step, root, tmp_path) == str(new)


@pytest.mark.parametrize("step", STEPS)
def test_the_fallback_ignores_a_sibling_rungs_new_result(step, tmp_path):
    """A crashed rung's log is older than results a parallel rung writes afterwards."""
    root = tmp_path / "out"
    _touch(
        root, "safety-eval/attaq/results.json", experiment="exp-ckpt_2000", mtime=NEW
    )
    _touch(root, "summary.csv", experiment="exp-ckpt_2000", mtime=NEW)
    (root / EXPERIMENT / "logs-dir").mkdir(parents=True)
    assert _emitted(step, root, tmp_path) is None


def _line_index(lines: list[str], needle: str) -> int:
    """Return the index of the first line that contains ``needle``."""
    return next(i for i, line in enumerate(lines) if needle in line)


@pytest.mark.parametrize("step", STEPS)
def test_an_empty_experiment_fails_before_the_eval(step):
    """An empty experiment makes the anchored search match nothing; refuse it early.

    The check runs before ``mkdir -p "$OUTPUT_BASE_PATH"``, so nothing is created
    in the bucket, and before the eval.
    """
    run = one_script(ENVS / step / "step.yaml", "GB_ARTIFACT_ID:sage_eval_results")
    lines = run.splitlines()
    check = _line_index(lines, '[ -n "$EXPERIMENT" ]')
    assert check < _line_index(lines, 'mkdir -p "$OUTPUT_BASE_PATH"')
    assert check < _line_index(lines, "/sage/cluster/gb/scripts/")
    env = {**os.environ, "EXPERIMENT": ""}
    r = subprocess.run(
        ["bash", "-ec", f"{lines[check]}\necho REACHED"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 1 and "REACHED" not in r.stdout
    assert "experiment is empty" in r.stdout


AWS_STEP = "aws/steps/sage-eval"

# Names the AWS step accepts, and names it refuses because a glob character or a
# '.'/'..' name would let the anchored search match outside the experiment.
GOOD_EXPERIMENTS = ["exp-ckpt_1000", "eval-l40s-350m-s7500", "team/exp.v2"]
BAD_EXPERIMENTS = [
    "*",
    "exp-*",
    "exp?",
    "exp[12]",
    "..",
    "team/../other",
    "./exp",
    "/exp",
    "exp/",
    "team//exp",
    "exp 1",
    "exp$HOME",
]


def _aws_config(experiment: str) -> dict:
    """The AWS step's default config with ``experiment`` set."""
    cfg = yaml.safe_load((ENVS / AWS_STEP / "step.yaml").read_text(encoding="utf-8"))
    config = cfg["config"]
    config["sage_eval_config"]["experiment"] = experiment
    return config


def _render_aws_run(experiment: str) -> str:
    """Fill the AWS run script as the launcher does at step creation."""
    run = one_script(ENVS / AWS_STEP / "step.yaml", "GB_ARTIFACT_ID:sage_eval_results")
    return fill_template(run, {"config": _aws_config(experiment)}, strict=True)


@pytest.mark.parametrize("experiment", GOOD_EXPERIMENTS)
def test_the_aws_step_renders_with_a_safe_experiment(experiment):
    """A safe name renders, and the run script exports it unchanged."""
    assert f'export EXPERIMENT="{experiment}"' in _render_aws_run(experiment)


@pytest.mark.parametrize("experiment", ["", *BAD_EXPERIMENTS])
def test_the_aws_step_refuses_an_unsafe_experiment_at_creation(experiment):
    """An empty or unsafe name fails the template fill, before anything launches."""
    with pytest.raises(Exception) as exc:
        _render_aws_run(experiment)
    causes = []
    err: BaseException | None = exc.value
    while err is not None:
        causes.append(str(err))
        err = err.__cause__
    assert any("sage_eval_config.experiment" in c for c in causes), causes


def _aws_name_check() -> str:
    """The AWS run script's bash checks on EXPERIMENT, from ``[ -n`` to ``esac``."""
    run = one_script(ENVS / AWS_STEP / "step.yaml", "GB_ARTIFACT_ID:sage_eval_results")
    lines = run.splitlines()
    start = _line_index(lines, '[ -n "$EXPERIMENT" ]')
    end = next(i for i in range(start, len(lines)) if lines[i].strip() == "esac")
    assert end < _line_index(lines, 'mkdir -p "$OUTPUT_BASE_PATH"')
    return "\n".join(lines[start : end + 1])


def _run_name_check(experiment: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-ec", f"{_aws_name_check()}\necho REACHED"],
        env={**os.environ, "EXPERIMENT": experiment},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("experiment", GOOD_EXPERIMENTS)
def test_the_aws_bash_check_accepts_a_safe_experiment(experiment):
    """The backstop passes the names the template accepts."""
    r = _run_name_check(experiment)
    assert r.returncode == 0 and "REACHED" in r.stdout, r.stdout + r.stderr


@pytest.mark.parametrize("experiment", BAD_EXPERIMENTS)
def test_the_aws_bash_check_refuses_an_unsafe_experiment(experiment):
    """The backstop refuses the names the template refuses."""
    r = _run_name_check(experiment)
    assert r.returncode == 1 and "REACHED" not in r.stdout, r.stdout + r.stderr
    assert "sage_eval_config.experiment" in r.stdout


@pytest.mark.parametrize("step", STEPS)
def test_traversal_order_decides_between_salad_and_a_sibling_eval(step, tmp_path):
    """Known limit: the step emits *a* result under the experiment, not its own eval's.

    With SALAD next to IFEval, whichever find reaches first is emitted. A crashed eval
    can therefore pass on a sibling's file until each eval gets its own output path.
    """
    root = tmp_path / "out"
    salad = _touch(root, SALAD)
    ifeval = _touch(root, "ifeval/export-1000/metrics-all.jsonl")
    assert _emitted(step, root, tmp_path) in {str(salad), str(ifeval)}
