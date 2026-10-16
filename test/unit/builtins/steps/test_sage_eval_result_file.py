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

REPO_ROOT = Path(__file__).resolve().parents[4]
ENVS = REPO_ROOT / "configurations/assets/environments/skypilot"

# The steps under test, and the variable each roots its search at. aws/sage-eval
# carries the same search but is not changed until it can be tested on AWS.
STEPS = {
    "lsf/ibm-bluevela/steps/sage-eval": "OUTPUT_BASE_PATH",
}

EXPERIMENT = "exp-ckpt_1000"
SALAD = "safety-eval/salad-bench/export-1000/base_set-result.json"


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


def _search_block(step: str) -> str:
    """The step's run-script lines from the first results search to the emit's `fi`."""
    cfg = yaml.safe_load((ENVS / step / "step.yaml").read_text())
    (run,) = [s for s in _strings(cfg) if "GB_ARTIFACT_ID:sage_eval_results" in s]
    lines = run.splitlines()
    start = next(i for i, l in enumerate(lines) if "RESULT_FILE=$(find" in l)
    emit = next(i for i, l in enumerate(lines) if 'if [ -n "$RESULT_FILE" ]' in l)
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
    # The fallback search compares against the eval's log; point it at a private copy.
    block = _search_block(step)
    assert "/tmp/sage-eval.log" in block
    block = block.replace("/tmp/sage-eval.log", str(log))
    env = {**os.environ, STEPS[step]: str(root), "EXPERIMENT": EXPERIMENT}
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
