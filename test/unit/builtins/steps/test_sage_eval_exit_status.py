"""The AWS sage-eval steps must report a failed eval as a failure (#488).

Each test cuts a block out of a step.yaml and runs it in bash with a stub eval.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

from .sage_eval_steps import one_script, shell_options

REPO_ROOT = Path(__file__).resolve().parents[4]
ENVS = REPO_ROOT / "configurations/assets/environments/skypilot"

SAGE_EVAL_STEPS = [
    "aws/steps/sage-eval",
]
# Each bcb step, and the file its stub eval writes on success (relative to
# OUTPUT_PATH) with the artifact id the step emits for it.
BCB_STEPS = {
    "aws/steps/sage-eval-bcb": ("bcb/code-metrics.json", "sage_eval_results"),
    "aws/steps/sage-eval-bcb-generate": ("bcb/task-predictions.jsonl", "predictions"),
    "aws/steps/sage-eval-bcb-execute": ("bcb/code-metrics.json", "sage_eval_results"),
}
GROUPED_STEPS = [
    "aws/steps/sage-eval-code-grouped",
    "aws/steps/sage-eval-multilingual-grouped",
    "aws/steps/sage-eval-olmes-grouped",
    "aws/steps/sage-eval-safety-grouped",
]

SCRIPTS_DIR = "/workspace/sage/sage/cluster/gb/scripts/"
# The line each bcb step runs its eval with.
BCB_EVAL_CALL = f"bash {SCRIPTS_DIR}${{GB_SCRIPT}} \\"
EXPERIMENT = "exp-ckpt_1000"
STALE = 0
ARTIFACT = re.compile(r"GB_ARTIFACT_ID:sage_eval_results GB_ARTIFACT_PATH:(.+)")


def _run_script(step: str) -> str:
    """Return a sage-eval or grouped step's run script.

    Args:
        step: A step path under ``ENVS``.

    Returns:
        The one script in the step that emits ``sage_eval_results``.
    """
    return one_script(ENVS / step / "step.yaml", "GB_ARTIFACT_ID:sage_eval_results")


def _indent(line: str) -> int:
    """Return the number of leading whitespace characters in ``line``.

    Args:
        line: One line of a script.

    Returns:
        The indentation width.
    """
    return len(line) - len(line.lstrip())


def _if_block(lines: list[str], start: int, opener: str) -> tuple[int, int]:
    """Find an ``if`` block in a script.

    Args:
        lines: The script's lines.
        start: The index to start looking from.
        opener: A substring of the block's ``if`` line.

    Returns:
        The indexes of the first ``opener`` line at or after ``start``, and of the
        ``fi`` at the same indentation that closes it.

    Raises:
        StopIteration: If no line contains ``opener``, or the block has no ``fi``.
    """
    head = next(i for i in range(start, len(lines)) if opener in lines[i])
    end = next(
        i
        for i in range(head + 1, len(lines))
        if lines[i].strip() == "fi" and _indent(lines[i]) == _indent(lines[head])
    )
    return head, end


def _eval_and_emit_block(step: str) -> str:
    """A sage-eval step's lines from the pre-eval marker to the emit's ``fi``,
    under the step's own shell options."""
    script = _run_script(step)
    lines = script.splitlines()
    start = next(i for i, l in enumerate(lines) if "EVAL_START_MARKER=" in l)
    _, end = _if_block(lines, start, 'if [ -n "$RESULT_FILE" ]')
    block = "\n".join(lines[start : end + 1])
    assert "{{" not in block and "{%" not in block, block
    return f"{shell_options(script)}\n{block}"


def _run_sage_eval(step: str, tmp_path: Path, stub: str):
    """Run the eval-and-emit block of ``step`` with ``stub`` as the eval script."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "stub.sh").write_text(stub)
    out = tmp_path / "out"
    (out / EXPERIMENT).mkdir(parents=True, exist_ok=True)
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()
    block = _eval_and_emit_block(step)
    assert SCRIPTS_DIR in block and "/tmp/sage-eval.log" in block
    block = block.replace(SCRIPTS_DIR, f"{scripts}/")
    block = block.replace("/tmp/sage-eval.log", str(tmp_path / "sage-eval.log"))
    env = {
        "PATH": os.environ["PATH"],
        "TMPDIR": str(tmpdir),
        "OUTPUT_BASE_PATH": str(out),
        "EXPERIMENT": EXPERIMENT,
        "GB_SCRIPT": "stub.sh",
    }
    result = subprocess.run(
        ["bash", "-c", block], env=env, capture_output=True, text=True, check=False
    )
    m = ARTIFACT.search(result.stdout)
    return result, out, (m.group(1) if m else None)


def _stale_result(out: Path) -> Path:
    """A results file from an earlier eval under the same experiment."""
    p = out / EXPERIMENT / "safety-eval/attaq/results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}")
    os.utime(p, (STALE, STALE))
    return p


@pytest.mark.parametrize("step", SAGE_EVAL_STEPS)
def test_a_failing_eval_script_fails_the_step(step, tmp_path):
    """With a sibling eval's results file on disk, a crashed eval must still fail,
    with the script's exit status and no artifact emitted."""
    out = tmp_path / "out"
    _stale_result(out)
    result, _, emitted = _run_sage_eval(
        step, tmp_path, "echo 'vLLM died at startup'\nexit 3\n"
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert emitted is None


@pytest.mark.parametrize("step", SAGE_EVAL_STEPS)
def test_a_passing_eval_emits_its_result(step, tmp_path):
    """A passing eval emits the results file it wrote under the experiment."""
    stub = f'mkdir -p "$OUTPUT_BASE_PATH/{EXPERIMENT}/ifeval"\n'
    stub += f'echo {{}} > "$OUTPUT_BASE_PATH/{EXPERIMENT}/ifeval/metrics-all.jsonl"\n'
    result, out, emitted = _run_sage_eval(step, tmp_path, stub)
    assert result.returncode == 0, result.stdout + result.stderr
    assert emitted == str(out / EXPERIMENT / "ifeval/metrics-all.jsonl")


@pytest.mark.parametrize("step", SAGE_EVAL_STEPS)
def test_the_fallback_finds_a_file_written_by_this_eval(step, tmp_path):
    """Only the fallback matches '*.csv'. The eval prints after writing the file, so
    the tee log is newer than the file; the pre-eval marker is older."""
    stub = f'mkdir -p "$OUTPUT_BASE_PATH/{EXPERIMENT}/code-eval"\n'
    stub += f'echo a,b > "$OUTPUT_BASE_PATH/{EXPERIMENT}/code-eval/summary.csv"\n'
    stub += "echo 'eval finished'\n"
    result, out, emitted = _run_sage_eval(step, tmp_path, stub)
    assert result.returncode == 0, result.stdout + result.stderr
    assert emitted == str(out / EXPERIMENT / "code-eval/summary.csv")


@pytest.mark.parametrize("step", SAGE_EVAL_STEPS)
def test_the_fallback_ignores_a_new_file_from_another_experiment(step, tmp_path):
    """A rung running in parallel under the same output dir writes a matching file
    during this eval; the fallback must not emit it as this rung's result."""
    stub = 'mkdir -p "$OUTPUT_BASE_PATH/exp-ckpt_2000/code-eval"\n'
    stub += 'echo a,b > "$OUTPUT_BASE_PATH/exp-ckpt_2000/code-eval/summary.csv"\n'
    stub += "echo 'eval finished'\n"
    result, _, emitted = _run_sage_eval(step, tmp_path, stub)
    assert result.returncode == 1, result.stdout + result.stderr
    assert emitted is None


@pytest.mark.parametrize("step", SAGE_EVAL_STEPS)
def test_the_fallback_ignores_a_file_older_than_the_eval(step, tmp_path):
    """A matching file from before the eval is not emitted, so the step fails."""
    out = tmp_path / "out"
    old = out / EXPERIMENT / "code-eval/old.csv"
    old.parent.mkdir(parents=True)
    old.write_text("a,b")
    os.utime(old, (STALE, STALE))
    result, _, emitted = _run_sage_eval(step, tmp_path, "echo 'eval finished'\n")
    assert result.returncode == 1, result.stdout + result.stderr
    assert emitted is None


def _failure_branch(step: str) -> str:
    """A grouped step's ``if bash <eval> | tee ...; then ... else ... fi`` block,
    under the step's own shell options."""
    script = _run_script(step)
    lines = script.splitlines()
    start, end = _if_block(lines, 0, f"if bash {SCRIPTS_DIR}")
    block = "\n".join(lines[start : end + 1])
    return f"{shell_options(script)}\n{block}"


def _bcb_eval_block(step: str) -> str:
    """A bcb step's run-script lines from the eval call to the end of the script.

    The run script is the one script in the step that calls the eval, found by
    content like the other steps' scripts.
    """
    run = one_script(ENVS / step / "step.yaml", BCB_EVAL_CALL)
    lines = run.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == BCB_EVAL_CALL)
    block = "\n".join(lines[start:])
    assert "{{" not in block and "{%" not in block, block
    return block


def _run_bcb(step: str, tmp_path: Path, stub: str):
    """Run the eval block of a bcb ``step`` with ``stub`` as the eval script."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "stub.sh").write_text(stub)
    output = tmp_path / "out"
    output.mkdir(exist_ok=True)
    block = _bcb_eval_block(step).replace(SCRIPTS_DIR, f"{scripts}/")
    block = block.replace("tee /tmp/", f"tee {tmp_path}/")
    env = {
        "PATH": os.environ["PATH"],
        "OUTPUT_PATH": str(output),
        "EVAL_NAME": "code-olmes-bigcodebench",
        "GB_SCRIPT": "stub.sh",
        "MODEL_SOURCE_S3": "s3://bucket/model",
    }
    result = subprocess.run(
        ["bash", "-c", block], env=env, capture_output=True, text=True, check=False
    )
    return result, output


@pytest.mark.parametrize("step", BCB_STEPS)
def test_a_failing_bcb_eval_fails_the_step(step, tmp_path):
    """With an earlier run's output on disk, a crashed eval must fail the step with
    its exit status, emit no artifact and write no .done marker."""
    rel, _ = BCB_STEPS[step]
    stale = tmp_path / "out" / rel
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    result, output = _run_bcb(step, tmp_path, "echo 'vLLM died at startup'\nexit 3\n")
    assert result.returncode == 3, result.stdout + result.stderr
    assert "(exit code: 3)" in result.stdout
    assert "GB_ARTIFACT_ID:" not in result.stdout
    assert not (output / "code-olmes-bigcodebench.done").exists()


@pytest.mark.parametrize("step", BCB_STEPS)
def test_a_passing_bcb_eval_emits_its_output(step, tmp_path):
    """A passing eval emits the results or predictions file it wrote."""
    rel, artifact_id = BCB_STEPS[step]
    stub = f'mkdir -p "$OUTPUT_PATH/{Path(rel).parent}"\necho {{}} > "$OUTPUT_PATH/{rel}"\n'
    result, output = _run_bcb(step, tmp_path, stub)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        f"GB_ARTIFACT_ID:{artifact_id} GB_ARTIFACT_PATH:{output / rel}" in result.stdout
    )


@pytest.mark.parametrize("step", GROUPED_STEPS)
def test_a_grouped_step_prints_the_evals_exit_code(step, tmp_path):
    """A failed eval in a group is reported with its own exit code, counted as a
    failure, and gets no .done marker."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "stub.sh").write_text("exit 7\n")
    output = tmp_path / "out"
    output.mkdir()
    block = _failure_branch(step).replace(SCRIPTS_DIR, f"{scripts}/")
    prelude = 'FAIL_COUNT=0\nFAILED_NAMES=""\neval_name=ifeval\ngb_script=stub.sh\n'
    epilogue = 'echo "count=${FAIL_COUNT} names=${FAILED_NAMES}"\n'
    env = {"PATH": os.environ["PATH"], "OUTPUT_PATH": str(output)}
    result = subprocess.run(
        ["bash", "-c", f"{prelude}{block}\n{epilogue}"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "FAILED: ifeval (exit code: 7)" in result.stdout
    assert "count=1 names= ifeval" in result.stdout
    assert not (output / "ifeval.done").exists()
