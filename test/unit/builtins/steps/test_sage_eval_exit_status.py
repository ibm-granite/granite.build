"""The AWS sage-eval steps must report a failed eval as a failure (#488).

The sage-eval step runs the eval as ``bash <script> 2>&1 | tee /tmp/sage-eval.log``.
Without ``set -o pipefail`` the pipeline's status was tee's, so under ``set -e`` a
failing eval script never stopped the step, and the step went on to emit any results
file under the experiment as SUCCESS. Its fallback search compared against the tee
log with ``-newer``; tee writes the log until the eval ends, so no file the eval wrote
was newer than it. The fallback now compares against a marker touched before the eval.

The sage-eval-bcb steps run the same pipeline without ``set -e``, so the eval's status
was ignored: the step wrote ``.done`` and emitted any results or predictions file it
found. They now check the eval's status right after the pipeline.

The grouped steps printed ``FAILED: <eval> (exit code: $?)`` after a ``$(date)`` in
the same word, which resets ``$?``, so the message always said 0.

These tests cut the real blocks out of each step.yaml and run them in bash, with the
eval script replaced by a stub.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

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
EXPERIMENT = "exp-ckpt_1000"
STALE = 0
ARTIFACT = re.compile(r"GB_ARTIFACT_ID:sage_eval_results GB_ARTIFACT_PATH:(.+)")


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


def _run_script(step: str) -> str:
    """The step's run script: the one string that emits sage_eval_results."""
    cfg = yaml.safe_load((ENVS / step / "step.yaml").read_text(encoding="utf-8"))
    (run,) = [s for s in _strings(cfg) if "GB_ARTIFACT_ID:sage_eval_results" in s]
    return run


def _shell_options(script: str) -> str:
    """The ``set`` lines at the top of a run script, before any other command."""
    options = []
    for line in script.splitlines():
        if re.match(r"\s*set\s+[-+]", line):
            options.append(line.strip())
        elif line.strip() and not line.strip().startswith("#"):
            break
    return "\n".join(options)


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _if_block(lines: list[str], start: int, opener: str) -> tuple[int, int]:
    """Indexes of the first ``opener`` line at or after ``start`` and of its ``fi``."""
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
    return f"{_shell_options(script)}\n{block}"


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
def test_the_fallback_ignores_a_file_older_than_the_eval(step, tmp_path):
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
    return f"{_shell_options(script)}\n{block}"


def _bcb_eval_block(step: str) -> str:
    """A bcb step's run-script lines from the eval call to the end of the script."""
    cfg = yaml.safe_load((ENVS / step / "step.yaml").read_text(encoding="utf-8"))
    launcher = cfg["environment_configs"]["Skypilot"]["default_launcher"]
    run = cfg["environment_configs"]["Skypilot"]["launchers"][launcher]["config"]["run"]
    lines = run.splitlines()
    start = next(
        i
        for i, l in enumerate(lines)
        if l.strip() == f"bash {SCRIPTS_DIR}${{GB_SCRIPT}} \\"
    )
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
    rel, artifact_id = BCB_STEPS[step]
    stub = f'mkdir -p "$OUTPUT_PATH/{Path(rel).parent}"\necho {{}} > "$OUTPUT_PATH/{rel}"\n'
    result, output = _run_bcb(step, tmp_path, stub)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        f"GB_ARTIFACT_ID:{artifact_id} GB_ARTIFACT_PATH:{output / rel}" in result.stdout
    )


@pytest.mark.parametrize("step", GROUPED_STEPS)
def test_a_grouped_step_prints_the_evals_exit_code(step, tmp_path):
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
