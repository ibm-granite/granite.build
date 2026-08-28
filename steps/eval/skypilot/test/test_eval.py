"""Tests for the exemplar eval entrypoint (../src/eval.sh).

Run from the step directory with `make test` (pytest recurses the whole tree).
The exemplar eval payload is a placeholder shell script baked into the custom
image; these cluster-agnostic unit tests invoke it directly and pin the two
contracts the step depends on: the fixed results.json output at
<output-dir>/results.json, and that the script does NOT print the Granite.build
``GB_ARTIFACT_ID`` line (the step.yaml run block registers the output instead).
"""

import json
import subprocess
from pathlib import Path

# The eval script lives beside the step, in src/.
_EVAL_SH = Path(__file__).resolve().parent.parent / "src" / "eval.sh"


def _run_eval(output_dir, model_path="some/model", tasks="", batch_size=8):
    """Invoke eval.sh with the flags the step.yaml passes.

    :param output_dir: Directory the results file is written into.
    :param model_path: Model path/id under evaluation.
    :param tasks: Comma-separated task names ("" => a placeholder task).
    :param batch_size: Per-device eval batch size.
    :returns: The CompletedProcess (stdout/stderr captured); non-zero exit raises.
    :raises subprocess.CalledProcessError: if eval.sh exits non-zero.
    """
    return subprocess.run(
        [
            "bash",
            str(_EVAL_SH),
            "--model-path",
            model_path,
            "--tasks",
            tasks,
            "--output-dir",
            str(output_dir),
            "--batch-size",
            str(batch_size),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_writes_results_at_fixed_path(tmp_path):
    """eval.sh writes results.json at <output_dir>/results.json, recording the
    model, batch size, and the requested tasks."""
    _run_eval(
        tmp_path, model_path="some/model", tasks="hellaswag,arc_easy", batch_size=4
    )

    results = tmp_path / "results.json"
    assert results.exists()
    data = json.loads(results.read_text())
    assert data["model_path"] == "some/model"
    assert data["batch_size"] == 4
    assert data["tasks"] == "hellaswag,arc_easy"


def test_defaults_to_placeholder_task(tmp_path):
    """An empty --tasks records a single 'placeholder' task rather than nothing."""
    _run_eval(tmp_path, tasks="")

    data = json.loads((tmp_path / "results.json").read_text())
    assert data["tasks"] == "placeholder"


def test_does_not_emit_artifact_line(tmp_path):
    """eval.sh must not print the artifact marker (either prefix): the step.yaml
    run block owns output registration for this fixed, single-file output."""
    proc = _run_eval(tmp_path, model_path="m")

    assert "GB_ARTIFACT_ID" not in proc.stdout
    assert "LLMB_ARTIFACT_ID" not in proc.stdout
    assert (tmp_path / "results.json").exists()
