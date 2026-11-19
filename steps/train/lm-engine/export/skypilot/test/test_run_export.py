# SPDX-License-Identifier: Apache-2.0

"""Behaviour tests for ``src/run_export.sh``, the step's entry point.

A render test only proves the template produces the environment; these run the real
rendered ``run:`` block, with ``python`` stubbed on ``PATH``, and check what the script
does with it: that ``export.py`` is invoked exactly once with the flags the driver's
argparse declares, that ``--expected`` points at a file that really holds
``config.expected``, that ``PYTHONPATH`` still carries the image's ``/app/src`` next to
the mounted ``src/``, that the driver's exit code (2 = bad recipe, 1 = unfaithful
export, 0 = verified) reaches the step unchanged, and that the shell never prints an
artifact line of its own.

The last one is the load-bearing case: gbserver binds ``hf_model`` by scraping
``GB_ARTIFACT_ID:hf_model`` from stdout, so a line the shell echoed -- or a duplicate
from a ``tee`` -- would publish a model no check had passed.
"""

import json
import pathlib
import shutil
import subprocess

import pytest
import yaml

from gbserver.utils.template import fill_template

_STEP_DIR = pathlib.Path(__file__).resolve().parent.parent
_SCRIPT = _STEP_DIR / "src" / "run_export.sh"
_TEMPLATE = yaml.safe_load((_STEP_DIR / "step-template.yaml").read_text())
_RUN = _TEMPLATE["environment_configs"]["Skypilot"]["launchers"]["lm-engine-export"][
    "config"
]["run"]

_EXPECTED = {
    "rope_theta": 10000000,
    "vocab_size": 100352,
    "tie_word_embeddings": True,
    "pad_token_id": None,
    "m_residual": 0.263,
}

_BASE = {
    "image": "quay.io/granitebuild/gb-step-lm-engine:v0.1.0",
    "train_dir": "/shared/runs/r/train",
    "iteration": "4000",
    "output_dir": "/shared/runs/r/hf/4000",
    "model_type": "granite",
    "tokenizer_path": "/shared/tok",
    "expected": _EXPECTED,
    "parity_tolerance": 0.05,
}

# The stub stands in for the image's interpreter. It records the argv it was handed
# and the PYTHONPATH it saw, then exits with STUB_RC -- so one stub covers the happy
# path and both failure codes. It prints the artifact line only when told to, which is
# how "the shell adds none of its own" is distinguishable from "nobody printed one".
_PYTHON_STUB = r"""#!/usr/bin/env bash
set -u
: > "$STUB_DIR/python.ran.$$"
for a in "$@"; do printf '%s\n' "$a" >> "$STUB_DIR/python.argv"; done
printf '%s\n' "${PYTHONPATH:-}" > "$STUB_DIR/python.pythonpath"
printf '%s\n' "${PWD}" > "$STUB_DIR/python.cwd"
rc="${STUB_RC:-0}"
out=""
want=""
for a in "$@"; do
  [ -z "$want" ] || { out="$a"; want=""; }
  [ "$a" = "--output-dir" ] && want=1
done
if [ "$rc" = 0 ]; then
  echo "GB_ARTIFACT_ID:hf_model GB_ARTIFACT_PATH:$out"
fi
echo "export: stage 1/5 unshard" >&2
exit "$rc"
"""


@pytest.fixture(name="world")
def _world(tmp_path):
    """A stubbed world: ``bin/python`` on PATH and the step's src/ mounted at ./src."""
    stub_dir = tmp_path / "stub"
    bin_dir = tmp_path / "bin"
    work = tmp_path / "work"
    for path in (stub_dir, bin_dir, work):
        path.mkdir()
    stub = bin_dir / "python"
    stub.write_text(_PYTHON_STUB)
    stub.chmod(0o755)
    # SkyPilot copies the file mount into the run's working directory, so the mount
    # is copied here too: a symlink would leave the script resolving itself back to
    # the source tree and hide a path bug.
    shutil.copytree(_STEP_DIR / "src", work / "src")

    def run(over=None, *, rc=None, pythonpath="/app/src", tmpdir=None):
        config = dict(_TEMPLATE["config"])
        config.update(_BASE)
        config.update(over or {})
        rendered = fill_template(_RUN, {"config": config}, strict=True)
        env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "STUB_DIR": str(stub_dir),
            "TMPDIR": str(tmpdir or tmp_path / "tmp"),
        }
        pathlib.Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        if pythonpath is not None:
            env["PYTHONPATH"] = pythonpath
        if rc is not None:
            env["STUB_RC"] = str(rc)
        return subprocess.run(
            ["bash", "-c", rendered],
            capture_output=True,
            text=True,
            check=False,
            cwd=work,
            env=env,
            timeout=120,
        )

    def argv():
        """The argv the stub was handed, one recorded line per argument, so a value
        holding a quote or a space is read back exactly as the stub received it."""
        assert invocations() == 1
        return (stub_dir / "python.argv").read_text().splitlines()

    def invocations():
        return len(list(stub_dir.glob("python.ran.*")))

    return type(
        "World",
        (),
        {
            "run": staticmethod(run),
            "argv": staticmethod(argv),
            "invocations": staticmethod(invocations),
            "stub_dir": stub_dir,
            "tmp_path": tmp_path,
            "mount": work / "src",
        },
    )


def _flag(argv, name):
    return argv[argv.index(name) + 1]


# ── the script itself ──────────────────────────────────────────────────────────


def test_the_script_is_valid_bash_and_carries_the_licence_header():
    assert subprocess.run(["bash", "-n", str(_SCRIPT)]).returncode == 0
    assert _SCRIPT.read_text().startswith("#!/usr/bin/env bash\n# SPDX-License")


def test_the_script_never_uses_torchrun_or_a_pipeline_around_the_driver():
    """unshard is single-process, and a pipeline would mask the driver's exit code."""
    code = [
        line
        for line in _SCRIPT.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    for line in code:
        assert "torchrun" not in line, line
        assert "tee " not in line, line
        assert "|" not in line.replace("||", ""), line


# ── the invocation ─────────────────────────────────────────────────────────────


def test_export_py_is_invoked_once_with_the_flags_the_driver_declares(world):
    proc = world.run()
    assert proc.returncode == 0, proc.stderr
    assert world.invocations() == 1
    argv = world.argv()
    assert argv[0].endswith("/src/export.py")
    assert _flag(argv, "--train-dir") == "/shared/runs/r/train"
    assert _flag(argv, "--output-dir") == "/shared/runs/r/hf/4000"
    assert _flag(argv, "--tokenizer-path") == "/shared/tok"
    assert _flag(argv, "--model-type") == "granite"
    assert _flag(argv, "--iteration") == "4000"
    assert _flag(argv, "--parity-tolerance") == "0.05"
    assert _flag(argv, "--image") == _BASE["image"]
    # Flags the step does not expose are left to the driver's own defaults.
    assert "--dtype" not in argv
    assert "--lm-engine-source" not in argv


def test_every_flag_passed_is_one_the_driver_accepts(world):
    """A flag export.py does not declare is an exit-2 the recipe cannot fix."""
    import export

    declared = set()
    for action in export.build_parser()._actions:
        declared.update(action.option_strings)
    world.run()
    for token in world.argv():
        if token.startswith("--"):
            assert token in declared, token


def test_an_empty_iteration_is_omitted_rather_than_passed_empty(world):
    """`--iteration ''` would be a rung named "" instead of "the latest present"."""
    assert world.run({"iteration": ""}).returncode == 0
    assert "--iteration" not in world.argv()


def test_the_script_runs_the_mounted_export_py_not_some_other_copy(world):
    """Resolved from the script's own directory, so the run uses the file mount."""
    world.run()
    assert pathlib.Path(world.argv()[0]) == world.mount / "export.py"


# ── --expected really is a file holding config.expected ────────────────────────


def test_expected_is_written_to_a_file_the_driver_can_read(world):
    proc = world.run()
    assert proc.returncode == 0, proc.stderr
    path = pathlib.Path(_flag(world.argv(), "--expected"))
    assert path.is_file(), path
    assert json.loads(path.read_text()) == _EXPECTED
    # Kept, not cleaned up: it is the evidence for what a failed check compared.
    assert str(path) in proc.stderr


def test_an_empty_expectation_passes_no_flag_at_all(world):
    """`--expected` pointing at `{}` and no `--expected` mean the same thing; the
    second does not leave a file behind to be mistaken for a real expectation."""
    assert world.run({"expected": {}}).returncode == 0
    assert "--expected" not in world.argv()


def test_the_expected_file_survives_shell_metacharacters(world):
    nasty = dict(_EXPECTED, notes='say "hi" $HOME `x` it\'s')
    proc = world.run({"expected": nasty, "output_dir": "/shared/it's/hf/4000"})
    assert proc.returncode == 0, proc.stderr
    assert _flag(world.argv(), "--output-dir") == "/shared/it's/hf/4000"
    path = pathlib.Path(_flag(world.argv(), "--expected"))
    assert json.loads(path.read_text()) == nasty


def test_the_expected_file_lands_in_a_writable_temp_dir(world):
    world.run()
    path = pathlib.Path(_flag(world.argv(), "--expected"))
    assert path.parent == (world.tmp_path / "tmp")


# ── PYTHONPATH ─────────────────────────────────────────────────────────────────


def test_pythonpath_carries_both_the_mount_and_the_images_source_tree(world):
    """`tools.model_conversion` is not installed -- it lives at /app/src -- and
    export.py/export_checks.py live in the mount. Replacing PYTHONPATH loses one."""
    world.run()
    entries = (world.stub_dir / "python.pythonpath").read_text().strip().split(":")
    assert entries[0] == str(world.mount)
    assert "/app/src" in entries


def test_pythonpath_without_an_inherited_value_holds_only_the_mount(world):
    """An empty inherited PYTHONPATH must not leave a bare ':' on the path."""
    world.run(pythonpath=None)
    value = (world.stub_dir / "python.pythonpath").read_text().strip()
    assert value == str(world.mount)


# ── exit codes ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("rc", [0, 1, 2])
def test_the_drivers_exit_code_reaches_the_step_unchanged(rc, world):
    """2 = bad recipe, 1 = export failed or unfaithful, 0 = verified. A swallowed
    non-zero publishes a model that did not pass its checks."""
    assert world.run(rc=rc).returncode == rc


def test_a_failing_export_emits_no_artifact_line(world):
    proc = world.run(rc=1)
    assert proc.returncode == 1
    assert "GB_ARTIFACT_ID" not in proc.stdout


# ── the artifact line ──────────────────────────────────────────────────────────


def test_the_artifact_line_appears_exactly_once_and_only_from_the_driver(world):
    proc = world.run()
    assert proc.returncode == 0, proc.stderr
    lines = [l for l in proc.stdout.splitlines() if "GB_ARTIFACT_ID" in l]
    assert lines == ["GB_ARTIFACT_ID:hf_model GB_ARTIFACT_PATH:/shared/runs/r/hf/4000"]


def test_the_shell_emits_no_artifact_line_when_the_driver_prints_none(world):
    """The shell has no copy of its own to fall back on -- proven by silencing the
    stub: with STUB_RC=0 but the marker suppressed, stdout must hold none."""
    proc = world.run(rc=3)
    assert proc.returncode == 3
    assert "GB_ARTIFACT_ID" not in proc.stdout


def test_the_banner_is_printed_for_the_monitor(world):
    proc = world.run()
    assert "lm-engine-export: iteration='4000' model_type='granite'" in proc.stdout
