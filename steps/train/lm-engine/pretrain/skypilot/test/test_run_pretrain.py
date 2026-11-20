# SPDX-License-Identifier: Apache-2.0

"""Behaviour tests for ``src/run_pretrain.sh``, the step's entry point.

A render test only proves the template produces the environment; these run the
script, with ``python`` and ``torchrun`` stubbed on ``PATH`` and a throwaway
``TRAIN_DIR``, and check what it does with it: that the checkpoint watcher starts
on rank 0 and only when emission is on, that a worker waits for rank 0's resume
decision instead of writing one, that the trainer's exit code reaches the step,
that the watcher's stop file is written even when the trainer fails, and that a
worker exits 0 rather than inheriting a rank test's status.
"""

import os
import pathlib
import subprocess
import time

import pytest

_STEP_DIR = pathlib.Path(__file__).resolve().parent.parent
_SCRIPT = _STEP_DIR / "src" / "run_pretrain.sh"
_TOKEN = "gb-build-tgt-abcdef123456-r2"

# One stub for all four helpers, dispatching on the module name the script
# passes, so the test exercises the real argv the script builds. ``-I`` is
# dropped the way an interpreter would consume it.
_PYTHON_STUB = r"""#!/usr/bin/env bash
set -u
args=()
for a in "$@"; do [ "$a" = "-I" ] || args+=("$a"); done
mod="$(basename "${args[0]}")"
rest=("${args[@]:1}")
value_of() {
  local want="$1" seen=""
  for a in "${rest[@]}"; do
    if [ -n "$seen" ]; then printf '%s' "$a"; return 0; fi
    [ "$a" = "$want" ] && seen=1
  done
}
echo "stub-python: $mod ${rest[*]}"
case "$mod" in
  data_check.py)
    if [ -n "${STUB_DATA_CHECK_RC:-}" ] && [ "${STUB_DATA_CHECK_RC}" != 0 ]; then
      echo "error: dataset_path is empty" >&2
      exit "$STUB_DATA_CHECK_RC"
    fi
    echo "data-check: ok"
    ;;
  resume.py)
    out="$(value_of --out)"
    : > "$STUB_DIR/resume.ran"
    printf '%s' "$(value_of --state-file)" > "$STUB_DIR/resume.state-file"
    if [ -n "${STUB_RESUME_ERROR:-}" ]; then
      printf '%s\n' '{"load": null, "seeded": false, "error": "no progress"}' > "$out"
      echo "error: no progress" >&2
      exit 1
    fi
    printf '%s\n' '{"load": null, "seeded": false, "error": null}' > "$out"
    ;;
  render_config.py)
    printf '%s' "$(value_of --out)" > "$STUB_DIR/render.out-path"
    printf 'stub: rendered config\n' > "$(value_of --out)"
    ;;
  ckpt_watch.py)
    stop="$(value_of --stop-file)"
    : > "$STUB_DIR/watch.ran"
    echo "stub-watcher: watching for $(value_of --rungs)"
    while [ ! -e "$stop" ]; do sleep 0.05; done
    echo "GB_ARTIFACT_ID:checkpoint_2000 GB_ARTIFACT_PATH:/x/global_step2000"
    ;;
  *)
    echo "stub-python: unexpected module $mod" >&2
    exit 97
    ;;
esac
"""

_TORCHRUN_STUB = r"""#!/usr/bin/env bash
set -u
: > "$STUB_DIR/torchrun.ran"
printf '%s\n' "stub-torchrun: $*" > "$STUB_DIR/torchrun.argv"
echo "step = 1 loss = 9.0"
[ -z "${STUB_TORCHRUN_SLEEP:-}" ] || sleep "$STUB_TORCHRUN_SLEEP"
exit "${STUB_TORCHRUN_RC:-0}"
"""


@pytest.fixture(name="world")
def _world(tmp_path):
    """A stubbed world: ``bin/`` on PATH, an empty TRAIN_DIR, and a runner."""
    if not (bash := subprocess.run(["bash", "-c", "true"]).returncode == 0):
        pytest.skip("bash not available")  # pragma: no cover
    assert bash
    stub_dir = tmp_path / "stub"
    bin_dir = tmp_path / "bin"
    train_dir = tmp_path / "train"
    for path in (stub_dir, bin_dir, train_dir):
        path.mkdir()
    for name, body in (("python", _PYTHON_STUB), ("torchrun", _TORCHRUN_STUB)):
        script = bin_dir / name
        script.write_text(body)
        script.chmod(0o755)

    def environment(*, rank="0", emit="false", phase="train", nnodes="1", **extra):
        env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "STUB_DIR": str(stub_dir),
            "TRAIN_DIR": str(train_dir),
            "LME_PHASE": phase,
            "LME_KNOBS_JSON": '{"experiment": "e"}',
            "LME_RUNGS": "2000",
            "LME_EMIT": emit,
            "LME_WATCH_INTERVAL": "1",
            "LME_RESUME_SRC": "",
            "LME_REQUIRE_PROGRESS": "true",
            "GB_ATTEMPT_TOKEN": _TOKEN,
            "SKYPILOT_NODE_RANK": rank,
            "SKYPILOT_NUM_NODES": nnodes,
            "SKYPILOT_NUM_GPUS_PER_NODE": "2",
            "SKYPILOT_NODE_IPS": "10.0.0.1\n10.0.0.2",
            "NODE_NAME": "vela-node-7",
            "LME_DECISION_WAIT_SECONDS": "3",
        }
        env.update(extra)
        return env

    def run(**kwargs):
        return subprocess.run(
            ["bash", str(_SCRIPT)],
            capture_output=True,
            text=True,
            check=False,
            cwd=tmp_path,
            env=environment(**kwargs),
            timeout=120,
        )

    def popen(**kwargs):
        return subprocess.Popen(
            ["bash", str(_SCRIPT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=tmp_path,
            env=environment(**kwargs),
        )

    def decide():
        """Pre-write rank 0's decision, as a worker would find it."""
        (train_dir / f".resume-decision.{_TOKEN}.json").write_text(
            '{"load": null, "seeded": false, "error": null}\n'
        )

    return type(
        "World",
        (),
        {
            "run": staticmethod(run),
            "popen": staticmethod(popen),
            "decide": staticmethod(decide),
            "train_dir": train_dir,
            "stub_dir": stub_dir,
            "tmp_path": tmp_path,
        },
    )


# ── the script itself ──────────────────────────────────────────────────────────


def test_the_script_is_valid_bash_and_executable():
    assert subprocess.run(["bash", "-n", str(_SCRIPT)]).returncode == 0
    assert os.access(_SCRIPT, os.X_OK)
    assert _SCRIPT.read_text().startswith("#!/usr/bin/env bash\n# SPDX-License")


# ── rank 0, the happy path ─────────────────────────────────────────────────────


def test_rank_0_renders_a_config_and_emits_the_final_artifact(world):
    proc = world.run()
    assert proc.returncode == 0, proc.stderr
    assert "lm-engine: node_rank=0 nnodes=1 gpus=2 master=10.0.0.1" in proc.stdout
    assert "k8s_node=vela-node-7" in proc.stdout
    assert f"GB_ARTIFACT_ID:checkpoint GB_ARTIFACT_PATH:{world.train_dir}" in (
        proc.stdout
    )
    # The rendered config is per node and per attempt, and it is echoed.
    out_path = (world.stub_dir / "render.out-path").read_text()
    assert out_path == f"{world.train_dir}/configs/lme_config.{_TOKEN}.node0.yml"
    assert "stub: rendered config" in proc.stdout
    # The trainer got the same file, and its output was teed per node.
    assert f"--config {out_path}" in (world.stub_dir / "torchrun.argv").read_text()
    assert (world.train_dir / "logs" / "node0-vela-node-7.log").is_file()


def test_rank_0_writes_the_decision_from_a_build_stable_state_file(world):
    proc = world.run()
    assert proc.returncode == 0, proc.stderr
    assert (world.stub_dir / "resume.ran").is_file()
    # Per attempt: the decision file name carries the attempt token.
    assert (world.train_dir / f".resume-decision.{_TOKEN}.json").is_file()
    # Per build: the progress baseline must outlive one attempt, so no token.
    state = (world.stub_dir / "resume.state-file").read_text()
    assert state == f"{world.train_dir}/.resume-state.json"
    assert _TOKEN not in state


def test_a_failing_resume_decision_fails_the_step(world):
    proc = world.run(STUB_RESUME_ERROR="1")
    assert proc.returncode == 1
    assert "no progress" in proc.stderr
    assert not (world.stub_dir / "torchrun.ran").exists()


# ── the watcher ────────────────────────────────────────────────────────────────


def test_the_watcher_runs_on_rank_0_when_emission_is_on(world):
    proc = world.run(emit="true")
    assert proc.returncode == 0, proc.stderr
    assert (world.stub_dir / "watch.ran").is_file()
    # Its stdout reaches the step's log, which is what dispatches the export target.
    assert "GB_ARTIFACT_ID:checkpoint_2000" in proc.stdout
    assert (world.train_dir / f".train-exited.{_TOKEN}").is_file()


def test_no_watcher_when_emission_is_off(world):
    proc = world.run(emit="false")
    assert proc.returncode == 0, proc.stderr
    assert not (world.stub_dir / "watch.ran").exists()
    assert not (world.train_dir / f".train-exited.{_TOKEN}").exists()


def test_no_watcher_on_a_worker_even_with_emission_on(world):
    world.decide()
    proc = world.run(rank="1", emit="true", nnodes="2")
    assert proc.returncode == 0, proc.stderr
    assert not (world.stub_dir / "watch.ran").exists()


def test_the_stop_file_is_written_when_the_trainer_fails(world):
    """Otherwise the watcher polls until the whole job is killed."""
    proc = world.run(emit="true", STUB_TORCHRUN_RC="3")
    assert proc.returncode == 3
    assert (world.train_dir / f".train-exited.{_TOKEN}").is_file()
    assert "GB_ARTIFACT_ID:checkpoint_2000" in proc.stdout


def test_the_exit_trap_releases_the_watcher_when_the_script_is_signalled(world):
    """A SIGTERM mid-run must release the watcher too, not only a clean exit."""
    proc = world.popen(emit="true", STUB_TORCHRUN_SLEEP="2")
    for _ in range(200):  # wait for the watcher to be up
        if (world.stub_dir / "watch.ran").exists():
            break
        time.sleep(0.05)
    assert (world.stub_dir / "watch.ran").is_file()
    proc.terminate()
    proc.communicate(timeout=60)
    assert (world.train_dir / f".train-exited.{_TOKEN}").is_file()


# ── exit codes ─────────────────────────────────────────────────────────────────


def test_the_trainers_exit_code_reaches_the_step_and_nothing_is_emitted(world):
    proc = world.run(STUB_TORCHRUN_RC="3")
    assert proc.returncode == 3
    assert "GB_ARTIFACT_ID:checkpoint GB_ARTIFACT_PATH:" not in proc.stdout


def test_a_worker_waits_for_the_decision_and_exits_0(world):
    """`[ "$RANK" = 0 ] && echo` as the last line would exit 1 on every worker."""
    world.decide()
    proc = world.run(rank="1", nnodes="2")
    assert proc.returncode == 0, proc.stderr
    # It read rank 0's decision rather than writing one.
    assert not (world.stub_dir / "resume.ran").exists()
    assert "GB_ARTIFACT_ID:checkpoint GB_ARTIFACT_PATH:" not in proc.stdout
    assert (world.stub_dir / "torchrun.ran").is_file()
    assert "--node_rank=1" in (world.stub_dir / "torchrun.argv").read_text()


def test_a_worker_times_out_cleanly_without_a_decision(world):
    proc = world.run(rank="1", nnodes="2", LME_DECISION_WAIT_SECONDS="1")
    assert proc.returncode == 1
    assert "no resume decision" in proc.stderr
    assert not (world.stub_dir / "torchrun.ran").exists()


# ── the data-check phase ───────────────────────────────────────────────────────


def test_the_data_check_phase_execs_and_never_reaches_the_trainer(world):
    proc = world.run(phase="data-check")
    assert proc.returncode == 0, proc.stderr
    assert "data-check: ok" in proc.stdout
    assert not (world.stub_dir / "torchrun.ran").exists()
    assert not (world.stub_dir / "resume.ran").exists()


def test_a_passing_data_check_emits_its_artifact(world):
    """Without this line `train` never dispatches.

    A binding is the only thing that orders two targets, and an output comes into
    existence only when its workload prints GB_ARTIFACT_ID for it. A recipe that
    orders `train` after a `data-check` target binds this artifact, so a silent
    data-check leaves the build waiting on a target that can never arrive.
    """
    proc = world.run(phase="data-check")
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if "GB_ARTIFACT_ID:" in ln]
    assert lines == [f"GB_ARTIFACT_ID:data_check GB_ARTIFACT_PATH:{world.train_dir}"]
    # the trainer's own artifact must not leak into the check phase
    assert "GB_ARTIFACT_ID:checkpoint " not in proc.stdout


def test_a_failing_data_check_emits_no_artifact(world):
    """A failed check must not publish an output a downstream target can bind."""
    proc = world.run(phase="data-check", STUB_DATA_CHECK_RC="1")
    assert proc.returncode == 1
    assert "GB_ARTIFACT_ID:" not in proc.stdout
