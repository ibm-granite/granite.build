"""Unit tests for the vllm-server step's allocation lifetime.

The step ends with an unbounded `wait "$SERVER_PID"`, so the SERVICE holds its
LSF allocation until something else downs the cluster. The only thing that does
is the teardown target — and every recipe in this repo gates teardown on the
TRAINER's checkpoint artifact (`binding: <train>.checkpoint`), which a failed
trainer never emits. gbserver's build schema has no on-failure/always semantics,
so the graph cannot express "tear down regardless".

Build d77546a9 demonstrated the cost: the trainer crashed, teardown never became
eligible, and 8 H100s stayed allocated until downed by hand. SERVICE clusters on
LSF also never autostop (skypilot.py:1817 passes autostop=None for the SSH/HPC
clouds), so there is no backstop underneath.

Hence a lifetime cap the server enforces on itself: a bound on how long it will
hold an allocation with nobody reclaiming it. It is a safety net, not the normal
path — teardown is still what ends a healthy run.
"""

from pathlib import Path

import yaml

from gbserver.utils.template import fill_template

REPO_ROOT = Path(__file__).resolve().parents[4]
VLLM_STEP_YAML = (
    REPO_ROOT
    / "configurations/assets/environments/skypilot/steps/distill/vllm-server/step.yaml"
)


def _render_run(overrides=None):
    cfg = yaml.safe_load(VLLM_STEP_YAML.read_text(encoding="utf-8"))
    vllm = {**cfg["config"]["vllm_config"], **(overrides or {})}
    run = cfg["environment_configs"]["Skypilot"]["launchers"]["vllm"]["config"]["run"]
    return fill_template(run, {"config": {"vllm_config": vllm}}, strict=False)


def test_lifetime_cap_bounds_the_allocation():
    """With a cap set, the server cannot hold its node indefinitely."""
    run = _render_run({"max_lifetime_seconds": 1800})

    assert "MAX_LIFETIME=1800" in run, "the cap did not reach the script"
    assert 'sleep "$MAX_LIFETIME"' in run, "no watchdog sleeping for the cap"
    assert 'kill "$SERVER_PID"' in run, "the watchdog never reaps the server"
    # And it must SAY so, since a reaped server means the consumer failed and the
    # reason has to be readable in the log rather than inferred from a bare exit.
    assert "lifetime cap of ${MAX_LIFETIME}s reached" in run


def test_lifetime_cap_defaults_to_unbounded():
    """Default behaviour is unchanged: 0 means hold until teardown, as today.

    Existing recipes (ifrl, identityrl) rely on the server outliving long
    training runs, so a cap must be opt-in rather than a surprise ceiling.
    """
    cfg = yaml.safe_load(VLLM_STEP_YAML.read_text(encoding="utf-8"))

    assert cfg["config"]["vllm_config"]["max_lifetime_seconds"] == 0

    run = _render_run({"max_lifetime_seconds": 0})
    assert 'wait "$SERVER_PID"' in run
