# SPDX-License-Identifier: Apache-2.0

"""Render tests for the lm-engine-pretrain launcher.

The ``run:`` block is filled with gbserver's strict renderer
(``fill_template(..., strict=True)``, the launcher-config fill in
``targetsteprun.py``: StrictUndefined plus the ``raise_error`` global), and the
result is compared against ``golden_run_scripts.json`` for three shapes: a
single-node train, a two-node train with mid-run emission and cluster NCCL
facts, and the data-check phase. The golden file makes an accidental change to
the job's environment visible in review.

The quoting test runs the rendered block for real, with a stub
``src/run_pretrain.sh`` that dumps its environment, so what the script receives
is checked after bash has unquoted it. Mode 1 only (no build).
"""

import json
import pathlib
import re
import subprocess

import pytest
import yaml

from gbserver.types.stepconfig import StepEnvironmentTypeConfig
from gbserver.utils.template import fill_objtemplate, fill_template

_STEP_DIR = pathlib.Path(__file__).resolve().parent.parent
_TEMPLATE = yaml.safe_load((_STEP_DIR / "step-template.yaml").read_text())
_SKY = _TEMPLATE["environment_configs"]["Skypilot"]
_LAUNCHER = _SKY["launchers"]["lm-engine-pretrain"]["config"]
_GOLDEN_PATH = pathlib.Path(__file__).resolve().parent / "golden_run_scripts.json"
_GOLDEN = json.loads(_GOLDEN_PATH.read_text())

_STUB = (
    "#!/usr/bin/env bash\n"
    "python3 -c 'import json,os; print(json.dumps({k: v for k, v in "
    'os.environ.items() if k.split("_")[0] in ("LME", "TRAIN", "NCCL", "WANDB", "GB")}))\'\n'
)

# The config each golden case renders from, as overrides on the template defaults.
# train_dir is required by the render, so every case sets it.
_CASES = {
    "single_node_train": {
        "image": "quay.io/granitebuild/gb-step-lm-engine:v0.1.0",
        "train_dir": "/shared/runs/g4-350m/train",
        "lme": {"experiment": "g4-350m", "max_steps": 4000},
    },
    "two_node_train_emit": {
        "image": "quay.io/granitebuild/gb-step-lm-engine:v0.1.0",
        "train_dir": "/shared/runs/g4-350m/train",
        "lme": {"experiment": "g4-350m", "max_steps": 4000},
        "ckpt_rungs": "2000,4000",
        "emit_checkpoint_artifacts": True,
        "checkpoint_watch_interval_seconds": 120,
        "resume_from_checkpoint_dir": "/shared/runs/earlier/train",
        "nccl_debug": "INFO",
        "nccl_env": {
            "NCCL_SOCKET_IFNAME": "eth0",
            "NCCL_IB_HCA": "mlx5_0,mlx5_1",
            "NCCL_CROSS_NIC": 1,
        },
        "wandb": {
            "project": "g4-pretrain",
            "mode": "online",
            "name": "g4-350m",
            "entity": "granite",
        },
        "compute_config": {"num_nodes": 2, "num_gpus_per_node": 8},
    },
    "data_check": {
        "image": "quay.io/granitebuild/gb-step-lm-engine:v0.1.0",
        "train_dir": "/shared/runs/g4-350m/train",
        "phase": "data-check",
        "lme": {"experiment": "g4-350m", "dataset_path": "/shared/corpus"},
    },
}


def _cfg(over: dict | None = None) -> dict:
    cfg = json.loads(json.dumps(_TEMPLATE["config"]))
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            cfg[key] = dict(cfg[key], **value)
        else:
            cfg[key] = value
    return cfg


def _render(over: dict | None = None) -> str:
    return fill_template(_LAUNCHER["run"], {"config": _cfg(over)}, strict=True)


def _render_error(over: dict | None = None) -> str:
    """The ``raise_error`` message a failing fill carries."""
    with pytest.raises(RuntimeError) as exc:
        fill_objtemplate(_LAUNCHER, {"config": _cfg(over)}, strict=True)
    err = exc.value
    while err.__cause__ is not None:
        err = err.__cause__
    assert isinstance(err, ValueError), repr(err)
    return str(err)


# ── the goldens ────────────────────────────────────────────────────────────────


def test_the_golden_file_covers_exactly_the_cases():
    assert set(_GOLDEN) == set(_CASES)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_rendered_run_block_matches_its_golden(case):
    """Regenerate with `python test/golden_run_scripts.py` after a deliberate change."""
    assert _render(_CASES[case]).splitlines() == _GOLDEN[case]


@pytest.mark.parametrize("case", sorted(_CASES))
def test_every_rendered_run_block_is_valid_bash(case):
    proc = subprocess.run(["bash", "-n"], input=_render(_CASES[case]), text=True)
    assert proc.returncode == 0


def test_an_undefined_value_fails_the_render():
    cfg = _cfg(_CASES["single_node_train"])
    del cfg["nccl_debug"]
    with pytest.raises(RuntimeError):
        fill_template(_LAUNCHER["run"], {"config": cfg}, strict=True)


# ── what the environment says ──────────────────────────────────────────────────


def test_a_single_node_run_disables_ib_and_skips_the_cluster_facts():
    run = _render(dict(_CASES["single_node_train"], nccl_env={"NCCL_IB_HCA": "mlx5_0"}))
    assert "export NCCL_IB_DISABLE=1" in run
    assert "NCCL_IB_HCA" not in run


def test_a_multi_node_run_exports_the_cluster_facts_verbatim():
    run = _render(_CASES["two_node_train_emit"])
    assert "export NCCL_IB_DISABLE=1" not in run
    # dictsort, so the goldens and a review diff are stable.
    assert run.index("export NCCL_CROSS_NIC=") < run.index("export NCCL_IB_HCA=")
    assert "export NCCL_SOCKET_IFNAME='eth0'" in run
    assert "export NCCL_DEBUG='INFO'" in run


def test_the_code_facts_are_the_steps_own_defaults():
    run = _render(_CASES["single_node_train"])
    for fact in (
        "export NCCL_TIMEOUT=10800000",
        "export CUDA_DEVICE_MAX_CONNECTIONS=1",
        "export PYTORCH_ALLOC_CONF=expandable_segments:True",
        "export OMP_NUM_THREADS=1",
        "export TOKENIZERS_PARALLELISM=false",
        "export TRITON_CACHE_DIR=/tmp/triton",
        "export HOME=/tmp",
        "export NCCL_IGNORE_CPU_AFFINITY=1",
        "export NCCL_SOCKET_NTHREADS=2",
        "export CUDA_DEVICE_ORDER=PCI_BUS_ID",
        "export CUDA_MODULE_LOADING=LAZY",
        "export NCCL_ALGO=Ring",
    ):
        assert fact in run, fact


def test_wandb_is_rendered_only_when_a_project_is_set():
    assert "WANDB_" not in _render(_CASES["single_node_train"])
    run = _render(_CASES["two_node_train_emit"])
    assert "export WANDB_PROJECT='g4-pretrain'" in run
    assert "export WANDB_MODE='online'" in run
    assert "export WANDB_ENTITY='granite'" in run


def test_the_attempt_token_changes_per_attempt_not_per_build():
    """A relaunch replays the targetrun and launch ids; only the cluster name moves."""
    run = _render(_CASES["single_node_train"])
    line = next(l for l in run.splitlines() if l.startswith("export GB_ATTEMPT_TOKEN="))
    assert "GB_SKYPILOT_CLUSTER_NAME" in line
    assert "GB_TARGETRUN_ID" not in line


def test_the_run_block_hands_off_to_the_entry_point():
    assert _render(_CASES["single_node_train"]).endswith("bash ./src/run_pretrain.sh")
    assert (_STEP_DIR / "src" / "run_pretrain.sh").is_file()


# ── render-time refusals ───────────────────────────────────────────────────────


def test_an_empty_train_dir_is_refused_when_the_step_is_created():
    assert "config.train_dir is required" in _render_error(
        dict(_CASES["single_node_train"], train_dir="")
    )


def test_an_unknown_phase_is_refused():
    assert "config.phase must be" in _render_error(
        dict(_CASES["single_node_train"], phase="finetune")
    )


def test_a_non_mapping_compute_config_is_refused():
    assert "compute_config must be a mapping" in _render_error(
        dict(_CASES["single_node_train"], compute_config="2")
    )


# ── what bash actually receives ────────────────────────────────────────────────


def test_quotes_and_json_survive_the_shell(tmp_path):
    over = dict(
        _CASES["two_node_train_emit"],
        train_dir="/shared/it's/train",
        lme={"experiment": "a'b", "notes": 'say "hi" $HOME `x`'},
        nccl_env={"NCCL_TOPO_FILE": "/etc/topo it's.xml"},
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "run_pretrain.sh").write_text(_STUB)
    proc = subprocess.run(
        ["bash", "-c", "set -eu\n" + _render(over)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    env = json.loads(proc.stdout.strip().splitlines()[-1])
    assert env["TRAIN_DIR"] == "/shared/it's/train"
    assert json.loads(env["LME_KNOBS_JSON"]) == over["lme"]
    assert env["NCCL_TOPO_FILE"] == "/etc/topo it's.xml"
    assert env["LME_EMIT"] == "true"
    assert env["LME_REQUIRE_PROGRESS"] == "true"
    assert env["LME_RUNGS"] == "2000,4000"
    assert env["LME_WATCH_INTERVAL"] == "120"
    assert env["LME_RESUME_SRC"] == "/shared/runs/earlier/train"
    assert env["LME_PHASE"] == "train"
    # Derived in the shell, from the one launch var that carries the attempt.
    assert env["GB_ATTEMPT_TOKEN"] == "local"


def test_the_attempt_token_is_the_cluster_name_when_gbserver_sets_it(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "run_pretrain.sh").write_text(_STUB)
    proc = subprocess.run(
        ["bash", "-c", "set -eu\n" + _render(_CASES["single_node_train"])],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env={
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "GB_SKYPILOT_CLUSTER_NAME": "gb-mybuild-pretrain-0123456789ab-r1",
            "GB_SKYPILOT_LAUNCH_ID": "0123456789ab-cdef",
        },
    )
    assert proc.returncode == 0, proc.stderr
    env = json.loads(proc.stdout.strip().splitlines()[-1])
    assert env["GB_ATTEMPT_TOKEN"] == "gb-mybuild-pretrain-0123456789ab-r1"


# ── the monitor ────────────────────────────────────────────────────────────────


def test_the_launcher_wires_up_the_shared_skypilot_monitor():
    assert _SKY["launchers"]["lm-engine-pretrain"]["monitors"] == ["skypilot_monitor"]
    monitor = _SKY["monitors"]["skypilot_monitor"]
    assert monitor["ref"] == "space://monitors/skypilot"
    StepEnvironmentTypeConfig.model_validate(_SKY)


def test_the_monitor_scrapes_the_log_while_the_job_runs():
    """on_completion parses the log once, at terminal status: a rung announced
    mid-run would not dispatch its export target until training ended."""
    config = _SKY["monitors"]["skypilot_monitor"]["config"]
    retrieval = config["log_retrieval"]
    assert "periodic" in retrieval["mode"]
    assert fill_template(retrieval["mode"], {"config": {}}) == "periodic"
    assert fill_template(retrieval["interval_seconds"], {"config": {}}) == "300"
    assert fill_template(config["poll_interval_seconds"], {"config": {}}) == "300"


def test_the_monitor_reports_progress_from_the_job_log():
    events = _SKY["monitors"]["skypilot_monitor"]["config"]["extra_event_configs"]
    assert len(events) == 1
    assert events[0]["event_type"] == "WORKLOAD_STATUS_EVENT"
    regex = events[0]["line_regex"]
    assert "lm-engine: node_rank=" in regex
    assert "step = " in regex
    # The banner run_pretrain.sh prints really does match.
    assert re.search(regex, "lm-engine: node_rank=0 nnodes=2 gpus=8 master=10.0.0.1")
    assert re.search(regex, "step = 100, loss = 4.2")
