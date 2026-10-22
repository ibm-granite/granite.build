"""The shipped builds stay internally consistent with the step contract.

Not a GB render test — it asserts that every build.yaml this step ships (the
exemplar a user copies, plus each per-cluster e2e fixture) would actually submit
against THIS step: the step URI, required config keys present, `output` names a
declared `model` output, single-node, `base_model` bound, and (for the exemplar)
no undefined `$${PARAM}`. Running the contract checks over the cluster fixtures too
means a fixture cannot drift from the step without a fast unit test failing.
"""

import re
from pathlib import Path

import pytest
import yaml

_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parent / "test-data"
_PARAM_RE = re.compile(r"\$\$\{([A-Z0-9_]+)\}")

# The exemplar plus one fixture per cluster test (test/<cluster>/ <-> test-data/<cluster>/).
BUILDS = ["build.yaml", "aws-sft/build.yaml", "slurm-sft/build.yaml"]


def _load(name):
    with open(_DATA / name) as f:
        return yaml.safe_load(f)


def _target(build):
    targets = build["granite.build"]["targets"]
    assert len(targets) == 1, "exemplar is a single-target sample"
    return next(iter(targets.values()))


def _step(target):
    steps = target["steps"]
    assert len(steps) == 1
    return steps[0]


@pytest.mark.parametrize("build", BUILDS)
def test_step_uri_is_this_step(build):
    step = _step(_target(_load(build)))
    # common.mk derives the URI from the authoring nesting (steps/train/nemo-rl).
    assert step["step_uri"] == "space://steps/train/nemo-rl"


@pytest.mark.parametrize("build", BUILDS)
def test_required_nemo_rl_config_keys_present_and_nonempty(build):
    nrl = _step(_target(_load(build)))["config"]["nemo_rl_config"]
    for key in ("algorithm", "config_name", "output"):
        assert key in nrl, f"missing required key {key!r}"
        assert str(nrl[key]).strip() != "", f"{key!r} must be non-empty"


@pytest.mark.parametrize("build", BUILDS)
def test_output_names_a_declared_model_output(build):
    target = _target(_load(build))
    nrl = _step(target)["config"]["nemo_rl_config"]
    outputs = target["outputs"]
    assert nrl["output"] in outputs, "output must name a declared outputs: entry"
    assert outputs[nrl["output"]]["type"] == "model"


@pytest.mark.parametrize("build", BUILDS)
def test_single_node(build):
    compute = _step(_target(_load(build)))["config"]["compute_config"]
    assert compute["num_nodes"] == 1


@pytest.mark.parametrize("build", BUILDS)
def test_base_model_input_present(build):
    target = _target(_load(build))
    assert "base_model" in target["inputs"]


@pytest.mark.parametrize("build", ["aws-sft/build.yaml", "slurm-sft/build.yaml"])
def test_cluster_fixture_requests_a_typed_gpu(build):
    # The step renders accelerators as "<gpu_model>:<num_gpus_per_node>"; a fixture
    # without a GPU type would provision a node NeMo-RL cannot train on.
    compute = _step(_target(_load(build)))["config"]["compute_config"]
    assert str(compute.get("gpu_model", "")).strip(), "gpu_model must be set"
    assert compute["num_gpus_per_node"] >= 1


def test_slurm_fixture_matches_the_bluevela_environment_constraints():
    # ibm-bluevela pins every job to H200 (sbatch constraint), so the fixture must
    # ask for H200, and must target that environment.
    repo = _HERE.parents[4]  # steps/train/nemo-rl/skypilot/test -> repo root
    env = yaml.safe_load(
        (
            repo
            / "configurations/assets/environments/skypilot/slurm/ibm-bluevela"
            / "environment.yaml"
        ).read_text()
    )
    target = _target(_load("slurm-sft/build.yaml"))
    assert (
        target["environment_uri"] == "space://environments/skypilot/slurm/ibm-bluevela"
    )
    constraint = env["config"]["sbatch_options"]["constraint"]
    compute = _step(target)["config"]["compute_config"]
    assert compute["gpu_model"].lower() == constraint


def test_every_referenced_param_is_defined():
    raw = (_DATA / "build.yaml").read_text()
    referenced = set(_PARAM_RE.findall(raw))
    defined = set(_load("parameters.yaml").keys())
    missing = referenced - defined
    assert not missing, f"params referenced but not defined: {sorted(missing)}"
