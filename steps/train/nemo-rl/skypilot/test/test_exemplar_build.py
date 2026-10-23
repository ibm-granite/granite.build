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


def _accelerators(build):
    config = _step(_target(_load(build)))["config"]
    return ((config.get("launcher_config") or {}).get("resources") or {}).get(
        "accelerators"
    )


def _parse_accelerators(spec):
    """Split a SkyPilot accelerators string into (type, count).

    A bare type ("A10G") means 1; a count may be fractional ("A10G:0.5").
    Returns (type, None) when the count is not a number.
    """
    gpu_type, sep, count = str(spec or "").partition(":")
    if not sep:
        return gpu_type.strip(), 1.0
    try:
        return gpu_type.strip(), float(count)
    except ValueError:
        return gpu_type.strip(), None


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("A10G:1", ("A10G", 1.0)),
        ("H200:8", ("H200", 8.0)),
        ("A10G", ("A10G", 1.0)),
        ("A10G:0.5", ("A10G", 0.5)),
        ("A10G:x", ("A10G", None)),
        ("", ("", 1.0)),
    ],
)
def test_parse_accelerators(spec, expected):
    assert _parse_accelerators(spec) == expected


@pytest.mark.parametrize("build", ["aws-sft/build.yaml", "slurm-sft/build.yaml"])
def test_cluster_fixture_requests_a_typed_gpu(build):
    # GPUs come only from launcher_config.resources.accelerators: without it the
    # node has no GPU, and NeMo-RL cannot train.
    spec = _accelerators(build)
    gpu_type, count = _parse_accelerators(spec)
    assert gpu_type, f"accelerators must name a GPU type, e.g. A10G:1 (got {spec!r})"
    assert (
        count is not None and count >= 1
    ), f"accelerators must request at least one whole GPU (got {spec!r})"


@pytest.mark.parametrize("build", BUILDS)
def test_gpus_are_not_requested_through_compute_config(build):
    # SkyPilot ignores compute_config GPUs; a count there would be a second,
    # unused copy that the launcher warns about.
    compute = _step(_target(_load(build)))["config"]["compute_config"]
    assert "num_gpus_per_node" not in compute
    assert "gpu_model" not in compute


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
    gpu_type, _ = _parse_accelerators(_accelerators("slurm-sft/build.yaml"))
    assert gpu_type.lower() == constraint


def test_every_referenced_param_is_defined():
    raw = (_DATA / "build.yaml").read_text()
    referenced = set(_PARAM_RE.findall(raw))
    defined = set(_load("parameters.yaml").keys())
    missing = referenced - defined
    assert not missing, f"params referenced but not defined: {sorted(missing)}"
