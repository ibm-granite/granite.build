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
    """Return the GPU request a shipped build makes.

    :param build: Path of the build under test-data/, e.g. ``"aws-sft/build.yaml"``.
    :returns: Its step's ``launcher_config.resources.accelerators`` value, or
        None when the build sets none (a missing or null layer counts as unset).
    """
    config = _step(_target(_load(build)))["config"]
    return ((config.get("launcher_config") or {}).get("resources") or {}).get(
        "accelerators"
    )


def _sky_accelerators(spec):
    """Parse an accelerators spec the way the launcher does: with SkyPilot.

    SkyPilot is imported, not skipped on: without it this check would report
    SKIPPED and a fixture with no GPU request would pass unseen (#542 review).
    The fixtures name GPU types in SkyPilot's bundled catalog (A10G, H200), so
    parsing them needs no network or kube context.

    :param spec: An accelerators value (e.g. ``"A10G:1"``, ``{"A100": 8}``).
    :returns: SkyPilot's ``{type: count}`` dict, or None when SkyPilot rejects
        the spec for any reason, so a bad spec fails the contract assertion
        instead of crashing the test.
    """
    import sky  # pylint: disable=import-outside-toplevel

    try:
        return sky.Resources(accelerators=spec).accelerators
    except Exception:  # pylint: disable=broad-except
        return None


@pytest.mark.parametrize("build", ["aws-sft/build.yaml", "slurm-sft/build.yaml"])
def test_cluster_fixture_requests_a_typed_gpu(build):
    # GPUs come only from launcher_config.resources.accelerators: without it the
    # node has no GPU, and NeMo-RL cannot train.
    spec = _accelerators(build)
    parsed = _sky_accelerators(spec)
    assert (
        parsed
    ), f"accelerators must be a SkyPilot GPU request, e.g. A10G:1 (got {spec!r})"
    ((gpu_type, count),) = parsed.items()
    assert gpu_type.strip(), f"accelerators must name a GPU type (got {spec!r})"
    assert count >= 1, f"accelerators must request at least one GPU (got {spec!r})"


@pytest.mark.parametrize("build", BUILDS)
def test_gpus_are_not_requested_through_compute_config(build):
    # SkyPilot ignores compute_config GPUs; a count there would be a second,
    # unused copy that the launcher warns about.
    compute = _step(_target(_load(build)))["config"]["compute_config"]
    assert "num_gpus_per_node" not in compute
    assert "gpu_model" not in compute


def test_slurm_fixture_uses_the_environment_image_cache():
    # ibm-bluevela's slurm_image_cache imports a registry image once into a
    # shared .sqsh and launches from it (#560). That only applies to a registry
    # ref: a hardcoded .sqsh path in the fixture is used as is, bypassing the
    # cache. Leave image_id unset so the step's pinned default is cached.
    repo = _HERE.parents[4]  # steps/train/nemo-rl/skypilot/test -> repo root
    env = yaml.safe_load(
        (
            repo
            / "configurations/assets/environments/skypilot/slurm/ibm-bluevela"
            / "environment.yaml"
        ).read_text()
    )
    assert env["config"].get("slurm_image_cache"), "ibm-bluevela lost its image cache"
    nrl = _step(_target(_load("slurm-sft/build.yaml")))["config"]["nemo_rl_config"]
    assert "image_id" not in nrl, "slurm-sft must use the step's registry image"


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
    parsed = _sky_accelerators(_accelerators("slurm-sft/build.yaml"))
    assert parsed, "slurm-sft must request GPUs with accelerators"
    ((gpu_type, _),) = parsed.items()
    assert gpu_type.lower() == constraint


def test_every_referenced_param_is_defined():
    raw = (_DATA / "build.yaml").read_text()
    referenced = set(_PARAM_RE.findall(raw))
    defined = set(_load("parameters.yaml").keys())
    missing = referenced - defined
    assert not missing, f"params referenced but not defined: {sorted(missing)}"
