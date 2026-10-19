"""The shipped exemplar stays internally consistent with the step contract.

Not a GB render test — it asserts the sample build.yaml/parameters.yaml a user
copies would actually submit against THIS step: required config keys present,
`output` names a declared `model` output, single-node, `base_model` bound, and
no undefined `$${PARAM}`.
"""

import re
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent
_DATA = _HERE.parent / "test-data"
_PARAM_RE = re.compile(r"\$\$\{([A-Z0-9_]+)\}")


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


def test_step_uri_is_this_step():
    step = _step(_target(_load("build.yaml")))
    # common.mk derives the URI from the authoring nesting (steps/train/nemo-rl).
    assert step["step_uri"] == "space://steps/train/nemo-rl"


def test_required_nemo_rl_config_keys_present_and_nonempty():
    nrl = _step(_target(_load("build.yaml")))["config"]["nemo_rl_config"]
    for key in ("algorithm", "config_name", "output"):
        assert key in nrl, f"missing required key {key!r}"
        assert str(nrl[key]).strip() != "", f"{key!r} must be non-empty"


def test_output_names_a_declared_model_output():
    target = _target(_load("build.yaml"))
    nrl = _step(target)["config"]["nemo_rl_config"]
    outputs = target["outputs"]
    assert nrl["output"] in outputs, "output must name a declared outputs: entry"
    assert outputs[nrl["output"]]["type"] == "model"


def test_single_node():
    compute = _step(_target(_load("build.yaml")))["config"]["compute_config"]
    assert compute["num_nodes"] == 1


def test_base_model_input_present():
    target = _target(_load("build.yaml"))
    assert "base_model" in target["inputs"]


def test_every_referenced_param_is_defined():
    raw = (_DATA / "build.yaml").read_text()
    referenced = set(_PARAM_RE.findall(raw))
    defined = set(_load("parameters.yaml").keys())
    missing = referenced - defined
    assert not missing, f"params referenced but not defined: {sorted(missing)}"
