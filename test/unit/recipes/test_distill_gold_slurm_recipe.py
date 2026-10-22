#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the SLURM distill-gold recipe (BlueVela SLURM).

The recipe is a build.yaml + parameters.yaml pair, so the thing that breaks it is
a parameter in one file and not the other — which on a live run surfaces only
after SLURM has queued an H100 allocation. Rendering catches one direction
(StrictUndefined raises on an undeclared marker); the opposite — a parameter
declared and never used — is asserted explicitly.

These also pin what makes this the SLURM recipe (vs the lsf/distill-gold sibling):
the three *-slurm step URIs, the SLURM environment URI, and the topology/scale
values carried over from gbansible/skypilot/kd_distill_bluevela_slurm_2node.yaml.
"""

import pathlib
import re

import pytest
import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_RECIPE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "recipes"
    / "granite4-gold-distillation"
    / "slurm"
    / "distill-gold"
)

# apply_parameters' delimiters: variable_start_string="$${", variable_end_string="}".
_MARKER = re.compile(r"\$\$\{([A-Za-z0-9_]+)\}")


@pytest.fixture
def rendered(tmp_path):
    """The recipe rendered against its own parameters.yaml (off-policy default)."""
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    params = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    return yaml.safe_load(apply_parameters(contents, [], params, str(tmp_path)))


def _targets(rendered):
    return rendered["granite.build"]["targets"]


def test_parameter_sets_match_exactly():
    """Every marker is declared, and every declaration is used.

    ON_POLICY and SKIP_ALIGN only appear inside the <% %> target conditionals
    (not a $${} marker), so they are added to the used set by hand.
    """
    text = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    used = set(_MARKER.findall(text))
    used.update({"ON_POLICY", "SKIP_ALIGN"})
    declared = set(get_params_from_file(str(_RECIPE / "parameters.yaml")))

    assert used - declared == set(), "markers with no declared parameter"
    assert declared - used == set(), "parameters declared but never used"


def test_no_single_dollar_substitution_markers():
    """A `${VAR}` typo is passed through untouched, not substituted.

    Only `$${` opens a marker; `$$` and `{{ }}` (gbserver's own Jinja, e.g.
    `env://{{ binding.path }}`) are both legitimate, so this looks only for the
    exact single-dollar-brace form.
    """
    text = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    assert "${" not in text.replace("$${", "")


def test_offpolicy_is_three_targets(rendered):
    """The default (ON_POLICY=false) run is align -> corpus -> train-gold."""
    assert list(_targets(rendered)) == ["align", "corpus", "train-gold"]


def test_targets_use_the_slurm_step_copies(rendered):
    """Each target routes to the SLURM step variant, not the LSF step.

    The LSF steps declare subtypes:[lsf] and gold reads the LSF topology contract,
    so a SLURM build must use the -slurm copies or it fails subtype validation /
    aborts on an unset MASTER_ADDR.
    """
    targets = _targets(rendered)
    assert targets["align"]["steps"][0]["step_uri"] == (
        "space://steps/distill/tokenizer-align-slurm"
    )
    assert targets["corpus"]["steps"][0]["step_uri"] == (
        "space://steps/distill/corpus-prep-slurm"
    )
    assert targets["train-gold"]["steps"][0]["step_uri"] == (
        "space://steps/distill/gold-slurm"
    )


def test_environment_is_the_slurm_bluevela_env(rendered):
    """Every target runs on the in-repo SLURM BlueVela environment."""
    uris = {t["environment_uri"] for t in _targets(rendered).values()}
    assert uris == {"space://environments/skypilot/slurm/ibm-bluevela"}


def test_three_lineage_inputs_on_train_gold(rendered):
    """The teacher, student and corpus are the trainer's inputs (lineage)."""
    inputs = _targets(rendered)["train-gold"]["inputs"]
    assert set(inputs) == {"teacher", "student", "corpus"}
    assert inputs["teacher"]["uri"].startswith("env:///")
    assert inputs["student"]["binding"] == "align.retagged_student"
    assert inputs["corpus"]["binding"] == "corpus.corpus"


def test_offpolicy_sets_no_vllm_servers(rendered):
    """ON_POLICY=false renders vllm_num_servers: 0 (every rank trains the corpus)."""
    gold = _targets(rendered)["train-gold"]["steps"][0]["config"]["gold_config"]
    assert gold["vllm_num_servers"] == 0
    assert gold["lmbda"] == 0.0


def test_response_template_is_the_granite_role_boundary(rendered):
    """The SLURM run uses the granite-native role boundary, not the LSF im_start.

    Mirrors kd_distill_bluevela_slurm_2node.yaml; no trailing newline, so none of
    the lsf recipe's doubled-backslash Jinja gotcha applies.
    """
    gold = _targets(rendered)["train-gold"]["steps"][0]["config"]["gold_config"]
    assert gold["response_template"] == "<|start_of_role|>assistant<|end_of_role|>"


def _onpolicy_rendered(tmp_path):
    """The recipe rendered as a ramped on-policy run (the --param overrides)."""
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    params = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    params["ON_POLICY"] = "true"
    params["LMBDA"] = "1.0"
    params["LMBDA_SCHEDULE"] = "linear"
    params["LMBDA_INIT"] = "0.0"
    params["VLLM_QUEUE"] = "gpu-high"
    return yaml.safe_load(apply_parameters(contents, [], params, str(tmp_path)))


def test_onpolicy_adds_server_and_teardown(tmp_path):
    """ON_POLICY=true adds the vllm-server and teardown targets."""
    rendered = _onpolicy_rendered(tmp_path)
    assert list(_targets(rendered)) == [
        "align",
        "corpus",
        "vllm-server",
        "train-gold",
        "teardown",
    ]


def test_onpolicy_uses_the_slurm_vllm_server_on_its_own_partition(tmp_path):
    """The server uses the SLURM step copy and a partition distinct from the trainer.

    gold-slurm reads the LSF topology-free vllm_url binding, but the server step must
    be the slurm subtype or the build fails validation; and the ramped run puts the
    server on gpu-high while the trainer stays on gpu-mid.
    """
    targets = _targets(_onpolicy_rendered(tmp_path))
    server = targets["vllm-server"]["steps"][0]
    assert server["step_uri"] == "space://steps/distill/vllm-server-slurm"
    server_zone = server["config"]["launcher_config"]["resources"]["zone"]
    trainer_zone = targets["train-gold"]["steps"][0]["config"]["launcher_config"][
        "resources"
    ]["zone"]
    assert server_zone == "gpu-high"
    assert trainer_zone == "gpu-mid"
    assert server_zone != trainer_zone


def test_onpolicy_ramps_lmbda_with_a_server(tmp_path):
    """Ramped on-policy: lmbda_schedule=linear, a non-zero target, and >=1 server.

    render_gold_config refuses linear at vllm_num_servers 0, so these must agree.
    """
    gold = _targets(_onpolicy_rendered(tmp_path))["train-gold"]["steps"][0]["config"][
        "gold_config"
    ]
    assert gold["lmbda_schedule"] == "linear"
    assert gold["lmbda_init"] == 0.0
    assert gold["lmbda"] == 1.0
    assert gold["vllm_num_servers"] >= 1


def test_scientific_notation_params_are_strings_in_parameters_yaml():
    """1.0e-05 must be quoted in parameters.yaml or PyYAML loads it as a float
    whose repr '1e-05' round-trips back through build.yaml as a string.

    Asserting on the raw file (not the rendered build) is the only place the quote
    is visible — once loaded, both forms are indistinguishable.
    """
    raw = (_RECIPE / "parameters.yaml").read_text(encoding="utf-8")
    for key in ("LEARNING_RATE", "MIN_LR"):
        assert re.search(rf'^{key}:\s*"[0-9.]+e-?[0-9]+"', raw, re.M), key
