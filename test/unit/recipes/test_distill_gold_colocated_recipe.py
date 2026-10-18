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

"""Unit tests for the SLURM colocated on-policy distill-gold recipe.

This recipe runs on-policy GOLD with an IN-ALLOCATION vLLM serve node: a single
train-gold target, no separate vllm-server/teardown, and the `vllm` input left
UNBOUND so gold-slurm takes its in-allocation split (serve on the last node of the
one allocation). That keeps all NCCL within a single SLURM allocation — unlike the
external-server path, which deadlocks on BlueVela. These tests pin that shape so a
later edit can't silently reintroduce the cross-allocation design.
"""

import pathlib
import re

import yaml

from gbcli.services.service_build import get_params_from_file
from gbcli.utils.buildutil import apply_parameters

_RECIPE = (
    pathlib.Path(__file__).resolve().parents[3]
    / "recipes"
    / "granite4-gold-distillation"
    / "slurm"
    / "distill-gold-colocated"
)

_MARKER = re.compile(r"\$\$\{([A-Za-z0-9_]+)\}")


def _rendered(tmp_path):
    contents = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    params = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    return yaml.safe_load(apply_parameters(contents, [], params, str(tmp_path)))


def _targets(rendered):
    return rendered["granite.build"]["targets"]


def test_parameter_sets_match_exactly():
    """Every $${marker} is declared and every declared param is used.

    This recipe has no <% %> conditionals, so (unlike the sibling recipes) there is
    no ON_POLICY to add by hand — the marker set and the declared set must match 1:1.
    """
    text = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    used = set(_MARKER.findall(text))
    declared = set(get_params_from_file(str(_RECIPE / "parameters.yaml")))
    assert used - declared == set(), "markers with no declared parameter"
    assert declared - used == set(), "parameters declared but never used"


def test_single_train_gold_target_no_server_or_teardown(tmp_path):
    """In-allocation design: ONE target, no external vllm-server / teardown."""
    targets = _targets(_rendered(tmp_path))
    assert list(targets) == ["train-gold"]
    assert targets["train-gold"]["steps"][0]["step_uri"] == (
        "space://steps/distill/gold-slurm"
    )


def test_vllm_input_is_unbound(tmp_path):
    """The `vllm` input MUST stay unbound — binding it selects the external-server
    path (cross-allocation NCCL, which deadlocks). Its absence selects gold-slurm's
    in-allocation split."""
    inputs = _targets(_rendered(tmp_path))["train-gold"]["inputs"]
    assert set(inputs) == {"student", "corpus", "teacher"}
    assert "vllm" not in inputs


def test_onpolicy_ramp_with_in_allocation_server(tmp_path):
    """vllm_num_servers>0 (carves the serve node) + linear lmbda ramp, and the
    key-verify is disabled for this image."""
    gold = _targets(_rendered(tmp_path))["train-gold"]["steps"][0]["config"][
        "gold_config"
    ]
    assert gold["vllm_num_servers"] >= 1
    assert gold["lmbda_schedule"] == "linear"
    assert gold["lmbda_init"] == 0.0
    assert gold["lmbda"] == 1.0
    assert gold["verify_trainer_accepts_keys"] is False


def test_num_nodes_exceeds_servers(tmp_path):
    """Need at least one trainer node: NUM_NODES must be > vllm_num_servers."""
    params = get_params_from_file(str(_RECIPE / "parameters.yaml"))
    assert int(params["NUM_NODES"]) > int(params["VLLM_NUM_SERVERS"])


def test_no_single_dollar_substitution_markers():
    """A `${VAR}` typo would reach the cluster literally; only `$${` is a marker."""
    text = (_RECIPE / "build.yaml").read_text(encoding="utf-8")
    assert "${" not in text.replace("$${", "")
