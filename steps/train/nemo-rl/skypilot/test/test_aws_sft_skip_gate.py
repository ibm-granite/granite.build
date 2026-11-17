# SPDX-License-Identifier: Apache-2.0

"""The aws-sft e2e's HF-push skip gate.

aws-sft pushes the trained model to the Hub, so it needs a write-capable
HF_TOKEN that survives the test setup. Without one it would provision EC2, train
and convert, and only then 403 on `hf upload`. The gate must decide that up
front, from the environment alone (no network at collection time).
"""

import importlib.util
import pathlib

import pytest

_MOD = (
    pathlib.Path(__file__).resolve().parent
    / "aws-sft"
    / ("test_skypilot_aws_nemo_rl_sft.py")
)


@pytest.fixture(scope="module")
def gate():
    spec = importlib.util.spec_from_file_location("aws_sft_e2e", _MOD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.hf_push_skip_reason


@pytest.mark.parametrize(
    "env,skips",
    [
        ({}, True),
        ({"HF_TOKEN": "hf_x"}, True),
        ({"GBTEST_SPS_ENABLE_ENV_VAR_OVERRIDE": "true"}, True),
        ({"HF_TOKEN": "", "GBTEST_SPS_ENABLE_ENV_VAR_OVERRIDE": "true"}, True),
        ({"HF_TOKEN": "hf_x", "GBTEST_SPS_ENABLE_ENV_VAR_OVERRIDE": "false"}, True),
        ({"HF_TOKEN": "hf_x", "GBTEST_SPS_ENABLE_ENV_VAR_OVERRIDE": "true"}, False),
        ({"HF_TOKEN": "hf_x", "GBTEST_SPS_ENABLE_ENV_VAR_OVERRIDE": "True"}, False),
    ],
)
def test_skips_unless_a_token_that_survives_setup(gate, env, skips):
    reason = gate(env)
    assert bool(reason) is skips
    if skips:
        assert "HF_TOKEN" in reason
