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

"""Every published step declares its inputs, and the declarations are pinned (#453).

A step that declares no `inputs:` cannot have a missing or misnamed input caught at
build validation: the non-strict config renderer passes an unresolved
`{{ bindings.<name>... }}` through as literal text, and the step fails on the cluster,
after queue time. So every step declares an `inputs:` block, even when it is only
`allow_unknown: true` with a comment saying why the step has no named inputs.

EXPECTED is the inventory, asserted against disk. Adding a step, or changing what one
requires, means changing this table — a decision, not a silent drift.
"""

import pathlib
import re

import pytest
import yaml

from gbserver.types.stepconfig import StepConfig

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
ASSETS_DIR = REPO_ROOT / "configurations" / "assets"

_SKY = "environments/skypilot/steps"
_LSF = "environments/skypilot/lsf/ibm-bluevela/steps"
_AWS = "environments/skypilot/aws/steps"

# step dir (relative to configurations/assets) -> (required, optional). Empty sets are
# the listed exceptions: each such step.yaml says why in a comment above `inputs:`.
EXPECTED = {
    # Distillation: not in any tag yet, so the inputs are required outright.
    f"{_SKY}/distill/tokenizer-align": (
        {"teacher", "student"},
        {"chat_template"},
    ),
    f"{_SKY}/distill/corpus-prep": ({"source_dataset", "tokenizer"}, set()),
    f"{_SKY}/distill/corpus-pin-check": ({"tokenizer"}, set()),
    f"{_SKY}/distill/sft": ({"student", "corpus"}, set()),
    f"{_SKY}/distill/gold": ({"student", "teacher", "corpus"}, {"vllm"}),
    f"{_SKY}/distill/eval": ({"student"}, {"teacher"}),
    f"{_SKY}/distill/hf-export": ({"train_output"}, {"expected_tokenizer"}),
    f"{_SKY}/distill/logit-precompute": (
        {"corpus", "teacher", "teacher_tokenizer"},
        set(),
    ),
    f"{_SKY}/distill/vllm-server": ({"model"}, set()),
    f"{_SKY}/distill/corpus-sources": (set(), set()),  # variable-length sources list
    f"{_SKY}/distill/gen-smoke": (set(), set()),  # variable-length rungs list
    f"{_SKY}/distill/probe": (set(), set()),  # diagnostic, literal paths
    # Shipped in v0.3.x: optional during the deprecation window, with the old config
    # key as a fallback.
    f"{_LSF}/sage-eval": (set(), {"model"}),
    f"{_LSF}/bfcl-eval": (set(), {"model"}),
    f"{_LSF}/openinstruct-sft": (set(), {"model"}),
    f"{_LSF}/openinstruct-rl": (set(), {"rm_url", "code_url"}),
    f"{_SKY}/dpk": (set(), {"docs"}),
    f"{_AWS}/openinstruct-sft": (set(), {"model"}),
    f"{_SKY}/digit": (set(), {"digit_input"}),
    "environments/skypilot-managed/kubernetes/steps/digit": (set(), {"digit_input"}),
    "steps/k8s/digit": (set(), {"digit_input"}),
    "environments/bash/steps/inference": ({"model"}, set()),
    "environments/bash/steps/inference-lora": ({"model"}, {"adapter"}),
    "environments/bash/steps/lora-finetune": ({"model"}, {"dataset"}),
    # Exceptions: no named inputs, by design.
    f"{_SKY}/byoc": (set(), set()),  # free-form command, any number of inputs
    f"{_LSF}/skypilot-teardown": (set(), set()),  # variable-length cluster list
    f"{_LSF}/rm-server": (set(), set()),
    f"{_LSF}/code-server": (set(), set()),
    f"{_LSF}/bcb-server": (set(), set()),
    f"{_LSF}/bfcl-export": (set(), set()),  # ordering-only gate inputs
    f"{_LSF}/sage-export": (set(), set()),
    f"{_LSF}/combined-export": (set(), set()),
    f"{_SKY}/sage": (set(), set()),
    "environments/skypilot-managed/kubernetes/steps/sage": (set(), set()),
    # AWS evals take the model as an S3 URI mounted by file_mounts.
    f"{_AWS}/sage-eval": (set(), set()),
    f"{_AWS}/sage-eval-bcb": (set(), set()),
    f"{_AWS}/sage-eval-bcb-generate": (set(), set()),
    f"{_AWS}/sage-eval-bcb-execute": (set(), set()),
    f"{_AWS}/sage-eval-multilingual-grouped": (set(), set()),
    f"{_AWS}/sage-eval-code-grouped": (set(), set()),
    f"{_AWS}/sage-eval-olmes-grouped": (set(), set()),
    f"{_AWS}/sage-eval-safety-grouped": (set(), set()),
    f"{_AWS}/bfcl-eval": (set(), set()),
    f"{_AWS}/openinstruct-rl": (set(), set()),
    "environments/bash/steps/hello": (set(), set()),
    "environments/docker/steps/hello": (set(), set()),
    "environments/runpod/steps/hello": (set(), set()),
    f"{_AWS}/hello": (set(), set()),
}

STEP_YAMLS = sorted(ASSETS_DIR.glob("**/step.yaml"))


def _rel(step_yaml: pathlib.Path) -> str:
    return step_yaml.parent.relative_to(ASSETS_DIR).as_posix()


def test_the_inventory_matches_disk():
    """A new or removed step must be added to (or dropped from) EXPECTED."""
    assert {_rel(p) for p in STEP_YAMLS} == set(EXPECTED)


@pytest.mark.parametrize("step_yaml", STEP_YAMLS, ids=_rel)
def test_every_step_declares_inputs(step_yaml):
    raw = yaml.safe_load(step_yaml.read_text(encoding="utf-8"))
    assert "inputs" in raw, (
        f"{_rel(step_yaml)} declares no `inputs:`. Declare what it reads, or "
        "`inputs: {allow_unknown: true}` with a comment saying why it has none."
    )


@pytest.mark.parametrize("step_yaml", STEP_YAMLS, ids=_rel)
def test_the_declaration_parses_and_matches(step_yaml):
    config = StepConfig.from_yaml(step_yaml)
    required, optional = EXPECTED[_rel(step_yaml)]
    assert set(config.inputs.required) == required
    assert set(config.inputs.optional) == optional


@pytest.mark.parametrize("step_yaml", STEP_YAMLS, ids=_rel)
def test_an_exception_says_why(step_yaml):
    """A step with no named inputs must explain itself above its `inputs:` block."""
    required, optional = EXPECTED[_rel(step_yaml)]
    if required or optional:
        return
    lines = step_yaml.read_text(encoding="utf-8").splitlines()
    # Block (`inputs:`) or flow (`inputs: {allow_unknown: true}`) style.
    at = next(i for i, line in enumerate(lines) if re.match(r"inputs:(\s|$)", line))
    comment = []
    for line in reversed(lines[:at]):
        if not line.startswith("#"):
            break
        comment.insert(0, line.lstrip("#").strip())
    why = " ".join(comment)
    assert len(why.split()) >= 4 and not re.match(
        r"(TODO|FIXME|XXX)\b", why, re.IGNORECASE
    ), f"{_rel(step_yaml)} declares no named inputs without a comment saying why"
