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

"""Tests for effective_target_priority_class_name: implicit input-pull /
output-push steps (hfpull, hfpush, lhpull, s3push, ...) are synthesized
server-side and never carry the user's per-step config.k8s, so they would run at
the cluster default priority. A transfer must not outrank the workload it serves,
so its priority is the *minimum* across the target's explicit steps.

Only two names are ordered (default-priority < high-priority); unset/empty or any
other name is the floor. high-priority is returned only when EVERY step is
high-priority; otherwise None (leave the implicit step unset -> cluster default)."""

import pytest

from gbserver.build.targetrun import (
    HIGH_PRIORITY_CLASS_NAME,
    effective_target_priority_class_name,
)
from gbserver.types.buildconfig import BuildTargetConfig, BuildTargetStepConfig

HIGH = HIGH_PRIORITY_CLASS_NAME  # "high-priority"


def _step(priority_class_name=..., *, no_config=False, no_k8s=False):
    """Build a step whose config.k8s.priority_class_name is set (or absent).

    - ``no_config=True``   -> step.config is None
    - ``no_k8s=True``      -> config has no ``k8s`` block
    - priority_class_name unset (the ``...`` sentinel) -> ``k8s`` present but the
      key absent
    - otherwise            -> ``k8s.priority_class_name`` = the given value
    """
    if no_config:
        return BuildTargetStepConfig(step_uri="space://steps/x", config=None)
    if no_k8s:
        return BuildTargetStepConfig(
            step_uri="space://steps/x", config={"download_config": {}}
        )
    if priority_class_name is ...:
        return BuildTargetStepConfig(step_uri="space://steps/x", config={"k8s": {}})
    return BuildTargetStepConfig(
        step_uri="space://steps/x",
        config={"k8s": {"priority_class_name": priority_class_name}},
    )


def _target(*steps):
    return BuildTargetConfig(
        environment_uri="space://environments/k8s", steps=list(steps)
    )


class TestEffectiveTargetPriorityClassName:
    def test_all_high_returns_high(self):
        cfg = _target(_step("high-priority"), _step("high-priority"))
        assert effective_target_priority_class_name(cfg) == HIGH

    def test_single_high_returns_high(self):
        assert (
            effective_target_priority_class_name(_target(_step("high-priority")))
            == HIGH
        )

    def test_high_and_default_returns_none(self):
        cfg = _target(_step("high-priority"), _step("default-priority"))
        assert effective_target_priority_class_name(cfg) is None

    def test_high_and_unset_key_returns_none(self):
        """A step with a k8s block but no priority_class_name is the floor."""
        cfg = _target(
            _step("high-priority"), _step()
        )  # second: k8s present, key absent
        assert effective_target_priority_class_name(cfg) is None

    def test_high_and_no_k8s_returns_none(self):
        cfg = _target(_step("high-priority"), _step(no_k8s=True))
        assert effective_target_priority_class_name(cfg) is None

    def test_high_and_no_config_returns_none(self):
        """step.config is None must not raise and counts as the floor."""
        cfg = _target(_step("high-priority"), _step(no_config=True))
        assert effective_target_priority_class_name(cfg) is None

    def test_single_default_returns_none(self):
        assert (
            effective_target_priority_class_name(_target(_step("default-priority")))
            is None
        )

    def test_all_unset_returns_none(self):
        cfg = _target(_step(), _step(no_k8s=True), _step(no_config=True))
        assert effective_target_priority_class_name(cfg) is None

    def test_no_steps_returns_none(self):
        assert effective_target_priority_class_name(_target()) is None

    @pytest.mark.parametrize(
        "steps",
        [
            [_step("high-priority"), _step("low")],  # unknown name is the floor
            [_step("low")],
            [_step("medium"), _step("high-priority")],
        ],
    )
    def test_unranked_name_treated_as_floor(self, steps):
        assert effective_target_priority_class_name(_target(*steps)) is None
