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
    TargetRun,
    effective_target_priority_class_name,
)
from gbserver.types.buildconfig import BuildTargetConfig, BuildTargetStepConfig


def _apply(targetstepconfig, target_config):
    """Call the injector without a TargetRun instance.

    ``_apply_implicit_step_priority_class_name`` reads only its arguments (never
    ``self``), so binding ``self=None`` exercises it directly, no fixture needed.
    """
    return TargetRun._apply_implicit_step_priority_class_name(
        None, targetstepconfig, target_config  # type: ignore[arg-type]
    )


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

    def test_empty_string_treated_as_floor(self):
        """An empty priority_class_name is the floor, not high-priority."""
        assert effective_target_priority_class_name(_target(_step(""))) is None
        cfg = _target(_step("high-priority"), _step(""))
        assert effective_target_priority_class_name(cfg) is None


class TestApplyImplicitStepPriorityClassName:
    """The injector holds the load-bearing guarantees: inject only when the target
    minimum is high-priority, deep-copy rather than mutate the queued config, don't
    clobber a value already set, and handle None target/config."""

    def test_injects_high_when_all_steps_high(self):
        step = _step(no_config=True)  # implicit step: no config of its own
        target = _target(_step("high-priority"), _step("high-priority"))
        result = _apply(step, target)
        assert result.config["k8s"]["priority_class_name"] == HIGH

    def test_leaves_config_none_when_floor(self):
        """Floor target minimum -> return the config untouched (no key added)."""
        step = _step(no_config=True)
        target = _target(_step("high-priority"), _step("default-priority"))
        result = _apply(step, target)
        assert result is step  # unchanged, same object
        assert result.config is None

    def test_does_not_mutate_original_when_injecting(self):
        """The injected key lands on a copy; the queued config is never mutated."""
        step = _step(no_config=True)
        target = _target(_step("high-priority"))
        result = _apply(step, target)
        assert result is not step
        assert step.config is None  # original untouched
        assert result.config["k8s"]["priority_class_name"] == HIGH

    def test_initializes_none_config_on_copy(self):
        step = _step(no_config=True)
        result = _apply(step, _target(_step("high-priority")))
        assert result.config == {"k8s": {"priority_class_name": HIGH}}

    def test_does_not_clobber_existing_priority(self):
        """A value already set on the implicit step wins over the target minimum."""
        step = BuildTargetStepConfig(
            step_uri="space://steps/hfpull",
            config={"k8s": {"priority_class_name": "preset"}},
        )
        target = _target(_step("high-priority"))
        result = _apply(step, target)
        assert result is step
        assert result.config["k8s"]["priority_class_name"] == "preset"

    def test_preserves_other_config_keys_when_injecting(self):
        """Injecting k8s must not drop the step's own store config."""
        step = BuildTargetStepConfig(
            step_uri="space://steps/hfpull", config={"hfpull_config": {"uri": "x"}}
        )
        result = _apply(step, _target(_step("high-priority")))
        assert result.config["hfpull_config"] == {"uri": "x"}
        assert result.config["k8s"]["priority_class_name"] == HIGH

    def test_none_target_config_returns_unchanged(self):
        step = _step(no_config=True)
        result = _apply(step, None)
        assert result is step
