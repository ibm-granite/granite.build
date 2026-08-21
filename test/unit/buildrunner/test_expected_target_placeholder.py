# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unreplaced FIXME placeholders must fail spec validation (issue #278)."""

import pytest
from libgbtest.buildrunner.buildtest import BuildTestSpecification


def _write(tmp_path, step_count):
    (tmp_path / "build.yaml").write_text(
        "granite.build:\n  name: d\n  targets: {t1: {}}\n"
    )
    p = tmp_path / "buildtest.yaml"
    p.write_text(
        "build_yaml: ./build.yaml\n"
        "tests: [runner]\n"
        "simulate_step_failure: false\n"
        "targets: [t1]\n"
        "target_expectations:\n"
        f"  - {{target_name: t1, input_artifact_count: 0, output_artifact_count: 1, "
        f"step_count: {step_count}, jobstats_count: 0}}\n"
    )
    return p


def test_unreplaced_placeholder_fails_validation(tmp_path):
    p = _write(tmp_path, "FIXME")
    with pytest.raises(Exception) as exc:
        BuildTestSpecification.from_yaml(p)
    msg = str(exc.value)
    # The field and offending token appear, plus our actionable wording so the
    # user knows a skeleton placeholder was left unreplaced (not a generic
    # "invalid integer" error).
    assert "step_count" in msg
    assert "FIXME" in msg
    assert "placeholder" in msg.lower()


def test_replaced_placeholder_loads(tmp_path):
    spec = BuildTestSpecification.from_yaml(_write(tmp_path, "-1"))
    assert spec.target_expectations[0].step_count == -1
