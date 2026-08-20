# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the gbtest render skeleton generator (issue #278)."""

import yaml
from libgbtest.buildrunner.buildtest_gen import derive_expectations, generate_skeleton

TWO_TARGET = """\
granite.build:
  name: demo
  targets:
    tokenize:
      environment_uri: space://environments/skypilot/aws
      outputs:
        tokens: {uri: "s3://b/t", type: dataset}
      steps:
        - step_uri: space://steps/command
    validate:
      environment_uri: space://environments/skypilot/aws
      inputs:
        tokens: {binding: tokenize.tokens}
      outputs:
        report: {uri: "env:///tmp/r", type: fileset}
      steps:
        - step_uri: space://steps/command
"""


def test_derive_expectations_counts(tmp_path):
    p = tmp_path / "build.yaml"
    p.write_text(TWO_TARGET)
    by = {e["target_name"]: e for e in derive_expectations(p)}
    assert by["tokenize"]["input_artifact_count"] == 0
    assert by["tokenize"]["output_artifact_count"] == 1
    assert by["validate"]["input_artifact_count"] == 1
    assert by["validate"]["output_artifact_count"] == 1


def test_generate_skeleton_emits_fixme_placeholders(tmp_path):
    p = tmp_path / "build.yaml"
    p.write_text(TWO_TARGET)
    text = generate_skeleton(p)
    doc = yaml.safe_load(text)  # loadable YAML (FIXME is a bare string)
    assert doc["simulate_step_failure"] is False  # step-retry off
    assert doc["tests"] == ["runner"]  # cancellation disabled
    assert sorted(doc["targets"]) == ["tokenize", "validate"]
    tok = next(e for e in doc["target_expectations"] if e["target_name"] == "tokenize")
    assert tok["output_artifact_count"] == 1  # derivable -> filled
    assert tok["step_count"] == "FIXME"  # non-derivable -> must replace
    assert tok["jobstats_count"] == -1  # not asserted at run time yet -> -1 (skip)
