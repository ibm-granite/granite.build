# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""`gbtest ... -f <build.yaml>` overrides the test's build_yaml (issue #278)."""

from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    BuildTestSpecification,
)
from libgbtest.buildrunner.gbtest import _split_build_yaml_flag


def _write_fixture(tmp_path):
    (tmp_path / "build.yaml").write_text(
        "granite.build:\n  name: d\n  targets: {t1: {}}\n"
    )
    (tmp_path / "exec.yaml").write_text(
        "granite.build:\n  name: d\n  targets: {t1: {}}\n"
    )
    bt = tmp_path / "buildtest.yaml"
    bt.write_text(
        "build_yaml: ./build.yaml\n"
        "tests: [runner]\n"
        "simulate_step_failure: false\n"
        "targets: [t1]\n"
        "target_expectations:\n"
        "  - {target_name: t1, input_artifact_count: 0, output_artifact_count: 1, "
        "step_count: -1, jobstats_count: 0}\n"
    )
    return bt


def test_from_yaml_uses_sibling_without_override(tmp_path):
    bt = _write_fixture(tmp_path)
    spec = BuildTestSpecification.from_yaml(bt)
    assert spec.build_yaml == str((tmp_path / "build.yaml").resolve())


def test_from_yaml_build_yaml_override(tmp_path):
    bt = _write_fixture(tmp_path)
    exec_build = tmp_path / "exec.yaml"
    spec = BuildTestSpecification.from_yaml(bt, build_yaml_override=str(exec_build))
    assert spec.build_yaml == str(exec_build.resolve())


def test_get_test_specification_applies_override(tmp_path):
    bt_dir = tmp_path
    _write_fixture(tmp_path)
    exec_build = tmp_path / "exec.yaml"

    class _Stub(AbstractYamlBuildRunnerTest):
        def _get_yaml_spec_dir(self):
            return bt_dir

    stub = _Stub()
    stub._build_yaml_override = str(exec_build)
    spec = stub._get_test_specification()
    assert spec.build_yaml == str(exec_build.resolve())


def test_split_build_yaml_flag_extracts_dash_f():
    override, rest = _split_build_yaml_flag(["-f", "exec.yaml", "-vv"])
    assert override == "exec.yaml"
    assert rest == ["-vv"]


def test_split_build_yaml_flag_none_when_absent():
    override, rest = _split_build_yaml_flag(["-vv", "-k", "runner"])
    assert override is None
    assert rest == ["-vv", "-k", "runner"]


def test_split_build_yaml_flag_trailing_bare_f_errors():
    # A trailing `-f` with no path is a usage error, not a silent no-op.
    with pytest.raises(ValueError, match="requires a build.yaml path"):
        _split_build_yaml_flag(["-vv", "-f"])


def test_split_build_yaml_flag_trailing_bare_long_flag_errors():
    with pytest.raises(ValueError, match="requires a build.yaml path"):
        _split_build_yaml_flag(["--build-yaml"])
