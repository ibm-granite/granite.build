# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the `gb build render` command (issue #278)."""

from pathlib import Path

import yaml
from click.testing import CliRunner

from gbcli.commands.command_build import cli as build_cli


def _template(tmp_path: Path) -> Path:
    tdir = tmp_path / "t"
    tdir.mkdir()
    bp = tdir / "build.yaml"
    bp.write_text(
        "granite.build:\n"
        "  name: d\n"
        "  targets:\n"
        "    t1:\n"
        "      environment_uri: space://environments/$${ENVIRONMENT}\n"
    )
    return bp


def test_render_prints_to_stdout(tmp_path):
    bp = _template(tmp_path)
    res = CliRunner().invoke(
        build_cli, ["render", "-f", str(bp), "--param", "ENVIRONMENT=skypilot/aws"]
    )
    assert res.exit_code == 0, res.output
    assert "skypilot/aws" in res.output and "$${" not in res.output


def test_render_writes_out_file(tmp_path):
    bp = _template(tmp_path)
    out = tmp_path / "resolved.yaml"
    res = CliRunner().invoke(
        build_cli,
        ["render", "-f", str(bp), "--param", "ENVIRONMENT=x", "-o", str(out)],
    )
    assert res.exit_code == 0, res.output
    doc = yaml.safe_load(out.read_text())
    assert doc["granite.build"]["targets"]["t1"]["environment_uri"].endswith("/x")


def test_render_missing_param_errors(tmp_path):
    tdir = tmp_path / "t2"
    tdir.mkdir()
    bp = tdir / "build.yaml"
    bp.write_text("granite.build:\n  name: $${NOPE}\n")
    res = CliRunner().invoke(build_cli, ["render", "-f", str(bp)])
    assert res.exit_code != 0
    assert "NOPE" in res.output
