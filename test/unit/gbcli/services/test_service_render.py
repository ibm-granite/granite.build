# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for gbcli.services.service_render.render_build_yaml (issue #278)."""

from pathlib import Path

import pytest
import yaml

from gbcli.services.service_render import RenderError, render_build_yaml


def _template(tmp_path: Path) -> Path:
    tdir = tmp_path / "tmpl"
    tdir.mkdir()
    (tdir / "build.yaml").write_text(
        "granite.build:\n"
        "  name: demo\n"
        "  targets:\n"
        "    t1:\n"
        "      environment_uri: space://environments/$${ENVIRONMENT}\n"
    )
    (tdir / "parameters.yaml").write_text("ENVIRONMENT: skypilot/slurm\n")
    return tdir / "build.yaml"


def test_render_uses_sibling_parameters_yaml(tmp_path):
    bp = _template(tmp_path)
    text = render_build_yaml(bp, cli_params=[], parameters_path=None)
    doc = yaml.safe_load(text)
    assert (
        doc["granite.build"]["targets"]["t1"]["environment_uri"]
        == "space://environments/skypilot/slurm"
    )
    assert "$${" not in text


def test_render_cli_param_overrides_file(tmp_path):
    bp = _template(tmp_path)
    text = render_build_yaml(
        bp, cli_params=["ENVIRONMENT=skypilot/aws"], parameters_path=None
    )
    assert "skypilot/aws" in (
        yaml.safe_load(text)["granite.build"]["targets"]["t1"]["environment_uri"]
    )


def test_render_missing_param_raises_render_error(tmp_path):
    tdir = tmp_path / "t"
    tdir.mkdir()
    bp = tdir / "build.yaml"
    bp.write_text("granite.build:\n  name: $${MISSING}\n")
    with pytest.raises(RenderError) as exc:
        render_build_yaml(bp, cli_params=[], parameters_path=None)
    assert "MISSING" in str(exc.value)


def test_render_does_not_write_side_effect_files(tmp_path):
    bp = _template(tmp_path)
    render_build_yaml(bp, cli_params=["ENVIRONMENT=x"], parameters_path=None)
    # apply_parameters writes parameters-applied.yaml into a scratch dir, not the
    # template dir.
    assert not (bp.parent / "parameters-applied.yaml").exists()
