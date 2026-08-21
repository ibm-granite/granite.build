# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for the `gbtest render` subcommand dispatch (issue #278)."""

from libgbtest.buildrunner import gbtest


def test_render_prints_skeleton(tmp_path, monkeypatch, capsys):
    bp = tmp_path / "build.yaml"
    bp.write_text(
        "granite.build:\n  name: d\n  targets:\n    t1:\n"
        "      environment_uri: space://environments/skypilot/slurm\n"
        "      outputs:\n        o: {uri: 'env:///tmp/x', type: fileset}\n"
        "      steps:\n        - step_uri: space://steps/command\n"
    )
    monkeypatch.setattr("sys.argv", ["gbtest", "render", str(bp)])
    rc = gbtest.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "target_expectations" in out and "FIXME" in out


def test_render_requires_build_yaml_arg(monkeypatch):
    monkeypatch.setattr("sys.argv", ["gbtest", "render"])
    assert gbtest.main() == 2
