# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Offline parameter rendering via `gb build describe --raw` and
`gb build start --dry-run` (issue #278 / PR #280 review).

The standalone `gb build render` subcommand was folded into the two commands
that already own the build file; both reuse the same `apply_parameters` engine,
so `gb` stays the single source of truth. These exercise the offline paths that
need no auth, space, or live server.
"""

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from gbcli.commands.command_build import cli as build_cli
from gbcli.services.service_build import build_describe, build_start

_PARAM_BUILD = (
    "granite.build:\n"
    "  name: demo-$${SUFFIX}\n"
    "  targets:\n"
    "    t1:\n"
    "      environment_uri: space://environments/$${ENVIRONMENT}\n"
)


def _write(dir_path: Path, text: str = "", name: str = "build.yaml") -> Path:
    p = dir_path / name
    p.write_text(text or _PARAM_BUILD, encoding="utf-8")
    return p


# --- gb build describe --raw --param / --parameters-path ---


def test_describe_raw_param_resolves(tmp_path):
    bp = _write(tmp_path)
    out = build_describe(
        github_token="",
        filename=str(bp),
        format="yaml",
        raw=True,
        params=["SUFFIX=1", "ENVIRONMENT=skypilot/aws"],
    )
    assert "$${" not in out
    doc = yaml.safe_load(out)
    assert doc["granite.build"]["name"] == "demo-1"
    assert (
        doc["granite.build"]["targets"]["t1"]["environment_uri"]
        == "space://environments/skypilot/aws"
    )


def test_describe_raw_no_param_is_verbatim(tmp_path):
    bp = _write(tmp_path)
    # No params -> the (possibly parameterized) file is dumped verbatim.
    out = build_describe(github_token="", filename=str(bp), format="yaml", raw=True)
    assert out == _PARAM_BUILD
    assert "$${ENVIRONMENT}" in out


def test_describe_raw_params_from_file(tmp_path):
    bp = _write(tmp_path)
    params_file = tmp_path / "params.yaml"
    params_file.write_text("SUFFIX: 9\nENVIRONMENT: bash\n", encoding="utf-8")
    out = build_describe(
        github_token="",
        filename=str(bp),
        format="yaml",
        raw=True,
        parameters_path=str(params_file),
    )
    doc = yaml.safe_load(out)
    assert doc["granite.build"]["name"] == "demo-9"
    assert doc["granite.build"]["targets"]["t1"]["environment_uri"].endswith("/bash")


def test_describe_raw_param_overrides_file(tmp_path):
    bp = _write(tmp_path)
    params_file = tmp_path / "params.yaml"
    params_file.write_text("SUFFIX: 9\nENVIRONMENT: bash\n", encoding="utf-8")
    out = build_describe(
        github_token="",
        filename=str(bp),
        format="yaml",
        raw=True,
        params=["ENVIRONMENT=k8s"],
        parameters_path=str(params_file),
    )
    doc = yaml.safe_load(out)
    # --param wins over the file; the file still supplies SUFFIX.
    assert doc["granite.build"]["name"] == "demo-9"
    assert doc["granite.build"]["targets"]["t1"]["environment_uri"].endswith("/k8s")


def test_describe_raw_missing_param_reports_error(tmp_path):
    bp = _write(tmp_path, "granite.build:\n  name: $${NOPE}\n")
    events = []

    def cb(callback_event=None, callback_args=None):
        events.append((callback_event, callback_args))

    result = build_describe(
        github_token="",
        filename=str(bp),
        format="yaml",
        raw=True,
        params=["UNRELATED=x"],
        callback=cb,
    )
    # An unresolved placeholder is reported via the error callback and yields no
    # output (the CLI then exits non-zero); it is not raised to the caller.
    assert result is None
    reasons = [a["reason"] for e, a in events if e == "error"]
    assert any("missing parameter" in r and "NOPE" in r for r in reasons)


def test_describe_raw_rejects_param_with_build_id():
    # A stored build_id is already fully resolved, so --param/--parameters-path
    # must not be re-applied to it; the CLI rejects the combination outright
    # rather than silently ignoring the params (PR #280 review, item #1).
    runner = CliRunner()
    result = runner.invoke(
        build_cli,
        [
            "describe",
            "00000000-0000-0000-0000-000000000000",
            "--raw",
            "--param",
            "SUFFIX=1",
            "--skip-version-check",
        ],
    )
    assert result.exit_code != 0
    assert "cannot be combined with a build_id" in result.output


# --- gb build start --dry-run --save-build-file (offline, no auth/space/submit) ---


def test_start_dry_run_returns_resolved(tmp_path, monkeypatch):
    monkeypatch.setenv("GB_CACHE", str(tmp_path / "cache"))
    bp = _write(tmp_path)
    out = build_start(
        github_token="",
        quiet=True,
        filename=str(bp),
        params=["SUFFIX=2", "ENVIRONMENT=bash"],
        dry_run=True,
    )
    assert "$${" not in out
    doc = yaml.safe_load(out)
    assert doc["granite.build"]["name"] == "demo-2"
    assert doc["granite.build"]["targets"]["t1"]["environment_uri"].endswith("/bash")


def test_start_dry_run_writes_save_build_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GB_CACHE", str(tmp_path / "cache"))
    bp = _write(tmp_path)
    dest = tmp_path / "resolved.yaml"
    out = build_start(
        github_token="",
        quiet=True,
        filename=str(bp),
        params=["SUFFIX=3", "ENVIRONMENT=k8s"],
        dry_run=True,
        save_build_file=str(dest),
    )
    assert dest.is_file()
    assert dest.read_text(encoding="utf-8") == out
    assert "space://environments/k8s" in out
