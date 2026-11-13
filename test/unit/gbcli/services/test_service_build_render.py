# Copyright Granite.Build Authors
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


# --- gb build describe --raw --param / --param-file ---


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


# --- layered --param-file ---


def _layers(tmp_path: Path):
    """Recipe parameters.yaml + two layers; each later file overrides a key."""
    _write(tmp_path, "SUFFIX: base\nENVIRONMENT: base\n", name="parameters.yaml")
    site = _write(tmp_path, "SUFFIX: site\nENVIRONMENT: site\n", name="site.yaml")
    smoke = _write(tmp_path, "SUFFIX: smoke\n", name="smoke.yaml")
    return str(site), str(smoke)


def test_describe_raw_layered_files_last_wins(tmp_path):
    bp = _write(tmp_path)
    site, smoke = _layers(tmp_path)
    out = build_describe(
        github_token="",
        filename=str(bp),
        format="yaml",
        raw=True,
        parameters_path=(site, smoke),
    )
    doc = yaml.safe_load(out)
    assert doc["granite.build"]["name"] == "demo-smoke"
    assert doc["granite.build"]["targets"]["t1"]["environment_uri"].endswith("/site")


def test_describe_raw_layer_falls_back_to_recipe_file(tmp_path):
    bp = _write(tmp_path)
    _write(tmp_path, "SUFFIX: base\nENVIRONMENT: base\n", name="parameters.yaml")
    smoke = _write(tmp_path, "SUFFIX: smoke\n", name="smoke.yaml")
    out = build_describe(
        github_token="",
        filename=str(bp),
        format="yaml",
        raw=True,
        parameters_path=(str(smoke),),
    )
    doc = yaml.safe_load(out)
    # A key the layer does not define falls back to the recipe's parameters.yaml.
    assert doc["granite.build"]["name"] == "demo-smoke"
    assert doc["granite.build"]["targets"]["t1"]["environment_uri"].endswith("/base")


def test_describe_raw_param_beats_every_file(tmp_path):
    bp = _write(tmp_path)
    site, smoke = _layers(tmp_path)
    out = build_describe(
        github_token="",
        filename=str(bp),
        format="yaml",
        raw=True,
        params=["SUFFIX=cli", "ENVIRONMENT=cli"],
        parameters_path=(site, smoke),
    )
    doc = yaml.safe_load(out)
    assert doc["granite.build"]["name"] == "demo-cli"
    assert doc["granite.build"]["targets"]["t1"]["environment_uri"].endswith("/cli")


def test_describe_raw_missing_layer_reports_error(tmp_path):
    bp = _write(tmp_path)
    events = []

    def cb(callback_event=None, callback_args=None):
        events.append((callback_event, callback_args))

    result = build_describe(
        github_token="",
        filename=str(bp),
        format="yaml",
        raw=True,
        parameters_path=(str(tmp_path / "nope.yaml"),),
        callback=cb,
    )
    assert result is None
    reasons = [a["reason"] for e, a in events if e == "error"]
    assert any("nope.yaml" in r for r in reasons)


def test_cli_describe_accepts_repeated_param_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GB_CONFIG", str(tmp_path / "gbcli"))
    bp = _write(tmp_path)
    site, smoke = _layers(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        build_cli,
        [
            "describe",
            "-f",
            str(bp),
            "--raw",
            "--param-file",
            site,
            "--param-file",
            smoke,
            "--skip-version-check",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "name: demo-smoke" in result.output
    assert "space://environments/site" in result.output


def test_cli_deprecated_parameters_path_still_works(tmp_path, monkeypatch):
    # --parameters-path is a deprecated alias of --param-file; mixing the two
    # keeps command-line order (smoke after site wins either way round).
    monkeypatch.setenv("GB_CONFIG", str(tmp_path / "gbcli"))
    bp = _write(tmp_path)
    site, smoke = _layers(tmp_path)
    runner = CliRunner()
    for args in (
        ["--parameters-path", site, "--parameters-path", smoke],
        ["--parameters-path", site, "--param-file", smoke],
        ["--param-file", site, "--parameters-path", smoke],
    ):
        result = runner.invoke(
            build_cli,
            ["describe", "-f", str(bp), "--raw", *args, "--skip-version-check"],
        )
        assert result.exit_code == 0, result.output
        assert "name: demo-smoke" in result.output
        assert "space://environments/site" in result.output


def test_describe_raw_bad_layer_blames_parameters_not_build_yaml(tmp_path):
    bp = _write(tmp_path)
    bad = _write(tmp_path, "- not\n- a mapping\n", name="bad.yaml")
    events, cb = _recorder()
    result = build_describe(
        github_token="",
        filename=str(bp),
        format="yaml",
        raw=True,
        parameters_path=(str(bad),),
        callback=cb,
    )
    assert result is None
    errors = _errors(events)
    assert len(errors) == 1, errors
    assert errors[0].startswith("Error applying build parameters:")
    assert "bad.yaml: parameters must be a mapping" in errors[0]
    assert "is invalid" not in errors[0]


def test_start_dry_run_layered_files(tmp_path, monkeypatch):
    monkeypatch.setenv("GB_CACHE", str(tmp_path / "cache"))
    bp = _write(tmp_path)
    site, smoke = _layers(tmp_path)
    out = build_start(
        github_token="",
        quiet=True,
        filename=str(bp),
        parameters_path=(site, smoke),
        dry_run=True,
    )
    doc = yaml.safe_load(out)
    assert doc["granite.build"]["name"] == "demo-smoke"
    assert doc["granite.build"]["targets"]["t1"]["environment_uri"].endswith("/site")


def test_start_dry_run_param_beats_every_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GB_CACHE", str(tmp_path / "cache"))
    bp = _write(tmp_path)
    site, smoke = _layers(tmp_path)
    out = build_start(
        github_token="",
        quiet=True,
        filename=str(bp),
        params=["SUFFIX=cli"],
        parameters_path=(site, smoke),
        dry_run=True,
    )
    doc = yaml.safe_load(out)
    assert doc["granite.build"]["name"] == "demo-cli"
    assert doc["granite.build"]["targets"]["t1"]["environment_uri"].endswith("/site")


def test_cli_start_dry_run_accepts_repeated_param_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GB_CONFIG", str(tmp_path / "gbcli"))
    monkeypatch.setenv("GB_CACHE", str(tmp_path / "cache"))
    bp = _write(tmp_path)
    site, smoke = _layers(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        build_cli,
        [
            "start",
            "-f",
            str(bp),
            "--dry-run",
            "--param-file",
            site,
            "--param-file",
            smoke,
            "--skip-version-check",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "demo-smoke" in result.output


def _recorder():
    events = []

    def cb(callback_event=None, callback_args=None):
        events.append((callback_event, callback_args))

    return events, cb


def _errors(events):
    return [str(a.get("reason", "")) for e, a in events if e == "error"]


def test_start_dry_run_bad_layer_reports_one_error(tmp_path, monkeypatch):
    monkeypatch.setenv("GB_CACHE", str(tmp_path / "cache"))
    bp = _write(tmp_path)
    bad = _write(tmp_path, "- not\n- a mapping\n", name="bad.yaml")
    events, cb = _recorder()
    out = build_start(
        github_token="",
        quiet=True,
        filename=str(bp),
        parameters_path=(str(bad),),
        dry_run=True,
        callback=cb,
    )
    assert out is None
    errors = _errors(events)
    assert len(errors) == 1, errors
    assert "bad.yaml: parameters must be a mapping" in errors[0]
    assert "unpack" not in errors[0]


def test_start_dry_run_relative_build_file(tmp_path, monkeypatch):
    monkeypatch.setenv("GB_CACHE", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    _write(tmp_path)
    _layers(tmp_path)
    out = build_start(
        github_token="",
        quiet=True,
        filename="build.yaml",
        parameters_path=("smoke.yaml",),
        dry_run=True,
    )
    doc = yaml.safe_load(out)
    # smoke.yaml sets SUFFIX; ENVIRONMENT falls back to the cwd's parameters.yaml.
    assert doc["granite.build"]["name"] == "demo-smoke"
    assert doc["granite.build"]["targets"]["t1"]["environment_uri"].endswith("/base")


@pytest.fixture
def offline_validate(monkeypatch, tmp_path):
    """Stub the server side of build_validate; capture the submitted build.yaml."""
    import base64
    import io
    import types
    import zipfile

    import gbcli.services.service_build as sb

    monkeypatch.setenv("GB_CACHE", str(tmp_path / "cache"))
    submitted = {}

    def fake_validate_build(token, api, archive, space_name, user, **kwargs):
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(archive))) as z:
            submitted["build.yaml"] = z.read("build.yaml").decode("utf-8")
        return {"errors": [], "warnings": []}

    monkeypatch.setattr(sb, "get_user", lambda token: types.SimpleNamespace(login="u"))
    monkeypatch.setattr(
        sb,
        "resolve_space",
        lambda token, space, cb=None: {"git_repo_uri": "x", "name": "default"},
    )
    monkeypatch.setattr(sb, "validate_build", fake_validate_build)
    return submitted


def test_validate_layered_files(tmp_path, offline_validate):
    from gbcli.services.service_build import build_validate

    bp = _write(tmp_path)
    site, smoke = _layers(tmp_path)
    events, cb = _recorder()
    result = build_validate(
        github_token="",
        quiet=True,
        filename=str(bp),
        params=["ENVIRONMENT=cli"],
        parameters_path=(site, smoke),
        callback=cb,
    )
    assert _errors(events) == []
    assert result
    doc = yaml.safe_load(offline_validate["build.yaml"])
    assert doc["granite.build"]["name"] == "demo-smoke"
    assert doc["granite.build"]["targets"]["t1"]["environment_uri"].endswith("/cli")


def test_validate_bad_layer_reports_one_error(tmp_path, offline_validate):
    from gbcli.services.service_build import build_validate

    bp = _write(tmp_path)
    bad = _write(tmp_path, "just a string\n", name="bad.yaml")
    events, cb = _recorder()
    build_validate(
        github_token="",
        quiet=True,
        filename=str(bp),
        parameters_path=(str(bad),),
        callback=cb,
    )
    errors = _errors(events)
    assert len(errors) == 1, errors
    assert "bad.yaml: parameters must be a mapping" in errors[0]
    assert "build.yaml" not in offline_validate


def test_dotted_param_sets_one_nested_key_of_file_map(tmp_path):
    # Files replace a map whole; a dotted --param sets just one nested key.
    bp = _write(tmp_path, "granite.build:\n  name: demo-$${M.x}-$${M.y}\n")
    _write(tmp_path, "M:\n  x: 1\n  y: 2\n", name="parameters.yaml")
    out = build_describe(
        github_token="",
        filename=str(bp),
        format="yaml",
        raw=True,
        params=["M.y=3"],
    )
    assert yaml.safe_load(out)["granite.build"]["name"] == "demo-1-3"
