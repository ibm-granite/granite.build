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

"""recipes/distill/distill-eval/rerun_command.py, against recorded build state.

The fixtures under fixtures/distill_eval_rerun/ are a distill build as the gb CLI
reports it (`gb build status/describe --format json`, and the archive's
parameters-applied.yaml): CKPT_LADDER 1000,2000, a screening subset on rung 1000
and safety+math+bfcl on rung 2000, with

    eval-olmes-gsm-symbolic-ck2000  failed on every attempt, its URI registered
    eval-salad-bench-ck1000         failed, then succeeded on the build's retry
    eval-bfcl-ck1000                failed
    eval-attaq-ck2000               never dispatched

No gbserver is involved: the gb CLI is replaced by a reader of those files.
"""

import copy
import importlib.util
import json
import pathlib
import shlex
import subprocess

import pytest
import yaml

_REPO = pathlib.Path(__file__).resolve().parents[3]
_RECIPE = _REPO / "recipes" / "distill" / "distill-eval"
_SCRIPT = _RECIPE / "rerun_command.py"
_SITE = _REPO / "recipes" / "distill" / "site.example.yaml"
_FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "distill_eval_rerun"
_BUILD_ID = "5f0c2d1e-7a3b-4c9d-8e6f-0123456789ab"
_ROOT = f"/shared/distill/run1/{_BUILD_ID}"

_spec = importlib.util.spec_from_file_location("distill_eval_rerun_command", _SCRIPT)
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

_CATALOG = yaml.safe_load((_RECIPE / "parameters.yaml").read_text(encoding="utf-8"))[
    "EVAL_CATALOG"
]


def _status():
    return json.loads((_FIXTURES / "distill_build_status.json").read_text())


def _definition():
    return json.loads((_FIXTURES / "distill_build_describe.json").read_text())


def _applied():
    return yaml.safe_load(
        (_FIXTURES / "distill_build_parameters_applied.yaml").read_text()
    )


def _mark_all_success(status):
    seen = set()
    for t in status["targets"]:
        t["status"] = "success"
        seen.add(t["target_name"])
    for d in _definition():
        if d["target_name"] not in seen:
            status["targets"].append(
                {
                    "target_name": d["target_name"],
                    "status": "success",
                    "input_artifacts": [],
                    "output_artifacts": [],
                }
            )
    return status


# ─── plan() ───────────────────────────────────────────────────────────────────


def test_the_recorded_build_reruns_exactly_its_failed_evals():
    params, notes = rc.plan(_status(), _definition(), _CATALOG)
    assert params["EVALS"] == "1000:bfcl;2000:olmes-gsm-symbolic,attaq"
    assert params["EXPORTS"] == (
        f"1000=env://{_ROOT}/export-1000,2000=env://{_ROOT}/export-2000"
    )
    assert notes == []


def test_a_fully_measured_build_needs_nothing():
    params, notes = rc.plan(_mark_all_success(_status()), _definition(), _CATALOG)
    assert params is None and notes == []


def test_whole_rungs_reruns_every_eval_of_a_failed_rung():
    params, _ = rc.plan(_status(), _definition(), _CATALOG, whole_rungs=True)
    rungs = dict(spec.split(":") for spec in params["EVALS"].split(";"))
    assert rungs["1000"].split(",") == ["olmes-gsm8k", "salad-bench", "bfcl"]
    assert len(rungs["2000"].split(",")) == 8


def test_a_lost_csv_with_its_evals_done_is_noted_not_planned():
    status = _mark_all_success(_status())
    for t in status["targets"]:
        if t["target_name"] == "export-sage-ck2000":
            t["status"] = "failed"
    params, notes = rc.plan(status, _definition(), _CATALOG)
    assert params is None
    assert len(notes) == 1 and "export-sage-ck2000" in notes[0]
    params, _ = rc.plan(status, _definition(), _CATALOG, whole_rungs=True)
    assert params["EVALS"].startswith("2000:")


def test_only_the_table_lost_is_noted():
    status = _mark_all_success(_status())
    for t in status["targets"]:
        if t["target_name"] == "export-combined":
            t["status"] = "failed"
    params, notes = rc.plan(status, _definition(), _CATALOG)
    assert params is None
    assert notes and "export-combined" in notes[0]


def test_a_lost_export_is_reported_not_planned():
    status = _status()
    status["targets"] = [
        t
        for t in status["targets"]
        if not (
            t["target_name"] == "export-2000" or t["target_name"].endswith("-ck2000")
        )
    ]
    params, notes = rc.plan(status, _definition(), _CATALOG)
    assert params["EVALS"] == "1000:bfcl"
    assert params["EXPORTS"] == f"1000=env://{_ROOT}/export-1000"
    assert any("rungs 2000" in n for n in notes)


def test_a_distill_eval_builds_models_come_from_its_eval_inputs():
    """A distill-eval build has no export-<N>; the eval's input is the model."""
    status = copy.deepcopy(_status())
    status["targets"] = [
        t
        for t in status["targets"]
        if t["target_name"].startswith(("eval-", "export-"))
        and not t["target_name"][7:].isdigit()
    ]
    for t in status["targets"]:
        for a in t["input_artifacts"]:
            a["uri"] = a["uri"].replace(f"env://{_ROOT}/export-", "hf://org/m-")
    definition = [
        d
        for d in _definition()
        if d["target_name"].startswith(("eval-", "export-"))
        and not d["target_name"][7:].isdigit()
    ]
    params, _ = rc.plan(status, definition, _CATALOG)
    assert params["EXPORTS"] == "1000=hf://org/m-1000,2000=hf://org/m-2000"


def test_a_rung_without_a_recorded_model_is_an_error():
    status = _status()
    for t in status["targets"]:
        if t["target_name"] == "export-1000":
            t["output_artifacts"] = []
        if t["target_name"].endswith("-ck1000"):
            t["input_artifacts"] = []
    with pytest.raises(rc.RerunError, match="1000"):
        rc.plan(status, _definition(), _CATALOG)


def test_an_eval_missing_from_the_catalog_is_an_error():
    definition = _definition() + [{"target_name": "eval-no-such-eval-ck1000"}]
    with pytest.raises(rc.RerunError):
        rc.plan(_status(), definition, _CATALOG)


@pytest.mark.parametrize(
    "name, matches",
    [
        ("eval-olmes-gsm8k-ck1000", "eval"),
        ("eval-bfcl-ck2000", "eval"),
        ("eval-include-ar-de-es-fr-ck500", "eval"),
        ("export-1000", "export"),
        ("export-sage-ck1000", "csv"),
        ("export-bfcl-ck1000", "csv"),
        ("export-combined", None),
        ("divergence-1000", None),
        # The retired recipes' names are not this recipe's targets.
        ("eval-transfer-1000", None),
        ("merge-sage-ck1000", None),
    ],
)
def test_target_name_patterns_are_distills(name, matches):
    got = {
        "eval": rc._EVAL_TARGET.match(name) and not name.startswith("eval-transfer"),
        "export": rc._EXPORT_TARGET.match(name),
        "csv": rc._CSV_TARGET.match(name),
    }
    hits = [k for k, v in got.items() if v]
    assert hits == ([matches] if matches else []), name


def test_the_patterns_name_targets_distill_renders(tmp_path):
    """Every eval/export/csv name the helper matches is one distill/build.yaml emits."""
    from gbcli.services.service_build import get_params_from_file
    from gbcli.utils.buildutil import apply_parameters

    distill = _REPO / "recipes" / "distill" / "distill"
    merged = {}
    for layer in (distill / "parameters.yaml", _SITE):
        merged.update(get_params_from_file(str(layer)))
    text = apply_parameters(
        (distill / "build.yaml").read_text(encoding="utf-8"),
        ["SCREEN_EVALS=olmes-gsm8k,bfcl", "FULL_EVAL_RUNGS=8000"],
        merged,
        str(tmp_path),
    )
    names = set(yaml.safe_load(text)["granite.build"]["targets"])
    assert {n for n in names if rc._EVAL_TARGET.match(n)}
    assert {n for n in names if rc._EXPORT_TARGET.match(n)}
    assert {n for n in names if rc._CSV_TARGET.match(n)}
    assert rc._COMBINED in names


# ─── overrides() and the render check ─────────────────────────────────────────


def test_overrides_carry_recipe_knobs_only():
    defaults = yaml.safe_load((_RECIPE / "parameters.yaml").read_text())
    site = set(yaml.safe_load(_SITE.read_text()))
    extra = rc.overrides(_applied(), defaults, site)
    # Read by distill-eval and different from its default.
    assert extra == {
        "BFCL_TEST_CATEGORIES": "simple",
        "EVAL_CATEGORIES": "safety,math,bfcl",
    }


def test_the_planned_parameters_render():
    params, _ = rc.plan(_status(), _definition(), _CATALOG)
    params.update(RUN_NAME="run1", EXPERIMENT="run1", RERUN_TAG="rerun-t")
    targets = rc.render_check(_SITE, {"EVAL_CATEGORIES": "safety,math,bfcl", **params})
    assert sorted(targets) == [
        "eval-attaq-ck2000",
        "eval-bfcl-ck1000",
        "eval-olmes-gsm-symbolic-ck2000",
        "export-bfcl-ck1000",
        "export-combined",
        "export-sage-ck2000",
    ]


# ─── main(), with the gb CLI replaced by the fixtures ─────────────────────────


@pytest.fixture(name="fake_gb")
def _fake_gb(monkeypatch):
    calls = []
    state = {"status": _status(), "applied": _applied()}

    def run(cmd, **kw):
        calls.append(cmd)
        assert cmd[0] == "gb" and kw["env"]["GB_ENVIRONMENT"]
        sub = cmd[2]
        if sub == "status":
            out = json.dumps(state["status"])
        elif sub == "describe":
            out = json.dumps(_definition())
        elif sub == "init":
            dest = pathlib.Path(cmd[3])
            dest.mkdir(parents=True)
            (dest / "parameters-applied.yaml").write_text(
                yaml.safe_dump(state["applied"])
            )
            out = ""
        else:
            raise AssertionError(cmd)
        return subprocess.CompletedProcess(cmd, 0, out, "")

    monkeypatch.setattr(rc.subprocess, "run", run)
    return calls, state


def _command(stdout):
    return shlex.split(stdout.replace("\\\n", " "))


def test_main_prints_a_distill_eval_command(fake_gb, capsys):
    calls, _ = fake_gb
    assert rc.main([_BUILD_ID, "--site", str(_SITE), "--rerun-tag", "rerun-x"]) == 0
    out = capsys.readouterr().out
    argv = _command(out)
    assert argv[:5] == [
        "gb",
        "build",
        "start",
        "-f",
        "recipes/distill/distill-eval/build.yaml",
    ]
    assert argv[5:7] == ["--parameters-path", str(_SITE)]
    params = dict(a.split("=", 1) for a in argv[8::2])
    assert argv[7::2] == ["--param"] * len(params)
    assert params["EVALS"] == "1000:bfcl;2000:olmes-gsm-symbolic,attaq"
    assert params["RERUN_TAG"] == "rerun-x"
    assert params["EXPERIMENT"] == "run1" and params["RUN_NAME"] == "run1"
    assert [c[2] for c in calls] == ["status", "describe", "init"]


def test_main_never_prints_site_values(fake_gb, capsys):
    assert rc.main([_BUILD_ID, "--site", str(_SITE)]) == 0
    out = capsys.readouterr().out
    params = dict(a.split("=", 1) for a in _command(out)[8::2])
    site = yaml.safe_load(_SITE.read_text())
    assert not set(params) & set(site)
    # EVAL_PARTITION is in the build's applied parameters and must not ride along.
    assert "gpu-preemptible" not in out


def test_main_defaults_to_the_users_site_file(fake_gb, capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert rc.main([_BUILD_ID]) == 1
    assert "pass --site" in capsys.readouterr().err
    site = tmp_path / ".granite.build" / "local-env" / "distill-ibm-site.yaml"
    site.parent.mkdir(parents=True)
    site.write_text(_SITE.read_text())
    assert rc.main([_BUILD_ID]) == 0
    argv = _command(capsys.readouterr().out)
    assert argv[5:7] == [
        "--parameters-path",
        "~/.granite.build/local-env/distill-ibm-site.yaml",
    ]


def test_main_refuses_a_running_build(fake_gb, capsys):
    _, state = fake_gb
    state["status"]["details"]["status"] = "running"
    assert rc.main([_BUILD_ID, "--site", str(_SITE)]) == 2
    assert capsys.readouterr().out == ""
    assert rc.main([_BUILD_ID, "--site", str(_SITE), "--allow-running"]) == 0


def test_main_with_nothing_to_rerun_prints_nothing(fake_gb, capsys):
    _, state = fake_gb
    state["status"] = _mark_all_success(state["status"])
    assert rc.main([_BUILD_ID, "--site", str(_SITE)]) == 0
    assert capsys.readouterr().out == ""


def test_main_refuses_to_reuse_the_builds_own_tag(fake_gb, capsys):
    _, state = fake_gb
    state["applied"]["RERUN_TAG"] = "t0"
    assert rc.main([_BUILD_ID, "--site", str(_SITE), "--rerun-tag", "t0"]) == 1
    assert "build's own" in capsys.readouterr().err


def test_a_gb_failure_is_reported(monkeypatch, capsys):
    monkeypatch.setattr(
        rc.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "unreachable"),
    )
    assert rc.main([_BUILD_ID, "--site", str(_SITE)]) == 1
    assert "unreachable" in capsys.readouterr().err
