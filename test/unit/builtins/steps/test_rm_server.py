"""Unit tests for the rm-server (reward model FastAPI) step asset.

rm-server is a declarative SERVICE step.yaml companion to bcb-server. It launches
open_instruct.servers.rm_server and emits the service URL as an rm_server_url binding
by scraping the FastAPI startup log line. These tests load the shipped asset, render
its run: block, and assert the launch command, env exports, and monitor URL extraction.
"""

import re
from pathlib import Path

import yaml

from gbserver.utils.template import fill_template

REPO_ROOT = Path(__file__).resolve().parents[4]
RM_STEP_YAML = (
    REPO_ROOT
    / "configurations/assets/environments/skypilot/lsf/ibm-bluevela/steps/rm-server/step.yaml"
)


def _load() -> dict:
    return yaml.safe_load(RM_STEP_YAML.read_text())


def _render_run(overrides: dict | None = None) -> str:
    cfg = _load()
    rm = {**cfg["config"]["rm_server_config"], **(overrides or {})}
    run = cfg["environment_configs"]["Skypilot"]["launchers"]["rm-server"]["config"][
        "run"
    ]
    return fill_template(run, {"config": {"rm_server_config": rm}}, strict=False)


class TestRmServerStep:
    def test_identity_and_output(self):
        cfg = _load()
        assert cfg["name"] == "rm-server"
        assert cfg["type"] == "SERVICE"
        assert "rm_server_config" in cfg["config"]
        assert cfg["outputs"]["optional"]["rm_server_url"]["type"] == "dataset"

    def test_launcher_shape(self):
        cfg = _load()
        rm = cfg["environment_configs"]["Skypilot"]["launchers"]["rm-server"]
        assert rm["type"] == "skypilot"
        assert rm["monitors"] == ["skypilot_monitor"]
        assert "open-instruct-3:0.1.0-noconda" in rm["config"]["image_id"]
        assert rm["config"]["resources"]["accelerators"]

    def test_launch_command(self):
        run = _render_run()
        assert "open_instruct.servers.rm_server" in run
        assert "exec $PYTHON_BIN -u -m open_instruct.servers.rm_server" in run

    def test_env_exports_when_set(self):
        run = _render_run(
            {
                "model_path": "/proj/models/phi4",
                "idle_timeout": "3600",
                "hf_token": "hf_x",
            }
        )
        assert 'export RM_SERVER_MODEL="/proj/models/phi4"' in run
        assert 'export IDLE_TIMEOUT="3600"' in run
        assert 'export HF_TOKEN="hf_x"' in run

    def test_env_absent_when_unset(self):
        run = _render_run({"idle_timeout": "", "hf_token": ""})
        assert "IDLE_TIMEOUT" not in run
        assert "HF_TOKEN" not in run

    def test_status_event_running(self):
        cfg = _load()
        events = cfg["environment_configs"]["Skypilot"]["monitors"]["skypilot_monitor"][
            "config"
        ]["event_configs"]
        status = next(e for e in events if e["event_type"] == "WORKLOAD_STATUS_EVENT")
        assert re.search(status["line_regex"], "Starting FastAPI server on host:8000")
        status_field = next(
            f for f in status["event_fields"] if f["field_name"] == "status"
        )
        assert status_field["field_value_template"] == "RUNNING"

    def test_monitor_url_tie_in(self):
        cfg = _load()
        events = cfg["environment_configs"]["Skypilot"]["monitors"]["skypilot_monitor"][
            "config"
        ]["event_configs"]
        newart = next(
            e for e in events if e["event_type"] == "NEWARTIFACT_IN_ENVIRONMENT_EVENT"
        )
        sample = "Starting FastAPI server on host123:8000"
        assert re.search(newart["line_regex"], sample)
        path_field = next(
            f for f in newart["event_fields"] if f["field_name"] == "path"
        )
        m = re.search(path_field["field_regex"], sample)
        assert m and m.group(0) == "host123:8000"
        # The scraped line is produced by Python logging (has a prefix) and may
        # carry a trailing CR; the path regex must still extract clean host:port.
        for sample in (
            "2026-06-15 10:00:00 INFO server: Starting FastAPI server on host123:8000",
            "Starting FastAPI server on host123:8000\r",
        ):
            m2 = re.search(path_field["field_regex"], sample)
            assert m2 and m2.group(0) == "host123:8000", sample
        binding_field = next(
            f for f in newart["event_fields"] if f["field_name"] == "binding"
        )
        assert "http://{{ fields.data.path }}" in binding_field["field_value_template"]
        binding_id_field = next(
            f for f in newart["event_fields"] if f["field_name"] == "binding_id"
        )
        assert binding_id_field["field_value_template"] == "rm_server_url"
