"""Unit tests for the code-server (code-execution FastAPI) step asset.

code-server is a CPU-only declarative SERVICE step.yaml companion to bcb-server/rm-server.
It launches a configurable open_instruct code server module and emits the service URL as a
code_server_url binding by scraping the FastAPI startup log line.
"""

import re
from pathlib import Path

import yaml

from gbserver.utils.template import fill_template

REPO_ROOT = Path(__file__).resolve().parents[4]
CODE_STEP_YAML = (
    REPO_ROOT
    / "configurations/assets/environments/skypilot/lsf/ibm-bluevela/steps/code-server/step.yaml"
)


def _load() -> dict:
    return yaml.safe_load(CODE_STEP_YAML.read_text())


def _render_run(overrides: dict | None = None) -> str:
    cfg = _load()
    cs = {**cfg["config"]["code_server_config"], **(overrides or {})}
    run = cfg["environment_configs"]["Skypilot"]["launchers"]["code-server"]["config"][
        "run"
    ]
    return fill_template(run, {"config": {"code_server_config": cs}}, strict=False)


class TestCodeServerStep:
    def test_identity_and_output(self):
        cfg = _load()
        assert cfg["name"] == "code-server"
        assert cfg["type"] == "SERVICE"
        assert "code_server_config" in cfg["config"]
        assert cfg["outputs"]["optional"]["code_server_url"]["type"] == "dataset"

    def test_launcher_is_cpu_only(self):
        cfg = _load()
        cs = cfg["environment_configs"]["Skypilot"]["launchers"]["code-server"]
        assert cs["type"] == "skypilot"
        assert cs["monitors"] == ["skypilot_monitor"]
        assert "open-instruct-3:0.1.0-noconda" in cs["config"]["image_id"]
        assert cs["config"]["resources"]["cpus"]
        assert "accelerators" not in cs["config"]["resources"]

    def test_launch_command(self):
        run = _render_run()
        assert "exec $PYTHON_BIN -u -m open_instruct.servers.code_server" in run
        assert "cd /stage" in run

    def test_container_path_is_parameterizable(self):
        run = _render_run({"container_path": "/usr/local/go/bin:/custom/bin"})
        assert 'export PATH="/usr/local/go/bin:/custom/bin:${PATH}"' in run

    def test_pythonpath_and_module_render(self):
        run = _render_run(
            {
                "container_pythonpath": "/workspace",
                "container_home": "/workspace",
                "module": "open_instruct.servers.code_server",
            }
        )
        assert 'export PYTHONPATH="/workspace:${PYTHONPATH:-}"' in run
        assert "cd /workspace" in run

    def test_path_exports_guarded(self):
        # default config → both exports present
        run = _render_run()
        assert 'export PATH="' in run
        assert 'export PYTHONPATH="' in run
        # empty values → guarded out entirely (no leading-":" PATH element)
        empty = _render_run({"container_path": "", "container_pythonpath": ""})
        assert 'export PATH="' not in empty
        assert 'export PYTHONPATH="' not in empty

    def test_idle_timeout_absent_when_unset(self):
        run = _render_run({"idle_timeout": ""})
        assert "IDLE_TIMEOUT" not in run

    def test_status_event_running(self):
        cfg = _load()
        events = cfg["environment_configs"]["Skypilot"]["monitors"]["skypilot_monitor"][
            "config"
        ]["event_configs"]
        status = next(e for e in events if e["event_type"] == "WORKLOAD_STATUS_EVENT")
        assert re.search(
            status["line_regex"], "Starting FastAPI server on codehost:9000"
        )
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
        sample = "Starting FastAPI server on codehost:9000"
        assert re.search(newart["line_regex"], sample)
        path_field = next(
            f for f in newart["event_fields"] if f["field_name"] == "path"
        )
        m = re.search(path_field["field_regex"], sample)
        assert m and m.group(0) == "codehost:9000"
        # log-prefixed and CRLF-terminated lines must still yield clean host:port
        for s in (
            "2026-06-15 10:00:00 INFO server: Starting FastAPI server on codehost:9000",
            "Starting FastAPI server on codehost:9000\r",
        ):
            m2 = re.search(path_field["field_regex"], s)
            assert m2 and m2.group(0) == "codehost:9000", s
        binding_field = next(
            f for f in newart["event_fields"] if f["field_name"] == "binding"
        )
        assert "http://{{ fields.data.path }}" in binding_field["field_value_template"]
        binding_id_field = next(
            f for f in newart["event_fields"] if f["field_name"] == "binding_id"
        )
        assert binding_id_field["field_value_template"] == "code_server_url"
