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

"""Unit tests for src/clearml_launch.py.

The wrapper's whole job is: run the selected NeMo-RL driver IN THIS PROCESS
(so ClearML's TensorBoard/stdout/GPU auto-capture, which only hooks the
Task.init() process, sees it), arming Task.init() first when ClearML is enabled
and importing nothing from clearml when it is not.
"""

import sys
import types

import clearml_launch
import pytest


class TestEnablement:
    _CREDS = {
        "CLEARML_API_HOST": "h",
        "CLEARML_API_ACCESS_KEY": "a",
        "CLEARML_API_SECRET_KEY": "s",
    }

    def test_off_is_never_enabled(self):
        assert clearml_launch.clearml_enabled("off", self._CREDS) is False

    def test_on_is_always_enabled(self):
        assert clearml_launch.clearml_enabled("on", {}) is True

    def test_auto_needs_all_three_creds(self):
        assert clearml_launch.clearml_enabled("auto", self._CREDS) is True
        partial = dict(self._CREDS, CLEARML_API_SECRET_KEY="")
        assert clearml_launch.clearml_enabled("auto", partial) is False
        assert clearml_launch.clearml_enabled("auto", {}) is False


class TestDriverExecution:
    def test_disabled_runs_driver_without_importing_clearml(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "clearml", raising=False)

        captured = {}

        def fake_run_path(path, run_name=None):
            captured["path"] = path
            captured["argv"] = list(sys.argv)
            captured["run_name"] = run_name

        monkeypatch.setattr(clearml_launch.runpy, "run_path", fake_run_path)
        # Importing clearml must fail loudly if the disabled path touches it.
        monkeypatch.setitem(sys.modules, "clearml", None)

        clearml_launch.main(
            [
                "--clearml-mode",
                "off",
                "--driver",
                "/opt/nemo-rl/examples/run_sft.py",
                "--",
                "--config",
                "cfg.yaml",
                "policy.model_name=/m",
            ]
        )

        assert captured["path"] == "/opt/nemo-rl/examples/run_sft.py"
        assert captured["run_name"] == "__main__"
        assert captured["argv"] == [
            "/opt/nemo-rl/examples/run_sft.py",
            "--config",
            "cfg.yaml",
            "policy.model_name=/m",
        ]

    def test_enabled_inits_and_closes_task_around_driver(self, monkeypatch):
        events = []

        class FakeTask:
            def close(self):
                events.append("close")

        def fake_init(project_name, task_name, output_uri):
            events.append(("init", project_name, task_name, output_uri))
            return FakeTask()

        fake_clearml = types.SimpleNamespace(Task=types.SimpleNamespace(init=fake_init))
        monkeypatch.setitem(sys.modules, "clearml", fake_clearml)

        def fake_run_path(path, run_name=None):
            events.append("run")

        monkeypatch.setattr(clearml_launch.runpy, "run_path", fake_run_path)

        clearml_launch.main(
            [
                "--clearml-mode",
                "on",
                "--clearml-project",
                "proj",
                "--clearml-task",
                "mytask",
                "--driver",
                "/opt/nemo-rl/examples/run_sft.py",
                "--",
                "--config",
                "cfg.yaml",
            ]
        )

        assert events == [
            ("init", "proj", "mytask", False),
            "run",
            "close",
        ]

    def test_task_closed_even_when_driver_raises(self, monkeypatch):
        events = []

        class FakeTask:
            def close(self):
                events.append("close")

        fake_clearml = types.SimpleNamespace(
            Task=types.SimpleNamespace(init=lambda **kw: FakeTask())
        )
        monkeypatch.setitem(sys.modules, "clearml", fake_clearml)

        def boom(path, run_name=None):
            raise RuntimeError("driver failed")

        monkeypatch.setattr(clearml_launch.runpy, "run_path", boom)

        with pytest.raises(RuntimeError, match="driver failed"):
            clearml_launch.main(
                [
                    "--clearml-mode",
                    "on",
                    "--driver",
                    "/opt/nemo-rl/examples/run_sft.py",
                    "--",
                    "--config",
                    "cfg.yaml",
                ]
            )
        assert events == ["close"]
