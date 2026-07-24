# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import ast
import asyncio
from pathlib import Path

import models
from services.impl.runner import Runner
from services.plugins import Seam, register_override
from tests.fakes.runners import FakeRunner


class _MinimalRunner(Runner):
    async def run(self):
        return "ran"


def _cfg():
    # Minimal valid TuningConfig stand-in: Runner.__init__ only truthiness-checks it.
    return {"config_id": "c1"}


def test_runner_default_supports_remote_cancel_is_false():
    r = _MinimalRunner(job_id="j1", run_config=_cfg())
    assert r.supports_remote_cancel() is False


def test_runner_default_cancel_is_noop():
    r = _MinimalRunner(job_id="j1", run_config=_cfg())
    asyncio.run(r.cancel())  # must not raise


def test_setup_py_declares_runner_entry_points():
    setup_src = Path(__file__).resolve().parent.parent.joinpath("setup.py").read_text()
    tree = ast.parse(setup_src)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "entry_points":
            # entry_points={"autotunex.runners": ["local = ...", "gb = ..."]}
            for k, v in zip(node.value.keys, node.value.values):
                if isinstance(k, ast.Constant):
                    found[k.value] = [
                        e.value for e in v.elts if isinstance(e, ast.Constant)
                    ]
    assert "autotunex.runners" in found, "missing autotunex.runners group"
    joined = " ".join(found["autotunex.runners"])
    assert "local = services.runners.local_runner:LocalRunner" in joined
    assert "gb = services.runners.gb_runner:GBRunner" in joined


class _StubDB:
    """Async DB stub capturing the calls Job.start() makes before dispatch."""

    def __init__(self):
        self.inserted_job = False

    def create_logging_table_sync(self):
        return None

    async def get_config(self, config_id):
        return {
            "name": "n",
            "tuner_type": "SFT",
            "rl_tuner_type": None,
            "config_data": {},
        }

    async def insert_job(self, run_config, config_snapshot=None):
        self.inserted_job = True
        return "job-123"

    async def insert_task(self, task):
        return None

    async def update_job_status(self, id, status):
        return None

    async def update_all_trial_status(self, job_id, status):
        return None


class _BG:
    def __init__(self):
        self.tasks = []

    def add_task(self, fn, *a, **kw):
        self.tasks.append((fn, a, kw))


async def test_start_dispatches_resolved_runner(monkeypatch):
    import services.job_service as js

    created = {}

    def _factory(**kwargs):
        runner = FakeRunner(**kwargs)
        created["runner"] = runner
        return runner

    register_override(Seam.RUNNER, "local", _factory)
    monkeypatch.setenv("AUTOTUNEX_RUNNER", "local")

    db = _StubDB()
    job = js.Job(db=db)
    bg = _BG()

    cfg = models.TuningConfig(
        config_id="c1", dataset_id="d1", model="m", experiment_name="test-exp"
    )
    result = await job.start(run_config=cfg, background_task=bg)

    assert result["job_id"] == "job-123"
    assert created["runner"].job_id == "job-123"
    # The scheduled background task is the resolved runner's run method.
    assert bg.tasks and bg.tasks[0][0] == created["runner"].run
