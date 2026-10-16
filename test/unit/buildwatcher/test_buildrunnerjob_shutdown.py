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

"""BuildRunnerJob: an orchestrator shutdown detaches, a user cancel still tears down.

The build-runner pod outlives the watcher, so ``on_build_orchestrator_shutdown()`` must
stop monitoring the job without marking the build CANCEL_REQUESTED or deleting the
job/pod. ``stop()`` (a user cancel) must still do both. Either request can arrive
before the build thread reaches ``start_and_wait()``; it must not be lost. The
Kubernetes calls and the admin storage are faked; the runner's own control flow is
what is under test.
"""

import asyncio
from types import SimpleNamespace

import pytest
from libgbtest.constants import HAS_K8S, requires_k8s

if HAS_K8S:  # buildrunnerjob imports kubernetes_asyncio at module level
    from gbserver.buildrunnerjob import buildrunnerjob as brj_mod
    from gbserver.buildrunnerjob.buildrunnerjob import BuildRunnerJob

from gbserver.types.status import Status

pytestmark = [pytest.mark.standalone, requires_k8s]

BUILD_ID = "11111111-2222-3333-4444-555555555555"


class _FakeApiClient:
    """Async context manager standing in for the kubernetes ApiClient."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _Recorder:
    """Records the side effects __start_job has on storage and the cluster."""

    def __init__(self):
        self.status_updates: list = []
        self.deleted_jobs: list[str] = []

    def update_fields(self, uuid, fields, should_update=None):
        self.status_updates.append(fields.get("status"))


def _job_runner(monkeypatch, recorder: _Recorder, on_first_read) -> "BuildRunnerJob":
    """Build a BuildRunnerJob with faked k8s/storage, bypassing its heavy __init__.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        recorder: collects storage updates and job deletions.
        on_first_read: called with the runner on the first job poll, to simulate
            the watcher's shutdown or a user cancel arriving while the job runs.

    Returns:
        A runner whose ``_BuildRunnerJob__start_job()`` can be awaited directly.
    """

    async def _create_api_client(**_kwargs):
        return _FakeApiClient()

    monkeypatch.setattr(
        brj_mod.AtomicApiClient, "create_api_client", _create_api_client
    )
    monkeypatch.setattr(brj_mod.client, "BatchV1Api", lambda api: None)
    monkeypatch.setattr(
        brj_mod, "get_admin_storage", lambda: SimpleNamespace(build_storage=recorder)
    )

    runner = BuildRunnerJob.__new__(BuildRunnerJob)
    runner.stored_build = SimpleNamespace(uuid=BUILD_ID)
    runner.monitoring_interval = 0.01
    runner.is_stop_requested = False
    runner.is_detach_requested = False
    runner.is_running = True
    reads = {"n": 0}

    async def _create(batchv1, body):
        return None

    async def _read(batchv1, name):
        reads["n"] += 1
        if reads["n"] == 1:
            on_first_read(runner)
        return SimpleNamespace()

    async def _delete(batchv1, name):
        recorder.deleted_jobs.append(name)

    runner._BuildRunnerJob__get_batchv1job_body = lambda: ("gb-build-runner-x", {})
    runner._BuildRunnerJob__create_namespaced_job_with_retry = _create
    runner._BuildRunnerJob__read_namespaced_job_with_retry = _read
    runner._BuildRunnerJob__delete_job_and_pod_with_retry = _delete
    return runner


@pytest.mark.timeout(30)
def test_orchestrator_shutdown_leaves_job_running(monkeypatch):
    """Shutdown ends monitoring with no CANCEL_REQUESTED write and no job delete."""
    recorder = _Recorder()
    runner = _job_runner(
        monkeypatch, recorder, lambda r: r.on_build_orchestrator_shutdown()
    )

    asyncio.run(runner._BuildRunnerJob__start_job())

    assert runner.is_detach_requested
    assert recorder.status_updates == [], "shutdown marked the build for cancel"
    assert recorder.deleted_jobs == [], "shutdown deleted the build-runner job"


@pytest.mark.timeout(30)
def test_stop_still_cancels_and_deletes_job(monkeypatch):
    """A user cancel (stop()) still marks CANCEL_REQUESTED and deletes the job."""
    recorder = _Recorder()
    runner = _job_runner(monkeypatch, recorder, lambda r: r.stop())

    asyncio.run(runner._BuildRunnerJob__start_job())

    assert recorder.status_updates == [Status.CANCEL_REQUESTED]
    assert recorder.deleted_jobs == ["gb-build-runner-x"]


def _unstarted_job_runner(monkeypatch) -> tuple["BuildRunnerJob", list]:
    """A BuildRunnerJob whose thread has not yet reached start_and_wait().

    Creating the job fails the test, and finalize_build_status calls are recorded.

    Returns:
        The runner and the list of statuses passed to finalize_build_status.
    """
    finalized: list = []
    monkeypatch.setattr(
        brj_mod,
        "finalize_build_status",
        lambda build_id, status, **_kw: finalized.append(status),
    )
    runner = BuildRunnerJob.__new__(BuildRunnerJob)
    runner.stored_build = SimpleNamespace(uuid=BUILD_ID)
    runner.is_stop_requested = False
    runner.is_detach_requested = False
    runner.is_running = False

    async def _no_job():
        pytest.fail("a job was created after the request arrived")

    runner._BuildRunnerJob__start_job = _no_job
    return runner, finalized


def test_shutdown_before_start_creates_no_job(monkeypatch):
    """An early shutdown is kept: no job, and the build is left for the next watcher."""
    runner, finalized = _unstarted_job_runner(monkeypatch)

    runner.on_build_orchestrator_shutdown()
    runner.start_and_wait()

    assert finalized == [], "the build should stay PENDING for the next watcher"


def test_stop_before_start_creates_no_job_and_cancels(monkeypatch):
    """An early user cancel is kept: no job, and the build is marked CANCELLED."""
    runner, finalized = _unstarted_job_runner(monkeypatch)

    runner.stop()
    runner.start_and_wait()

    assert finalized == [Status.CANCELLED]
