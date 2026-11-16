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

"""Tests for ``gbserver.api.environment_queue``.

squeue lines follow Slurm 25.11 output field for field; every user, account,
host and id is invented. SSH, git and the Secret Manager are faked.
"""

import contextlib
import logging
import textwrap
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gbserver.api import environment_queue as eq
from gbserver.api.utils import NO_ACCESSIBLE_SPACE
from gbserver.environment._skypilot_metadata import (
    comment_value,
    task_metadata_comment,
)
from gbserver.environment.skypilot import Skypilot
from gbserver.types.environmentconfig import EnvironmentConfig
from gbserver.utils.ssh_tunnel import SshTunnelError

SVC = "svc-builder"


def job_line(
    job_id, state, *, name="gb-x", comment="(null)", gres="N/A", nodes="1", cpus="8"
):
    start = "N/A" if state == "PENDING" else "2026-11-10T09:00:05"
    nodelist = "" if state == "PENDING" else "n01"
    reason = "QOSGrpGRES" if state == "PENDING" else "None"
    return "|".join(
        [
            job_id,
            SVC,
            "grp_demo",
            "gpu",
            "normal",
            state,
            reason,
            "2026-11-10T09:00:00",
            start,
            "1-00:00:00",
            "0:42",
            nodes,
            cpus,
            "64G",
            nodelist,
            gres,
            name,
            comment,
        ]
    )


def tag(**over):
    meta = {
        "build_id": "b-1",
        "gb_env": "DEV",
        "space_name": "public",
        "username": "alice",
        "build_config_name": "sft run",
        "target_name": "train",
        "targetstep_uri": "space://steps/train",
        "targetrun_id": "t-1",
        "targetsteprun_id": "s-1",
    }
    meta.update(over)
    return task_metadata_comment(meta)


ALICE = tag()
BOB = tag(
    build_id="b-2",
    gb_env="STAGING",
    space_name="my space",
    username="bob",
    build_config_name="eval",
    target_name="bench",
)
JOBS = "\n".join(
    [
        job_line(
            "100",
            "RUNNING",
            name="gb-sft-run-train-1a2b",
            comment=ALICE,
            gres="gres/gpu:h200:4",
            nodes="2",
        ),
        job_line(
            "101", "PENDING", name="gb-eval-bench-3c4d", comment=BOB, gres="gres/gpu:8"
        ),
        job_line(
            "102", "RUNNING", name="bash", comment="(null)"
        ),  # service-account job without a tag
    ]
)
LOAD = "\n".join(
    [
        "RUNNING|2|gres/gpu:h200:4",
        "PENDING|1|gres/gpu:8",
        "RUNNING|1|N/A",
        "RUNNING|4|gres:gpu:8",
        "PENDING|1|N/A",
    ]
)


# ------------------------------------------------------------------ parsing


class TestParseGpuGres:
    @pytest.mark.parametrize(
        "gres, expected",
        [
            ("gres/gpu:8", (8, None, True)),
            ("gres:gpu:8", (8, None, True)),
            ("gres/gpu:h200:4", (4, "h200", True)),
            ("gpu:h100:1", (1, "h100", True)),
            (None, (None, None, True)),
            ("gres/foo:1", (None, None, True)),
            ("gres/gpu:h200", (None, None, False)),
        ],
    )
    def test_cases(self, gres, expected):
        assert eq.parse_gpu_gres(gres) == expected


class TestParseGbComment:
    def test_reads_the_writer(self):
        t = eq.parse_gb_comment(ALICE)
        assert (t.build_id, t.gb_env, t.space, t.username, t.build_name) == (
            "b-1",
            "DEV",
            "public",
            "alice",
            "sft_run",
        )

    def test_every_writer_key_has_a_model_field(self):
        assert eq._GB_COMMENT_KEYS <= set(eq.GbJobTag.model_fields)

    def test_repeated_key_is_rejected(self):
        assert eq.parse_gb_comment("build_id=a;space=x;space=y") is None

    @pytest.mark.parametrize(
        "comment", [None, "", "free text", "build_name=x;target=y"]
    )
    def test_not_ours(self, comment):
        assert eq.parse_gb_comment(comment) is None


class TestParseJobLine:
    def test_gpus_are_per_node_times_nodes(self):
        j = eq.parse_job_line(
            job_line("100", "RUNNING", gres="gres/gpu:h200:4", nodes="2")
        )
        assert (j.gpus_per_node, j.gpus, j.gpu_model) == (4, 8, "h200")

    def test_pipe_in_comment_stays_in_comment(self):
        j = eq.parse_job_line(job_line("100", "RUNNING", comment="a|b|c"))
        assert j.comment == "a|b|c" and j.name == "gb-x" and j.state == "RUNNING"

    def test_pending_array(self):
        j = eq.parse_job_line(job_line("103_[0-9%4]", "PENDING"))
        assert (j.array_job_id, j.start_time, j.nodelist, j.reason) == (
            "103",
            None,
            None,
            "QOSGrpGRES",
        )

    @pytest.mark.parametrize(
        "line",
        [
            "",
            "too|few",
            job_line("100", "COMPLETED"),
            job_line("100", "RUNNING", nodes="x"),
            job_line("abc", "RUNNING"),
            job_line("100", "RUNNING", gres="gres/gpu:h200"),
        ],
    )
    def test_rejects(self, line):
        assert eq.parse_job_line(line) is None

    def test_unparsed_lines_are_counted(self):
        jobs, bad = eq.parse_jobs(JOBS + "\ngarbage")
        assert len(jobs) == 3 and bad == 1


AWKWARD = [
    "a|b",
    "x;target=evil",
    "k=v",
    'it\'s "quoted"',
    "two\nlines",
    "tab\there",
    "nb sp",
    "em space",
    "ünï cödé",
]


@pytest.mark.parametrize("raw", AWKWARD)
def test_round_trip_through_the_real_writers(raw):
    """Names gbserver writes come back intact, whatever the user put in them."""
    comment = task_metadata_comment(
        {
            "build_id": "b-9",
            "gb_env": "DEV",
            "space_name": raw,
            "username": "alice",
            "build_config_name": raw,
            "target_name": raw,
            "targetstep_uri": "space://steps/x",
            "targetrun_id": "t-9",
            "targetsteprun_id": "s-9",
        }
    )
    name = Skypilot._cluster_name_for(
        "0123456789abcdef", target_name=raw, build_id="b-9", build_config_name=raw
    )
    assert "\n" not in comment and "\n" not in name
    job = eq.parse_job_line(job_line("700", "RUNNING", name=name, comment=comment))
    assert job is not None and job.state == "RUNNING" and job.user == SVC
    assert job.name == name and job.comment == comment
    assert (job.gb.build_name, job.gb.target, job.gb.space) == (comment_value(raw),) * 3
    assert (job.gb.build_id, job.gb.step_id) == ("b-9", "s-9")


def test_load_counts_cover_every_user():
    counts, bad = eq.parse_load(LOAD + "\nCOMPLETED|1|N/A")
    assert (counts.running, counts.pending, counts.running_gpus, bad) == (3, 2, 40, 1)


def test_jobs_command_quotes_the_user():
    assert eq.squeue_jobs_command(SVC).startswith(
        f"squeue -h -a -t RUNNING,PENDING -u {SVC} -o '%i|"
    )
    assert "-u 'svc builder;id'" in eq.squeue_jobs_command("svc builder;id")
    assert eq.squeue_jobs_command(SVC).endswith("|%j|%k'")
    assert "%j" not in eq.SQUEUE_LOAD and "%k" not in eq.SQUEUE_LOAD


# ------------------------------------------------------------ registry/asset


class TestRegistry:
    def test_parses_config(self):
        reg = eq._load_registry(
            '{"cluster-a": {"space": "ops", "environment": "skypilot/slurm/cluster-a"}}'
        )
        assert reg == {
            "cluster-a": eq.QueueEnvironment(
                space="ops", env_path="skypilot/slurm/cluster-a"
            )
        }

    @pytest.mark.parametrize("raw", ["", "not json", '{"a": {"space": "x"}}', "[]"])
    def test_bad_or_empty_config_is_empty(self, raw):
        assert eq._load_registry(raw) == {}

    def test_unknown_environment_is_404(self, monkeypatch):
        monkeypatch.setattr(eq, "QUEUE_ENVIRONMENT_REGISTRY", {})
        with pytest.raises(HTTPException) as exc:
            eq.lookup_queue_environment("nowhere")
        assert exc.value.status_code == 404


ASSET = textwrap.dedent(f"""\
    name: sky-slurm-cluster-a
    type: Skypilot
    subtype: slurm
    config:
      cluster: cluster-a
      cluster_ssh_configs:
        slurm:
          - Host: other
            HostName: login.other.example
            User: nobody
            IdentityKey: OTHER_KEY
          - Host: cluster-a
            HostName:
              - login1.cluster-a.example
              - login2.cluster-a.example
            User: {SVC}
            IdentityKey: CLUSTER_A_SSH_KEY
    """)


@pytest.fixture
def asset_config(tmp_path):
    p = tmp_path / "environment.yaml"
    p.write_text(ASSET)
    return EnvironmentConfig.from_yaml(p)


def _cfg(**over):
    base = {"name": "x", "type": "Skypilot", "subtype": "slurm", "config": {}}
    base.update(over)
    return EnvironmentConfig(**base)


class TestConnectionFromEnvConfig:
    def test_reads_the_host_named_by_cluster(self, asset_config):
        conn = eq.connection_from_env_config("cluster-a", asset_config)
        assert conn == eq.QueueConnection(
            ("login1.cluster-a.example", "login2.cluster-a.example"),
            SVC,
            "CLUSTER_A_SSH_KEY",
        )

    def test_single_host_without_cluster(self):
        cfg = _cfg(
            config={
                "cluster_ssh_configs": {
                    "slurm": [
                        {
                            "Host": "h",
                            "HostName": "login.example",
                            "User": "u",
                            "IdentityKey": "K",
                        }
                    ]
                }
            }
        )
        assert eq.connection_from_env_config("x", cfg).login_nodes == ("login.example",)

    @pytest.mark.parametrize(
        "over, reason",
        [
            ({"subtype": "lsf"}, "not slurm"),
            ({"config": {}}, "no matching"),
            (
                {
                    "config": {
                        "cluster_ssh_configs": {
                            "slurm": [
                                {
                                    "Host": "h",
                                    "HostName": [],
                                    "User": "u",
                                    "IdentityKey": "K",
                                }
                            ]
                        }
                    }
                },
                "missing HostName",
            ),
            (
                {
                    "config": {
                        "cluster_ssh_configs": {
                            "slurm": [
                                {
                                    "Host": "h",
                                    "HostName": "n",
                                    "User": "u",
                                    "IdentityFile": "~/.ssh/k",
                                }
                            ]
                        }
                    }
                },
                "missing IdentityKey",
            ),
        ],
    )
    def test_incomplete_is_503_with_the_reason_logged(self, over, reason, caplog):
        with caplog.at_level(logging.WARNING), pytest.raises(HTTPException) as exc:
            eq.connection_from_env_config("x", _cfg(**over))
        assert exc.value.status_code == 503 and reason not in exc.value.detail
        assert reason in caplog.text


class TestResolveConnection:
    record = eq.QueueEnvironment(space="ops", env_path="skypilot/slurm/cluster-a")

    @pytest.fixture(autouse=True)
    def _fakes(self, monkeypatch, asset_config):
        monkeypatch.setattr(eq, "_connections", {})
        space = SimpleNamespace(git_repo_uri="file:///spaces/ops")
        monkeypatch.setattr(
            eq,
            "get_admin_storage",
            lambda: SimpleNamespace(
                space_storage=SimpleNamespace(
                    get_by_name=lambda n: space if n == "ops" else None
                )
            ),
        )
        monkeypatch.setattr(
            eq,
            "get_space_environment_uri",
            lambda repo, path: f"{repo}/environments/{path}",
        )
        self.state = SimpleNamespace(loads=0, fail=False, threads=[])

        def load(_uri):
            self.state.loads += 1
            if self.state.fail:
                raise RuntimeError("git clone of file:///spaces/ops failed")
            return asset_config, None

        monkeypatch.setattr(eq.Environment, "load_environment_config", load)
        real_to_thread = eq.asyncio.to_thread

        async def spy(fn, *a, **k):
            self.state.threads.append(fn.__name__)
            return await real_to_thread(fn, *a, **k)

        monkeypatch.setattr(eq.asyncio, "to_thread", spy)
        self.key = AsyncMock(return_value="KEY-MATERIAL")
        monkeypatch.setattr(eq, "fetch_space_ssh_key", self.key)

    @pytest.mark.asyncio
    async def test_blocking_work_runs_off_the_loop_and_is_cached(self):
        conn, key = await eq.resolve_connection("cluster-a", self.record)
        await eq.resolve_connection("cluster-a", self.record)
        assert self.state.threads == ["_load_asset"] and self.state.loads == 1
        assert conn.username == SVC and key == "KEY-MATERIAL"
        self.key.assert_awaited_once_with("ops", "CLUSTER_A_SSH_KEY")

    @pytest.mark.asyncio
    async def test_unknown_space_is_503(self):
        with pytest.raises(HTTPException) as exc:
            await eq.resolve_connection(
                "cluster-a", eq.QueueEnvironment(space="nope", env_path="p")
            )
        assert exc.value.status_code == 503

    @pytest.mark.asyncio
    async def test_load_failure_keeps_uri_and_error_out_of_the_body(self):
        self.state.fail = True
        with pytest.raises(HTTPException) as exc:
            await eq.resolve_connection("cluster-a", self.record)
        assert exc.value.status_code == 503
        assert "file://" not in exc.value.detail and "clone" not in exc.value.detail

    @pytest.mark.asyncio
    async def test_missing_key_secret_is_503_not_404(self):
        self.key.side_effect = HTTPException(404, "ssh key secret not found")
        with pytest.raises(HTTPException) as exc:
            await eq.resolve_connection("cluster-a", self.record)
        assert exc.value.status_code == 503


# -------------------------------------------------------------------- route

CALLER = {"login": "alice"}


@pytest.fixture
def client():
    app = eq.queue_api

    @app.middleware("http")
    async def _as_caller(request, call_next):
        request.state.data = {"user": SimpleNamespace(login=CALLER["login"])}
        return await call_next(request)

    try:
        yield TestClient(app)
    finally:
        app.user_middleware.pop(0)
        app.middleware_stack = None


class FakeTunnel:
    host = "login1.cluster-a.example"

    def __init__(self, rc=0, stderr="", run_error=None):
        self.outputs = {eq.squeue_jobs_command(SVC): JOBS, eq.SQUEUE_LOAD: LOAD}
        self.rc, self.stderr, self.run_error, self.commands = rc, stderr, run_error, []

    async def run_remote(self, command, raise_on_error=False):
        self.commands.append(command)
        if self.run_error:
            raise self.run_error
        return self.rc, self.outputs.get(command, ""), self.stderr


@pytest.fixture
def queue(monkeypatch):
    """One configured environment backed by a fake tunnel; caller sees `public`."""
    monkeypatch.setattr(
        eq,
        "QUEUE_ENVIRONMENT_REGISTRY",
        {"cluster-a": eq.QueueEnvironment("ops", "skypilot/slurm/cluster-a")},
    )
    monkeypatch.setattr(eq, "_snapshots", {})
    monkeypatch.setattr(eq, "_locks", {})
    monkeypatch.setattr(eq, "_connections", {})
    monkeypatch.setattr(
        eq,
        "resolve_connection",
        AsyncMock(
            return_value=(
                eq.QueueConnection(("login1.cluster-a.example",), SVC, "K"),
                "KEY",
            )
        ),
    )
    state = SimpleNamespace(
        tunnel=FakeTunnel(), opens=[], open_error=None, scope=["public"]
    )
    monkeypatch.setattr(eq, "scope_space_name_filter", lambda request: state.scope)

    @contextlib.asynccontextmanager
    async def fake_open(*args, **kwargs):
        state.opens.append(kwargs)
        if state.open_error:
            raise state.open_error
        yield state.tunnel

    monkeypatch.setattr(eq, "open_login_tunnel", fake_open)
    return state


def jobs_of(client, query=""):
    r = client.get(f"/cluster-a/queue{query}")
    assert r.status_code == 200, r.text
    return {j["job_id"]: j for j in r.json()["jobs"]}


def test_lists_the_service_accounts_jobs_and_whole_queue_counts(queue, client):
    body = client.get("/cluster-a/queue").json()
    assert queue.tunnel.commands == [eq.squeue_jobs_command(SVC), eq.SQUEUE_LOAD]
    assert queue.opens[0]["key_material"] == "KEY"
    assert body["counts"] == {"running": 3, "pending": 2, "running_gpus": 40}
    assert sorted(j["job_id"] for j in body["jobs"]) == ["100", "101", "102"]


def test_job_in_one_of_my_spaces_is_full(queue, client):
    job = jobs_of(client)["100"]
    assert not job["redacted"] and job["mine"]
    assert job["name"] == "gb-sft-run-train-1a2b" and job["gb"]["step_id"] == "s-1"


def test_job_outside_my_spaces_has_scheduler_fields_only(queue, client):
    job = jobs_of(client)["101"]
    assert job["redacted"] and not job["mine"]
    assert (job["name"], job["comment"], job["gb"]) == (None, None, None)
    assert (job["state"], job["partition"], job["gpus"], job["reason"]) == (
        "PENDING",
        "gpu",
        8,
        "QOSGrpGRES",
    )


def test_untagged_job_fails_closed(queue, client):
    job = jobs_of(client)["102"]
    assert job["redacted"] and job["name"] is None and job["gb"] is None


def test_space_is_compared_in_the_tags_spelling(queue, client):
    queue.scope = ["my space"]  # the tag stores "my_space"
    jobs = jobs_of(client)
    assert not jobs["101"]["redacted"] and jobs["100"]["redacted"]


def test_no_accessible_space_redacts_everything(queue, client):
    queue.scope = NO_ACCESSIBLE_SPACE
    assert all(j["redacted"] for j in jobs_of(client).values())


def test_super_admin_sees_every_tagged_job(queue, client):
    queue.scope = ""
    jobs = jobs_of(client)
    assert not jobs["100"]["redacted"] and not jobs["101"]["redacted"]
    assert jobs["102"]["redacted"]  # still no readable tag


def test_filters_run_on_the_callers_view(queue, client):
    assert list(jobs_of(client, "?mine=true")) == ["100"]
    assert jobs_of(client, "?username=bob") == {}  # hidden from this caller
    queue.scope = ["public", "my space"]
    assert list(jobs_of(client, "?space=my space&state=pending")) == ["101"]
    assert list(jobs_of(client, "?gb_env=STAGING")) == ["101"]


def test_cache_shares_one_session(queue, client):
    client.get("/cluster-a/queue")
    client.get("/cluster-a/queue?state=running")
    assert len(queue.opens) == 1


def test_squeue_failure_is_503_cached_and_generic(queue, client):
    queue.tunnel = FakeTunnel(rc=1, stderr="slurm_load_jobs error: timed out")
    r1, r2 = client.get("/cluster-a/queue"), client.get("/cluster-a/queue")
    assert (r1.status_code, r2.status_code) == (503, 503) and len(queue.opens) == 1
    assert "slurm_load_jobs" not in r1.json()["detail"]


def test_failed_tunnel_open_clears_the_connection_cache(queue, client):
    eq._connections["cluster-a"] = "stale"
    queue.open_error = HTTPException(503, "no reachable login nodes")
    assert client.get("/cluster-a/queue").status_code == 503
    assert "cluster-a" not in eq._connections


def test_ssh_error_mid_session_clears_the_connection_cache(queue, client):
    eq._connections["cluster-a"] = "stale"
    queue.tunnel = FakeTunnel(run_error=SshTunnelError("session dropped"))
    r = client.get("/cluster-a/queue")
    assert r.status_code == 503 and "dropped" not in r.json()["detail"]
    assert "cluster-a" not in eq._connections


def test_bad_state_is_422(queue, client):
    assert client.get("/cluster-a/queue?state=done").status_code == 422


def test_unknown_environment_is_404(queue, client):
    assert client.get("/nowhere/queue").status_code == 404
