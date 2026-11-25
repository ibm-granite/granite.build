#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Tests for ``gbserver.api.environment_queue`` and ``gbserver.api.slurm_output``.

squeue lines follow Slurm 25.11 output field for field; every user, account,
host and id is invented. SSH, git and the Secret Manager are faked.
"""

import contextlib
import logging
import textwrap
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import unquote

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gbserver.api import environment_queue as eq
from gbserver.api import slurm_output as so
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
        "RUNNING|2|gres/gpu:h200:4|gpu",
        "PENDING|1|gres/gpu:8|gpu,gpu-high",
        "RUNNING|1|N/A|cpu",
        "RUNNING|4|gres:gpu:8|gpu-high",
        "PENDING|1|N/A|cpu",
    ]
)


def sinfo_line(host, state, gres, used, partition):
    return f"{host:<128}{state:<64}{gres:<256}{used:<256}{partition:<64}"


SINFO = "\n".join(
    [
        sinfo_line("n01", "mixed", "gpu:h200:8(S:0-1)", "gpu:h200:3(IDX:0-2)", "gpu*"),
        sinfo_line(
            "n01", "mixed", "gpu:h200:8(S:0-1)", "gpu:h200:3(IDX:0-2)", "gpu-high"
        ),
        sinfo_line("n02", "idle", "gpu:h200:8(S:0-1)", "gpu:h200:0", "gpu*"),
        sinfo_line(
            "n03", "allocated", "gpu:h100:8(S:0-1)", "gpu:h100:8(IDX:0-7)", "gpu*"
        ),
        sinfo_line("n04", "drained", "gpu:h100:8(S:0-1)", "gpu:h100:0", "gpu*"),
        sinfo_line("n05", "idle~", "gpu:h100:8(S:0-1)", "gpu:h100:0", "gpu*"),
        sinfo_line("n06", "down*", "gpu:8(S:0-1)", "gpu:0", "gpu*"),
        sinfo_line("c01", "idle", "(null)", "gpu:0", "cpu"),
    ]
)


def sacct_line(
    job_id,
    state,
    *,
    elapsed="3600",
    tres="billing=8,cpu=8,gres/gpu:h200=4,gres/gpu=4,mem=240G,node=1",
    comment="",
):
    return "|".join(
        [
            job_id,
            "gb-sft-run-train-1a2b",
            "gpu",
            "normal",
            state,
            "2026-11-10T09:00:00",
            "2026-11-10T09:00:05",
            "2026-11-10T10:00:05",
            elapsed,
            "1",
            tres,
            "0:0",
            comment,
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
        assert so.parse_gpu_gres(gres) == expected


class TestParseGbComment:
    def test_reads_the_writer(self):
        t = so.parse_gb_comment(ALICE)
        assert (t.build_id, t.gb_env, t.space, t.username, t.build_name) == (
            "b-1",
            "DEV",
            "public",
            "alice",
            "sft_run",
        )

    def test_every_writer_key_has_a_model_field(self):
        assert so._GB_COMMENT_KEYS <= set(so.GbJobTag.model_fields)

    def test_repeated_key_is_rejected(self):
        assert so.parse_gb_comment("build_id=a;space=x;space=y") is None

    @pytest.mark.parametrize(
        "comment", [None, "", "free text", "build_name=x;target=y"]
    )
    def test_not_ours(self, comment):
        assert so.parse_gb_comment(comment) is None


class TestParseJobLine:
    def test_gpus_are_per_node_times_nodes(self):
        j = so.parse_job_line(
            job_line("100", "RUNNING", gres="gres/gpu:h200:4", nodes="2")
        )
        assert (j.gpus_per_node, j.gpus, j.gpu_model) == (4, 8, "h200")

    def test_pipe_in_comment_stays_in_comment(self):
        j = so.parse_job_line(job_line("100", "RUNNING", comment="a|b|c"))
        assert j.comment == "a|b|c" and j.name == "gb-x" and j.state == "RUNNING"

    def test_pending_array(self):
        j = so.parse_job_line(job_line("103_[0-9%4]", "PENDING"))
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
        assert so.parse_job_line(line) is None

    def test_unparsed_lines_are_counted(self):
        jobs, bad = so.parse_jobs(JOBS + "\ngarbage")
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
    job = so.parse_job_line(job_line("700", "RUNNING", name=name, comment=comment))
    assert job is not None and job.state == "RUNNING" and job.user == SVC
    assert job.name == name and job.comment == comment
    assert (job.gb.build_name, job.gb.target, job.gb.space) == (comment_value(raw),) * 3
    assert (job.gb.build_id, job.gb.step_id) == ("b-9", "s-9")


def test_load_counts_cover_every_user():
    counts, bad = so.parse_load(LOAD + "\nCOMPLETED|1|N/A|gpu\nRUNNING|1|N/A")
    assert (counts.running, counts.pending, counts.running_gpus, bad) == (3, 2, 40, 2)
    assert counts.running_gpus_by_partition == {"gpu": 8, "cpu": 0, "gpu-high": 32}


def test_jobs_command_quotes_the_user():
    assert so.squeue_jobs_command(SVC).startswith(
        f"squeue -h -a -t RUNNING,PENDING -u {SVC} -o '%i|"
    )
    assert "-u 'svc builder;id'" in so.squeue_jobs_command("svc builder;id")
    assert so.squeue_jobs_command(SVC).endswith("|%j|%k'")
    assert "%j" not in so.SQUEUE_LOAD and "%k" not in so.SQUEUE_LOAD


# ------------------------------------------------------------- path / asset


class TestParseQueueEnvironment:
    def test_space_and_asset_path(self):
        rec = eq.parse_queue_environment("ops", "skypilot/slurm/cluster-a/")
        assert rec == eq.QueueEnvironment("ops", "skypilot/slurm/cluster-a")
        assert rec.key == "ops/skypilot/slurm/cluster-a"

    @pytest.mark.parametrize(
        "space, path",
        [
            ("ops", ""),
            ("ops", "a//b"),
            ("ops", ".."),
            ("ops", "a/../b"),
            ("ops", "./a"),
            ("ops", ".hidden"),
            ("ops", "a b"),
            ("ops", "a" * 300),
            ("", "a"),
        ],
    )
    def test_malformed_is_404(self, space, path):
        with pytest.raises(HTTPException) as exc:
            eq.parse_queue_environment(space, path)
        assert exc.value.status_code == 404


class TestSpaceEnvironmentUri:
    def test_db_space_uses_a_subdirectory(self, monkeypatch):
        monkeypatch.setattr(
            eq.db_space, "get_space_config_uri", lambda row: "db://spaces/ops"
        )
        assert (
            eq.space_environment_uri(object(), "skypilot/slurm/cluster-a")
            == "db://spaces/ops#subdirectory=environments/skypilot/slurm/cluster-a"
        )

    def test_git_space_appends_to_its_subdirectory(self, monkeypatch):
        monkeypatch.setattr(
            eq.db_space,
            "get_space_config_uri",
            lambda row: "git+ssh://git@git.example/o/ops.git@cfg#subdirectory=defs",
        )
        uri = eq.space_environment_uri(object(), "skypilot/slurm/cluster-a")
        assert uri.startswith("git+ssh://git@git.example/o/ops.git@cfg")
        assert unquote(uri).endswith("defs/environments/skypilot/slurm/cluster-a")


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


def test_non_slurm_asset_is_404():
    with pytest.raises(HTTPException) as exc:
        eq.connection_from_env_config("x", _cfg(subtype="lsf"))
    assert exc.value.status_code == 404 and exc.value.detail == "environment not found"


class TestResolveConnection:
    record = eq.QueueEnvironment(space="ops", env_path="skypilot/slurm/cluster-a")

    @pytest.fixture(autouse=True)
    def _fakes(self, monkeypatch, asset_config):
        monkeypatch.setattr(eq, "_connections", {})
        space = SimpleNamespace(name="ops")
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
            "space_environment_uri",
            lambda row, path: f"file:///spaces/{row.name}/environments/{path}",
        )
        self.state = SimpleNamespace(loads=[], fail=None, threads=[])

        def load(uri):
            self.state.loads.append(uri)
            if self.state.fail:
                raise self.state.fail
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
        assert self.state.threads == ["_load_asset"]
        assert self.state.loads == [
            "file:///spaces/ops/environments/skypilot/slurm/cluster-a"
        ]
        assert conn.username == SVC and key == "KEY-MATERIAL"
        self.key.assert_awaited_once_with("ops", "CLUSTER_A_SSH_KEY")

    @pytest.mark.asyncio
    async def test_unknown_space_is_404(self):
        with pytest.raises(HTTPException) as exc:
            await eq.resolve_connection(
                "cluster-a", eq.QueueEnvironment(space="nope", env_path="p")
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_path_without_an_environment_yaml_is_404(self):
        self.state.fail = eq.UnknownEnvironmentType("environment.yaml not found")
        with pytest.raises(HTTPException) as exc:
            await eq.resolve_connection("cluster-a", self.record)
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_load_failure_keeps_uri_and_error_out_of_the_body(self):
        self.state.fail = RuntimeError("git clone of file:///spaces/ops failed")
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
        self.outputs = {
            so.squeue_jobs_command(SVC): JOBS,
            so.SQUEUE_LOAD: LOAD,
            so.SINFO_NODES: SINFO,
            **{so.sacct_history_command(SVC, d): SACCT for d in so.HISTORY_DAYS},
        }
        self.rc, self.stderr, self.run_error, self.commands = rc, stderr, run_error, []

    async def run_remote(self, command, raise_on_error=False):
        self.commands.append(command)
        if self.run_error:
            raise self.run_error
        return self.rc, self.outputs.get(command, ""), self.stderr


@pytest.fixture
def queue(monkeypatch):
    """One environment backed by a fake tunnel; caller has `ops` and `public`."""
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
        tunnel=FakeTunnel(), opens=[], open_error=None, scope=["ops", "public"]
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


KEY = "ops/skypilot/slurm/cluster-a"
URL = f"/{KEY}/queue"


def jobs_of(client, query=""):
    r = client.get(f"{URL}{query}")
    assert r.status_code == 200, r.text
    return {j["job_id"]: j for j in r.json()["jobs"]}


def test_lists_the_service_accounts_jobs_and_whole_queue_counts(queue, client):
    body = client.get(f"/{KEY}/queue").json()
    assert queue.tunnel.commands == [so.squeue_jobs_command(SVC), so.SQUEUE_LOAD]
    assert queue.opens[0]["key_material"] == "KEY"
    assert body["counts"] == {
        "running": 3,
        "pending": 2,
        "running_gpus": 40,
        "running_gpus_by_partition": {"gpu": 8, "cpu": 0, "gpu-high": 32},
    }
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
    queue.scope = ["ops", "my space"]  # the tag stores "my_space"
    jobs = jobs_of(client)
    assert not jobs["101"]["redacted"] and jobs["100"]["redacted"]


def test_only_the_environments_space_redacts_everything_else(queue, client):
    queue.scope = ["ops"]
    assert all(j["redacted"] for j in jobs_of(client).values())


def test_super_admin_sees_every_tagged_job(queue, client):
    queue.scope = ""
    jobs = jobs_of(client)
    assert not jobs["100"]["redacted"] and not jobs["101"]["redacted"]
    assert jobs["102"]["redacted"]  # still no readable tag


def test_filters_run_on_the_callers_view(queue, client):
    assert list(jobs_of(client, "?mine=true")) == ["100"]
    assert jobs_of(client, "?username=bob") == {}  # hidden from this caller
    queue.scope = ["ops", "public", "my space"]
    assert list(jobs_of(client, "?space=my space&state=pending")) == ["101"]
    assert list(jobs_of(client, "?gb_env=STAGING")) == ["101"]


def test_cache_shares_one_session(queue, client):
    client.get(f"/{KEY}/queue")
    client.get(f"/{KEY}/queue?state=running")
    assert len(queue.opens) == 1


def test_squeue_failure_is_503_cached_and_generic(queue, client):
    queue.tunnel = FakeTunnel(rc=1, stderr="slurm_load_jobs error: timed out")
    r1, r2 = client.get(f"/{KEY}/queue"), client.get(f"/{KEY}/queue")
    assert (r1.status_code, r2.status_code) == (503, 503) and len(queue.opens) == 1
    assert "slurm_load_jobs" not in r1.json()["detail"]


def test_failed_tunnel_open_clears_the_connection_cache(queue, client):
    eq._connections[KEY] = "stale"
    queue.open_error = HTTPException(503, "no reachable login nodes")
    assert client.get(f"/{KEY}/queue").status_code == 503
    assert KEY not in eq._connections


def test_ssh_error_mid_session_clears_the_connection_cache(queue, client):
    eq._connections[KEY] = "stale"
    queue.tunnel = FakeTunnel(run_error=SshTunnelError("session dropped"))
    r = client.get(f"/{KEY}/queue")
    assert r.status_code == 503 and "dropped" not in r.json()["detail"]
    assert KEY not in eq._connections


def test_bad_state_is_422(queue, client):
    assert client.get(f"/{KEY}/queue?state=done").status_code == 422


def test_response_names_the_environment(queue, client):
    assert client.get(URL).json()["environment"] == KEY


@pytest.mark.parametrize("scope", [["public"], NO_ACCESSIBLE_SPACE])
def test_space_without_access_is_404_before_any_ssh(queue, client, scope):
    queue.scope = scope
    r = client.get(URL)
    assert r.status_code == 404 and r.json()["detail"] == "environment not found"
    assert queue.opens == [] and not eq.resolve_connection.await_count


def test_malformed_path_is_404(queue, client):
    assert client.get("/ops/a/%2e%2e/b/queue").status_code == 404
    assert client.get("/ops/.hidden/queue").status_code == 404
    assert client.get("/ops/queue").status_code == 404


def test_caches_are_bounded(monkeypatch):
    monkeypatch.setattr(eq, "_MAX_CACHED_ENVIRONMENTS", 3)
    cache = {"stale": 0, "a": 1, "b": 1}
    eq._bound(cache, lambda v: v == 0)
    assert list(cache) == ["a", "b"]
    cache["c"] = 1
    eq._bound(cache, lambda v: v == 0)
    assert list(cache) == ["b", "c"]


@pytest.mark.parametrize("kind", ["capacity", "history?days=7"])
def test_capacity_and_history_need_the_space_too(queue, client, kind):
    queue.scope = ["public"]
    assert client.get(f"/{KEY}/{kind}").status_code == 404
    assert queue.opens == []


# ----------------------------------------------------------------- capacity


@pytest.mark.parametrize(
    "state, usable",
    [
        ("idle", True),
        ("mixed", True),
        ("allocated", True),
        ("planned", True),
        ("mixed-", True),
        ("drained", False),
        ("draining", False),
        ("down*", False),
        ("idle~", False),
        ("idle#", False),
        ("maint", False),
        ("reserved", False),
        ("rebooting", False),
        ("powered_down", False),
        ("unknown", False),
        ("mixed$", False),
    ],
)
def test_node_is_usable(state, usable):
    assert so.node_is_usable(state) is usable


def test_sinfo_nodes_are_deduplicated_across_partitions():
    nodes, bad = so.parse_sinfo_nodes(SINFO + "\ntoo few fields")
    assert bad == 1 and len(nodes) == 7
    assert nodes["n01"].partitions == {
        "gpu",
        "gpu-high",
    }  # default-partition "*" stripped
    assert (nodes["n01"].gpus, nodes["n01"].gpus_used, nodes["n01"].model) == (
        8,
        3,
        "h200",
    )
    assert nodes["c01"].gpus == 0


def test_capacity_counts_free_gpus_on_usable_nodes_only():
    nodes, _ = so.parse_sinfo_nodes(SINFO)
    models, parts = so.summarize_capacity(nodes)
    by = {m.gpu_model: m for m in models}
    assert (by["h200"].usable_nodes, by["h200"].gpus_used, by["h200"].gpus_free) == (
        2,
        3,
        13,
    )
    assert (by["h100"].nodes, by["h100"].usable_nodes, by["h100"].gpus_free) == (
        3,
        1,
        0,
    )
    assert (by[None].nodes, by[None].usable_nodes) == (1, 0)  # unnamed model, down
    p = {x.partition: x for x in parts}
    assert (p["gpu"].nodes, p["gpu"].gpus_free) == (6, 13) and p[
        "gpu-high"
    ].gpus_free == 5


def test_capacity_route(queue, client):
    r = client.get(f"/{KEY}/capacity")
    assert r.status_code == 200, r.text
    body = r.json()
    assert queue.tunnel.commands == [so.SINFO_NODES]
    assert {m["gpu_model"]: m["gpus_free"] for m in body["models"]} == {
        "h100": 0,
        "h200": 13,
        None: 0,
    }
    assert "n01" not in r.text  # aggregates only


# ------------------------------------------------------------------ history

SACCT = "\n".join(
    [
        sacct_line("300", "COMPLETED", comment=ALICE),
        sacct_line(
            "301", "FAILED", elapsed="1800", tres="cpu=8,gres/gpu=8,node=1", comment=BOB
        ),
        sacct_line("302", "CANCELLED by 4242", comment=ALICE + "|tail"),
        sacct_line(
            "303", "RUNNING", comment=ALICE
        ),  # still running: skipped, not unparsed
        sacct_line("304", "PREEMPTED", comment=""),
    ]
)


def test_history_command_is_fixed_per_window():
    cmd = so.sacct_history_command(SVC, 7)
    assert cmd.startswith(
        f"sacct -X -n -P -u {SVC} -S now-7days -E now --format=JobID,"
    )
    assert cmd.endswith(",ExitCode,Comment")
    with pytest.raises(AssertionError):
        so.sacct_history_command(SVC, 3)


@pytest.mark.parametrize(
    "tres, expected",
    [
        ("gres/gpu:h200=4,gres/gpu=4", (4, "h200")),
        ("cpu=8,gres/gpu=8", (8, None)),
        ("gres/gpu:h100=16", (16, "h100")),
        ("cpu=8,mem=4G", (None, None)),
        (None, (None, None)),
    ],
)
def test_alloc_gpus(tres, expected):
    assert so.parse_alloc_gpus(tres) == expected


def test_history_parsing():
    jobs, bad = so.parse_history(SACCT + "\ngarbage")
    by = {j.job_id: j for j in jobs}
    assert sorted(by) == ["300", "301", "302", "304"] and bad == 1
    assert (by["300"].gpus, by["300"].gpu_model, by["300"].gpu_hours) == (
        4,
        "h200",
        4.0,
    )
    assert by["301"].gpu_hours == 4.0
    assert by["302"].state == "CANCELLED" and by["302"].comment.endswith("|tail")
    assert by["304"].gb is None


def test_history_route_visibility_and_totals(queue, client):
    r = client.get(f"/{KEY}/history?days=7")
    assert r.status_code == 200, r.text
    body = r.json()
    jobs = {j["job_id"]: j for j in body["jobs"]}
    assert body["comments_recorded"] is True
    assert not jobs["300"]["redacted"] and jobs["300"]["mine"]
    assert jobs["301"]["redacted"] and jobs["301"]["gb"] is None  # bob's space
    assert jobs["304"]["redacted"]  # no tag: fails closed
    assert body["totals"]["jobs"] == 4
    mine = client.get(f"/{KEY}/history?days=7&mine=true").json()
    assert sorted(j["job_id"] for j in mine["jobs"]) == ["300", "302"]
    assert mine["totals"] == {
        "jobs": 2,
        "gpu_hours": 8.0,
        "gpu_hours_by_model": {"h200": 8.0},
    }


def test_history_windows_are_cached_separately(queue, client):
    client.get(f"/{KEY}/history?days=7")
    client.get(f"/{KEY}/history?days=7")
    client.get(f"/{KEY}/history?days=30")
    assert queue.tunnel.commands == [
        so.sacct_history_command(SVC, 7),
        so.sacct_history_command(SVC, 30),
    ]


def test_history_rejects_other_windows(queue, client):
    assert client.get(f"/{KEY}/history?days=3").status_code == 422


def test_no_comments_recorded_is_flagged(queue, client):
    queue.tunnel.outputs[so.sacct_history_command(SVC, 1)] = sacct_line(
        "400", "COMPLETED"
    )
    body = client.get(f"/{KEY}/history?days=1").json()
    assert body["comments_recorded"] is False and body["jobs"][0]["redacted"]


def test_kinds_share_one_connection_but_cache_separately(queue, client):
    client.get(f"/{KEY}/queue")
    client.get(f"/{KEY}/capacity")
    client.get(f"/{KEY}/queue")
    assert len(queue.opens) == 2
