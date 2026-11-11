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
import textwrap
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from gbserver.api import environment_queue as eq
from gbserver.environment._skypilot_metadata import task_metadata_comment
from gbserver.types.environmentconfig import EnvironmentConfig


def fixed(
    job_id,
    user,
    state,
    *,
    gres="N/A",
    nodes="1",
    cpus="8",
    part="gpu",
    reason="None",
    account="grp_demo",
    nodelist="n01",
):
    return "|".join(
        [
            job_id,
            user,
            account,
            part,
            "normal",
            state,
            reason,
            "2026-11-10T09:00:00",
            "N/A" if state == "PENDING" else "2026-11-10T09:00:05",
            "1-00:00:00",
            "0:42",
            nodes,
            cpus,
            "64G",
            "" if state == "PENDING" else nodelist,
            gres,
        ]
    )


ALICE_TAG = task_metadata_comment(
    {
        "build_id": "b-alice",
        "gb_env": "DEV",
        "space_name": "public",
        "username": "alice",
        "build_config_name": "sft run",
        "target_name": "train",
        "targetstep_uri": "space://steps/train",
        "targetrun_id": "t-1",
        "targetsteprun_id": "s-1",
    }
)
BOB_TAG = task_metadata_comment(
    {
        "build_id": "b-bob",
        "gb_env": "STAGING",
        "space_name": "my space",
        "username": "bob",
        "build_config_name": "eval",
        "target_name": "bench",
        "targetrun_id": "t-2",
        "targetsteprun_id": "s-2",
    }
)

FIXED = "\n".join(
    [
        fixed("100", "svc-builder", "RUNNING", gres="gres/gpu:h200:4", nodes="2"),
        fixed("101", "svc-builder", "PENDING", gres="gres/gpu:8", reason="QOSGrpGRES"),
        fixed("102", "carol", "RUNNING"),
        fixed("103_[0-9%4]", "dave", "PENDING", reason="Dependency"),
    ]
)
NAMES = "\n".join(
    [
        "100|gb-sft-run-train-1a2b",
        "101|gb-eval-bench-3c4d",
        "102|notebook",
        "103_[0-9%4]|sweep",
    ]
)
COMMENTS = "\n".join(
    [f"100|{ALICE_TAG}", f"101|{BOB_TAG}", "102|carol's own note", "103_[0-9%4]|(null)"]
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
            ("gres/gpu:h200:2,gres/foo:1", (2, "h200", True)),
            (None, (None, None, True)),
            ("gres/foo:1", (None, None, True)),
            ("gres/gpu:h200", (None, None, False)),  # gpu, but no count we can read
        ],
    )
    def test_cases(self, gres, expected):
        assert eq.parse_gpu_gres(gres) == expected


class TestParseGbComment:
    def test_round_trips_the_writer(self):
        tag = eq.parse_gb_comment(ALICE_TAG)
        assert (tag.build_id, tag.gb_env, tag.space, tag.username) == (
            "b-alice",
            "DEV",
            "public",
            "alice",
        )
        assert tag.build_name == "sft_run"  # sanitized at write time
        assert tag.step_id == "s-1"

    def test_every_writer_key_has_a_model_field(self):
        assert eq._GB_COMMENT_KEYS <= set(eq.GbJobTag.model_fields)

    @pytest.mark.parametrize(
        "comment", [None, "", "free text", "build_name=x;target=y"]
    )
    def test_not_ours(self, comment):
        assert eq.parse_gb_comment(comment) is None


class TestParseFixedLine:
    def test_gpus_are_per_node_times_nodes(self):
        job = eq.parse_fixed_line(
            fixed("100", "u", "RUNNING", gres="gres/gpu:h200:4", nodes="2")
        )
        assert (job.gpus_per_node, job.gpus, job.gpu_model, job.nodes) == (
            4,
            8,
            "h200",
            2,
        )

    def test_pending_array(self):
        job = eq.parse_fixed_line(
            fixed("103_[0-9%4]", "u", "PENDING", reason="Dependency")
        )
        assert (job.array_job_id, job.start_time, job.nodelist, job.reason) == (
            "103",
            None,
            None,
            "Dependency",
        )

    @pytest.mark.parametrize(
        "line",
        [
            "",
            "too|few",
            fixed("100", "u", "COMPLETED"),  # unexpected state
            fixed("100", "u", "RUNNING", nodes="x"),  # non-numeric field
            fixed("abc", "u", "RUNNING"),  # not a job id
            fixed("100", "u", "RUNNING") + "|extra",  # one field too many
            fixed("100", "u", "RUNNING", gres="gres/gpu:h200"),
        ],
    )
    def test_rejects(self, line):
        assert eq.parse_fixed_line(line) is None


class TestParseQueue:
    def test_joins_on_job_id(self):
        jobs, bad = eq.parse_queue(FIXED, NAMES, COMMENTS)
        assert bad == 0
        by = {j.job_id: j for j in jobs}
        assert (
            by["100"].name == "gb-sft-run-train-1a2b"
            and by["100"].gb.username == "alice"
        )
        assert by["102"].comment == "carol's own note" and by["102"].gb is None
        assert by["103_[0-9%4]"].comment is None

    def test_pipe_in_a_name_cannot_shift_columns(self):
        jobs, bad = eq.parse_queue(
            fixed("200", "eve", "RUNNING"), "200|a|RUNNING|forged", "200|(null)"
        )
        assert bad == 0 and jobs[0].state == "RUNNING" and jobs[0].user == "eve"
        assert jobs[0].name == "a|RUNNING|forged"

    def test_a_name_cannot_create_a_job(self):
        """Only the fixed-format query creates jobs; text lines just annotate them."""
        jobs, _ = eq.parse_queue(
            fixed("200", "eve", "RUNNING"), "200|x\n999|svc-builder", ""
        )
        assert [j.job_id for j in jobs] == ["200"]

    def test_repeated_id_in_a_text_column_is_dropped(self):
        """A newline in one job's text can't forge another job's tag."""
        forged = f"201|x\n200|{ALICE_TAG}"
        jobs, bad = eq.parse_queue(
            fixed("200", "eve", "RUNNING") + "\n" + fixed("201", "eve", "RUNNING"),
            "",
            f"200|(null)\n{forged}",
        )
        by = {j.job_id: j for j in jobs}
        assert by["200"].gb is None and by["200"].comment is None
        assert bad == 2

    def test_bad_fixed_lines_are_counted(self):
        jobs, bad = eq.parse_queue(FIXED + "\ngarbage", NAMES, COMMENTS)
        assert len(jobs) == 4 and bad == 1


def test_commands_are_constants():
    assert eq.SQUEUE_FIXED.startswith("squeue -h -a -t RUNNING,PENDING -o '%i|")
    assert "%j" not in eq.SQUEUE_FIXED and "%k" not in eq.SQUEUE_FIXED
    assert eq.SQUEUE_NAME.endswith("-o '%i|%j'") and eq.SQUEUE_COMMENT.endswith(
        "-o '%i|%k'"
    )


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


ASSET = textwrap.dedent("""\
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
            User: svc-builder
            IdentityKey: CLUSTER_A_SSH_KEY
    """)


@pytest.fixture
def asset_config(tmp_path):
    p = tmp_path / "environment.yaml"
    p.write_text(ASSET)
    return EnvironmentConfig.from_yaml(p)


class TestConnectionFromEnvConfig:
    def test_reads_the_host_named_by_cluster(self, asset_config):
        conn = eq.connection_from_env_config("cluster-a", asset_config)
        assert conn == eq.QueueConnection(
            ("login1.cluster-a.example", "login2.cluster-a.example"),
            "svc-builder",
            "CLUSTER_A_SSH_KEY",
        )

    def _cfg(self, **over):
        base = {"name": "x", "type": "Skypilot", "subtype": "slurm", "config": {}}
        base.update(over)
        return EnvironmentConfig(**base)

    def test_single_host_without_cluster(self):
        cfg = self._cfg(
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
        "over, detail",
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
    def test_incomplete_is_503(self, over, detail):
        with pytest.raises(HTTPException) as exc:
            eq.connection_from_env_config("x", self._cfg(**over))
        assert exc.value.status_code == 503 and detail in exc.value.detail


class TestResolveConnection:
    record = eq.QueueEnvironment(space="ops", env_path="skypilot/slurm/cluster-a")

    @pytest.fixture(autouse=True)
    def _fresh(self, monkeypatch, asset_config):
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
        self.seen = {}
        monkeypatch.setattr(
            eq,
            "get_space_environment_uri",
            lambda repo, path: self.seen.setdefault(
                "uri", f"{repo}/environments/{path}"
            ),
        )
        self.loads = 0

        def load(_uri):
            self.loads += 1
            return asset_config, None

        monkeypatch.setattr(eq.Environment, "load_environment_config", load)
        self.key = AsyncMock(return_value="KEY-MATERIAL")
        monkeypatch.setattr(eq, "fetch_space_ssh_key", self.key)

    @pytest.mark.asyncio
    async def test_reads_the_asset_from_the_configured_space_and_caches(self):
        conn, key = await eq.resolve_connection("cluster-a", self.record)
        await eq.resolve_connection("cluster-a", self.record)
        assert (
            self.seen["uri"]
            == "file:///spaces/ops/environments/skypilot/slurm/cluster-a"
        )
        assert conn.username == "svc-builder" and key == "KEY-MATERIAL"
        self.key.assert_awaited_once_with("ops", "CLUSTER_A_SSH_KEY")
        assert self.loads == 1

    @pytest.mark.asyncio
    async def test_unknown_space_is_503(self):
        with pytest.raises(HTTPException) as exc:
            await eq.resolve_connection(
                "cluster-a", eq.QueueEnvironment(space="nope", env_path="p")
            )
        assert exc.value.status_code == 503

    @pytest.mark.asyncio
    async def test_missing_key_secret_is_503_not_404(self):
        self.key.side_effect = HTTPException(404, "ssh key secret not found")
        with pytest.raises(HTTPException) as exc:
            await eq.resolve_connection("cluster-a", self.record)
        assert exc.value.status_code == 503 and "not found" in exc.value.detail


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


@pytest.fixture
def as_user(monkeypatch):
    return lambda login: monkeypatch.setitem(CALLER, "login", login)


class FakeTunnel:
    host = "login1.cluster-a.example"

    def __init__(self, outputs=None, rc=0, stderr=""):
        self.outputs = outputs or {
            eq.SQUEUE_FIXED: FIXED,
            eq.SQUEUE_NAME: NAMES,
            eq.SQUEUE_COMMENT: COMMENTS,
        }
        self.rc, self.stderr, self.commands = rc, stderr, []

    async def run_remote(self, command, raise_on_error=False):
        self.commands.append(command)
        return self.rc, self.outputs.get(command, ""), self.stderr


@pytest.fixture
def queue(monkeypatch):
    """One configured environment backed by a fake tunnel."""
    monkeypatch.setattr(
        eq,
        "QUEUE_ENVIRONMENT_REGISTRY",
        {"cluster-a": eq.QueueEnvironment("ops", "skypilot/slurm/cluster-a")},
    )
    monkeypatch.setattr(eq, "_snapshots", {})
    monkeypatch.setattr(eq, "_locks", {})
    monkeypatch.setattr(
        eq,
        "resolve_connection",
        AsyncMock(
            return_value=(
                eq.QueueConnection(("login1.cluster-a.example",), "svc-builder", "K"),
                "KEY",
            )
        ),
    )
    monkeypatch.setattr(eq, "is_super_admin", lambda request: False)
    state = SimpleNamespace(tunnel=FakeTunnel(), opens=[])

    @contextlib.asynccontextmanager
    async def fake_open(*args, **kwargs):
        state.opens.append(kwargs)
        yield state.tunnel

    monkeypatch.setattr(eq, "open_login_tunnel", fake_open)
    return state


def jobs_of(client, query=""):
    r = client.get(f"/cluster-a/queue{query}")
    assert r.status_code == 200, r.text
    return {j["job_id"]: j for j in r.json()["jobs"]}


def test_route_runs_three_constant_commands_with_a_held_key(queue, client):
    body = client.get("/cluster-a/queue").json()
    assert queue.tunnel.commands == [eq.SQUEUE_FIXED, eq.SQUEUE_NAME, eq.SQUEUE_COMMENT]
    assert (
        queue.opens[0]["key_material"] == "KEY"
        and queue.opens[0]["budget_s"] == eq.GBSERVER_ENV_QUEUE_SSH_BUDGET_S
    )
    assert (
        body["counts"] == {"running": 2, "pending": 2} and body["unparsed_lines"] == 0
    )


def test_own_job_is_full(queue, client):
    job = jobs_of(client)["100"]
    assert job["mine"] and not job["redacted"]
    assert job["name"] == "gb-sft-run-train-1a2b" and job["gb"]["step_id"] == "s-1"


def test_others_tagged_job_hides_name_comment_and_tag_but_build_name(queue, client):
    job = jobs_of(client)["101"]
    assert job["redacted"] and not job["mine"]
    assert job["name"] is None and job["comment"] is None
    assert job["gb"] == {k: None for k in eq.GbJobTag.model_fields} | {
        "build_name": "eval"
    }
    assert (job["user"], job["gpus"]) == ("svc-builder", 8)


def test_others_untagged_job_hides_comment_only(queue, client):
    job = jobs_of(client)["102"]
    assert job["redacted"] and job["comment"] is None and job["name"] == "notebook"


def test_admin_sees_everything(queue, monkeypatch, client):
    monkeypatch.setattr(eq, "is_super_admin", lambda request: True)
    jobs = jobs_of(client)
    assert not any(j["redacted"] for j in jobs.values())
    assert jobs["101"]["name"] == "gb-eval-bench-3c4d" and not jobs["101"]["mine"]


def test_mine_and_tag_filters(queue, as_user, client):
    assert list(jobs_of(client, "?mine=true")) == ["100"]
    # Hidden fields on others' jobs never match.
    assert jobs_of(client, "?username=bob") == {}
    as_user("bob")
    assert list(jobs_of(client, "?mine=true")) == ["101"]
    # Filters compare in the tag's sanitized spelling ("my space" → "my_space").
    assert list(jobs_of(client, "?space=my space")) == ["101"]
    assert list(jobs_of(client, "?gb_env=STAGING&state=pending")) == ["101"]


def test_slurm_user_filter_and_gb_only(queue, client):
    assert sorted(jobs_of(client, "?user=svc-builder")) == ["100", "101"]
    assert sorted(jobs_of(client, "?gb_only=true")) == ["100", "101"]


def test_cache_shares_one_session(queue, client):
    client.get("/cluster-a/queue")
    client.get("/cluster-a/queue?state=running")
    assert len(queue.opens) == 1


def test_failures_are_cached_too(queue, client):
    queue.tunnel = FakeTunnel(rc=1, stderr="slurm_load_jobs error: timed out")
    r1 = client.get("/cluster-a/queue")
    r2 = client.get("/cluster-a/queue")
    assert (r1.status_code, r2.status_code) == (503, 503)
    assert "timed out" in r2.json()["detail"]
    assert len(queue.opens) == 1


def test_bad_state_is_422(queue, client):
    assert client.get("/cluster-a/queue?state=done").status_code == 422


def test_unknown_environment_is_404(queue, client):
    assert client.get("/nowhere/queue").status_code == 404
