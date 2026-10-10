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

"""Tests for fetching one job's content from the store its index rows name."""

from types import SimpleNamespace

from gbserver.lineage.db_service import DBLineageService
from gbserver.storage.stored_lineage_job import StoredLineageJob
from gbserver.storage.stored_lineage_row import JobStore, StoredLineageRow
from gbserver.utils.redaction import REDACTED

A = "lh://prod/ns/tables/a"
B = "lh://prod/ns/tables/b"

SECRET_PARAMS = {"step": {"api_key": "s3cret", "commit_hash": "abc"}}


def index_row(job_id, job_store, retrieve=None):
    attributes = {
        "job": {"id": job_id, "namespace": "space/build", "status": "success"}
    }
    if retrieve:
        attributes["retrieve"] = retrieve
    return StoredLineageRow(
        job_id=job_id, input=A, output=B, job_store=job_store, attributes=attributes
    )


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def get_rows_by_job(self, job_id):
        return [r for r in self.rows if r.job_id == job_id]


class ByUuid:
    def __init__(self, items=None):
        self.items = items or {}

    def get_by_uuid(self, uuid):
        return self.items.get(uuid)

    def get_by_where(self, where):
        return [s for s in self.items.values() if s.target_id == where["target_id"]]


def dumpable(**fields):
    return SimpleNamespace(**fields, model_dump=lambda mode=None: dict(fields))


def admin(jobs=None, builds=None, targets=None, steps=None):
    return SimpleNamespace(
        lineage_job_storage=SimpleNamespace(
            get_job=lambda job_id: (jobs or {}).get(job_id)
        ),
        build_storage=ByUuid(builds),
        target_storage=ByUuid(targets),
        step_storage=ByUuid(steps),
    )


def allow(build):
    return None


def deny(build):
    raise PermissionError("no")


def detail(
    rows, job_id="j1", authorize=allow, wandb_runs=None, authorize_entry=None, **storage
):
    return DBLineageService(storage=Rows(rows)).get_job_detail(
        job_id,
        authorize_build=authorize,
        admin_storage=admin(**storage),
        wandb_runs=wandb_runs or (lambda job_id: []),
        authorize_entry=authorize_entry,
    )


def test_an_unknown_job_is_none():
    assert detail([]) is None


def test_lineage_job_store_reads_the_job_table_and_redacts():
    job = StoredLineageJob(
        job_id="j1",
        attributes={
            "payload": {
                "job_input_params": SECRET_PARAMS,
                "execution_stats": {"rows": 3},
            }
        },
    )
    result = detail([index_row("j1", JobStore.LINEAGE_JOB)], jobs={"j1": job})
    assert result["job_store"] == "lineage_job"
    assert result["detail_available"] is True
    assert result["detail"]["execution_stats"] == {"rows": 3}
    assert result["detail"]["job_input_params"]["step"]["api_key"] == REDACTED
    assert result["detail"]["job_input_params"]["step"]["commit_hash"] == "abc"
    assert result["inputs"] == [A] and result["outputs"] == [B]


def test_a_missing_job_table_row_degrades_with_a_reason():
    result = detail([index_row("j1", JobStore.LINEAGE_JOB)])
    assert result["detail_available"] is False
    assert "lineage job table" in result["detail_error"]
    assert result["job_id"] == "j1"


TARGETS = {"retrieve": {"build_id": "b1", "target_run_uuid": "t1"}}


def targets_storage():
    return dict(
        builds={"b1": dumpable(uuid="b1", build_archive="big")},
        targets={"t1": dumpable(uuid="t1", build_id="b1", target_name="train")},
        steps={"s1": dumpable(uuid="s1", target_id="t1", step_name="run")},
    )


def test_targets_store_returns_the_target_with_its_steps():
    result = detail([index_row("j1", JobStore.TARGETS, **TARGETS)], **targets_storage())
    assert result["detail_available"] is True
    assert result["build_id"] == "b1"
    assert result["build"]["build_archive"] == ""
    assert result["target"]["target_name"] == "train"
    assert [s["step_name"] for s in result["target"]["steps"]] == ["run"]


def test_targets_store_withholds_a_build_the_caller_may_not_read():
    result = detail(
        [index_row("j1", JobStore.TARGETS, **TARGETS)],
        authorize=deny,
        **targets_storage(),
    )
    assert result["detail_available"] is False
    assert "target" not in result
    assert "Not authorized" in result["detail_error"]


def test_targets_store_reports_a_build_not_on_this_server():
    result = detail([index_row("j1", JobStore.TARGETS, **TARGETS)])
    assert result["detail_available"] is False
    assert "not on this server" in result["detail_error"]


def test_wandb_store_reads_the_run_config():
    run = SimpleNamespace(
        config={"job_input_params": SECRET_PARAMS}, url="https://wandb/run"
    )
    result = detail([index_row("j1", JobStore.WANDB)], wandb_runs=lambda job_id: [run])
    assert result["detail_available"] is True
    assert result["origin_url"] == "https://wandb/run"
    assert result["detail"]["job_input_params"]["step"]["api_key"] == REDACTED


def test_a_deleted_wandb_run_degrades():
    result = detail([index_row("j1", JobStore.WANDB)])
    assert result["detail_available"] is False
    assert "W&B" in result["detail_error"]


def test_a_failing_store_degrades_rather_than_raising():
    def boom(job_id):
        raise RuntimeError("wandb is down")

    result = detail([index_row("j1", JobStore.WANDB)], wandb_runs=boom)
    assert result["detail_available"] is False
    assert "wandb is down" in result["detail_error"]


def test_other_store_returns_the_index_entry_only():
    result = detail([index_row("j1", JobStore.OTHER)])
    assert result["job_store"] == "other"
    assert result["detail_available"] is False
    assert result["job"]["status"] == "success"


def test_lineage_job_store_withholds_a_job_outside_the_callers_spaces():
    job = StoredLineageJob(
        job_id="j1", attributes={"payload": {"job_input_params": SECRET_PARAMS}}
    )
    seen = []

    def outsider(entry):
        seen.append(entry["space_name"])
        return False

    result = detail(
        [index_row("j1", JobStore.LINEAGE_JOB)],
        jobs={"j1": job},
        authorize_entry=outsider,
    )
    assert seen == ["space"]
    assert result["detail_available"] is False
    assert "detail" not in result


def test_wandb_store_withholds_a_job_outside_the_callers_spaces():
    def runs(job_id):
        raise AssertionError("W&B must not be read for an unauthorized caller")

    result = detail(
        [index_row("j1", JobStore.WANDB)],
        wandb_runs=runs,
        authorize_entry=lambda entry: False,
    )
    assert result["detail_available"] is False
