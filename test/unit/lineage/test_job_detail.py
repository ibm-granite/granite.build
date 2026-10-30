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

"""Tests for fetching one job's content from its build or the lineage job table."""

from types import SimpleNamespace

import pytest

from gbserver.lineage.db_service import DBLineageService
from gbserver.storage.stored_lineage_row import JobStore, StoredLineageRow

A = "lh://prod/ns/tables/a"
B = "lh://prod/ns/tables/b"


def index_row(job_id, job_store, retrieve=None, origin=None):
    attributes = {
        "job": {"id": job_id, "namespace": "space/build", "status": "success"}
    }
    if retrieve:
        attributes["retrieve"] = retrieve
    if origin:
        attributes["origin"] = origin
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


def admin(jobs=None, builds=None, targets=None, steps=None, artifacts=None):
    return SimpleNamespace(
        artifact_registry=ByUuid(artifacts),
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


def detail(rows, job_id="j1", authorize=allow, authorize_entry=None, **storage):
    return DBLineageService(storage=Rows(rows)).get_job_detail(
        job_id,
        authorize_build=authorize,
        admin_storage=admin(**storage),
        authorize_entry=authorize_entry,
    )


def test_an_unknown_job_is_none():
    assert detail([]) is None


TARGETS = {"retrieve": {"build_id": "b1", "target_run_uuid": "t1"}}


def targets_storage():
    return dict(
        builds={"b1": dumpable(uuid="b1", build_archive="big")},
        targets={"t1": dumpable(uuid="t1", build_id="b1", target_name="train")},
        steps={"s1": dumpable(uuid="s1", target_id="t1", step_name="run")},
    )


REGISTRATION = {"retrieve": {"build_id": "art1"}}


def test_a_registration_row_shows_only_the_index():
    """A registration's build_id holds its artifact UUID, which names no build
    here: the index is shown, whatever the registry holds."""
    result = detail(
        [index_row("j1", JobStore.WANDB, **REGISTRATION)],
        authorize_entry=lambda e: True,
        artifacts={"art1": SimpleNamespace(uuid="art1", created_by_target_id="t1")},
        **targets_storage(),
    )
    assert result["detail_available"] is False
    assert result["detail_error"] == "The build is not on this server"
    assert "target" not in result


def test_a_build_reader_outside_the_jobs_space_gets_no_space_gated_fields():
    """The build check releases the build and target; owner, namespace and tags
    stay behind the space check."""
    result = detail(
        [index_row("j1", JobStore.WANDB, **TARGETS)],
        authorize_entry=lambda e: False,
        **targets_storage(),
    )
    assert result["detail_available"] is True
    assert result["target"]["target_name"] == "train"
    for key in _SPACE_GATED:
        assert key not in result


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
    assert result["detail_denied"] is True


def test_a_build_not_on_this_server_is_reported():
    result = detail([index_row("j1", JobStore.TARGETS, **TARGETS)])
    assert result["detail_available"] is False
    assert "not on this server" in result["detail_error"]
    assert result["detail_denied"] is False


def test_other_store_returns_the_index_entry_only():
    result = detail([index_row("j1", JobStore.OTHER)])
    assert result["job_store"] == "other"
    assert result["detail_available"] is False
    assert result["job"]["status"] == "success"


_SPACE_GATED = ("job_namespace", "space_name", "owner", "tags", "origin")


def test_an_unauthorized_detail_returns_only_the_global_fields():
    """The listing entry carries owner, namespace, tags and origin ids; a caller
    who fails the space check gets the trace fields and nothing else."""
    result = detail(
        [index_row("j1", JobStore.LINEAGE_JOB)], authorize_entry=lambda entry: False
    )
    assert result["job_id"] == "j1"
    assert result["job"] == {"status": "success"}
    for key in _SPACE_GATED:
        assert key not in result


def test_an_unavailable_store_returns_only_the_global_fields_to_an_outsider():
    result = detail([index_row("j1", JobStore.OTHER)], authorize_entry=lambda e: False)
    for key in _SPACE_GATED:
        assert key not in result
    # Withheld, not missing, so the drawer says so instead of a bare N/A.
    assert result["detail_denied"] is True


def test_an_imported_job_returns_the_full_entry_to_a_reader():
    """An import has no store of its own: a reader still sees owner and namespace."""
    result = detail([index_row("j1", JobStore.OTHER)], authorize_entry=lambda e: True)
    assert result["detail_available"] is False
    assert result["detail_denied"] is False
    assert result["job_namespace"] == "space/build"
    assert result["space_name"] == "space"


def test_a_pruned_build_returns_the_full_entry_to_a_reader():
    result = detail(
        [index_row("j1", JobStore.TARGETS, **TARGETS)], authorize_entry=lambda e: True
    )
    assert result["detail_available"] is False
    assert result["job_namespace"] == "space/build"
    # The build's id is not in the reason: it was never authorized.
    assert (
        "b1" not in result["detail_error"]
        and "not on this server" in result["detail_error"]
    )


def test_a_build_the_caller_may_not_read_withholds_the_entry():
    """The build's own check wins over the space check on the entry."""
    result = detail(
        [index_row("j1", JobStore.TARGETS, **TARGETS)],
        authorize=deny,
        authorize_entry=lambda e: True,
        **targets_storage(),
    )
    assert result["detail_denied"] is True
    for key in _SPACE_GATED:
        assert key not in result


def test_an_authorized_detail_returns_the_full_entry():
    result = detail(
        [index_row("j1", JobStore.LINEAGE_JOB)], authorize_entry=lambda entry: True
    )
    assert result["job_namespace"] == "space/build"
    assert result["space_name"] == "space"


@pytest.mark.parametrize("job_store", [JobStore.LINEAGE_JOB, JobStore.WANDB])
def test_a_granite_build_job_is_read_from_its_build_whatever_indexed_it(job_store):
    result = detail([index_row("j1", job_store, **TARGETS)], **targets_storage())
    assert result["detail_available"] is True
    assert result["job_store"] == job_store.value
    assert result["build_id"] == "b1"
    assert result["target"]["target_name"] == "train"
    assert [s["step_name"] for s in result["target"]["steps"]] == ["run"]


@pytest.mark.parametrize(
    "job_store", [s for s in JobStore if s != JobStore.LINEAGE_JOB]
)
def test_a_job_with_no_build_shows_only_the_index(job_store):
    result = detail(
        [index_row("j1", job_store, retrieve={"job_id": "j1"})],
        authorize_entry=lambda entry: True,
    )
    assert result["detail_available"] is False
    assert result["detail_error"] == "Only the index is recorded"
    assert "target" not in result and "detail" not in result
    assert result["job_namespace"] == "space/build"


def test_a_target_of_another_build_is_not_shown():
    storage = targets_storage()
    storage["targets"]["t1"] = dumpable(uuid="t1", build_id="other", target_name="x")
    result = detail(
        [index_row("j1", JobStore.WANDB, **TARGETS)],
        authorize_entry=lambda entry: True,
        **storage,
    )
    assert result["detail_available"] is False
    assert result["detail_error"] == "The target run is not on this server"
    assert "target" not in result
    assert result["build_id"] == "b1" and result["target_run_uuid"] == "t1"


def test_a_build_the_index_names_but_this_server_lacks_shows_the_index():
    result = detail(
        [index_row("j1", JobStore.WANDB, **TARGETS)], authorize_entry=lambda e: True
    )
    assert result["detail_available"] is False
    assert result["detail_error"] == "The build is not on this server"
    assert result["build_id"] == "b1" and result["target_run_uuid"] == "t1"


def test_a_failing_build_read_degrades_rather_than_raising():
    def boom(uuid):
        raise RuntimeError("SELECT secret")

    storage = admin(**targets_storage())
    storage.build_storage.get_by_uuid = boom
    result = DBLineageService(
        storage=Rows([index_row("j1", JobStore.TARGETS, **TARGETS)])
    ).get_job_detail("j1", authorize_build=allow, admin_storage=storage)
    assert result["detail_available"] is False
    # The storage error text stays in the log.
    assert result["detail_error"] == "Could not read the build"


def test_an_outsider_gets_no_target_ids():
    result = detail(
        [index_row("j1", JobStore.LINEAGE_JOB, **TARGETS)],
        authorize_entry=lambda entry: False,
    )
    assert result["target_run_uuid"] is None
    assert not result.get("build_id")


def test_a_lineage_event_with_no_build_reads_the_job_table_and_redacts():
    job = SimpleNamespace(
        attributes={
            "payload": {
                "job_input_params": {"api_key": "secret"},
                "execution_stats": {"rows": 3},
            }
        }
    )
    result = detail(
        [index_row("j1", JobStore.LINEAGE_JOB, retrieve={"job_id": "j1"})],
        authorize_entry=lambda entry: True,
        jobs={"j1": job},
    )
    assert result["detail_available"] is True
    assert result["detail"]["execution_stats"] == {"rows": 3}
    assert result["detail"]["job_input_params"] != {"api_key": "secret"}


def test_a_lineage_event_missing_from_the_job_table_shows_the_index():
    result = detail(
        [index_row("j1", JobStore.LINEAGE_JOB, retrieve={"job_id": "j1"})],
        authorize_entry=lambda entry: True,
    )
    assert result["detail_available"] is False
    assert result["detail_error"] == "Only the index is recorded"


def test_a_lineage_events_payloads_are_withheld_from_an_outsider():
    job = SimpleNamespace(attributes={"payload": {"execution_stats": {"rows": 3}}})
    result = detail(
        [index_row("j1", JobStore.LINEAGE_JOB, retrieve={"job_id": "j1"})],
        authorize_entry=lambda entry: False,
        jobs={"j1": job},
    )
    assert "detail" not in result or not result["detail"]
    assert result["detail_denied"] is True


def test_an_older_row_with_origin_ids_is_read_from_its_build():
    result = detail(
        [
            index_row(
                "j1",
                JobStore.TARGETS,
                origin={"ids": {"build_id": "b1", "target_run_uuid": "t1"}},
            )
        ],
        **targets_storage(),
    )
    assert result["detail_available"] is True
    assert result["target"]["target_name"] == "train"


def test_a_missing_build_is_reported_as_denied_to_an_outsider():
    # The same answer as for a build that is present, so an outsider cannot tell
    # whether a build in another space still exists.
    result = detail(
        [index_row("j1", JobStore.WANDB, **TARGETS)], authorize_entry=lambda e: False
    )
    assert result["detail_error"] == "Not authorized to read this build"
    assert result["detail_denied"] is True
    for key in _SPACE_GATED:
        assert key not in result


def test_a_target_with_no_build_shows_only_the_index():
    # A W&B registration tags its producing target but no build: not a build read.
    result = detail(
        [index_row("j1", JobStore.WANDB, retrieve={"target_run_uuid": "t1"})],
        authorize_entry=lambda entry: True,
        **targets_storage(),
    )
    assert result["detail_available"] is False
    assert result["detail_error"] == "Only the index is recorded"
    assert result["detail_denied"] is False
    assert "target" not in result
