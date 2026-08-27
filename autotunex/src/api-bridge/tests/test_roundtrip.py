# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""In-memory SQLite round-trip tests for the api-bridge Database methods.

Exercises each Database method through insert->read against a real (SQLite)
engine built from the Core metadata. This is the coverage the bridge lacked
while it was raw-pymysql/MySQL-only.
"""

import pytest
from sqlalchemy import create_engine

from api_bridge.database import Database
from api_bridge.tables import metadata


@pytest.fixture
def db():
    from sqlalchemy import event
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _record):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    metadata.create_all(engine)
    return Database(engine=engine)


def seed_user(db, email="u@example.com"):
    """Insert a user and return its id (str)."""
    return str(db.insert_user(email))


def test_insert_and_get_user_roundtrip(db):
    user_id = seed_user(db, "person@example.com")

    result = db.get_user("person@example.com")

    assert result is not None
    assert str(result["id"]) == user_id
    assert result["email"] == "person@example.com"
    assert result["created_at"] is not None  # ISO string from get_utc_timestamp


def test_get_user_is_case_insensitive(db):
    seed_user(db, "MixedCase@Example.com")

    assert db.get_user("mixedcase@example.com") is not None


def test_get_user_returns_none_when_absent(db):
    assert db.get_user("nobody@example.com") is None


def _config(user_id, name="cfg", data=None):
    from api_bridge import models

    return models.Configuration(
        user_id=user_id,
        name=name,
        tuner_type="bayesian",
        config_data=data or {"lr": {"min": 1e-5, "max": 1e-3}},
    )


def test_insert_and_get_config_roundtrip(db):
    user_id = seed_user(db)
    config_id = str(db.insert_configuration(_config(user_id)))

    result = db.get_config(config_id)

    assert str(result["id"]) == config_id
    assert result["config_data"] == {"lr": {"min": 1e-5, "max": 1e-3}}  # parsed dict


def test_get_configs_lists_user_configs_with_config_data_parsed(db):
    user_id = seed_user(db)
    db.insert_configuration(_config(user_id, name="a"))
    db.insert_configuration(_config(user_id, name="b"))

    results = db.get_configs(user_id=user_id)

    assert len(results) == 2
    assert all(isinstance(r["config_data"], dict) for r in results)
    assert all(r["associated_jobs"] == [] for r in results)


def test_get_config_by_name_and_user_roundtrip(db):
    user_id = seed_user(db)
    db.insert_configuration(_config(user_id, name="named"))

    result = db.get_config_by_name_and_user("named", user_id)

    assert result is not None
    assert result["name"] == "named"
    assert isinstance(result["config_data"], dict)


def test_update_configuration_roundtrip(db):
    from api_bridge import models

    user_id = seed_user(db)
    config_id = str(db.insert_configuration(_config(user_id, name="orig")))

    updated = models.Configuration(
        id=config_id,
        user_id=user_id,
        name="renamed",
        tuner_type="grid_search",
        config_data={"lr": {"min": 1e-4, "max": 1e-2}},
    )
    db.update_configuration(updated)

    result = db.get_config(config_id)
    assert result["name"] == "renamed"
    assert result["tuner_type"] == "grid_search"
    assert result["config_data"] == {"lr": {"min": 1e-4, "max": 1e-2}}


def _dataset(user_id, name="ds"):
    from api_bridge import models

    return models.DatasetInfo(user_id=user_id, name=name, description="desc")


def test_insert_and_get_dataset_roundtrip(db):
    user_id = seed_user(db)
    created = db.insert_dataset(_dataset(user_id, "d1"))

    result = db.get_dataset(dataset_id=str(created.id), user_id=user_id)

    assert result is not None
    assert result["name"] == "d1"


def test_insert_dataset_duplicate_name_raises_400(db):
    from fastapi import HTTPException

    user_id = seed_user(db)
    db.insert_dataset(_dataset(user_id, "dup"))

    with pytest.raises(HTTPException) as exc:
        db.insert_dataset(_dataset(user_id, "dup"))
    assert exc.value.status_code == 400


def test_check_dataset_exists(db):
    user_id = seed_user(db)
    created = db.insert_dataset(_dataset(user_id, "exists"))

    assert db.check_dataset_exists(str(created.id)) is True
    assert db.check_dataset_exists("00000000-0000-0000-0000-000000000000") is False


def test_get_datasets_lists_with_associated_jobs(db):
    user_id = seed_user(db)
    db.insert_dataset(_dataset(user_id, "a"))
    db.insert_dataset(_dataset(user_id, "b"))

    results = db.get_datasets(user_id)

    assert len(results) == 2
    assert all(r["associated_jobs"] == [] for r in results)


def test_update_dataset_metadata_marks_dataset_ready(db):
    # Attaching an artifact means the dataset now has readable data, so the write
    # path must flip status to 'ready'. Without it the row keeps its 'empty'
    # server-default and the main service refuses to preview it ("Unable to load
    # dataset"), even though the tuning pipeline populated real records/artifact.
    user_id = seed_user(db)
    created = db.insert_dataset(_dataset(user_id, "registered"))

    result = db.update_dataset_metadata(
        id=str(created.id),
        user_id=user_id,
        metadata={
            "train_records": 334,
            "train_file_size": 783904,
            "validation_records": 84,
            "validation_file_size": 425639,
            "artifact_id": None,
            "artifact_url": "hf://huggingface.co/datasets/org/registered_abcd1234",
        },
    )

    assert result["status"] == "ready"
    assert db.get_dataset(dataset_id=str(created.id), user_id=user_id)["status"] == "ready"


def _tuning_config(user_id, config_id, dataset_id, job_id=None):
    from api_bridge import models

    return models.TuningConfig(
        id=job_id,
        user_id=user_id,
        config_id=config_id,
        dataset_id=dataset_id,
        model="meta-llama/Llama-2-7b-hf",
        experiment_name="exp-1",
    )


def seed_job(db, user_id, config_id, dataset_id, job_id="33333333-3333-3333-3333-333333333333"):
    return str(db.insert_job(_tuning_config(user_id, config_id, dataset_id, job_id)))


def test_insert_and_get_job_roundtrip(db):
    user_id = seed_user(db)
    config_id = str(db.insert_configuration(_config(user_id)))
    dataset_id = str(db.insert_dataset(_dataset(user_id)).id)

    job_id = seed_job(db, user_id, config_id, dataset_id)
    result = db.get_job_by_id(job_id)

    assert result is not None
    assert str(result["id"]) == job_id
    assert result["experiment_name"] == "exp-1"
    assert result["model_source"] == "huggingface"  # server_default applied


def test_insert_job_with_missing_config_raises_404(db):
    from fastapi import HTTPException

    user_id = seed_user(db)
    dataset_id = str(db.insert_dataset(_dataset(user_id)).id)

    with pytest.raises(HTTPException) as exc:
        seed_job(db, user_id, "00000000-0000-0000-0000-0000000000ff", dataset_id)
    assert exc.value.status_code == 404


def test_insert_gb_task_roundtrip(db):
    user_id = seed_user(db)
    config_id = str(db.insert_configuration(_config(user_id)))
    dataset_id = str(db.insert_dataset(_dataset(user_id)).id)
    job_id = seed_job(db, user_id, config_id, dataset_id)

    task_id = db.insert_gb_task(job_id=job_id, build_id="44444444-4444-4444-4444-444444444444")

    assert task_id is not None


def test_insert_trial_and_result_roundtrip(db):
    from sqlalchemy import select as sa_select

    from api_bridge import model as bridge_models
    from api_bridge.tables import results, trials

    user_id = seed_user(db)
    config_id = str(db.insert_configuration(_config(user_id)))
    dataset_id = str(db.insert_dataset(_dataset(user_id)).id)
    job_id = seed_job(db, user_id, config_id, dataset_id)

    trial = bridge_models.Trial(
        id="trial-abc", job_id=job_id, status=bridge_models.TrialStatus.RUNNING, config={"lr": 0.1}
    )
    db.insert_trial(trial)
    db.insert_result(
        {
            "id": job_id,
            "job_id": job_id,
            "trial_id": "trial-abc",
            "metric": "loss",
            "metrics": {"loss": 0.5},
        }
    )

    with db._engine.connect() as conn:
        t = conn.execute(sa_select(trials)).mappings().first()
        r = conn.execute(sa_select(results)).mappings().first()
    assert t["config"] == {"lr": 0.1}  # JSON round-trips as dict
    assert r["metrics"] == {"loss": 0.5}


def test_record_logs_preserves_batch_order(db):
    from sqlalchemy import select as sa_select

    from api_bridge.tables import log_entries

    user_id = seed_user(db)
    config_id = str(db.insert_configuration(_config(user_id)))
    dataset_id = str(db.insert_dataset(_dataset(user_id)).id)
    job_id = seed_job(db, user_id, config_id, dataset_id)

    buffer = [
        {
            "job_id": job_id,
            "trial_id": None,
            "level": "INFO",
            "filename": "f",
            "message": f"m{i}",
            "iteration": i,
            "epoch": None,
            "timestamp": None,
        }
        for i in range(5)
    ]
    assert db.insert_logs(buffer) is True

    with db._engine.connect() as conn:
        rows = conn.execute(sa_select(log_entries).order_by(log_entries.c.id)).mappings().all()
    assert [r["message"] for r in rows] == ["m0", "m1", "m2", "m3", "m4"]


def test_update_job_status_roundtrip(db):
    from api_bridge import model as bridge_models

    user_id = seed_user(db)
    config_id = str(db.insert_configuration(_config(user_id)))
    dataset_id = str(db.insert_dataset(_dataset(user_id)).id)
    job_id = seed_job(db, user_id, config_id, dataset_id)

    db.update_job_status(id=job_id, status=bridge_models.JobStatus.RUNNING)

    assert db.get_job_by_id(job_id)["status"] == "RUNNING"


def test_structure_check_passes_when_tables_exist(db):
    db.test_db_connection_and_structure()  # must not raise


def test_structure_check_raises_when_table_missing():
    from fastapi import HTTPException
    from sqlalchemy import create_engine

    engine = create_engine("sqlite://")  # empty: no tables created
    empty_db = Database(engine=engine)

    with pytest.raises(HTTPException) as exc:
        empty_db.test_db_connection_and_structure()
    assert exc.value.status_code == 500
