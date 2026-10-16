# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""In-memory SQLite round-trip test for ``Database.insert_metrics``.

Mirrors ``tests/test_roundtrip.py``'s ``test_record_logs_preserves_batch_order``:
exercises the ``training_metrics`` Core table via a real (SQLite) engine built
from the shared metadata, reusing that module's FK-seeding helpers.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, event, select
from sqlalchemy.pool import StaticPool

from api_bridge import log_service
from api_bridge import model as bridge_models
from api_bridge.database import Database
from api_bridge.tables import metadata, training_metrics
from tests.test_roundtrip import _config, _dataset, seed_job, seed_user


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )

    @event.listens_for(engine, "connect")
    def _fk_on(dbapi_conn, _record):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")

    metadata.create_all(engine)
    return Database(engine=engine)


def test_insert_metrics_roundtrip(db):
    user_id = seed_user(db)
    config_id = str(db.insert_configuration(_config(user_id)))
    dataset_id = str(db.insert_dataset(_dataset(user_id)).id)
    job_id = seed_job(db, user_id, config_id, dataset_id)

    buffer = [
        {
            "job_id": job_id,
            "trial_id": "t1",
            "global_step": s,
            "epoch": 0.1,
            "loss": float(s),
            "grad_norm": 1.0,
            "learning_rate": 1e-6,
            "split": "train",
            "extra": {},
        }
        for s in (10, 20)
    ]
    assert db.insert_metrics(buffer) is True

    with db._engine.connect() as conn:
        rows = (
            conn.execute(select(training_metrics).order_by(training_metrics.c.id)).mappings().all()
        )
    assert [r["loss"] for r in rows] == [10.0, 20.0]
    assert rows[0]["trial_id"] == "t1"


def test_record_metrics_sanitizes_a_diverged_nan_loss(db):
    """NaN survives pydantic float parsing but breaks a MySQL DOUBLE bind.

    On SQLite it lands as NULL either way, so this asserts the sanitize happens in
    the service — the batch must not be handed to the driver carrying NaN, since
    there the whole INSERT fails and every row in it is lost behind an HTTP 200
    ``{"success": false}``.
    """
    user_id = seed_user(db)
    config_id = str(db.insert_configuration(_config(user_id)))
    dataset_id = str(db.insert_dataset(_dataset(user_id)).id)
    job_id = seed_job(db, user_id, config_id, dataset_id)
    service = log_service.LogService(db)
    metric = bridge_models.TrainingMetric(
        job_id=job_id, trial_id="t1", global_step=1, loss=float("nan"), epoch=1.0
    )

    # This subproject installs no pytest-asyncio (see its pyproject), so the one
    # async service method is driven directly rather than via an async test.
    result = asyncio.run(service.record_metrics([metric]))

    assert result["success"] is True
    with db._engine.connect() as conn:
        row = conn.execute(select(training_metrics)).mappings().one()
    assert row["loss"] is None
    assert row["epoch"] == 1.0


def test_training_metric_rejects_a_null_job_id():
    """job_id is NOT NULL: a null must fail validation, not the INSERT.

    Reaching the database with a null yields a caught IntegrityError, which the
    route reports as HTTP 200 ``{"success": false}`` — a silent loss rather than a
    422 the caller can act on.
    """
    with pytest.raises(ValidationError):
        bridge_models.TrainingMetric(job_id=None, global_step=1)
