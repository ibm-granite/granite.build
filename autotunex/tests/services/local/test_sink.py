"""DbTrialSink persistence driven from a worker thread, and id coercion.

The sink is worker-thread only: every test that persists drives it through
``asyncio.to_thread`` (as a Ray callback would), while the event loop stays free
to service the coroutines the sink schedules onto it.
"""

from __future__ import annotations

import asyncio
import logging
import pickle
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from autotunex.db.tables import ConfigurationTable, DatasetTable, JobTable, UserTable
from autotunex.db.tables.log_entries import LogEntryTable
from autotunex.db.tables.results import ResultTable
from autotunex.db.tables.training_metrics import TrainingMetricTable
from autotunex.db.tables.trials import TrialTable
from autotunex.models.status import DatasetStatus, RunStatus
from autotunex.services.local.protocols import LogRecord, TrainingMetricRecord
from autotunex.services.local.sink import DbTrialSink, SinkLogHandler


async def _seed_job(factory: async_sessionmaker[AsyncSession]) -> UUID:
    async with factory() as session:
        user = UserTable(id=uuid4(), email=f"{uuid4()}@example.com", role="user")
        session.add(user)
        await session.commit()
        config = ConfigurationTable(
            id=uuid4(),
            user_id=str(user.id),
            name="c",
            tuner_type="lora",
            rl_tuner_type=None,
            config_data={"a": 1},
        )
        dataset = DatasetTable(
            id=uuid4(),
            user_id=str(user.id),
            name="alpaca",
            data_format="jsonl",
            status=DatasetStatus.READY,
            artifact_url="s3://data/alpaca",
        )
        session.add_all([config, dataset])
        await session.commit()
        job = JobTable(
            id=uuid4(),
            user_id=str(user.id),
            status=RunStatus.PENDING,
            config_id=config.id,
            dataset_id=dataset.id,
            model="ibm/granite",
            model_source="huggingface",
            experiment_name="exp",
            tuning_type="lora",
            config_snapshot={
                "name": "c",
                "tuner_type": "lora",
                "rl_tuner_type": None,
                "config_data": {"a": 1},
            },
        )
        session.add(job)
        await session.commit()
        return job.id


async def test_sink_persists_trial_from_worker_thread(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed_job(factory)
    loop = asyncio.get_running_loop()
    sink = DbTrialSink(session_factory=factory, loop=loop, job_id=job_id)

    await asyncio.to_thread(sink.trial_started, "ray_abcdef0001", {"lr": 1})
    await asyncio.to_thread(sink.trial_result, "ray_abcdef0001", "loss", {"loss": 0.5})
    await asyncio.to_thread(sink.trial_completed, "ray_abcdef0001")

    trial_id = DbTrialSink.coerce_trial_id("ray_abcdef0001")
    async with factory() as session:
        trial = await session.get(TrialTable, trial_id)
        result = (
            await session.execute(select(ResultTable).where(ResultTable.trial_id == trial_id))
        ).scalar_one_or_none()
    assert trial is not None and trial.status == RunStatus.COMPLETED
    assert result is not None and result.metrics == {"loss": 0.5}


async def test_sink_marks_trial_error_from_worker_thread(engine: AsyncEngine) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed_job(factory)
    loop = asyncio.get_running_loop()
    sink = DbTrialSink(session_factory=factory, loop=loop, job_id=job_id)

    await asyncio.to_thread(sink.trial_started, "ray_0002", None)
    await asyncio.to_thread(sink.trial_error, "ray_0002")

    async with factory() as session:
        trial = await session.get(TrialTable, "ray_0002")
    assert trial is not None and trial.status == RunStatus.ERROR


async def test_log_handler_flushes_buffered_records_to_the_sink(
    engine: AsyncEngine,
) -> None:
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed_job(factory)
    loop = asyncio.get_running_loop()
    sink = DbTrialSink(session_factory=factory, loop=loop, job_id=job_id)
    handler = SinkLogHandler(sink, trial_id=None)
    logger = logging.getLogger(f"test_sink_handler_{uuid4().hex}")
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    def emit_then_flush() -> None:
        logger.info("hello from worker")
        handler.flush()

    await asyncio.to_thread(emit_then_flush)

    async with factory() as session:
        rows = (
            (await session.execute(select(LogEntryTable).where(LogEntryTable.job_id == job_id)))
            .scalars()
            .all()
        )
    logger.removeHandler(handler)
    assert any(row.message == "hello from worker" for row in rows)


async def test_sink_log_persists_a_trial_tagged_entry(engine: AsyncEngine) -> None:
    """A ``LogRecord`` carrying a trial id is persisted tagged with that trial.

    This is the trial-level path the trainer's ``_SinkCallback`` uses: the job's
    driver logs are tagged ``trial_id=None`` (above), while each trial lifecycle
    line carries the trial's id so ``log_entries`` can be filtered per trial.
    """
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed_job(factory)
    loop = asyncio.get_running_loop()
    sink = DbTrialSink(session_factory=factory, loop=loop, job_id=job_id)

    await asyncio.to_thread(
        sink.log,
        LogRecord(
            trial_id="t01",
            level="INFO",
            filename=None,
            message="Trial t01 started",
            iteration=None,
            epoch=None,
        ),
    )

    async with factory() as session:
        rows = (
            (await session.execute(select(LogEntryTable).where(LogEntryTable.job_id == job_id)))
            .scalars()
            .all()
        )
    assert any(row.trial_id == "t01" and row.message == "Trial t01 started" for row in rows)


async def test_sink_training_metric_persists_a_row_from_a_worker_thread(
    engine: AsyncEngine,
) -> None:
    """The stream-parse end and the repository end are each covered; this is the bridge.

    ``DbTrialSink.training_metric`` is the only place ``coerce_trial_id`` is applied
    to a metric row, so it is also the only place a trial id that charts can join
    against is established.
    """
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed_job(factory)
    loop = asyncio.get_running_loop()
    sink = DbTrialSink(session_factory=factory, loop=loop, job_id=job_id)

    await asyncio.to_thread(
        sink.training_metric,
        TrainingMetricRecord(
            trial_id="t01",
            global_step=12,
            epoch=0.5,
            loss=1.25,
            grad_norm=2.0,
            learning_rate=1e-6,
            split="train",
            extra={"k": "v"},
        ),
    )

    async with factory() as session:
        rows = (
            (
                await session.execute(
                    select(TrainingMetricTable).where(TrainingMetricTable.job_id == job_id)
                )
            )
            .scalars()
            .all()
        )
    assert [(r.trial_id, r.global_step, r.loss, r.extra) for r in rows] == [
        ("t01", 12, 1.25, {"k": "v"})
    ]


async def test_sink_training_metric_coerces_an_overlong_ray_trial_id(
    engine: AsyncEngine,
) -> None:
    """``trials.id``/``training_metrics.trial_id`` are VARCHAR(16); Ray ids run longer.

    The coercion must match the one ``_SinkCallback`` applies when it writes the
    ``trials`` row, or ``GET /jobs/{id}/trials/{trial_id}/metrics`` would never join.
    """
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed_job(factory)
    loop = asyncio.get_running_loop()
    sink = DbTrialSink(session_factory=factory, loop=loop, job_id=job_id)
    raw = "ray_" + "b" * 40

    await asyncio.to_thread(
        sink.training_metric,
        TrainingMetricRecord(
            trial_id=raw,
            global_step=1,
            epoch=None,
            loss=None,
            grad_norm=None,
            learning_rate=None,
            split="train",
            extra=None,
        ),
    )

    async with factory() as session:
        row = (
            await session.execute(
                select(TrainingMetricTable).where(TrainingMetricTable.job_id == job_id)
            )
        ).scalar_one()
    assert row.trial_id == DbTrialSink.coerce_trial_id(raw)
    assert len(row.trial_id or "") <= 16


async def test_sink_training_metric_keeps_a_null_trial_id_null(engine: AsyncEngine) -> None:
    """The final-training run has no ``trials`` row, so job-level metrics carry no trial."""
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed_job(factory)
    loop = asyncio.get_running_loop()
    sink = DbTrialSink(session_factory=factory, loop=loop, job_id=job_id)

    await asyncio.to_thread(
        sink.training_metric,
        TrainingMetricRecord(
            trial_id=None,
            global_step=3,
            epoch=None,
            loss=0.5,
            grad_norm=None,
            learning_rate=None,
            split="eval",
            extra=None,
        ),
    )

    async with factory() as session:
        row = (
            await session.execute(
                select(TrainingMetricTable).where(TrainingMetricTable.job_id == job_id)
            )
        ).scalar_one()
    assert row.trial_id is None
    assert row.split == "eval"


class _FakeSearcher:
    """Stands in for a Ray ``search_alg``/``scheduler`` *instance*.

    The ``autotune`` optimizer replaces the ``search_alg`` string in the shared
    ``param_space["tune_config"]`` with a live searcher object (e.g.
    ``LimitedDiscrepancySearch``), which Ray then hands back inside every
    ``trial.config`` — and which the ``trials.config`` JSON column cannot store.
    """


class _FakeScalar:
    """Mimics a NumPy scalar: not JSON-serializable, but unwraps via ``.item()``.

    Integer search spaces sampled by Ray can surface as ``numpy.int64`` (not a
    ``int`` subclass), so a JSON-safe coercion must recover the value rather than
    stringify it away.
    """

    def item(self) -> int:
        return 42


async def test_sink_stores_a_json_safe_trial_config(engine: AsyncEngine) -> None:
    """A trial config carrying Ray machinery still persists, JSON-safe.

    Ray hands back each trial's config with the ``autotune`` machinery the
    optimizer stuffed into the shared ``param_space`` — including live
    ``search_alg``/``scheduler`` instances the ``trials.config`` JSON column
    cannot encode. The sink reduces the config to a JSON-safe form: sampled
    hyperparameters are preserved exactly, a NumPy-style scalar is unwrapped to
    its value, and any remaining unserializable object collapses to a stable
    ``module.qualname`` marker so the row records *what* ran without the object.
    """
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    job_id = await _seed_job(factory)
    loop = asyncio.get_running_loop()
    sink = DbTrialSink(session_factory=factory, loop=loop, job_id=job_id)

    await asyncio.to_thread(
        sink.trial_started,
        "t01",
        {
            "learning_rate": 1e-6,
            "r": _FakeScalar(),
            "tune_config": {"search_alg": _FakeSearcher(), "metric": "loss"},
        },
    )

    async with factory() as session:
        trial = await session.get(TrialTable, "t01")
    assert trial is not None and trial.config is not None
    assert trial.config["learning_rate"] == 1e-6  # primitive preserved exactly
    assert trial.config["r"] == 42  # NumPy-style scalar unwrapped, not stringified
    assert trial.config["tune_config"]["metric"] == "loss"  # nested primitive preserved
    assert trial.config["tune_config"]["search_alg"].endswith("._FakeSearcher")


async def test_sink_pickles_to_a_hollow_shell_that_keeps_only_the_job_id(
    engine: AsyncEngine,
) -> None:
    """The sink survives pickling by shedding its unpicklable resources.

    Ray Tune serializes ``RunConfig.callbacks`` — and therefore the sink each
    callback wraps — into the experiment's ``tuner.pkl`` when the ``Tuner`` is
    built. The event loop and the async engine behind the session factory both
    hold unpicklable resources (thread locks, weakrefs), so ``__getstate__``
    keeps only ``job_id``. The live sink is untouched — pickling never happens on
    the trial write path — and the hollow shell is only ever read back by
    ``Tuner.restore``, which the local runner never calls.
    """
    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    loop = asyncio.get_running_loop()
    job_id = uuid4()
    sink = DbTrialSink(session_factory=factory, loop=loop, job_id=job_id)

    restored = pickle.loads(pickle.dumps(sink))

    assert sink._loop is loop  # the live sink keeps its resources
    assert sink._session_factory is factory
    assert restored._job_id == job_id  # the shell preserves job attribution
    assert not hasattr(restored, "_loop")
    assert not hasattr(restored, "_session_factory")


def test_coerce_trial_id_bounds_length() -> None:
    coerced = DbTrialSink.coerce_trial_id("x" * 40)

    assert len(coerced) <= 16


def test_coerce_trial_id_leaves_short_ids_unchanged() -> None:
    coerced = DbTrialSink.coerce_trial_id("ray_abcdef0001")

    assert coerced == "ray_abcdef0001"
