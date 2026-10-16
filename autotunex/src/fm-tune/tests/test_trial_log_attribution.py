"""Regression tests for per-trial log attribution under concurrent trials.

The driver process funnels *every* trial's output through one root-logger
`BufferedLogHandler`: Ray runs with `log_to_driver=True`, so worker stdout is
forwarded into the driver, where `main.py` replaces `sys.stdout` with
`PrintLogger` -> `logging`. If the handler carries a single mutable `trial_id`
that Tune callbacks overwrite, those forwarded lines are stamped with whichever
trial fired a callback most recently -- trial A's logs land under trial B.
"""

import json
import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

from autotune.callbacks.logging_service import BufferedLogHandler
from autotune.callbacks.tuner_callback import CustomLoggerCallback


def _trial(tid):
    t = MagicMock()
    t.trial_id = tid
    t.config = {"lr": 0.1}
    t.trainable_name = "trainer"
    t.status = "RUNNING"
    return t


@pytest.fixture
def driver():
    """Mirror the driver process: one BufferedLogHandler on the root logger."""
    db = MagicMock()
    handler = BufferedLogHandler(job_id="job-1", db=db, buffer_size=10_000)
    handler.record_data = MagicMock()  # never touch the network in tests
    root = logging.getLogger()
    root.addHandler(handler)
    prev_level = root.level
    root.setLevel(logging.INFO)
    # Stands in for main.py's PrintLogger(logger): Ray's log-forwarding thread
    # writes worker stdout here, and it propagates to root.
    forwarded = logging.getLogger("autotune.forwarded")
    cb = CustomLoggerCallback(job_id="job-1", handler=handler)
    try:
        yield db, handler, forwarded, cb
    finally:
        root.removeHandler(handler)
        root.setLevel(prev_level)


def _rows(handler, db):
    """Every log entry the handler produced, buffered or already flushed."""
    flushed = [r for c in db.insert_logs.call_args_list for r in (c.kwargs.get("buffer") or [])]
    return flushed + list(handler.buffer)


def _find(rows, needle):
    return [r for r in rows if needle in r["message"]]


class TestTrialAttribution:
    def test_forwarded_worker_log_is_not_stamped_with_a_stale_trial_id(self, driver):
        db, handler, forwarded, cb = driver
        A = _trial("trial-AAAA")

        cb.on_trial_start(iteration=0, trials=[A], trial=A)
        forwarded.info("cluster resources: {'CPU': 8}")  # job-level, no trial owner

        rows = _find(_rows(handler, db), "cluster resources")
        assert rows, "expected the job-level record to be buffered"
        assert rows[0]["trial_id"] is None, (
            f"job-level log was misattributed to {rows[0]['trial_id']!r}; the handler "
            "leaked the last trial id set by a Tune callback"
        )

    def test_two_concurrent_trials_do_not_cross_label(self, driver):
        # Both trials are RUNNING; Tune interleaves their callbacks while Ray's
        # log-forwarding thread pushes each trial's stdout into the driver.
        db, handler, forwarded, cb = driver
        A, B = _trial("trial-AAAA"), _trial("trial-BBBB")

        cb.on_trial_start(iteration=0, trials=[A, B], trial=A)
        cb.on_trial_start(iteration=0, trials=[A, B], trial=B)
        forwarded.info("[trial-AAAA] step 1 loss=0.53")
        cb.on_trial_result(iteration=1, trials=[A, B], trial=A, result={"loss": 0.5, "config": {}})
        forwarded.info("[trial-BBBB] step 2 loss=0.41")

        # A forwarded line naming trial X must never be stored under trial Y.
        leaked = [
            (r["message"], r["trial_id"])
            for r in _rows(handler, db)
            if r["message"].startswith("[trial-")
            and r["trial_id"] is not None
            and f"[{r['trial_id']}]" not in r["message"]
        ]
        assert not leaked, f"records labelled with the wrong trial: {leaked}"

    def test_callback_logs_keep_their_own_trial_id(self, driver):
        # The fix must not throw away correct attribution: a callback's own log
        # lines still belong to that callback's trial.
        db, handler, forwarded, cb = driver
        A = _trial("trial-AAAA")
        cb.on_trial_start(iteration=0, trials=[A], trial=A)

        rows = _find(_rows(handler, db), "Trial_trial-AAAA Started")
        assert rows, "expected the callback's own start log to be recorded"
        assert rows[0]["trial_id"] == "trial-AAAA"

    def test_concurrent_callbacks_on_two_threads_stay_isolated(self, driver):
        # Two threads each log "inside" a different trial. A process-global
        # trial_id cannot keep these apart; a context-scoped one can.
        db, handler, forwarded, cb = driver
        start = threading.Barrier(2)

        def run(tid, msg):
            trial = _trial(tid)
            start.wait(5)
            cb.on_trial_start(iteration=0, trials=[trial], trial=trial)
            forwarded.info(msg, extra={"trial_id": tid})

        threads = [
            threading.Thread(target=run, args=("trial-AAAA", "own-line-A")),
            threading.Thread(target=run, args=("trial-BBBB", "own-line-B")),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)

        rows = _rows(handler, db)
        for tid, msg in (("trial-AAAA", "own-line-A"), ("trial-BBBB", "own-line-B")):
            found = _find(rows, msg)
            assert found, f"{msg} was not recorded"
            assert found[0]["trial_id"] == tid, f"{msg} was stored under {found[0]['trial_id']!r}, expected {tid!r}"


class TestConcurrentBufferSafety:
    def test_records_emitted_during_a_slow_flush_are_not_dropped(self):
        # flush() posts the buffer, then clears it. Anything a concurrent
        # emit() appended during that (blocking, un-timed-out) POST must
        # survive rather than be discarded.
        db = MagicMock()
        handler = BufferedLogHandler(job_id="job-1", db=db, buffer_size=10_000)
        entered = threading.Event()
        release = threading.Event()

        def slow_insert(buffer=None):
            entered.set()
            release.wait(5)

        db.insert_logs.side_effect = slow_insert
        handler.emit(logging.LogRecord("t", logging.INFO, __file__, 1, "first", None, None))

        t = threading.Thread(target=handler.flush)
        t.start()
        assert entered.wait(5), "flush never reached the destination"
        handler.emit(logging.LogRecord("t", logging.INFO, __file__, 1, "during-flush", None, None))
        release.set()
        t.join(5)

        flushed = [r["message"] for c in db.insert_logs.call_args_list for r in (c.kwargs.get("buffer") or [])]
        remaining = [r["message"] for r in handler.buffer]
        assert "during-flush" in remaining + flushed, "a record emitted during flush was discarded by the buffer reset"


class TestWorkerSideAttribution:
    """A trial worker process owns exactly one trial, so it labels at the source."""

    def _worker_handler(self):
        db = MagicMock()
        handler = BufferedLogHandler(job_id="job-1", db=db, buffer_size=10_000)
        log = logging.getLogger()
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        return db, handler, log

    def test_bind_trial_id_labels_records_from_any_thread(self):
        from autotune.logging_setup import bind_trial_id

        db, handler, log = self._worker_handler()
        try:
            bind_trial_id("trial-CCCC")
            # HF Trainer / dataloader threads log too; a context-scoped id would
            # miss them, so the worker binds the handler default instead.
            t = threading.Thread(target=lambda: log.info("worker-thread-line"))
            t.start()
            t.join(5)
            log.info("worker-main-line")

            rows = _rows(handler, db)
            for msg in ("worker-thread-line", "worker-main-line"):
                found = _find(rows, msg)
                assert found, f"{msg} not recorded"
                assert found[0]["trial_id"] == "trial-CCCC", f"{msg} stored under {found[0]['trial_id']!r}"
        finally:
            log.removeHandler(handler)

    def test_bind_trial_id_is_a_noop_without_a_trial(self):
        from autotune.logging_setup import bind_trial_id

        db, handler, log = self._worker_handler()
        try:
            bind_trial_id(None)
            assert handler.trial_id is None
        finally:
            log.removeHandler(handler)


class TestBridgeRuntimeEnv:
    """Worker processes on other nodes need the bridge settings passed to them."""

    def test_empty_when_bridge_is_off(self, monkeypatch):
        from autotune.logging_setup import bridge_runtime_env

        monkeypatch.delenv("AUTOTUNE_JOB_ID", raising=False)
        monkeypatch.delenv("AUTOTUNE_ENDPOINT_URL", raising=False)
        # Must stay empty so ray.init() is unchanged for the default (bridge-off) run.
        assert bridge_runtime_env() == {}

    def test_carries_bridge_settings_when_on(self, monkeypatch):
        from autotune.logging_setup import bridge_runtime_env

        monkeypatch.setenv("AUTOTUNE_JOB_ID", "job-1")
        monkeypatch.setenv("AUTOTUNE_ENDPOINT_URL", "http://bridge/fmtune/api")
        assert bridge_runtime_env() == {
            "runtime_env": {
                "env_vars": {
                    "AUTOTUNE_JOB_ID": "job-1",
                    "AUTOTUNE_ENDPOINT_URL": "http://bridge/fmtune/api",
                }
            }
        }


class TestHttpFlushPayload:
    """The HTTP path is what the AutotuneX api-bridge actually uses.

    `flush()` detaches the buffer into a local `batch` before sending, so the
    send helpers must serialize *that batch*. A helper that reads `self.buffer`
    instead posts the already-emptied buffer — an empty array — and silently
    drops every log line. The DB path uses a mocked `db`, so only an assertion
    on the POST body catches this.
    """

    def _handler(self):
        return BufferedLogHandler(job_id="job-1", endpoint_url="http://bridge/fmtune/api", buffer_size=10_000)

    def _emit(self, handler, *messages):
        for m in messages:
            handler.emit(logging.LogRecord("t", logging.INFO, __file__, 1, m, None, None))

    def test_flush_posts_every_buffered_record(self):
        handler = self._handler()
        self._emit(handler, "line-0", "line-1", "line-2")

        with patch("autotune.callbacks.logging_service.requests.post") as post:
            post.return_value = MagicMock(raise_for_status=MagicMock())
            handler.flush()

        assert post.call_count == 1, "expected exactly one POST to /record_logs"
        sent = json.loads(post.call_args.kwargs["data"])
        assert [r["message"] for r in sent] == ["line-0", "line-1", "line-2"], (
            f"POST body carried {len(sent)} record(s); the send helper is not serializing the batch flush() detached"
        )

    def test_record_emitted_during_a_slow_post_is_sent_exactly_once(self):
        # Ray's log-forwarding thread keeps emitting while the POST is in flight.
        handler = self._handler()
        self._emit(handler, "before-post")
        entered, release = threading.Event(), threading.Event()
        bodies = []

        def slow_post(*a, **kw):
            bodies.append(json.loads(kw["data"]))
            entered.set()
            release.wait(5)
            return MagicMock(raise_for_status=MagicMock())

        with patch("autotune.callbacks.logging_service.requests.post", side_effect=slow_post):
            t = threading.Thread(target=handler.flush)
            t.start()
            assert entered.wait(5), "flush never reached the endpoint"
            self._emit(handler, "during-post")
            release.set()
            t.join(5)

            assert [r["message"] for r in bodies[0]] == ["before-post"]
            handler.flush()  # the concurrent record must still be pending

        delivered = [r["message"] for body in bodies for r in body]
        assert delivered.count("during-post") == 1, (
            f"'during-post' delivered {delivered.count('during-post')} time(s); "
            "expected exactly once (0 = dropped by the buffer reset, 2 = resent)"
        )
