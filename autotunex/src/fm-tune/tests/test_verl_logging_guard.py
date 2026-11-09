"""Pins driver_multi_verl's defence against dependencies silencing the trial's logging."""

import logging
import os

import pytest

pytest.importorskip("verl")


@pytest.fixture
def root_state():
    level, handlers = logging.root.level, [(h, h.level) for h in logging.root.handlers]
    yield
    logging.disable(logging.NOTSET)
    logging.root.setLevel(level)
    for h, lvl in handlers:
        h.setLevel(lvl)


def test_driver_sets_sagemaker_log_level_default():
    import autotune.trainers.driver_multi_verl  # noqa: F401  (sets it at import)

    assert os.environ.get("SAGEMAKER_CONTAINER_LOG_LEVEL")


def test_sagemaker_root_logger_respects_the_default(root_state, monkeypatch):
    """The dependency that silenced every RL trial log: ERROR unless the env var says otherwise."""
    mhcs = pytest.importorskip("model_hosting_container_standards.logging_config")
    logging.root.setLevel(logging.INFO)

    monkeypatch.delenv("SAGEMAKER_CONTAINER_LOG_LEVEL", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    mhcs.configure_root_logger()
    assert logging.root.level == logging.ERROR  # the upstream default

    logging.root.setLevel(logging.INFO)
    monkeypatch.setenv("SAGEMAKER_CONTAINER_LOG_LEVEL", "INFO")
    mhcs.configure_root_logger()
    assert logging.root.level == logging.INFO


def test_restore_logging_undoes_raised_levels(root_state, capsys):
    from autotune.trainers.driver_multi_verl import _restore_logging, _snapshot_logging

    handler = logging.StreamHandler()
    logging.root.addHandler(handler)
    try:
        logging.root.setLevel(logging.INFO)
        handler.setLevel(logging.NOTSET)
        snap = _snapshot_logging()
        logging.root.setLevel(logging.ERROR)
        handler.setLevel(logging.ERROR)
        _restore_logging(snap, "test")
        assert logging.root.level == logging.INFO
        assert handler.level == logging.NOTSET
        assert "raised the root logger" in capsys.readouterr().out
    finally:
        logging.root.removeHandler(handler)


def test_restore_logging_undoes_disable(root_state, capsys):
    from autotune.trainers.driver_multi_verl import _restore_logging, _snapshot_logging

    snap = _snapshot_logging()
    logging.disable(logging.INFO)
    _restore_logging(snap, "test")
    assert logging.root.manager.disable == 0
    assert "re-enabling" in capsys.readouterr().out


def test_restore_logging_is_quiet_when_nothing_changed(root_state, capsys):
    from autotune.trainers.driver_multi_verl import _restore_logging, _snapshot_logging

    _restore_logging(_snapshot_logging(), "test")
    assert capsys.readouterr().out == ""
