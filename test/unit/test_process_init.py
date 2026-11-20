#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""ensure_process_initialized() honors the same env vars as the click root group."""

from gbserver import cli


def test_worker_init_reads_click_env_vars(monkeypatch, tmp_path):
    log_file = tmp_path / "logs" / "gbserver.txt"
    monkeypatch.setenv("GBSERVER_LOG_FILE", str(log_file))
    monkeypatch.setenv("GBSERVER_LOG_LEVEL", "debug")
    calls = {}
    monkeypatch.setattr(cli, "init_process", lambda **kw: calls.update(kw) or None)
    monkeypatch.setattr(cli, "process_initialized", False)
    cli.ensure_process_initialized()
    assert calls["log_file"] == str(log_file)
    assert calls["log_level"] == "debug"


def test_worker_init_skipped_when_already_initialized(monkeypatch):
    monkeypatch.setattr(cli, "process_initialized", True)
    monkeypatch.setattr(
        cli, "init_process", lambda **kw: (_ for _ in ()).throw(AssertionError)
    )
    cli.ensure_process_initialized()


def test_root_api_startup_runs_process_init(monkeypatch):
    """Serving root_api directly still initializes the process on startup."""
    import asyncio

    from gbserver.api import root_api as root_api_mod

    calls = []
    monkeypatch.setattr(cli, "process_initialized", False)
    monkeypatch.setattr(cli, "init_process", lambda **kw: calls.append(kw))
    monkeypatch.setattr(root_api_mod, "_ANALYTICS_ENABLED", False)
    monkeypatch.setattr(root_api_mod, "GBSERVER_EVENT_PUBLISHING_ENABLED", False)
    asyncio.run(root_api_mod._start_background_tasks())
    assert len(calls) == 1
