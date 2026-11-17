#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0


"""uvicorn app factory for the REST server.

Kept apart from root_api so the process is initialized (logging, standalone
storage, retries) *before* root_api's module body runs and logs, and so
importing root_api itself has no side effects (tests build a TestClient from it
and must keep the storage factory they chose).
"""

from fastapi import FastAPI


def create_app() -> FastAPI:
    """uvicorn calls this once per worker process (``factory=True``).

    This is the only supported way to serve the REST API: importing root_api on
    its own does not initialize the process.
    """
    from gbserver import cli

    # Spawned workers run the init; a process that already ran the CLI keeps the
    # user's --log-level/--log-file.
    cli.ensure_process_initialized()
    from gbserver.api.root_api import root_api

    return root_api
