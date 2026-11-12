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
