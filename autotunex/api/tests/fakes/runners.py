# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

from services.impl.runner import Runner


class FakeRunner(Runner):
    """In-memory runner double for tests. Records that run() was invoked."""

    def __init__(self, job_id, run_config, db=None, logging_handler=None):
        super().__init__(job_id, run_config)
        self.db = db
        self.logging_handler = logging_handler
        self.ran = False
        self.cancelled = False

    async def run(self):
        self.ran = True

    async def cancel(self):
        self.cancelled = True

    def supports_remote_cancel(self):
        return True
