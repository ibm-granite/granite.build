# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Job service for api-bridge (minimal wrapper)."""

from api_bridge import database, models


class Job:
    def __init__(self, db: database.Database):
        self.db = db

    def get_by_id(self, job_id: str):
        """Return a job row by id, or None."""
        return self.db.get_job_by_id(job_id)

    def create(self, config: models.TuningConfig) -> str:
        """Insert a job row and return its id (str)."""
        return str(self.db.insert_job(config))
