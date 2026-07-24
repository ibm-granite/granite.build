# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import models


class Runner(ABC):
    def __init__(self, job_id: str, run_config: models.TuningConfig):
        if not job_id:
            raise RuntimeError("job_id cannot be None")

        if not run_config:
            raise RuntimeError("run_config cannot be None")
        self.job_id = job_id
        self.run_config = run_config

    @abstractmethod
    def run(self):
        pass

    def supports_remote_cancel(self) -> bool:
        """Whether cancelling requires a remote call. Local runners: no."""
        return False

    async def cancel(self) -> None:
        """Cancel a running job. Default (local) behavior is a no-op;
        remote runners (e.g. Granite Build) override this."""
        return None
