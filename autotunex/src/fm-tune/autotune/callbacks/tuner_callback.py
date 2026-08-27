# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
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

import logging
from typing import Optional, Protocol

from ray.tune import Callback
from ray.tune.experiment.trial import Trial

from autotune.callbacks.logging_service import BufferedLogHandler, RecordType

logger = logging.getLogger(__name__)


class HandlerProtocol(Protocol):
    """Protocol defining the interface for database operations."""

    def get_job_id(self) -> str:
        """Get the job ID."""
        ...

    def set_trial_id(self, trial_id: str) -> None:
        """Set the trial ID."""
        ...

    def flush(self) -> str:
        """Get the job ID."""
        ...


class CustomLoggerCallback(Callback):
    def __init__(self, job_id=None, handler: Optional[BufferedLogHandler] = None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.job_id = job_id
        self.handler = handler

    def on_trial_start(self, iteration, trials, trial: Trial):
        if self.handler:
            self.handler.set_trial_id(trial.trial_id)
        data = {
            "id": trial.trial_id,
            "job_id": self.handler.get_job_id(),
            "status": "RUNNING",
            "config": self.sanitized_result({"config": trial.config}),
        }

        self.handler.record_data(data, record_type=RecordType.RECORD_TRIAL)

        self.logger.info(f"::::::::::::::: Trial_{trial.trial_id} Initialized :::::::::::::::\n")

        self.logger.info(f"trial_id: {trial.trial_id}")
        self.logger.info(f"iterations: {iteration}")
        self.logger.info(f"trial_fn: {trial.trainable_name}")
        self.logger.info(f"trial_status: {trial.status}\n")
        self.logger.info(">>>>>>>>>>>>> trial_config <<<<<<<<<<<<<\n")
        for key, value in trial.config.items():
            self.logger.info(f"{key}: {value}")
        self.logger.info(f"::::::::::::::: Trial_{trial.trial_id} Started :::::::::::::::\n")
        if self.handler:
            self.handler.flush()

    def on_trial_result(self, iteration, trials, trial, result, **info):
        if self.handler:
            self.handler.set_trial_id(trial.trial_id)
        self.logger.info(f"--------- Trial_{trial.trial_id} Result Start -----------")
        self.logger.info(f"trial_id: {trial.trial_id}")
        self.logger.info(f"iterations: {iteration}")
        self.logger.info(f"trial_fn: {trial.trainable_name}")
        self.logger.info(f"trial_config: {trial.config}")
        self.logger.info(f"trial_status: {trial.status}")
        self.logger.info(f"Trial {trial.trial_id} reported result: {result}")
        self.logger.info(f"--------- Result for Trial_{trial.trial_id} End -----------")
        self.logger.info(f"......... TRIAL_JOB_ID.....{self.job_id}.............")
        # job.insert_trial_results(id=self.job_id, result=result)
        result = {
            **result,
            "id": self.handler.get_job_id(),
            "job_id": self.handler.get_job_id(),
            "config": self.sanitized_result(result),
        }
        print("sanitized results: ", result)
        self.handler.record_data(result, record_type=RecordType.RECORD_RESULT)
        status = {"id": trial.trial_id, "status": "COMPLETED"}
        self.handler.record_data(status, record_type=RecordType.UPDATE_STATUS)

        if self.handler:
            self.handler.set_trial_id(None)
            self.handler.flush()

    def on_trial_complete(self, iteration, trials, trial, **info):
        if self.handler:
            self.handler.set_trial_id(trial.trial_id)

        self.logger.info(f"--------- Trial_{trial.trial_id} Completed -----------")
        if self.handler:
            self.handler.set_trial_id(None)
            self.handler.flush()

    def on_trial_error(self, iteration, trials, trial: Trial, **info):
        if self.handler:
            self.handler.set_trial_id(trial.trial_id)
        self.logger.error(f"Error occured during trial_{trial.trial_id} execution")
        status = {"id": trial.trial_id, "status": "ERROR"}
        self.handler.record_data(status, record_type=RecordType.UPDATE_STATUS)
        if self.handler:
            self.handler.set_trial_id(None)
            self.handler.flush()

    def sanitized_result(self, data):
        config_copy = data["config"].copy() if isinstance(data["config"], dict) else {}
        # Remove or convert non-serializable objects.  ``tune_config`` may
        # carry live instances of the search algorithm (e.g. BLDS / Hyperopt)
        # and the trial scheduler (e.g. AsyncHyperBandScheduler when
        # scheduler='asha'); both are not JSON-serializable, so replace them
        # with their class names before the buffered log handler tries to
        # ``json.dumps`` the trial record.
        tc = config_copy.get("tune_config")
        if isinstance(tc, dict):
            tc_copy = tc.copy()
            for key in ("search_alg", "scheduler"):
                if key in tc_copy and not isinstance(tc_copy[key], (str, int, float, bool, type(None))):
                    tc_copy[key] = tc_copy[key].__class__.__name__
            config_copy["tune_config"] = tc_copy
        return config_copy
