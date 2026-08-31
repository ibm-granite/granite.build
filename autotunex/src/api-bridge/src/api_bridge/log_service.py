# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import math
import uuid
from typing import Any

from fastapi import HTTPException

from api_bridge import database
from api_bridge import model as bridge_models

logger = logging.getLogger(__name__)


def _sanitize_nan(obj):
    """Recursively replace NaN/Infinity floats with None so the result is JSON-serializable."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nan(v) for v in obj]
    return obj


class LogService:
    def __init__(self, db: database.Database):
        self.db = db

    async def record_logs(self, logs: list[bridge_models.LogEntry]):
        """
        Record log entries.

        Args:
            logs: list of LogRecord objects
        """
        try:
            result = self.db.insert_logs(logs)
            if result:
                return {"message": "logs inserted", "success": True}
            else:
                return {"message": "failed to insert logs", "success": False}
        except Exception as e:
            logger.error("Failed to insert logs", exc_info=e)
            raise HTTPException(status_code=400, detail=f"Something went wrong: {e}")

    async def insert_trial(self, data: bridge_models.Trial) -> str:
        try:
            result = self.db.insert_trial(data=data)
            return result
        except Exception as e:
            logger.error("Failed to insert trial", exc_info=e)
            raise HTTPException(status_code=400, detail=f"Something went wrong: {e}")

    def update_job_status(self, id: str, status: bridge_models.JobStatus) -> bool:
        self.db.update_job_status(id=id, status=status)
        if status == "TERMINATED" or status == "ERROR":
            return self.db.update_all_trial_status(job_id=id, status=status)

    def update_trial_status(self, id: str, status: bridge_models.JobStatus) -> bool:
        self.db.update_trial_status(trial_id=id, status=status)

    def is_valid_uuid(self, s: str) -> bool:
        try:
            # Try to create a UUID object from the string
            uuid.UUID(s)
            return True
        except ValueError:
            return False

    async def status_updates(self, data: bridge_models.UpdateStatus):
        try:
            if data.id and self.is_valid_uuid(data.id):
                self.update_job_status(id=data.id, status=data.status)
            elif data.id and not self.is_valid_uuid(data.id):
                self.update_trial_status(id=data.id, status=data.status)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(
                status_code=400,
                detail=f"Error occured while updating job/trial status: {e!s}",
            )

    def parse_result(self, data):
        if data.get("metric") == "loss":
            result = {
                "loss": (
                    None
                    if data.get("loss") is None or math.isnan(data.get("loss"))
                    else data.get("loss")
                ),
                "train_loss": (
                    None
                    if data.get("train_loss") is None or math.isnan(data.get("train_loss"))
                    else data.get("train_loss")
                ),
                "total_time": (
                    None
                    if data.get("time_total_s") is None or math.isnan(data.get("time_total_s"))
                    else data.get("time_total_s")
                ),
            }
            logger.debug(result)
        else:
            return json.dumps({"error": "Unsupported metric"}, indent=4)

        return result

    async def insert_trial_results(self, id: str, result: Any):
        try:
            result["job_id"] = id
            result["metric"] = "loss"
            result["metrics"] = self.parse_result(result)
            logger.info(f"Parsed result: {result}")
            response = self.db.insert_result(metadata=result)
            return _sanitize_nan(response)
        except Exception as e:
            logger.error("Failed to insert result", exc_info=e)
            raise HTTPException(status_code=400, detail=f"something went wrong: {e}")
