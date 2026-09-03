# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class LogEntry(BaseModel):
    id: int | None = Field(None, description="Auto-incrementing primary key")
    job_id: str | None = Field(..., description="Job ID as a 36-character UUID")
    trial_id: str | None
    level: str | None = Field(None, description="Log level (e.g., INFO, WARNING, ERROR)")
    filename: str | None = Field(None, description="Name of the file where the log was generated")
    message: str | None = Field(None, description="Log message")
    iteration: int | None = Field(None, description="Iteration number")
    epoch: float | None = Field(None, description="Epoch number")
    timestamp: datetime | None = Field(None, description="Timestamp of the log entry")

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class TrainingMetric(BaseModel):
    # Non-optional: training_metrics.job_id is NOT NULL, so a null here would pass
    # validation and then fail the INSERT as a caught IntegrityError — reported to
    # the caller as HTTP 200 {"success": false}. Reject it at the edge instead.
    job_id: str = Field(..., description="Job ID as a 36-character UUID")
    trial_id: str | None = None
    global_step: int = Field(..., description="Global training step")
    epoch: float | None = None
    loss: float | None = None
    grad_norm: float | None = None
    learning_rate: float | None = None
    split: str = Field("train", description="'train' or 'eval'")
    extra: dict[str, Any] | None = None

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    TERMINATED = "TERMINATED"
    ERROR = "ERROR"
    COMPLETED = "COMPLETED"


class UpdateStatus(BaseModel):
    id: str
    status: JobStatus | None = JobStatus.PENDING


class TrialStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    TERMINATED = "TERMINATED"
    ERROR = "ERROR"
    COMPLETED = "COMPLETED"


class Trial(BaseModel):
    id: str
    job_id: UUID
    status: TrialStatus
    config: dict[str, Any] | None

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class Result(BaseModel):
    id: UUID | None = None
    job_id: UUID
    trial_id: str
    metric: str
    loss: float
    train_loss: float
    eval_loss: float
    total_time: float | None
    time_total_s: float | None

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]


class BootstrapConfig(BaseModel):
    name: str
    tuner_type: str
    rl_tuner_type: str | None = None
    config_data: dict[str, Any]


class BootstrapDataset(BaseModel):
    name: str
    artifact_uri: str


class BootstrapJob(BaseModel):
    model: str
    experiment_name: str
    tuning_type: str | None = None
    model_source: str = "huggingface"
    seed: int = 42


class BootstrapRequest(BaseModel):
    job_id: str
    build_id: str | None = None
    config: BootstrapConfig
    dataset: BootstrapDataset
    job: BootstrapJob
