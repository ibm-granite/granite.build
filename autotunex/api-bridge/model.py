# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class LogEntry(BaseModel):
    id: Optional[int] = Field(None, description="Auto-incrementing primary key")
    job_id: Optional[str] = Field(..., description="Job ID as a 36-character UUID")
    trial_id: Optional[str]
    level: Optional[str] = Field(
        None, description="Log level (e.g., INFO, WARNING, ERROR)"
    )
    filename: Optional[str] = Field(
        None, description="Name of the file where the log was generated"
    )
    message: Optional[str] = Field(None, description="Log message")
    iteration: Optional[int] = Field(None, description="Iteration number")
    epoch: Optional[float] = Field(None, description="Epoch number")
    timestamp: Optional[datetime] = Field(
        None, description="Timestamp of the log entry"
    )

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
    status: Optional[JobStatus] = JobStatus.PENDING


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
    config: Optional[Dict[str, Any]]

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


class Result(BaseModel):
    id: Optional[UUID] = None
    job_id: UUID
    trial_id: str
    metric: str
    loss: float
    train_loss: float
    eval_loss: float
    total_time: Optional[float]
    time_total_s: Optional[float]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]


class BootstrapConfig(BaseModel):
    name: str
    tuner_type: str
    rl_tuner_type: Optional[str] = None
    config_data: Dict[str, Any]


class BootstrapDataset(BaseModel):
    name: str
    artifact_uri: str


class BootstrapJob(BaseModel):
    model: str
    experiment_name: str
    tuning_type: Optional[str] = None
    model_source: str = "huggingface"
    seed: int = 42


class BootstrapRequest(BaseModel):
    job_id: str
    build_id: Optional[str] = None
    config: BootstrapConfig
    dataset: BootstrapDataset
    job: BootstrapJob
