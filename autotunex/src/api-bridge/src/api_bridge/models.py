# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Pydantic models for api-bridge CRUD endpoints.

These are local copies of the models from api/models.py, kept here so that
api-bridge is fully self-contained and does not depend on the api package.
"""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PipelineType(str, Enum):
    PREFIX = "prefix"
    PEFT = "peft"
    SFT = "sft"


class TuningType(str, Enum):
    ALORA = "alora"
    LOHA = "loha"
    LOKR = "lokr"
    LORA = "lora"
    P_TUNING = "p_tuning"
    PREFIX_TUNING = "prefix_tuning"
    PROMPT_TUNING = "prompt_tuning"
    SFT = "sft"
    VERA = "vera"


class Status(str, Enum):
    CREATED = "CREATED"
    ADDED = "ADDED"
    UPDATED = "UPDATED"
    DELETED = "DELETED"


class TrialStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    TERMINATED = "TERMINATED"
    ERROR = "ERROR"
    COMPLETED = "COMPLETED"


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    TERMINATED = "TERMINATED"
    ERROR = "ERROR"
    COMPLETED = "COMPLETED"


class Roles(str, Enum):
    ADMIN = "admin"
    USER = "user"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class TuningConfig(BaseModel):
    id: UUID | None = Field(None, description="Unique job identifier (auto-generated)")
    user_id: str | None = Field(None, description="User ID (auto-set from auth)")
    status: JobStatus | None = Field(JobStatus.PENDING, description="Current job status")
    seed: int | None = Field(42, description="Random seed for reproducibility")
    config_id: str = Field(..., description="Configuration ID to use for tuning")
    dataset_id: str = Field(..., description="Dataset ID for training")
    model: str = Field(
        ...,
        description="Foundation model name (e.g., 'meta-llama/Llama-2-7b-hf')",
        examples=["meta-llama/Llama-2-7b-hf", "mistralai/Mistral-7B-v0.1"],
    )
    experiment_name: str = Field(
        ..., description="Unique experiment name", examples=["my-chatbot-v1"]
    )
    tuning_type: TuningType | None = Field(
        None, description="Fine-tuning method (LORA, PREFIX_TUNING, etc.)"
    )
    ray_address: str | None = Field(
        None, description="Ray cluster address for distributed training"
    )
    cleanup: bool | None = Field(
        True, description="Clean up intermediate artifacts after completion"
    )
    save_history: bool | None = Field(True, description="Save training history and metrics")
    autotune: bool = Field(True, description="Enable autotuning of hyperparameters")
    build_id: str | None = Field(None, description="Build ID to associate with a gb_tasks record")

    class Config:
        json_schema_extra = {
            "example": {
                "config_id": "550e8400-e29b-41d4-a716-446655440000",
                "dataset_id": "660e8400-e29b-41d4-a716-446655440000",
                "model": "meta-llama/Llama-2-7b-hf",
                "experiment_name": "customer-support-bot",
                "tuning_type": "lora",
                "seed": 42,
            }
        }


class Response(BaseModel):
    id: UUID | None = Field(None, description="Resource ID if applicable")
    status: Status = Field(..., description="Operation status")
    message: str | None = Field(None, description="Human-readable status message")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "CREATED",
                "message": "Configuration created successfully",
            }
        }


class Configuration(BaseModel):
    id: UUID | None = Field(None, description="Unique configuration identifier")
    user_id: str | None = Field(None, description="Owner user ID")
    name: str = Field(
        ...,
        description="Configuration name",
        examples=["lora-config-1", "prefix-tuning-aggressive"],
    )
    tuner_type: str = Field(
        ...,
        description="HPO algorithm type",
        examples=["bayesian", "grid_search", "random_search"],
    )
    artifact_id: str | None = Field(None, description="Associated artifact identifier")
    artifact_url: str | None = Field(None, description="URL to configuration artifact")
    config_data: dict[str, Any] = Field(..., description="Hyperparameter search space definition")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "lora-bayesian-config",
                "tuner_type": "bayesian",
                "config_data": {
                    "learning_rate": {"min": 1e-5, "max": 1e-3, "log": True},
                    "lora_rank": {"values": [8, 16, 32, 64]},
                    "lora_alpha": {"min": 8, "max": 128},
                    "batch_size": {"values": [4, 8, 16]},
                },
            }
        }


class AuthUser(BaseModel):
    email: str
    role: Roles
    impersonating: str | None = None


class DatasetInfo(BaseModel):
    id: UUID | None = Field(None, description="Unique dataset identifier")
    user_id: str | None = Field(None, description="Owner user ID")
    name: str = Field(
        ...,
        description="Dataset name",
        examples=["customer-qa-pairs", "product-reviews"],
    )
    description: str = Field(
        ...,
        description="Dataset description and purpose",
        examples=["Customer support Q&A pairs for chatbot training"],
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "customer-support-qa",
                "description": "10K customer support conversations with responses",
            }
        }


class SimpleJobResponse(BaseModel):
    """Simplified JobResponse without build_status."""

    id: UUID | None = None
    experiment_name: str | None = None
    status: JobStatus | None = None
    model: str | None = None
    config_name: str | None = None
    dataset: str | None = None
    created_at: datetime
    updated_at: datetime


class SimpleConfiguration(BaseModel):
    """Simplified Configuration without config_data."""

    id: UUID | None = Field(None, description="Unique configuration identifier")
    name: str | None = Field(None, description="Configuration name")
    tuner_type: str | None = Field(None, description="HPO algorithm type")
    artifact_id: str | None = None
    artifact_url: str | None = None
    config_data: dict | None = None
    associated_jobs: list[SimpleJobResponse]
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @field_validator("config_data", mode="before")
    @classmethod
    def force_config_data_null(cls, v):
        return None


class DatasetResponse(DatasetInfo):
    """Dataset info with associated jobs."""

    artifact_id: str | None = None
    artifact_url: str | None = None
    associated_jobs: list[SimpleJobResponse]
    train_file: str | None = None
    train_records: int | None = None
    train_file_size: int | None = None
    validation_file: str | None = None
    validation_records: int | None = None
    validation_file_size: int | None = None
    created_at: datetime
    updated_at: datetime


class User(BaseModel):
    id: UUID | None = Field(None, description="Unique user identifier")
    email: str = Field(..., description="User email address")
    role: Roles = Field(..., description="role of the user")
    created_at: datetime | None = Field(None, description="Account creation timestamp")
    updated_at: datetime | None = Field(None, description="Last login timestamp")
    jobs: list[SimpleJobResponse] | None = None
    configs: list[SimpleConfiguration] | None = None
    datasets: list[DatasetResponse] | None = None
