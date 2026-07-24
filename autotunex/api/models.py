# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel, Field, root_validator, field_validator
from typing import List, Optional, Union, Any, Dict
from uuid import UUID
from datetime import datetime
from enum import Enum
from utils import utc_now_string
from constants import RITS_TTL


class PipelineType(str, Enum):
    PREFIX = "prefix"
    PEFT = "peft"
    SFT = "sft"


class ModelSource(str, Enum):
    HUGGINGFACE = "huggingface"
    DMF = "dmf"
    CUSTOM_PATH = "custom_path"


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


class JobStats(BaseModel):
    total: int
    pending: int
    running: int
    paused: int
    terminated: int
    error: int
    completed: int


class TuningConfig(BaseModel):
    id: Optional[UUID] = Field(
        None, description="Unique job identifier (auto-generated)"
    )
    user_id: Optional[str] = Field(None, description="User ID (auto-set from auth)")
    status: Optional[JobStatus] = Field(
        JobStatus.PENDING, description="Current job status"
    )
    seed: Optional[int] = Field(42, description="Random seed for reproducibility")
    config_id: str = Field(..., description="Configuration ID to use for tuning")
    dataset_id: str = Field(..., description="Dataset ID for training")
    model: str = Field(
        ...,
        description="Model identifier (HF: 'org/model', DMF: 'lh://prod/...' or UUID)",
        examples=[
            "meta-llama/Llama-2-7b-hf",
            "mistralai/Mistral-7B-v0.1",
            "lh://prod/granite_dot_build.public/models/model_shared/granite-2b-base/20250319T181102",
        ],
    )
    model_source: ModelSource = Field(
        ModelSource.HUGGINGFACE, description="Source of the model (huggingface or dmf)"
    )
    experiment_name: str = Field(
        ..., description="Unique experiment name", examples=["my-chatbot-v1"]
    )
    tuning_type: Optional[TuningType] = Field(
        None, description="Fine-tuning method (LORA, PREFIX_TUNING, etc.)"
    )
    ray_address: Optional[str] = Field(
        None, description="Ray cluster address for distributed training"
    )
    cleanup: Optional[bool] = Field(
        True, description="Clean up intermediate artifacts after completion"
    )
    save_history: Optional[bool] = Field(
        True, description="Save training history and metrics"
    )
    autotune: bool = Field(True, description="Enable autotuning of hyperparameters")
    additional_info: Optional[dict] = Field(
        None,
        description="Additional metadata for DMF models (namespace, base_model, revision, etc.)",
    )
    reward_function_code: Optional[str] = Field(
        None,
        description="Python code for the reward function (Online RL only). Injected into build.yaml for GB runner.",
    )
    reward_function_name: Optional[str] = Field(
        None, description="Name of the reward function to call (default: compute_score)"
    )

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
    id: Optional[UUID] = Field(None, description="Resource ID if applicable")
    status: Status = Field(..., description="Operation status")
    message: Optional[str] = Field(None, description="Human-readable status message")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "CREATED",
                "message": "Configuration created successfully",
            }
        }


class Roles(str, Enum):
    ADMIN = "admin"
    USER = "user"


class AuthUser(BaseModel):
    email: str
    role: Roles
    impersonating: Optional[str] = None


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


class UserMetadata(BaseModel):
    number_of_jobs: Optional[int] = Field(None, description="number of jobs")
    number_of_configurations: Optional[int] = Field(
        None, description="number of configurations including system configurations"
    )
    number_of_datasets: Optional[int] = Field(None, description="number of datasets")


class DmfMetadata(BaseModel):
    label: str = Field(
        ...,
        description="Human-readable model label/name",
        examples=["customer-support-v1", "finance-qa-bot"],
    )
    variant: str = Field(
        ...,
        description="Model variant identifier",
        examples=["lora-r16", "prefix-tuned", "full-finetuned"],
    )
    type: str = Field(
        ...,
        description="Model task type",
        examples=["text-generation", "classification", "question-answering"],
    )
    size: str = Field(
        ..., description="Model size", examples=["7B", "13B", "70B", "1.3B"]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "label": "customer-support-chatbot-v2",
                "variant": "lora-rank-16",
                "type": "text-generation",
                "size": "7B",
            }
        }


class DatasetInfo(BaseModel):
    id: Optional[UUID] = Field(None, description="Unique dataset identifier")
    user_id: Optional[str] = Field(None, description="Owner user ID")
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


class Trial(BaseModel):
    id: str
    job_id: UUID
    status: TrialStatus
    config: Optional[Dict[str, Any]]

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)


# Below are old pydantic classes for creating detailed configurations.
# For now we will be directly storing the JSON in mysql table to keep it simple
class Config(BaseModel):
    id: Optional[UUID] = Field(default_factory=UUID)
    user_id: Optional[str] = None
    name: str
    tuner_type: str
    artifact_id: Optional[str] = None
    artifact_url: Optional[str] = None
    config_data: Dict[str, Any]
    associated_jobs: List[TuningConfig]
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class SystemConfig(BaseModel):
    id: Optional[str] = None
    config_id: Optional[UUID]
    num_cpus_per_worker: int
    num_gpus_per_worker: int


class TuneConfig(BaseModel):
    id: Optional[str] = None
    config_id: Optional[UUID]
    mode: str
    metric: str
    search_alg: str
    scheduler: str
    num_samples: int
    max_concurrent_trials: int
    max_discrepancy: int


class TrainConfig(BaseModel):
    id: Optional[str] = None
    config_id: Optional[UUID]
    num_epochs: int
    max_train_steps: int
    num_warmup_steps: int
    seed: int
    training_iteration: int
    precision: str
    use_flash_attn: bool = False
    use_gradient_chkpt: bool = False


class PreprocessConfig(BaseModel):
    id: Optional[str] = None
    config_id: Optional[UUID]
    pad_to_max_length: bool = False
    use_slow_tokenizer: bool = False
    input_sequence: str
    output_sequence: str


class AdapterTunerConfig(BaseModel):
    id: Optional[str] = None
    parameter_name: Optional[str]
    strategy: Optional[str] = "choice"
    values: List[Union[str, int, float]]  # This will handle the JSON field
    default_value: Union[str, int, float]


class Hyperparameters(BaseModel):
    config_id: Optional[UUID]
    parameters: Dict[str, List[AdapterTunerConfig]]


# Complete configuration with relationships
class Configurations(Config):
    system_config: Optional[SystemConfig] = None
    tune_config: Optional[TuneConfig] = None
    train_config: Optional[TrainConfig] = None
    preprocess_config: Optional[PreprocessConfig] = None
    hyperparameters: Optional[Hyperparameters] = None

    @root_validator(pre=True)
    def assign_config_id(cls, values):
        # Use the 'id' field of Config to assign to all `config_id` fields
        config_id = values.get("id")
        if config_id:
            if values.get("system_config"):
                values["system_config"]["config_id"] = config_id
            if values.get("tune_config"):
                values["tune_config"]["config_id"] = config_id
            if values.get("train_config"):
                values["train_config"]["config_id"] = config_id
            if values.get("preprocess_config"):
                values["preprocess_config"]["config_id"] = config_id
            if values.get("hyperparameters"):
                values["hyperparameters"]["config_id"] = config_id
        return values


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


class F1_Score(BaseModel):
    id: Optional[UUID] = None
    result_id: UUID
    accuracy: float
    f1_micro: float
    f1_macro: float
    f1_weighted: float


class Rouge_Score(BaseModel):
    id: Optional[UUID] = None
    result_id: UUID
    rouge1: float
    rouge2: float
    rougeL: float
    rougeLsum: float


class FileInfo(BaseModel):
    path: str
    file_size: int
    file_hash: str
    created: datetime


class ModelInfo(BaseModel):
    model_id: Optional[str] = None
    user: Optional[str]
    model_label: str
    base_model: str
    size: Optional[str] = None
    revision: str
    open: bool
    product_name: str
    files: Optional[List[FileInfo]] = None
    dmf_url: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class UpdateStatus(BaseModel):
    id: str
    status: Optional[JobStatus] = JobStatus.PENDING


class TaskType(str, Enum):
    RITS = "RITS"
    TUNING = "TUNING"
    DOWNLOAD = "DOWNLOAD"


class Task(BaseModel):
    id: Optional[UUID] = None
    job_id: Optional[str]
    build_id: Optional[str] = None
    status: Optional[JobStatus] = JobStatus.PENDING
    type: TaskType
    pr_url: Optional[str] = None
    artifact_id: Optional[str] = None
    artifact_uri: Optional[str] = None
    build_status: Union[Dict[str, Any], None] = None
    started_at: Optional[str] = Field(default_factory=utc_now_string)
    updated_at: Optional[str] = Field(default_factory=utc_now_string)
    rits_url: Optional[str] = None


class JobResponse(TuningConfig):
    id: Optional[UUID] = None
    num_trials: int = 0
    config_name: str
    rl_tuning_type: Optional[str] = None
    logs: Optional[List[LogEntry]] = None
    gb_logs: Optional[Dict[str, Any]] = None
    build_status: Optional[Dict[str, Any]] = None
    task: Optional[Task] = None
    created_at: datetime
    updated_at: datetime


class PushToRits(BaseModel):
    job_id: UUID
    suffix: Optional[str] = "autotunex"
    base_model_if_lora: Optional[str] = None
    model_checkpoint: Optional[str] = None
    rits_deployment_reference: Optional[str] = None
    model_table: Optional[str] = "model_shared"
    ns: Optional[str] = "granite_dot_build.public"
    ttl: Optional[str] = RITS_TTL

    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]


class SimpleJobResponse(BaseModel):
    """Simplified JobResponse without build_status for UserData"""

    id: Optional[UUID] = None
    experiment_name: Optional[str] = None
    status: Optional[JobStatus] = None
    model: Optional[str] = None
    config_name: Optional[str] = None
    dataset: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class SimpleConfiguration(BaseModel):
    """Simplified Configuration without config_data for UserData"""

    id: Optional[UUID] = Field(None, description="Unique configuration identifier")
    user_id: Optional[str] = None
    name: Optional[str] = Field(
        None,
        description="Configuration name",
        examples=["lora-config-1", "prefix-tuning-aggressive"],
    )
    tuner_type: Optional[str] = Field(
        None,
        description="HPO algorithm type",
        examples=["bayesian", "grid_search", "random_search"],
    )
    rl_tuner_type: Optional[str] = Field(
        None,
        description="HPO algorithm type for RL training",
        examples=["bayesian", "grid_search", "random_search"],
    )
    artifact_id: Optional[str] = None
    artifact_url: Optional[str] = None
    config_data: Optional[dict] = None
    associated_jobs: List[SimpleJobResponse] = []
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @field_validator("config_data", mode="before")
    @classmethod
    def force_config_data_null(cls, v):
        return None


class DatasetResponse(DatasetInfo):
    """Dataset info with associated jobs"""

    artifact_id: Optional[str] = None
    artifact_url: Optional[str] = None
    associated_jobs: List[SimpleJobResponse] = []
    train_file: Optional[str] = None
    train_records: Optional[int] = None
    train_file_size: Optional[int] = None
    validation_file: Optional[str] = None
    validation_records: Optional[int] = None
    validation_file_size: Optional[int] = None
    created_at: datetime
    updated_at: datetime


class User(BaseModel):
    id: Optional[UUID] = Field(None, description="Unique user identifier")
    email: str = Field(..., description="User email address")
    role: Roles = Field(..., description="role of the user")
    created_at: Optional[datetime] = Field(
        None, description="Account creation timestamp"
    )
    updated_at: Optional[datetime] = Field(None, description="Last login timestamp")
    jobs: Optional[List[SimpleJobResponse]] = None
    configs: Optional[List[SimpleConfiguration]] = None
    datasets: Optional[List[DatasetResponse]] = None


class EstimateMemoryUsage(BaseModel):
    model_size_billion_params: float
    precision: str = "bf16"
    batch_size: int = 1
    sequence_length: int = 128
    use_gradient_checkpointing: bool = True
    zero_stage: int = 3
    use_lora: bool = False
    gpu_size_gb: int = 80


class EstimateMemoryUsageRequest(BaseModel):
    model_name: str
    config_id: UUID
    gpu_memory: Optional[int] = 80


class EstimateMemoryUsageResponse(BaseModel):
    model_size_billion_params: float
    gpu_memory_gb: float
    cpu_memory_gb: float
    num_gpus: int
    weights_memory: float
    optimizer_memory: float
    gradients_memory: float
    activations_memory: float


class Configuration(BaseModel):
    id: Optional[UUID] = Field(None, description="Unique configuration identifier")
    user_id: Optional[str] = Field(None, description="Owner user ID")
    name: str = Field(
        ...,
        description="Configuration name",
        examples=["lora-config-1", "prefix-tuning-aggressive"],
    )
    tuner_type: Optional[str] = Field(
        None,
        description="HPO algorithm type",
        examples=["bayesian", "grid_search", "random_search"],
    )
    rl_tuner_type: Optional[str] = Field(
        None,
        description="HPO algorithm type for RL training",
        examples=["bayesian", "grid_search", "random_search"],
    )
    config_data: Dict[str, Any] = Field(
        ..., description="Hyperparameter search space definition"
    )
    associated_jobs: List[SimpleJobResponse] = None


# ---------------------------------------------------------------------------
# Chat models
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: user or assistant")
    content: Optional[str] = Field(None, description="Message content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversation message history")
    context: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Conversation context state"
    )
    thread_id: Optional[str] = Field(
        None,
        description="Stable conversation thread id; enables server-side message/tool-result memory across turns",
    )


class ChatResponse(BaseModel):
    output: str = Field(..., description="Assistant response text")
    context: Dict[str, Any] = Field(
        default_factory=dict, description="Updated conversation context"
    )
    tool_calls_made: Optional[List[Dict[str, Any]]] = Field(
        None, description="Tools invoked during this turn"
    )


class GenerateTestSolutionsRequest(BaseModel):
    prompts: List[List[Dict[str, str]]] = Field(
        ..., description="List of message arrays (each is a VERL prompt)"
    )


class GenerateTestSolutionsResponse(BaseModel):
    solutions: List[str] = Field(..., description="Generated solution strings")
