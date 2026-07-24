# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Query
from fastapi.responses import FileResponse

import dependencies
from auth import get_current_user
import models as api
from services import (
    job_service,
    user_service,
    db_service,
)  # for type annotations Job / User / Database

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/api/job/estimate_usages",
    tags=["Tunings"],
    summary="Estimate memory usage",
    response_description="Estimate memory usage for tuning a model",
)
async def estimate_usages(
    config: api.EstimateMemoryUsageRequest,
    job: job_service.Job = Depends(dependencies.get_job_service),
) -> api.EstimateMemoryUsageResponse:
    """
    Estimate the resource requirement to start the tuning.
    """
    return await job.estimate_memory_usage(config)


@router.post(
    "/api/job",
    tags=["Tunings"],
    summary="Start a new fine-tuning job",
    response_description="Job details including job ID and status",
)
async def start_tuning(
    config: api.TuningConfig,
    background_task: BackgroundTasks,
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """
    Start a new fine-tuning job with the specified configuration.

    This endpoint initiates a fine-tuning job for a foundation model using the provided
    configuration, dataset, and hyperparameters. The job runs asynchronously in the background.

    **Required Parameters:**
    - **config_id**: ID of the configuration to use
    - **dataset_id**: ID of the dataset to train on
    - **model**: Name/ID of the foundation model to fine-tune
    - **experiment_name**: Unique name for this tuning experiment

    **Optional Parameters:**
    - **tuning_type**: Type of tuning (LORA, PREFIX_TUNING, etc.)
    - **seed**: Random seed for reproducibility (default: 42)
    - **ray_address**: Ray cluster address for distributed training
    - **cleanup**: Whether to cleanup resources after completion (default: true)

    **Returns:**
    - Job ID and initial status
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    config.user_id = user_id
    # return True
    return await job.start(config, background_task)


@router.get(
    "/api/job/{id}",
    tags=["Tunings"],
    summary="Get job status and details",
    response_description="Complete job information including status, configuration, and metrics",
)
async def get_job(
    id: str,
    include_logs: bool = Query(True, description="Include logs in the response"),
    log_limit: int = Query(
        1000, ge=1, description="Maximum number of log lines to return"
    ),
    all_logs: bool = Query(False, description="Return all logs, ignoring log_limit"),
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """
    Retrieve detailed information about a specific fine-tuning job.

    Returns comprehensive job details including:
    - Current status (PENDING, RUNNING, COMPLETED, ERROR, etc.)
    - Configuration used
    - Start and end timestamps
    - Progress metrics
    - Associated trials

    **Parameters:**
    - **id**: Unique job identifier (UUID)
    - **include_logs**: Whether to include logs (default: true)
    - **log_limit**: Max log lines to return (default: 1000)
    - **all_logs**: If true, return all logs ignoring log_limit

    **Returns:**
    - Complete job object with all associated metadata

    **Errors:**
    - 404: Job not found or user doesn't have access
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    return await job.get_job(
        id, user_id, include_logs=include_logs, log_limit=log_limit, all_logs=all_logs
    )


@router.get(
    "/api/job/by_build_id/{build_id}",
    tags=["Tunings"],
    summary="Get job status and details by Granite Build build_id",
    response_description="Complete job information, resolved via the Granite Build build_id",
)
async def get_job_by_build_id(
    build_id: str,
    include_logs: bool = Query(True, description="Include logs in the response"),
    log_limit: int = Query(
        1000, ge=1, description="Maximum number of log lines to return"
    ),
    all_logs: bool = Query(False, description="Return all logs, ignoring log_limit"),
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """
    Retrieve detailed information about a job, looked up by its Granite Build
    build_id instead of its job_id.

    Returns the exact same payload as `GET /api/job/{id}` — this endpoint
    only differs in how the job is located.

    **Parameters:**
    - **build_id**: Granite Build build UUID (stored in `gb_tasks.build_id`)
    - **include_logs**: Whether to include logs (default: true)
    - **log_limit**: Max log lines to return (default: 1000)
    - **all_logs**: If true, return all logs ignoring log_limit

    **Returns:**
    - Complete job object with all associated metadata

    **Errors:**
    - 404: No job found for this build_id, or user doesn't have access
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    return await job.get_job_by_build_id(
        build_id,
        user_id,
        include_logs=include_logs,
        log_limit=log_limit,
        all_logs=all_logs,
    )


@router.get(
    "/api/job/{id}/config",
    tags=["Tunings"],
    summary="Get the configuration snapshot used by a job",
    response_description="Configuration as it was when the job was created, with staleness indicator",
)
async def get_job_config(
    id: str,
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """
    Retrieve the configuration snapshot for a specific job.

    Returns the config as it was at job creation time. If the live config
    has been updated since, `is_stale` will be true.

    **Returns:**
    - name, tuner_type, rl_tuner_type, config_data, is_stale
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    return await job.get_job_config_snapshot(id, user_id)


@router.get(
    "/api/job/{id}/logs",
    tags=["Tunings"],
    summary="Get paginated job logs in descending order",
)
async def get_job_logs_paginated(
    id: str,
    before_id: int = Query(
        0, ge=0, description="Return logs with id < before_id (0 = latest)"
    ),
    limit: int = Query(50, ge=1, le=500, description="Number of log entries per page"),
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    db: db_service.Database = Depends(dependencies.get_database),
):
    """
    Fetch log entries for a job in descending order (newest first) with cursor pagination.

    Use `before_id` to load older pages. On first call, omit `before_id` or pass 0
    to get the latest logs. Use the `id` of the last log in the response as `before_id`
    for the next page.
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    await db.get_job(id, user_id, include_logs=False)
    return await db.get_logs_page(id, before_id, limit)


@router.get(
    "/api/job/push_to_rits/{id}",
    tags=["Tunings"],
    summary="Get RITS publication info for a job",
    response_description="RITS publication metadata",
)
async def get_rits_by_job_id(
    id: str,
    auth_user: api.AuthUser = Depends(get_current_user),
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """
    Get RITS publication information for a job.

    Check if a model has been published to RITS for inference serving.
    Returns publication metadata if available.

    **Parameters:**
    - **id**: Unique job identifier (UUID)

    **Returns:**
    - RITS publication details or null if not published

    **Use Cases:**
    - Check publication status
    - Get inference endpoint information
    - Verify deployment readiness
    """
    return await job.get_rits_for_job(id)


@router.post(
    "/api/job/push_to_rits/{id}",
    tags=["Tunings"],
    summary="Publish model to RITS",
    response_description="RITS publication confirmation",
)
async def push_to_rits_by_job_id(
    id: str,
    background_task: BackgroundTasks,
    auth_user: api.AuthUser = Depends(get_current_user),
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """
    Publish a fine-tuned model to RITS (Runtime Inference Service).

    Deploy the trained model to IBM's inference serving platform for
    real-time predictions. The publication happens asynchronously in
    the background.

    **What this does:**
    - Packages the model for inference
    - Uploads to RITS infrastructure
    - Creates inference endpoint
    - Configures serving parameters

    **Parameters:**
    - **id**: Job ID with completed training

    **Returns:**
    - Publication request confirmation with tracking info

    **Requirements:**
    - Job must be in COMPLETED status
    - Model artifacts must be available
    - User must have RITS access

    **Next Steps:**
    After publication, use the inference endpoint to make predictions
    with your fine-tuned model.
    """
    return await job.publish_to_rits(id=id, background_task=background_task)


@router.get(
    "/api/jobs",
    tags=["Tunings"],
    summary="List all jobs for current user",
    response_description="Array of all jobs belonging to the authenticated user",
)
async def get_jobs(
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """
    Retrieve all fine-tuning jobs for the authenticated user.

    Returns a list of all jobs created by the current user, including:
    - Job IDs
    - Experiment names
    - Current status
    - Creation timestamps
    - Model and dataset information

    Jobs are typically sorted by creation time (newest first).

    **Returns:**
    - Array of job objects
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    return await job.get_jobs(user_id)


@router.get(
    "/api/jobs/stats",
    tags=["Tunings"],
    summary="Get tuning job statistics for the current user",
    response_description="Total job count and a breakdown by status",
)
async def get_job_stats(
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    job: job_service.Job = Depends(dependencies.get_job_service),
) -> api.JobStats:
    """
    Retrieve aggregate statistics for the authenticated user's fine-tuning jobs.

    Returns the total number of jobs and a count for each possible status
    (PENDING, RUNNING, PAUSED, TERMINATED, ERROR, COMPLETED).

    **Returns:**
    - `total`: total number of jobs owned by the user
    - `pending`, `running`, `paused`, `terminated`, `error`, `completed`:
      per-status counts (0 if the user has no jobs in that status)
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    return await job.get_job_stats(user_id)


@router.delete(
    "/api/job/{job_id}",
    tags=["Tunings"],
    summary="Delete a fine-tuning job",
    response_description="Boolean indicating success of deletion",
)
async def delete_job(
    job_id: str,
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    job: job_service.Job = Depends(dependencies.get_job_service),
) -> bool:
    """
    Permanently delete a fine-tuning job and all associated data.

    This operation:
    - Removes the job record from the database
    - Deletes all associated trials and logs
    - Cleans up output artifacts and model checkpoints
    - Cannot be undone

    **Parameters:**
    - **job_id**: Unique job identifier (UUID)

    **Returns:**
    - `true` if deletion was successful
    - `false` if job was not found or couldn't be deleted

    **Errors:**
    - 403: User doesn't have permission to delete this job
    - 404: Job not found
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    return await job.delete_job(job_id, user_id)


@router.get(
    "/api/job/{job_id}/trials",
    tags=["Tunings"],
    summary="Get all trials for a job",
    response_description="Array of trial objects with configurations and results",
)
async def get_trials_by_job_id(
    job_id: str,
    auth_user: api.AuthUser = Depends(get_current_user),
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """
    Retrieve all hyperparameter tuning trials for a specific job.

    Each trial represents one hyperparameter configuration that was tested
    during the AutoML/HPO process. Returns detailed information including:
    - Trial configurations (hyperparameters used)
    - Training metrics and performance
    - Trial status and timestamps
    - Resource utilization

    **Parameters:**
    - **job_id**: Unique job identifier (UUID)

    **Returns:**
    - Array of trial objects sorted by performance or creation time

    **Use Cases:**
    - Analyze which hyperparameters performed best
    - Compare trial configurations
    - Debug failed trials
    """
    return await job.get_trials_by_job_id(job_id)


@router.get(
    "/api/job/{job_id}/trials/logs",
    tags=["Tunings"],
    summary="Get logs for all trials in a job",
    response_description="Aggregated logs from all trials",
)
async def get_trials_log_by_job_id(
    job_id: str,
    auth_user: api.AuthUser = Depends(get_current_user),
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """
    Retrieve aggregated training logs from all trials in a job.

    Returns comprehensive logging information including:
    - Training progress (loss, accuracy, etc.)
    - System messages and warnings
    - Error traces
    - Resource utilization logs

    **Parameters:**
    - **job_id**: Unique job identifier (UUID)

    **Returns:**
    - Array of log entries with timestamps and severity levels

    **Log Levels:**
    - INFO: General progress updates
    - WARNING: Non-critical issues
    - ERROR: Failures and exceptions

    **Use Cases:**
    - Monitor training progress
    - Debug trial failures
    - Analyze performance issues
    """
    return await job.get_trials_logs_by_job_id(job_id)


@router.get(
    "/api/job/trial/{trial_id}/logs",
    tags=["Tunings"],
    summary="Get paginated logs for a specific trial",
    response_description="Paginated training logs for the specified trial",
)
async def get_trial_logs_by_id(
    trial_id: str,
    before_id: int = Query(
        0, ge=0, description="Return logs with id < before_id (0 = latest)"
    ),
    limit: int = Query(50, ge=1, le=500, description="Number of log entries per page"),
    auth_user: api.AuthUser = Depends(get_current_user),
    db: db_service.Database = Depends(dependencies.get_database),
):
    """
    Fetch trial log entries in descending order (newest first) with cursor pagination.

    Use `before_id` to load older pages. On first call, omit `before_id` or pass 0
    to get the latest logs. Use the `id` of the last log in the response as `before_id`
    for the next page.
    """
    return await db.get_trial_logs_page(trial_id, before_id, limit)


@router.get(
    "/api/job/{job_id}/results",
    tags=["Tunings"],
    summary="Get aggregated results for a job",
    response_description="Job results including best configuration and performance metrics",
)
async def get_trial_results_by_job_id(
    job_id: str,
    auth_user: api.AuthUser = Depends(get_current_user),
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """
    Get the aggregated results and best performing configuration for a job.

    Returns:
    - Best performing hyperparameter configuration
    - Final evaluation metrics (accuracy, loss, F1, etc.)
    - Comparison across all trials
    - Training history and convergence information

    **Parameters:**
    - **job_id**: Unique job identifier (UUID)

    **Returns:**
    - Results object with best configuration and metrics

    **Note:** Only available after job completion
    """
    return await job.get_results_by_job_id(job_id)


@router.get(
    "/api/job/{job_id}/result_report",
    tags=["Tunings"],
    summary="List downloadable result files",
    response_description="Array of available files (CSV reports, JSON configs, etc.)",
)
async def get_job_assets(
    job_id: str,
    auth_user: api.AuthUser = Depends(get_current_user),
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """
    Get a list of all downloadable result files for a completed job.

    Available files typically include:
    - **results.csv**: Detailed metrics for all trials
    - **best_config.json**: Winning hyperparameter configuration
    - **training_history.csv**: Training metrics over time
    - **model_checkpoints**: Trained model artifacts

    **Parameters:**
    - **job_id**: Unique job identifier (UUID)

    **Returns:**
    - Array of file objects with names, sizes, and download URLs

    **Usage:**
    1. Call this endpoint to get available files
    2. Use the filename with `/api/job/{job_id}/result_report/{filename}` to download
    """
    result = await job.list_job_assets(job_id=job_id)
    return result


@router.post(
    "/api/job/{job_id}/prepare_download",
    tags=["Tunings"],
    summary="Start download preparation in the background",
    response_description="Task ID and status for tracking download preparation",
)
async def prepare_download(
    job_id: str,
    background_tasks: BackgroundTasks,
    auth_user: api.AuthUser = Depends(get_current_user),
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """
    Initiate background preparation of a ZIP archive for all job assets.

    Returns a task_id immediately. Poll GET /api/task/{task_id} for status.
    When status is COMPLETED, call GET /api/job/{job_id}/download_all_assets?task_id={task_id}
    to download the prepared file.
    """
    result = await job.prepare_download(job_id=job_id)
    if result["status"] == "PENDING":
        background_tasks.add_task(
            job.execute_download_preparation, job_id, result["task_id"]
        )
    return result


@router.get(
    "/api/job/{job_id}/download_all_assets",
    tags=["Tunings"],
    summary="Download all result files as a ZIP archive",
    response_description="ZIP archive containing all result files",
)
async def download_all_assets(
    job_id: str,
    task_id: Optional[str] = None,
    auth_user: api.AuthUser = Depends(get_current_user),
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """
    Download all result files for a completed job as a single ZIP archive.

    **Parameters:**
    - **job_id**: Unique job identifier (UUID)
    - **task_id**: Optional task ID from prepare_download. If provided, serves the
      pre-prepared ZIP instead of building one on-the-fly.

    **Returns:**
    - ZIP file download containing all assets (CSVs, JSONs, model checkpoints, etc.)

    **Errors:**
    - 404: No results found for the job
    - 400: Job not completed or error creating archive
    """
    if task_id:
        zip_path, zip_filename = await job.get_prepared_download(task_id)
    else:
        zip_path, zip_filename = await job.download_all_assets(job_id=job_id)

    return FileResponse(
        path=zip_path,
        filename=zip_filename,
        media_type="application/zip",
    )


@router.get(
    "/api/job/{job_id}/result_report/{filename}",
    tags=["Tunings"],
    summary="Download a specific result file",
    response_description="File download (CSV, JSON, or binary model file)",
)
async def download_job_result(
    job_id: str,
    filename: str,
    auth_user: api.AuthUser = Depends(get_current_user),
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """
    Download a specific result file from a completed job.

    **Common Files:**
    - **results.csv**: Trial results in CSV format
    - **best_config.json**: Best hyperparameter configuration
    - **training_history.csv**: Training metrics over epochs
    - **model.pth** / **adapter_model.bin**: Model checkpoints

    **Parameters:**
    - **job_id**: Unique job identifier (UUID)
    - **filename**: Name of the file to download (from result_report endpoint)

    **Returns:**
    - File download with appropriate content-type headers

    **Errors:**
    - 404: File not found or job doesn't exist
    - 403: User doesn't have access to this job
    """
    result = await job.download_job_result(job_id=job_id, filename=filename)
    return result


@router.get(
    "/api/task/{id}",
    tags=["Tasks"],
    summary="Get task details",
    response_description="Task information with status and progress",
)
async def get_task(
    id: str,
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """
    Retrieve details about a specific task within a job.

    Tasks represent individual operations within a tuning job, such as:
    - Data preprocessing
    - Trial execution
    - Model evaluation
    - Result aggregation

    **Parameters:**
    - **id**: Unique task identifier (UUID)

    **Returns:**
    - Task object with status, progress, and metadata

    **Use Cases:**
    - Track progress of specific operations
    - Debug task failures
    - Monitor resource usage per task
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    return await job.get_task(id, user_id)


@router.get(
    "/api/tasks/{job_id}",
    tags=["Tasks"],
    summary="Get all tasks for a job",
    response_description="Array of task objects",
)
async def get_tasks_for_job(
    job_id: str,
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """
    Retrieve all tasks associated with a specific job.

    Returns a list of all tasks that make up the tuning job, including:
    - Task types and purposes
    - Execution status
    - Start/end times
    - Resource usage
    - Error messages (if failed)

    **Parameters:**
    - **job_id**: Unique job identifier (UUID)

    **Returns:**
    - Array of task objects ordered by execution sequence

    **Use Cases:**
    - Monitor job execution pipeline
    - Identify bottlenecks
    - Debug failures at task level
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    return await job.get_tasks(job_id, user_id)


@router.post("/api/job/{job_id}", tags=["Utils"], include_in_schema=False)
async def update_asset(
    job_id: str,
    job: job_service.Job = Depends(dependencies.get_job_service),
):
    """
    Update assets as per new format
    """
    return await job.update_to_new_output_asset(job_id=job_id)
