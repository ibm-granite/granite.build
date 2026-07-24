# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import logging
import math
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import models as api
import paths
from constants import RITS_TTL
from fastapi import BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from services import db_service, file_service, gb_service, logging_service
from services.plugins import Seam, resolve
from services.yaml_service import YAMLManager
from utils import (
    build_dmf_url,
    extract_artifact_identifier,
    extract_chars,
    extract_github_url,
    get_granite_model_params,
    get_utc_timestamp,
    is_gb_enabled,
    is_valid_uuid,
    parse_gb_message,
    time_elapsed,
    utc_now_string,
)

gb: gb_service.GBService = gb_service.GBService()

logger = logging.getLogger(__name__)


class Job:
    def __init__(self, db: db_service.Database):
        self.db = db
        self.dmf = resolve(Seam.REGISTRY, db=db)
        self.is_running = False
        self.monitor_thread = None
        self._main_loop = None

    async def start(
        self, run_config: api.TuningConfig, background_task: BackgroundTasks
    ):
        config_dict = await self.db.get_config(run_config.config_id)
        run_config.tuning_type = config_dict["tuner_type"]
        config_snapshot = {
            "name": config_dict.get("name"),
            "tuner_type": config_dict.get("tuner_type"),
            "rl_tuner_type": config_dict.get("rl_tuner_type"),
            "config_data": config_dict.get("config_data"),
        }
        job_id = await self.db.insert_job(run_config, config_snapshot=config_snapshot)
        task = api.Task(job_id=str(job_id), type=api.TaskType.TUNING)
        await self.db.insert_task(task=task)
        try:
            runner = resolve(
                Seam.RUNNER,
                job_id=job_id,
                run_config=run_config,
                db=self.db,
                logging_handler=logging_service.BufferedLogHandler(db=self.db),
            )
            background_task.add_task(runner.run)
        except Exception as e:
            await self.update_job_status(id=job_id, status=api.JobStatus.ERROR)
            logger.error(f"Something went wrong: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Something went wrong")

        return {"job_id": str(job_id), "status": api.JobStatus.PENDING}

    async def get_job_config_snapshot(self, job_id: str, user_id: str) -> dict:
        return await self.db.get_job_config_snapshot(job_id, user_id)

    async def update_job_status(self, id: str, status: api.JobStatus) -> bool:
        await self.db.update_job_status(id=id, status=status)
        if status == "TERMINATED" or status == "ERROR":
            return await self.db.update_all_trial_status(job_id=id, status=status)

    async def get_jobs(self, user_id: str = None) -> list[api.JobResponse]:
        return await self.db.get_jobs(user_id)

    async def get_job_stats(self, user_id: str) -> dict:
        return await self.db.get_job_stats(user_id)

    async def delete_job(self, id: str, user_id) -> bool:
        try:
            if is_gb_enabled():
                job = await self.get_job(id=id, user_id=user_id)
                if job and job.get("status") in (
                    api.JobStatus.PENDING,
                    api.JobStatus.RUNNING,
                ):
                    # build_id/pr_url live in gb_tasks, not in the jobs table
                    task_dict = await self.db.get_task_by_job_id(
                        job_id=id, type=api.TaskType.TUNING
                    )
                    if task_dict:
                        build_id = task_dict.get("build_id")
                        pr_url = task_dict.get("pr_url")
                        if build_id:
                            await gb.cancel_gb_build(build_id)
                            logger.info(f"Cancelled Job in Granite Build: {build_id}")
                        elif pr_url:
                            # Fallback: cancel via CLI subprocess for legacy pr_url-only jobs
                            delete_command = ["build", "cancel", pr_url]
                            result = await gb.command_executor(delete_command)
                            output = result.strip().replace("\r", "\n")
                            if "error" in output.lower():
                                raise Exception(output)
                            logger.info(
                                f"Cancelled Job in Granite Build via CLI: {pr_url}"
                            )
            return await self.db.delete_job(id, user_id)
        except Exception as e:
            logger.error(f"result: {e}")
            raise HTTPException(status_code=400, detail=str(e))

    async def insert_trial(self, data: api.Trial) -> str:
        try:
            result = await self.db.insert_trial(data=data)
            return result
        except Exception as e:
            raise HTTPException(status_code=409, detail=f"Something went wrong: {e}")

    async def get_trials_by_job_id(self, job_id: str) -> list[api.Trial]:
        return await self.db.get_trials_by_job_id(job_id=job_id)

    async def get_trials_logs_by_job_id(self, job_id: str) -> list[api.LogEntry]:
        return await self.db.get_trials_logs_by_job_id(job_id=job_id)

    async def get_trial_logs_by_id(self, trial_id: str) -> list[api.LogEntry]:
        return await self.db.get_trials_logs_by_id(trial_id=trial_id)

    async def get_trial_logs_page(
        self, trial_id: str, before_id: int = 0, limit: int = 50
    ) -> dict:
        return await self.db.get_trial_logs_page(
            trial_id=trial_id, before_id=before_id, limit=limit
        )

    async def terminate_jobs(self, status: api.JobStatus) -> bool:
        if is_gb_enabled():
            logger.debug("Jobs can't be terminated as GB is enabled")
            return False
        jobs = await self.get_jobs()
        updated_jobs = []
        for job in jobs:
            if job["status"] not in {"TERMINATED", "COMPLETED", "ERROR"}:
                await self.update_job_status(id=job["id"], status=status)
                updated_jobs.append(job["id"])
        return True

    async def update_trial_status(self, id: str, status: api.JobStatus) -> bool:
        await self.db.update_trial_status(trial_id=id, status=status)

    async def insert_trial_results(
        self, id: str, result: Union[api.Result, api.F1_Score, api.Rouge_Score]
    ) -> api.Result:
        try:
            result["job_id"] = id
            result["metric"] = "loss"
            result["metrics"] = self.parse_result(result)
            logger.info(f"Parsed result: {result}")
            response = await self.db.insert_result(metadata=result)
            return response
        except Exception as e:
            raise HTTPException(status_code=409, detail=f"something went wrong: {e}")

    async def get_results_by_job_id(self, job_id: str) -> list[api.Result]:
        return await self.db.get_results_by_job_id(job_id=job_id)

    def get_job_results_directory(self, job_id: str, revision: str = None) -> Path:
        """Get the results directory for a specific job."""
        if is_gb_enabled():
            output_dir = Path(paths.dmf_cache()) / "dmf_models" / str(job_id) / revision
            return output_dir
        else:
            AUTOTUNE_RESULTS_PATH = paths.results_path()
            output_dir = (
                Path(AUTOTUNE_RESULTS_PATH) / "output" / str(job_id) / "results"
            )
            logger.debug(f"get_job_results_directory: {output_dir}")
            return output_dir

    async def list_job_assets(self, job_id: str) -> List[Dict[str, str]]:
        """List all assets in the job's results directory."""
        try:
            if is_gb_enabled():
                task_by_job_id = await self.db.get_task_by_job_id(
                    job_id, api.TaskType.TUNING
                )
                if task_by_job_id is None:
                    raise Exception(f"No assets found for job_id: {job_id}")
                task = api.Task(**task_by_job_id)
                if task.pr_url is None:
                    raise Exception(f"Github PR URL not found for {job_id}")
                if (
                    task is not None
                    and task.pr_url is not None
                    and task.build_status is not None
                    and task.build_status.get("details", {}).get("status") == "success"
                ):
                    job = await self.db.get_job_by_id(id=job_id)
                    if job is not None and job["output_artifacts"] is not None:
                        logger.debug(f"Found job artifacts: {job_id}")
                        assets = json.loads(job["output_artifacts"])
                        return assets
                    logger.warning(f"Job artifacts not found: {job_id}")
                    logger.debug(f"Fetching checkpoints from GB: {job_id}")
                    result = self.dmf.get_checkpoints(task.artifact_uri)
                    logger.debug(f"result: {result}")
                    files = result[0]["files"]
                    for file in files:
                        file["size"] = file["file_size"]
                        file["modified"] = get_utc_timestamp(file["created"])
                        file["filename"] = os.path.basename(urlparse(file["path"]).path)
                        file["published"] = True
                    logger.debug(f"list_job_assets : gb_enabled, {result}")
                    data = self.remove_duplicates_by_name(result[0]["files"])
                    await self.db.push_job_artifacts(
                        job_id=job_id, output_artifacts=data
                    )
                    return data
                else:
                    raise Exception(
                        f"Assets are not available because the job status is {task.build_status.get('details', {}).get('status')}."
                    )
            else:
                results_dir = self.get_job_results_directory(job_id)
                logger.debug(f"results_dir: {results_dir}")
                if not results_dir.exists():
                    raise HTTPException(
                        status_code=404,
                        detail=f"Results directory not found for job {job_id}",
                    )

                assets = []
                for (
                    file_path
                ) in results_dir.iterdir():  # list files in results directory
                    if file_path.is_file():  # only include files, not directories
                        assets.append(
                            {
                                "filename": file_path.name,
                                "size": file_path.stat().st_size,
                                "modified": datetime.fromtimestamp(
                                    file_path.stat().st_mtime
                                ).isoformat(),
                            }
                        )

                return assets
        except Exception as e:
            logger.info(f"list all assets {e}", exc_info=True)
            raise HTTPException(status_code=400, detail=str(e))

    def get_asset_path(
        self, job_id: str, filename: str, revision: str = None
    ) -> Optional[Path]:
        """Get the full path for a specific asset."""
        job_dir = self.get_job_results_directory(job_id=job_id, revision=revision)
        # Search for the file recursively in the job directory
        for file_path in job_dir.rglob(filename):
            if file_path.is_file():
                return file_path

        return None

    async def download_job_result(self, job_id: str, filename: str):
        try:
            asset_path = None
            if is_gb_enabled():
                logger.debug("GB is enabled")
                task_dict = await self.db.get_task_by_job_id(
                    job_id, api.TaskType.TUNING
                )
                task = api.Task(**task_dict)
                logger.debug(f"task, {task}")
                if not task.artifact_uri:
                    raise Exception(f"No artifact url found for job {job_id}")
                # Let's check if asset is already available
                model, revision = extract_artifact_identifier(task.artifact_uri)
                asset_path = self.get_asset_path(
                    job_id=model, filename=filename, revision=revision
                )
                logger.debug(f"got asset path: {asset_path}")
                if not asset_path:
                    job_assets = await self.list_job_assets(job_id=job_id)
                    logger.debug(f"job_assets, {job_assets}")
                    filtered_assets = [
                        asset.get("path")
                        for asset in job_assets
                        if asset.get("filename") == filename
                    ]
                    if len(filtered_assets) == 0:
                        raise Exception(
                            f"Path not found for filename {filename} in job_id {job_id}"
                        )
                    logger.debug(f"filtered_assets: {filtered_assets}")
                    download_status = self.dmf.pull_checkpoint_file(
                        artifact_url=task.artifact_uri, file_paths=filtered_assets
                    )
                    logging.debug(f"download_status is success: {download_status}")
                    asset_path = self.get_asset_path(
                        job_id=model, filename=filename, revision=revision
                    )
            else:
                asset_path = self.get_asset_path(job_id, filename)
            if not asset_path:
                raise Exception("Asset not found")

            return FileResponse(
                path=str(asset_path),
                filename=filename,
                media_type="application/octet-stream",
            )
        except Exception as e:
            logger.exception(e)
            raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

    async def download_all_assets(self, job_id: str):
        """Download all assets for a job as a ZIP file."""
        try:
            job = await self.db.get_job_by_id(id=job_id)
            experiment_name = job["experiment_name"] if job else job_id
            zip_filename = f"{experiment_name}_assets.zip"

            if is_gb_enabled():
                task_dict = await self.db.get_task_by_job_id(
                    job_id, api.TaskType.TUNING
                )
                if task_dict is None:
                    raise Exception(f"No task found for job_id: {job_id}")
                task = api.Task(**task_dict)
                if not task.artifact_uri:
                    raise Exception(f"No artifact URI found for job {job_id}")

                download_path = await asyncio.to_thread(
                    self.dmf.pull_all_checkpoint_files,
                    artifact_url=task.artifact_uri,
                )
                logger.info(f"All assets pulled to: {download_path}")

                zip_path = await asyncio.to_thread(
                    file_service.zip_folder,
                    download_path,
                    output_zip=zip_filename,
                    output_dir=os.path.dirname(download_path),
                )
            else:
                results_dir = self.get_job_results_directory(job_id)
                if not results_dir.exists() or not any(results_dir.iterdir()):
                    raise HTTPException(
                        status_code=404,
                        detail=f"No results found for job {job_id}",
                    )

                zip_path = await asyncio.to_thread(
                    file_service.zip_folder,
                    str(results_dir),
                    output_zip=zip_filename,
                    output_dir=str(results_dir.parent),
                )

            logger.info(f"download_all_assets: ZIP created at {zip_path}")
            return zip_path, zip_filename
        except HTTPException:
            raise
        except Exception as e:
            logger.exception(e)
            raise HTTPException(
                status_code=400,
                detail=f"Error creating ZIP archive: {str(e)}",
            )

    async def prepare_download(self, job_id: str) -> dict:
        """Create a DOWNLOAD task and return the task_id and status immediately."""
        existing = await self.db.get_task_by_job_id(job_id, api.TaskType.DOWNLOAD)
        if existing:
            status = existing.get("status")
            if status == api.JobStatus.COMPLETED:
                return {"task_id": existing["id"], "status": "COMPLETED"}
            if status in (api.JobStatus.PENDING, api.JobStatus.RUNNING):
                return {"task_id": existing["id"], "status": status}

        task = api.Task(
            job_id=job_id, type=api.TaskType.DOWNLOAD, status=api.JobStatus.PENDING
        )
        task_id = await self.db.insert_task(task)
        return {"task_id": task_id, "status": "PENDING"}

    async def execute_download_preparation(self, job_id: str, task_id: str):
        """Background task: prepare the ZIP and update task status."""
        try:
            task = api.Task(
                id=task_id,
                job_id=job_id,
                type=api.TaskType.DOWNLOAD,
                status=api.JobStatus.RUNNING,
            )
            await self.db.update_task(task)

            zip_path, zip_filename = await self.download_all_assets(job_id=job_id)

            task.status = api.JobStatus.COMPLETED
            task.build_status = {"zip_path": zip_path, "zip_filename": zip_filename}
            task.updated_at = utc_now_string()
            await self.db.update_task(task)
        except Exception as e:
            logger.exception(f"Download preparation failed for job {job_id}: {e}")
            task = api.Task(
                id=task_id,
                job_id=job_id,
                type=api.TaskType.DOWNLOAD,
                status=api.JobStatus.ERROR,
            )
            task.build_status = {"error": str(e)}
            task.updated_at = utc_now_string()
            await self.db.update_task(task)

    async def get_prepared_download(self, task_id: str) -> tuple:
        """Get the zip_path and zip_filename from a completed DOWNLOAD task."""
        task_dict = await self.db.get_task(task_id)
        if not task_dict:
            raise HTTPException(status_code=404, detail="Download task not found")
        if task_dict.get("status") != api.JobStatus.COMPLETED:
            raise HTTPException(
                status_code=400,
                detail=f"Download not ready. Status: {task_dict.get('status')}",
            )
        build_status = task_dict.get("build_status", {})
        if isinstance(build_status, str):
            build_status = json.loads(build_status)
        zip_path = build_status.get("zip_path")
        zip_filename = build_status.get("zip_filename")
        if not zip_path or not os.path.exists(zip_path):
            raise HTTPException(
                status_code=404, detail="Prepared ZIP file not found on disk"
            )
        return zip_path, zip_filename

    async def cleanup_expired_downloads(self, max_age_minutes: int = 60):
        """Delete expired download tasks and their ZIP files."""
        expired = await self.db.get_expired_download_tasks(max_age_minutes)
        for task_dict in expired:
            try:
                build_status = task_dict.get("build_status", {})
                if isinstance(build_status, str):
                    build_status = json.loads(build_status)
                zip_path = (build_status or {}).get("zip_path")
                if zip_path and os.path.exists(zip_path):
                    os.remove(zip_path)
                    logger.info(f"Cleaned up expired download ZIP: {zip_path}")
                await self.db.delete_task(task_dict["id"])
            except Exception as e:
                logger.warning(
                    f"Failed to cleanup download task {task_dict.get('id')}: {e}"
                )

    def parse_result(self, data):
        result = {}
        if data.get("metric") == "accuracy":
            result = {
                "loss": data.get("loss"),
                "accuracy": data.get("accuracy"),
                "total_time": data.get("time_total_s"),
            }
        elif data.get("metric") == "rougeL":
            result = {
                "loss": data.get("loss"),
                # "rouge1": data.get("rouge1"),
                # "rouge2": data.get("rouge2"),
                "rougeL": data.get("rougeL"),
                # "rougeLsum": data.get("rougeLsum"),
                "total_time": data.get("time_total_s"),
            }
        elif data.get("metric") == "precision":
            result = {
                "loss": data.get("loss"),
                "precision": data.get("precision"),
                "total_time": data.get("time_total_s"),
            }
        elif data.get("metric") == "f1":
            result = {
                "loss": data.get("loss"),
                "f1": data.get("f1"),
                "total_time": data.get("time_total_s"),
            }
        elif data.get("metric") == "loss":
            result = {
                "loss": (
                    None
                    if data.get("loss") is None or math.isnan(data.get("loss"))
                    else data.get("loss")
                ),
                "train_loss": (
                    None
                    if data.get("train_loss") is None
                    or math.isnan(data.get("train_loss"))
                    else data.get("train_loss")
                ),
                "total_time": (
                    None
                    if data.get("time_total_s") is None
                    or math.isnan(data.get("time_total_s"))
                    else data.get("time_total_s")
                ),
            }
        else:
            return json.dumps({"error": "Unsupported metric"}, indent=4)

        return result

    async def update_to_new_output_asset(self, job_id: str):
        try:
            job_data = await self.db.get_job(id=job_id)
            model_folder_name = job_data["model"].split("/")[-1]
            AUTOTUNE_RESULTS_PATH = paths.results_path()
            output_dir = f"{AUTOTUNE_RESULTS_PATH}/output/{str(job_id)}"
            ray_result_folder = os.path.join(output_dir, "ray_results")
            ray_result_path = file_service.zip_folder(
                ray_result_folder,
                "ray_results.zip",
                os.path.join(output_dir, "results"),
            )
            logger.info(f"ray_result_path: {ray_result_path}")

            if job_data.get("autotune", None):
                tuned_model_folder = os.path.join(
                    output_dir, f"{model_folder_name}-autotuned"
                )
                tuned_model_path = file_service.zip_folder(
                    tuned_model_folder,
                    f"{model_folder_name}-autotuned.zip",
                    os.path.join(output_dir, "results"),
                )
                logger.info(f"tuned_model_path: {tuned_model_path}")
            return True
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(status_code=400, detail=f"File not found: {str(e)}")

    async def status_updates(self, data: api.UpdateStatus):
        try:
            if data.id and is_valid_uuid(data.id):
                await self.update_job_status(id=data.id, status=data.status)
            elif data.id and not is_valid_uuid(data.id):
                await self.update_trial_status(id=data.id, status=data.status)
        except Exception as e:
            logger.error(e, exc_info=True)
            raise HTTPException(
                status_code=400,
                detail=f"Error occured while updating job/trial status: {str(e)}",
            )

    def remove_duplicates_by_name(self, objects):
        seen = set()
        result = []
        seen.add("silent.log")
        for obj in objects:
            name = obj["filename"]
            if name not in seen:
                seen.add(name)
                result.append(obj)

        return result

    async def get_task(self, id: str, user_id: str = None) -> Optional[api.Task]:
        try:
            return await self.db.get_task(id=id)
        except Exception as e:
            logger.error(f"error occured in get_task_by_id: {e}")
            raise HTTPException(
                status_code=400, detail=f"Error occured in get_task_by_id: {str(e)}"
            )

    async def get_tasks(self, job_id: str, user_id: str = None) -> list[api.Task]:
        try:
            return await self.db.get_tasks(job_id=job_id)
        except Exception as e:
            logger.error(f"error occured in get_task_by_id: {e}")
            raise HTTPException(
                status_code=400, detail=f"Error occured in get_tasks: {str(e)}"
            )

    async def get_job(
        self,
        id: str,
        user_id: str,
        include_logs: bool = True,
        log_limit: int = 1000,
        all_logs: bool = False,
    ) -> api.JobResponse:
        """Get job details with optional GB task information."""
        logger.debug("Getting job with id: %s for user: %s", id, user_id)

        result = await self.db.get_job(
            id,
            user_id,
            include_logs=include_logs,
            log_limit=log_limit,
            all_logs=all_logs,
        )

        if not is_gb_enabled():
            logger.debug("GB is not enabled, returning basic job result")
            return result

        logger.debug("GB is enabled, processing task information")
        task = await self._get_or_create_task(id)

        if task is None:
            logger.warning("Failed to get or create task for job: %s", id)
            return result

        # Update task status if needed
        await self._update_task_from_gb(task, id)
        await self._handle_failed_build(task, id, result)
        await self._handle_cancelled_build(task=task, job_id=id, job_result=result)
        # Handle completed builds
        await self._process_completed_build(task, id, result)

        # Construct enhanced result
        enhanced_result = self._build_enhanced_result(result, task)

        logger.debug("Completed get_job for id: %s", id)
        return enhanced_result

    async def get_job_by_build_id(
        self,
        build_id: str,
        user_id: str,
        include_logs: bool = True,
        log_limit: int = 1000,
        all_logs: bool = False,
    ) -> api.JobResponse:
        """Resolve a Granite Build build_id to its job_id, then return the same
        enriched payload as get_job()."""
        job_id = await self.db.get_job_id_by_build_id(build_id)
        return await self.get_job(
            id=job_id,
            user_id=user_id,
            include_logs=include_logs,
            log_limit=log_limit,
            all_logs=all_logs,
        )

    async def _get_or_create_task(self, job_id: str) -> api.Task:
        """Get existing task or create new one if it doesn't exist."""
        task_data = await self.db.get_task_by_job_id(
            job_id=job_id, type=api.TaskType.TUNING
        )

        if task_data is not None:
            logger.debug("Found existing task for job: %s", job_id)
            return api.Task(**task_data)

        # Create new task
        task = api.Task(job_id=job_id, type=api.TaskType.TUNING)
        logger.debug("Creating new task for job: %s", job_id)
        task.id = await self.db.insert_task(task=task)
        return task

    async def _update_task_from_gb(self, task: api.Task, job_id: str) -> None:
        """Update task with latest information from GB if PR URL exists."""
        logger.debug(f"Updating task from GB: {task.id}")
        if task.pr_url is None and task.build_id is None:
            return

        if task.build_status is not None and (
            task.build_status.get("details", {}).get("status") == "failed"
            or task.build_status.get("details", {}).get("status") == "success"
        ):
            logger.debug(
                f"Found build status, exiting from _update_task_from_gb: {task.id}"
            )
            return

        try:
            logger.debug("Fetching GB status for task: %s", task.id)
            build_id = task.build_id if task.build_id else task.pr_url
            task.build_status = await gb.get_gb_status(build_id=build_id)

            if task.build_status is None:
                logger.warning("No build status returned from GB for task: %s", task.id)
                return

            details = task.build_status.get("details")
            if details is None:
                logger.warning("No details in build status for task: %s", task.id)
                return

            # Update task with build details
            task.pr_url = details.get("source_pr", task.pr_url)
            task.build_id = details.get("build_id", task.build_id)
            task.started_at = details.get("started_at", task.started_at)
            task.updated_at = details.get("updated_at", task.updated_at)

            # Update job status based on build status
            build_status = details.get("status")
            if build_status == "running":
                if task.type == api.TaskType.TUNING:
                    await self.db.update_job_status(
                        id=job_id, status=api.JobStatus.RUNNING
                    )
                task.status = api.JobStatus.RUNNING

            # Update database
            await self.db.update_task(task=task)

            logger.debug("Updated task %s with build status: %s", task.id, build_status)
        except Exception as e:
            logger.error("Failed to update task from GB: %s", str(e))

    async def _process_completed_build(
        self, task: api.Task, job_id: str, job_result: dict = None
    ) -> None:
        """Process completed builds to extract artifacts and update job status."""
        logger.debug(
            f"Processing completed tasks: {task.id} with status: {task.status}"
        )
        if (
            task.build_status is None
            or task.build_status.get("details", {}).get("status") != "success"
        ):
            logger.debug(
                f"Exiting Processing of completed tasks: {task.id} with status: {task.status}"
            )
            return

        if job_result is None:
            job_result = await self.db.get_job_by_id(id=job_id)

        # Handle successful build with missing artifact
        if task.artifact_id is None:
            await self._extract_artifact_from_build(task, job_id)

        # Update job status to completed if build succeeded
        if job_result["status"] != api.JobStatus.COMPLETED:
            logger.debug("Updating job %s status to COMPLETED", job_id)
            await self.db.update_job_status(id=job_id, status=api.JobStatus.COMPLETED)
            job_result["status"] = "COMPLETED"

        if task.status != api.JobStatus.COMPLETED:
            logger.debug("Updating task %s status to Completed", task.id)
            task.status = api.JobStatus.COMPLETED
            await self.db.update_task(task=task)

        # Invalidate cached status for completed builds
        build_id = task.build_id if task.build_id else task.pr_url
        if build_id:
            gb.invalidate_status_cache(str(build_id))

        if (
            task.type == api.TaskType.RITS
            and task.status == api.JobStatus.COMPLETED
            and task.rits_url is None
        ):
            logger.debug("Updating task %s with RITS url", task.id)
            build_history = task.build_status.get("build_history", [])
            if build_history:
                for build in build_history:
                    parsed_message = parse_gb_message(build["description"])
                    if parsed_message is not None:
                        logger.debug(
                            f"Updated task {task.id} with RITS url: {parsed_message['full_url']}"
                        )
                        task.rits_url = parsed_message["full_url"]
                        break
                await self.db.update_task(task=task)

    async def _handle_failed_build(
        self, task: api.Task, job_id: str, job_result: dict = None
    ) -> None:
        """Handle failed builds by updating job status."""
        if (
            task.build_status is None
            or task.build_status.get("details", {}).get("status") != "failed"
        ):
            return

        if job_result is None:
            job_result = await self.db.get_job_by_id(id=job_id)

        if (
            job_result["status"] != api.JobStatus.ERROR
            and task.type == api.TaskType.TUNING
        ):
            logger.debug("Updating job %s status to ERROR due to build failure", job_id)
            await self.db.update_job_status(id=job_id, status=api.JobStatus.ERROR)
            job_result["status"] = "ERROR"

        if task.status != api.JobStatus.ERROR:
            logger.debug("Updating task %s status to Completed", task.id)
            task.status = api.JobStatus.ERROR
            await self.db.update_task(task=task)

        # Invalidate cached status for failed builds
        build_id = task.build_id if task.build_id else task.pr_url
        if build_id:
            gb.invalidate_status_cache(str(build_id))

    async def _handle_cancelled_build(
        self, task: api.Task, job_id: str, job_result: dict = None
    ) -> None:
        """Handle failed builds by updating job status."""
        if (
            task.build_status is None
            or task.build_status.get("details", {}).get("status") != "cancelled"
        ):
            return

        if job_result is None:
            job_result = await self.db.get_job_by_id(id=job_id)

        if (
            job_result["status"] != api.JobStatus.TERMINATED
            and task.type == api.TaskType.TUNING
        ):
            logger.debug("Updating job %s status to CANCELLED", job_id)
            await self.db.update_job_status(id=job_id, status=api.JobStatus.TERMINATED)
            job_result["status"] = "TERMINATED"

        if task.status != api.JobStatus.TERMINATED:
            logger.debug("Updating task %s status to CANCELLED/TERMINATED", task.id)
            task.status = api.JobStatus.TERMINATED
            await self.db.update_task(task=task)

        # Invalidate cached status for cancelled builds
        build_id = task.build_id if task.build_id else task.pr_url
        if build_id:
            gb.invalidate_status_cache(str(build_id))

    async def _extract_artifact_from_build(self, task: api.Task, job_id: str) -> None:
        """Extract artifact information from successful build."""
        try:
            # output_artifacts = task.build_status.get("output_artifacts", [])
            targets = task.build_status.get("targets", [])
            if not targets or len(targets[0]) == 0:
                logger.warning("No targets found for task: %s", task.id)
                return

            output_artifact = targets[0].get("output_artifacts", [])
            if not output_artifact or len(output_artifact[0]) == 0:
                logger.warning("No output artifacts found for task: %s", task.id)
                return
            first_artifact = output_artifact[0]
            task.artifact_id = first_artifact.get("artifact_id")
            task.artifact_uri = first_artifact.get("uri")

            if task.artifact_id:
                logger.debug(
                    "Extracted artifact %s for task: %s", task.artifact_id, task.id
                )
                await self.db.update_task(task=task)

        except (IndexError, KeyError, TypeError) as e:
            logger.error(
                "Failed to extract artifact from build for task %s: %s", task.id, str(e)
            )

    def _build_enhanced_result(self, base_result: dict, task: api.Task) -> dict:
        """Build the enhanced result with task information."""
        del base_result["output_artifacts"]
        if task.artifact_uri:
            model, revision = extract_artifact_identifier(task.artifact_uri)
            dmf_url = build_dmf_url(model, revision)
        else:
            dmf_url = None
        enhanced_result = {
            **base_result,
            "task_id": task.id,
            "build_status": task.build_status,
            "github_pr_url": task.pr_url,
            "artifact_id": task.artifact_id,
            "artifact_uri": task.artifact_uri,
            "dmf_url": dmf_url,
            # "created_at": task.started_at,
            "updated_at": task.updated_at,
        }

        return enhanced_result

    async def get_running_jobs(self) -> list[api.JobResponse]:
        try:
            return await self.db.get_running_jobs()
        except Exception as e:
            logger.error(f"Error fetching running jobs: {str(e)}")
            return []

    async def check_and_update_jobs(self) -> list[api.JobResponse]:
        try:
            running_jobs = await self.get_running_jobs()
            logger.debug(f"check_and_update_jobs: running_jobs: {running_jobs}")
            if not running_jobs or len(running_jobs) == 0:
                logger.info("No running jobs found")
                return
            for job in running_jobs:
                try:
                    job_id = job["id"]
                    task_dict = await self.db.get_task_by_job_id(
                        job_id=job_id, type=api.TaskType.TUNING
                    )
                    logger.debug(f"check_and_update_jobs: task_dict: {task_dict}")
                    if task_dict:
                        task = api.Task(**task_dict)
                        await self._update_task_from_gb(task=task, job_id=job_id)
                        await self._handle_failed_build(
                            task=task, job_id=job_id, job_result=job
                        )
                        await self._handle_cancelled_build(
                            task=task, job_id=job_id, job_result=job
                        )
                        await self._process_completed_build(task, job_id, job)

                except Exception as e:
                    logger.warning(f"Unexpected error occured: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error in check_and_update_job_status: {str(e)}")

    async def check_and_update_tasks(self):
        try:
            pending_tasks = await self.db.get_pending_tasks()
            logger.debug("check_and_update_tasks: pending_tasks:")
            if not pending_tasks or len(pending_tasks) == 0:
                logger.info("No running jobs found")
                return
            for task_dict in pending_tasks:
                try:
                    if task_dict:
                        task = api.Task(**task_dict)
                        job_result = await self.db.get_job_by_id(id=task.job_id)
                        await self._update_task_from_gb(task=task, job_id=task.job_id)
                        await self._handle_failed_build(
                            task=task, job_id=task.job_id, job_result=job_result
                        )
                        await self._handle_cancelled_build(
                            task=task, job_id=task.job_id, job_result=job_result
                        )
                        await self._process_completed_build(
                            task=task, job_id=task.job_id, job_result=job_result
                        )

                except Exception as e:
                    logger.warning(f"Unexpected error occured: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error in check_and_update_job_status: {str(e)}")

    def _background_monitor(self):
        """
        Background thread function that runs the monitoring loop.
        Uses the main event loop via run_coroutine_threadsafe to avoid
        creating a separate loop (aiomysql pool is bound to the main loop).
        """
        while self.is_running:
            try:
                if self._main_loop is not None and self._main_loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(
                        self.check_and_update_tasks(), self._main_loop
                    )
                    future.result(timeout=120)
                gb.cleanup_stale_cache()
                logger.info("Job status check completed, sleeping for 5 minutes...")
                time.sleep(300)
            except Exception as e:
                logger.error(f"Error in background monitor: {e!r}")
                time.sleep(60)

    def start_monitoring(self):
        """
        Start the background monitoring thread.
        Captures the current (main) event loop so the background thread
        can submit coroutines to it via run_coroutine_threadsafe.
        """
        if self.is_running:
            logger.warning("Monitor is already running")
            return

        try:
            self._main_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._main_loop = None
            logger.warning("No running event loop — background monitor may not work")

        self.is_running = True
        self.monitor_thread = threading.Thread(
            target=self._background_monitor, daemon=True
        )
        self.monitor_thread.start()
        logger.info("Job status monitoring started")

    def stop_monitoring(self):
        """
        Stop the background monitoring thread
        """
        self.is_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("Job status monitoring stopped")

    async def publish_to_rits(self, id: str, background_task: BackgroundTasks):
        if is_gb_enabled():
            rits_task_id = uuid.uuid4()
            rits_task = api.Task(
                id=str(rits_task_id), job_id=str(id), type=api.TaskType.RITS
            )
            task_id = await self.db.insert_task(task=rits_task)

            payload = api.PushToRits(job_id=id, ttl=RITS_TTL)
            background_task.add_task(self.push_to_rits, payload, task_id)
            return {
                "status": "success",
                "message": f"Push to RITS started with task: {task_id}",
            }
        else:
            return {"message": "Tuning in granite build is disabled"}

    async def push_to_rits(self, payload: api.PushToRits, task_id: str):
        if is_gb_enabled():
            try:
                job_dict = await self.db.get_job_by_id(id=payload.job_id)
                if job_dict is not None:
                    payload.suffix = extract_chars(job_dict.get("experiment_name"))
                    model_name = job_dict.get("model").split("/")[1]
                    model_detail = self.dmf.get_model_detail(model_label=model_name)
                    if (
                        model_detail.get("model_label") is not None
                        and model_detail.get("revision") is not None
                    ):
                        payload.base_model_if_lora = f"{model_detail.get('model_label')}/{model_detail.get('revision')}"
                    if model_detail.get("model_label") is not None:
                        payload.rits_deployment_reference = (
                            f"{model_detail.get('model_label').replace('.', '-')}"
                        )
                logger.info(f"Initializing Push to Rits: {payload.job_id}")
                task_dict = await self.db.get_task_by_job_id(
                    job_id=payload.job_id, type=api.TaskType.TUNING
                )
                task = api.Task(**task_dict)
                model, revision = extract_artifact_identifier(task.artifact_uri)
                payload.model_checkpoint = f"{model}.{revision}"
                logger.debug(f"task: {task}")

                build_name = f"temp_yaml/jobs/{payload.job_id}/rits"
                logger.debug(
                    f"Rits build yaml already exist: {os.path.exists(f'{build_name}/build.yaml')}"
                )
                if not os.path.exists(f"{build_name}/build.yaml"):
                    command = [
                        "build",
                        "init",
                        build_name,
                        "--from-template",
                        "ModelToRITS",
                    ]
                    logger.debug(f"command for ModelToRITS build init: {command}")
                    result = await gb.command_executor(command)
                    logger.debug(f"{result}")
                logger.info(f"Granite build initialized for ModelToRITS: {build_name}")
                yaml_file = YAMLManager(f"{build_name}/build.yaml")
                config = yaml_file.read_yaml()
                # config["granite.build"]["targets"]["custom"]["steps"][0]["config"]["custom_code_config"][
                #     "setup_command"] = "git checkout fix_lora && pip install -r requirements.txt && pip install ."
                config["granite.build"]["targets"]["custom"]["steps"][0]["config"][
                    "custom_code_config"
                ]["start_command"] = self.build_rits_start_cmd(payload)
                logger.debug(f"Granite build config: {config}")
                yaml_file.write_yaml(config)
                logger.info(f"Rits build.yaml created: {build_name}")
                command = ["build", "start", "-f", f"{build_name}/build.yaml"]
                logger.info(f"command for build start: {command}")
                result = await gb.command_executor(command)
                output = result.strip().replace("\r", "\n")
                logger.info(f"output: {output}")
                url, build_id = extract_github_url(result)
                logger.info(f"Github PR url: {url} or uuid: {build_id}")
                if url is not None or build_id is not None:
                    rits_task_dict = await self.get_task(id=task_id)
                    rits_task = api.Task(**rits_task_dict)
                    rits_task.pr_url = url.strip() if url else None
                    rits_task.build_id = build_id.strip() if build_id else None
                    rits_task.updated_at = utc_now_string()
                    task_id = await self.db.update_task(task=rits_task)
                    logger.debug(f"task_id: {task_id}")
                    return rits_task
                else:
                    logger.error(f"No Github pull url found: {url}")
                    raise Exception(f"No Github pull url found: {url}")
            except Exception as e:
                logger.error(f"Error in push_to_rits: {str(e)}")
        else:
            return {"message": "Tuning in granite build is disabled"}

    def build_rits_start_cmd(self, config: api.PushToRits):
        cmd = (
            f"python main.py "
            f"--your_deployment_suffix {config.suffix} "
            f"--base_model_if_lora {config.base_model_if_lora} "
            f"--model_checkpoint {config.model_checkpoint} "
            f"--rits_deployment_reference {config.rits_deployment_reference} "
            f"--model_table {config.model_table} "
            f"--input_folder $INPUT_PATH "
            f"--output_folder $OUTPUT_PATH "
            f'--ns "granite_dot_build.public" '
            f"--ttl {config.ttl}"
        )
        return cmd

    async def get_rits_for_job(self, job_id):
        rits_task_dict = await self.db.get_task_by_job_id(
            job_id=job_id, type=api.TaskType.RITS
        )
        if rits_task_dict is None:
            return None
        rits_task = api.Task(**rits_task_dict)
        updated_rits_task = await self.execute_task_update(task=rits_task)
        difference = time_elapsed(updated_rits_task.updated_at)
        if updated_rits_task.status in (api.JobStatus.TERMINATED, api.JobStatus.ERROR):
            return None
        if difference >= int(RITS_TTL.replace("m", "")):
            return None
        return updated_rits_task

    async def execute_task_update(self, task: api.Task) -> api.Task:
        await self._update_task_from_gb(task=task, job_id=task.job_id)
        await self._handle_failed_build(task=task, job_id=task.job_id)
        await self._handle_cancelled_build(task=task, job_id=task.job_id)
        await self._process_completed_build(task=task, job_id=task.job_id)
        updated_task_dict = await self.db.get_task_by_job_id(
            job_id=task.job_id, type=api.TaskType.RITS
        )
        if updated_task_dict is None:
            return None
        updated_task = api.Task(**updated_task_dict)
        return updated_task

    async def estimate_memory_usage(
        self,
        config: api.EstimateMemoryUsageRequest,
    ) -> api.EstimateMemoryUsageResponse:
        """
        Estimate the resource requirement to start the tuning.
        Supports both SFT and RL (PPO, DPO, KTO) configurations.
        """
        logger.debug(f"Estimating memory usage for config: {config}")
        # Lazy import: the autotune training core is an optional IBM dependency,
        # absent in a credential-free install. Import it only when this
        # estimation path is actually exercised.
        from autotune.utils import estimate_memory_usage, parse_model_parameters

        if (
            config.model_name.startswith("ibm-granite/granite-4.0")
            and parse_model_parameters(config.model_name) is None
        ):
            model_params = get_granite_model_params(config.model_name)
        else:
            model_params = parse_model_parameters(config.model_name)

        if model_params is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unable to parse model parameters from model name: {config.model_name}",
            )
        configuration = await self.db.get_config(config_id=config.config_id)

        if configuration is None:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration with id {config.config_id} not found",
            )

        config_data = configuration.get("config_data", {})
        tuner_type = configuration.get("tuner_type")
        rl_tuner_type = configuration.get("rl_tuner_type")
        is_rl = rl_tuner_type is not None

        precision = (
            config_data.get("training_config", {})
            .get("precision", {})
            .get("default", None)
        )
        sequence_length = (
            config_data.get("training_config", {})
            .get("max_length", {})
            .get("default", None)
        )

        if is_rl:
            # For RL configs, get batch_size from tuners_rl_config
            rl_hyperparams = (
                config_data.get("tuners_rl_config", {})
                .get(rl_tuner_type, {})
                .get("hyperparams", {})
            )
            batch_size_config = rl_hyperparams.get("per_device_train_batch_size", {})
            if isinstance(batch_size_config, dict) and batch_size_config.get("values"):
                batch_size = batch_size_config["values"][-1]
            else:
                # DPO/KTO may not have batch_size; use a conservative default
                batch_size = 4
        else:
            # For SFT configs, get batch_size from tuners_config
            batch_size = (
                config_data.get("tuners_config", {})
                .get(tuner_type, {})
                .get("hyperparams", {})
                .get("per_device_train_batch_size", {})
                .get("values", [])[-1]
            )

        logger.debug(f"model_params: {model_params} billion parameters")
        logger.debug(f"sequence_length: {sequence_length}")
        logger.debug(f"precision: {precision}")
        logger.debug(f"batch_size: {batch_size}")
        if is_rl:
            logger.debug(f"rl_tuner_type: {rl_tuner_type}")

        try:
            result = estimate_memory_usage(
                model_size_billion_params=model_params,
                precision="bf16",
                batch_size=batch_size,
                sequence_length=sequence_length,
                use_gradient_checkpointing=True,
                zero_stage=3,
                use_lora=True,
            )

            # For RL training, adjust estimates for additional models in memory
            if is_rl:
                weights_mem = result.get("weights_memory", 0)
                optimizer_mem = result.get("optimizer_memory", 0)
                gradients_mem = result.get("gradients_memory", 0)

                if rl_tuner_type == "ppo":
                    # PPO loads 4 models: policy (trained) + reference (frozen)
                    #   + value (trained) + reward (frozen)
                    # Reference & reward: weights only (frozen)
                    # Value model: weights + optimizer + gradients (LoRA-trained)
                    additional_mem = weights_mem * 3 + optimizer_mem + gradients_mem
                else:
                    # DPO/KTO: policy (trained) + reference (frozen)
                    # Reference model: weights only
                    additional_mem = weights_mem

                result["gpu_memory_gb"] = result["gpu_memory_gb"] + additional_mem
                result["cpu_memory_gb"] = result["cpu_memory_gb"] + additional_mem
                result["num_gpus"] = math.ceil(result["gpu_memory_gb"] / 80)

            logger.debug(f"Estimated memory usage result: {result}")
            return result
        except Exception as e:
            logger.error(f"Error estimating memory usage: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Error estimating memory usage: {str(e)}",
            )
