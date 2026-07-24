# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse
from dependencies import get_gb_service, get_job_service
from services import job_service, gb_service
from typing import Dict, Any
import models as api

router = APIRouter()


@router.get("/api/autotune_yaml")
async def read_autotune_yaml(gb: gb_service.GBService = Depends(get_gb_service)):
    content = gb.load_yaml()
    return content


@router.post("/api/create_autotune_yaml")
async def create_autotune_yaml(
    config: dict, gb: gb_service.GBService = Depends(get_gb_service)
):
    return gb.create_yaml(config)


@router.post("/api/insert_trial_result", tags=["Utils"])
async def insert_trial_result(
    data: Dict[str, Any], job: job_service.Job = Depends(get_job_service)
):
    """
    insert trial result
    """

    result = await job.insert_trial_results(id=data["id"], result=data)
    return result


@router.post("/api/record_trial", tags=["Utils"])
async def insert_trials(
    config: api.Trial, job: job_service.Job = Depends(get_job_service)
):
    """
    Insert job trials
    """
    return await job.insert_trial(data=config)


@router.post("/api/update_status", tags=["Utils"])
async def update_status(
    data: api.UpdateStatus, job: job_service.Job = Depends(get_job_service)
):
    """
    Update job and trial status
    """
    result = await job.status_updates(data=data)
    return result


@router.get("/", deprecated=True, include_in_schema=False)
def root():
    """Redirects to the documentation."""
    return RedirectResponse(url="/fmtune/docs")
