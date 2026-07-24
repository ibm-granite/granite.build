# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

from dependencies import get_gb_service
from fastapi import APIRouter, Depends, Query
from services import gb_service

router = APIRouter()


@router.get("/api/gb/logs/{job_id}")
async def get_gb_logs(
    job_id: str,
    fetch_all: bool = Query(
        False,
        alias="all",
        description="Paginate through all log pages instead of returning only the first page",
    ),
    gb: gb_service.GBService = Depends(get_gb_service),
):
    """
    Get the Job status for job id
    """
    return await gb.get_gb_logs_by_job_id(job_id=job_id, fetch_all=fetch_all)


@router.get("/api/gb/status/{build_id}")
async def get_gb_status(
    build_id: str, gb: gb_service.GBService = Depends(get_gb_service)
):
    """
    Get the GB build status for job id
    """
    return await gb.get_gb_status(build_id=build_id)
