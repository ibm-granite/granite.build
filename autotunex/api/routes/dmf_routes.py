# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

import dependencies
import models as api
from auth import get_current_user
from fastapi import APIRouter, Depends, HTTPException
from services import (  # for type annotations user_service.User / dmf_service.Dmf
    dmf_service,
    user_service,
)
from starlette.concurrency import run_in_threadpool

router = APIRouter()


@router.get(
    "/api/dmf/models",
    tags=["DMF"],
    summary="Get published models from DMF",
    response_description="Array of models published by the current user",
)
async def get_published_models(
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    dmf: dmf_service.Dmf = Depends(dependencies.get_dmf_service),
) -> list[api.ModelInfo]:
    """
    Retrieve models published to Data Model Factory (DMF) by the current user.

    Data Model Factory is IBM's model registry for storing and sharing
    fine-tuned models. This endpoint returns:
    - Model IDs and names
    - Model variants and labels
    - Publication metadata
    - Model size and type information
    - Access URLs

    **Returns:**
    - Array of ModelInfo objects for user's published models

    **Use Case:**
    - View your published models
    - Get model IDs for deployment
    - Verify successful publications
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    results = await dmf.get_models(user_id=user_id)
    return results


@router.get("/api/dmf/all_models", tags=["DMF"])
async def get_all_published_models(
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    dmf: dmf_service.Dmf = Depends(dependencies.get_dmf_service),
) -> list[api.ModelInfo]:
    """
    Get all published models from all the autotunex user in Data Model Factory
    """
    response = await user.get_user(auth_user.email)
    if response["role"] == api.Roles.ADMIN:
        results = await dmf.get_all_models()
        return results
    else:
        raise HTTPException(
            status_code=401, detail={"status": 401, "message": "Unauthorized"}
        )


@router.post(
    "/api/dmf/model/{job_id}",
    tags=["DMF"],
    summary="Publish model to Data Model Factory",
    response_description="Publication confirmation with model details",
)
async def publish_model(
    job_id: str,
    dmf_metadata: api.DmfMetadata,
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    dmf: dmf_service.Dmf = Depends(dependencies.get_dmf_service),
):
    """
    Publish a fine-tuned model from a completed job to Data Model Factory.

    This makes the model available in IBM's model registry for:
    - Deployment to inference services
    - Sharing with team members
    - Version control and tracking
    - Production use

    **Parameters:**
    - **job_id**: ID of completed tuning job with trained model
    - **dmf_metadata**: Publication metadata

    **Request Body:**
    ```json
    {
      "label": "customer-support-bot-v1",
      "variant": "lora-r16",
      "type": "text-generation",
      "size": "7B"
    }
    ```

    **Metadata Fields:**
    - **label**: Human-readable model name/label
    - **variant**: Model variant or version identifier
    - **type**: Model type (text-generation, classification, etc.)
    - **size**: Model size (7B, 13B, etc.)

    **Returns:**
    - Publication confirmation with DMF model ID

    **Errors:**
    - 404: Job not found or incomplete
    - 400: Invalid metadata
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    result = await dmf.publish_model(
        job_id=job_id, metadata=dmf_metadata, user_id=user_id
    )
    return result


@router.delete(
    "/api/dmf/model/{job_id}",
    tags=["DMF"],
    summary="Delete model from Data Model Factory",
    response_description="Deletion confirmation",
)
async def delete_model(
    job_id: str,
    auth_user: api.AuthUser = Depends(get_current_user),
    user: user_service.User = Depends(dependencies.get_user_service),
    dmf: dmf_service.Dmf = Depends(dependencies.get_dmf_service),
):
    """
    Remove a published model from Data Model Factory.

    This operation:
    - Removes the model from DMF registry
    - Deletes model artifacts from storage
    - Revokes access to the model
    - Does not delete the original training job

    **Parameters:**
    - **job_id**: ID of the job whose model should be deleted

    **Returns:**
    - Deletion confirmation with status

    **Errors:**
    - 404: Model not found in DMF
    - 403: User doesn't have permission
    - 409: Model is currently deployed (must undeploy first)

    **Note:** This only removes the DMF publication, not the local
    model files from the completed job.
    """
    user_id = (await user.get_user(auth_user.email))["id"]
    result = await dmf.delete_model(job_id=job_id, user_id=user_id)
    return result


@router.post(
    "/api/dmf/model_card",
    tags=["DMF"],
    summary="Get DMF model card/README",
    response_description="Model card content with documentation",
)
async def get_dmf_model_card(
    request: Dict[str, str],
    auth_user: api.AuthUser = Depends(get_current_user),
    dmf: dmf_service.Dmf = Depends(dependencies.get_dmf_service),
) -> Dict[str, Any]:
    """
    Fetch model card/README documentation for a DMF model.

    This endpoint retrieves comprehensive model documentation from DMF including:
    - Model description and purpose
    - Training details and hyperparameters
    - Performance metrics and benchmarks
    - Usage guidelines and examples
    - Limitations and known issues

    **Request Body:**
    ```json
    {
        "namespace": "base_training",
        "table": "model_shared",
        "model_label": "granite-3.2-8b-instruct",
        "revision": "r250219a"
    }
    ```

    **Returns:**
    - Model card content (markdown or structured format)

    **Errors:**
    - 400: Invalid parameters or DMF API error
    - 500: Server configuration error
    """
    namespace = request.get("namespace")
    table = request.get("table", "model_shared")
    model_label = request.get("model_label")
    revision = request.get("revision")

    if not all([namespace, model_label, revision]):
        raise HTTPException(
            status_code=400,
            detail={
                "status": "error",
                "message": "Missing required parameters: namespace, model_label, revision",
            },
        )

    result = await run_in_threadpool(
        dmf.get_model_card,
        namespace=namespace,
        table=table,
        model_label=model_label,
        revision=revision,
    )
    return result


@router.post(
    "/api/dmf/search",
    tags=["DMF"],
    summary="Search DMF models",
    response_description="Search results with matching models",
)
async def search_dmf_models(
    request: Dict[str, str],
    auth_user: api.AuthUser = Depends(get_current_user),
    dmf: dmf_service.Dmf = Depends(dependencies.get_dmf_service),
) -> Dict[str, Any]:
    """
    Search for models in the DMF registry.

    **Request Body:**
    ```json
    {
        "query": "granite"
    }
    ```

    **Returns:**
    - Search results containing matching models

    **Errors:**
    - 400: Invalid parameters or DMF API error
    - 500: Server configuration error
    """
    query = request.get("query")

    if not query:
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": "Missing required parameter: query"},
        )

    result = await run_in_threadpool(dmf.search_models, query=query)
    return result
