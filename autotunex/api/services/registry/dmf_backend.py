# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from typing import List

import models as api
import paths
from fastapi import HTTPException
from services import db_service
from services.registry.base import ModelRegistry
from utils import extract_artifact_identifier, is_gb_enabled

logger = logging.getLogger(__name__)


class DmfRegistry(ModelRegistry):
    def __init__(self, db: db_service.Database):
        self.db = db

    def dmf_connect(self):
        from lakehouse.assets import Model  # lazy: heavy IBM SDK, only on demand
        from lakehouse.wrappers import LakehouseIceberg

        lakehouse = LakehouseIceberg(config="env")
        model = Model(lh=lakehouse)
        return model

    def get_checkpoints(self, artifact_url: str) -> List[api.ModelInfo]:
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # List parameters
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        """
        namespace (str)       : Namespace under which the model will be pushed. -- See Getting Started > Lakehouse reference
        table (str)           : Table name, either 'model' or 'model_shared'. Defaults to 'model_shared'
        model_label (str)     : Model name / Model label
        revision (str)        : Model checkpoint revision.
        starts_with (str)     : Filters down the results to only display models starting with the specified prefix (if MODEL_LABEL is missing)
        extra_fields ([str])  : Extra model fields to be returned (Check the list of columns in the model table) E.g.: ["size","files"]
        """
        try:
            # P2-FU1: preserved verbatim — `is_gb_enabled` (no parens) is always truthy.
            # Do NOT add () without a behavior-change sign-off.
            if is_gb_enabled and artifact_url:
                model_label, revision = extract_artifact_identifier(artifact_url)
                models = self.dmf_connect().list_models(
                    model_label=model_label,
                    namespace="granite_dot_build.public",
                    revision=revision,
                    table="model_shared",
                    extra_fields=[
                        "model_id",
                        "product_name",
                        "base_model",
                        "size",
                        "files",
                    ],
                )
                models[0]["user"] = None
                logger.debug(f"get_checkpoints_models: {models}")
                return models
            else:
                logger.error(f"Artifact URL is empty: {artifact_url}")
                raise Exception("Artifact url cannot be empty")
        except Exception as e:
            logger.error(f"DMF_get_checkpoints {e}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "message": f"Issue occured on while getting checkpoints from DMF: {e}",
                },
            )

    def pull_all_checkpoint_files(self, artifact_url: str) -> str:
        """
        Pull all checkpoint files for a model to DMF_CACHE.
        Uses model.pull() (full model download) instead of pull_files().
        Returns the local path where files were downloaded.
        """
        try:
            # P2-FU1: preserved verbatim — `is_gb_enabled` (no parens) is always truthy.
            # Do NOT add () without a behavior-change sign-off.
            if is_gb_enabled and artifact_url:
                model_label, revision = extract_artifact_identifier(artifact_url)
                loc = self.dmf_connect().pull(
                    model=model_label,
                    namespace="granite_dot_build.public",
                    table="model_shared",
                    revision=revision,
                    force_download=False,
                )
                logger.info(f"pull_all_checkpoint_files: model downloaded to {loc}")
                return loc
            else:
                logger.error(f"Artifact URL is empty: {artifact_url}")
                raise Exception("Artifact url cannot be empty")
        except Exception as e:
            logger.error(f"DMF_pull_all_checkpoint_files: {e}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "message": f"Issue occured while downloading all checkpoint files from DMF: {e}",
                },
            )

    def pull_checkpoint_file(self, artifact_url: str, file_paths: list[str]) -> bool:
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # List parameters
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        """
        namespace (str)       : Namespace under which the model will be pushed. -- See Getting Started > Lakehouse reference
        table (str)           : Table name, either 'model' or 'model_shared'. Defaults to 'model_shared'
        model_label (str)     : Model name / Model label
        revision (str)        : Model checkpoint revision.
        starts_with (str)     : Filters down the results to only display models starting with the specified prefix (if MODEL_LABEL is missing)
        extra_fields ([str])  : Extra model fields to be returned (Check the list of columns in the model table) E.g.: ["size","files"]
        """
        try:
            # P2-FU1: preserved verbatim — `is_gb_enabled` (no parens) is always truthy.
            # Do NOT add () without a behavior-change sign-off.
            if is_gb_enabled and artifact_url:
                model, revision = extract_artifact_identifier(artifact_url)
                models = self.dmf_connect().pull_files(
                    model=model,
                    namespace="granite_dot_build.public",
                    table="model_shared",
                    revision=revision,
                    files=file_paths,
                )
                logger.info(f"models: {models}")
                return True
            else:
                logger.error(f"Artifact URL is empty: {artifact_url}")
                raise Exception("Artifact url cannot be empty")
        except Exception as e:
            logger.error(f"DMF_pull_checkpoint_file: {e}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "message": f"Issue occured on while downloading checkpoint file from DMF: {e}",
                },
            )

    async def get_models(self, user_id: str) -> List[api.ModelInfo]:
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # List parameters
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        """
        namespace (str)       : Namespace under which the model will be pushed. -- See Getting Started > Lakehouse reference
        table (str)           : Table name, either 'model' or 'model_shared'. Defaults to 'model_shared'
        model_label (str)     : Model name / Model label
        revision (str)        : Model checkpoint revision.
        starts_with (str)     : Filters down the results to only display models starting with the specified prefix (if MODEL_LABEL is missing)
        extra_fields ([str])  : Extra model fields to be returned (Check the list of columns in the model table) E.g.: ["size","files"]
        """
        try:
            if not is_gb_enabled():
                models = self.dmf_connect().list_models(
                    namespace="autotunex",
                    table="model_shared",
                    extra_fields=[
                        "model_id",
                        "product_name",
                        "base_model",
                        "size",
                        "files",
                    ],
                )
                jobs = await self.db.get_jobs(user_id=user_id)
                job_ids = {job["id"] for job in jobs}

                filtered_models = [
                    model for model in models if model["revision"] in job_ids
                ]
                for model in filtered_models:
                    job = await self.db.get_job_by_id(id=model["revision"])
                    if job:
                        user = (
                            await self.db.get_user_by_id(job["user_id"])
                            if job.get("user_id")
                            else None
                        )
                        model["user"] = user["email"]
                    else:
                        logger.warning(
                            f"No job found for revision: {model['revision']}"
                        )
                        model["user"] = None
                return filtered_models
            else:
                return await self.db.get_gb_published_models(user_id=user_id)
        except Exception as e:
            logger.exception(f"DMF_get_models: {e}")
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "message": f"Issue occured while listing models from DMF: {e}",
                },
            )

    async def get_all_models(self) -> List[api.ModelInfo]:
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # List parameters
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        """
        namespace (str)       : Namespace under which the model will be pushed. -- See Getting Started > Lakehouse reference
        table (str)           : Table name, either 'model' or 'model_shared'. Defaults to 'model_shared'
        model_label (str)     : Model name / Model label
        revision (str)        : Model checkpoint revision.
        starts_with (str)     : Filters down the results to only display models starting with the specified prefix (if MODEL_LABEL is missing)
        extra_fields ([str])  : Extra model fields to be returned (Check the list of columns in the model table) E.g.: ["size","files"]
        """
        try:
            if not is_gb_enabled():
                models = self.dmf_connect().list_models(
                    namespace="autotunex",
                    table="model_shared",
                    extra_fields=[
                        "model_id",
                        "product_name",
                        "base_model",
                        "size",
                        "files",
                    ],
                )
                for model in models:
                    job = await self.db.get_job_by_id(id=model["revision"])
                    if job:
                        user = (
                            await self.db.get_user_by_id(job["user_id"])
                            if job.get("user_id")
                            else None
                        )
                        model["user"] = user["email"]
                    else:
                        logger.warning(
                            f"No job found for revision: {model['revision']}"
                        )
                        model["user"] = None
                return models
            else:
                return await self.db.get_gb_published_models()

        except Exception as e:
            logger.error(f"DMF_get_all_models: {e}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "message": f"Issue occured while getting all models list from DMF: {e}",
                },
            )

    async def publish_model(self, job_id: str, metadata: api.DmfMetadata, user_id: str):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # List parameters
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        """
        namespace (str)       : Namespace under which the model will be pushed. -- See Getting Started > Lakehouse reference [required]
        table (str)           : Table name, either 'model' or 'model_shared'. Defaults to 'model_shared'
        label (str)           : Name of your model. Usual format {model_type}-{size}-{variant} [required]
        source_path (str)     : Source directory where the the model checkpoint files are located
        base_model (str)      : Base model of the checkpoint you are pushing -- exmaple: granite-3b-base-v1
        type (str)            : Model type -- example: granite, sandstone [required]
        size (str)            : Model size -- example: 1b, 3b, 13b, 20b [required]
        revision (str)        : Model checkpoint revision. Autogenerated timestamp by default with format `YYYYMMDDTHHMMSS`. E.g.: `20240613T122734`
        variant (str)         : Model variant -- example: base, instruct, lm, etc [required]
        steps (str)           : Model steps -- example: 2500
        open (bool)           : True/False flag to indicate if the checkpoint is open or restricted. Only valid on model_shared table.
        overwrite (str,opt)   : Indicates previous upload of the same model checkpoint should be overwritten. Defaults to False
        product_name (str,opt): Product name
        use_aspera (bool)     : True/False flag to indicate if the file transfer should be done using Aspera (default False)
        """

        try:
            job = await self.db.get_job(id=job_id, user_id=user_id)
            logger.info(job["experiment_name"])
            AUTOTUNE_RESULTS_PATH = paths.results_path()
            output_dir = f"{AUTOTUNE_RESULTS_PATH}/output/{str(job_id)}"
            model_dir = os.path.join(output_dir, "results")
            logger.info("model_dir", model_dir)

            response = self.dmf_connect().push(
                namespace="autotunex",
                table="model_shared",
                label=f"autotunex.{metadata.label}",
                model_dir=model_dir,
                base_model=job["model"],
                revision=job["id"],
                type=metadata.type,
                size=metadata.size,
                variant=metadata.variant,
                product_name="autotunex",
                open=False,
                overwrite=True,
                comments=f"https://vail.cf.res.ibm.com/autotune?tuning={job['id']}",
            )

            logger.info(response)
            return {"status": "Published", "message": response}
        except Exception as e:
            logger.error(
                f"Issue occured while publishing model file to DMF: {e}", exc_info=True
            )
            raise HTTPException(status_code=400, detail=f"{e}")

    async def delete_model(self, job_id: str, user_id: str):
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # Delete a model's checkpoint
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        """
        namespace (str)       : Namespace under which the model will be pushed. -- See Getting Started > Lakehouse reference
        table (str,opt)       : Table name, either 'model' or 'model_shared'. Defaults to 'model_shared'
        model (str)           : Name of your model. Usual format {model_type}-{size}-{variant}-{version}
        revision (str)        : Model checkpoint revision.
        """
        try:
            job = await self.db.get_job(id=job_id, user_id=user_id)

            self.dmf_connect().delete(
                namespace="autotunex",
                table="model_shared",
                model=f"autotunex.{job['experiment_name']}",
                revision=job_id,
            )
            return True
        except Exception as e:
            logger.error(
                f"Issue occured while deleting model in DMF: {e}", exc_info=True
            )
            raise HTTPException(status_code=400, detail=f"{e}")

    def get_model_detail(
        self,
        model_label: str,
        namespace: str = "base_training",
        table: str = "model_shared",
    ):
        models = self.dmf_connect().list_models(
            model_label=model_label,
            namespace=namespace,
            table=table,
            extra_fields=["size", "model_id"],
        )

        if models and len(models) > 0:
            return models[0]
        else:
            return None

    def get_model_card(
        self, namespace: str, table: str, model_label: str, revision: str
    ):
        """
        Fetch model card/README from DMF API.

        Parameters:
            namespace (str): DMF namespace (e.g., "base_training")
            table (str): Table name (e.g., "model_shared")
            model_label (str): Model label/name (e.g., "granite-3.2-8b-instruct")
            revision (str): Model revision (e.g., "r250219a")

        Returns:
            Dict containing model card content

        Raises:
            HTTPException: If API call fails
        """
        import requests

        dmf_api_url = os.getenv("DMF_API_URL", "https://api.dmf.vpc-int.res.ibm.com")
        dmf_token = os.getenv("LAKEHOUSE_TOKEN")

        if not dmf_token:
            logger.error("LAKEHOUSE_TOKEN not set in environment")
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "message": "DMF authentication not configured",
                },
            )

        try:
            url = f"{dmf_api_url}/v2/model/model_readme"
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {dmf_token}",
                "Content-Type": "application/json",
            }
            payload = {
                "token": dmf_token,
                "namespace": namespace,
                "table": table,
                "model_label": model_label,
                "revision": revision,
            }

            logger.info(f"Fetching model card for {model_label}@{revision}")
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            # Check response status
            if response.status_code == 404:
                logger.warning(f"Model card not found for {model_label}@{revision}")
                raise HTTPException(
                    status_code=404,
                    detail={
                        "status": "error",
                        "message": f"Model card not available for {model_label}",
                    },
                )

            response.raise_for_status()

            data = response.json()

            # DMF returns format: {"readme": "...", "yaml": "..."}
            if not data.get("readme"):
                logger.warning(f"Model card has no readme content for {model_label}")
                raise HTTPException(
                    status_code=404,
                    detail={
                        "status": "error",
                        "message": "Model card has no documentation content",
                    },
                )

            logger.info(f"Successfully fetched model card for {model_label}")
            return data

        except requests.exceptions.HTTPError as e:
            # Capture response body for better error messages
            error_detail = ""
            if e.response is not None:
                status_code = e.response.status_code
                try:
                    error_body = e.response.json()
                    error_detail = error_body.get("message", str(error_body))
                except Exception:
                    error_detail = e.response.text[:200] if e.response.text else ""

                # Provide user-friendly messages
                if status_code == 500:
                    user_message = f"Model card not available for {model_label}. The model may not have documentation."
                elif status_code == 404:
                    user_message = f"Model card not found for {model_label}"
                else:
                    user_message = f"Failed to fetch model card: {error_detail}"
            else:
                user_message = f"Failed to fetch model card from DMF: {str(e)}"

            logger.error(
                f"DMF model card fetch failed for {model_label}@{revision}: {e} - {error_detail}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=404
                if e.response and e.response.status_code in [404, 500]
                else 400,
                detail={"status": "error", "message": user_message},
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"DMF model card fetch failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "message": f"Failed to fetch model card from DMF: {str(e)}",
                },
            )

    def search_models(self, query: str):
        """
        Search for models in DMF.

        Parameters:
            query (str): Search query string

        Returns:
            Dict containing search results with 'data' array of models

        Raises:
            HTTPException: If API call fails
        """
        import requests

        dmf_api_url = os.getenv(
            "DMF_API_URL", "https://dmf-datamodel.cash.sl.cloud9.ibm.com"
        )
        dmf_token = os.getenv("LAKEHOUSE_TOKEN")

        if not dmf_token:
            logger.error("LAKEHOUSE_TOKEN not set in environment")
            raise HTTPException(
                status_code=500,
                detail={
                    "status": "error",
                    "message": "DMF authentication not configured",
                },
            )

        try:
            url = f"{dmf_api_url}/v2/model/search"
            headers = {
                "accept": "application/json",
                "Authorization": f"Bearer {dmf_token}",
                "Content-Type": "application/json",
            }
            payload = {"query": query}

            logger.info(f"Searching DMF models with query: {query}")
            response = requests.post(
                url, json=payload, headers=headers, timeout=10, verify=False
            )

            response.raise_for_status()

            data = response.json()
            logger.info(
                f"DMF search returned {len(data.get('data', []))} results for query: {query}"
            )
            return data

        except requests.exceptions.HTTPError as e:
            error_detail = ""
            if e.response is not None:
                try:
                    error_body = e.response.json()
                    error_detail = error_body.get("message", str(error_body))
                except Exception:
                    error_detail = e.response.text[:200] if e.response.text else ""

            logger.error(
                f"DMF model search failed for query '{query}': {e} - {error_detail}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "message": f"Failed to search DMF models: {error_detail or str(e)}",
                },
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"DMF model search failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail={
                    "status": "error",
                    "message": f"Failed to search DMF models: {str(e)}",
                },
            )
