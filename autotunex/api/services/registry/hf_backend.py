# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace Hub model registry — read-only OSS backend.

Read operations (list/detail/card/checkpoints/pull) are backed by the HF Hub.
Write operations (publish/delete) are not supported and return HTTP 501.
get_models/get_all_models delegate to the local registry, which tracks models
this platform has trained (the Hub has no notion of "our jobs").

`huggingface_hub` is imported lazily and ships in the optional `hf` extra.
"""

import logging

from fastapi import HTTPException
from services.registry.base import ModelRegistry
from services.registry.local_backend import LocalRegistry

logger = logging.getLogger(__name__)

_PUBLISH_UNSUPPORTED = {
    "status": "error",
    "message": "Publishing is not supported by the HuggingFace registry backend.",
}


class HuggingFaceRegistry(ModelRegistry):
    def __init__(self, db):
        super().__init__(db)
        self._local = LocalRegistry(db)

    def _api(self):
        from huggingface_hub import HfApi  # lazy

        return HfApi()

    def get_model_detail(
        self, model_label, namespace="base_training", table="model_shared"
    ):
        info = self._api().model_info(model_label)
        return {
            "model_id": getattr(info, "id", model_label),
            "model_label": model_label,
            "revision": "main",
        }

    def get_model_card(self, namespace, table, model_label, revision):
        from huggingface_hub import ModelCard  # lazy

        try:
            card = ModelCard.load(model_label)
            return {"readme": str(card.content), "yaml": ""}
        except Exception as e:
            logger.warning("HF model card load failed for %s: %s", model_label, e)
            raise HTTPException(
                status_code=404,
                detail={
                    "status": "error",
                    "message": f"Model card not available for {model_label}",
                },
            )

    def search_models(self, query):
        results = self._api().list_models(search=query)
        data = [
            {
                "model_id": m.id,
                "model_label": m.id,
                "namespace": m.id.split("/")[0] if "/" in m.id else m.id,
                "base_model": m.id,
                "revision": "main",
            }
            for m in results
        ]
        return {"data": data}

    def get_checkpoints(self, artifact_url):
        info = self._api().model_info(artifact_url)
        return [{"name": s.rfilename} for s in getattr(info, "siblings", [])]

    def pull_all_checkpoint_files(self, artifact_url):
        from huggingface_hub import snapshot_download  # lazy

        return snapshot_download(repo_id=artifact_url)

    def pull_checkpoint_file(self, artifact_url, file_paths):
        from huggingface_hub import hf_hub_download  # lazy

        for f in file_paths:
            hf_hub_download(repo_id=artifact_url, filename=f)
        return True

    async def get_models(self, user_id):
        return await self._local.get_models(user_id)

    async def get_all_models(self):
        return await self._local.get_all_models()

    async def publish_model(self, job_id, metadata, user_id):
        raise HTTPException(status_code=501, detail=_PUBLISH_UNSUPPORTED)

    async def delete_model(self, job_id, user_id):
        raise HTTPException(status_code=501, detail=_PUBLISH_UNSUPPORTED)
