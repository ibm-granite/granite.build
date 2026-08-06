# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Local-disk model registry — the zero-config default backend.

Models live under MODELS_DIR (AUTOTUNEX_MODELS_DIR, default
<AUTOTUNE_RESULTS_PATH>/registry) as:

    <MODELS_DIR>/<label>/<job_id>/{<files>, model.json}

`model.json` mirrors the dict shape produced by the DMF backend so frontend / MCP
consumers are unaffected. No IBM dependencies; the existing job DB supplies
ownership and user enrichment.
"""

import json
import logging
import shutil
from pathlib import Path

import paths
from fastapi import HTTPException
from services.registry.base import ModelRegistry

logger = logging.getLogger(__name__)


class LocalRegistry(ModelRegistry):
    def __init__(self, db):
        super().__init__(db)

    @property
    def _models_dir(self) -> Path:
        p = Path(paths.models_dir())
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _iter_manifests(self):
        """Yield (path, dict) for every model.json under MODELS_DIR."""
        for manifest in self._models_dir.glob("*/*/model.json"):
            try:
                yield manifest, json.loads(manifest.read_text())
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Skipping unreadable manifest %s: %s", manifest, e)

    def _model_dir_for_job(self, job_id: str) -> Path | None:
        for manifest, _ in self._iter_manifests():
            if manifest.parent.name == job_id:
                return manifest.parent
        return None

    async def publish_model(self, job_id, metadata, user_id):
        try:
            job = await self.db.get_job(id=job_id, user_id=user_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            results_path = paths.results_path()
            src = Path(results_path) / "output" / str(job_id) / "results"
            dest = self._models_dir / metadata.label / str(job_id)
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            manifest = {
                "model_id": str(job_id),
                "user": None,
                "model_label": metadata.label,
                "base_model": job.get("model"),
                "size": metadata.size,
                "revision": str(job_id),
                "open": False,
                "product_name": "autotunex",
                "files": [p.name for p in dest.iterdir() if p.is_file()],
            }
            (dest / "model.json").write_text(json.dumps(manifest, indent=2))
            return {"status": "Published", "message": str(dest)}
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Local publish_model failed: %s", e, exc_info=True)
            raise HTTPException(status_code=400, detail=f"{e}")

    def get_model_detail(
        self, model_label, namespace="base_training", table="model_shared"
    ):
        for _, data in self._iter_manifests():
            if data.get("model_label") == model_label:
                return data
        return None

    async def _enrich_user(self, model):
        job = await self.db.get_job_by_id(id=model["revision"])
        if job:
            user = (
                await self.db.get_user_by_id(job["user_id"])
                if job.get("user_id")
                else None
            )
            model["user"] = user["email"] if user else None
        else:
            logger.warning("No job found for revision: %s", model["revision"])
            model["user"] = None
        return model

    async def get_models(self, user_id):
        jobs = await self.db.get_jobs(user_id=user_id)
        job_ids = {job["id"] for job in jobs}
        out = []
        for _, data in self._iter_manifests():
            if data.get("revision") in job_ids:
                out.append(await self._enrich_user(data))
        return out

    async def get_all_models(self):
        out = []
        for _, data in self._iter_manifests():
            out.append(await self._enrich_user(data))
        return out

    async def delete_model(self, job_id, user_id):
        job = await self.db.get_job(id=job_id, user_id=user_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        target = self._model_dir_for_job(str(job_id))
        if target and target.exists():
            shutil.rmtree(target)
        return True

    def get_checkpoints(self, artifact_url):
        # artifact_url is treated as the job_id locally.
        target = self._model_dir_for_job(str(artifact_url))
        if not target:
            raise HTTPException(status_code=404, detail="Model not found")
        return [{"name": p.name} for p in target.iterdir() if p.is_file()]

    def pull_all_checkpoint_files(self, artifact_url):
        target = self._model_dir_for_job(str(artifact_url))
        if not target:
            raise HTTPException(status_code=404, detail="Model not found")
        return str(target)  # already local

    def pull_checkpoint_file(self, artifact_url, file_paths):
        target = self._model_dir_for_job(str(artifact_url))
        if not target:
            raise HTTPException(status_code=404, detail="Model not found")
        missing = [f for f in file_paths if not (target / f).exists()]
        if missing:
            raise HTTPException(status_code=404, detail=f"Files not found: {missing}")
        return True

    def get_model_card(self, namespace, table, model_label, revision):
        for manifest, data in self._iter_manifests():
            if data.get("model_label") == model_label:
                readme = manifest.parent / "README.md"
                if readme.exists():
                    return {"readme": readme.read_text(), "yaml": ""}
        raise HTTPException(
            status_code=404,
            detail={
                "status": "error",
                "message": f"Model card not available for {model_label}",
            },
        )

    def search_models(self, query):
        data = []
        for _, d in self._iter_manifests():
            if query.lower() in d.get("model_label", "").lower():
                item = dict(d)
                item["namespace"] = (
                    d.get("namespace") or d.get("product_name") or "autotunex"
                )
                data.append(item)
        return {"data": data}
