#!/usr/bin/env python3

# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import re
from typing import Dict, List, Literal, Optional, Tuple, cast

import wandb
from huggingface_hub import dataset_info, model_info

from gbserver.lineage.openlineage_service import LineageService
from gbserver.lineage.openlineage_utils import (
    get_hf_artifact_uri,
    get_huggingface_hub_url,
    parse_hf_uri,
    parse_hf_url,
)
from gbserver.types.constants import (
    GBSERVER_WANDB_API_KEY,
    GBSERVER_WANDB_BASE_URL,
    GBSERVER_WANDB_ENTITY,
    GBSERVER_WANDB_PROJECT,
)
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class WandBLineageService(LineageService):

    def __init__(self):
        wandb.login(key=GBSERVER_WANDB_API_KEY, host=GBSERVER_WANDB_BASE_URL)
        self._runs = {}

    def _get_run(self, run_id: str, job_name: str):
        if run_id in self._runs:
            return self._runs[run_id]

        run = wandb.init(
            project=GBSERVER_WANDB_PROJECT,
            entity=GBSERVER_WANDB_ENTITY,
            id=run_id,
            name=job_name,
        )

        self._runs[run_id] = run
        return run

    def emit_event(self, event: Dict) -> None:
        try:
            run_id = event["run"]["runId"]
            job_name = event["job"]["name"]
            event_type = event["eventType"]

            run = self._get_run(run_id, job_name)

            for inp in event.get("inputs", []):
                resource_name = self._dataset_name(inp)
                resource_type = self._get_hf_type(inp)
                artifact_type = (
                    resource_type if resource_type in ("model", "dataset", "bucket") else "dataset"
                )

                if self._is_huggingface_resource(inp):
                    self._register_hf_reference(run, inp, resource_name, is_output=False)
                else:
                    artifact = wandb.Artifact(name=resource_name, type=artifact_type, metadata=inp)
                    run.use_artifact(artifact)

            for out in event.get("outputs", []):
                resource_name = self._dataset_name(out)
                resource_type = self._get_hf_type(out)
                artifact_type = (
                    resource_type if resource_type in ("model", "dataset", "bucket") else "dataset"
                )

                if self._is_huggingface_resource(out):
                    self._register_hf_reference(run, out, resource_name, is_output=True)
                else:
                    artifact = wandb.Artifact(name=resource_name, type=artifact_type, metadata=out)
                    run.log_artifact(artifact)

            run_facets = event.get("run", {}).get("facets", {})
            job_facets = event.get("job", {}).get("facets", {})
            namespace = event.get("job", {}).get("namespace", "")

            config_update = {
                "job_name": job_name,
                "job_namespace": namespace,
                "event_type": event_type,
                "producer": event.get("producer", ""),
                "schemaURL": event.get("schemaURL", ""),
            }

            tags = run_facets.get("tags", {})
            if tags:
                config_update["build_id"] = tags.get("build_id", "")
                config_update["target_id"] = tags.get("target_id", "")
                config_update["username"] = tags.get("username", "")
                config_update["space_name"] = tags.get("space_name", "")

            source_code = run_facets.get("source_code", {})
            if source_code:
                config_update["source_code_url"] = source_code.get("url", "")

            job_input_params = run_facets.get("job_input_params")
            if job_input_params is not None:
                config_update["job_input_params"] = job_input_params

            execution_stats = run_facets.get("execution_stats")
            if execution_stats is not None:
                config_update["execution_stats"] = execution_stats

            job_details = run_facets.get("job_details", {})
            if job_details:
                config_update["job_id"] = job_details.get("job_id", "")
                config_update["job_type"] = job_details.get("job_type", "")
                config_update["category"] = job_details.get("category", "")
                config_update["job_status"] = job_details.get("job_status", "")
                config_update["job_started_at"] = job_details.get("job_started_at", "")
                config_update["job_completed_at"] = job_details.get("job_completed_at", "")
                config_update["release_id"] = job_details.get("release_id", "")
                config_update["owner"] = job_details.get("owner", "")
                config_update["job_output_stats"] = job_details.get("job_output_stats", {})

            if "documentation" in job_facets:
                doc = job_facets["documentation"]
                if isinstance(doc, dict) and "description" in doc:
                    config_update["description"] = doc["description"]

            run.config.update(config_update)

            run.summary["last_event_time"] = event.get("eventTime")

            if "tags" in run_facets:
                tags_dict = run_facets["tags"]
                tags_list = [f"{k}={v}" for k, v in tags_dict.items() if not k.startswith("_")]
                if tags_list:
                    run.tags = list(run.tags) + tags_list

            if "documentation" in job_facets:
                doc_facet = job_facets["documentation"]
                if isinstance(doc_facet, dict) and "description" in doc_facet:
                    run.notes = doc_facet["description"]

            run.log({"openlineage_event": event})

            if event_type == "FAIL":
                run.finish(exit_code=1)
                self._runs.pop(run_id, None)

            elif event_type == "COMPLETE":
                run.finish()
                self._runs.pop(run_id, None)

            logger.info("Processed %s event for run %s", event_type, run_id)

        except Exception as e:
            logger.error("Failed to process lineage event: %s", e)
            raise

    def get_run_lineage(self, run_id: str) -> Optional[Dict]:
        try:
            api = wandb.Api()
            path = (
                f"{GBSERVER_WANDB_ENTITY}/{GBSERVER_WANDB_PROJECT}/{run_id}"
                if GBSERVER_WANDB_ENTITY
                else f"{GBSERVER_WANDB_PROJECT}/{run_id}"
            )
            run = api.run(path)
        except Exception:
            return None

        inputs: List[Dict] = []
        outputs: List[Dict] = []

        for artifact in run.used_artifacts():
            # Should we filter out WandB system artifacts here? For now, we include all artifacts to ensure we capture Hugging Face references, but we might want to revisit this logic in the future
            # if self._is_wandb_system_artifact(artifact):
            #     continue
            inputs.append(self._artifact_to_openlineage_dataset(artifact))

        for artifact in run.logged_artifacts():
            # if self._is_wandb_system_artifact(artifact):
            #     continue
            outputs.append(self._artifact_to_openlineage_dataset(artifact))

        job_name = run.config.get("job_name", run.name or "unknown")
        event_type = run.config.get("event_type", "OTHER")
        event_time = run.summary.get("last_event_time", run.createdAt)
        namespace = f"{run.entity}/{run.project}"

        tags_facet: Dict[str, str] = {}
        if run.tags:
            for tag in run.tags:
                if "=" in tag:
                    key, value = tag.split("=", 1)
                    tags_facet[key] = value

        run_facets = {"tags": tags_facet} if tags_facet else {}

        job_facets: Dict[str, Dict] = {}
        if run.notes:
            job_facets["documentation"] = {
                "_producer": "gbserver",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DocumentationJobFacet.json#/$defs/DocumentationJobFacet",
                "description": run.notes,
            }

        return {
            "eventType": event_type,
            "eventTime": event_time,
            "run": {"runId": run_id, "facets": run_facets},
            "job": {"namespace": namespace, "name": job_name, "facets": job_facets},
            "inputs": inputs,
            "outputs": outputs,
            "producer": "https://github.ibm.com/granite-dot-build/gbserver",
            "schemaURL": "https://openlineage.io/spec/2-0-2/OpenLineage.json#/$defs/RunEvent",
        }

    @staticmethod
    def _is_wandb_system_artifact(artifact: wandb.Artifact) -> bool:
        return artifact.type.startswith("wandb-") or artifact.name.startswith("run-")

    @staticmethod
    def _artifact_to_openlineage_dataset(artifact: wandb.Artifact) -> Dict:
        meta = artifact.metadata or {}
        repo_id = meta.get("repo_id")
        artifact_type = meta.get("artifact_type")
        url = meta.get("url")
        if repo_id and artifact_type:
            uri = get_hf_artifact_uri(repo_id, artifact_type)
            namespace = repo_id.split("/")[0] if "/" in repo_id else repo_id
            name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        elif url:
            org, name, artifact_type = parse_hf_url(url)
            namespace = org
            uri = get_hf_artifact_uri(
                f"{org}/{name}",
                cast(Literal["model", "dataset", "bucket"], artifact_type),
            )
        elif meta.get("uri") or meta.get("namespace") or meta.get("name"):
            uri = meta.get("uri", artifact.name)
            namespace = meta.get("namespace", "N/A")
            name = meta.get("name", artifact.name)
        else:
            uri = "N/A"
            namespace = "N/A"
            name = artifact.name
        return {
            "namespace": namespace,
            "name": name,
            "uri": uri,
            "facets": meta,
        }

    def _sanitize_artifact_name(self, name: str) -> str:
        sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
        sanitized = re.sub(r"_+", "_", sanitized)
        return sanitized

    def _dataset_name(self, dataset: Dict) -> str:
        name = dataset.get("name", "unknown")
        return self._sanitize_artifact_name(name)

    def _get_hf_type(self, resource: Dict) -> Optional[str]:
        uri = resource.get("uri", "")
        if uri.startswith("hf://"):
            _, _, artifact_type = parse_hf_uri(uri)
            return artifact_type

        facets = resource.get("facets", {})
        if isinstance(facets, dict):
            artifact_uri = facets.get("artifact_uri", "")
            if artifact_uri.startswith("hf://"):
                _, _, artifact_type = parse_hf_uri(artifact_uri)
                return artifact_type

        namespace = resource.get("namespace", "").lower()
        if "huggingface://datasets" in namespace or "huggingface://dataset" in namespace:
            return "dataset"
        elif "huggingface://models" in namespace or "huggingface://model" in namespace:
            return "model"
        elif "huggingface://buckets" in namespace or "huggingface://bucket" in namespace:
            return "bucket"
        elif "huggingface" in namespace:
            return "dataset"
        return None

    def _is_huggingface_resource(self, resource: Dict) -> bool:
        return self._get_hf_type(resource) is not None

    def _hf_resource_exists(self, resource_id: str, resource_type: str) -> bool:
        try:
            if resource_type == "model":
                model_info(resource_id)
            elif resource_type == "dataset":
                dataset_info(resource_id)
            elif resource_type == "bucket":
                from huggingface_hub import HfApi

                HfApi().bucket_info(bucket_id=resource_id)
            else:
                return False
            return True
        except Exception:
            return False

    def _register_hf_reference(
        self,
        run: wandb.sdk.wandb_run.Run,
        resource: Dict,
        resource_name: str,
        is_output: bool = False,
    ) -> None:
        uri = resource.get("uri", "")
        org, name, _ = parse_hf_uri(uri)
        resource_id = f"{org}/{name}"
        resource_type = self._get_hf_type(resource)

        artifact_type = (
            resource_type if resource_type in ("model", "dataset", "bucket") else "dataset"
        )

        if not self._hf_resource_exists(resource_id, artifact_type):
            hf_url = get_huggingface_hub_url(artifact_type, resource_id)

            artifact = wandb.Artifact(
                name=resource_name,
                type=artifact_type,
                description=f"Hugging Face {resource_type} reference",
                metadata={
                    "url": hf_url,
                    "repo_id": resource_id,
                    "registry": "huggingface",
                    "artifact_type": artifact_type,
                },
            )
            artifact.add_reference(uri=hf_url, name=name, checksum=False)

            if is_output:
                run.log_artifact(artifact)
                logger.info("Logged HF %s output: %s", resource_type, resource_id)
            else:
                run.use_artifact(artifact)
                logger.info("Registered HF %s input: %s", resource_type, resource_id)
        else:
            hf_url = get_huggingface_hub_url(artifact_type, resource_id)

            artifact = wandb.Artifact(
                name=resource_name,
                type=artifact_type,
                description=f"Hugging Face {resource_type}",
                metadata={
                    "uri": hf_url,
                    "repo_id": resource_id,
                    "registry": "huggingface",
                    "artifact_type": artifact_type,
                },
            )

            if is_output:
                run.log_artifact(artifact)
                logger.info("Logging existing HF %s output: %s", resource_type, resource_id)
            else:
                run.use_artifact(artifact)
                logger.info("Using existing HF %s input: %s", resource_type, resource_id)

    def search_lineage_by_tags(
        self, tags: list, limit: int = 10, offset: int = 0
    ) -> Tuple[int, list]:
        try:
            api = wandb.Api()

            project_path = (
                f"{GBSERVER_WANDB_ENTITY}/{GBSERVER_WANDB_PROJECT}"
                if GBSERVER_WANDB_ENTITY
                else GBSERVER_WANDB_PROJECT
            )

            runs = api.runs(
                project_path,
                filters={"tags": {"$in": tags}} if tags else {},
            )

            all_runs = list(runs)
            total_count = len(all_runs)

            paginated_runs = all_runs[offset : offset + limit]

            results = []
            for run in paginated_runs:
                lineage = self.get_run_lineage(run.id)
                if lineage:
                    results.append(lineage)

            logger.info(
                "Found %d runs (page) matching tags: %s, total: %d",
                len(results),
                tags,
                total_count,
            )
            return total_count, results

        except Exception as e:
            logger.error("Failed to search lineage by tags: %s", e)
            return 0, []

    def search_runs_by_artifact(
        self, repo_id: str, limit: int = 10, offset: int = 0
    ) -> Tuple[int, list]:
        try:
            api = wandb.Api()

            project_path = (
                f"{GBSERVER_WANDB_ENTITY}/{GBSERVER_WANDB_PROJECT}"
                if GBSERVER_WANDB_ENTITY
                else GBSERVER_WANDB_PROJECT
            )

            runs = api.runs(project_path)

            matching_runs = []
            for run in runs:
                lineage = self.get_run_lineage(run.id)
                if lineage:
                    inputs = lineage.get("inputs", [])
                    outputs = lineage.get("outputs", [])

                    has_repo = False
                    for inp in inputs:
                        if repo_id == inp.get("facets", {}).get("repo_id", ""):
                            has_repo = True
                            break

                    if not has_repo:
                        for out in outputs:
                            if repo_id == out.get("facets", {}).get("repo_id", ""):
                                has_repo = True
                                break

                    if has_repo:
                        matching_runs.append(lineage)

            total_count = len(matching_runs)

            paginated_results = matching_runs[offset : offset + limit]

            logger.info(
                "Found %d runs (page) with artifact: %s, total: %d",
                len(paginated_results),
                repo_id,
                total_count,
            )
            return total_count, paginated_results

        except Exception as e:
            logger.error("Failed to search lineage by artifact: %s", e)
            return 0, []
