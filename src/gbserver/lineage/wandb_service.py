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
from collections import deque
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import wandb
from huggingface_hub import dataset_info, model_info

from gbcommon.uri.hf import HfURI
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

_TAG_CONFIG_KEYS = ("build_id", "target_id", "username", "space_name")
_PASSTHROUGH_FACET_KEYS = ("job_input_params", "execution_stats")
_JOB_DETAIL_KEYS = (
    "job_id",
    "job_type",
    "category",
    "job_status",
    "job_started_at",
    "job_completed_at",
    "release_id",
    "owner",
    "job_output_stats",
)


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
            resume="allow",
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
                    resource_type
                    if resource_type in ("model", "dataset", "bucket")
                    else "dataset"
                )

                if self._is_huggingface_resource(inp):
                    self._register_hf_reference(
                        run, inp, resource_name, is_output=False
                    )
                else:
                    artifact = wandb.Artifact(
                        name=resource_name, type=artifact_type, metadata=inp
                    )
                    run.use_artifact(artifact)

            for out in event.get("outputs", []):
                resource_name = self._dataset_name(out)
                resource_type = self._get_hf_type(out)
                artifact_type = (
                    resource_type
                    if resource_type in ("model", "dataset", "bucket")
                    else "dataset"
                )

                if self._is_huggingface_resource(out):
                    self._register_hf_reference(run, out, resource_name, is_output=True)
                else:
                    artifact = wandb.Artifact(
                        name=resource_name, type=artifact_type, metadata=out
                    )
                    run.log_artifact(artifact)

            run_facets = event.get("run", {}).get("facets", {})
            job_facets = event.get("job", {}).get("facets", {})
            namespace = event.get("job", {}).get("namespace", "")

            config_update: Dict[str, Any] = {
                "job_name": job_name,
                "job_namespace": namespace,
                "event_type": event_type,
                "producer": event.get("producer", ""),
                "schemaURL": event.get("schemaURL", ""),
            }

            tags = run_facets.get("tags", {})
            for key in _TAG_CONFIG_KEYS:
                if key in tags:
                    config_update[key] = tags[key]

            source_code = run_facets.get("source_code", {})
            if source_code.get("url"):
                config_update["source_code_url"] = source_code["url"]

            for key in _PASSTHROUGH_FACET_KEYS:
                if run_facets.get(key) is not None:
                    config_update[key] = run_facets[key]

            job_details = run_facets.get("job_details", {})
            for key in _JOB_DETAIL_KEYS:
                if key in job_details:
                    config_update[key] = job_details[key]

            doc = job_facets.get("documentation", {})
            if isinstance(doc, dict) and doc.get("description"):
                config_update["description"] = doc["description"]

            run.config.update(config_update)

            run.summary["last_event_time"] = event.get("eventTime")

            if "tags" in run_facets:
                tags_dict = run_facets["tags"]
                tags_list = [
                    f"{k}={v}" for k, v in tags_dict.items() if not k.startswith("_")
                ]
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

    def _get_run_lineage(self, run_id: str) -> Optional[Dict]:
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

        config = run.config or {}
        job_name = config.get("job_name", run.name or "unknown")
        event_type = config.get("event_type", "OTHER")
        event_time = run.summary.get("last_event_time", run.createdAt)
        namespace = f"{run.entity}/{run.project}"

        tags_facet: Dict[str, str] = {}
        if run.tags:
            for tag in run.tags:
                if "=" in tag:
                    key, value = tag.split("=", 1)
                    tags_facet[key] = value

        run_facets: Dict[str, Any] = {}
        if tags_facet:
            run_facets["tags"] = tags_facet

        for key in _PASSTHROUGH_FACET_KEYS:
            if config.get(key) is not None:
                run_facets[key] = config[key]

        source_code_url = config.get("source_code_url")
        if source_code_url is not None:
            run_facets["source_code"] = {
                "url": source_code_url,
                "commit_hash": "",
                "path": "",
            }

        job_details = {k: config[k] for k in _JOB_DETAIL_KEYS if k in config}
        if job_details:
            run_facets["job_details"] = job_details

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
        if (
            "huggingface://datasets" in namespace
            or "huggingface://dataset" in namespace
        ):
            return "dataset"
        elif "huggingface://models" in namespace or "huggingface://model" in namespace:
            return "model"
        elif (
            "huggingface://buckets" in namespace or "huggingface://bucket" in namespace
        ):
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
            resource_type
            if resource_type in ("model", "dataset", "bucket")
            else "dataset"
        )

        hf_url = get_huggingface_hub_url(artifact_type, resource_id)
        hf_uri_with_host = HfURI.parse(uri).custom_str()
        metadata = {
            "repo_id": resource_id,
            "registry": "huggingface",
            "artifact_type": artifact_type,
            "uri": hf_uri_with_host,
            "url": hf_url,
        }
        metadata.update(resource)
        metadata["uri"] = hf_uri_with_host
        metadata["url"] = hf_url

        if not self._hf_resource_exists(resource_id, artifact_type):

            artifact = wandb.Artifact(
                name=resource_name,
                type=artifact_type,
                description=f"Hugging Face {resource_type} reference",
                metadata=metadata,
            )
            artifact.add_reference(uri=hf_url, name=name, checksum=False)

            if is_output:
                run.log_artifact(artifact)
                logger.info("Logged HF %s output: %s", resource_type, resource_id)
            else:
                run.use_artifact(artifact)
                logger.info("Registered HF %s input: %s", resource_type, resource_id)
        else:
            artifact = wandb.Artifact(
                name=resource_name,
                type=artifact_type,
                description=f"Hugging Face {resource_type}",
                metadata=metadata,
            )

            if is_output:
                run.log_artifact(artifact)
                logger.info(
                    "Logging existing HF %s output: %s", resource_type, resource_id
                )
            else:
                run.use_artifact(artifact)
                logger.info(
                    "Using existing HF %s input: %s", resource_type, resource_id
                )

    def _resolve_artifact_by_url(self, api, url: str):
        org, name, artifact_type = parse_hf_url(url)
        repo_id = f"{org}/{name}"

        project_path = (
            f"{GBSERVER_WANDB_ENTITY}/{GBSERVER_WANDB_PROJECT}"
            if GBSERVER_WANDB_ENTITY
            else GBSERVER_WANDB_PROJECT
        )

        search_types = (
            [artifact_type] if artifact_type else ["model", "dataset", "bucket"]
        )
        for art_type in search_types:
            try:
                type_obj = api.artifact_type(art_type, project_path)
                for collection in type_obj.collections():
                    for artifact in collection.artifacts():
                        meta = artifact.metadata or {}
                        if meta.get("repo_id") == repo_id:
                            return artifact
            except Exception:
                continue
        return None

    @staticmethod
    def _with_version(name: str) -> str:
        if ":" in name:
            return name
        return f"{name}:latest"

    def _resolve_url_from_uri(self, uri: str) -> str:
        org, name, artifact_type = parse_hf_uri(uri)
        repo_id = f"{org}/{name}"
        return get_huggingface_hub_url(artifact_type, repo_id)

    def _get_artifact_names_from_url(self, url: str) -> List[str]:
        org, name, _ = parse_hf_url(url)
        candidates = [self._sanitize_artifact_name(name)]
        repo_id_sanitized = self._sanitize_artifact_name(f"{org}/{name}")
        if repo_id_sanitized != candidates[0]:
            candidates.append(repo_id_sanitized)
        return candidates

    def get_artifact_graph(
        self,
        artifact_name: Optional[str] = None,
        artifact_url: Optional[str] = None,
        artifact_type: Optional[str] = None,
        max_depth: int = 10,
        direction: str = "downstream",
    ) -> Optional[Dict]:
        try:
            api = wandb.Api()
            root_artifact = None

            if artifact_name:
                hf_prefixes = ("datasets/", "models/", "buckets/", "spaces/")
                has_version = ":" in artifact_name
                if artifact_name.startswith(hf_prefixes):
                    parts = artifact_name.split("/")
                    hf_type = parts[0].rstrip("s")
                    repo_id = "/".join(parts[1:])
                    artifact_url = get_huggingface_hub_url(hf_type, repo_id)
                elif not has_version and artifact_name.count("/") == 1:
                    org, name = artifact_name.split("/")
                    candidates = [
                        self._sanitize_artifact_name(name),
                        self._sanitize_artifact_name(artifact_name),
                    ]
                    for candidate in candidates:
                        try:
                            full_name = f"{GBSERVER_WANDB_ENTITY}/{GBSERVER_WANDB_PROJECT}/{self._with_version(candidate)}"
                            root_artifact = api.artifact(full_name)
                            break
                        except Exception:
                            continue
                elif artifact_name.count("/") < 2:
                    full_name = f"{GBSERVER_WANDB_ENTITY}/{GBSERVER_WANDB_PROJECT}/{self._with_version(artifact_name)}"
                    root_artifact = api.artifact(full_name)
                else:
                    full_name = self._with_version(artifact_name)
                    root_artifact = api.artifact(full_name)

            if root_artifact is None and artifact_url:
                if artifact_url.startswith("hf://"):
                    artifact_url = self._resolve_url_from_uri(artifact_url)
                for candidate in self._get_artifact_names_from_url(artifact_url):
                    try:
                        full_name = f"{GBSERVER_WANDB_ENTITY}/{GBSERVER_WANDB_PROJECT}/{self._with_version(candidate)}"
                        root_artifact = api.artifact(full_name)
                        break
                    except Exception:
                        continue
                if root_artifact is None:
                    root_artifact = self._resolve_artifact_by_url(api, artifact_url)

            if root_artifact is None:
                logger.warning(
                    "Artifact not found: artifact_name=%s, artifact_url=%s",
                    artifact_name,
                    artifact_url,
                )
                return None
        except Exception as e:
            logger.error(
                "Error resolving artifact: artifact_name=%s, artifact_url=%s, error=%s",
                artifact_name,
                artifact_url,
                e,
            )
            return None

        if artifact_type and root_artifact.type != artifact_type:
            raise ValueError(
                f"Artifact type mismatch: expected '{artifact_type}', "
                f"but artifact '{root_artifact.name}' has type '{root_artifact.type}'"
            )

        root_id = root_artifact.qualified_name
        root_node = {
            "id": root_id,
            "node_type": "artifact",
            "name": root_artifact.name,
            "artifact_type": root_artifact.type,
            "is_root": True,
            "metadata": root_artifact.metadata or {},
        }

        if direction == "both":
            down = self._traverse_graph(root_artifact, root_id, max_depth, "downstream")
            up = self._traverse_graph(root_artifact, root_id, max_depth, "upstream")

            node_map: Dict[str, Dict] = {root_id: root_node}
            for n in down["nodes"] + up["nodes"]:
                node_map[n["id"]] = n
            node_map[root_id] = root_node

            edge_set: set = set()
            edges: List[Dict] = []
            for edge in down["edges"] + up["edges"]:
                key = (edge["source"], edge["target"])
                if key not in edge_set:
                    edge_set.add(key)
                    edges.append(edge)

            return {
                "root_id": root_id,
                "nodes": list(node_map.values()),
                "edges": edges,
                "truncated": down["truncated"] or up["truncated"],
            }

        result = self._traverse_graph(root_artifact, root_id, max_depth, direction)
        result["nodes"].insert(0, root_node)
        return result

    def _traverse_graph(
        self,
        root_artifact,
        root_id: str,
        max_depth: int,
        direction: str,
    ) -> Dict:
        nodes: List[Dict] = []
        edges: List[Dict] = []
        visited_artifacts: set = {root_id}
        visited_runs: set = set()
        truncated = False

        queue: deque = deque()
        queue.append(("artifact", root_artifact, 0))

        while queue:
            item_type, item, depth = queue.popleft()

            if depth >= max_depth:
                truncated = True
                continue

            if item_type == "artifact":
                if direction == "downstream":
                    next_runs = list(item.used_by())
                else:
                    try:
                        producer = item.logged_by()
                    except (AttributeError, Exception):
                        producer = None
                    next_runs = [producer] if producer else []

                for run in next_runs:
                    if not hasattr(run, "id") or not hasattr(run, "entity"):
                        continue
                    run_id = f"{run.entity}/{run.project}/{run.id}"
                    edges.append({"source": item.qualified_name, "target": run_id})

                    if run_id not in visited_runs:
                        visited_runs.add(run_id)
                        run_name = getattr(run, "name", None) or run.id
                        run_config = getattr(run, "config", {}) or {}
                        run_tags = list(getattr(run, "tags", None) or [])
                        nodes.append(
                            {
                                "id": run_id,
                                "node_type": "run",
                                "name": run_name,
                                "artifact_type": None,
                                "is_root": False,
                                "metadata": {
                                    "run_id": run.id,
                                    "job_name": run_config.get("job_name", run_name),
                                    "job_namespace": run_config.get(
                                        "job_namespace", ""
                                    ),
                                    "job_type": run_config.get("job_type", ""),
                                    "state": getattr(run, "state", None),
                                    "created_at": getattr(run, "createdAt", None),
                                    "job_id": run_config.get("job_id", ""),
                                    "job_status": run_config.get("job_status", ""),
                                    "job_started_at": run_config.get(
                                        "job_started_at", ""
                                    ),
                                    "job_completed_at": run_config.get(
                                        "job_completed_at", ""
                                    ),
                                    "release_id": run_config.get("release_id", ""),
                                    "category": run_config.get("category", ""),
                                    "owner": run_config.get("owner", ""),
                                    "source_code_details": {
                                        "url": run_config.get("source_code_url", ""),
                                        "commit_hash": "",
                                        "path": "",
                                    },
                                    "job_input_params": run_config.get(
                                        "job_input_params", {}
                                    ),
                                    "execution_stats": run_config.get(
                                        "execution_stats", {}
                                    ),
                                    "job_output_stats": run_config.get(
                                        "job_output_stats", {}
                                    ),
                                },
                                "tags": run_tags,
                            }
                        )
                        queue.append(("run", run, depth + 1))

            elif item_type == "run":
                if direction == "downstream":
                    next_artifacts = list(item.logged_artifacts())
                else:
                    next_artifacts = list(item.used_artifacts())

                for artifact in next_artifacts:
                    if self._is_wandb_system_artifact(artifact):
                        continue

                    art_id = artifact.qualified_name
                    run_id = f"{item.entity}/{item.project}/{item.id}"
                    edges.append({"source": run_id, "target": art_id})

                    if art_id not in visited_artifacts:
                        visited_artifacts.add(art_id)
                        nodes.append(
                            {
                                "id": art_id,
                                "node_type": "artifact",
                                "name": artifact.name,
                                "artifact_type": artifact.type,
                                "is_root": False,
                                "metadata": artifact.metadata or {},
                            }
                        )
                        queue.append(("artifact", artifact, depth + 1))

        return {
            "root_id": root_id,
            "nodes": nodes,
            "edges": edges,
            "truncated": truncated,
        }

    def count_events_by_tags(
        self, tags: list, required_tags: Optional[list] = None
    ) -> int:
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
            required = set(required_tags or [])
            total = 0
            # run.log({"openlineage_event": <dict>}) flattens the dict in
            # history, so there is no top-level "openlineage_event" column.
            # Count rows by a stable flattened sub-key instead.
            marker = "openlineage_event.eventType"
            for run in runs:
                if required and not required.issubset(set(run.tags or [])):
                    continue
                for row in run.scan_history(keys=[marker]):
                    if row.get(marker) is not None:
                        total += 1
            return total
        except Exception as e:
            logger.error("Failed to count events by tags: %s", e)
            return 0

    def count_runs_by_tags(
        self, tags: list, required_tags: Optional[list] = None
    ) -> int:
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
            required = set(required_tags or [])
            total = 0
            for run in runs:
                if required and not required.issubset(set(run.tags or [])):
                    continue
                total += 1
            return total
        except Exception as e:
            logger.error("Failed to count runs by tags: %s", e)
            return 0

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
                lineage = self._get_run_lineage(run.id)
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
