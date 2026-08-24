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

import logging
import re
from collections import deque
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, cast

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
from gbserver.lineage.wandb_jobstats import LINEAGE_PRODUCER_URL
from gbserver.types.constants import (
    GBSERVER_WANDB_API_KEY,
    GBSERVER_WANDB_BASE_URL,
    GBSERVER_WANDB_ENTITY,
    GBSERVER_WANDB_LOG_LEVEL,
    GBSERVER_WANDB_PROJECT,
    GBSERVER_WANDB_QUIET,
)
from gbserver.utils.logger import get_log_level, get_logger

logger = get_logger(__name__)

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
        # Route wandb's Python-logger messages through gbserver's root handler
        # (CustomFormatter) so any that surface match the gbserver log format,
        # instead of wandb's own handler printing them raw. See issue #181 Task 1.
        wandb_logger = logging.getLogger("wandb")
        wandb_logger.setLevel(get_log_level(GBSERVER_WANDB_LOG_LEVEL))
        for handler in list(wandb_logger.handlers):
            wandb_logger.removeHandler(handler)
        wandb_logger.propagate = True
        self._runs = {}

    def _get_run(self, run_id: str, job_name: str):
        if run_id in self._runs:
            return self._runs[run_id]

        try:
            run = self._init_run(run_id, job_name)
        except Exception:
            # init() registers the run before it can fail, so a failure can leave a
            # partial run behind: a late one on wandb.run, an early one nothing.
            # Release it rather than leaking a live run and its sync thread for the
            # life of this (daemon) process. Not about id reuse — ids are random
            # now — but about not stranding an open run.
            self._finish_quietly(self._partial_run_for(run_id), run_id)
            raise

        self._runs[run_id] = run
        return run

    @staticmethod
    def _partial_run_for(run_id: str) -> Any:
        """The global wandb run, but only if it is the one opened for ``run_id``.

        ``wandb.init`` publishes near the end of a successful init, so a *late*
        init failure leaves this call's partial run on the module-global
        ``wandb.run`` and it must be released. The global is shared, though: it
        may hold an unrelated run, and finishing that would mark a healthy run
        failed. An unreadable id counts as not-ours — releasing the wrong run
        corrupts it, while failing to release ours only leaves the id in use
        until restart.
        """
        run = getattr(wandb, "run", None)
        if run is None:
            return None
        try:
            # ``Run.id`` is a decorated property, so reading it can raise more than
            # AttributeError. This runs inside the caller's ``except`` block, ahead of
            # a bare ``raise`` — letting anything escape here would mask the init
            # error that is the real failure. Unreadable therefore means not-ours.
            observed_id = run.id
        except Exception:  # noqa: BLE001 — any failure to read the id means not-ours
            return None
        return run if observed_id == run_id else None

    @staticmethod
    def _finish_quietly(run: Any, run_id: str) -> None:
        """Finish a run, never letting a teardown error mask the real one."""
        if run is None:
            return
        try:
            run.finish(exit_code=1)
        except Exception as cleanup_error:  # pragma: no cover - defensive
            logger.warning("Failed to release wandb run %s: %s", run_id, cleanup_error)

    def _init_run(self, run_id: str, job_name: str):
        """Open (or resume) the wandb run backing this lineage event."""
        return wandb.init(
            project=GBSERVER_WANDB_PROJECT,
            entity=GBSERVER_WANDB_ENTITY,
            id=run_id,
            name=job_name,
            # "never", not "allow": run ids are fresh random uuids
            # (WandBLineageStore._build_events_for_target), so a resume can never
            # legitimately happen. Under "allow" a uuid collision or a bug that
            # reused an id would silently APPEND to an existing run; "never" turns
            # that into a visible error instead of quiet lineage corruption.
            resume="never",
            # These runs are lineage *events*, not training runs. wandb's default
            # code/git capture would snapshot the lineage-watcher process's own
            # working tree (the gbserver checkout) into every recorded target —
            # producing misleading code/, diff.patch, diff_<hash>.patch files and
            # leaking the recorder's source/diff into wandb. Disable all of it.
            # For the same reason, console capture earns nothing here: there is no
            # workload stdout worth showing in the wandb UI, only the watcher's own
            # logs. It also actively misbehaves — the stderr wrapper is uninstalled
            # *after* finish() marks the run done, so any log line written in that
            # window hits a callback that rejects writes to a finished run and
            # raises UsageError (caught and logged by wandb's redirect.py). With
            # wandb's own logging at DEBUG that is self-inflicted: its teardown
            # messages trip their own wrapper. No redirect, no window, no noise.
            settings=wandb.Settings(
                quiet=GBSERVER_WANDB_QUIET,
                save_code=False,
                disable_code=True,
                disable_git=True,
                console="off",
            ),
        )

    def _release_run(self, run_id: str) -> None:
        """Finish and forget a run so it does not leak into ``self._runs``.

        Run ids are random now, so this is no longer about making an id reusable.
        It still does real work within one process lifetime: ``self._runs`` holds
        open wandb runs, and an event that fails partway through would otherwise
        leave a live run and its background sync thread behind for as long as the
        process lives -- and the lineage watcher is a long-lived daemon.

        A no-op for an id with no open run, so error paths need not know whether
        the run was ever opened or a terminal event already finished it.
        """
        self._finish_quietly(self._runs.pop(run_id, None), run_id)

    # wandb run modes in which a live backend IS available, so artifact
    # registration can proceed. Any other mode (offline/disabled/dryrun/...) is
    # treated as "no live backend" and artifact registration is skipped. Using
    # an online allowlist rather than an offline denylist means a new or renamed
    # non-live mode fails safe (skip) instead of raising against a dead backend.
    _ONLINE_MODES = ("online", "run", "shared")

    def _is_offline(self, run: Any) -> bool:
        """Check whether a wandb run lacks a live backend for artifact ops.

        Prefers the documented ``run.settings.mode`` (e.g. "online"/"offline"/
        "disabled"/"dryrun"), treating anything not in ``_ONLINE_MODES`` as
        offline. Falls back to the ``run.offline`` attribute for wandb versions
        that do not expose settings on the run. Defaults to False (treat as
        online) if neither is available.
        """
        mode = getattr(getattr(run, "settings", None), "mode", None)
        if isinstance(mode, str):
            return mode not in self._ONLINE_MODES
        return bool(getattr(run, "offline", False))

    def _register_artifacts(self, run: Any, event: Dict) -> None:
        """Register input and output artifacts for the run.

        Requires a live wandb backend; callers must skip this in offline mode.
        """
        for direction, resources in (
            ("input", event.get("inputs", [])),
            ("output", event.get("outputs", [])),
        ):
            is_output = direction == "output"
            for resource in resources:
                resource_name = self._dataset_name(resource)
                resource_type = self._get_hf_type(resource)
                artifact_type = (
                    resource_type
                    if resource_type in ("model", "dataset", "bucket")
                    else "dataset"
                )

                if self._is_huggingface_resource(resource):
                    self._register_hf_reference(
                        run, resource, resource_name, is_output=is_output
                    )
                else:
                    artifact = wandb.Artifact(
                        name=resource_name, type=artifact_type, metadata=resource
                    )
                    if is_output:
                        run.log_artifact(artifact)
                    else:
                        run.use_artifact(artifact)

    def emit_event(self, event: Dict) -> None:
        run_id: Optional[str] = None
        try:
            run_id = event["run"]["runId"]
            job_name = event["job"]["name"]
            event_type = event["eventType"]

            run = self._get_run(run_id, job_name)

            # Artifact registration requires a live wandb backend; in offline
            # mode it raises. Skip only the artifact block here (run config,
            # facets, tags and the event log below still apply offline).
            if self._is_offline(run):
                logger.warning(
                    "wandb offline mode; skipping artifact lineage registration for run %s",
                    run_id,
                )
            else:
                self._register_artifacts(run, event)

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
            for key, value in tags.items():
                if not key.startswith("_"):
                    config_update[key] = value

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

            run.config.update(config_update, allow_val_change=True)

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
            # Free the id for the retry, as in _get_run. A terminal event already
            # finished and popped the run, so this is a no-op there — not a
            # second finish().
            if run_id is not None:
                self._release_run(run_id)
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
            "producer": LINEAGE_PRODUCER_URL,
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

    def filter_unrecorded(
        self,
        target_ids: set[str],
        expected_counts: Optional[dict[str, int]] = None,
        on_query_error: Optional[Callable[[Exception], None]] = None,
    ) -> set[str]:
        """Return the subset of ``target_ids`` not yet recorded in wandb.

        Bounded to the given candidates: it queries only for runs tagged with one
        of ``target_ids`` (a server-side ``{"tags": {"$in": [...]}}`` filter)
        rather than scanning the whole project. Each recorded run carries a
        ``target_id=<uuid>`` tag (see WandBLineageStore._build_events_for_target),
        so we derive which candidates are already recorded from that tag and
        return the rest. The tag is the ONLY way to do this: run ids are random
        uuids carrying no target information at all, so a run whose ``target_id``
        tag is missing is invisible here and unreclaimable — which is why the
        emitter must put that tag on every event it writes.

        A single target emits one run per output artifact (or one run when it has
        no outputs), so presence of *a* tagged run does not mean the target is
        fully recorded: a prior scan that crashed part-way through emitting a
        target's runs leaves some runs tagged but the lineage incomplete. To
        avoid masking such a partial record (which would leave a permanent gap,
        since the target would never be re-selected), we *count* the tagged runs
        per candidate and treat a target as recorded only when its run count
        meets or exceeds ``expected_counts[tid]``. When ``expected_counts`` is
        ``None`` or lacks a candidate's key, that candidate falls back to the
        presence check (recorded once >=1 run exists) — the pre-count behavior,
        which keeps older fully-recorded runs (that predate this check) from
        being needlessly re-recorded.

        This is a CORRECTNESS mechanism, not an optimization. Run ids are random
        uuids, so re-recording a target wandb already has writes a second set of
        runs rather than resuming the first — there is no idempotency underneath
        to fall back on. Hence the failure mode is fail CLOSED: on any error this
        returns an EMPTY set (record nothing) and reports the error through
        ``on_query_error``, and the caller is expected to abort and retry rather
        than treat the empty result as "all recorded".

        One consequence of random ids, ACCEPTED and deliberately not fixed here:
        a target whose emission crashed partway through leaves runs that can never
        be completed, since the missing ones would get fresh ids. Re-recording
        emits a full new set, so the target ends up with more runs than
        ``expected_counts[tid]`` and the ``>=`` above passes from then on.

        What that actually costs is small: the re-record DOES write every output,
        so the lineage is complete and correct -- what remains is duplicate runs in
        the wandb UI. The count being inflated would mask a *further* partial
        emission of the same target, but that needs a second mid-emission crash on
        a target whose count is already inflated: a target is only reconciled once
        it is SUCCESS with a finished_at, and a finished target gains no new
        outputs (output_artifacts accumulates during the run, see
        buildrunner.__merge_output_artifacts, and the whole set lives in the one
        StoredTargetRun row), so expected_run_count is stable. A double race, not
        an operational failure mode.

        Do not "fix" this by making run ids deterministic again (a hash of target
        uuid / artifact name / index, or any content-derived scheme). That is where
        this code came from, and it fails far worse: wandb does not allow a DELETED
        run to be recreated under its original id, so once a run is deleted -- which
        happened, intentionally -- a derived id recomputes to that tombstoned id
        forever and the target becomes permanently unrecordable. Commit 5824ae99
        could only stop the futile retries, not record the lineage. Random ids
        remove that failure by construction: a fresh uuid has never been seen, so
        it can never have been deleted. Bounded over-recording is the strictly
        better trade, and it holds without depending on nobody ever deleting a run.

        Nor does a hybrid work (derive the id, fall back to a random one when wandb
        rejects it as deleted): the fallback run is itself unaddressable, so the
        next scan recomputes the same rejected hash and emits *another* random run
        for that output, growing duplicates without bound instead of once.

        Tightening ``>=`` to ``==`` is also wrong: a target with extra runs would
        then read unrecorded on every scan and re-emit a duplicate set each time.

        If this ever does need closing, the missing state is LOCAL, not in the id:
        record which outputs have already been emitted on the granite.build side
        (output_artifacts already says which are expected) instead of inferring it
        by counting what reached the sink. That leaves ids random and deletions
        harmless.
        """
        if not target_ids:
            return set()
        try:
            api = wandb.Api()
            project_path = (
                f"{GBSERVER_WANDB_ENTITY}/{GBSERVER_WANDB_PROJECT}"
                if GBSERVER_WANDB_ENTITY
                else GBSERVER_WANDB_PROJECT
            )
            # Bound the query to this scan's candidates: match only runs whose
            # target_id tag is one of ours. This is a set-membership ($in) filter,
            # not the $regex the IBM wandb backend times out on, so it stays cheap
            # and reliable regardless of how many historical runs exist. A single
            # query serves all candidates; the counting below is a post-filter
            # over the fetched runs, never a per-target query.
            candidate_tags = [f"target_id={tid}" for tid in target_ids]
            run_counts: dict[str, int] = {}
            for run in api.runs(
                project_path, filters={"tags": {"$in": candidate_tags}}
            ):
                # A run can carry more than one target_id tag (e.g. a run resumed
                # across targets), so credit every candidate tag it carries. Use a
                # set so a run counts at most once per target even if the same tag
                # appears twice on it.
                for target_id in {
                    tag.split("=", 1)[1]
                    for tag in (run.tags or [])
                    if tag.startswith("target_id=")
                }:
                    # Only count candidates we actually asked about; ignore empty
                    # values and any unrelated tag the backend returns.
                    if target_id in target_ids:
                        run_counts[target_id] = run_counts.get(target_id, 0) + 1
            recorded: set[str] = set()
            for tid in target_ids:
                count = run_counts.get(tid, 0)
                if count == 0:
                    continue
                expected = (expected_counts or {}).get(tid)
                # No expected count for this target → presence check (>=1);
                # otherwise require the full set of runs.
                if expected is None or count >= expected:
                    recorded.add(tid)
            return target_ids - recorded
        except (
            Exception
        ) as e:  # noqa: BLE001 — best-effort; failure records nothing (fail closed)
            logger.error("Failed to filter unrecorded target ids from wandb: %s", e)
            if on_query_error is not None:
                # Tell the caller this empty set is a fail-CLOSED default, not a
                # verdict that everything is already recorded. Both halves matter:
                # the empty set alone would read as "nothing left to do" and let a
                # caller advance its checkpoint over unprocessed work, so the
                # callback is what makes the difference visible. Reported before
                # returning, and kept inside the handler: the contract is that this
                # method never raises, so a misbehaving callback must not turn a
                # degraded query into one.
                try:
                    on_query_error(e)
                except Exception:  # noqa: BLE001 — callback must not break the contract
                    logger.exception(
                        "on_query_error callback raised while reporting a "
                        "filter_unrecorded failure; ignoring it."
                    )
            # Fail CLOSED: return nothing to record. Run ids are random, so
            # re-recording a target that wandb already has creates DUPLICATE runs
            # instead of resuming the existing ones — an unanswered query must
            # never be read as "not recorded". The caller retries next pass.
            return set()

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
