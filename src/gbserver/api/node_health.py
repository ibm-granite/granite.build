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

"""
Node health API endpoints for monitoring Kubernetes node failure patterns.

Queries node failure data directly from storage. The NodeHealthTracker
(monitoring/alerting) runs in the BuildWatcher process; this API only
needs read access to the shared storage layer.
"""

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request

from gbserver.api.utils import is_super_admin
from gbserver.storage.node_failure_storage import INodeFailureStorage
from gbserver.storage.stored_node_failure import StoredNodeFailure
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

node_health_api = FastAPI()


def __get_node_failure_storage() -> INodeFailureStorage:
    """Get the node failure storage from admin storage, or raise 503."""
    try:
        from gbserver.storage.singleton_storage import get_admin_storage

        return get_admin_storage().node_failure_storage
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Node health tracking is not enabled",
        )


def _stored_failure_to_dict(failure: StoredNodeFailure) -> Dict[str, Any]:
    """Convert a StoredNodeFailure to a dict for API responses."""
    meta = failure.metadata or {}
    return {
        "node_name": failure.node_name,
        "build_id": failure.build_id,
        "launch_id": failure.launch_id,
        "failure_type": failure.failure_type,
        "timestamp": failure.created_time.isoformat(),
        "retry_count": failure.retry_count,
        "metadata": meta,
        "namespace": meta.get("namespace", ""),
        "cluster": meta.get("cluster", ""),
    }


@node_health_api.get("/summary")
async def get_failure_summary(request: Request) -> Dict[str, Dict]:
    """
    Get infrastructure failure summary for all Kubernetes nodes.

    Returns statistics about K8s infrastructure failures that triggered build
    retries, aggregated across all clusters/namespaces used by gbserver.
    """
    storage = __get_node_failure_storage()
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, storage.get_failure_summary)


@node_health_api.get("/problematic")
async def get_problematic_nodes(
    request: Request,
    threshold: Optional[int] = Query(
        None,
        description="Failure count threshold (defaults to 5)",
    ),
    minutes: Optional[int] = Query(
        None,
        description="Time window in minutes (defaults to 30)",
    ),
) -> Dict[str, List[str]]:
    """
    Get list of Kubernetes nodes exceeding infrastructure failure threshold.
    """
    storage = __get_node_failure_storage()

    t = threshold if threshold is not None else 5
    m = minutes if minutes is not None else 30

    loop = asyncio.get_event_loop()
    nodes = await loop.run_in_executor(
        None, lambda: storage.get_problematic_nodes(threshold=t, minutes=m)
    )
    return {"problematic_nodes": nodes}


@node_health_api.get("/nodes/{node_name}/failures")
async def get_node_failures(
    request: Request,
    node_name: str,
    minutes: Optional[int] = Query(
        None,
        description="Time window in minutes (defaults to 30)",
    ),
) -> Dict:
    """
    Get recent infrastructure failures for a specific Kubernetes node.
    """
    storage = __get_node_failure_storage()

    m = minutes if minutes is not None else 30

    loop = asyncio.get_event_loop()
    failures = await loop.run_in_executor(
        None, lambda: storage.get_recent_failures(node_name=node_name, minutes=m)
    )

    return {
        "node_name": node_name,
        "failure_count": len(failures),
        "failures": [_stored_failure_to_dict(f) for f in failures],
    }


@node_health_api.post("/nodes/{node_name}/resolve")
async def resolve_node_failures(request: Request, node_name: str) -> Dict[str, Any]:
    """
    Mark all unresolved failures for a node as resolved.

    Requires super admin privileges.

    Sets resolved=True on all unresolved failures for the node.
    Records are NOT deleted — they remain for audit purposes.
    """
    if not is_super_admin(request):
        raise HTTPException(
            status_code=403,
            detail="Super admin privileges required to modify node health",
        )

    storage = __get_node_failure_storage()
    loop = asyncio.get_event_loop()
    count = await loop.run_in_executor(None, storage.resolve_node_failures, node_name)

    logger.info("Resolved %d failures for node %s via API", count, node_name)

    return {
        "status": "success",
        "message": f"Resolved {count} failures for {node_name}",
        "resolved_count": count,
    }


@node_health_api.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for node health tracking service."""
    # Verifies storage is set
    __get_node_failure_storage()

    return {
        "status": "healthy",
        "service": "node-health-tracker",
    }
