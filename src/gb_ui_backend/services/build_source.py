"""Shared data types for build discovery (K8s and other sources)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class BuildStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SUSPENDED = "suspended"
    SUBMITTED = "submitted"
    DELETED = "deleted"
    INVALID = "invalid"


@dataclass
class BuildResources:
    cpu: str = ""
    memory: str = ""
    gpu: str = ""
    storage: str = ""
    replicas: int = 1


@dataclass
class StepRun:
    uuid: str
    build_id: str
    name: str
    status: BuildStatus
    status_msg: str = ""


@dataclass
class TargetRun:
    uuid: str
    build_id: str
    name: str
    status: BuildStatus
    status_msg: str = ""
    appwrapper_name: str = ""
    appwrapper_state: str = ""
    pod_placement: Dict[str, str] = field(default_factory=dict)


@dataclass
class K8sResource:
    kind: str
    name: str
    namespace: str
    cluster: str
    status: str
    build_status: BuildStatus
    created_time: Optional[datetime] = None
    resources: Optional[BuildResources] = None
    failure_reason: str = ""
    failure_message: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Build:
    uuid: str
    name: str
    space_name: str
    username: str
    status: BuildStatus
    source_uri: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_time: Optional[datetime] = None
    updated_time: Optional[datetime] = None
    failure_reason: str = ""
    failure_message: str = ""
    resources: Optional[BuildResources] = None
    k8s_resources: List[K8sResource] = field(default_factory=list)
    k8s_status: Dict[str, Any] = field(default_factory=dict)
