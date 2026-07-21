"""Abstract build data models. Copied from gb_dashboard.services.build_source."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, auto
from typing import Any, List, Optional, Set


class BuildStatus(StrEnum):
    """Build status enum matching gbserver."""

    SUBMITTED = auto()
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    INVALID = auto()
    CANCELLED = auto()
    CANCEL_REQUESTED = auto()
    SUSPENDED = auto()
    DELETED = auto()


@dataclass
class BuildStatusInfo:
    """Derived status flags for a build."""

    is_completed: bool = False
    has_failures: bool = False


@dataclass
class Build:
    """Core build data model."""

    uuid: str
    name: str
    space_name: str
    username: str
    status: BuildStatus
    source_uri: str = ""
    created_time: Optional[datetime] = None
    updated_time: Optional[datetime] = None
    data_sources: Set[str] = field(default_factory=set)
    status_info: BuildStatusInfo = field(default_factory=BuildStatusInfo)
    # gbserver-enriched fields
    gbserver_status: Optional[str] = None
    gbserver_space_name: Optional[str] = None
    gbserver_name: Optional[str] = None
    # K8s resources (empty in gbmcp — no K8s dashboard DB)
    k8s_resources: List[Any] = field(default_factory=list)
