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
Types for events.
"""

import dataclasses
from asyncio import Event
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, auto
from typing import Any, Dict, List, Optional, Self, Type

from gbserver.types.artifact import ArtifactType
from gbserver.types.metrics import Metric
from gbserver.types.status import Status
from gbserver.utils.utils import get_time


class BuildLogLevel(StrEnum):
    """Used internally, but so that they have string values that will show up in the database in column with meaningful values.
    However, these may be exposed later if/when we allow searching by level in the db.

    Args:
        StrEnum (_type_): _description_
    """

    INFO = auto()
    ERROR = auto()
    WARNING = auto()


class BuildEventType(StrEnum):
    """The type of an event during a build."""

    NEWARTIFACT_IN_ENVIRONMENT_EVENT = auto()
    NEW_MULTIARTIFACT_IN_ENVIRONMENT_EVENT = auto()
    ARTIFACT_EVENT = auto()
    ARTIFACT_PUSHED_EVENT = auto()
    MESSAGE_EVENT = auto()
    """These are messages generated during the run in the remote compute server. """
    STATUS_EVENT = auto()
    TERMINATE_EVENT = auto()
    WORKLOAD_STATUS_EVENT = auto()
    METRICS_EVENT = auto()
    VALIDATION_DATA_EVENT = auto()
    # LOG_EVENT = auto()
    # """Log events are typically internal log messages produced by the internals of gbserver, but which may still be useful to both developers and end users. """

    def is_internal_event(self: Self) -> bool:
        """Is this a Build framework internal event?"""
        internal_events = (
            BuildEventType.NEWARTIFACT_IN_ENVIRONMENT_EVENT,
            BuildEventType.NEW_MULTIARTIFACT_IN_ENVIRONMENT_EVENT,
            BuildEventType.TERMINATE_EVENT,
        )
        return self in internal_events


@dataclass
class EventPayload:
    """Base class for an event's payload."""

    data: Optional[dict] = field(default_factory=dict, repr=False)

    @classmethod
    def payload_parser(
        cls: Type[Self], event_type: BuildEventType, data: Any
    ) -> "EventPayload":
        """Factory method given event type and data."""
        match event_type:
            case BuildEventType.NEWARTIFACT_IN_ENVIRONMENT_EVENT:
                return ArtifactEventPayload(**data)
            case BuildEventType.NEW_MULTIARTIFACT_IN_ENVIRONMENT_EVENT:
                return MultiArtifactEventPayload(**data)
            case BuildEventType.ARTIFACT_EVENT:
                return CreatedArtifactEventPayload(**data)
            case BuildEventType.ARTIFACT_PUSHED_EVENT:
                return ArtifactPushedEventPayload(**data)
            case BuildEventType.MESSAGE_EVENT:
                return BuildEventMessagePayload(**data)
            case BuildEventType.STATUS_EVENT:
                return BuildEventStatusPayload(**data)
            case BuildEventType.WORKLOAD_STATUS_EVENT:
                return BuildEventWorkloadStatusPayload(**data)
            case BuildEventType.TERMINATE_EVENT:
                return BuildEventTerminatePayload(**data)
            case BuildEventType.METRICS_EVENT:
                # Convert metric dicts back to Metric objects
                metrics_data = data.copy()
                if "metrics" in metrics_data and metrics_data["metrics"]:
                    metrics_data["metrics"] = [
                        Metric(**m) if isinstance(m, dict) else m
                        for m in metrics_data["metrics"]
                    ]
                return BuildEventMetricsPayload(**metrics_data)
            case BuildEventType.VALIDATION_DATA_EVENT:
                return BuildEventValidationDataPayload(**data)
        self = cls(data=data)
        return self


@dataclass
class EntityRunMetadata:
    """Metadata about a run."""

    build_id: Optional[str] = field(default="")
    username: Optional[str] = field(default="")
    type: Optional[str] = field(default="")
    target_name: Optional[str] = field(default="")
    targetrun_id: Optional[str] = field(default="")
    targetsteprun_id: Optional[str] = field(default="")
    targetstep_uri: Optional[str] = field(default="")
    target_step_index: Optional[int] = None
    target_hash: str = ""
    skipped_for_prerun_target_id: str = ""

    @classmethod
    def from_dict(cls: Type[Self], xs: dict) -> Self:
        """Factory method to construct from a dict."""
        return cls(
            build_id=xs.get("build_id", ""),
            username=xs.get("username", ""),
            type=xs.get("type", ""),
            target_name=xs.get("target_name", ""),
            targetrun_id=xs.get("targetrun_id", ""),
            targetsteprun_id=xs.get("targetsteprun_id", ""),
            targetstep_uri=xs.get("targetstep_uri", ""),
            target_step_index=xs.get("target_step_index", None),
            target_hash=xs.get("target_hash", ""),
            skipped_for_prerun_target_id=xs.get("skipped_for_prerun_target_id", ""),
        )

    def to_dict(self: Self) -> dict:
        """Convert into a dict."""
        return dataclasses.asdict(self)


@dataclass
class BuildEvent(Event):
    """An event during a build."""

    run_metadata: EntityRunMetadata
    type: BuildEventType = BuildEventType.NEWARTIFACT_IN_ENVIRONMENT_EVENT
    payload: Optional[EventPayload] = None
    timestamp: datetime = field(default_factory=get_time)
    source: str = "build-framework"

    @classmethod
    def from_dict(cls: Type[Self], xs: dict) -> Self:
        """Factory method to construct from a dict."""
        default_build_event = BuildEvent(run_metadata=EntityRunMetadata())

        run_metadata = xs.get("run_metadata", default_build_event.run_metadata)
        run_metadata = EntityRunMetadata.from_dict(run_metadata)

        event_type = xs.get("type", default_build_event.type)

        payload = xs.get("payload", default_build_event.payload)
        if payload is not None:
            payload = EventPayload.payload_parser(event_type, payload)

        timestamp = xs.get("timestamp", default_build_event.timestamp)

        source = xs.get("source", default_build_event.source)

        return cls(
            run_metadata=run_metadata,
            type=event_type,
            payload=payload,
            timestamp=timestamp,
            source=source,
        )

    def to_dict(self: Self) -> dict:
        """Convert into a dict."""
        return dataclasses.asdict(self)

    def to_json_dict(self: Self) -> dict:
        """Create a dictionary that can be passed to json.dumps() without giving a TypeError."""
        build_event_dict = self.to_dict()
        build_event_dict["timestamp"] = (
            self.timestamp.isoformat()
        )  # Make json.dumps() work.
        build_event_dict["type"] = self.type.name
        # Handle metrics payload - Metric is a Pydantic model that needs explicit serialization
        if isinstance(self.payload, BuildEventMetricsPayload):
            build_event_dict["payload"]["metrics"] = [
                m.model_dump(mode="json") for m in self.payload.metrics
            ]
        # To save on storage in gb_events, remove the empty keys that will default to the same values when we deserialize
        self._remove_empty_key_values(build_event_dict["run_metadata"])
        self._remove_empty_key_values(build_event_dict["payload"])
        return build_event_dict

    def _remove_empty_key_values(self: Self, xs: dict) -> None:
        keys_to_remove = []
        for key, value in xs.items():
            if value is None or value == "" or value == {}:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del xs[key]

    @classmethod
    def from_json_dict(cls: Type[Self], xs: dict) -> "BuildEvent":
        """Create a BuildEvent from a dictionary that was created by json.loads() on a string generated by to_json_dict()."""
        build_event_dict = xs.copy()
        # Fix the time stamp
        timestamp = xs["timestamp"]
        timestamp = datetime.fromisoformat(timestamp)
        build_event_dict["timestamp"] = timestamp
        # Fix the type
        type_name = xs["type"]
        build_event_dict["type"] = BuildEventType(type_name.lower())
        # Create the event
        build_event = BuildEvent.from_dict(xs=build_event_dict)
        return build_event


@dataclass
class CreatedArtifactEventPayload(EventPayload):
    """Payload for created a new aritfact event."""

    # Required initializations
    # uri = lh://lake-staging.cloud/granite_dot_build.public/tables/digit_input data URI
    uri: Optional[str] = None
    # Defaulting initializations
    binding_id: Optional[str] = ""
    type: Optional[ArtifactType] = ArtifactType.UNDEFINED


@dataclass
class ArtifactPushedEventPayload(EventPayload):
    """Payload for when a newly created aritfact gets pushed."""

    # Required initializations
    # uri = lh://lake-staging.cloud/granite_dot_build.public/tables/digit_input data URI
    uri: Optional[str] = None
    binding_id: str = ""
    type: Optional[ArtifactType] = ArtifactType.UNDEFINED


@dataclass
class ArtifactEventPayload(EventPayload):
    """Payload for an artifact event."""

    binding_id: str = ""
    binding: Optional[Any] = None


@dataclass
class MultiArtifactEventPayload(EventPayload):
    """Payload for an event where multiple artifacts are created."""

    artifacts: List[ArtifactEventPayload] = field(default_factory=list)


@dataclass
class BuildEventMessagePayload(EventPayload):
    """Payload for a message event."""

    level: str = ""
    msg: str = ""


@dataclass
class BuildEventTerminatePayload(EventPayload):
    """Payload for a build terminate event."""

    msg: str = ""


@dataclass
class BuildEventStatusPayload(EventPayload):
    """Payload for a build status event."""

    status: Status = Status.PENDING
    msg: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class BuildEventWorkloadStatusPayload(EventPayload):
    """Payload for a workload status event."""

    status: Status = Status.PENDING


@dataclass
class BuildEventMetricsPayload(EventPayload):
    """Payload for a metrics event."""

    metrics: List[Metric] = field(default_factory=list)


@dataclass
class BuildEventValidationDataPayload(EventPayload):
    """
    Validation data, usually produced by dynamic validators
    and recommenders during a dry run.
    """

    data: Any = None


def create_message_event(
    source: str,
    build_id: str,
    level: BuildLogLevel,
    message: str,
    tiggering_event: Optional[BuildEvent] = None,
) -> BuildEvent:
    """Create BuildEvent of type BuildEventEvEntype.MESSAGE_EVENTi, for which the event payload contains the log level and message.
    If provided, the tiggering_event contains the build_id and other run_metadata included in the event.

    Args:
        source: the name of the event source applied to the returned BuildEvent
        build_id (str): id of the build assigned to the returned BuildEvent
        level (BuildLogLevel): log level (info, warning, error)
        message (str): The log message.

        tiggering_event (Optional[BuildEvent], optional): Used to copy in the run_metadata into the returned event. Defaults to None,
        in which case, only the given build_id is included in the run_metadata for the returned event.

    Returns:
        BuildEvent:
    """
    if tiggering_event is None:
        run_metadata = EntityRunMetadata()
        run_metadata.build_id = build_id
    else:
        run_metadata = tiggering_event.run_metadata
        assert run_metadata.build_id == build_id
    event_type = BuildEventType.MESSAGE_EVENT
    _payload = {"level": level.name, "msg": message}
    payload = EventPayload.payload_parser(event_type=event_type, data=_payload)
    return BuildEvent(
        run_metadata=run_metadata, type=event_type, payload=payload, source=source
    )
