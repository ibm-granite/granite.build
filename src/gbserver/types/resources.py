"""Resources module."""

from enum import Enum, StrEnum, auto

from pydantic import BaseModel


class ResourceType(StrEnum):
    """Resource Type implementation."""

    gpu = auto()
    cpu = auto()
    memory = auto()


class ResourceSpec(BaseModel):
    """
    Describes a resource (gpu, cpu, memory, etc) specification

    Args:
        BaseModel (_type_): _description_
    """

    name: str
    type: ResourceType
    amount: float
