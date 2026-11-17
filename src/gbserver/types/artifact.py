#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""Artifact type definitions for the build registry."""

from enum import StrEnum, auto


class ArtifactType(StrEnum):
    """Enumeration of supported artifact types."""

    # These are DEPREDCATED, but will be kept so we can deserialize old entries.
    SEED_DATA = auto()
    GENERATED_DATA = auto()
    TUNING_DATA = auto()
    BASE_MODEL = auto()
    TUNED_MODEL = auto()

    UNDEFINED = ""
    # Active artifact types (LH + HuggingFace)
    MODEL = auto()
    DATASET = auto()
    FILESET = auto()
    TABLE = auto()
    BUCKET = auto()  # HuggingFace bucket storage


class ArtifactStoreType(StrEnum):
    """Enumeration of artifact storage backends."""

    LAKEHOUSE = auto()
    COS = auto()
    LOCAL = auto()
