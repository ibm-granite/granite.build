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

"""Artifact module."""

from enum import StrEnum, auto


class ArtifactType(StrEnum):
    # These are DEPREDCATED, but will be kept so we can deserialize old entries.
    """Artifact Type implementation."""
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
    """Artifact Store Type implementation."""

    LAKEHOUSE = auto()
    COS = auto()
    LOCAL = auto()
