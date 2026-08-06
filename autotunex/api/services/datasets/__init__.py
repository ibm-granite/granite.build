# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""`datasets` package — focused split of the former dataset_service.py.
`Dataset` (upload/view/delete) and `DatasetIntelligence` (LLM parsing/mapping)."""

from .intelligence import DatasetIntelligence  # noqa: F401
from .service import Dataset  # noqa: F401

__all__ = ["Dataset", "DatasetIntelligence"]
