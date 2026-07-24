# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Backward-compatible shim. Implementation now lives in the `datasets` package
(`service.Dataset`, `intelligence.DatasetIntelligence`). Existing imports
`from services import dataset_service` and `dataset_service.Dataset` keep working.
Do not add logic here — edit services/datasets/."""

from .datasets import Dataset, DatasetIntelligence  # noqa: F401
