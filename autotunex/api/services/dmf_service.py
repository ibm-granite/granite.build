# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Backward-compat shim. Real code lives in services/registry/dmf_backend.py.

`Dmf` is retained as an alias so existing imports (dependencies.py, mcp_server.py,
job_service.py, server.py) keep working unchanged during the strangler migration.
"""

from services.registry.dmf_backend import DmfRegistry as Dmf  # noqa: F401
