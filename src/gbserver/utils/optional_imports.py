# Copyright Granite.Build Authors
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

"""Centralized availability flags for optional dependencies.

Import a flag to check whether an optional package is installed,
instead of scattering try/except ImportError blocks across the codebase.

Usage::

    from gbserver.utils.optional_imports import HAS_LAKEHOUSE

    if HAS_LAKEHOUSE:
        from lakehouse import LakehouseIceberg
        ...
"""

HAS_LAKEHOUSE: bool
try:
    import lakehouse  # noqa: F401

    HAS_LAKEHOUSE = True
except ImportError:
    HAS_LAKEHOUSE = False

HAS_SKYPILOT: bool
try:
    import sky  # noqa: F401

    HAS_SKYPILOT = True
except ImportError:
    HAS_SKYPILOT = False

HAS_IBM_SDK: bool
try:
    import ibm_cloud_sdk_core  # noqa: F401

    HAS_IBM_SDK = True
except ImportError:
    HAS_IBM_SDK = False

HAS_RABBITMQ: bool
try:
    import aio_pika  # noqa: F401

    HAS_RABBITMQ = True
except ImportError:
    HAS_RABBITMQ = False

HAS_NATS: bool
try:
    import nats  # noqa: F401

    HAS_NATS = True
except ImportError:
    HAS_NATS = False

HAS_ASYNCSSH: bool
try:
    import asyncssh  # noqa: F401

    HAS_ASYNCSSH = True
except ImportError:
    HAS_ASYNCSSH = False
