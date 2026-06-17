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

"""Shared auto-discovery for the secret-manager families.

Both ``SpaceSecretManager`` and ``UserSecretManager`` register their backends by
scanning their package for modules named ``<key><BaseName>.py`` exposing a
``<Key><BaseName>`` class. This helper implements that scan once so the two
families cannot drift (e.g. one skipping helper modules like ``factory.py`` and
the other not).
"""

import importlib
import os
from typing import Dict, Type

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def discover_secret_managers(
    package_file: str,
    package_name: str,
    base_class: Type,
    registry: Dict[str, Type],
) -> None:
    """Populate ``registry`` with ``key -> subclass`` for one secret-manager package.

    A module ``<key><BaseName>.py`` (e.g. ``localusersecretmanager.py`` for base
    ``UserSecretManager``) is registered under ``<key>`` (``local``). Modules whose
    name does not end in the lowercased base-class name (e.g. ``factory.py``) and the
    base module itself are skipped, so helper modules are never mistaken for backends.

    No-op if ``registry`` is already populated.
    """
    if len(registry) != 0:
        return
    package_dir = os.path.dirname(package_file)
    base_name = base_class.__name__  # e.g. "UserSecretManager"
    suffix = base_name.lower()  # e.g. "usersecretmanager"
    self_module = os.path.basename(package_file)

    for filename in os.listdir(package_dir):
        if not filename.endswith(".py"):
            continue
        if filename in ("__init__.py", self_module):
            continue
        module_name = filename[:-3]
        # Only "<key><base_name>.py" modules are backends; skip helpers (factory.py).
        if not module_name.lower().endswith(suffix) or len(module_name) <= len(suffix):
            continue
        key_name = module_name[: -len(base_name)].lower()
        type_name = key_name.capitalize() + base_name
        try:
            module = importlib.import_module(f".{module_name}", package=package_name)
            if not hasattr(module, type_name):
                logger.error(
                    "Module %s does not contain expected type class %s",
                    module_name,
                    type_name,
                )
                continue
            handler_class = getattr(module, type_name)
            if isinstance(handler_class, type) and issubclass(
                handler_class, base_class
            ):
                registry[key_name] = handler_class
            else:
                logger.error(
                    "Ignoring %s since it is not a subclass of %s",
                    type_name,
                    base_name,
                )
        except ImportError as e:
            logger.error("Error importing module %s: %s", type_name, e)
        except Exception as e:
            logger.error("Error loading secret manager type from %s: %s", type_name, e)
