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
    force: bool = False,
) -> None:
    """Populate ``registry`` with ``key -> subclass`` for one secret-manager package.

    A module ``<key><BaseName>.py`` (e.g. ``localusersecretmanager.py`` for base
    ``UserSecretManager``) is registered under ``<key>`` (``local``). Modules whose
    name does not end in the lowercased base-class name (e.g. ``factory.py``) and the
    base module itself are skipped, so helper modules are never mistaken for backends.

    Unlike the other subsystem loaders (which run once at package import), this is
    also called on an API request path (``gbserver.api.secrets``) that resolves a
    space's secret manager. To keep that path cheap it is a **no-op once the
    registry is populated**; pass ``force=True`` to rebuild anyway (for tests that
    reload modules). When it does (re)build, it goes through the shared
    :func:`~gbcommon.plugins.rebuild_registry` contract so the reload-safe
    clear-in-place semantics are identical to the other subsystems.
    """
    if len(registry) != 0 and not force:
        return
    # Import locally to avoid a hard import-time dependency and keep parity with
    # the other subsystems, which import the registrar lazily too.
    from gbcommon.plugins import (
        GROUP_SECRET_MANAGERS,
        PluginRegistrar,
        keys_by_name,
        rebuild_registry,
    )

    package_dir = os.path.dirname(package_file)
    base_name = base_class.__name__  # e.g. "UserSecretManager"
    suffix = base_name.lower()  # e.g. "usersecretmanager"
    self_module = os.path.basename(package_file)

    # Key = the discovered name (module <key> prefix in-tree, entry-point name for
    # plugins), filed lowercased plus verbatim. Both passes register through this
    # registrar.
    registrar = PluginRegistrar(registry, f"{base_name} key", keys_by_name)

    def populate() -> None:
        for filename in os.listdir(package_dir):
            if not filename.endswith(".py"):
                continue
            if filename in ("__init__.py", self_module):
                continue
            module_name = filename[:-3]
            # Only "<key><base_name>.py" modules are backends; skip helpers (factory.py).
            if not module_name.lower().endswith(suffix) or len(module_name) <= len(
                suffix
            ):
                continue
            key_name = module_name[: -len(base_name)].lower()
            type_name = key_name.capitalize() + base_name
            try:
                module = importlib.import_module(
                    f".{module_name}", package=package_name
                )
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
                    registrar.add(handler_class, key_name)
                else:
                    logger.error(
                        "Ignoring %s since it is not a subclass of %s",
                        type_name,
                        base_name,
                    )
            except ImportError as e:
                logger.error("Error importing module %s: %s", type_name, e)
            except Exception as e:
                logger.error(
                    "Error loading secret manager type from %s: %s", type_name, e
                )

        # Discover secret managers shipped by separately-installed plugin packages.
        # Runs after the in-tree scan so the core-wins rule protects built-ins. The
        # one group feeds both the Space and User families — the ``issubclass``
        # filter in ``discover`` routes each class to the family it belongs to.
        registrar.discover(GROUP_SECRET_MANAGERS, base_class)

    rebuild_registry(registry, populate)
