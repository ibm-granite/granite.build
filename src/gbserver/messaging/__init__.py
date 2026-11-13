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

"""
messaging package initializer
Automatically discovers all concrete MessagingBase subclasses in
this package and exposes them via `discover_backends()`.
"""

import importlib
import inspect
import pkgutil
from typing import Dict, Type


def discover_backends() -> Dict[str, "type"]:
    """
    Walk every module under `messaging.*`, import it, and return a mapping:
        {lowercase_classname: ConcreteSubclass}
    """
    from gbserver.messaging.messaging_base import MessagingBase  # <— root module import

    backends: Dict[str, Type[MessagingBase]] = {}

    for _, modname, _ in pkgutil.walk_packages(__path__, __name__ + "."):
        try:
            module = importlib.import_module(modname)
        except ImportError:
            continue  # Skip backends with missing dependencies (e.g., aio_pika, nats)

        for obj in vars(module).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, MessagingBase)
                and obj is not MessagingBase
                # Skip backends whose optional dependency is not installed, rather
                # than offering one that would only fail at instantiation.
                and obj.is_available()
            ):
                key = obj.__name__.lower()  # e.g. "rabbitmqbase"
                backends[key] = obj

    return backends
