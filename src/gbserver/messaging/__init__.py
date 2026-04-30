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
            if inspect.isclass(obj) and issubclass(obj, MessagingBase) and obj is not MessagingBase:
                key = obj.__name__.lower()  # e.g. "rabbitmqbase"
                backends[key] = obj

    return backends
