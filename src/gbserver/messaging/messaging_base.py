#  messaging_base.py  (generic abstractions)

import abc
from dataclasses import dataclass
from gbserver.messaging import discover_backends
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Union
import yaml


JSON = Union[
    Dict[str, Any],   # objects
    List[Any],        # arrays
    str,
    int,
    float,
    bool,
    None,
]


@dataclass(frozen=True, slots=True)
class Address:
    """
    Canonical identifier for a logical channel.
    * exchange may be None for brokers that don't use exchanges (kafka).
    * routing_key is optional; the backend can append suffixes.
    """
    exchange: str | None
    queue: str
    routing_key: str | None = None

    #helpers
    def rk(self, suffix: str | None = None) -> str:
        if suffix:
            return f"{self.queue}.{self.routing_key}.{suffix}" if self.routing_key else f"{self.queue}.{suffix}"
        return f"{self.queue}.{self.routing_key}" if self.routing_key else self.queue


class MessagingBase(abc.ABC):
    """Abstract interface a concrete broker must satisfy."""

    def __init__(self, addr: Address):
        self.addr: Address = addr

    @abc.abstractmethod
    async def setup(self) -> None: ...

    @abc.abstractmethod
    async def publish(self, payload: JSON, suffix: str) -> None: ...

    @abc.abstractmethod
    async def consume_stream(self, handler: Callable[[bytes,str], Awaitable[None]]) -> None: ...

    @abc.abstractmethod
    async def run(self) -> None: ...

    @abc.abstractmethod
    async def close(self) -> None: ...


    @staticmethod
    def from_yaml(src: Union[str, Path], *, strict_env: bool = True) -> "MessagingBase":

        from gbserver.utils.env_expand import expand_env
        raw = Path(src).read_text() if Path(src).exists() else str(src)

        # expand env vars via shared util
        expanded = expand_env(raw, strict=strict_env)

        data: Dict[str, Any] = yaml.safe_load(expanded)

        backend_name = str(data.get("type", "")).lower()
        if not backend_name:
            raise ValueError("YAML must contain top-level `type:` (e.g. RabbitMQBase)")

        backends = discover_backends()
        cls = backends.get(backend_name)
        if cls is None:
            raise ValueError(
                f"Unknown messaging backend '{backend_name}'. Available: {', '.join(backends)}"
            )

        address = Address(**data.get("address", {}))
        kwargs = {k: v for k, v in data.items() if k not in {"type", "address"}}
        return cls(address, **kwargs)
