# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Generic plugin registry for AutoTuneX extension points ("seams").

A *seam* is a named extension point (runner, model registry, auth, chat,
storage). Implementations register under an entry-point group and are selected
by name. Selection order: explicit `name` arg -> env var -> historical
fallback. Implementations are imported LAZILY (only when selected), so a
deployment never imports backends it does not use.
"""

import logging
import os
from enum import Enum
from importlib.metadata import entry_points
from typing import Any, Callable

from utils import is_gb_enabled

logger = logging.getLogger(__name__)


class Seam(str, Enum):
    RUNNER = "autotunex.runners"
    REGISTRY = "autotunex.model_registries"
    AUTH = "autotunex.auth_providers"
    CHAT = "autotunex.chat_providers"
    STORAGE = "autotunex.storage_backends"


class UnknownProviderError(RuntimeError):
    """Raised when a selected provider name is not registered for a seam."""


# Test-only injection: seam -> {name: factory}. Checked before entry-points.
_OVERRIDES: dict[Seam, dict[str, Callable[..., Any]]] = {}


def register_override(seam: Seam, name: str, factory: Callable[..., Any]) -> None:
    """Register an in-process provider factory (used by tests)."""
    _OVERRIDES.setdefault(seam, {})[name] = factory


def clear_overrides() -> None:
    """Remove all registered overrides."""
    _OVERRIDES.clear()


def _load_entry_point_factory(seam: Seam, name: str):
    """Return the loaded entry-point object for (seam, name), or None."""
    eps = entry_points(group=seam.value)
    for ep in eps:
        if ep.name == name:
            return ep.load()  # LAZY: target module imports only here
    return None


def _installed_names(seam: Seam) -> list[str]:
    names = list(_OVERRIDES.get(seam, {}).keys())
    names += [ep.name for ep in entry_points(group=seam.value)]
    return sorted(set(names))


_ENV_VARS: dict[Seam, str] = {
    Seam.RUNNER: "AUTOTUNEX_RUNNER",
    Seam.REGISTRY: "AUTOTUNEX_REGISTRY",
    Seam.AUTH: "AUTOTUNEX_AUTH",
    Seam.CHAT: "AUTOTUNEX_CHAT",
    Seam.STORAGE: "AUTOTUNEX_STORAGE",
}


def _can_import(mod: str) -> bool:
    """True if `mod` is importable without importing it.

    Checks ``sys.modules`` first so that test stubs registered without a
    ``__spec__`` (which cause ``find_spec`` to raise ``ValueError`` on
    Python 3.13+) are treated as present.
    """
    import sys
    import importlib.util

    if mod in sys.modules:
        return True
    try:
        return importlib.util.find_spec(mod) is not None
    except ValueError:
        # Defense-in-depth: find_spec raises ValueError for a module in sys.modules
        # with __spec__=None. The sys.modules check above already covers that case
        # today; this guard stays in case that check is ever narrowed.
        return True


def _fallback(seam: Seam) -> str:
    """Historical default per seam (preserves pre-plugin behavior)."""
    if seam == Seam.REGISTRY:
        # Import-probe (not GB-coupled): IBM deployments have lakehouse installed
        # -> dmf (preserves today's behavior); pure-OSS -> local disk.
        return "dmf" if _can_import("lakehouse") else "local"
    if seam in (Seam.RUNNER, Seam.STORAGE):
        return "gb" if is_gb_enabled() else "local"
    if seam == Seam.AUTH:
        # Mirror auth.OIDC_ENABLED exactly: all three OIDC vars must be set.
        return (
            "w3id"
            if (
                os.getenv("OIDC_CLIENT_ID")
                and os.getenv("OIDC_CLIENT_SECRET")
                and os.getenv("OIDC_SECURITY_ENDPOINT")
            )
            else "dev"
        )
    if seam == Seam.CHAT:
        # Probe: a configured OpenAI-compatible endpoint (Ollama/LM Studio/OpenAI)
        # wins; otherwise the historical LiteLLM default.
        return "openai_compatible" if os.getenv("OPENAI_BASE_URL") else "litellm"
    raise UnknownProviderError(f"No fallback defined for seam {seam.value!r}")


def _select_name(seam: Seam, name: str | None) -> str:
    if name is not None:
        return name
    env_val = os.getenv(_ENV_VARS[seam])
    if env_val:
        return env_val
    return _fallback(seam)


def resolve(seam: Seam, name: str | None = None, **kwargs) -> Any:
    """Resolve and instantiate the named provider for `seam`."""
    selected = _select_name(seam, name)

    override = _OVERRIDES.get(seam, {}).get(selected)
    if override is not None:
        logger.info("Resolved seam %s -> %r (override)", seam.value, selected)
        return override(**kwargs)

    factory = _load_entry_point_factory(seam, selected)
    if factory is not None:
        logger.info("Resolved seam %s -> %r (entry-point)", seam.value, selected)
        return factory(**kwargs)

    available = _installed_names(seam)
    logger.error(
        "No provider %r registered for seam %s. Installed: %s",
        selected,
        seam.value,
        available,
    )
    raise UnknownProviderError(
        f"No provider named {selected!r} registered for group "
        f"{seam.value!r}. Installed providers: {available or '[]'}."
    )
