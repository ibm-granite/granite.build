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

"""Out-of-tree plugin discovery via Python packaging entry points.

granite.build's subsystems (environments, secret managers, asset stores, URI
handlers, ...) each keep a registry populated by scanning their own package
directory. This module adds the *second* discovery source: implementations
shipped by a separately-installed package that declares
`[project.entry-points."<group>"]` in its ``pyproject.toml``.

The entry-point table **is** the plugin manifest — no bespoke manifest file is
needed. Each subsystem defines a well-known group name (see the constants
below) and, after its in-tree directory scan, calls one of the iterators here
and folds the results into its own registry using its own key-derivation rule.

Discovery is defensive by construction: a plugin group that cannot be
enumerated, or a single entry point that fails to import/load, is logged and
skipped so that one broken plugin never prevents the core (or its other
plugins) from starting. This mirrors the try/except-and-log style the in-tree
loaders already use.
"""

from collections.abc import Hashable
from importlib.metadata import EntryPoint, entry_points
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Type,
)

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Entry-point group names — the plugin API surface.
#
# Centralized here so a typo becomes an import error rather than a silently
# empty discovery pass. Keep in sync with docs/plugins/README.md.
# ---------------------------------------------------------------------------

# Groups wired to a discovery pass today (see the four registry loaders).
GROUP_ENVIRONMENTS = "gbserver.environments"
GROUP_SECRET_MANAGERS = "gbserver.secret_managers"
GROUP_ASSET_STORES = "gbserver.asset_stores"
GROUP_URI_HANDLERS = "gbserver.uri_handlers"

# Groups reserved for subsystems that will adopt this same mechanism in
# follow-up PRs. Declared now so the plugin package and its docs can be written
# against stable names.
GROUP_AUTH_PROVIDERS = "gbserver.auth_providers"
GROUP_RESILIENCE_STRATEGIES = "gbserver.resilience_strategies"
GROUP_BUILTIN_STEPS = "gbserver.builtin_steps"
GROUP_CLI_PLUGINS = "gbcli.plugins"


# Loaded entry points, cached per group. A group's entry-point table is fixed
# for the life of the process (installed packages don't change under a running
# server), and a single group can be consumed by more than one registry pass —
# the secret-manager group, for instance, feeds both the Space and the User
# family. Loading once and caching means each entry point is imported (and any
# load failure logged) exactly once per group, not once per consuming pass.
_loaded_groups: Dict[str, Tuple[Tuple[str, Any], ...]] = {}


def _clear_entry_point_cache() -> None:
    """Drop the per-group load cache. For tests that swap ``entry_points``."""
    _loaded_groups.clear()


def _load_group(group: str) -> Tuple[Tuple[str, Any], ...]:
    """Enumerate and ``load()`` *group* once, returning ``(name, obj)`` pairs.

    Enumeration failure -> ``()``; a single entry point that fails to load is
    logged (once) and dropped. The result is memoized in ``_loaded_groups``.
    """
    try:
        eps: Iterable[EntryPoint] = entry_points(group=group)
    except Exception as e:  # pragma: no cover - defensive; enumerate rarely fails
        logger.error("Error enumerating plugin entry points for group %s: %s", group, e)
        _loaded_groups[group] = ()
        return ()

    loaded: List[Tuple[str, Any]] = []
    for ep in eps:
        try:
            obj = ep.load()
        except Exception as e:
            # One broken plugin must not take down the core or its siblings.
            logger.error(
                "Error loading plugin entry point %s=%s (group %s): %s",
                ep.name,
                ep.value,
                group,
                e,
            )
            continue
        loaded.append((ep.name, obj))

    result = tuple(loaded)
    _loaded_groups[group] = result
    return result


def iter_entry_point_objects(group: str) -> Iterable[Tuple[str, Any]]:
    """Yield ``(name, loaded_object)`` for every entry point in *group*.

    ``loaded_object`` is whatever the entry point points at — a class, a
    module, a function. Callers that require a class should use
    :func:`iter_entry_point_classes` instead.

    Never raises: a failure to enumerate the group, or to ``load()`` an
    individual entry point, is logged at ERROR and that entry point is skipped.
    A group with no entry points yields nothing (the no-plugin-installed case).
    The group is loaded at most once (see :data:`_loaded_groups`); repeat calls
    for the same group replay the cached objects without re-importing.
    """
    cached = _loaded_groups.get(group)
    if cached is None:
        cached = _load_group(group)
    yield from cached


def iter_entry_point_classes(
    group: str, base_class: Type
) -> Iterable[Tuple[str, Type]]:
    """Yield ``(name, cls)`` for entry points in *group* that are subclasses of *base_class*.

    Two kinds of mismatch are handled differently:

    - A loaded object that is **not a class** is a genuine misconfiguration (the
      entry point points at a function or module where a class is required) and
      is logged at ERROR.
    - A loaded object that **is a class but not a subclass** of *base_class* is
      skipped quietly (DEBUG). A single group may legitimately be consumed by
      more than one registry that filter by base class — the secret-manager
      group, for instance, feeds both the space and user families, so each pass
      sees (and must ignore without complaint) the other family's classes.
    """
    for name, obj in iter_entry_point_objects(group):
        if not isinstance(obj, type):
            logger.error(
                "Ignoring plugin entry point %s (group %s): %r is not a class",
                name,
                group,
                obj,
            )
            continue
        if issubclass(obj, base_class):
            yield name, obj
        else:
            logger.debug(
                "Skipping plugin entry point %s (group %s): %s is not a subclass of %s",
                name,
                group,
                obj.__name__,
                base_class.__name__,
            )


# ---------------------------------------------------------------------------
# PluginRegistrar — one place that owns the register/collision rule.
# ---------------------------------------------------------------------------

# Given a class and its (optional) entry-point name, return the registry keys
# the class should be filed under. The name is None for the in-tree pass (keys
# derived from the class itself) and the entry-point name for the plugin pass.
KeysOf = Callable[[Type, Optional[str]], Iterable[Hashable]]


class PluginRegistrar:
    """Files subsystem implementations into a registry with a uniform rule.

    Every pluggable subsystem keeps a ``dict`` mapping some key (URI scheme,
    environment type name, URI class, ...) to an implementation class. Both the
    in-tree directory scan and the entry-point plugin scan register through this
    one object, so the collision policy lives in a single place:

    **Core wins.** A key already bound to a *different* class is left untouched
    and the newcomer is skipped with a WARNING — so a plugin can only *add* to
    the registry, never silently shadow a built-in (or an earlier plugin).

    The only thing that varies per subsystem is how a class maps to its key(s);
    the caller supplies that as ``keys_of(cls, name)``.
    """

    def __init__(
        self,
        registry: MutableMapping[Any, Type],
        label: str,
        keys_of: KeysOf,
    ) -> None:
        """``registry`` is the subsystem's own dict; mutated in place.

        Typed ``MutableMapping[Any, Type]`` rather than ``Dict[Hashable, Type]``
        so a subsystem can pass its own concretely-keyed registry (e.g.
        ``Dict[str, Type]`` or the asset store's ``Dict[type, Type]``) without a
        variance error — ``dict`` is invariant in its key type.

        ``label`` names the key in log messages (e.g. ``"URI scheme"``).
        ``keys_of(cls, name)`` yields the registry keys ``cls`` belongs under.
        """
        self.registry = registry
        self.label = label
        self.keys_of = keys_of

    def add(self, cls: Type, name: Optional[str] = None) -> None:
        """Register ``cls`` under each of its keys, honoring core-wins.

        Collision is decided **once for the whole class**, not per key: if *any*
        of its keys is already bound to a different value, the entire
        registration is refused. ``keys_by_name`` files one class under several
        key strings (a lowercase alias plus the verbatim name), so a per-key
        check would let a plugin whose lowercase key collides with a built-in
        still slip in under its case-variant key (e.g. built-in ``build`` +
        plugin ``Build``) — shadowing the built-in under a different case and
        defeating core-wins. All-or-nothing closes that hole.

        A class that yields *no* keys (e.g. a discovered handler that forgot to
        override its key method and inherits the base ``[]``) is filed nowhere
        and would silently never resolve; warn so its author gets a diagnostic
        rather than a handler that is quietly ignored at runtime.
        """
        keys = list(self.keys_of(cls, name))
        if not keys:
            logger.warning(
                "%s: %s produced no keys and was not registered; "
                "check that its key derivation is implemented",
                self.label,
                _value_label(cls),
            )
            return

        # Core wins: if any key is already owned by a different value, refuse the
        # whole registration so the newcomer can't shadow it under a sibling key.
        for key in keys:
            existing = self.registry.get(key)
            if existing is not None and existing is not cls:
                logger.warning(
                    "%s %r already registered to %s; ignoring %s",
                    self.label,
                    _key_label(key),
                    _value_label(existing),
                    _value_label(cls),
                )
                return

        for key in keys:
            self.registry[key] = cls

    def discover(self, group: str, base_class: Type) -> None:
        """Run the entry-point plugin pass for ``group`` into this registry.

        Call this *after* the in-tree scan so the core-wins rule protects the
        built-in implementations. **Never raises** — the entry point loader
        guards ``load()``, and each ``add()`` is guarded here so that a plugin
        with a broken key derivation (``keys_of`` raising, or returning a
        non-iterable — e.g. a single class where a list is expected) is logged
        and skipped rather than aborting discovery. These loaders run at package
        import time, so an unguarded failure would prevent startup entirely.
        """
        for name, cls in iter_entry_point_classes(group, base_class):
            try:
                self.add(cls, name)
            except Exception as e:
                logger.error(
                    "Error registering plugin %s=%s (group %s): %s",
                    name,
                    _value_label(cls),
                    group,
                    e,
                )

    def discover_objects(
        self, group: str, predicate: Optional[Callable[[Any], bool]] = None
    ) -> None:
        """Run an entry-point plugin pass for ``group`` where the value is an object.

        The counterpart to :meth:`discover` for subsystems whose registry value
        is not a class subclassing a common base but an object keyed by name —
        the CLI, for instance, registers ``click`` command objects. It routes
        each loaded object through :meth:`add` under the entry-point ``name`` and
        applies the same core-wins rule and per-entry guard as :meth:`discover`.
        Use it with a name-deriving ``keys_of`` (e.g. :func:`keys_by_name`).

        ``predicate`` optionally validates each loaded object before it is filed.
        An object that fails it is skipped with a WARNING — the object-valued
        analogue of :func:`iter_entry_point_classes`' subclass check, so a plugin
        that points its entry point at the wrong kind of object (e.g. a function
        where a ``click`` command is required) is rejected rather than filed and
        mis-handled downstream.
        """
        for name, obj in iter_entry_point_objects(group):
            if predicate is not None and not predicate(obj):
                logger.warning(
                    "Ignoring plugin entry point %s=%s (group %s): "
                    "not a valid %s implementation",
                    name,
                    _value_label(obj),
                    group,
                    self.label,
                )
                continue
            try:
                self.add(obj, name)
            except Exception as e:
                logger.error(
                    "Error registering plugin %s=%s (group %s): %s",
                    name,
                    _value_label(obj),
                    group,
                    e,
                )


def _key_label(key: Hashable) -> Any:
    """Human-readable form of a registry key for log messages.

    Registry keys are usually strings, but the asset-store registry is keyed by
    URI *class*; show its name rather than its ``repr``.
    """
    return getattr(key, "__name__", key)


def _value_label(value: Any) -> Any:
    """Human-readable form of a registry *value* for log messages.

    Values are usually classes (``__name__``), but some subsystems file other
    objects — a ``click`` command has ``name`` instead, and a thunk may have
    neither — so fall back through both before ``repr``. Uses ``is not None``
    rather than truthiness so a present-but-empty label (e.g. ``name=""``) is
    kept as-is instead of being treated as missing.
    """
    for attr in ("__name__", "name"):
        label = getattr(value, attr, None)
        if label is not None:
            return label
    return value


def rebuild_registry(
    registry: MutableMapping[Any, Any], populate: Callable[[], None]
) -> None:
    """Clear ``registry`` in place, then repopulate it via ``populate``.

    The single reset-and-rebuild contract shared by every subsystem loader, so
    the reload-safety fix does not have to be re-derived per subsystem.

    The clear is **in place** (``dict.clear()``), never a rebind to a fresh
    ``{}``: the registries are long-lived ``ClassVar`` dicts and callers may hold
    a reference to the dict object, so rebinding would strand those references on
    the old, now-stale dict.

    Rebuilding from scratch (rather than adding on top of whatever is already
    there) is what makes a loader idempotent and reload-safe. Without it, a
    re-run after ``importlib.reload`` of a subsystem module leaves the registry
    pointing at the pre-reload class: the reloaded module defines a *new* class
    object, but :meth:`PluginRegistrar.add`'s core-wins guard
    (``existing is not cls``) sees the key already bound to the stale class and
    refuses to replace it. Clearing first lets the in-tree scan re-file the
    current classes, and the plugin ``discover`` pass then adds on top with
    core-wins protecting the freshly-registered built-ins.

    ``populate`` does the subsystem's in-tree scan (``registrar.add(...)``)
    followed by its ``registrar.discover(group, base_class)`` plugin pass.
    """
    registry.clear()
    populate()


# ---------------------------------------------------------------------------
# Ready-made ``keys_of`` callbacks — the two supported key-derivation shapes.
#
# There are exactly two ways a subsystem derives a registry key, and every
# subsystem uses one of them:
#
# * ``keys_by_name`` — the key **is the name** (the module name in-tree, the
#   entry-point name for a plugin). This is the industry-standard entry-point
#   convention (cf. the Python packaging guide and OpenStack stevedore, where
#   the entry-point name, owned by the plugin author, is the lookup key). Use it
#   whenever an implementation is selected by a name a user writes in config
#   (environment ``type``, secret-manager ``type``, auth provider, retry
#   strategy ``type``).
#
# * ``keys_from_method`` — the class **advertises its capability keys** by
#   answering a method. Use it when the key is a capability the class dispatches
#   on rather than a chosen name, and one class may own several keys: URI
#   handlers key on the scheme(s) they serve (``git``, ``git+ssh``) and asset
#   stores on the URI class(es) they back. The resolver looks up by that
#   capability, not by plugin name, so name-as-key does not apply here.
#
# A subsystem picks one when constructing its PluginRegistrar instead of
# re-spelling the same lambda.
# ---------------------------------------------------------------------------


def _require_name(name: Optional[str]) -> str:
    """A name-derived ``keys_of`` requires a name; misuse is a bug, not a warning."""
    if not name:
        raise ValueError("this keys_of derives the key from a name, but none was given")
    return name


def keys_by_name(_cls: Type, name: Optional[str]) -> Iterable[Hashable]:
    """Keys = the name lowercased *and* the name exactly as declared.

    The canonical name-as-key helper for every name-keyed subsystem (secret
    managers, environments, auth providers, retry strategies). The declared name
    is preserved verbatim so a plugin's own casing stays reachable: an entry
    point ``AWSBatch`` registers under both ``awsbatch`` and ``AWSBatch``, so a
    build referencing ``type: AWSBatch`` resolves. (An earlier version used
    ``str.capitalize()``, which lowercases every character after the first and
    left internal capitals unreachable.) In-tree names are already a single
    ``str.capitalize()`` (module ``k8s`` -> ``K8s``), so this keeps the
    historical ``{k8s, K8s}`` keys unchanged while no longer mangling plugins.

    Filing both the lowercased and verbatim forms is a superset of a
    lowercase-only key: a caller that lowercases before lookup is unaffected,
    and one that passes the declared casing now also resolves.
    """
    resolved = _require_name(name)
    lowered = resolved.lower()
    # Dedup so an already-lowercase name doesn't file the same key twice.
    return (lowered,) if resolved == lowered else (lowered, resolved)


def keys_from_method(method_name: str) -> KeysOf:
    """Build a ``keys_of`` that asks the class for its capability keys via ``method_name``.

    Used when the registration keys come from the class rather than its name —
    e.g. ``get_supported_schemes`` (URI) or ``get_supported_uri_classes``
    (asset stores). The discovered ``name`` is unused.
    """

    def keys_of(cls: Type, _name: Optional[str]) -> Iterable[Hashable]:
        return getattr(cls, method_name)()

    return keys_of
