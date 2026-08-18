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

"""Unit tests for the entry-point plugin discovery mechanism.

Exercises the shared helper in ``gbcommon.plugins`` and the four loaders wired
to it, without installing a real plugin package: ``entry_points`` is
monkeypatched to return hand-built ``EntryPoint`` objects. The four loaders'
in-tree registries are snapshotted and restored so a test never leaks a fake
class into another test (or the rest of the suite).
"""

from importlib.metadata import EntryPoint

import pytest

import gbcommon.plugins as plugins

# ---------------------------------------------------------------------------
# Helpers: build fake EntryPoint objects that resolve to arbitrary objects.
# EntryPoint.load() imports `module` and getattrs `attr`; we bypass that by
# registering the targets in a fake module we inject into sys.modules.
# ---------------------------------------------------------------------------


def _make_entry_points(monkeypatch, mapping):
    """Install a fake module exposing ``mapping`` values and return EntryPoints.

    ``mapping`` is ``{ep_name: (attr_name, object_or_RAISE)}``. A value of the
    sentinel ``RAISE`` makes that entry point's ``load()`` raise, to test
    per-entry-point failure isolation.
    """
    import sys
    import types

    mod = types.ModuleType("_gbtest_fake_plugin")
    eps = []
    for ep_name, (attr, obj) in mapping.items():
        if obj is not RAISE:
            setattr(mod, attr, obj)
        eps.append(
            EntryPoint(
                name=ep_name,
                value=f"_gbtest_fake_plugin:{attr}",
                group="ignored",
            )
        )
    monkeypatch.setitem(sys.modules, "_gbtest_fake_plugin", mod)
    return eps


RAISE = object()  # sentinel: entry point whose load() should raise


def _patch_entry_points(monkeypatch, group_to_eps):
    """Patch gbcommon.plugins.entry_points to serve the given per-group lists."""

    def fake_entry_points(*, group):
        return group_to_eps.get(group, [])

    monkeypatch.setattr(plugins, "entry_points", fake_entry_points)
    # The group load-cache may already hold results read from the *real*
    # entry_points at package-import time (some subsystems discover on import);
    # swapping the source invalidates it.
    plugins._clear_entry_point_cache()


@pytest.fixture(autouse=True)
def _clear_plugin_group_cache():
    """Isolate the per-group load cache so one test's fake entry points don't
    leak into the next (the cache is keyed by group and lives for the process)."""
    plugins._clear_entry_point_cache()
    yield
    plugins._clear_entry_point_cache()


# ---------------------------------------------------------------------------
# Shared helper: gbcommon.plugins
# ---------------------------------------------------------------------------


class _Base:
    pass


class _Good(_Base):
    pass


class _NotSubclass:
    pass


def test_iter_objects_yields_loaded_objects(monkeypatch):
    eps = _make_entry_points(monkeypatch, {"good": ("Good", _Good)})
    _patch_entry_points(monkeypatch, {"g": eps})
    assert list(plugins.iter_entry_point_objects("g")) == [("good", _Good)]


def test_iter_objects_empty_group_is_noop(monkeypatch):
    _patch_entry_points(monkeypatch, {})
    assert list(plugins.iter_entry_point_objects("nonexistent")) == []


def test_iter_classes_filters_non_subclass_quietly(monkeypatch, caplog):
    """A valid class that isn't a subclass is filtered without an ERROR.

    A shared group (e.g. secret managers) is consumed by more than one registry,
    so each pass legitimately sees the other's classes and must ignore them
    without crying wolf.
    """
    import logging

    caplog.set_level(logging.DEBUG)
    eps = _make_entry_points(
        monkeypatch,
        {"good": ("Good", _Good), "other": ("Other", _NotSubclass)},
    )
    _patch_entry_points(monkeypatch, {"g": eps})
    result = list(plugins.iter_entry_point_classes("g", _Base))
    assert result == [("good", _Good)]
    # Skipped quietly: DEBUG, never ERROR.
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_iter_classes_non_class_is_error(monkeypatch, caplog):
    """An entry point pointing at a non-class is a real misconfig → ERROR."""
    eps = _make_entry_points(monkeypatch, {"fn": ("a_function", lambda: None)})
    _patch_entry_points(monkeypatch, {"g": eps})
    result = list(plugins.iter_entry_point_classes("g", _Base))
    assert result == []
    assert "is not a class" in caplog.text


def test_iter_objects_isolates_load_failure(monkeypatch, caplog):
    """A single failing load() is logged and skipped; siblings still yield."""
    eps = _make_entry_points(
        monkeypatch,
        {"boom": ("Boom", RAISE), "good": ("Good", _Good)},
    )
    _patch_entry_points(monkeypatch, {"g": eps})
    result = list(plugins.iter_entry_point_objects("g"))
    assert result == [("good", _Good)]
    assert "Error loading plugin entry point boom" in caplog.text


def test_enumerate_failure_is_contained(monkeypatch, caplog):
    def boom(*, group):
        raise RuntimeError("registry unreadable")

    monkeypatch.setattr(plugins, "entry_points", boom)
    assert list(plugins.iter_entry_point_objects("g")) == []
    assert "Error enumerating plugin entry points" in caplog.text


# ---------------------------------------------------------------------------
# PluginRegistrar
# ---------------------------------------------------------------------------


def test_registrar_add_files_under_all_keys():
    reg: dict = {}
    registrar = plugins.PluginRegistrar(
        reg, "thing", keys_of=lambda cls, name: [name, name.upper()]
    )
    registrar.add(_Good, "x")
    assert reg == {"x": _Good, "X": _Good}


def test_registrar_core_wins_on_collision(caplog):
    reg = {"x": _Good}
    registrar = plugins.PluginRegistrar(reg, "thing", keys_of=lambda cls, name: [name])
    registrar.add(_NotSubclass, "x")  # different class, same key
    assert reg["x"] is _Good  # core kept
    assert "already registered" in caplog.text


def test_registrar_reregister_same_class_is_quiet(caplog):
    reg: dict = {}
    registrar = plugins.PluginRegistrar(reg, "thing", keys_of=lambda cls, name: [name])
    registrar.add(_Good, "x")
    registrar.add(_Good, "x")  # idempotent: same class, no warning
    assert reg["x"] is _Good
    assert "already registered" not in caplog.text


def test_keys_of_helpers():
    assert list(plugins.keys_by_name_lower(_Good, "Foo")) == ["foo"]
    # An in-tree name is already capitalized (module k8s -> "K8s"), giving the
    # historical {lower, cased} pair.
    assert list(plugins.keys_by_name_cased(_Good, "K8s")) == ["k8s", "K8s"]
    # An already-lowercase name files exactly one key, not a duplicate pair.
    assert list(plugins.keys_by_name_cased(_Good, "foo")) == ["foo"]
    # A plugin's internal capitals survive verbatim so `type: AWSBatch` resolves
    # (the old str.capitalize() would have mangled this to "Awsbatch").
    assert list(plugins.keys_by_name_cased(_Good, "AWSBatch")) == [
        "awsbatch",
        "AWSBatch",
    ]

    class HasKeys:
        @classmethod
        def get_supported_schemes(cls):
            return ["a", "b"]

    keys_of = plugins.keys_from_method("get_supported_schemes")
    assert list(keys_of(HasKeys, None)) == ["a", "b"]


def test_registrar_warns_when_class_produces_no_keys(caplog):
    """A discovered class whose key derivation yields nothing is filed nowhere;
    warn so its author isn't left with a handler that silently never resolves."""
    reg: dict = {}
    registrar = plugins.PluginRegistrar(reg, "URI scheme", keys_of=lambda cls, name: [])
    registrar.add(_Good, "forgot_to_override")
    assert reg == {}
    assert "produced no keys" in caplog.text
    assert "_Good" in caplog.text


def test_group_loaded_once_and_cached(monkeypatch):
    """Two passes over the same group import each entry point once, not twice
    (the secret-manager group feeds both the Space and User families)."""
    load_calls = {"n": 0}

    class _CountingEP(EntryPoint):
        def load(self):  # type: ignore[override]
            load_calls["n"] += 1
            return _Good

    ep = _CountingEP(name="good", value="_x:_Good", group="ignored")
    _patch_entry_points(monkeypatch, {"shared": [ep]})

    first = list(plugins.iter_entry_point_objects("shared"))
    second = list(plugins.iter_entry_point_objects("shared"))
    assert first == second == [("good", _Good)]
    assert load_calls["n"] == 1  # loaded once, replayed from cache the second time


def test_broken_entry_point_logged_once_across_passes(monkeypatch, caplog):
    """A failing load() is logged a single time even when the group is consumed
    by multiple passes — no double error per broken plugin."""
    eps = _make_entry_points(monkeypatch, {"boom": ("Boom", RAISE)})
    _patch_entry_points(monkeypatch, {"shared": eps})

    list(plugins.iter_entry_point_objects("shared"))
    list(plugins.iter_entry_point_objects("shared"))
    assert caplog.text.count("Error loading plugin entry point boom") == 1


def test_registrar_discover_routes_through_add(monkeypatch):
    reg: dict = {}
    registrar = plugins.PluginRegistrar(reg, "thing", keys_of=lambda cls, name: [name])
    eps = _make_entry_points(monkeypatch, {"good": ("Good", _Good)})
    _patch_entry_points(monkeypatch, {"g": eps})
    registrar.discover("g", _Base)
    assert reg == {"good": _Good}


def test_registrar_discover_isolates_keys_of_raising(monkeypatch, caplog):
    """A plugin whose key derivation raises is skipped; siblings still register.

    Guards the startup guarantee: these loaders run at import time, so an
    unguarded keys_of failure would abort subsystem import.
    """

    def keys_of(cls, name):
        if name == "boom":
            raise RuntimeError("bad key derivation")
        return [name]

    reg: dict = {}
    registrar = plugins.PluginRegistrar(reg, "thing", keys_of=keys_of)
    eps = _make_entry_points(
        monkeypatch, {"boom": ("Boom", _Good), "good": ("Good2", _Good)}
    )
    _patch_entry_points(monkeypatch, {"g": eps})
    registrar.discover("g", _Base)
    assert reg == {"good": _Good}  # boom skipped, good survives
    assert "Error registering plugin boom" in caplog.text


def test_registrar_discover_isolates_non_iterable_keys(monkeypatch, caplog):
    """keys_of returning a non-iterable (e.g. a single class) is skipped, not fatal."""
    reg: dict = {}
    # Simulates the common get_supported_uri_classes() -> SingleClass mistake.
    registrar = plugins.PluginRegistrar(reg, "thing", keys_of=lambda cls, name: cls)
    eps = _make_entry_points(monkeypatch, {"bad": ("Bad", _Good)})
    _patch_entry_points(monkeypatch, {"g": eps})
    registrar.discover("g", _Base)  # must not raise
    assert reg == {}
    assert "Error registering plugin bad" in caplog.text


# ---------------------------------------------------------------------------
# URI loader
# ---------------------------------------------------------------------------


@pytest.fixture
def uri_registry_snapshot():
    from gbcommon.uri.uri import URI

    saved = dict(URI.uri_handler_classes)
    yield URI
    URI.uri_handler_classes.clear()
    URI.uri_handler_classes.update(saved)


def test_uri_plugin_adds_new_scheme(monkeypatch, uri_registry_snapshot):
    from gbcommon.uri.uri import URI

    class DummyURI(URI):
        @staticmethod
        def get_supported_schemes():
            return ["dummy"]

    eps = _make_entry_points(monkeypatch, {"dummy": ("DummyURI", DummyURI)})
    _patch_entry_points(monkeypatch, {plugins.GROUP_URI_HANDLERS: eps})

    URI._load_urihandlers()
    assert URI.uri_handler_classes["dummy"] is DummyURI


def test_uri_plugin_collision_core_wins(monkeypatch, caplog, uri_registry_snapshot):
    from gbcommon.uri.uri import URI

    core_for_hf = URI.uri_handler_classes["hf"]

    class ShadowHfURI(URI):
        @staticmethod
        def get_supported_schemes():
            return ["hf"]

    eps = _make_entry_points(monkeypatch, {"hf": ("ShadowHfURI", ShadowHfURI)})
    _patch_entry_points(monkeypatch, {plugins.GROUP_URI_HANDLERS: eps})

    URI._load_urihandlers()
    assert URI.uri_handler_classes["hf"] is core_for_hf
    assert "already registered" in caplog.text


def test_uri_noop_when_no_plugins(monkeypatch, uri_registry_snapshot):
    from gbcommon.uri.uri import URI

    baseline = dict(URI.uri_handler_classes)
    _patch_entry_points(monkeypatch, {})
    URI._load_urihandlers()
    assert URI.uri_handler_classes == baseline


# ---------------------------------------------------------------------------
# Environment loader
# ---------------------------------------------------------------------------


@pytest.fixture
def env_registry_snapshot():
    from gbserver.environment.environment import Environment

    saved = dict(Environment.environment_types)
    yield Environment
    Environment.environment_types.clear()
    Environment.environment_types.update(saved)


def test_environment_plugin_registers_declared_name_verbatim(
    monkeypatch, env_registry_snapshot
):
    """The declared entry-point name is preserved as-is (plus a lowercase alias),
    so a plugin's own casing stays reachable — `type: DummyEnv` resolves. The old
    str.capitalize() would have filed this only under 'Dummyenv' and lost it."""
    from gbserver.environment.environment import Environment

    # A bare subclass; the loader only touches the registry, not instantiation.
    DummyEnv = type("DummyEnv", (Environment,), {})

    eps = _make_entry_points(monkeypatch, {"DummyEnv": ("DummyEnv", DummyEnv)})
    _patch_entry_points(monkeypatch, {plugins.GROUP_ENVIRONMENTS: eps})

    Environment._load_environment_types()
    assert Environment.environment_types["dummyenv"] is DummyEnv
    assert Environment.environment_types["DummyEnv"] is DummyEnv
    # The mangled middle-cased form is no longer produced.
    assert "Dummyenv" not in Environment.environment_types


def test_environment_plugin_lowercase_name_single_key(
    monkeypatch, env_registry_snapshot
):
    """An already-lowercase entry-point name files exactly one key (no dup pair)."""
    from gbserver.environment.environment import Environment

    LowerEnv = type("LowerEnv", (Environment,), {})
    eps = _make_entry_points(monkeypatch, {"lowerenv": ("LowerEnv", LowerEnv)})
    _patch_entry_points(monkeypatch, {plugins.GROUP_ENVIRONMENTS: eps})

    Environment._load_environment_types()
    assert Environment.environment_types["lowerenv"] is LowerEnv


def test_environment_plugin_collision_core_wins(
    monkeypatch, caplog, env_registry_snapshot
):
    from gbserver.environment.environment import Environment

    # Seed a known core type ourselves rather than depending on a specific
    # built-in (e.g. "k8s" is only registered when its optional dependency is
    # installed, which is not the case in the standalone CI env). This keeps the
    # test hermetic under any dependency set and under parallel execution.
    CoreEnv = type("CoreEnv", (Environment,), {})
    Environment.environment_types["seeded"] = CoreEnv
    Environment.environment_types["Seeded"] = CoreEnv

    ShadowEnv = type("ShadowEnv", (Environment,), {})
    eps = _make_entry_points(monkeypatch, {"seeded": ("ShadowEnv", ShadowEnv)})
    _patch_entry_points(monkeypatch, {plugins.GROUP_ENVIRONMENTS: eps})

    Environment._load_environment_types()
    assert Environment.environment_types["seeded"] is CoreEnv  # core kept
    assert "already registered" in caplog.text


# ---------------------------------------------------------------------------
# Assetstore loader
# ---------------------------------------------------------------------------


@pytest.fixture
def assetstore_registry_snapshot():
    from gbserver.asset.assetstore import Assetstore

    saved = dict(Assetstore.assetstore_types)
    yield Assetstore
    Assetstore.assetstore_types.clear()
    Assetstore.assetstore_types.update(saved)


def test_assetstore_plugin_collision_core_wins(
    monkeypatch, caplog, assetstore_registry_snapshot
):
    from gbcommon.uri.hf import HfURI
    from gbserver.asset.assetstore import Assetstore

    core_hf_store = Assetstore.assetstore_types[HfURI]

    class ShadowStore(Assetstore):
        @classmethod
        def get_supported_uri_classes(cls):
            return [HfURI]

    eps = _make_entry_points(monkeypatch, {"shadow": ("ShadowStore", ShadowStore)})
    _patch_entry_points(monkeypatch, {plugins.GROUP_ASSET_STORES: eps})

    # Force the loader past its len-guard by clearing, then re-populating via
    # the in-tree pass + the plugin pass in a single call.
    Assetstore.assetstore_types.clear()
    Assetstore._load_assetstore_types()
    assert Assetstore.assetstore_types[HfURI] is core_hf_store
    assert "already registered" in caplog.text


# ---------------------------------------------------------------------------
# Secret manager loader (shared helper, both families)
# ---------------------------------------------------------------------------


def test_secret_manager_plugin_discovered_alongside_intree(monkeypatch):
    """The plugin pass runs after (and adds to) the in-tree directory scan."""
    from gbserver.spacesecretmanager import spacesecretmanager as sm_module
    from gbserver.spacesecretmanager.spacesecretmanager import SpaceSecretManager
    from gbserver.utils import secretmanager_discovery

    class DummySpaceSM(SpaceSecretManager):
        pass

    eps = _make_entry_points(monkeypatch, {"dummy": ("DummySpaceSM", DummySpaceSM)})
    _patch_entry_points(monkeypatch, {plugins.GROUP_SECRET_MANAGERS: eps})

    registry: dict = {}
    secretmanager_discovery.discover_secret_managers(
        package_file=sm_module.__file__,
        package_name="gbserver.spacesecretmanager",
        base_class=SpaceSecretManager,
        registry=registry,
    )
    # In-tree backends are present AND the plugin's dummy was added on top.
    assert "local" in registry
    assert registry.get("dummy") is DummySpaceSM


def test_secret_manager_plugin_routed_by_base_class(monkeypatch):
    """One group feeds both families; issubclass routing keeps them separate.

    A ``SpaceSecretManager`` subclass declared in the group must NOT land in a
    ``UserSecretManager`` registry.
    """
    from gbserver.spacesecretmanager.spacesecretmanager import SpaceSecretManager
    from gbserver.usersecretmanager import usersecretmanager as usm_module
    from gbserver.usersecretmanager.usersecretmanager import UserSecretManager
    from gbserver.utils import secretmanager_discovery

    class DummySpaceSM(SpaceSecretManager):
        pass

    eps = _make_entry_points(monkeypatch, {"dummy": ("DummySpaceSM", DummySpaceSM)})
    _patch_entry_points(monkeypatch, {plugins.GROUP_SECRET_MANAGERS: eps})

    registry: dict = {}
    secretmanager_discovery.discover_secret_managers(
        package_file=usm_module.__file__,
        package_name="gbserver.usersecretmanager",
        base_class=UserSecretManager,
        registry=registry,
    )
    # The Space-family dummy is filtered out of the User-family registry.
    assert "dummy" not in registry
