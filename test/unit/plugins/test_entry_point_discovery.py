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


def test_registrar_core_wins_is_all_or_nothing_across_keys(caplog):
    """A collision on any one key refuses the whole registration.

    With keys_by_name a class is filed under both a lowercase alias and its
    verbatim name. A newcomer colliding on the lowercase key must not still slip
    in under its case-variant key (built-in `build` vs plugin `Build`), which
    would shadow the built-in under a different case and defeat core-wins.
    """
    reg = {"build": _Good}  # built-in, filed under its lowercase key
    registrar = plugins.PluginRegistrar(reg, "thing", plugins.keys_by_name)
    registrar.add(_NotSubclass, "Build")  # keys: {"build", "Build"}
    assert reg == {"build": _Good}  # newcomer filed under neither key
    assert "already registered" in caplog.text


def test_registrar_reregister_same_class_is_quiet(caplog):
    reg: dict = {}
    registrar = plugins.PluginRegistrar(reg, "thing", keys_of=lambda cls, name: [name])
    registrar.add(_Good, "x")
    registrar.add(_Good, "x")  # idempotent: same class, no warning
    assert reg["x"] is _Good
    assert "already registered" not in caplog.text


def test_keys_of_helpers():
    # The single canonical name-as-key helper: {lower, verbatim}, deduped.
    # An in-tree name is already capitalized (module k8s -> "K8s").
    assert list(plugins.keys_by_name(_Good, "K8s")) == ["k8s", "K8s"]
    # An already-lowercase name files exactly one key, not a duplicate pair.
    assert list(plugins.keys_by_name(_Good, "foo")) == ["foo"]
    assert list(plugins.keys_by_name(_Good, "Foo")) == ["foo", "Foo"]
    # A plugin's internal capitals survive verbatim so `type: AWSBatch` resolves
    # (the old str.capitalize() would have mangled this to "Awsbatch").
    assert list(plugins.keys_by_name(_Good, "AWSBatch")) == [
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


def test_uri_loader_rebuilds_registry(monkeypatch, uri_registry_snapshot):
    """The loader rebuilds the registry from scratch on every run: a stale entry
    left over from a prior registration is replaced by the module's current
    class, not kept by the core-wins guard.

    This guards the reload path the test conftest relies on. After
    ``importlib.reload(gbcommon.uri.git)`` the module's ``GitURI`` is a brand-new
    class object; re-running ``_load_urihandlers`` must register that new object.
    A loader that reused the registry would hit core-wins and keep the
    pre-reload class, so ``URI.get_uri`` would hand back instances whose class
    differs from ``gbcommon.uri.git.GitURI`` — defeating any ``monkeypatch`` that
    targets the module attribute (the exact regression behind the git-base
    resolver test cloning for real instead of using its patched clone).
    """
    import gbcommon.uri.git
    from gbcommon.uri.uri import URI

    # A distinct class standing in for a stale pre-reload registration.
    class StaleGitURI(gbcommon.uri.git.GitURI):
        pass

    _patch_entry_points(monkeypatch, {})
    schemes = gbcommon.uri.git.GitURI.get_supported_schemes()
    for scheme in schemes:
        URI.uri_handler_classes[scheme] = StaleGitURI

    URI._load_urihandlers()

    for scheme in schemes:
        assert URI.uri_handler_classes[scheme] is gbcommon.uri.git.GitURI


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
    """A plugin declaring an existing built-in's key cannot shadow it.

    The loader rebuilds the registry from scratch each call (in-tree scan first,
    plugin ``discover`` last), so we let the plugin collide with a genuine
    in-tree environment. ``bash`` is the one built-in guaranteed present under
    any dependency set (the local executor has no optional dependency), so the
    test stays hermetic in standalone CI without pre-seeding — which the rebuild
    would clear anyway.
    """
    from gbserver.environment.environment import Environment

    class ShadowBash(Environment):
        pass

    eps = _make_entry_points(monkeypatch, {"bash": ("ShadowBash", ShadowBash)})
    _patch_entry_points(monkeypatch, {plugins.GROUP_ENVIRONMENTS: eps})

    Environment._load_environment_types()
    # The in-tree Bash class was registered first, so the plugin is refused.
    assert Environment.environment_types["bash"] is not ShadowBash
    assert Environment.environment_types["bash"].__name__ == "Bash"
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

    # force=True re-exercises the loader (clear + in-tree scan + plugin pass)
    # even though the registry is already populated from import.
    Assetstore._load_assetstore_types(force=True)
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


# ---------------------------------------------------------------------------
# Shared reset/rebuild lifecycle: rebuild_registry
#
# The four loaders above route their reset through this one contract. These
# tests pin the contract itself and its reload-safety for each newly-converted
# loader (URI's is test_uri_loader_rebuilds_registry above).
# ---------------------------------------------------------------------------


def test_rebuild_registry_clears_in_place():
    """The registry is cleared IN PLACE (not rebound) and refilled by populate.

    In place matters: callers (and ClassVar consumers) may hold a reference to
    the dict object, so the same object must survive the rebuild.
    """
    registry = {"stale": object()}
    identity = id(registry)
    marker = object()

    def populate():
        registry["fresh"] = marker

    plugins.rebuild_registry(registry, populate)

    assert id(registry) == identity  # same object, not rebound
    assert "stale" not in registry  # cleared
    assert registry["fresh"] is marker  # repopulated


def test_environment_loader_rebuilds_registry(monkeypatch, env_registry_snapshot):
    """A stale registration is replaced by the loader's fresh in-tree class,
    rather than kept by the core-wins guard (the PR #289 class of bug — the
    environment loader previously never cleared)."""
    from gbserver.environment.environment import Environment

    class StaleEnv(Environment):
        pass

    _patch_entry_points(monkeypatch, {})
    Environment.environment_types["stale-key"] = StaleEnv

    Environment._load_environment_types()

    # The stale key is gone (the loader rebuilt from scratch), and a real
    # in-tree environment resolves to the module's current class.
    assert "stale-key" not in Environment.environment_types


def test_assetstore_loader_force_rebuilds(monkeypatch, assetstore_registry_snapshot):
    """A forced rebuild clears in place, so a stale entry left in the populated
    registry is dropped and the in-tree class re-filed (reload-safe)."""
    from gbcommon.uri.hf import HfURI
    from gbserver.asset.assetstore import Assetstore

    core_hf_store = Assetstore.assetstore_types[HfURI]

    class StaleStore(Assetstore):
        @classmethod
        def get_supported_uri_classes(cls):
            return [HfURI]

    _patch_entry_points(monkeypatch, {})
    Assetstore.assetstore_types[HfURI] = StaleStore

    Assetstore._load_assetstore_types(force=True)

    # The in-tree scan re-files HfURI's real core store, replacing the stale one.
    assert Assetstore.assetstore_types[HfURI] is core_hf_store


def test_secret_manager_loader_force_rebuilds(monkeypatch):
    """The secret-manager loader keeps a populated-registry no-op for the API hot
    path, but force=True rebuilds in place — replacing a stale entry."""
    from gbserver.spacesecretmanager import spacesecretmanager as sm_module
    from gbserver.spacesecretmanager.spacesecretmanager import SpaceSecretManager
    from gbserver.utils import secretmanager_discovery

    class StaleSM(SpaceSecretManager):
        pass

    _patch_entry_points(monkeypatch, {})
    registry: dict = {"local": StaleSM}

    # Without force, an already-populated registry is left untouched (hot path).
    secretmanager_discovery.discover_secret_managers(
        package_file=sm_module.__file__,
        package_name="gbserver.spacesecretmanager",
        base_class=SpaceSecretManager,
        registry=registry,
    )
    assert registry["local"] is StaleSM  # no-op, stale entry kept

    # With force, the registry is rebuilt in place: the stale "local" is replaced
    # by the real in-tree class and the same dict object is reused.
    identity = id(registry)
    secretmanager_discovery.discover_secret_managers(
        package_file=sm_module.__file__,
        package_name="gbserver.spacesecretmanager",
        base_class=SpaceSecretManager,
        registry=registry,
        force=True,
    )
    assert id(registry) == identity
    assert "local" in registry
    assert registry["local"] is not StaleSM


# ---------------------------------------------------------------------------
# Auth providers
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_provider_registry_snapshot():
    from gbserver.api import auth_providers

    saved = dict(auth_providers.provider_types)
    # Start empty so the (load-once-guarded) loader rebuilds under the test's
    # patched entry points, then restore the real registry on teardown.
    auth_providers.provider_types.clear()
    yield auth_providers
    auth_providers.provider_types.clear()
    auth_providers.provider_types.update(saved)


def test_auth_provider_plugin_registered_by_name(
    monkeypatch, auth_provider_registry_snapshot
):
    """A plugin provider is filed under its entry-point name (lower + verbatim)."""
    from gbserver.api.auth_providers import AuthProvider

    class DummyProvider(AuthProvider):
        @property
        def provider_name(self):
            return "dummy"

        def identify_token(self, token):
            return False

        def validate_token(self, token):
            return (None, "nope")

    eps = _make_entry_points(monkeypatch, {"dummy": ("DummyProvider", DummyProvider)})
    _patch_entry_points(monkeypatch, {plugins.GROUP_AUTH_PROVIDERS: eps})

    auth_provider_registry_snapshot._load_auth_providers()
    assert auth_provider_registry_snapshot.provider_types["dummy"] is DummyProvider


def test_auth_provider_plugin_collision_core_wins(
    monkeypatch, caplog, auth_provider_registry_snapshot
):
    """A plugin cannot shadow a built-in provider (core-wins)."""
    from gbserver.api.auth_providers import AuthProvider

    class ShadowGitHub(AuthProvider):
        @property
        def provider_name(self):
            return "github"

        def identify_token(self, token):
            return False

        def validate_token(self, token):
            return (None, "nope")

    eps = _make_entry_points(monkeypatch, {"github": ("ShadowGitHub", ShadowGitHub)})
    _patch_entry_points(monkeypatch, {plugins.GROUP_AUTH_PROVIDERS: eps})

    auth_provider_registry_snapshot._load_auth_providers()
    # The in-tree GitHubAuthProvider was registered first, so the plugin is refused.
    assert auth_provider_registry_snapshot.provider_types["github"].__name__ == (
        "GitHubAuthProvider"
    )
    assert "already registered" in caplog.text


def test_auth_provider_build_list_multi_order(
    monkeypatch, auth_provider_registry_snapshot
):
    """build_provider_list preserves the JWT-before-opaque order for 'multi'."""
    monkeypatch.setenv("GBSERVER_IBMID_CLIENT_ID", "test-id")
    _patch_entry_points(monkeypatch, {})

    providers = auth_provider_registry_snapshot.build_provider_list("multi")
    assert [p.provider_name for p in providers] == ["ibmid", "github"]


# ---------------------------------------------------------------------------
# Resilience strategies
# ---------------------------------------------------------------------------


@pytest.fixture
def retry_strategy_registry_snapshot():
    from gbserver.resilience.retry_handler import RetryStrategy

    saved = dict(RetryStrategy.strategy_types)
    # Start empty so the (load-once-guarded) loader rebuilds under the test's
    # patched entry points, then restore on teardown.
    RetryStrategy.strategy_types.clear()
    yield RetryStrategy
    RetryStrategy.strategy_types.clear()
    RetryStrategy.strategy_types.update(saved)


def test_retry_strategy_plugin_registered_by_name(
    monkeypatch, retry_strategy_registry_snapshot
):
    """A plugin strategy is discovered and looked up by its config ``type`` name."""
    from gbserver.resilience.retry_handler import RetryStrategy

    class DummyStrategy(RetryStrategy):
        def should_retry(self, event):
            return False

    eps = _make_entry_points(monkeypatch, {"Dummy": ("DummyStrategy", DummyStrategy)})
    _patch_entry_points(monkeypatch, {plugins.GROUP_RESILIENCE_STRATEGIES: eps})

    RetryStrategy._load_retry_strategies()
    # Reachable under the verbatim config type and its lowercase alias.
    assert RetryStrategy.strategy_types["Dummy"] is DummyStrategy
    assert RetryStrategy.strategy_types["dummy"] is DummyStrategy


def test_retry_strategy_builtin_types_present(
    monkeypatch, retry_strategy_registry_snapshot
):
    """The in-tree config types are registered under their verbatim names."""
    from gbserver.resilience.retry_handler import RetryStrategy

    _patch_entry_points(monkeypatch, {})
    RetryStrategy._load_retry_strategies()
    assert "UnhealthyInsufficientPods" in RetryStrategy.strategy_types
    assert "AnyFailure" in RetryStrategy.strategy_types


def test_retry_strategy_plugin_collision_core_wins(
    monkeypatch, caplog, retry_strategy_registry_snapshot
):
    """A plugin cannot shadow a built-in strategy type (core-wins)."""
    from gbserver.resilience.retry_handler import RetryStrategy

    class ShadowStrategy(RetryStrategy):
        def should_retry(self, event):
            return False

    eps = _make_entry_points(
        monkeypatch, {"AnyFailure": ("ShadowStrategy", ShadowStrategy)}
    )
    _patch_entry_points(monkeypatch, {plugins.GROUP_RESILIENCE_STRATEGIES: eps})

    RetryStrategy._load_retry_strategies()
    assert RetryStrategy.strategy_types["AnyFailure"] is not ShadowStrategy
    assert "already registered" in caplog.text


# ---------------------------------------------------------------------------
# PluginRegistrar.discover_objects (object-valued registries, e.g. the CLI)
# ---------------------------------------------------------------------------


def test_registrar_discover_objects_files_by_name(monkeypatch):
    """discover_objects routes loaded objects (not classes) through add()."""
    marker = object()
    eps = _make_entry_points(monkeypatch, {"thing": ("thing", marker)})
    _patch_entry_points(monkeypatch, {"g": eps})

    reg: dict = {}
    registrar = plugins.PluginRegistrar(reg, "thing", plugins.keys_by_name)
    registrar.discover_objects("g")
    assert reg["thing"] is marker


def test_registrar_add_non_class_value_and_collision(caplog):
    """add() files a non-class value and logs a collision without needing __name__."""
    first = object()
    second = object()
    reg: dict = {}
    registrar = plugins.PluginRegistrar(reg, "thing", plugins.keys_by_name)
    registrar.add(first, "x")
    registrar.add(second, "x")  # collision: core (first) wins
    assert reg["x"] is first
    assert "already registered" in caplog.text


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_command_registry_snapshot():
    from gbcli.cli import GraniteBuildCLI

    saved = dict(GraniteBuildCLI.command_types)
    # Start empty so the (load-once-guarded) loader rebuilds under the test's
    # patched entry points, then restore on teardown.
    GraniteBuildCLI.command_types.clear()
    yield GraniteBuildCLI
    GraniteBuildCLI.command_types.clear()
    GraniteBuildCLI.command_types.update(saved)


def test_cli_plugin_command_registered_by_name(
    monkeypatch, cli_command_registry_snapshot
):
    """A plugin command is filed under its entry-point name and resolves to its
    click command object."""
    import click

    @click.command("dummy")
    def dummy_cmd():
        pass

    eps = _make_entry_points(monkeypatch, {"dummy": ("dummy_cmd", dummy_cmd)})
    _patch_entry_points(monkeypatch, {plugins.GROUP_CLI_PLUGINS: eps})

    cli_command_registry_snapshot._load_commands()
    assert "dummy" in cli_command_registry_snapshot.command_types
    # get_command resolves the registered value to the click command object.
    resolved = cli_command_registry_snapshot().get_command(None, "dummy")
    assert resolved is dummy_cmd


def test_cli_plugin_collision_core_wins(
    monkeypatch, caplog, cli_command_registry_snapshot
):
    """A plugin cannot shadow an in-tree command (core-wins).

    Asserts against the classmethod loader / registry directly rather than
    constructing ``GraniteBuildCLI`` (whose __init__ reconfigures logging, which
    would detach the handler caplog relies on)."""
    import click

    @click.command("build")
    def shadow_build():
        pass

    eps = _make_entry_points(monkeypatch, {"build": ("shadow_build", shadow_build)})
    _patch_entry_points(monkeypatch, {plugins.GROUP_CLI_PLUGINS: eps})

    cli_command_registry_snapshot._load_commands()
    # The in-tree build loader was registered first, so the plugin is refused.
    assert cli_command_registry_snapshot.command_types["build"] is not shadow_build
    assert "already registered" in caplog.text


def test_cli_plugin_case_variant_cannot_shadow_builtin(
    monkeypatch, cli_command_registry_snapshot
):
    """A case-variant plugin name cannot shadow an in-tree command either.

    The in-tree `build` is keyed under `build`; a plugin `Build` collides on that
    key and, with all-or-nothing core-wins, is filed under neither `build` nor
    `Build`. So `gb Build` must not resolve to the plugin — the exact bypass
    a per-key collision check would have allowed.
    """
    import click

    @click.command("Build")
    def shadow_build():
        pass

    eps = _make_entry_points(monkeypatch, {"Build": ("shadow_build", shadow_build)})
    _patch_entry_points(monkeypatch, {plugins.GROUP_CLI_PLUGINS: eps})

    # (The collision WARNING is asserted by the registrar-level test;
    # constructing GraniteBuildCLI reconfigures logging and detaches caplog.)
    cli = cli_command_registry_snapshot()
    assert "Build" not in cli.command_types
    assert cli.get_command(None, "Build") is not shadow_build


def test_cli_noop_when_no_plugins(monkeypatch, cli_command_registry_snapshot):
    """With no plugin installed, only in-tree commands are listed."""
    _patch_entry_points(monkeypatch, {})
    cli_command_registry_snapshot._load_commands()
    names = sorted(
        n for n in cli_command_registry_snapshot.command_types if n != "dataset"
    )
    assert "build" in names
    assert "version" in names
    assert "dataset" not in names  # hidden built-in stays hidden


def test_cli_mixed_case_plugin_listed_once(monkeypatch, cli_command_registry_snapshot):
    """A mixed-case plugin command name is filed under both cases but listed once.

    keys_by_name files `MyCmd` under both `mycmd` and `MyCmd`; list_commands must
    show a single canonical entry, while get_command still resolves either form.
    """
    import click

    @click.command("MyCmd")
    def mycmd():
        pass

    eps = _make_entry_points(monkeypatch, {"MyCmd": ("mycmd", mycmd)})
    _patch_entry_points(monkeypatch, {plugins.GROUP_CLI_PLUGINS: eps})

    cli = cli_command_registry_snapshot()
    listed = cli.list_commands(None)
    assert listed.count("mycmd") == 1
    assert "MyCmd" not in listed  # not shown twice under the verbatim key
    assert cli.get_command(None, "MyCmd") is mycmd
    assert cli.get_command(None, "mycmd") is mycmd


def test_cli_hidden_name_blocked_case_insensitively(
    monkeypatch, cli_command_registry_snapshot
):
    """A plugin claiming a hidden built-in's name is blocked in either case.

    `command_dataset.py` is skipped by the in-tree scan, so a plugin can file
    `Dataset` under both `dataset` and `Dataset`. get_command must refuse both,
    not just the lowercase form, so the hidden guard can't be bypassed by case.
    """
    import click

    @click.command("Dataset")
    def shadow_dataset():
        pass

    eps = _make_entry_points(
        monkeypatch, {"Dataset": ("shadow_dataset", shadow_dataset)}
    )
    _patch_entry_points(monkeypatch, {plugins.GROUP_CLI_PLUGINS: eps})

    cli = cli_command_registry_snapshot()
    assert cli.get_command(None, "Dataset") is None
    assert cli.get_command(None, "dataset") is None


def test_auth_build_list_skips_unregistered_name(
    monkeypatch, auth_provider_registry_snapshot
):
    """A mode naming an unregistered provider degrades gracefully, not KeyError."""
    _patch_entry_points(monkeypatch, {})
    ap = auth_provider_registry_snapshot
    # Force a mode that references a name absent from the registry.
    monkeypatch.setitem(ap._AUTH_MODES, "broken", ["nonexistent"])

    # Falls back to github rather than raising KeyError mid-request.
    providers = ap.build_provider_list("broken")
    assert [p.provider_name for p in providers] == ["github"]


def test_cli_non_command_plugin_rejected(
    monkeypatch, caplog, cli_command_registry_snapshot
):
    """A gbcli.plugins entry point that is not a click command is skipped with a
    warning, never filed (so _resolve_command can't later invoke it)."""

    def not_a_command():
        return "should-never-be-invoked"

    eps = _make_entry_points(monkeypatch, {"weird": ("not_a_command", not_a_command)})
    _patch_entry_points(monkeypatch, {plugins.GROUP_CLI_PLUGINS: eps})

    cli_command_registry_snapshot._load_commands(force=True)
    assert "weird" not in cli_command_registry_snapshot.command_types
    assert "not a valid" in caplog.text


def test_discover_objects_predicate_skips_and_warns(monkeypatch, caplog):
    """discover_objects filters out objects failing the predicate, with a warning."""
    good, bad = object(), object()
    eps = _make_entry_points(monkeypatch, {"good": ("good", good), "bad": ("bad", bad)})
    _patch_entry_points(monkeypatch, {"g": eps})

    reg: dict = {}
    registrar = plugins.PluginRegistrar(reg, "thing", plugins.keys_by_name)
    registrar.discover_objects("g", predicate=lambda o: o is good)
    assert reg == {"good": good}
    assert "not a valid" in caplog.text
