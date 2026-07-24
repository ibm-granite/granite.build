# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest config for api/ tests.

asyncio_mode=auto lets async test functions run without a per-test marker.
No project-wide pytest config exists; these tests are run ad hoc with
`python -m pytest api/tests -v` from the `api/` directory.
"""

import sys
import types

import pytest
from services.plugins import clear_overrides


def _install_pyarrow_stub_if_missing():
    """Make ``import pyarrow`` resolvable when the wheel is not installed.

    ``services.file_service`` imports ``pyarrow``/``pyarrow.parquet`` at module
    top level, but the characterization tests here only exercise the pure-Python
    JSONL paths (``stream_split_jsonl``, ``remap_jsonl_file``, ``get_jsonl_data``)
    — none of which touch pyarrow. In sandboxed CI where pyarrow cannot be
    installed, we register a minimal stub so the module imports. If real pyarrow
    is present this is a no-op, so it never shadows the genuine library.
    """
    try:
        import pyarrow  # noqa: F401

        return
    except ImportError:
        pass

    pa = types.ModuleType("pyarrow")
    # pandas' compat layer probes pyarrow.__version__ at import time.
    pa.__version__ = "0.0.0"
    pq = types.ModuleType("pyarrow.parquet")
    pa.parquet = pq
    sys.modules.setdefault("pyarrow", pa)
    sys.modules.setdefault("pyarrow.parquet", pq)


def _install_autotune_stub_if_missing():
    """Make ``from autotune.utils import get_autotune_dataset_types`` resolvable
    when the ``autotune`` library is not installed.

    ``services.datasets.intelligence`` imports ``get_autotune_dataset_types``
    from ``autotune.utils`` at module top level, but the tests here only
    exercise the pure-Python methods (``_has_io_structure``,
    ``_create_direct_mapping_strategy``, ``_validate_strategy_on_sample``) which
    never call autotune. In sandboxed CI where autotune cannot be installed, we
    register a minimal stub so the module imports. If real autotune is present
    this is a no-op, so it never shadows the genuine library.
    """
    try:
        import autotune.utils  # noqa: F401

        return
    except ImportError:
        pass

    autotune = types.ModuleType("autotune")
    utils = types.ModuleType("autotune.utils")

    def get_autotune_dataset_types():
        return {}

    def parse_model_parameters(model_name):
        return None

    def estimate_memory_usage(**kwargs):
        return {}

    def get_autotune_config(**kwargs):
        return {}

    utils.get_autotune_dataset_types = get_autotune_dataset_types
    utils.parse_model_parameters = parse_model_parameters
    utils.estimate_memory_usage = estimate_memory_usage
    utils.get_autotune_config = get_autotune_config
    autotune.utils = utils
    sys.modules.setdefault("autotune", autotune)
    sys.modules.setdefault("autotune.utils", utils)


def _install_tuspyserver_stub_if_missing():
    """Stub ``tuspyserver`` when the wheel is not installed (PyPI blocked in CI).

    ``services.datasets.tus_app`` imports ``create_tus_router`` at module top.
    The stub lets import-health tests load the module; live tus behavior is
    verified separately in a deps-equipped environment. No-op if real tuspyserver
    is present.
    """
    try:
        import tuspyserver  # noqa: F401

        return
    except ImportError:
        pass

    mod = types.ModuleType("tuspyserver")

    def create_tus_router(**kwargs):
        from fastapi import APIRouter

        return APIRouter()  # empty router; satisfies mount, exercises no tus logic

    mod.create_tus_router = create_tus_router
    sys.modules.setdefault("tuspyserver", mod)


def _install_db_stubs_if_missing():
    """Stub pymysql/aiomysql/dbutils when absent so services.db_service imports
    in CI without a MySQL driver. No-op when the real drivers are installed.
    Tests that touch the DB use a mocked Database object, never these stubs.

    services.db_service imports at module top level: ``import pymysql``,
    ``import aiomysql``, and ``from dbutils.pooled_db import PooledDB``. The
    dbutils.pooled_db submodule must therefore expose a ``PooledDB`` attribute.
    """

    class _LenientModule(types.ModuleType):
        # db_service evaluates attribute accesses at import time (e.g. the
        # ``aiomysql.Pool`` type hint on line 49, ``pymysql.cursors.DictCursor``
        # at call time). Return a permissive placeholder for any attribute so
        # the module imports without the real driver. Submodules (e.g.
        # ``pymysql.cursors``) resolve to another lenient module.
        def __getattr__(self, item):
            sub = _LenientModule(f"{self.__name__}.{item}")
            setattr(self, item, sub)
            return sub

    for name in ("pymysql", "aiomysql"):
        try:
            __import__(name)
        except ImportError:
            sys.modules.setdefault(name, _LenientModule(name))
    # dbutils + dbutils.pooled_db submodule
    # (db_service does `from dbutils.pooled_db import PooledDB`).
    try:
        __import__("dbutils.pooled_db")
    except ImportError:
        dbutils_mod = _LenientModule("dbutils")
        dbutils_mod.__path__ = []  # mark as a package so submodule import resolves
        pooled = _LenientModule("dbutils.pooled_db")
        pooled.PooledDB = object
        dbutils_mod.pooled_db = pooled
        sys.modules.setdefault("dbutils", dbutils_mod)
        sys.modules.setdefault("dbutils.pooled_db", pooled)


def _install_gbcli_stub_if_missing():
    """Stub ``gbcli`` when the wheel is not installed (IBM-internal package).

    ``services.gb_service`` imports multiple ``gbcli`` sub-modules at module top
    level and calls ``configureGBWorkingEnv()`` immediately. The stub lets
    ``services.job_service`` (which imports ``gb_service``) load in OSS / CI
    environments where ``gbcli`` is unavailable. All attribute accesses on the
    stub return a callable no-op sentinel so that the import chain completes
    without raising. No-op when real gbcli is present.

    IMPORTANT: Only the *sub*-modules (not the top-level ``gbcli`` package) are
    registered in sys.modules. Python's import machinery resolves
    ``from gbcli.x.y import z`` by looking up ``gbcli.x.y`` directly in
    sys.modules without needing the top-level ``gbcli`` entry. Meanwhile
    ``importlib.util.find_spec("gbcli")`` returns ``None`` (package not found),
    which preserves the existing ``@pytest.mark.skipif`` guard in
    ``test_import_health.py::test_server_entrypoints_import``.
    """
    try:
        import gbcli  # noqa: F401

        return
    except ImportError:
        pass

    import importlib.util as _ilu

    class _Stub:
        """Callable no-op sentinel returned for any attribute on a stub module."""

        def __init__(self, name=""):
            self._name = name

        def __call__(self, *a, **kw):
            return None

        def __repr__(self):
            return f"_GbStub({self._name!r})"

    class _LenientGbModule(types.ModuleType):
        """Module stub: returns a _Stub for any attribute that isn't set."""

        def __getattr__(self, item):
            stub = _Stub(f"{self.__name__}.{item}")
            # Cache on the module dict to avoid repeated __getattr__ calls.
            object.__setattr__(self, item, stub)
            return stub

    for subpath in (
        "gbcli.utils",
        "gbcli.utils.cli_config",
        "gbcli.utils.gbconstants",
        "gbcli.utils.gbcredentials",
        "gbcli.utils.gbserver",
        "gbcli.utils.log_query",
        "gbcli.utils.utils",
        "gbcli.services",
        "gbcli.services.service_build",
    ):
        if subpath not in sys.modules:
            mod = _LenientGbModule(subpath)
            mod.__path__ = []
            mod.__spec__ = _ilu.spec_from_loader(subpath, loader=None)
            sys.modules[subpath] = mod
        # Wire parent → child attribute, but SKIP wiring the root "gbcli" parent
        # so it is never added to sys.modules (preserves find_spec("gbcli")==None).
        parts = subpath.split(".")
        if len(parts) > 1:
            parent_name = ".".join(parts[:-1])
            if parent_name in sys.modules:
                setattr(sys.modules[parent_name], parts[-1], sys.modules[subpath])


def _install_lakehouse_stub_if_missing():
    """Stub ``lakehouse`` when the IBM Data Mesh Framework wheel is absent.

    ``services.dmf_service`` imports ``lakehouse.assets.Model`` and
    ``lakehouse.wrappers.LakehouseIceberg`` at module top level. The stub lets
    ``services.job_service`` (which imports ``dmf_service``) load in OSS / CI
    environments without the DMF wheel. No-op when real lakehouse is present.
    """
    try:
        import lakehouse  # noqa: F401

        return
    except ImportError:
        pass

    class _LenientLhModule(types.ModuleType):
        def __call__(self, *a, **kw):
            return None

        def __getattr__(self, item):
            sub = _LenientLhModule(f"{self.__name__}.{item}")
            setattr(self, item, sub)
            return sub

    for dotted in ("lakehouse", "lakehouse.assets", "lakehouse.wrappers"):
        if dotted not in sys.modules:
            mod = _LenientLhModule(dotted)
            mod.__path__ = []
            sys.modules[dotted] = mod
        parts = dotted.split(".")
        if len(parts) > 1:
            parent = sys.modules[".".join(parts[:-1])]
            setattr(parent, parts[-1], sys.modules[dotted])


_install_pyarrow_stub_if_missing()
_install_autotune_stub_if_missing()
_install_db_stubs_if_missing()
_install_tuspyserver_stub_if_missing()
_install_gbcli_stub_if_missing()
_install_lakehouse_stub_if_missing()


def _has_pytest_asyncio() -> bool:
    try:
        import pytest_asyncio  # noqa: F401

        return True
    except ImportError:
        return False


_HAS_PYTEST_ASYNCIO = _has_pytest_asyncio()


def pytest_configure(config):
    # When pytest-asyncio is installed, asyncio_mode=auto lets async test
    # functions run without a per-test marker. The option only exists when the
    # plugin is loaded — setting it otherwise raises, so guard on presence.
    if _HAS_PYTEST_ASYNCIO:
        config.option.asyncio_mode = "auto"


if not _HAS_PYTEST_ASYNCIO:
    import asyncio
    import inspect

    def pytest_pyfunc_call(pyfuncitem):
        """Fallback async test runner when pytest-asyncio is not installed.

        Some environments that carry the heavy runtime deps (pyarrow, tuspyserver,
        DB drivers) do not have pytest-asyncio. Rather than tie the suite to one
        interpreter, run coroutine test functions on a fresh event loop here. This
        hook is registered ONLY when the plugin is absent, so it never competes with
        pytest-asyncio when that is installed. Returning True tells pytest the call
        was handled.
        """
        test_fn = pyfuncitem.obj
        if not inspect.iscoroutinefunction(test_fn):
            return None
        sig = inspect.signature(test_fn)
        kwargs = {
            name: pyfuncitem.funcargs[name]
            for name in sig.parameters
            if name in pyfuncitem.funcargs
        }
        asyncio.run(test_fn(**kwargs))
        return True


@pytest.fixture(autouse=True)
def _reset_overrides():
    """Ensure registry overrides never leak between tests."""
    clear_overrides()
    yield
    clear_overrides()
