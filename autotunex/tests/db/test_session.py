"""Engine construction.

The MySQL session-timezone argument is the only backend-specific connection
setting, and the only one worth testing here — everything else is dialect-neutral.

These tests deliberately go beyond comparing the returned dict to a literal. An
earlier version did only that, and passed while shipping ``{"time_zone": ...}`` —
a PyMySQL/aiomysql parameter that ``asyncmy`` rejects outright, so every real
request against MySQL failed with ``TypeError`` on connect. Asserting a dict
equals itself proves nothing about whether the driver accepts it.
"""

from __future__ import annotations

import datetime
import inspect
import ssl
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from autotunex.core.config import Settings
from autotunex.db.session import build_connect_args, build_pool_kwargs, build_ssl_context

_QUEUEPOOL_ONLY_KEYS = frozenset(
    {"pool_size", "max_overflow", "pool_timeout", "pool_recycle", "pool_use_lifo"}
)
"""Pool arguments that belong to ``QueuePool`` and blow up SQLite's ``StaticPool``.

Passing any of these to ``create_async_engine`` for a SQLite URL raises
``TypeError`` at engine-construction time — the exact regression
``build_pool_kwargs`` gates against, and the one the test suite (in-memory SQLite)
would hit first.
"""


def _pool_settings(**overrides: int | float | bool) -> Settings:
    """Return ``Settings`` with the connection-pool knobs pinned.

    Init kwargs outrank environment variables and ``.env`` in pydantic-settings, so
    these values are deterministic regardless of the developer's environment.
    """
    # dict[str, Any], not the narrower value type, so ``Settings(**values)`` type-checks
    # against BaseSettings' special init params (``_env_file`` and friends).
    values: dict[str, Any] = {
        "database_pool_size": 10,
        "database_max_overflow": 5,
        "database_pool_timeout_seconds": 30.0,
        "database_pool_recycle_seconds": 1800,
        "database_pool_pre_ping": True,
        "database_pool_use_lifo": True,
    }
    values.update(overrides)
    return Settings(**values)


def test_mysql_urls_pin_the_session_timezone_to_utc() -> None:
    """Without this MySQL converts TIMESTAMP reads into the server's zone."""
    args = build_connect_args("mysql+asyncmy://user:pw@host/autotune")

    assert args == {"init_command": "SET time_zone = '+00:00'"}


def test_the_mysql_argument_is_one_asyncmy_actually_accepts() -> None:
    """``asyncmy`` has no ``time_zone`` parameter, unlike PyMySQL and aiomysql.

    This is the assertion the original tests lacked. Passing an unsupported
    keyword raises ``TypeError`` at connect time, which no amount of dict
    comparison would reveal.
    """
    args = build_connect_args("mysql+asyncmy://user:pw@host/autotune")

    accepted = set(inspect.signature(_asyncmy_connection_init()).parameters)
    assert set(args) <= accepted, f"asyncmy rejects {set(args) - accepted}"


def test_the_mysql_init_command_sets_utc_not_some_other_zone() -> None:
    """A syntactically valid init_command pointing at the wrong zone still breaks reads."""
    command = build_connect_args("mysql://user:pw@host/autotune")["init_command"]

    assert "time_zone" in command
    assert "+00:00" in command


def test_sqlite_urls_get_no_extra_arguments() -> None:
    assert build_connect_args("sqlite+aiosqlite:///./autotunex.db") == {}


def test_postgres_urls_get_no_extra_arguments() -> None:
    assert build_connect_args("postgresql+asyncpg://user:pw@host/autotune") == {}


def test_a_bare_mysql_scheme_is_recognized() -> None:
    """The driver suffix is optional in a URL."""
    args = build_connect_args("mysql://user:pw@host/autotune")

    assert args == {"init_command": "SET time_zone = '+00:00'"}


def test_mysql_urls_carry_the_ssl_context_alongside_the_timezone() -> None:
    """Managed MySQL (IBM Cloud) refuses non-TLS auth, so the context must reach the driver."""
    context = ssl.create_default_context()

    args = build_connect_args("mysql+asyncmy://user:pw@host/autotune", context)

    assert args["ssl"] is context
    assert args["init_command"] == "SET time_zone = '+00:00'"


def test_ssl_is_a_keyword_asyncmy_actually_accepts() -> None:
    """As with init_command, an unsupported keyword would raise TypeError at connect."""
    args = build_connect_args("mysql://user:pw@host/autotune", ssl.create_default_context())

    accepted = set(inspect.signature(_asyncmy_connection_init()).parameters)
    assert set(args) <= accepted, f"asyncmy rejects {set(args) - accepted}"


def test_no_ssl_context_leaves_the_mysql_args_unchanged() -> None:
    """The default (no TLS) path must stay byte-for-byte what it was before."""
    assert build_connect_args("mysql+asyncmy://user:pw@host/autotune") == {
        "init_command": "SET time_zone = '+00:00'"
    }


def test_non_mysql_urls_ignore_a_supplied_ssl_context() -> None:
    """asyncpg/aiosqlite take TLS differently; ``ssl`` here is a MySQL-only concern."""
    args = build_connect_args(
        "postgresql+asyncpg://user:pw@host/autotune", ssl.create_default_context()
    )

    assert args == {}


def test_build_ssl_context_returns_none_without_a_ca_path() -> None:
    """No CA configured means no TLS parameters — the local SQLite/Postgres default."""
    assert build_ssl_context(None) is None


def test_build_ssl_context_verifies_the_server_against_the_given_ca(tmp_path: Path) -> None:
    """A CA path yields a context that actually checks the server's identity."""
    ca = tmp_path / "ca.pem"
    _write_self_signed_ca(ca)

    context = build_ssl_context(str(ca))

    assert context is not None
    assert context.verify_mode is ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_ssl_mode_require_encrypts_without_verifying_and_needs_no_ca() -> None:
    """``require`` gives an encrypted channel with no server authentication.

    This is the "connect without a certificate" path: IBM Cloud Databases
    refuses plaintext auth, but accepts TLS whose peer cert we do not verify.
    """
    context = build_ssl_context(None, "require")

    assert context is not None
    assert context.verify_mode is ssl.CERT_NONE
    assert context.check_hostname is False


def test_ssl_mode_disable_sends_no_tls_even_when_a_ca_is_configured(tmp_path: Path) -> None:
    """An explicit ``disable`` wins over a stray CA path — the local-dev default."""
    ca = tmp_path / "ca.pem"
    _write_self_signed_ca(ca)

    assert build_ssl_context(str(ca), "disable") is None


def test_ssl_mode_verify_without_a_ca_is_a_configuration_error() -> None:
    """Verification needs something to check against — fail loudly, not silently insecure."""
    with pytest.raises(ValueError, match="database_ssl_ca"):
        build_ssl_context(None, "verify")


def test_ssl_mode_verify_relaxes_x509_strict_for_ibm_clouds_ca(tmp_path: Path) -> None:
    """IBM Cloud's CA marks basicConstraints non-critical; OpenSSL 3.x strict mode rejects it.

    Clearing ``VERIFY_X509_STRICT`` is the whole reason ``verify`` works against
    IBM Cloud Databases at all — chain and hostname checks stay on.
    """
    ca = tmp_path / "ca.pem"
    _write_self_signed_ca(ca)

    context = build_ssl_context(str(ca), "verify")

    assert context is not None
    assert context.verify_mode is ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert not (context.verify_flags & ssl.VERIFY_X509_STRICT)


def test_ssl_mode_none_derives_verify_when_a_ca_is_present(tmp_path: Path) -> None:
    """Backward compatibility: a lone ``database_ssl_ca`` still means verified TLS."""
    ca = tmp_path / "ca.pem"
    _write_self_signed_ca(ca)

    context = build_ssl_context(str(ca), None)

    assert context is not None
    assert context.verify_mode is ssl.CERT_REQUIRED


def test_pool_kwargs_always_carry_pre_ping_for_every_dialect() -> None:
    """Pre-ping is pool-agnostic, so it is passed for SQLite, MySQL and Postgres alike."""
    settings = _pool_settings(database_pool_pre_ping=True)

    for url in (
        "sqlite+aiosqlite:///./autotunex.db",
        "mysql+asyncmy://user:pw@host/autotune",
        "postgresql+asyncpg://user:pw@host/autotune",
    ):
        assert build_pool_kwargs(url, settings)["pool_pre_ping"] is True


def test_pool_kwargs_for_sqlite_omit_the_queuepool_only_arguments() -> None:
    """SQLite's async pool is not a QueuePool; the sizing knobs would raise TypeError.

    This is the regression guard: the dev default and the whole test suite run on
    SQLite, so shipping a QueuePool-only argument here breaks every request and
    every test at engine construction.
    """
    kwargs = build_pool_kwargs("sqlite+aiosqlite:///./autotunex.db", _pool_settings())

    assert _QUEUEPOOL_ONLY_KEYS.isdisjoint(kwargs)
    assert kwargs == {"pool_pre_ping": True}


def test_pool_kwargs_for_mysql_include_the_queuepool_sizing_and_recycle() -> None:
    """Managed MySQL is where warm reuse and recycling actually pay off."""
    kwargs = build_pool_kwargs("mysql+asyncmy://user:pw@host/autotune", _pool_settings())

    assert set(kwargs) >= _QUEUEPOOL_ONLY_KEYS


def test_pool_kwargs_for_postgres_include_the_queuepool_sizing() -> None:
    """Postgres is a server database too, so it gets the same QueuePool tuning as MySQL."""
    kwargs = build_pool_kwargs("postgresql+asyncpg://user:pw@host/autotune", _pool_settings())

    assert set(kwargs) >= _QUEUEPOOL_ONLY_KEYS


def test_pool_kwargs_pass_the_configured_values_straight_through() -> None:
    """Every knob is operator-tunable; the builder must not silently substitute its own."""
    settings = _pool_settings(
        database_pool_size=17,
        database_max_overflow=3,
        database_pool_timeout_seconds=12.5,
        database_pool_recycle_seconds=900,
        database_pool_pre_ping=False,
        database_pool_use_lifo=False,
    )

    kwargs = build_pool_kwargs("mysql+asyncmy://user:pw@host/autotune", settings)

    assert kwargs == {
        "pool_pre_ping": False,
        "pool_size": 17,
        "max_overflow": 3,
        "pool_timeout": 12.5,
        "pool_recycle": 900,
        "pool_use_lifo": False,
    }


async def test_the_sqlite_pool_kwargs_are_ones_create_async_engine_accepts() -> None:
    """As with the connect args, prove the driver accepts them rather than asserting a dict.

    ``create_async_engine`` builds the pool eagerly, so an argument the SQLite pool
    rejects raises here at construction time — exactly the failure mode gating exists
    to prevent.
    """
    from sqlalchemy.ext.asyncio import create_async_engine

    url = "sqlite+aiosqlite:///:memory:"
    engine = create_async_engine(url, **build_pool_kwargs(url, _pool_settings()))
    try:
        assert engine is not None
    finally:
        await engine.dispose()


def _write_self_signed_ca(path: Path) -> None:
    """Write a throwaway self-signed CA PEM so ``load_verify_locations`` has real input."""
    cryptography = pytest.importorskip(
        "cryptography", reason="the mysql extra (cryptography) is not installed"
    )
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    del cryptography  # only imported to gate the skip
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-ca")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC))
        .not_valid_after(datetime.datetime(2040, 1, 1, tzinfo=datetime.UTC))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))


def _asyncmy_connection_init() -> Callable[..., Any]:
    """Return ``asyncmy.connection.Connection.__init__`` for signature inspection.

    Skips rather than fails when the optional ``mysql`` extra is not installed, so
    the suite still runs on a SQLite-only checkout.
    """
    import pytest

    connection = pytest.importorskip(
        "asyncmy.connection", reason="the mysql extra is not installed"
    )
    init: Callable[..., Any] = connection.Connection.__init__
    return init
