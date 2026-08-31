"""Startup validation for the auth settings.

Each rule is a crash, not a runtime surprise — misconfiguration is caught
before the app ever accepts a request.

Every construction here passes ``_env_file=None``. This module is the one place
that must build a ``Settings`` *without* the shared ``make_settings`` factory —
it asserts on defaults and on validation rules, so pinning the fields under test
would make it assert its own arguments back. Disabling dotenv reading is
therefore the only insulation available: without it, a developer with
``AUTOTUNEX_STANDALONE_EMAIL`` in their local ``.env`` sees
``test_defaults_are_standalone_with_no_narrowing`` fail for a reason that has
nothing to do with the code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr, ValidationError

from autotunex.core.config import Settings
from tests.conftest import make_settings


@pytest.fixture(autouse=True)
def _without_autotunex_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every ``AUTOTUNEX_`` variable for the duration of each test here.

    ``_env_file=None`` stops pydantic-settings reading the ``.env`` *file*, but
    an exported ``AUTOTUNEX_STANDALONE_EMAIL`` still outranks a field default —
    and this is the one module that cannot defend itself by pinning the field,
    because the default is the thing under assertion. Autouse, so a test added
    later inherits the insulation instead of rediscovering the problem.

    Caveat: a future test in this module that genuinely needs to assert
    environment-variable *parsing* — e.g. that ``AUTOTUNEX_AUTH_PROVIDERS=disabled``
    raises ``SettingsError`` because the value must be JSON — would be silently
    defanged by this fixture, since it strips the very variable being asserted on
    before the test body runs. This fixture is function-scoped, though, so such a
    test remains free to call ``monkeypatch.setenv(...)`` itself, later, inside its
    own body — that call runs after this fixture's stripping and is not undone by it.
    """
    for name in [key for key in os.environ if key.startswith("AUTOTUNEX_")]:
        monkeypatch.delenv(name)


def test_defaults_are_standalone_with_no_narrowing() -> None:
    settings = Settings(_env_file=None, environment="test")

    assert settings.auth_providers == ["disabled"]
    assert settings.standalone_email is None
    assert settings.standalone_role == "admin"


def test_gb_tags_defaults_to_autotunex() -> None:
    settings = Settings(_env_file=None, environment="test")

    assert settings.gb_tags == "autotunex"


def test_api_key_setting_no_longer_exists() -> None:
    assert not hasattr(Settings(_env_file=None, environment="test"), "api_key")


def test_disabled_cannot_combine_with_another_provider() -> None:
    with pytest.raises(ValidationError, match="disabled"):
        Settings(_env_file=None, environment="test", auth_providers=["disabled", "disabled"])


def test_an_empty_provider_list_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="test", auth_providers=[])


def test_prod_with_auth_disabled_refuses_to_start() -> None:
    with pytest.raises(ValidationError, match="disabled"):
        Settings(_env_file=None, environment="prod", auth_providers=["disabled"])


def test_prod_with_auth_disabled_and_the_opt_in_flag_starts() -> None:
    settings = Settings(
        _env_file=None,
        environment="prod",
        auth_providers=["disabled"],
        allow_insecure_no_auth=True,
    )

    assert settings.auth_providers == ["disabled"]


def test_prod_with_auth_disabled_still_refuses_without_the_opt_in_flag() -> None:
    with pytest.raises(ValueError, match="authentication disabled"):
        Settings(_env_file=None, environment="prod", auth_providers=["disabled"])


def test_prod_cannot_start_while_disabled_is_the_only_provider_that_exists() -> None:
    """Prod has no legal configuration in this phase, by construction.

    The positive counterpart — prod starting fine with a real provider — lands
    in the API-key phase, which is the first to add a second provider name. The
    name of this test says what it actually asserts; do not rename it to
    something that reads as a success case.
    """
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="prod")


def test_dev_with_auth_disabled_starts_fine() -> None:
    settings = Settings(_env_file=None, environment="dev", auth_providers=["disabled"])

    assert settings.environment == "dev"


def test_an_unknown_standalone_role_is_rejected() -> None:
    with pytest.raises(ValidationError, match="role"):
        Settings(_env_file=None, environment="test", standalone_role="superuser")


def test_api_key_provider_requires_at_least_one_key() -> None:
    with pytest.raises(ValidationError, match="api_keys"):
        Settings(_env_file=None, environment="test", auth_providers=["api_key"], api_keys={})


def test_api_key_provider_rejects_a_raw_key_where_a_digest_belongs() -> None:
    """Finding 1: a raw key pasted into ``api_keys`` must not survive into the crash.

    ``raw_key`` is a distinctive synthetic value (not a realistic-looking
    secret) chosen so the non-disclosure assertion below is meaningful: the
    old message interpolated the key verbatim via ``{key!r}``, so this
    assertion fails against that message and only passes once the key is
    withheld and the offending entry is identified by its email instead.
    """
    raw_key = "zzz-raw-key-must-never-appear-in-a-validation-error-zzz"
    with pytest.raises(ValidationError, match="digest") as exc_info:
        Settings(
            _env_file=None,
            environment="test",
            auth_providers=["api_key"],
            api_keys={raw_key: "a@example.com"},
        )

    assert raw_key not in str(exc_info.value)


def test_api_key_provider_rejects_an_uppercase_digest() -> None:
    """``^[0-9a-f]{64}$`` is lowercase-only, and ``hexdigest()`` produces lowercase.

    An uppercase digest is the shape of a value pasted from some other tool; it
    would pass a length check, start cleanly, and then never match anything,
    which is precisely the failure mode rule 3 exists to prevent.
    """
    with pytest.raises(ValidationError, match="digest"):
        Settings(
            _env_file=None,
            environment="test",
            auth_providers=["api_key"],
            api_keys={"A" * 64: "a@example.com"},
        )


def test_api_key_provider_accepts_a_valid_sha256_digest() -> None:
    digest = "a" * 64
    settings = Settings(
        _env_file=None,
        environment="test",
        auth_providers=["api_key"],
        api_keys={digest: "a@example.com"},
    )

    assert settings.api_keys == {digest: "a@example.com"}


def test_disabled_cannot_combine_with_api_key() -> None:
    digest = "a" * 64
    with pytest.raises(ValidationError, match="disabled"):
        Settings(
            _env_file=None,
            environment="test",
            auth_providers=["disabled", "api_key"],
            api_keys={digest: "a@example.com"},
        )


def test_prod_with_api_key_enabled_starts_fine() -> None:
    """The positive counterpart Phase 1's plan could not write.

    Phase 1 had only ``"disabled"`` in the Literal, so prod had no legal
    configuration at all and its test could only assert the failure. This is the
    escape.
    """
    digest = "a" * 64
    settings = Settings(
        _env_file=None,
        environment="prod",
        auth_providers=["api_key"],
        api_keys={digest: "a@example.com"},
    )

    assert settings.environment == "prod"


_OIDC_KWARGS: dict[str, Any] = {
    # `Any`, not `object`: this dict is spread as `Settings(**kwargs)`, and
    # `BaseSettings.__init__` has dozens of precisely-typed `_cli_*` /
    # `_env_*` keyword parameters. Against `dict[str, object]`, mypy strict
    # must reject the spread because `object` cannot satisfy any of them; only
    # `Any` lets the unpack through while keeping every value here concrete.
    "_env_file": None,
    "environment": "test",
    "auth_providers": ["oidc"],
    "oidc_issuer": "https://idp.example.com/oauth2",
    "oidc_jwks_uri": "https://idp.example.com/oauth2/jwks",
    "oidc_audience": "my-client-id",
}


def test_oidc_provider_requires_issuer_jwks_and_audience() -> None:
    with pytest.raises(ValidationError, match=r"oidc_issuer|oidc_jwks_uri|oidc_audience"):
        Settings(_env_file=None, environment="test", auth_providers=["oidc"])


@pytest.mark.parametrize("missing", ["oidc_issuer", "oidc_jwks_uri", "oidc_audience"])
def test_oidc_provider_names_whichever_setting_is_missing(missing: str) -> None:
    """Each of the three is independently required — audience most of all.

    granite.build made audience checking conditional on a client id being set,
    so leaving it unset silently accepted tokens minted for a *different*
    application on the same issuer. This is the startup check that closes that.
    """
    kwargs = dict(_OIDC_KWARGS)
    del kwargs[missing]

    with pytest.raises(ValidationError, match=missing):
        Settings(**kwargs)


@pytest.mark.parametrize("emptied", ["oidc_issuer", "oidc_jwks_uri", "oidc_audience"])
def test_oidc_provider_treats_an_empty_setting_as_unset(emptied: str) -> None:
    """``AUTOTUNEX_OIDC_AUDIENCE=`` in a ``.env`` is ``""``, not ``None``.

    This is the shape a misconfiguration actually takes — writing the variable
    and leaving the value off is far likelier than omitting the line — and an
    ``is None`` test let it walk straight through the validator built to catch
    it. Startup then succeeded and every token was rejected instead, since no
    real ``aud`` equals ``""``: a silent, uniform 401 with nothing in the
    configuration to explain it. Failing at startup is the whole point of this
    validator, so empty has to count as unset.
    """
    kwargs = dict(_OIDC_KWARGS)
    kwargs[emptied] = ""

    with pytest.raises(ValidationError, match=emptied):
        Settings(**kwargs)


@pytest.mark.parametrize(
    ("blanked", "whitespace"),
    [
        ("oidc_issuer", " "),
        ("oidc_jwks_uri", "\t"),
        ("oidc_audience", "   "),
    ],
)
def test_oidc_provider_treats_a_whitespace_only_setting_as_unset(
    blanked: str, whitespace: str
) -> None:
    """One layer below the empty-string case above, and just as reachable.

    ``AUTOTUNEX_OIDC_AUDIENCE="   "`` is truthy, so the bare ``if not value``
    check the previous fix installed lets it straight through: startup
    succeeds, and every token is rejected with the same uniform 401 the
    empty-string gate exists to prevent, with nothing in the configuration to
    explain why. No real ``iss``, JWKS URI, or ``aud`` is ever whitespace, so
    whitespace-only has to fail this gate the same way empty does.
    """
    kwargs = dict(_OIDC_KWARGS)
    kwargs[blanked] = whitespace

    with pytest.raises(ValidationError, match=blanked):
        Settings(**kwargs)


def test_oidc_provider_starts_fine_once_fully_configured() -> None:
    settings = Settings(**_OIDC_KWARGS)

    assert settings.oidc_algorithms == ["RS256"]
    assert settings.oidc_email_claims == ["email", "emailAddress"]
    assert settings.oidc_leeway_seconds == 30


def test_oidc_leeway_seconds_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="test", oidc_leeway_seconds=301)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="test", oidc_leeway_seconds=-1)


def test_prod_with_oidc_enabled_starts_fine() -> None:
    settings = Settings(**{**_OIDC_KWARGS, "environment": "prod"})

    assert settings.environment == "prod"


# 32+ characters, deliberately: `session_secret` now enforces this floor
# (RFC 7518 §3.2's own minimum for an HS256 key; see the field's docstring in
# `core/config.py`), and PyJWT would also emit `InsecureKeyLengthWarning` on
# every mint/verify with anything shorter. `oidc_client_secret` below is left
# short on purpose — that value belongs to the IdP, not to us.
_SESSION_SECRET = "test-config-session-secret-at-least-32-characters-long"

# `.invalid` (RFC 2606) hosts, deliberately, unlike `_OIDC_KWARGS` above: a
# mutation probe that drops one of these stubs would otherwise dial the real
# `idp.example.com` and hang instead of failing fast.
_SESSION_KWARGS: dict[str, Any] = {
    # `Any`, not `object` — see `_OIDC_KWARGS` above for why the spread below
    # needs it.
    "_env_file": None,
    "environment": "test",
    "auth_providers": ["session"],
    "oidc_issuer": "https://idp.invalid/oauth2",
    "oidc_jwks_uri": "https://idp.invalid/oauth2/jwks",
    "oidc_audience": "my-client-id",
    "oidc_client_id": "my-client-id",
    "oidc_client_secret": "shh",
    "oidc_authorization_endpoint": "https://idp.invalid/authorize",
    "oidc_token_endpoint": "https://idp.invalid/token",
    "public_base_url": "https://autotunex.invalid",
    "session_secret": _SESSION_SECRET,
}

_RULE_4_SETTINGS = [
    "oidc_issuer",
    "oidc_jwks_uri",
    "oidc_audience",
    "oidc_client_id",
    "oidc_client_secret",
    "oidc_authorization_endpoint",
    "oidc_token_endpoint",
    "public_base_url",
    "session_secret",
]


def test_session_provider_requires_the_full_bff_settings_block() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="test", auth_providers=["session"])


@pytest.mark.parametrize(
    "missing",
    [
        "oidc_client_id",
        "oidc_client_secret",
        "oidc_authorization_endpoint",
        "oidc_token_endpoint",
        "public_base_url",
        "session_secret",
    ],
)
def test_session_provider_names_whichever_bff_setting_is_missing(missing: str) -> None:
    """``session_secret`` above all — see the note on its field docstring.

    An unset HS256 key would let anyone mint a session cookie for any email
    address, which is why it is in this list rather than defaulting to a
    randomly-generated fallback.
    """
    kwargs = dict(_SESSION_KWARGS)
    del kwargs[missing]

    with pytest.raises(ValidationError, match=missing):
        Settings(**kwargs)


@pytest.mark.parametrize("emptied", _RULE_4_SETTINGS)
def test_session_provider_treats_an_empty_setting_as_unset(emptied: str) -> None:
    """Mirrors ``test_oidc_provider_treats_an_empty_setting_as_unset`` above.

    Rule 4 must use ``_is_unset``, not a bare ``is None`` or ``not value``
    check, for the same reason rule 2 does: ``AUTOTUNEX_SESSION_SECRET=`` with
    nothing after the ``=`` parses to ``""``, and an ``is None`` test would let
    that walk straight through the validator built to catch it.

    ``session_secret`` itself is the one exception in this parametrization:
    ``Field(min_length=32)`` rejects an empty ``SecretStr`` before rule 4's
    ``@model_validator`` ever runs, so that case is rejected by the length
    floor, not by ``_is_unset`` — the error still names ``session_secret``,
    so this test still passes for it, just not for the reason it passes for
    every other setting here. See
    ``test_a_32_character_whitespace_only_session_secret_is_still_treated_as_unset``
    below for a value that clears the length floor and actually reaches
    ``_is_unset``.
    """
    kwargs = dict(_SESSION_KWARGS)
    kwargs[emptied] = ""

    with pytest.raises(ValidationError, match=emptied):
        Settings(**kwargs)


@pytest.mark.parametrize("blanked", _RULE_4_SETTINGS)
def test_session_provider_treats_a_whitespace_only_setting_as_unset(blanked: str) -> None:
    """Mirrors ``test_oidc_provider_treats_a_whitespace_only_setting_as_unset`` above.

    One layer below the empty-string case: a whitespace-only value is truthy,
    so a bare ``if not value`` check misses it, and no real secret, endpoint,
    or URL is ever whitespace.

    ``session_secret`` itself is the one exception in this parametrization:
    three spaces is shorter than the 32-character floor
    ``Field(min_length=32)`` enforces, so that case is rejected by the length
    check before rule 4's ``_is_unset`` ever runs — the error still names
    ``session_secret``, so this test still passes for it, just not for the
    reason it passes for every other setting here. See
    ``test_a_32_character_whitespace_only_session_secret_is_still_treated_as_unset``
    below for a whitespace-only value that clears the length floor and
    actually reaches ``_is_unset``.
    """
    kwargs = dict(_SESSION_KWARGS)
    kwargs[blanked] = "   "

    with pytest.raises(ValidationError, match=blanked):
        Settings(**kwargs)


def test_a_32_character_whitespace_only_session_secret_is_still_treated_as_unset() -> None:
    """The ``session_secret`` case the two parametrized tests above cannot reach.

    32 spaces clears ``Field(min_length=32)`` — the length floor counts
    characters, not their content — so this is the one whitespace-only value
    that actually reaches rule 4's ``@model_validator``, and it must still be
    rejected by ``_is_unset`` rather than accepted for merely being long
    enough. Without this test, a change from ``_is_unset`` to a bare
    ``is None`` check would pass every other rule-4 assertion above (each one
    caught upstream by ``Field(min_length=32)`` for this field) and only fail
    here.
    """
    kwargs = dict(_SESSION_KWARGS)
    kwargs["session_secret"] = " " * 32

    with pytest.raises(ValidationError, match="session_secret"):
        Settings(**kwargs)


def test_session_provider_starts_fine_once_fully_configured() -> None:
    settings = Settings(**_SESSION_KWARGS)

    assert settings.session_ttl_hours == 8
    assert settings.session_cookie_same_site == "lax"


def test_session_provider_still_requires_the_rule_2_oidc_settings() -> None:
    kwargs = dict(_SESSION_KWARGS)
    del kwargs["oidc_audience"]

    with pytest.raises(ValidationError, match="oidc_audience"):
        Settings(**kwargs)


def test_cors_allow_origins_rejects_a_wildcard() -> None:
    """A later task attaches ``CORSMiddleware(..., allow_credentials=True)`` using this list.

    Starlette echoes back the request's own ``Origin`` with credentials
    allowed whenever ``"*"`` is in ``allow_origins`` and ``allow_credentials``
    is true — a wildcard origin combined with credentialed CORS is wrong
    regardless, so this is rejected unconditionally, not only when a session
    provider is configured.
    """
    with pytest.raises(ValidationError, match="cors_allow_origins"):
        Settings(_env_file=None, environment="test", cors_allow_origins=["*"])


def test_cross_site_cookies_require_an_explicit_cors_allowlist() -> None:
    kwargs = dict(_SESSION_KWARGS)
    kwargs["session_cookie_same_site"] = "none"

    with pytest.raises(ValidationError, match="cors_allow_origins"):
        Settings(**kwargs)


def test_cross_site_cookies_are_fine_with_an_explicit_cors_allowlist() -> None:
    kwargs = dict(_SESSION_KWARGS)
    kwargs["session_cookie_same_site"] = "none"
    kwargs["cors_allow_origins"] = ["https://ui.example.com"]

    settings = Settings(**kwargs)

    assert settings.cors_allow_origins == ["https://ui.example.com"]


def test_session_ttl_hours_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="test", session_ttl_hours=25)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="test", session_ttl_hours=0)


def test_a_session_secret_shorter_than_32_characters_is_rejected() -> None:
    """RFC 7518 §3.2's own floor for an HS256 key, not a stylistic choice.

    A short secret previously passed every check here — non-empty,
    non-whitespace — and then made every mint and every verify emit PyJWT's
    own ``InsecureKeyLengthWarning``, since the secret doubles as the HS256
    signing key. 31 characters, one below the floor, pins the boundary.
    """
    kwargs = dict(_SESSION_KWARGS)
    kwargs["session_secret"] = "x" * 31

    with pytest.raises(ValidationError, match="session_secret"):
        Settings(**kwargs)


def test_a_session_secret_of_32_characters_is_accepted() -> None:
    """The other side of the same boundary: 32 is enough, not just "close"."""
    kwargs = dict(_SESSION_KWARGS)
    kwargs["session_secret"] = "x" * 32

    settings = Settings(**kwargs)

    assert settings.session_secret is not None
    assert settings.session_secret.get_secret_value() == "x" * 32


def test_database_ssl_mode_defaults_to_auto() -> None:
    """``None`` derives from ``database_ssl_ca`` — verify if set, else no TLS."""
    settings = Settings(_env_file=None, environment="test")

    assert settings.database_ssl_mode is None


def test_dataset_defaults_are_sensible() -> None:
    settings = make_settings()

    assert settings.dataset_upload_max_bytes == 5 * 1024**3
    assert settings.dataset_storage_backend == "auto"
    assert settings.llmb_command == "llmb"
    assert settings.hf_token_env == "HF_TOKEN"


def test_dataset_staging_dir_is_under_the_storage_dir() -> None:
    settings = make_settings()

    assert settings.dataset_staging_dir == settings.dataset_storage_dir / ".staging"


def test_dataset_upload_concurrency_and_timeouts_have_defaults() -> None:
    settings = Settings(_env_file=None, environment="test")

    assert settings.dataset_upload_max_concurrent == 2
    assert settings.dataset_processing_timeout_seconds == 3600.0
    assert settings.dataset_push_timeout_seconds == 1800.0


def test_dataset_upload_max_concurrent_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="test", dataset_upload_max_concurrent=0)


def test_forcing_huggingface_without_a_token_env_var_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("GB_TOKEN", raising=False)

    with pytest.raises(ValueError, match="huggingface"):
        make_settings(dataset_storage_backend="huggingface")


def test_forcing_huggingface_without_the_gb_token_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    monkeypatch.delenv("GB_TOKEN", raising=False)

    # HF token alone is not enough — the GB token authenticates the llmb CLI.
    with pytest.raises(ValueError, match="GB_TOKEN"):
        make_settings(dataset_storage_backend="huggingface")


def test_forcing_huggingface_with_both_token_env_vars_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    monkeypatch.setenv("GB_TOKEN", "gb_xxx")

    settings = make_settings(dataset_storage_backend="huggingface")

    assert settings.dataset_storage_backend == "huggingface"


def test_forcing_huggingface_in_standalone_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both tokens present, so the ONLY reason to fail is the standalone conflict.
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    monkeypatch.setenv("GB_TOKEN", "gb_xxx")

    with pytest.raises(ValueError, match="standalone"):
        make_settings(dataset_storage_backend="huggingface", gb_environment="standalone")


def test_forcing_huggingface_in_lsf_standalone_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The remote LSF/SkyPilot standalone variant genuinely needs HF-hosted data
    # (a local file:// cannot reach the cluster); its push limitation is a
    # separate, deferred non-goal. So huggingface must remain valid there.
    monkeypatch.setenv("HF_TOKEN", "hf_xxx")
    monkeypatch.setenv("GB_TOKEN", "gb_xxx")

    settings = make_settings(
        dataset_storage_backend="huggingface",
        gb_environment="standalone",
        lsf_cluster="example-cluster",
    )

    assert settings.dataset_storage_backend == "huggingface"
    assert settings.lsf_cluster == "example-cluster"


def test_auto_in_standalone_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    # auto degrades to local in standalone; it must NOT be rejected at startup.
    settings = make_settings(dataset_storage_backend="auto", gb_environment="standalone")

    assert settings.dataset_storage_backend == "auto"
    assert settings.gb_environment == "standalone"


def test_llm_settings_default_to_unset() -> None:
    settings = Settings(_env_file=None, environment="test")

    assert settings.llm_base_url is None
    assert settings.llm_api_key is None
    assert settings.llm_model is None
    assert settings.llm_configured is False


def test_all_three_llm_settings_together_are_configured() -> None:
    settings = Settings(
        _env_file=None,
        environment="test",
        llm_base_url="https://gw.example/v1",
        llm_api_key="sk-test",
        llm_model="some-model",
    )

    assert settings.llm_configured is True


def test_partial_llm_configuration_is_rejected() -> None:
    with pytest.raises(ValidationError, match="partially configured"):
        Settings(
            _env_file=None,
            environment="test",
            llm_base_url="https://gw.example/v1",
        )


def test_llm_api_key_never_appears_in_repr() -> None:
    settings = Settings(
        _env_file=None,
        environment="test",
        llm_base_url="https://gw.example/v1",
        llm_api_key="super-secret-value",
        llm_model="some-model",
    )

    assert "super-secret-value" not in repr(settings)
    assert isinstance(settings.llm_api_key, SecretStr)


def test_job_backend_defaults_to_none() -> None:
    settings = Settings(_env_file=None, environment="test")

    assert settings.job_backend == "none"
    assert settings.job_runtime_image is None
    assert settings.job_trainer_ref == "main"


def test_job_backend_llmb_requires_runtime_image_repo_and_output() -> None:
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None, environment="test", job_backend="llmb")

    message = str(exc_info.value)
    assert "job_runtime_image" in message
    assert "job_trainer_repo" in message
    assert "job_output_uri_root" in message
    assert "gb_server_url" in message  # reconcile needs it or every job parks at pending


def test_job_backend_llmb_valid_when_required_fields_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GB_TOKEN", "gb_xxx")
    settings = Settings(
        _env_file=None,
        environment="test",
        job_backend="llmb",
        job_runtime_image="registry.example/tuner:1",
        job_trainer_repo="https://example.com/trainer.git",
        job_output_uri_root="s3://bucket/runs",
        gb_server_url="https://gbserver.example",
    )

    assert settings.job_backend == "llmb"
    assert settings.gb_server_url == "https://gbserver.example"


def test_job_backend_llmb_requires_the_gb_token_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GB_TOKEN", raising=False)

    with pytest.raises(ValidationError, match="GB_TOKEN"):
        Settings(
            _env_file=None,
            environment="test",
            job_backend="llmb",
            job_runtime_image="registry.example/tuner:1",
            job_trainer_repo="https://example.com/trainer.git",
            job_output_uri_root="s3://bucket/runs",
            gb_server_url="https://gbserver.example",
        )


def test_reconcile_settings_default_when_backend_is_none() -> None:
    settings = Settings(_env_file=None, environment="test")

    assert settings.gb_server_url is None
    assert settings.job_reconcile_interval_seconds == 30
    assert settings.job_reconcile_concurrency == 5


def test_hf_viewer_preview_defaults() -> None:
    settings = Settings(_env_file=None, environment="test")

    assert settings.hf_preview_enabled is True
    assert settings.hf_viewer_base_url == "https://datasets-server.huggingface.co"
    assert settings.hf_viewer_timeout_seconds == 2.5


def test_hf_viewer_preview_fields_are_overridable() -> None:
    settings = Settings(
        _env_file=None,
        environment="test",
        hf_preview_enabled=False,
        hf_viewer_base_url="https://viewer.internal",
        hf_viewer_timeout_seconds=5.0,
    )

    assert settings.hf_preview_enabled is False
    assert settings.hf_viewer_base_url == "https://viewer.internal"
    assert settings.hf_viewer_timeout_seconds == 5.0


def test_hf_viewer_timeout_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="test", hf_viewer_timeout_seconds=0)


# --- Local & bash job backends ---------------------------------------------


def test_job_backend_local_requires_no_cluster_settings() -> None:
    """``local`` runs the HPO in-process, so it needs none of the llmb inputs."""
    settings = Settings(_env_file=None, environment="test", job_backend="local")

    assert settings.job_backend == "local"


def test_gb_environment_reads_the_unprefixed_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """``GB_ENVIRONMENT`` is granite.build's own var — read *without* the prefix."""
    monkeypatch.setenv("GB_ENVIRONMENT", "standalone")

    settings = Settings(_env_file=None, environment="test")

    assert settings.gb_environment == "standalone"


def test_gb_environment_ignores_the_autotunex_prefixed_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The explicit alias exists precisely so the ``AUTOTUNEX_`` prefix does not apply."""
    monkeypatch.setenv("AUTOTUNEX_GB_ENVIRONMENT", "standalone")

    settings = Settings(_env_file=None, environment="test")

    assert settings.gb_environment is None


def test_gb_environment_is_normalized_case_insensitively(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fold ``GB_ENVIRONMENT=STANDALONE`` (granite.build's uppercase) to lowercase.

    AutoTuneX compares against lowercase ``"standalone"``, so the value is trimmed and folded.
    """
    monkeypatch.setenv("GB_ENVIRONMENT", "  STANDALONE  ")

    settings = Settings(_env_file=None, environment="test")

    assert settings.gb_environment == "standalone"


def test_bash_and_local_settings_have_sensible_defaults() -> None:
    settings = Settings(_env_file=None, environment="test")

    assert settings.bash_fm_tune_root is None
    assert settings.bash_fm_tune_extra == "full,mlx"
    assert settings.bash_backend == "mlx"
    assert settings.local_ray_address is None


def test_llmb_standalone_does_not_require_custom_code_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bash variant (``GB_ENVIRONMENT=standalone``) drops the custom_code inputs."""
    monkeypatch.setenv("GB_TOKEN", "t")

    settings = Settings(
        _env_file=None,
        environment="test",
        job_backend="llmb",
        gb_environment="standalone",
        gb_server_url="http://localhost:9000",
    )

    assert settings.gb_environment == "standalone"
    assert settings.job_runtime_image is None


def test_llmb_custom_code_still_requires_cluster_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without ``GB_ENVIRONMENT=standalone``, the current required set is unchanged."""
    monkeypatch.setenv("GB_TOKEN", "t")

    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            environment="test",
            job_backend="llmb",
            gb_server_url="http://x",
        )

    message = str(exc_info.value)
    assert "job_runtime_image" in message
    assert "job_trainer_repo" in message
    assert "job_output_uri_root" in message


def test_local_output_dir_defaults_under_artifact_dir() -> None:
    settings = Settings(_env_file=None, environment="test")

    assert settings.local_output_dir == settings.artifact_dir / "local"


def test_local_output_dir_tracks_an_overridden_artifact_dir() -> None:
    """The validator recomputes the default so the relationship survives an override."""
    settings = Settings(_env_file=None, environment="test", artifact_dir=Path("/data/art"))

    assert settings.local_output_dir == Path("/data/art/local")


def test_local_output_dir_is_respected_when_set_explicitly() -> None:
    """An explicit ``local_output_dir`` is never overwritten by the validator."""
    settings = Settings(
        _env_file=None,
        environment="test",
        artifact_dir=Path("/data/art"),
        local_output_dir=Path("/custom/out"),
    )

    assert settings.local_output_dir == Path("/custom/out")


def test_prod_refuses_debug_true() -> None:
    """Debug mode leaks tracebacks; never allow it in production."""
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="prod", debug=True, allow_insecure_no_auth=True)


def test_prod_allows_debug_false() -> None:
    settings = Settings(
        _env_file=None, environment="prod", debug=False, allow_insecure_no_auth=True
    )

    assert settings.debug is False


# --- LSF / SkyPilot standalone runner ---------------------------------------


def test_lsf_settings_have_sensible_defaults() -> None:
    settings = Settings(_env_file=None, environment="test")

    assert settings.lsf_cluster is None
    assert settings.lsf_environment_uri is None
    assert settings.lsf_image is None
    assert settings.lsf_accelerators is None
    assert settings.lsf_venv_path == "/step_venv"
    assert settings.lsf_cuda_home == "/opt/share/cuda-12.9"
    assert settings.lsf_num_cpus_per_node == 32
    assert settings.lsf_total_memory_per_node == "256Gi"
    assert settings.lsf_poll_interval_seconds == 30


def test_llmb_lsf_requires_environment_uri_image_and_trainer_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting lsf_cluster under standalone selects LSF, which needs its inputs."""
    monkeypatch.setenv("GB_TOKEN", "t")

    with pytest.raises(ValidationError) as exc_info:
        Settings(
            _env_file=None,
            environment="test",
            job_backend="llmb",
            gb_environment="standalone",
            gb_server_url="http://localhost:9000",
            lsf_cluster="example-cluster",
        )

    message = str(exc_info.value)
    assert "lsf_environment_uri" in message
    assert "lsf_image" in message
    assert "job_trainer_repo" in message


def test_llmb_lsf_valid_when_required_fields_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GB_TOKEN", "t")

    settings = Settings(
        _env_file=None,
        environment="test",
        job_backend="llmb",
        gb_environment="standalone",
        gb_server_url="http://localhost:9000",
        lsf_cluster="example-cluster",
        lsf_environment_uri="space://environments/skypilot/lsf/example-cluster",
        lsf_image="registry.example.com/tuner:1",
        job_trainer_repo="https://example.com/trainer.git",
    )

    assert settings.lsf_cluster == "example-cluster"


def test_llmb_standalone_bash_unaffected_by_lsf_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without lsf_cluster, standalone stays the bash variant (no LSF inputs required)."""
    monkeypatch.setenv("GB_TOKEN", "t")

    settings = Settings(
        _env_file=None,
        environment="test",
        job_backend="llmb",
        gb_environment="standalone",
        gb_server_url="http://localhost:9000",
    )

    assert settings.lsf_cluster is None
    assert settings.job_trainer_repo is None


# --- Cancellation -------------------------------------------------------


def test_local_cancel_timeout_seconds_defaults_to_30() -> None:
    """``_env_file=None``, not a bare ``Settings()``.

    This repo's real ``.env`` sets ``job_backend=llmb``, which requires
    ``GB_TOKEN`` to be exported in the shell (see the module docstring's
    rationale) — a bare ``Settings()`` would fail on that unrelated
    requirement rather than exercising this field's default. The autouse
    fixture above strips every ``AUTOTUNEX_`` env var, so no override of the
    new setting survives to this assertion.
    """
    settings = Settings(_env_file=None, environment="test")

    assert settings.local_cancel_timeout_seconds == 30.0
