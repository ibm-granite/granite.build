# Environment variable constants used exclusively in test contexts.
# These env vars are generally passed to distributed components (i.e. steps, buildrunners, etc)
# that are then responsible for implementing/following their implications.
#
# Kept separate from constants.py to avoid mixing production and test config.

import os
from typing import Optional

from gbcommon.types.gbenvconfig import getenv_boolean

_GBTEST_PREFIX = "GBTEST_"

# Controls whether HuggingFace Hub I/O is mocked. Set by tests that lack real
# (or write) HuggingFace access — e.g. forked-PR CI, which has no HF_TOKEN. It is
# all-or-nothing: when truthy, every HF op (push/pull/exists/delete and
# resource-group resolution) short-circuits before touching the Hub; when
# unset/empty/falsy, all ops run for real. Propagated to remote jobs/pods via env var
# so they mock identically. Read at call time (not import time) so tests can
# toggle it by setting/unsetting the env var without any patching.
ENV_VAR_GBTEST_MOCK_HF = f"{_GBTEST_PREFIX}MOCK_HF"

# Stack of prior GBTEST_MOCK_HF values saved by enable_hf_mocks() so that
# disable_hf_mocks() restores the previous value instead of clobbering a
# suite-level default (e.g. one forced by mock mode) for sibling tests.
_HF_MOCK_SAVED: list[Optional[str]] = []


def is_hf_mocked() -> bool:
    """Return True if HuggingFace Hub I/O should be mocked (all ops).

    A plain boolean read of GBTEST_MOCK_HF using the repo-standard parsing, so
    "true"/"1"/"yes"/"on" all enable mocking and unset/empty are both False.
    Defaulting it on under GBTEST_MODE=mock is a separate, deliberate test-init
    step that writes the resolved value back to the environment (see
    pytest_sessionstart in test/conftest.py) — this function stays a dumb read so
    it behaves identically in a dispatched worker or pod, which only ever receives
    the forwarded env var.

    Returns:
        bool: True when GBTEST_MOCK_HF parses as truthy.
    """
    return getenv_boolean(ENV_VAR_GBTEST_MOCK_HF)


def enable_hf_mocks() -> None:
    """Enable HF mocking for this process and any remote jobs/pods.

    Sets GBTEST_MOCK_HF=true in the environment; the previous value is saved and
    restored by the matching disable_hf_mocks(). Since is_hf_mocked() reads the
    env var at call time, this takes effect immediately without patching, and the
    env var is forwarded to remote pods so they mock too.
    """
    _HF_MOCK_SAVED.append(os.environ.get(ENV_VAR_GBTEST_MOCK_HF))
    os.environ[ENV_VAR_GBTEST_MOCK_HF] = "true"


def disable_hf_mocks() -> None:
    """Restore GBTEST_MOCK_HF to the value saved by the matching enable_hf_mocks().

    Restores the prior value (or removes the var if there was none), so per-test
    enable/disable does not clobber a suite-level default set outside the test.
    """
    prior = _HF_MOCK_SAVED.pop() if _HF_MOCK_SAVED else None
    if prior is None:
        os.environ.pop(ENV_VAR_GBTEST_MOCK_HF, None)
    else:
        os.environ[ENV_VAR_GBTEST_MOCK_HF] = prior


# Causes the supporting environments that implement step-level retry to inject
# an initial failure event to trigger the step retry in the environment, if the step supports retries.
# Any environment that supports retries using Environment.with_retry_handler() will
# be subject to this injection via with_retry_handler().
ENV_VAR_GBTEST_SIMULATE_FAILURE_SCENARIO = f"{_GBTEST_PREFIX}SIMULATE_FAILURE_SCENARIO"


def is_failure_simulated() -> bool:
    """Return True if failure simulation is enabled (GBTEST_SIMULATE_FAILURE_SCENARIO=true in env)."""
    return os.getenv(ENV_VAR_GBTEST_SIMULATE_FAILURE_SCENARIO, "").lower() == "true"


def enable_failure_simulation() -> None:
    """Enable failure simulation for this process and any remote jobs/pods.

    Sets GBTEST_SIMULATE_FAILURE_SCENARIO in the environment. The env var is also
    forwarded to remote pods via get_exported_gbtest_env_vars().
    """
    os.environ[ENV_VAR_GBTEST_SIMULATE_FAILURE_SCENARIO] = "true"


def disable_failure_simulation() -> None:
    """Disable failure simulation by removing GBTEST_SIMULATE_FAILURE_SCENARIO from the environment."""
    os.environ.pop(ENV_VAR_GBTEST_SIMULATE_FAILURE_SCENARIO, None)


# Which environment's HF resource group a STANDALONE run pushes to (e.g.
# gbspace-public-staging). Defaults to EMPTY, meaning the production
# gbspace-public: a real standalone user must land in the production group. Only
# a test run opts into a redirect, by setting this to STAGING/DEV explicitly (the
# extended-tests Makefile targets do). Do not give this a non-empty default —
# that silently sends real users to a staging group they cannot write.
ENV_VAR_GBTEST_STANDALONE_ENVIRONMENT = f"{_GBTEST_PREFIX}STANDALONE_ENVIRONMENT"
DEFAULT_GBTEST_STANDALONE_ENVIRONMENT = ""


def standalone_rg_environment() -> str:
    """Return the environment whose HF resource group a STANDALONE run targets.

    Read at call time (not import time) so a test can set/unset the env var
    without patching.

    Returns:
        str: ``"STAGING"``/``"DEV"`` when a test run has explicitly opted into a
        redirect, or ``""`` (the default, and when the var is set empty) meaning
        the production resource group.
    """
    return os.getenv(
        ENV_VAR_GBTEST_STANDALONE_ENVIRONMENT, DEFAULT_GBTEST_STANDALONE_ENVIRONMENT
    ).strip()


# The set of all GBTEST_ env var names defined in this module.
_GBTEST_EXPORTED_ENV_VARS = {
    ENV_VAR_GBTEST_MOCK_HF,
    ENV_VAR_GBTEST_SIMULATE_FAILURE_SCENARIO,
    ENV_VAR_GBTEST_STANDALONE_ENVIRONMENT,
}


def get_exported_gbtest_env_vars() -> dict[str, str]:
    """Return the GBTEST_ environment variables defined in this module that are currently set.

    Only returns vars explicitly declared here (not arbitrary GBTEST_* vars from the
    environment), so callers never accidentally forward test secrets or API keys.

    Returns:
        dict[str, str]: mapping of env var name → value for each known GBTEST_
        variable that is currently set in the environment.
    """
    return {k: v for k, v in os.environ.items() if k in _GBTEST_EXPORTED_ENV_VARS}
