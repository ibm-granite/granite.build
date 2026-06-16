# Environment variable constants used exclusively in test contexts.
# These env vars are generally passed to distributed components (i.e. steps, buildrunners, etc)
# that are then responsible for implementing/following their implications.
#
# Kept separate from constants.py to avoid mixing production and test config.

import os
from typing import Optional

_GBTEST_PREFIX = "GBTEST_"

# Controls which HF operations are mocked. Set by tests that lack real (or
# write) HuggingFace access. The value is a comma-separated, case-insensitive
# list of op names (see HF_OP_* below); the token "all" mocks everything and an
# empty/unset value mocks nothing. Propagated to remote jobs/pods via env var so
# they mock the same ops. Read at call time (not import time) so tests can toggle
# it by setting/unsetting the env var without any patching.
#
# Selective mocking lets forked-PR CI mock the token-requiring ops (push, exists,
# delete, resource_group) while letting pull run for real against public repos
# (which download anonymously, no HF_TOKEN needed).
ENV_VAR_GBTEST_MOCKED_HF_OPS = f"{_GBTEST_PREFIX}MOCKED_HF_OPS"

# Canonical HF operation names that can be selectively mocked.
HF_OP_PULL = "pull"
HF_OP_PUSH = "push"
HF_OP_EXISTS = "exists"
HF_OP_DELETE = "delete"
HF_OP_RESOURCE_GROUP = "resource_group"
HF_OPS_ALL = frozenset(
    {HF_OP_PULL, HF_OP_PUSH, HF_OP_EXISTS, HF_OP_DELETE, HF_OP_RESOURCE_GROUP}
)

# Sentinel value in GBTEST_MOCKED_HF_OPS meaning "mock every HF op".
_HF_OPS_ALL_TOKEN = "all"

# Stack of prior GBTEST_MOCKED_HF_OPS values saved by enable_hf_mocks() so that
# disable_hf_mocks() restores the previous value instead of clobbering a
# suite-level default (e.g. one exported by the Makefile) for sibling tests.
_HF_OPS_SAVED: list[Optional[str]] = []


def hf_mocked_ops() -> set[str]:
    """Parse GBTEST_MOCKED_HF_OPS into the set of HF ops to mock.

    Returns:
        set[str]: lowercased op names to mock. The token "all" expands to
        HF_OPS_ALL; an empty/unset env var yields an empty set.
    """
    raw = os.getenv(ENV_VAR_GBTEST_MOCKED_HF_OPS, "")
    ops = {tok.strip().lower() for tok in raw.split(",") if tok.strip()}
    if _HF_OPS_ALL_TOKEN in ops:
        return set(HF_OPS_ALL)
    return ops


def is_hf_mocked(op: str) -> bool:
    """Return True if the given HF op should be mocked.

    Args:
        op: One of the HF_OP_* operation names (e.g. HF_OP_PULL).

    Returns:
        bool: True if ``op`` is listed in GBTEST_MOCKED_HF_OPS (or it lists "all").
    """
    return op.lower() in hf_mocked_ops()


def enable_hf_mocks(*ops: str) -> None:
    """Enable mocking of the given HF ops for this process and any remote jobs/pods.

    Sets GBTEST_MOCKED_HF_OPS in the environment; the previous value is saved and
    restored by the matching disable_hf_mocks(). Since is_hf_mocked() reads the
    env var at call time, this takes effect immediately without patching, and the
    env var is forwarded to remote pods so they mock the same ops.

    Args:
        *ops: HF_OP_* names to mock. Defaults to all ops when none are given.
    """
    _HF_OPS_SAVED.append(os.environ.get(ENV_VAR_GBTEST_MOCKED_HF_OPS))
    os.environ[ENV_VAR_GBTEST_MOCKED_HF_OPS] = (
        ",".join(ops) if ops else _HF_OPS_ALL_TOKEN
    )


def disable_hf_mocks() -> None:
    """Restore GBTEST_MOCKED_HF_OPS to the value saved by the matching enable_hf_mocks().

    Restores the prior value (or removes the var if there was none), so per-test
    enable/disable does not clobber a suite-level default set outside the test.
    """
    prior = _HF_OPS_SAVED.pop() if _HF_OPS_SAVED else None
    if prior is None:
        os.environ.pop(ENV_VAR_GBTEST_MOCKED_HF_OPS, None)
    else:
        os.environ[ENV_VAR_GBTEST_MOCKED_HF_OPS] = prior


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


# The set of all GBTEST_ env var names defined in this module.
_GBTEST_EXPORTED_ENV_VARS = {
    ENV_VAR_GBTEST_MOCKED_HF_OPS,
    ENV_VAR_GBTEST_SIMULATE_FAILURE_SCENARIO,
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
