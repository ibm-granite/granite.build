# Environment variable constants used exclusively in test contexts.
# These env vars are generally passed to distributed components (i.e. steps, buildrunners, etc)
# that are then responsible for implementing/following their implications.
#
# Kept separate from constants.py to avoid mixing production and test config.

import os

# True when GBTEST_MOCK_HF_CALLS is set in the environment (e.g. inside a hfpush/pull step 
# launched by a test-driven build runner). This should be propogated and applied to all 
# hfpull/push step implementations.
ENV_VAR_GBTEST_MOCK_HF_CALLS = "GBTEST_MOCK_HF_CALLS"
GBTEST_MOCK_HF = os.getenv(ENV_VAR_GBTEST_MOCK_HF_CALLS, "").lower() == "true"

# Causes the supporting environments that implement step-level retry to inject
# an initial failure event to trigger the step retry in the environment, if the step supports retries.
# Any environment that supports retries using Environment.with_retry_handler() will 
# be subject to this injection via with_retry_handler().
ENV_VAR_GBTEST_SIMULATE_FAILURE_SCENARIO = "GBTEST_SIMULATE_FAILURE_SCENARIO"
GBTEST_SIMULATE_FAILURE_SCENARIO = os.getenv(ENV_VAR_GBTEST_MOCK_HF_CALLS, "").lower() == "true"
