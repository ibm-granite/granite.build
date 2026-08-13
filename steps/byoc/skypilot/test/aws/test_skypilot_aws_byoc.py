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

"""Integration test: the byoc step runs end to end on Skypilot/aws (real EC2).

This is the aws counterpart of ``test_skypilot_byoc.py`` (slurm). It is a
**step-level** test: it lives beside the byoc step's Makefile in a per-cluster
subdir (``steps/byoc/skypilot/test/aws/``), with its fixtures in the matching
``test-data/aws/``, and is developed and run independently of the repository's
central suite (it is not in ``testpaths``). Run it via ``make test`` with the
repo-root ``.venv`` activated and AWS credentials exported::

    make -C steps/byoc/skypilot test

``make test`` depends on ``make space``, so the git-ignored ``space/`` directory
that the ``buildtest.yaml``'s ``space_uri`` points at is always rendered before
pytest runs — this test therefore assumes the Space already exists and does no
rendering of its own.

Exercises the public-image ``byoc`` step as a real build on a cloud backend: it
git-clones a tiny public repo (``octocat/Hello-World``) in ``setup``, runs a
``setup_command`` that writes a sentinel, then a ``command`` in ``run`` that
requires that sentinel (so a SUCCESS proves ``setup_command`` ran), registering
one output artifact.

Design notes:
  * Identical to the slurm fixture except ``environment_uri`` points at
    ``space://environments/skypilot/aws`` — the SAME step (``space://steps/byoc``)
    is proven on real EC2, resolved through the shared skypilot ``base_uris``
    chain to ``configurations/assets``.
  * **Bare node** (``image: ""``) so the command runs directly on the EC2
    instance (no container image needed).
  * **HF-free** (``env://`` in/out) so it needs no HF_TOKEN — only AWS creds.

**Real EC2 — never runs by accident.** It is in the extended suite only AND
skips unless boto3's credential env vars are present (see
:func:`gbserver.environment.skypilot.aws_credentials_present`), so no instance
is ever provisioned without credentials explicitly exported into the
environment.
"""

from pathlib import Path

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

from gbserver.environment.skypilot import aws_credentials_present

pytestmark = pytest.mark.skypilot_integration


# Real-infra build test (provisions a real EC2 instance via Skypilot) — only run
# in the extended suite (make extended-tests), and only when AWS credentials are
# present in the environment so it can never provision EC2 by accident. The
# credential gate is the shared gbserver predicate (aws_credentials_present).
@extended_testing_only
@pytest.mark.skipif(
    not aws_credentials_present(),
    reason=(
        "AWS credentials not in environment "
        "(set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE)"
    ),
)
class TestSkypilotAwsByoc(AbstractYamlBuildRunnerTest):
    """byoc clones a public repo and runs a command end to end on aws (EC2)."""

    def _get_yaml_spec_dir(self) -> Path:
        # Fixtures (build.yaml/buildtest.yaml) live in the test-data/ dir that
        # mirrors this file, resolved by the repo's test/ ↔ test-data/ helper —
        # which keys off the first `test/` segment, so it works from both homes
        # of this test (see steps/README.md, "Two test modes"):
        #   Mode 1 (authoring)  steps/byoc/skypilot/test/aws/
        #       -> steps/byoc/skypilot/test-data/aws/  (co-located)
        #   Mode 2 (published)  test/steps/byoc/skypilot/aws/
        #       -> test-data/steps/byoc/skypilot/aws/  (parallel top-level tree)
        # In Mode 1 the Space is rendered by `make space` (the `make test`
        # prerequisite) before this test runs.
        return get_test_data_dir_for(__file__)
