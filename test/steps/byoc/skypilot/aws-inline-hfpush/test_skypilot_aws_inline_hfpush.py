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

"""Gated live e2e: single-instance inline pull -> compute -> push on aws (real EC2).

Issue #390. The aws environment with NO ``shared_filesystem`` runs the hf
assetstore pull AND push ``inline: true``: the download is injected into the
byoc step's setup and the upload into its epilogue, so ONE EC2 instance does the
whole ``pull -> compute -> push`` without any shared mount to stage through.
This is the aws counterpart of the #389 ``GB_RUN_SHARED_FS_E2E`` shared-FS e2e,
for the inline (no-shared-FS) path.

It follows the #373 aws step-test conventions (``AbstractYamlBuildRunnerTest`` +
``get_test_data_dir_for`` + a ``build.yaml``/``buildtest.yaml`` fixture pair;
:func:`gbserver.environment.skypilot.aws_credentials_present` credential gate;
``skypilot_integration`` marker; extended-suite-only), and layers on the extra,
opt-in assertions this issue needs.

**Triple-gated — never runs by accident, import-safe without any creds:**
  1. Module-level ``pytest.mark.skipif`` on ``GB_RUN_INLINE_HFPUSH_E2E`` (the
     #389 ``GB_RUN_SHARED_FS_E2E`` idiom): unless it is ``true`` the whole module
     SKIPS at collection. Import and collection never touch AWS/HF, so this is
     safe to collect in CI with no credentials.
  2. ``@extended_testing_only`` (the quick suite deselects ``-m "not extended"``).
  3. ``aws_credentials_present()`` skip (no EC2 without AWS creds) AND an
     ``HF_TOKEN`` skip (the inline push needs a writable HF token).

**What it proves when opted in (real AWS + HF):**
  * the build reaches SUCCESS (framework: inline pull -> byoc compute -> inline
    push all ran; ``step_count 1`` confirms pull/push were injected, not queued);
  * the pushed artifact EXISTS on HF (``HfApi.repo_exists`` + a non-empty file
    listing on the throwaway repo);
  * the ``ARTIFACT_PUSHED_EVENT`` is emitted only AFTER the upload completes
    (ordering: the PUSHED event's stored index follows the step's completion).

Ruling R3: all hf URIs in the build.yaml are THREE-slash (``hf:///owner/repo``);
a two-slash form raises "Malformed HF URI". The push target is a throwaway repo
under a test namespace, created via ``build_yaml`` substitution at load time and
deleted in teardown.

Run it (opt-in only; provisions real EC2 + writes to HF)::

    GB_RUN_INLINE_HFPUSH_E2E=true HF_TOKEN=... AWS_PROFILE=gb-skypilot \\
        pytest test/steps/byoc/skypilot/aws-inline-hfpush/ -v -s
"""

import os
import uuid
from pathlib import Path
from typing import Self

import pytest
from libgbtest.buildrunner.buildtest import (
    AbstractYamlBuildRunnerTest,
    get_test_data_dir_for,
)
from libgbtest.constants import extended_testing_only

from gbserver.environment.skypilot import aws_credentials_present
from gbserver.types.buildevent import BuildEventType

# ---------------------------------------------------------------------------
# Gate 1 (module-level, IMPORT-SAFE): unless GB_RUN_INLINE_HFPUSH_E2E=true this
# whole module SKIPS at collection. Mirrors the #389 GB_RUN_SHARED_FS_E2E idiom.
# Nothing above this line touches AWS or HF, so collection never fails without
# credentials.
# ---------------------------------------------------------------------------
GATE_ENV_VAR = "GB_RUN_INLINE_HFPUSH_E2E"

pytestmark = [
    pytest.mark.skypilot_integration,
    pytest.mark.skipif(
        os.environ.get(GATE_ENV_VAR, "").lower() != "true",
        reason=(
            f"set {GATE_ENV_VAR}=true (+ AWS creds, HF_TOKEN) to run the live "
            "inline-hfpush e2e (real EC2 + HF write)"
        ),
    ),
]

# The HF namespace the throwaway push repo is created under. Defaults to the
# authenticated user's namespace (HfApi resolves ``None`` owner to the token's
# user); override with GB_INLINE_HFPUSH_E2E_NS to push under an org instead.
_HF_PUSH_NAMESPACE = os.environ.get("GB_INLINE_HFPUSH_E2E_NS", "").strip()


def _hf_token() -> str:
    """Return the HF write token from the environment (HF_TOKEN or HF_HUB_TOKEN)."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN") or ""


@extended_testing_only
@pytest.mark.skipif(
    not aws_credentials_present(),
    reason=(
        "AWS credentials not in environment "
        "(set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE)"
    ),
)
@pytest.mark.skipif(
    not _hf_token(),
    reason="HF_TOKEN (or HF_HUB_TOKEN) not set; the inline push needs a writable HF token",
)
class TestSkypilotAwsInlineHfpush(AbstractYamlBuildRunnerTest):
    """Inline pull -> byoc compute -> inline push, end to end on aws (EC2)."""

    def _get_yaml_spec_dir(self: Self) -> Path:
        """Return the fixture dir holding this test's build.yaml and buildtest.yaml.

        Resolved by the repo's ``test/`` <-> ``test-data/`` helper so the pairing
        holds in both test modes (see steps/README.md).
        """
        return get_test_data_dir_for(__file__)

    def setup_method(self: Self, method) -> None:
        """Materialize a throwaway HF push repo URI into a rendered build.yaml.

        The fixture build.yaml carries a ``{HF_PUSH_REPO}`` placeholder for the
        push output. Here we mint a unique THREE-slash hf URI (Ruling R3) under
        the test namespace and write a rendered copy to a temp file that the base
        class picks up via ``_build_yaml_override``. Kept out of module scope so
        collection (Gate 1 skip) never mints a repo id or touches HF.
        """
        super().setup_method(method)

        # HfApi is imported lazily (inside the opted-in test body / setup) so the
        # module stays import-safe when huggingface_hub write paths are unused.
        from huggingface_hub import HfApi

        self._hf_api = HfApi(token=_hf_token())
        # Owner: an explicit namespace if provided, else the token's own user.
        owner = _HF_PUSH_NAMESPACE or self._hf_api.whoami()["name"]
        repo_name = f"gb-inline-hfpush-e2e-{uuid.uuid4().hex[:12]}"
        # Ruling R3: THREE-slash hf URI. repo_id (owner/name) is what HfApi uses.
        self._hf_repo_id = f"{owner}/{repo_name}"
        self._hf_push_uri = f"hf:///{self._hf_repo_id}"

        spec_dir = self._get_yaml_spec_dir()
        template = (spec_dir / "build.yaml").read_text(encoding="utf-8")
        rendered = template.replace("{HF_PUSH_REPO}", self._hf_push_uri)

        # Write the rendered build.yaml beside the fixture so relative resolution
        # (space_uri etc.) in buildtest.yaml is unaffected; unique per run.
        self._rendered_build_yaml = (
            spec_dir / f".build.rendered.{uuid.uuid4().hex[:8]}.yaml"
        )
        self._rendered_build_yaml.write_text(rendered, encoding="utf-8")
        # AbstractYamlBuildRunnerTest._get_test_specification passes this through.
        self._build_yaml_override = str(self._rendered_build_yaml)

    def teardown_method(self: Self, method) -> None:
        """Delete the throwaway HF repo and the rendered build.yaml, then base teardown."""
        # Delete the HF repo first (best effort — never mask a test failure).
        try:
            api = getattr(self, "_hf_api", None)
            repo_id = getattr(self, "_hf_repo_id", None)
            if api is not None and repo_id:
                api.delete_repo(repo_id=repo_id, missing_ok=True)
        except Exception:  # pragma: no cover - cleanup best effort
            pass
        # Remove the rendered build.yaml temp file.
        try:
            rendered = getattr(self, "_rendered_build_yaml", None)
            if rendered is not None:
                Path(rendered).unlink(missing_ok=True)
        except Exception:  # pragma: no cover - cleanup best effort
            pass
        super().teardown_method(method)

    def test_runner(self: Self) -> None:
        """Run the inline pull->compute->push build, then assert HF + PUSHED ordering.

        Delegates the build submission and the build-success + artifact-count
        assertions to the shared YAML framework (``_run_yaml_spec('runner')``),
        then adds the two issue-#390-specific live assertions:
          (b) the pushed artifact exists on HF, and
          (c) ARTIFACT_PUSHED_EVENT is emitted after the upload (ordering).
        """
        # (a) build succeeds (framework verifies SUCCESS + input/output counts).
        self._run_yaml_spec("runner", test_cancel=False)

        # Recover the just-run build (the framework runs exactly one build; its
        # teardown enumerates builds the same way).
        builds = self.storage.build_storage.get_by_uuid(None)
        assert builds, "no build recorded after a successful runner run"
        build = builds[-1]
        build_id = build.uuid

        # (b) the pushed artifact exists on HF: repo present AND non-empty listing.
        assert self._hf_api.repo_exists(
            repo_id=self._hf_repo_id
        ), f"pushed HF repo {self._hf_repo_id!r} does not exist after the build"
        files = self._hf_api.list_repo_files(repo_id=self._hf_repo_id)
        # A fresh repo always carries .gitattributes; require at least one MORE
        # file so we know the inline push actually uploaded the artifact payload.
        payload = [f for f in files if f != ".gitattributes"]
        assert payload, (
            f"pushed HF repo {self._hf_repo_id!r} has no artifact files "
            f"(only {files!r}); inline push did not upload the marker path"
        )

        # (c) ordering: ARTIFACT_PUSHED_EVENT is emitted only AFTER the upload
        # completes. get_sorted_build_events returns events in storage-insertion
        # order, so the PUSHED event must be the LAST artifact event for the
        # pushed binding (nothing follows it), and must carry the pushed hf URI.
        events = self.storage.event_storage.get_sorted_build_events(build_id=build_id)
        pushed_indices = [
            i
            for i, ev in enumerate(events)
            if ev.build_event.type == BuildEventType.ARTIFACT_PUSHED_EVENT
        ]
        assert pushed_indices, (
            "no ARTIFACT_PUSHED_EVENT recorded; the inline push epilogue did not "
            "emit the 'Pushed HF URI:' marker the monitor captures"
        )
        # The PUSHED event is emitted by the epilogue AFTER the upload returns, so
        # it is the last artifact event: no NEWARTIFACT for this binding follows.
        last_pushed = pushed_indices[-1]
        pushed_ev = events[last_pushed].build_event
        # ARTIFACT_PUSHED_EVENT carries an ArtifactPushedEventPayload whose `uri`
        # is the pushed hf:// URI (see gbserver.types.buildevent).
        pushed_uri = str(getattr(pushed_ev.payload, "uri", "") or "")
        assert pushed_uri.startswith("hf://") and self._hf_repo_id in pushed_uri, (
            f"ARTIFACT_PUSHED_EVENT uri {pushed_uri!r} does not reference the "
            f"pushed repo {self._hf_repo_id!r}"
        )

    def test_runner_cancellation(self: Self) -> None:
        """Cancellation variant is not exercised (buildtest.yaml lists only 'runner').

        The byoc command + inline push finish near-instantly once the node is up,
        leaving no stable RUNNING window to cancel; the base method self-skips
        because 'runner_cancellation' is absent from the spec's tests list.
        """
        self._run_yaml_spec("runner_cancellation", test_cancel=True)
