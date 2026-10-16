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

"""Build tests whose ``hf://`` outputs are verified on the Hub, then deleted.

A fixture's build.yaml names its pushed datasets with two placeholders:

  ``hf://huggingface.co/datasets/__GBTEST_HF_PUSH_ORG__/<name>___GBTEST_HF_PUSH_RUN__``

Before the build runs, :class:`HfPushCleanupMixin` replaces them with the push
org (``GBTEST_HF_PUSH_ORG``, default ``ibm-research`` — the org the nightly's
token writes, as the slurm tests' pushes do) and a fresh per-run suffix, and runs
the rendered copy. So the test knows each dataset's exact id: after a successful
build it asserts each one exists — proof the push landed, not just that the push
step exited 0 — and, pass or fail, it deletes whichever exist, so runs do not pile
datasets up in the org.

Pushing needs an ``HF_TOKEN`` that can write the org, and deleting needs one that
can delete there. Locally, to push to your own namespace with your own token::

    GBTEST_HF_PUSH_ORG=<your-hf-user> GBTEST_SPS_ENABLE_ENV_VAR_OVERRIDE=true \\
        HF_TOKEN=<token> pytest ...

(the override keeps the secret manager's ``hf-token`` from replacing yours). A
token that may push but not delete only logs a warning: cleanup is best effort.
"""

import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Self

from libgbtest.buildrunner.buildtest import AbstractYamlBuildRunnerTest

logger = logging.getLogger(__name__)

ENV_VAR_GBTEST_HF_PUSH_ORG = "GBTEST_HF_PUSH_ORG"
DEFAULT_GBTEST_HF_PUSH_ORG = "ibm-research"

ORG_PLACEHOLDER = "__GBTEST_HF_PUSH_ORG__"
RUN_PLACEHOLDER = "__GBTEST_HF_PUSH_RUN__"

_DATASET_URI = re.compile(r"hf://huggingface\.co/datasets/([\w.-]+/[\w.-]+)")


def hf_push_org() -> str:
    """The HF org (or user namespace) the fixtures push to."""
    return os.environ.get(ENV_VAR_GBTEST_HF_PUSH_ORG) or DEFAULT_GBTEST_HF_PUSH_ORG


def render_push_build_yaml(src: Path, dest_dir: Path) -> tuple[Path, list[str]]:
    """Write ``src`` with the push placeholders filled into ``dest_dir``.

    :param src: The fixture's build.yaml.
    :param dest_dir: Directory for the rendered copy.
    :returns: The rendered build.yaml and the dataset ids it pushes to.
    :raises AssertionError: If ``src`` pushes to no placeholder-named dataset.
    """
    text = src.read_text(encoding="utf-8")
    run = uuid.uuid4().hex[:8]
    text = text.replace(ORG_PLACEHOLDER, hf_push_org()).replace(RUN_PLACEHOLDER, run)
    repos = sorted({m for m in _DATASET_URI.findall(text) if m.endswith(run)})
    assert repos, f"{src} has no hf:// dataset output named with {RUN_PLACEHOLDER}"
    dest = dest_dir / src.name
    dest.write_text(text, encoding="utf-8")
    return dest, repos


class HfPushCleanupMixin(AbstractYamlBuildRunnerTest):
    """Verify and delete a YAML build test's ``hf://`` dataset outputs.

    Put it before the test's base class. The fixture's build.yaml must name its
    pushed datasets with :data:`ORG_PLACEHOLDER` and :data:`RUN_PLACEHOLDER`.
    """

    def _run_yaml_spec(self: Self, test_key: str, test_cancel: bool) -> None:
        from huggingface_hub import HfApi

        with tempfile.TemporaryDirectory(prefix="gbtest-hfpush-") as tmp:
            build_yaml, repos = render_push_build_yaml(
                self._get_yaml_spec_dir() / "build.yaml", Path(tmp)
            )
            self._build_yaml_override = str(build_yaml)
            api = HfApi(token=os.environ.get("HF_TOKEN") or None)
            try:
                super()._run_yaml_spec(test_key, test_cancel)
                if not test_cancel:
                    # A cancelled build may stop before or after a push, so only
                    # the full run asserts.
                    missing = [
                        r for r in repos if not api.repo_exists(r, repo_type="dataset")
                    ]
                    assert not missing, f"hf:// outputs not on the Hub: {missing}"
            finally:
                del self._build_yaml_override
                for repo in repos:
                    try:
                        api.delete_repo(repo, repo_type="dataset", missing_ok=True)
                    except Exception as e:  # pylint: disable=broad-exception-caught
                        logger.warning("could not delete test dataset %s: %s", repo, e)
