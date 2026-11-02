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

"""Static checks on the step's Dockerfile: the supply-chain and runtime rules.

The image is too large to build in a unit test, so this pins what review would
otherwise have to catch: every base is digest-pinned, Open-Instruct is pinned to
a full commit, the image sets no non-root USER (SkyPilot needs root), and the
paths step.yaml calls exist.
"""

import pathlib
import re

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_DOCKERFILE = (_STEP_DIR / "Dockerfile").read_text()
_TEMPLATE = (_STEP_DIR / "step-template.yaml").read_text()


def _instructions(name: str) -> list[str]:
    return re.findall(rf"^{name}\s+(.+)$", _DOCKERFILE, re.M)


def test_every_base_is_digest_pinned():
    froms = _instructions("FROM")
    assert froms
    for line in froms:
        assert re.search(r"@sha256:[0-9a-f]{64}", line), line


def test_open_instruct_pinned_to_a_full_commit():
    assert re.search(r"^ARG OPEN_INSTRUCT_COMMIT=[0-9a-f]{40}$", _DOCKERFILE, re.M)
    assert 'test "$(git rev-parse HEAD)" = "${OPEN_INSTRUCT_COMMIT}"' in _DOCKERFILE


def test_locked_install():
    assert "uv sync --frozen" in _DOCKERFILE
    assert re.search(r"clearml==\d+\.\d+\.\d+", _DOCKERFILE)


def test_runs_as_root_for_skypilot():
    # SkyPilot's docker runtime saves root's HOME to /etc/profile.d during setup, so a
    # non-root default USER logs in with HOME=/root and provisioning fails.
    assert not _instructions("USER")
    assert not re.search(r"^RUN useradd", _DOCKERFILE, re.M)


def test_records_commit_without_git_commit_env():
    assert "OPEN_INSTRUCT_COMMIT=${OPEN_INSTRUCT_COMMIT}" in _DOCKERFILE
    assert not re.search(r"^\s*GIT_COMMIT=", _DOCKERFILE, re.M)


def test_scripts_land_where_step_yaml_calls_them():
    assert "COPY src/ /opt/openinstruct-sft/" in _DOCKERFILE
    for script in re.findall(r"/opt/openinstruct-sft/(\w+\.sh)", _TEMPLATE):
        assert (_STEP_DIR / "src" / script).is_file(), script


_PATCHED = {
    "pillow": "13.0.0",
    "nltk": "3.10.3",
    "litellm": "1.84.0",
    "PyJWT": "2.14.0",
    "Authlib": "1.9.0",
    "fastmcp": "3.2.0",
    "anyio": "4.14.2",
    "GitPython": "3.1.59",
    "certifi": "2026.10.19",
    "cryptography": "46.0.7",
    "wandb": "0.31.0",
}


def test_security_patches_pinned():
    for pkg, version in _PATCHED.items():
        assert f'"{pkg}=={version}"' in _DOCKERFILE, pkg


def test_nsight_profiler_purged():
    assert (
        "apt-get purge -y cuda-nsight-compute-12-8 nsight-compute-2025.1.1"
        in _DOCKERFILE
    )


def test_vllm_removed_not_upgraded():
    assert "uv pip uninstall --python .venv/bin/python vllm" in _DOCKERFILE
    assert not re.search(r'"vllm==', _DOCKERFILE)


def test_patches_run_after_the_frozen_sync():
    sync = _DOCKERFILE.index("uv sync --frozen")
    assert sync < _DOCKERFILE.index("uv pip uninstall --python .venv/bin/python vllm")
    assert sync < _DOCKERFILE.index('"pillow==13.0.0"')


def test_finetune_patch_applied_before_install():
    assert (
        "COPY patches/finetune_local_model.py /tmp/finetune_local_model.py"
        in _DOCKERFILE
    )
    assert (
        "python /tmp/finetune_local_model.py open_instruct/finetune.py" in _DOCKERFILE
    )
    assert _DOCKERFILE.index(
        "finetune_local_model.py open_instruct/finetune.py"
    ) < _DOCKERFILE.index("uv sync --frozen")
