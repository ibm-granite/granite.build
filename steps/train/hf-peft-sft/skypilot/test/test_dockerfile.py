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

"""Static checks on the image recipe: supply chain and runtime layout.

The image is too large to build in a unit test, so this pins what review would
otherwise catch: digest-pinned bases, a fully hashed lock with the spec's exact
pins, root (SkyPilot), and that every path the step calls exists in src/.
"""

import pathlib
import re

_STEP_DIR = pathlib.Path(__file__).resolve().parents[1]
_DOCKERFILE = (_STEP_DIR / "Dockerfile").read_text()
_LOCK = (_STEP_DIR / "requirements.lock").read_text()
_IN = (_STEP_DIR / "requirements.in").read_text()

_PINS = {
    "peft": "0.21.2",
    "trl": "1.17.1",
    "transformers": "5.22.1",
    "accelerate": "1.15.0",
    "bitsandbytes": "0.50.2",
    "deepspeed": "0.19.9",
    "datasets": "5.1.0",
    "clearml": "2.1.12",
}


def _instructions(name: str) -> list[str]:
    return re.findall(rf"^{name}\s+(.+)$", _DOCKERFILE, re.M)


def _lock_entries() -> dict[str, str]:
    """name -> version for every top-level requirement in the lock."""
    return {
        m.group(1).lower(): m.group(2)
        for m in re.finditer(r"^([A-Za-z0-9_.\-]+)==([^\s\\]+)", _LOCK, re.M)
    }


def test_every_base_is_digest_pinned():
    froms = _instructions("FROM")
    assert len(froms) >= 2
    for line in froms:
        assert re.search(r"@sha256:[0-9a-f]{64}", line), line


def test_direct_pins_match_the_spec():
    for pkg, version in _PINS.items():
        assert f"{pkg}=={version}" in _IN, pkg
        assert _lock_entries()[pkg] == version, pkg
    assert re.search(r"^torch==\d+\.\d+\.\d+", _IN, re.M)


def test_every_lock_entry_is_pinned_and_hashed():
    blocks = re.split(r"\n(?=[A-Za-z0-9])", _LOCK.split("\n", 1)[1])
    reqs = [b for b in blocks if re.match(r"[A-Za-z0-9]", b)]
    assert len(reqs) > 50
    for block in reqs:
        assert "==" in block.splitlines()[0], block
        assert "--hash=sha256:" in block, block.splitlines()[0]


def test_install_requires_hashes():
    assert "--require-hashes" in _DOCKERFILE
    assert "requirements.lock" in _DOCKERFILE


def test_no_excluded_packages():
    entries = _lock_entries()
    for pkg in ("flash-attn", "flash_attn", "vllm", "unsloth"):
        assert pkg not in entries, pkg


def test_runs_as_root_for_skypilot():
    assert not _instructions("USER")
    assert not re.search(r"^RUN useradd", _DOCKERFILE, re.M)


def test_layout():
    assert "COPY src/ /opt/hf-peft-sft/" in _DOCKERFILE
    assert "PATH=/opt/hf-peft-sft/.venv/bin:${PATH}" in _DOCKERFILE
    assert "HF_PEFT_SFT_HOME=/opt/hf-peft-sft" in _DOCKERFILE


def test_venv_readable_without_a_copy_up():
    # chmod the venv in its own install layer; a later recursive chmod over
    # /opt/hf-peft-sft would copy the whole venv layer up again.
    install = _DOCKERFILE.index("--require-hashes")
    copy_src = _DOCKERFILE.index("COPY src/")
    assert install < copy_src
    after = _DOCKERFILE[copy_src:]
    assert "chmod -R a+rX /opt/hf-peft-sft " not in after
    assert "chmod -R a+rX /opt/hf-peft-sft\n" not in after


def test_makefile_identity():
    mk = (_STEP_DIR / "Makefile").read_text()
    assert re.search(r"^STEP_NAME := hf-peft-sft$", mk, re.M)
    assert re.search(r"^REGISTRY \?= quay\.io/granitebuild$", mk, re.M)
    assert re.search(r"^IMAGE_VERSION \?= peft-v0\.21\.2-\d+$", mk, re.M)
    assert "include ../../../common.mk" in mk


def test_every_script_and_config_the_step_calls_exists():
    run = (_STEP_DIR / "src" / "hf_peft_sft_run.sh").read_text()
    for name in re.findall(r'cfg="(\w+\.yaml)"', run) + re.findall(
        r"accelerate/(\w+\.yaml)", run
    ):
        assert (_STEP_DIR / "src" / "accelerate" / name).is_file(), name
    template = _STEP_DIR / "step-template.yaml"
    if template.exists():  # written in Task 8
        for script in re.findall(
            r"/opt/hf-peft-sft/(\w+\.(?:sh|py))", template.read_text()
        ):
            assert (_STEP_DIR / "src" / script).is_file(), script
