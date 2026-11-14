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

# What fms-hf-tuning 3.2.2's caps resolve to (the spec's Tech Stack).
_RESOLVED = {
    "fms-hf-tuning": "3.2.2",
    "torch": "2.8.0",
    "transformers": "4.55.4",
    "trl": "0.19.1",
    "peft": "0.19.1",
    "accelerate": "1.15.0",
    "datasets": "4.8.5",
    "clearml": "2.1.12",
}
_FLASH_ATTN = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
    "flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)


def _instructions(name: str) -> list[str]:
    return re.findall(rf"^{name}\s+(.+)$", _DOCKERFILE, re.M)


def _lock_entries() -> dict[str, str]:
    return {
        m.group(1).lower(): m.group(2)
        for m in re.finditer(r"^([A-Za-z0-9_.\-]+)==([^\s\\]+)", _LOCK, re.M)
    }


def test_every_base_is_digest_pinned():
    froms = _instructions("FROM")
    assert len(froms) == 2
    for line in froms:
        assert re.search(r"@sha256:[0-9a-f]{64}", line), line
    assert froms[1].startswith("nvcr.io/nvidia/cuda:12.8.2-base-ubuntu24.04@")


def test_direct_pins():
    assert "fms-hf-tuning==3.2.2" in _IN
    assert "clearml==2.1.12" in _IN
    assert f"flash-attn @ {_FLASH_ATTN}" in _IN


def test_lock_resolves_the_spec_versions():
    entries = _lock_entries()
    for pkg, version in _RESOLVED.items():
        assert entries.get(pkg) == version, (pkg, entries.get(pkg))


def test_flash_attn_wheel_is_hashed_in_the_lock():
    block = _LOCK[_LOCK.index(f"flash-attn @ {_FLASH_ATTN}") :].split("\n#")[0]
    assert re.search(r"--hash=sha256:[0-9a-f]{64}", block)


def test_every_lock_entry_is_pinned_and_hashed():
    blocks = re.split(r"\n(?=[A-Za-z0-9])", _LOCK.split("\n", 1)[1])
    reqs = [b for b in blocks if re.match(r"[A-Za-z0-9]", b)]
    assert len(reqs) > 50
    for block in reqs:
        first = block.splitlines()[0]
        assert "==" in first or " @ https://" in first, first
        assert "--hash=sha256:" in block, first


def test_install_requires_hashes():
    assert "--require-hashes --no-deps -r /tmp/requirements.lock" in _DOCKERFILE


def test_managed_python_ships_no_pip():
    install = _DOCKERFILE.index("--require-hashes")
    removal = re.search(
        r"rm -rf /opt/uv-python/\*/lib/python3\.12/site-packages/pip "
        r"/opt/uv-python/\*/lib/python3\.12/site-packages/pip-\*\.dist-info "
        r"/opt/uv-python/\*/lib/python3\.12/ensurepip",
        _DOCKERFILE,
    )
    assert removal and install < removal.start()


def test_no_bytecode_at_build_time():
    assert re.findall(r"UV_COMPILE_BYTECODE=(\d)", _DOCKERFILE) == ["0"]
    assert _DOCKERFILE.index("UV_COMPILE_BYTECODE=0") < _DOCKERFILE.index(
        "--require-hashes"
    )


def test_runs_as_root_for_skypilot():
    assert not _instructions("USER")
    assert not re.search(r"^RUN useradd", _DOCKERFILE, re.M)


def test_layout():
    assert "COPY src/ /opt/fms-hf-tuning/" in _DOCKERFILE
    assert "PATH=/opt/fms-hf-tuning/.venv/bin:${PATH}" in _DOCKERFILE
    assert "FMS_HF_TUNING_HOME=/opt/fms-hf-tuning" in _DOCKERFILE


def test_venv_readable_without_a_copy_up():
    install = _DOCKERFILE.index("--require-hashes")
    copy_src = _DOCKERFILE.index("COPY src/")
    assert install < copy_src
    after = _DOCKERFILE[copy_src:]
    assert "chmod -R a+rX /opt/fms-hf-tuning " not in after
    assert "chmod -R a+rX /opt/fms-hf-tuning\n" not in after


def test_makefile_identity():
    mk = (_STEP_DIR / "Makefile").read_text()
    assert re.search(r"^STEP_NAME := fms-hf-tuning$", mk, re.M)
    assert re.search(r"^REGISTRY \?= quay\.io/granitebuild$", mk, re.M)
    assert re.search(r"^IMAGE_VERSION \?= fms-v3\.2\.2-\d+$", mk, re.M)
    assert "include ../../../common.mk" in mk


def test_every_script_and_config_the_step_calls_exists():
    run = (_STEP_DIR / "src" / "fms_hf_tuning_run.sh").read_text()
    for name in re.findall(r'cfg="(\w+\.yaml)"', run):
        assert (_STEP_DIR / "src" / "accelerate" / name).is_file(), name
    for script in re.findall(r"\$\{home\}/(\w+\.(?:sh|py))", run):
        assert (_STEP_DIR / "src" / script).is_file(), script
    template = (_STEP_DIR / "step-template.yaml").read_text()
    scripts = re.findall(r"/opt/fms-hf-tuning/(\w+\.(?:sh|py))", template)
    assert set(scripts) == {
        "fms_hf_tuning_guard.sh",
        "fms_hf_tuning_setup.sh",
        "fms_hf_tuning_run.sh",
    }
    for script in scripts:
        assert (_STEP_DIR / "src" / script).is_file(), script


def test_dockerignore_keeps_bytecode_out_of_the_image():
    # Local test runs leave src/__pycache__; COPY src/ would ship it.
    ignore = (_STEP_DIR / ".dockerignore").read_text().split()
    assert "**/__pycache__" in ignore and "**/*.pyc" in ignore
