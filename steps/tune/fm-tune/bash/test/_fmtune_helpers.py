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

"""Shared helpers for the fmtune_run.sh unit tests (not a test module).

The stub ``main.py`` records its argv and cwd as JSON, then fakes a saved model at
the root of ``<output_dir>`` (where ``--model_at_root`` leaves it). The ``env``
fixture that pairs with it is in conftest.py.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "src" / "fmtune_run.sh"

_STUB_MAIN = r"""
import json, os, sys
args = sys.argv[1:]
rec = {"argv": args, "cwd": os.getcwd(), "log_path": os.environ.get("LOG_PATH"),
       "path": os.environ.get("PATH")}
if os.environ.get("STUB_CONFIG_COPY"):
    cfg = args[args.index("--config_file") + 1]
    rec["config_text"] = open(cfg).read()
json.dump(rec, open(os.environ["STUB_RECORD"], "w"))
if os.environ.get("STUB_EXIT"):
    sys.exit(int(os.environ["STUB_EXIT"]))
out = args[args.index("--output_dir") + 1]
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "adapter_config.json"), "w").write("{}")
"""


def _run(env: dict, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
    )


def _record(env: dict) -> dict:
    return json.loads(pathlib.Path(env["STUB_RECORD"]).read_text())


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _git_repo(env, tmp_path):
    """A local file:// repo on branch rel holding the stub fm-tune under sub/fm-tune."""
    if shutil.which("git") is None:  # pragma: no cover
        pytest.skip("git not available")
    repo = tmp_path / "repo"
    shutil.copytree(env["FMTUNE_SOURCE_PATH"], repo / "sub" / "fm-tune")
    _git("init", "-q", "-b", "rel", cwd=repo)
    _git("add", ".", cwd=repo)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x", cwd=repo)
    _git(
        "config", "uploadpack.allowAnySHA1InWant", "true", cwd=repo
    )  # as GitHub serves any SHA
    env.update(
        FMTUNE_SOURCE_PATH="",
        FMTUNE_SOURCE_REPO=f"file://{repo}",
        FMTUNE_SOURCE_REF="rel",
        FMTUNE_SOURCE_SUBDIR="sub/fm-tune",
    )
    return repo


def _head(path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo, msg):
    _git("add", "-A", ".", cwd=repo)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", msg, cwd=repo)


def _clone_lock(env) -> pathlib.Path:
    key = subprocess.run(
        [
            "bash",
            "-c",
            'printf "%s@%s" "$0" "$1" | cksum | cut -d" " -f1',
            env["FMTUNE_SOURCE_REPO"],
            env["FMTUNE_SOURCE_REF"],
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return pathlib.Path(env["FMTUNE_CACHE_DIR"]) / f"src-{key}.lock"


def _stub(bin_dir: pathlib.Path, name: str, body: str) -> None:
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / name).write_text("#!/bin/sh\n" + body)
    (bin_dir / name).chmod(0o755)


def _git_spy(env, tmp_path):
    """Put a git first on PATH that records each call and fails."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "git").write_text(f"#!/bin/sh\ntouch {tmp_path}/git-ran\nexit 1\n")
    (bin_dir / "git").chmod(0o755)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    return tmp_path / "git-ran"
