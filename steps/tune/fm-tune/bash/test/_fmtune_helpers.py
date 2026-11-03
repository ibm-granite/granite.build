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
if os.environ.get("STUB_HOOK"):  # a test's step that runs while "training"
    import subprocess
    stage = args[args.index("--output_dir") + 1]
    subprocess.run(["/bin/sh", "-c", os.environ["STUB_HOOK"]], check=True,
                   env=dict(os.environ, STUB_STAGE=stage))
if os.environ.get("STUB_EXIT"):
    sys.exit(int(os.environ["STUB_EXIT"]))
out = args[args.index("--output_dir") + 1]
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "adapter_config.json"), "w").write("{}")
"""


def _run(env: dict, cwd=None) -> subprocess.CompletedProcess:
    """Run fmtune_run.sh with exactly ``env`` (nothing inherited) and capture it.

    :param env: the whole environment for the script.
    :param cwd: the directory to start in; None means the test's own.
    :returns: the finished process, with text stdout and stderr; never raises on
        a non-zero exit.
    """
    return subprocess.run(
        ["bash", str(_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
    )


def _clear_output(env: dict) -> None:
    """Remove the run's output dir, so a test can run the script again.

    The script refuses a non-empty output dir, and a successful run leaves the
    model there.

    :param env: the script env; its ``FMTUNE_OUTPUT_DIR`` is removed if present.
    """
    shutil.rmtree(env["FMTUNE_OUTPUT_DIR"], ignore_errors=True)


def _record(env: dict) -> dict:
    """Read what the stub main.py recorded on its last run.

    :param env: the script env; its ``STUB_RECORD`` names the record file.
    :returns: the record: ``argv``, ``cwd``, ``log_path``, ``path``, and
        ``config_text`` when ``STUB_CONFIG_COPY`` was set.
    """
    return json.loads(pathlib.Path(env["STUB_RECORD"]).read_text())


def _flag(argv: list[str], name: str) -> str:
    """Return the value that follows flag ``name`` in ``argv``.

    :param argv: an argument list, as in ``_record(env)["argv"]``.
    :param name: the flag, for example ``"--output_dir"``.
    :returns: the next argument; raises ValueError if the flag is absent.
    """
    return argv[argv.index(name) + 1]


def _staged_as(argv: list[str], out) -> bool:
    """Whether main.py was given a staging dir for output dir ``out``.

    The script trains in ``<out>.partial.<random>`` and renames it onto ``out``.

    :param argv: main.py's arguments, as in ``_record(env)["argv"]``.
    :param out: the output dir the model is published to.
    :returns: True when ``--output_dir`` is ``<out>.partial.`` plus a suffix.
    """
    stage = _flag(argv, "--output_dir")
    prefix = f"{out}.partial."
    return stage.startswith(prefix) and len(stage) > len(prefix)


def _git(*args, cwd):
    """Run one git command quietly, failing the test if it fails.

    :param args: git's arguments.
    :param cwd: the directory to run git in.
    """
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
    """Return the commit SHA that HEAD names in the repo at ``path``.

    :param path: a git work tree.
    :returns: the full 40-character SHA.
    """
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo, msg):
    """Stage every change in ``repo`` and commit it.

    :param repo: a git work tree, as made by ``_git_repo``.
    :param msg: the commit message.
    """
    _git("add", "-A", ".", cwd=repo)
    _git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", msg, cwd=repo)


def _clone_lock(env) -> pathlib.Path:
    """Return the clone-mirror lock path the script uses for the env's repo@ref.

    Computes the key the way fmtune_run.sh does (cksum of ``<repo>@<ref>``).

    :param env: the script env; reads ``FMTUNE_SOURCE_REPO``, ``FMTUNE_SOURCE_REF``
        and ``FMTUNE_CACHE_DIR``.
    :returns: ``<cache>/src-<key>.lock``; it need not exist.
    """
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
    """Write an executable /bin/sh script ``bin_dir/name``, for putting on PATH.

    :param bin_dir: the directory for the stub; created if missing.
    :param name: the command it stands in for, for example ``"mv"``.
    :param body: the script's text after the ``#!/bin/sh`` line.
    """
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
