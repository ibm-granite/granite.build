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

"""The byoc step brings its own git when the image has none (#529).

byoc's ``setup`` clones the repo before ``setup_command`` runs, so a build
cannot install git itself. The step's default image (fedora-minimal) and the
official pytorch/pytorch images ship without git, and on some clusters (enroot
on LSF) the container is not root, so a system package manager cannot run.

The launcher's ``setup`` and ``run`` blocks are rendered the way gbserver
renders them and run in bash with a PATH that holds only stubs, so each case
controls exactly which of git / conda / apt-get / dnf / microdnf exist and what
``id -u`` reports. ``HOME`` is ``/``, as inside an enroot container, which is
not writable by the test user.
"""

import pathlib
import shutil
import subprocess

import pytest
import yaml

from gbserver.utils.template import fill_template

_REPO = pathlib.Path(__file__).resolve().parents[3]
_TEMPLATE = _REPO / "steps" / "misc" / "byoc" / "skypilot" / "step-template.yaml"
_SHA = "0123456789abcdef0123456789abcdef01234567"
_BASH = shutil.which("bash") or ""

pytestmark = pytest.mark.skipif(not _BASH, reason="bash not available")

# Logs one line per call (tool path, then tab-separated args) and acts out just
# enough of each tool for setup/run to continue.
_LOG = (
    'printf "%s" "$0" >> "$TRACE"; printf "\\t%s" "$@" >> "$TRACE"; echo >> "$TRACE"\n'
)

_GIT = (
    "#!/bin/bash\n"
    + _LOG
    + 'case "$1" in\n'
    + "  --version) echo 'git version 2.56.0' ;;\n"
    + '  clone) mkdir -p "$3" ;;\n'
    + f'  -C) [ "$3" = rev-parse ] && echo {_SHA} ;;\n'
    + "esac\n"
    + "exit 0\n"
)

# conda writes ~/.cache/conda and, when the image's package cache is not
# writable, ~/.conda/pkgs. With HOME=/ a non-root conda fails with
# "Permission denied: '/.cache'" (seen in condaforge/miniforge3 run as uid 1000),
# so the stub fails the same way, then writes its scratch and puts a git into
# the requested prefix.
_CONDA = (
    "#!/bin/bash\n"
    + _LOG
    + 'if [ "$HOME" = / ]; then echo "Permission denied: \'/.cache\'" >&2; exit 1; fi\n'
    + 'mkdir -p "$HOME/.cache/conda" "${CONDA_PKGS_DIRS:-$HOME/.conda/pkgs}/git-2.56.0"\n'
    + 'prefix=""\n'
    + 'while [ $# -gt 0 ]; do [ "$1" = -p ] && prefix="$2"; shift; done\n'
    + 'mkdir -p "$prefix/bin" && cp "$GIT_STUB" "$prefix/bin/git"\n'
)

# A system package manager: installing git puts it on PATH.
_PKG = (
    "#!/bin/bash\n"
    + _LOG
    + 'for a in "$@"; do [ "$a" = git ] && cp "$GIT_STUB" "$BIN/git"; done\n'
    + "exit 0\n"
)

_ID = '#!/bin/bash\n[ "$1" = -u ] && echo "$FAKE_UID"\n'

# Real tools the step and the stubs call; everything else is absent.
_REAL = ("rm", "mkdir", "cp")


def _launcher() -> dict:
    step = yaml.safe_load(_TEMPLATE.read_text())
    return step["environment_configs"]["Skypilot"]["launchers"]["byoc"]["config"]


def _render(block: str, **byoc_config) -> str:
    defaults = yaml.safe_load(_TEMPLATE.read_text())["config"]["byoc_config"]
    data = {"config": {"byoc_config": {**defaults, **byoc_config}}}
    return fill_template(_launcher()[block], data, strict=True)


class _Node:
    """A workdir plus a PATH holding only the given tools."""

    def __init__(self, tmp_path: pathlib.Path, tools=(), uid: int = 1000):
        self.work = tmp_path / "work"
        self.work.mkdir()
        self.bin = tmp_path / "bin"
        self.bin.mkdir()
        self.trace = tmp_path / "trace"
        self.trace.touch()
        self.git_stub = tmp_path / "git-stub"
        self.write(self.git_stub, _GIT)
        for name in _REAL:
            real = shutil.which(name)
            assert real, f"{name} not on PATH"
            (self.bin / name).symlink_to(real)
        self.write(self.bin / "id", _ID)
        stubs = {"git": _GIT, "conda": _CONDA}
        for name in tools:
            self.write(self.bin / name, stubs.get(name, _PKG))
        self.uid = uid

    @staticmethod
    def write(path: pathlib.Path, text: str) -> None:
        path.write_text(text)
        path.chmod(0o755)

    def run(self, script: str) -> subprocess.CompletedProcess:
        env = {
            "PATH": str(self.bin),
            "HOME": "/",
            "TRACE": str(self.trace),
            "GIT_STUB": str(self.git_stub),
            "BIN": str(self.bin),
            "FAKE_UID": str(self.uid),
        }
        return subprocess.run(
            [_BASH, "-c", script],
            cwd=self.work,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def calls(self, tool: str) -> list[list[str]]:
        """Argv of every call to *tool*, matched on the basename of its path."""
        out = []
        for line in self.trace.read_text().splitlines():
            path, *args = line.split("\t")
            if pathlib.Path(path).name == tool:
                out.append([path, *args])
        return out

    def clone_git(self) -> str:
        """Path of the git that ran the clone."""
        (clone,) = [c for c in self.calls("git") if c[1] == "clone"]
        return clone[0]


_SETUP = {"repo": "https://example.com/org/repo.git", "command": "true"}


def test_git_in_the_image_is_used_as_is(tmp_path):
    node = _Node(tmp_path, tools=("git", "conda", "apt-get"), uid=0)
    proc = node.run(_render("setup", **_SETUP))
    assert proc.returncode == 0, proc.stderr
    assert node.clone_git() == str(node.bin / "git")
    assert not node.calls("conda")
    assert not node.calls("apt-get")


def test_conda_installs_git_for_a_non_root_container(tmp_path):
    """enroot on LSF: no root, HOME=/ (read-only), conda in the image."""
    node = _Node(tmp_path, tools=("conda", "apt-get"), uid=1000)
    proc = node.run(_render("setup", ref="v1", **_SETUP))
    assert proc.returncode == 0, proc.stderr
    (create,) = node.calls("conda")  # pylint: disable=unbalanced-tuple-unpacking
    args = create[1:]
    assert args[:2] == ["create", "-y"]
    assert args[args.index("-p") + 1] == str(node.work / ".byoc-git")
    assert args[args.index("-c") + 1] == "conda-forge"
    assert "--override-channels" in args
    assert args[-1] == "git"
    installed = str(node.work / ".byoc-git" / "bin" / "git")
    assert node.clone_git() == installed
    (checkout,) = [c for c in node.calls("git") if "checkout" in c]
    assert checkout == [installed, "-C", "code", "checkout", "v1"]
    assert not node.calls("apt-get")


def test_conda_scratch_does_not_stay_in_the_workdir(tmp_path):
    """The package cache is ~150 MB; only the git prefix is kept."""
    node = _Node(tmp_path, tools=("conda",), uid=1000)
    proc = node.run(_render("setup", **_SETUP))
    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in node.work.iterdir()) == [".byoc-git", "code"]


def test_a_broken_git_left_by_an_earlier_attempt_is_reinstalled(tmp_path):
    """A relaunch reuses the per-run workdir; an install cut off mid-way can
    leave bin/git without the libraries it loads."""
    node = _Node(tmp_path, tools=("conda",), uid=1000)
    leftover = node.work / ".byoc-git" / "bin" / "git"
    leftover.parent.mkdir(parents=True)
    node.write(
        leftover,
        "#!/bin/bash\n"
        + _LOG
        + "echo 'git: error while loading shared libraries: libpcre2-8.so.0' >&2\n"
        + "exit 127\n",
    )
    proc = node.run(_render("setup", **_SETUP))
    assert proc.returncode == 0, proc.stderr
    assert len(node.calls("conda")) == 1
    assert node.clone_git() == str(leftover)
    assert leftover.read_text() == node.git_stub.read_text()


@pytest.mark.parametrize("manager", ["apt-get", "dnf", "microdnf"])
def test_root_without_conda_uses_the_system_package_manager(tmp_path, manager):
    """fedora-minimal (the default image) has dnf/microdnf; Debian images apt-get."""
    node = _Node(tmp_path, tools=(manager,), uid=0)
    proc = node.run(_render("setup", **_SETUP))
    assert proc.returncode == 0, proc.stderr
    installs = [c for c in node.calls(manager) if "install" in c]
    assert len(installs) == 1 and "git" in installs[0]
    assert node.clone_git() == str(node.bin / "git")


def _refused(proc: subprocess.CompletedProcess) -> bool:
    """The step stopped with its own message, not a bare `git: command not found`."""
    return (
        proc.returncode not in (0, 127)
        and "command not found" not in proc.stderr
        and any(line.startswith("byoc:") for line in proc.stderr.splitlines())
    )


def test_non_root_without_conda_fails_before_cloning(tmp_path):
    """apt-get exists but cannot run without root, so it is not tried."""
    node = _Node(tmp_path, tools=("apt-get",), uid=1000)
    proc = node.run(_render("setup", **_SETUP))
    assert _refused(proc), proc.stderr
    assert not node.calls("apt-get")
    assert not (node.work / "code").exists()


def test_root_without_any_package_manager_fails_before_cloning(tmp_path):
    node = _Node(tmp_path, tools=(), uid=0)
    proc = node.run(_render("setup", **_SETUP))
    assert _refused(proc), proc.stderr
    assert not (node.work / "code").exists()


def test_conda_retries_with_the_classic_solver(tmp_path):
    """The pytorch/pytorch:2.6.0 image's conda 25.1.0 (libmamba 2.0.5) reports
    every conda-forge package as not available; `--solver classic` installs git."""
    node = _Node(tmp_path, tools=("conda",), uid=1000)
    node.write(
        node.bin / "conda",
        _CONDA.replace(
            'prefix=""\n',
            'case " $* " in *" --solver classic "*) ;; *) exit 1 ;; esac\nprefix=""\n',
        ),
    )
    proc = node.run(_render("setup", **_SETUP))
    assert proc.returncode == 0, proc.stderr
    first, retry = node.calls("conda")  # pylint: disable=unbalanced-tuple-unpacking
    assert "--solver" not in first
    assert retry[retry.index("--solver") + 1] == "classic"
    assert node.clone_git() == str(node.work / ".byoc-git" / "bin" / "git")


def test_a_failed_conda_install_fails_setup(tmp_path):
    node = _Node(tmp_path, tools=("conda",), uid=1000)
    node.write(node.bin / "conda", "#!/bin/bash\n" + _LOG + "exit 3\n")
    proc = node.run(_render("setup", **_SETUP))
    assert proc.returncode == 3, proc.stderr
    assert len(node.calls("conda")) == 2
    assert not (node.work / "code").exists()


def test_the_installed_prefix_does_not_shadow_the_images_tools(tmp_path):
    """conda's git brings its own perl, openssl, curl...; the command must still
    get the image's."""
    node = _Node(tmp_path, tools=(), uid=1000)
    (node.work / ".byoc-git" / "bin").mkdir(parents=True)
    node.write(node.work / ".byoc-git" / "bin" / "git", _GIT)
    node.write(node.work / ".byoc-git" / "bin" / "perl", "#!/bin/bash\necho prefix\n")
    node.write(node.bin / "perl", "#!/bin/bash\necho image\n")
    proc = node.run(_render("run", **{**_SETUP, "command": "perl"}))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines()[-1] == "image"


def test_run_records_the_commit_with_the_installed_git(tmp_path):
    """run is a separate shell: it must find setup's git to record lineage."""
    node = _Node(tmp_path, tools=(), uid=1000)
    (node.work / ".byoc-git" / "bin").mkdir(parents=True)
    node.write(node.work / ".byoc-git" / "bin" / "git", _GIT)
    proc = node.run(_render("run", **_SETUP))
    assert proc.returncode == 0, proc.stderr
    assert (
        f"GB_STEP_METADATA_KEY:commit_hash GB_STEP_METADATA_VALUE:{_SHA}"
        in proc.stdout.splitlines()
    )
