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
``id -u`` reports. ``HOME`` is ``/`` unless a test says otherwise, as inside an
enroot container, where it is not writable.
"""

import pathlib
import shutil
import subprocess

import pytest
import yaml

from gbserver.utils.template import fill_template

_REPO = pathlib.Path(__file__).resolve().parents[3]
_TEMPLATE = yaml.safe_load(
    (_REPO / "steps" / "aux" / "byoc" / "skypilot" / "step-template.yaml").read_text()
)
_LAUNCHER = _TEMPLATE["environment_configs"]["Skypilot"]["launchers"]["byoc"]["config"]
_DEFAULTS = _TEMPLATE["config"]["byoc_config"]
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

# A git whose binary is there but cannot start, as when a shared library is
# missing.
_BROKEN_GIT = (
    "#!/bin/bash\n"
    + _LOG
    + "echo 'git: error while loading shared libraries: libpcre2-8.so.0' >&2\n"
    + "exit 127\n"
)

# conda writes its cache under ${XDG_CACHE_HOME:-~/.cache} and, when the image's
# package cache is not writable, its packages under ${CONDA_PKGS_DIRS:-~/.conda/pkgs}.
# With HOME=/ a non-root conda fails with "Permission denied: '/.cache'" (seen in
# condaforge/miniforge3 run as uid 1000), so the stub fails the same way. Then it
# fills the requested prefix: git, plus the perl that conda-forge's git depends on.
_CONDA = (
    "#!/bin/bash\n"
    + _LOG
    + 'cache="${XDG_CACHE_HOME:-$HOME/.cache}"; pkgs="${CONDA_PKGS_DIRS:-$HOME/.conda/pkgs}"\n'
    + 'for d in "$cache" "$pkgs"; do\n'
    + '  case "$d" in //*) echo "Permission denied: \'$d\'" >&2; exit 1 ;; esac\n'
    + "done\n"
    + 'if [ -n "${NEED_CONDARC:-}" ] && [ ! -f "$HOME/.condarc" ]; then\n'
    + '  echo "CondaHTTPError: no proxy configured" >&2; exit 1\n'
    + "fi\n"
    + 'mkdir -p "$cache/conda" "$pkgs/git-2.56.0"\n'
    + 'prefix=""\n'
    + 'while [ $# -gt 0 ]; do [ "$1" = -p ] && prefix="$2"; shift; done\n'
    + 'mkdir -p "$prefix/bin" && cp "$GIT_STUB" "$prefix/bin/git"\n'
    + "printf '#!/bin/bash\\necho conda-perl\\n' > \"$prefix/bin/perl\"\n"
    + 'chmod +x "$prefix/bin/perl"\n'
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
_REAL = ("rm", "mkdir", "cp", "ln", "chmod")


def _render(block: str, **byoc_config) -> str:
    """Render the launcher's ``setup`` or ``run`` with gbserver's strict renderer."""
    data = {"config": {"byoc_config": {**_DEFAULTS, **byoc_config}}}
    return fill_template(_LAUNCHER[block], data, strict=True)


class _Node:
    """A workdir plus a PATH holding only the given tools."""

    def __init__(self, tmp_path: pathlib.Path, tools=(), uid: int = 1000):
        """Lay out the workdir, the PATH dir and the call trace under *tmp_path*.

        :param tools: tool stubs to put on PATH: ``git`` and ``conda`` get their
            own stubs, any other name (``apt-get``, ``dnf``, ``microdnf``) a
            package manager that installs git. A tool not listed is absent.
        :param uid: what ``id -u`` reports; 0 makes the container root.
        """
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
        """Write an executable script."""
        path.write_text(text)
        path.chmod(0o755)

    def run(self, script: str, **env_extra: str) -> subprocess.CompletedProcess:
        """Run *script* in bash from the workdir with only this node's PATH."""
        env = {
            "PATH": str(self.bin),
            "HOME": "/",
            "TRACE": str(self.trace),
            "GIT_STUB": str(self.git_stub),
            "BIN": str(self.bin),
            "FAKE_UID": str(self.uid),
            **env_extra,
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

    @property
    def installed_git(self) -> str:
        """Where setup puts the git it installs."""
        return str(self.work / ".byoc-git" / "bin" / "git")


_SETUP = {"repo": "https://example.com/org/repo.git", "command": "true"}


def _refused(proc: subprocess.CompletedProcess) -> bool:
    """The step stopped with its own message, not a bare `git: command not found`."""
    return (
        proc.returncode not in (0, 127)
        and "command not found" not in proc.stderr
        and any(line.startswith("byoc:") for line in proc.stderr.splitlines())
    )


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
    assert args[args.index("-p") + 1].startswith(str(node.work / ".byoc-git"))
    assert args[args.index("-c") + 1] == "conda-forge"
    assert "--override-channels" in args
    assert args[-1] == "git"
    assert node.clone_git() == node.installed_git
    (checkout,) = [c for c in node.calls("git") if "checkout" in c]
    assert checkout == [node.installed_git, "-C", "code", "checkout", "v1"]
    assert not node.calls("apt-get")


def test_conda_keeps_the_users_home(tmp_path):
    """Proxy, channel_alias and mirror settings live in ~/.condarc (and ~/.netrc);
    a conda that does not see the user's HOME cannot reach conda-forge there."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".condarc").write_text("proxy_servers: {https: http://proxy:3128}\n")
    node = _Node(tmp_path, tools=("conda",), uid=1000)
    proc = node.run(_render("setup", **_SETUP), HOME=str(home), NEED_CONDARC="1")
    assert proc.returncode == 0, proc.stderr
    assert node.clone_git() == node.installed_git


def test_conda_scratch_does_not_stay_in_the_workdir(tmp_path):
    """The package cache is ~150 MB; only the git prefix is kept."""
    node = _Node(tmp_path, tools=("conda",), uid=1000)
    proc = node.run(_render("setup", **_SETUP))
    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in node.work.iterdir()) == [".byoc-git", "code"]


def test_a_failed_conda_install_leaves_nothing_and_fails_setup(tmp_path):
    """Non-root, so there is no other installer; nothing may pile up across
    relaunches on a shared workdir."""
    node = _Node(tmp_path, tools=("conda",), uid=1000)
    node.write(
        node.bin / "conda",
        "#!/bin/bash\n"
        + _LOG
        + 'mkdir -p "$CONDA_PKGS_DIRS/half-downloaded"\n'
        + "exit 3\n",
    )
    proc = node.run(_render("setup", **_SETUP))
    assert _refused(proc), proc.stderr
    assert len(node.calls("conda")) == 1
    assert not list(node.work.iterdir())


def test_root_falls_back_to_the_package_manager_when_conda_fails(tmp_path):
    """e.g. egress allows the distro mirrors but not conda.anaconda.org."""
    node = _Node(tmp_path, tools=("conda", "apt-get"), uid=0)
    node.write(node.bin / "conda", "#!/bin/bash\n" + _LOG + "exit 1\n")
    proc = node.run(_render("setup", **_SETUP))
    assert proc.returncode == 0, proc.stderr
    assert [c for c in node.calls("apt-get") if "install" in c]
    assert node.clone_git() == str(node.bin / "git")
    assert not (node.work / ".byoc-git").exists()


def test_installed_git_replaces_a_broken_image_git(tmp_path):
    """The image's git exists but cannot start; the clone must use the new one."""
    node = _Node(tmp_path, tools=("conda",), uid=1000)
    node.write(node.bin / "git", _BROKEN_GIT)
    proc = node.run(_render("setup", **_SETUP))
    assert proc.returncode == 0, proc.stderr
    assert node.clone_git() == node.installed_git


def test_an_install_that_leaves_git_broken_fails_with_a_byoc_message(tmp_path):
    """apt-get reports git as installed already, but the image's git still cannot
    start."""
    node = _Node(tmp_path, tools=("apt-get",), uid=0)
    node.write(node.bin / "apt-get", "#!/bin/bash\n" + _LOG + "exit 0\n")
    node.write(node.bin / "git", _BROKEN_GIT)
    proc = node.run(_render("setup", **_SETUP))
    assert _refused(proc), proc.stderr
    assert not [c for c in node.calls("git") if c[1] == "clone"]


def test_a_broken_git_left_by_an_earlier_attempt_is_reinstalled(tmp_path):
    """A relaunch reuses the per-run workdir; an install cut off mid-way can
    leave bin/git without the libraries it loads."""
    node = _Node(tmp_path, tools=("conda",), uid=1000)
    leftover = node.work / ".byoc-git" / "bin" / "git"
    leftover.parent.mkdir(parents=True)
    node.write(leftover, _BROKEN_GIT)
    proc = node.run(_render("setup", **_SETUP))
    assert proc.returncode == 0, proc.stderr
    assert len(node.calls("conda")) == 1
    assert node.clone_git() == node.installed_git
    assert leftover.read_text() == node.git_stub.read_text()


def test_a_broken_leftover_git_does_not_hide_a_package_manager_git(tmp_path):
    """The leftover .byoc-git/bin comes first on PATH; when the package manager
    installs git instead of conda, the leftover must not shadow it."""
    node = _Node(tmp_path, tools=("apt-get",), uid=0)
    leftover = node.work / ".byoc-git" / "bin" / "git"
    leftover.parent.mkdir(parents=True)
    node.write(leftover, _BROKEN_GIT)
    proc = node.run(_render("setup", **_SETUP))
    assert proc.returncode == 0, proc.stderr
    assert node.clone_git() == str(node.bin / "git")
    assert not (node.work / ".byoc-git").exists()


@pytest.mark.parametrize("manager", ["apt-get", "dnf", "microdnf"])
def test_root_without_conda_uses_the_system_package_manager(tmp_path, manager):
    """fedora-minimal (the default image) has dnf/microdnf; Debian images apt-get."""
    node = _Node(tmp_path, tools=(manager,), uid=0)
    proc = node.run(_render("setup", **_SETUP))
    assert proc.returncode == 0, proc.stderr
    installs = [c for c in node.calls(manager) if "install" in c]
    assert len(installs) == 1 and "git" in installs[0]
    assert node.clone_git() == str(node.bin / "git")


def test_a_failed_package_manager_fails_with_a_byoc_message(tmp_path):
    """Root without real privileges (enroot --container-remap-root) cannot run
    apt-get; the step must say so instead of failing at the clone."""
    node = _Node(tmp_path, tools=("apt-get",), uid=0)
    node.write(
        node.bin / "apt-get",
        "#!/bin/bash\n"
        + _LOG
        + "echo 'E: setgroups 65534 failed - setgroups (1: Operation not permitted)' >&2\n"
        + "exit 100\n",
    )
    proc = node.run(_render("setup", **_SETUP))
    assert _refused(proc), proc.stderr
    assert not (node.work / "code").exists()


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


def test_run_records_the_commit_with_the_git_setup_installed(tmp_path):
    """run is a separate shell: it must find the git setup installed, at the path
    setup put it, to record lineage. A broken image git must not win either."""
    node = _Node(tmp_path, tools=("conda",), uid=1000)
    node.write(node.bin / "git", _BROKEN_GIT)
    setup = node.run(_render("setup", **_SETUP))
    assert setup.returncode == 0, setup.stderr
    proc = node.run(_render("run", **_SETUP))
    assert proc.returncode == 0, proc.stderr
    assert (
        f"GB_STEP_METADATA_KEY:commit_hash GB_STEP_METADATA_VALUE:{_SHA}"
        in proc.stdout.splitlines()
    )


def test_the_installed_git_does_not_shadow_the_images_tools(tmp_path):
    """conda's git brings its own perl, openssl, curl...; setup_command and the
    command must still get the image's."""
    node = _Node(tmp_path, tools=("conda",), uid=1000)
    node.write(node.bin / "perl", "#!/bin/bash\necho image-perl\n")
    setup = node.run(_render("setup", **{**_SETUP, "setup_command": "perl"}))
    assert setup.returncode == 0, setup.stderr
    assert setup.stdout.splitlines()[-1] == "image-perl"
    proc = node.run(_render("run", **{**_SETUP, "command": "perl"}))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.splitlines()[-1] == "image-perl"
