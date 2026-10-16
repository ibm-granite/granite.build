"""Unit tests for the sage-eval steps' scratch TMPDIR.

The sage-eval steps point TMPDIR at a per-task scratch directory. Programs bind
unix sockets under tempfile.gettempdir(), i.e. that directory: vLLM's V1 engine
its ZMQ ipc sockets, and Python < 3.13 multiprocessing its pymp-* listeners. A
unix socket path may be at most 107 characters (sizeof(sockaddr_un.sun_path) - 1).

When the directory was named after SKYPILOT_TASK_ID, which embeds the cluster
name, the socket path ran to about 150 characters, and on BlueVela the dense
granite-4.1-3b model failed SALAD-Bench at engine start:

    zmq.error.ZMQError: ipc path "/opt/nvme/granitebuild/tmp-sky-2026-10-11-03-40-54-
    185155_gb-distill-350m-model-s-eval-salad-bench-m1-72e9eaa5-641_1/420bb520-...
    " is longer than 107 characters

These tests run each step's TMPDIR setup in bash, both the NVMe branch and the
/tmp fallback, under a sandbox root that stands in for /, and check that the
directory is new, private and short enough for the longest socket path.
"""

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
ENVIRONMENTS = REPO_ROOT / "configurations/assets/environments"
STEPS = [
    ENVIRONMENTS / "skypilot/lsf/ibm-bluevela/steps/sage-eval/step.yaml",
    ENVIRONMENTS / "skypilot/aws/steps/sage-eval/step.yaml",
    ENVIRONMENTS / "skypilot/aws/steps/sage-eval-code-grouped/step.yaml",
]
# Other steps that set TMPDIR, and why they are not tested here.
OTHER_TMPDIR_STEPS = {
    # Sets TMPDIR to its own VLLM_RPC_BASE_PATH, /tmp/vllm-<user>-<LSF job id>.
    ENVIRONMENTS
    / "skypilot/steps/distill/vllm-server/step.yaml": "short by construction",
}
# Where steps live: environment assets, the builtin steps, and the samples.
STEP_ROOTS = [ENVIRONMENTS, REPO_ROOT / "src/gbserver/builtins", REPO_ROOT / "samples"]
SETS_TMPDIR = re.compile(r"^\s*(export\s+)?TMPDIR=", re.M)

# The task id of the failing BlueVela run above.
TASK_ID = (
    "sky-2026-10-11-03-40-54-185155_"
    "gb-distill-350m-model-s-eval-salad-bench-m1-72e9eaa5-641_1"
)
# The longest login name useradd accepts.
LONGEST_USER = "u" * 32
SUN_PATH_MAX = 107
# The longest name bound under TMPDIR: a vLLM ipc socket is a uuid4, and a
# multiprocessing listener is pymp-XXXXXXXX/listener-XXXXXXXX (31 characters).
LONGEST_SOCKET_NAME = "420bb520-b3a8-41b9-b2f3-cf3dd3b25e7c"
NVME_TEST = '[ -d "/opt/nvme" ]'


def _scripts(node):
    """Yield every string in a parsed step.yaml that sets TMPDIR.

    Args:
        node: A parsed YAML node (dict, list or scalar).

    Yields:
        Each string value, at any depth, with a line assigning TMPDIR.
    """
    if isinstance(node, dict):
        for value in node.values():
            yield from _scripts(value)
    elif isinstance(node, list):
        for value in node:
            yield from _scripts(value)
    elif isinstance(node, str) and SETS_TMPDIR.search(node):
        yield node


def _run_script(path):
    """Return the one run script in ``path`` that sets TMPDIR.

    Args:
        path: A step.yaml file.

    Returns:
        The script text.
    """
    scripts = list(_scripts(yaml.safe_load(path.read_text(encoding="utf-8"))))
    assert len(scripts) == 1, f"{path}: expected one script setting TMPDIR"
    return scripts[0]


def _tmpdir_setup(script, root):
    """Cut the TMPDIR setup out of a run script and point it at a sandbox.

    The setup runs from the NVMe test to the ``fi`` after the last line that
    assigns TMPDIR. The NVMe test becomes ``[ "$NVME" = 1 ]`` so the caller can
    pick the branch, and /opt/nvme and /tmp move under ``root``.

    Args:
        script: A run script from :func:`_run_script`.
        root: The directory that stands in for /.

    Returns:
        The setup, as a bash snippet.
    """
    lines = script.splitlines()
    start = next(i for i, line in enumerate(lines) if NVME_TEST in line)
    last = max(i for i, line in enumerate(lines) if SETS_TMPDIR.match(line))
    end = next(i for i in range(last, len(lines)) if lines[i].strip() == "fi")
    setup = "\n".join(lines[start : end + 1])
    assert setup.count(NVME_TEST) == 1, f"expected one {NVME_TEST} in:\n{setup}"
    setup = setup.replace(NVME_TEST, '[ "$NVME" = 1 ]')
    # In one pass, so the root (itself under /tmp in pytest) is not rewritten.
    return re.sub(r"/opt/nvme|/tmp/", lambda m: f"{root}{m.group()}", setup)


def _shell_options(script):
    """Return the ``set`` lines at the top of a run script, before any command.

    Args:
        script: A run script from :func:`_run_script`.

    Returns:
        Those lines, so a snippet runs with the step's own options and no others.
    """
    options = []
    for line in script.splitlines():
        if re.match(r"\s*set\s+[-+]", line):
            options.append(line.strip())
        elif line.strip() and not line.strip().startswith("#"):
            break
    return "\n".join(options)


def _run_setup(path, nvme, root, after):
    """Run a step's TMPDIR setup in bash, then ``after``.

    The setup runs with the step's own shell options (some steps have no
    ``set -e``), under ``root`` standing in for /.

    Args:
        path: A step.yaml file.
        nvme: Take the NVMe branch if True, the /tmp fallback if False.
        root: The directory that stands in for /.
        after: Bash to run once the setup is done.

    Returns:
        The finished :class:`subprocess.CompletedProcess`.
    """
    (root / "tmp").mkdir(parents=True, exist_ok=True)
    if nvme:
        (root / "opt/nvme" / LONGEST_USER).mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": os.environ["PATH"],
        "USER": LONGEST_USER,
        "NVME": "1" if nvme else "0",
        "SKYPILOT_TASK_ID": TASK_ID,
    }
    script = _run_script(path)
    setup = _tmpdir_setup(script, root)
    return subprocess.run(
        ["bash", "-c", f"{_shell_options(script)}\n{setup}\n{after}"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _tmpdir(path, nvme, root):
    """Run a step's TMPDIR setup in bash and return the TMPDIR it sets.

    Args:
        path: A step.yaml file.
        nvme: Take the NVMe branch if True, the /tmp fallback if False.
        root: The directory that stands in for /.

    Returns:
        The TMPDIR the setup created, under ``root``.
    """
    result = _run_setup(path, nvme, root, 'printf %s "$TMPDIR"')
    assert result.returncode == 0, f"{path}: {result.stderr}"
    tmpdir = Path(result.stdout)
    assert tmpdir.is_relative_to(root), f"{path}: {tmpdir} is outside the sandbox"
    return tmpdir


def test_every_step_that_sets_tmpdir_is_listed():
    """A new step that sets TMPDIR must be added to STEPS or OTHER_TMPDIR_STEPS."""
    found = {
        path
        for root in STEP_ROOTS
        for path in root.rglob("step.yaml")
        if SETS_TMPDIR.search(path.read_text(encoding="utf-8"))
    }
    assert found == set(STEPS) | set(OTHER_TMPDIR_STEPS)


@pytest.mark.parametrize("nvme", [True, False], ids=["nvme", "tmp"])
@pytest.mark.parametrize(
    "path", STEPS, ids=lambda p: str(p.parent.relative_to(ENVIRONMENTS))
)
def test_a_socket_under_tmpdir_fits_in_sun_path(path, nvme, tmp_path):
    """The longest socket path under TMPDIR fits, in either branch."""
    tmpdir = f"/{_tmpdir(path, nvme, tmp_path).relative_to(tmp_path)}"
    assert len(f"{tmpdir}/{LONGEST_SOCKET_NAME}") <= SUN_PATH_MAX, tmpdir


@pytest.mark.parametrize("nvme", [True, False], ids=["nvme", "tmp"])
@pytest.mark.parametrize(
    "path", STEPS, ids=lambda p: str(p.parent.relative_to(ENVIRONMENTS))
)
def test_each_run_gets_a_new_private_tmpdir(path, nvme, tmp_path):
    """Two runs of one task never share a directory, and neither is readable by
    others, so neither can inherit a directory someone else made."""
    first = _tmpdir(path, nvme, tmp_path)
    second = _tmpdir(path, nvme, tmp_path)
    assert first != second
    for tmpdir in (first, second):
        assert tmpdir.is_dir()
        assert stat.S_IMODE(tmpdir.stat().st_mode) == 0o700
        assert tmpdir.stat().st_uid == os.getuid()


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write a mode-500 directory")
@pytest.mark.parametrize("nvme", [True, False], ids=["nvme", "tmp"])
@pytest.mark.parametrize(
    "path", STEPS, ids=lambda p: str(p.parent.relative_to(ENVIRONMENTS))
)
def test_a_mktemp_failure_stops_the_script(path, nvme, tmp_path):
    """If the scratch directory cannot be made, the script stops there, rather
    than going on with an empty TMPDIR (tempfile would then use the tmpfs)."""
    parent = tmp_path / ("opt/nvme" / Path(LONGEST_USER) if nvme else "tmp")
    parent.mkdir(parents=True)
    parent.chmod(0o500)
    try:
        result = _run_setup(path, nvme, tmp_path, "echo reached")
    finally:
        parent.chmod(0o700)
    assert result.returncode != 0
    assert "reached" not in result.stdout
