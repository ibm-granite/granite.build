"""Tests for SkyPilot HPC (slurm/lsf) control-plane SSH resilience.

Covers the three defenses added after a bluevela launch failed with
``ValueError: Failed to get partitions for cluster bluevela`` whose real cause was
``Connection timed out during banner exchange``:

1. the retry classifier treats an SSH banner/session timeout as transient (and
   still treats an SSH *auth* rejection as fatal),
2. the pre-launch ``echo`` reachability probe is bounded and never fatal,
3. a failure traceback is logged as ONE record so line-per-record log ingestion
   cannot shred it.
"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gbserver.environment.skypilot import (
    Skypilot,
    _is_transient_provision_error,
    _log_remote_stacktrace,
)
from gbserver.types.environmentconfig import EnvironmentConfig

# The verbatim failure from the production runner log (build
# 00cb68b4-1f58-4802-b802-adcf681e254a), ANSI colour codes included, since the
# classifier sees the raw string.
PROD_BANNER_FAILURE = (
    "Failed to get partitions for cluster bluevela: sky.exceptions.CommandError: "
    "Command scontrol show partitions -o failed with return code 255.\n"
    "\x1b[31mFailed to get Slurm partitions.\x1b[0m\n\n"
    "Connection timed out during banner exchange\n"
)


@pytest.fixture
def slurm_env():
    config = EnvironmentConfig(
        name="test-slurm",
        type="Skypilot",
        config={"default_cloud": "slurm"},
    )
    return Skypilot(event_q=asyncio.Queue(), environment_config=config)


# ---------------------------------------------------------------------------
# 1. Retry classification
# ---------------------------------------------------------------------------


def test_production_banner_timeout_is_transient():
    """The exact production failure must retry, not fail the build outright."""
    assert _is_transient_provision_error(ValueError(PROD_BANNER_FAILURE)) is True


@pytest.mark.parametrize(
    "msg",
    [
        # Unambiguously-SSH wording: retried on every cloud, since no other path
        # produces it.
        "Connection timed out during banner exchange",
        "Failed to get Slurm partitions.",
        "Failed to query Slurm jobs.",
        "kex_exchange_identification: Connection closed by remote host",
        "ssh_exchange_identification: read: Connection reset by peer",
        "Connection closed by remote host",
    ],
)
def test_ssh_flakiness_is_transient(msg):
    assert _is_transient_provision_error(ValueError(msg)) is True
    # Cloud-independent: still transient when the cloud is known, either way.
    assert _is_transient_provision_error(ValueError(msg), cloud="slurm") is True
    assert _is_transient_provision_error(ValueError(msg), cloud="k8s") is True


# Generic TCP/DNS wording. The identical text comes out of k8s/gcp/aws paths
# (registry blips, image pull, cloud-API hiccups), where a persistent misconfig
# would otherwise be retried with a full teardown between attempts.
_GENERIC_NETWORK_MSGS = [
    "Connection timed out",
    "Connection reset by peer",
    "No route to host",
    "Temporary failure in name resolution",
]


@pytest.mark.parametrize("msg", _GENERIC_NETWORK_MSGS)
@pytest.mark.parametrize("cloud", ["slurm", "lsf", "slurm/bluevela", "LSF"])
def test_generic_network_error_is_transient_on_hpc(msg, cloud):
    """On the HPC SSH path these mean the control-plane SSH blipped — retry.

    Accepts a bare cloud or a full infra string, and is case-insensitive.
    """
    assert _is_transient_provision_error(ValueError(msg), cloud=cloud) is True


@pytest.mark.parametrize("msg", _GENERIC_NETWORK_MSGS)
@pytest.mark.parametrize("cloud", ["k8s", "gcp", "aws", "kubernetes"])
def test_generic_network_error_not_transient_off_hpc(msg, cloud):
    """Off the HPC path the same text may be a persistent misconfig — don't retry."""
    assert _is_transient_provision_error(ValueError(msg), cloud=cloud) is False


@pytest.mark.parametrize("msg", _GENERIC_NETWORK_MSGS)
def test_generic_network_error_not_transient_without_cloud(msg):
    """With no cloud supplied, stay conservative and do not retry."""
    assert _is_transient_provision_error(ValueError(msg)) is False


def test_hpc_cloud_does_not_rescue_auth_rejection():
    """Cloud scoping must not override the non-transient (auth) tuple."""
    msg = "Permission denied (publickey). Connection timed out"
    assert _is_transient_provision_error(ValueError(msg), cloud="slurm") is False


@pytest.mark.parametrize(
    "msg",
    [
        # Auth rejections carry the same exit-255 vocabulary as the transient SSH
        # blips but will never succeed on retry, so they must stay fatal.
        "Failed to get partitions for cluster bluevela: CommandError: Command "
        "scontrol show partitions -o failed with return code 255.\n"
        "ubuntu@bluevela: Permission denied (publickey,password).",
        "Host key verification failed.",
        "Too many authentication failures",
        "Permission denied, please try again.",
        "no such identity: /keys/id_rsa: No such file or directory",
        # Pre-existing permanent config errors must not regress.
        "Catalog does not contain any instances",
        "No launchable resource found",
    ],
)
def test_auth_and_config_failures_stay_fatal(msg):
    assert _is_transient_provision_error(ValueError(msg)) is False


@pytest.mark.parametrize(
    "msg",
    [
        # Generic cloud-provisioning timeouts, not SSH: SkyPilot uses this wording
        # across azure/gcp/k8s (e.g. k8s "Timed out waiting for apt update"), so
        # matching it would retry unrelated non-HPC failures.
        "Timed out waiting for apt update",
        "Operation timed out while creating disk",
    ],
)
def test_generic_cloud_timeouts_are_not_retried(msg):
    assert _is_transient_provision_error(ValueError(msg)) is False


def test_auth_rejection_wins_over_transient_substring():
    """Both an auth rejection and a timeout => fatal (non-transient wins)."""
    msg = (
        "Connection timed out during banner exchange\n" "Permission denied (publickey)."
    )
    assert _is_transient_provision_error(ValueError(msg)) is False


# ---------------------------------------------------------------------------
# 2. Pre-launch echo probe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_skipped_for_non_hpc_cloud(slurm_env):
    """k8s/aws have no shared SSH config; the probe must not spawn anything."""
    with patch("asyncio.create_subprocess_exec") as spawn:
        await slurm_env._probe_hpc_login_node("k8s", "some-cluster")
    spawn.assert_not_called()


@pytest.mark.asyncio
async def test_probe_disabled_by_zero_timeout(slurm_env, tmp_path):
    cfg = tmp_path / ".slurm"
    cfg.mkdir()
    (cfg / "config").write_text("Host bluevela\n")
    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("gbserver.types.constants.GBSERVER_SKYPILOT_SSH_PROBE_TIMEOUT_S", 0),
        patch("asyncio.create_subprocess_exec") as spawn,
    ):
        await slurm_env._probe_hpc_login_node("slurm", "bluevela")
    spawn.assert_not_called()


@pytest.mark.asyncio
async def test_probe_runs_echo_against_skypilot_ssh_config(slurm_env, tmp_path):
    """A plain `echo` through ~/.slurm/config — never a scheduler command."""
    cfg_dir = tmp_path / ".slurm"
    cfg_dir.mkdir()
    (cfg_dir / "config").write_text("Host bluevela\n    User me\n")

    proc = MagicMock()
    proc.communicate = _async_return((b"gbserver probe\n", b""))
    proc.returncode = 0

    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("asyncio.create_subprocess_exec", side_effect=_spawns(proc)) as spawn,
    ):
        await slurm_env._probe_hpc_login_node("slurm", "bluevela")

    args = spawn.call_args[0]
    assert args[0] == "ssh"
    assert "echo" in args
    assert "bluevela" in args
    assert "-F" in args
    assert str(cfg_dir / "config") in args
    # BatchMode prevents the probe itself from blocking on a password prompt.
    assert "BatchMode=yes" in args
    joined = " ".join(args)
    for scheduler_cmd in ("scontrol", "squeue", "sbatch", "bhosts", "bsub"):
        assert scheduler_cmd not in joined


@pytest.mark.asyncio
async def test_probe_timeout_kills_child_and_does_not_raise(slurm_env, tmp_path):
    """A hung probe is killed and reaped: wait_for alone leaks the ssh child."""
    cfg_dir = tmp_path / ".slurm"
    cfg_dir.mkdir()
    (cfg_dir / "config").write_text("Host bluevela\n")

    proc = MagicMock()

    async def _hang():
        await asyncio.sleep(3600)

    proc.communicate = _hang
    proc.wait = _async_return(0)

    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("gbserver.types.constants.GBSERVER_SKYPILOT_SSH_PROBE_TIMEOUT_S", 1),
        patch("asyncio.create_subprocess_exec", side_effect=_spawns(proc)),
    ):
        # Must return (not raise) so a probe quirk never blocks a good launch.
        await asyncio.wait_for(
            slurm_env._probe_hpc_login_node("slurm", "bluevela"), timeout=30
        )

    proc.kill.assert_called_once()


@pytest.mark.asyncio
async def test_probe_failure_is_not_fatal(slurm_env, tmp_path):
    """A non-zero probe still proceeds to launch; the classifier is the backstop."""
    cfg_dir = tmp_path / ".slurm"
    cfg_dir.mkdir()
    (cfg_dir / "config").write_text("Host bluevela\n")

    proc = MagicMock()
    proc.communicate = _async_return((b"", b"ssh: connect to host ... timed out"))
    proc.returncode = 255

    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("asyncio.create_subprocess_exec", side_effect=_spawns(proc)),
    ):
        await slurm_env._probe_hpc_login_node("slurm", "bluevela")


@pytest.mark.asyncio
async def test_probe_noop_without_ssh_config(slurm_env, tmp_path):
    """No materialized config (env defines none) => nothing to probe through."""
    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("asyncio.create_subprocess_exec") as spawn,
    ):
        await slurm_env._probe_hpc_login_node("slurm", "bluevela")
    spawn.assert_not_called()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _async_return(value):
    async def _inner(*_a, **_kw):
        return value

    return _inner


def _spawns(proc):
    """Async side_effect for create_subprocess_exec (a bare coroutine is
    awaitable only once)."""

    async def _inner(*_a, **_kw):
        return proc

    return _inner


class TestRemoteStacktraceLogging:
    """``_log_remote_stacktrace`` surfaces the SkyPilot API server's traceback.

    Regression guard for the bluevela/SLURM failure that reported only
    ``OSError: [Errno 30] Read-only file system`` with no frames and no path:
    the exception crossed the API-server boundary, so ``exc_info=True`` had no
    ``__traceback__`` to render and the errno-only OSError carried no filename.
    """

    def test_logs_server_traceback_when_present(self):
        exc = OSError(30, "Read-only file system")
        setattr(exc, "stacktrace", 'File "/sky/backend.py", line 9\nOSError: ...')
        with patch("gbserver.environment.skypilot.logger") as mock_logger:
            _log_remote_stacktrace(exc, "provision test-cluster")
        assert mock_logger.error.called
        logged = " ".join(str(a) for a in mock_logger.error.call_args[0])
        assert "/sky/backend.py" in logged
        assert "provision test-cluster" in logged

    def test_silent_when_no_server_traceback(self):
        # Locally-raised exceptions have a real traceback already; adding an
        # empty "server traceback" line would just be noise.
        with patch("gbserver.environment.skypilot.logger") as mock_logger:
            _log_remote_stacktrace(OSError(30, "Read-only file system"), "ctx")
        mock_logger.error.assert_not_called()
