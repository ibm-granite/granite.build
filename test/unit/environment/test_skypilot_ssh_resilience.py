"""Tests for SkyPilot HPC (slurm/lsf) control-plane SSH resilience.

Covers the defenses added after a bluevela launch failed with
``ValueError: Failed to get partitions for cluster bluevela`` whose real cause was
``Connection timed out during banner exchange``:

1. the retry classifier treats an SSH banner/session timeout as transient (and
   still treats an SSH *auth* rejection as fatal),
2. the login-node reachability probe selects a reachable ``HostName`` and fails the
   launch fast when none answers (see ``_probe_ssh_hostname`` / ``_ssh_probe_cmd``),
3. a failure traceback is logged as ONE record so line-per-record log ingestion
   cannot shred it.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gbserver.environment.skypilot import (
    _is_transient_provision_error,
    _log_remote_stacktrace,
    _probe_ssh_hostname,
    _ssh_probe_cmd,
)

# The verbatim failure from the production runner log (build
# 00cb68b4-1f58-4802-b802-adcf681e254a), ANSI colour codes included, since the
# classifier sees the raw string.
PROD_BANNER_FAILURE = (
    "Failed to get partitions for cluster bluevela: sky.exceptions.CommandError: "
    "Command scontrol show partitions -o failed with return code 255.\n"
    "\x1b[31mFailed to get Slurm partitions.\x1b[0m\n\n"
    "Connection timed out during banner exchange\n"
)


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
    # Bare form only: the kex_/ssh_exchange_identification variants are
    # unambiguous and stay cloud-independent above.
    "Connection closed by remote host",
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


# ---------------------------------------------------------------------------
# 2. Login-node reachability probe
# ---------------------------------------------------------------------------
class TestSshProbeCmd:
    """`_ssh_probe_cmd` — the argv builder for the reachability probe."""

    def test_argv_shape_and_no_verification_flags(self):
        with patch(
            "gbserver.types.constants.ENABLE_SSH_HOST_KEY_VERIFICATION", False
        ):
            cmds = _ssh_probe_cmd("/tmp/cfg", "bluevela", 12)
        assert cmds[:3] == ["ssh", "-F", "/tmp/cfg"]
        assert "BatchMode=yes" in cmds
        assert "ConnectTimeout=12" in cmds
        assert "StrictHostKeyChecking=no" in cmds
        assert "UserKnownHostsFile=/dev/null" in cmds
        # Destination then `echo <msg>` are always the last three tokens.
        assert cmds[-3:] == ["bluevela", "echo", "gbserver probe"]

    def test_strict_toggle_keeps_verification(self):
        with patch(
            "gbserver.types.constants.ENABLE_SSH_HOST_KEY_VERIFICATION", True
        ):
            cmds = _ssh_probe_cmd("/tmp/cfg", "bluevela", 5)
        assert "StrictHostKeyChecking=no" not in cmds
        assert "UserKnownHostsFile=/dev/null" not in cmds


class TestProbeSshHostname:
    """`_probe_ssh_hostname` — blocking ssh-echo probe of one candidate."""

    def _host(self, **extra):
        return {"Host": "bluevela", "HostName": "login2.ex.com", "User": "gb", **extra}

    def test_returncode_zero_is_reachable(self):
        seen = {}

        def fake_run(cmds, **_kw):
            seen["cmds"] = cmds
            # The rendered config (-F <path>) must carry the chosen HostName.
            cfg_path = cmds[cmds.index("-F") + 1]
            seen["cfg"] = Path(cfg_path).read_text(encoding="utf-8")
            return MagicMock(returncode=0, stderr=b"")

        with patch("gbserver.environment.skypilot.subprocess.run", fake_run):
            assert _probe_ssh_hostname(self._host(), {}) is True
        assert seen["cmds"][-3:] == ["bluevela", "echo", "gbserver probe"]
        assert "HostName login2.ex.com" in seen["cfg"]
        assert "ssh_probe_timeout_s" not in seen["cfg"]  # synthetic key stripped

    def test_nonzero_returncode_is_unreachable(self):
        with patch(
            "gbserver.environment.skypilot.subprocess.run",
            return_value=MagicMock(returncode=255, stderr=b"timed out"),
        ):
            assert _probe_ssh_hostname(self._host(), {}) is False

    def test_timeout_is_unreachable(self):
        with patch(
            "gbserver.environment.skypilot.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="ssh", timeout=5),
        ):
            assert _probe_ssh_hostname(self._host(), {}) is False

    def test_per_host_timeout_used_as_connecttimeout(self):
        captured = {}

        def fake_run(cmds, **kw):
            captured["cmds"] = cmds
            captured["timeout"] = kw.get("timeout")
            return MagicMock(returncode=0, stderr=b"")

        with patch("gbserver.environment.skypilot.subprocess.run", fake_run):
            _probe_ssh_hostname(self._host(ssh_probe_timeout_s=7), {})
        assert "ConnectTimeout=7" in captured["cmds"]
        assert captured["timeout"] == 7 + 5  # subprocess wait = ConnectTimeout + buffer

    def test_nonpositive_deployment_default_falls_back(self):
        # The probe is mandatory. A host that pins no timeout inherits the deployment
        # default; if that is mis-set non-positive it falls back to
        # DEFAULT_SSH_PROBE_TIMEOUT_S (30) and still probes — never disabled.
        captured = {}

        def fake_run(cmds, **_kw):
            captured["cmds"] = cmds
            return MagicMock(returncode=0, stderr=b"")

        with (
            patch("gbserver.types.constants.GBSERVER_SKYPILOT_SSH_PROBE_TIMEOUT_S", 0),
            patch("gbserver.environment.skypilot.subprocess.run", fake_run),
        ):
            assert _probe_ssh_hostname(self._host(), {}) is True
        assert "ConnectTimeout=30" in captured["cmds"]
