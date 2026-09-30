"""Tests for SkyPilot HPC (slurm/lsf) control-plane SSH resilience.

Covers the defenses added after a bluevela launch failed with
``ValueError: Failed to get partitions for cluster bluevela`` whose real cause was
``Connection timed out during banner exchange``:

1. the retry classifier (``_is_transient_provision_error``) treats an SSH
   banner/session timeout as transient (and still treats an SSH *auth* rejection as
   fatal),
2. the narrower login-node failover classifier (``_is_transient_ssh_error``) fires
   on those same SSH control-plane blips but NOT on capacity errors — so a wedged
   login node fails over to a candidate while a full cluster does not,
3. a failure traceback is logged as ONE record so line-per-record log ingestion
   cannot shred it.
"""

from unittest.mock import patch

import pytest

from gbserver.environment.skypilot import (
    _is_transient_provision_error,
    _is_transient_ssh_error,
    _log_remote_stacktrace,
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
# 2. Login-node failover classifier (_is_transient_ssh_error)
# ---------------------------------------------------------------------------
class TestIsTransientSshError:
    """`_is_transient_ssh_error` — the narrower classifier that gates login-node
    failover. It must match SSH control-plane blips (so a wedged login node rotates
    to a candidate) but NOT capacity errors (a full cluster is not the node's fault)
    or auth rejections (a bad key never succeeds on retry)."""

    @pytest.mark.parametrize(
        "msg",
        [
            # Unambiguously-SSH wording: a reason to fail over on any cloud.
            "Connection timed out during banner exchange",
            "Failed to get partitions for cluster bluevela",
            "Failed to get Slurm partitions.",
            "Failed to query Slurm jobs.",
            "kex_exchange_identification: Connection closed by remote host",
            "ssh_exchange_identification: read: Connection reset by peer",
        ],
    )
    def test_ssh_control_plane_blip_triggers_failover(self, msg):
        assert _is_transient_ssh_error(ValueError(msg)) is True
        # Cloud-independent for the unambiguous SSH substrings.
        assert _is_transient_ssh_error(ValueError(msg), cloud="slurm") is True
        assert _is_transient_ssh_error(ValueError(msg), cloud="k8s") is True

    @pytest.mark.parametrize("msg", _GENERIC_NETWORK_MSGS)
    @pytest.mark.parametrize("cloud", ["slurm", "lsf", "slurm/bluevela", "LSF"])
    def test_generic_network_error_fails_over_on_hpc(self, msg, cloud):
        """Generic TCP/DNS wording is an SSH blip only on the HPC path — fail over."""
        assert _is_transient_ssh_error(ValueError(msg), cloud=cloud) is True

    @pytest.mark.parametrize("msg", _GENERIC_NETWORK_MSGS)
    @pytest.mark.parametrize("cloud", ["k8s", "gcp", "aws", None])
    def test_generic_network_error_no_failover_off_hpc(self, msg, cloud):
        assert _is_transient_ssh_error(ValueError(msg), cloud=cloud) is False

    @pytest.mark.parametrize(
        "msg",
        [
            # Capacity/resource failures ARE retried by the provision classifier,
            # but every login node hits them alike, so they must NOT rotate.
            "Failed to acquire resources in normal for cluster bluevela",
            "Failed to provision all possible launchable resources",
            "Resources unavailable",
        ],
    )
    @pytest.mark.parametrize("cloud", ["slurm", "lsf", None])
    def test_capacity_error_does_not_fail_over(self, msg, cloud):
        assert _is_transient_ssh_error(ValueError(msg), cloud=cloud) is False
        # Sanity: the provision classifier still treats capacity as retriable.
        assert _is_transient_provision_error(ValueError(msg), cloud="slurm") is True

    @pytest.mark.parametrize(
        "msg",
        [
            "ubuntu@bluevela: Permission denied (publickey,password).",
            "Host key verification failed.",
            "Too many authentication failures",
            "no such identity: /keys/id_rsa: No such file or directory",
        ],
    )
    def test_auth_rejection_does_not_fail_over(self, msg):
        """A bad key never succeeds on another node — rotating would waste attempts."""
        assert _is_transient_ssh_error(ValueError(msg), cloud="slurm") is False

    def test_auth_rejection_wins_over_ssh_substring(self):
        """Both an SSH blip and an auth rejection => no failover (auth wins)."""
        msg = (
            "Connection timed out during banner exchange\n"
            "Permission denied (publickey)."
        )
        assert _is_transient_ssh_error(ValueError(msg), cloud="slurm") is False
