import subprocess
from types import SimpleNamespace

import pytest

from gbserver.environment.shared_fs import build_provider
from gbserver.environment.shared_fs.base import resolve_shared_workdir
from gbserver.environment.shared_fs.config import EfsConfig, SharedFilesystemConfig
from gbserver.environment.shared_fs.efs import EfsProvider


def test_efs_config_valid_with_fsid_and_region():
    sf = SharedFilesystemConfig.model_validate(
        {
            "provider": "efs",
            "mount_point": "/mnt/gb-shared",
            "efs": {"file_system_id": "fs-0abc", "region": "us-east-1"},
        }
    )
    assert sf.mount_point == "/mnt/gb-shared"
    assert sf.efs.tls is True
    assert sf.efs.derived_dns_name() == "fs-0abc.efs.us-east-1.amazonaws.com"


def test_efs_config_valid_with_dns_name():
    sf = SharedFilesystemConfig.model_validate(
        {
            "provider": "efs",
            "mount_point": "/mnt/gb-shared",
            "efs": {"dns_name": "fs-0abc.efs.eu-west-1.amazonaws.com"},
        }
    )
    assert sf.efs.derived_dns_name() == "fs-0abc.efs.eu-west-1.amazonaws.com"


def test_provider_must_be_efs():
    with pytest.raises(ValueError):
        SharedFilesystemConfig.model_validate(
            {
                "provider": "s3",
                "mount_point": "/mnt/x",
                "efs": {"file_system_id": "fs-1"},
            }
        )


def test_efs_block_required():
    with pytest.raises(ValueError, match="requires an 'efs' block"):
        SharedFilesystemConfig.model_validate(
            {"provider": "efs", "mount_point": "/mnt/x"}
        )


def test_mount_point_trailing_slash_normalized():
    # A trailing slash would otherwise never match the chmod-walk's mount-root
    # sentinel, so the loop would climb to / and chmod /mnt and / (harmless but
    # sloppy). Normalize it in the validator.
    sf = SharedFilesystemConfig.model_validate(
        {
            "provider": "efs",
            "mount_point": "/mnt/gb-shared/",
            "efs": {"file_system_id": "fs-1", "region": "us-east-1"},
        }
    )
    assert sf.mount_point == "/mnt/gb-shared"


def test_mount_point_must_be_absolute():
    with pytest.raises(ValueError, match="must be absolute"):
        SharedFilesystemConfig.model_validate(
            {
                "provider": "efs",
                "mount_point": "rel/path",
                "efs": {"file_system_id": "fs-1", "region": "us-east-1"},
            }
        )


def test_efs_requires_a_target():
    with pytest.raises(ValueError, match="file_system_id or dns_name"):
        SharedFilesystemConfig.model_validate(
            {"provider": "efs", "mount_point": "/mnt/x", "efs": {}}
        )


def test_efs_fsid_requires_region_for_nfs_fallback():
    with pytest.raises(ValueError, match="'region' is required"):
        SharedFilesystemConfig.model_validate(
            {
                "provider": "efs",
                "mount_point": "/mnt/x",
                "efs": {"file_system_id": "fs-1"},
            }
        )


def test_efs_cleanup_zone_must_be_in_region():
    with pytest.raises(ValueError, match="cleanup_zone"):
        SharedFilesystemConfig.model_validate(
            {
                "provider": "efs",
                "mount_point": "/mnt/x",
                "efs": {
                    "file_system_id": "fs-1",
                    "region": "us-east-1",
                    "cleanup_zone": "us-west-2a",
                },
            }
        )


def test_local_scratch_must_be_absolute():
    with pytest.raises(ValueError, match="local_scratch"):
        SharedFilesystemConfig.model_validate(
            {
                "provider": "efs",
                "mount_point": "/mnt/x",
                "local_scratch": "rel/scratch",
                "efs": {"file_system_id": "fs-1", "region": "us-east-1"},
            }
        )


def test_local_scratch_absolute_ok_and_default_none():
    sf = SharedFilesystemConfig.model_validate(
        {
            "provider": "efs",
            "mount_point": "/mnt/x",
            "local_scratch": "/opt/dlami/nvme/gb-scratch",
            "efs": {"file_system_id": "fs-1", "region": "us-east-1"},
        }
    )
    assert sf.local_scratch == "/opt/dlami/nvme/gb-scratch"
    sf2 = SharedFilesystemConfig.model_validate(
        {
            "provider": "efs",
            "mount_point": "/mnt/x",
            "efs": {"file_system_id": "fs-1", "region": "us-east-1"},
        }
    )
    assert sf2.local_scratch is None


def test_efs_cleanup_zone_in_region_ok():
    sf = SharedFilesystemConfig.model_validate(
        {
            "provider": "efs",
            "mount_point": "/mnt/x",
            "efs": {
                "file_system_id": "fs-1",
                "region": "us-east-1",
                "cleanup_zone": "us-east-1a",
            },
        }
    )
    assert sf.efs.cleanup_zone == "us-east-1a"


def _env(cfg: dict):
    return SimpleNamespace(config=cfg)


def test_resolve_none_config():
    assert resolve_shared_workdir(None) is None


def test_resolve_plain_shared_workdir():
    assert resolve_shared_workdir(_env({"shared_workdir": "/proj/x"})) == "/proj/x"


def test_resolve_shared_filesystem_mount_point():
    root = resolve_shared_workdir(
        _env(
            {
                "shared_filesystem": {
                    "provider": "efs",
                    "mount_point": "/mnt/gb-shared",
                    "efs": {"file_system_id": "fs-1", "region": "us-east-1"},
                }
            }
        )
    )
    assert root == "/mnt/gb-shared"


def test_resolve_neither():
    assert resolve_shared_workdir(_env({})) is None


def test_resolve_both_set_defensive_error():
    with pytest.raises(ValueError, match="both"):
        resolve_shared_workdir(
            _env(
                {
                    "shared_workdir": "/proj/x",
                    "shared_filesystem": {
                        "provider": "efs",
                        "mount_point": "/mnt/y",
                        "efs": {"file_system_id": "fs-1", "region": "us-east-1"},
                    },
                }
            )
        )


def _bash_ok(script: str):
    proc = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr


def test_efs_mount_prologue_installs_nfs_client_and_falls_back_to_nfs4():
    p = EfsProvider(
        "/mnt/gb-shared",
        EfsConfig(file_system_id="fs-0abc", region="us-east-1", tls=True),
    )
    shell = p.mount_prologue()
    assert "mount.nfs4" in shell  # nfs client install guard
    assert "mountpoint -q /mnt/gb-shared" in shell
    assert "mount.efs" in shell  # efs-utils preferred path
    assert "fs-0abc.efs.us-east-1.amazonaws.com:/" in shell  # nfs4 fallback DNS
    assert "-o tls" in shell
    assert "failed" in shell  # fail-fast message
    _bash_ok(shell)


def test_efs_mount_prologue_is_root_safe_no_bare_sudo():
    """Regression (#393): a containerized step runs as root in a minimal image
    (e.g. debian:12-slim) that has NO `sudo`; the prologue must gate sudo on the
    effective uid ($SUDO) rather than calling bare `sudo`, else the in-container
    EFS mount dies with 'sudo: not found' and the workload fails."""
    p = EfsProvider(
        "/mnt/gb-shared",
        EfsConfig(file_system_id="fs-0abc", region="us-east-1", tls=True),
    )
    shell = p.mount_prologue()
    # Defines a uid-gated $SUDO and uses it for the privileged commands...
    assert "id -u" in shell and "SUDO=" in shell
    assert "$SUDO mount" in shell
    assert "$SUDO mkdir" in shell
    assert "$SUDO apt-get" in shell
    # ...and never calls bare `sudo` (absent when running as root in a container).
    assert "sudo mount" not in shell
    assert "sudo mkdir" not in shell
    assert "sudo apt-get" not in shell
    _bash_ok(shell)


def test_efs_mount_prologue_warns_when_tls_requested_but_efs_utils_absent():
    """Regression (#389 review): `-o tls` only encrypts on the mount.efs path;
    plain nfs4 (the common fallback on stock images) cannot do EFS TLS, so a
    tls=true config that falls back must warn loudly rather than silently mount in
    cleartext while claiming encryption in transit."""
    p = EfsProvider(
        "/mnt/gb-shared",
        EfsConfig(file_system_id="fs-0abc", region="us-east-1", tls=True),
    )
    shell = p.mount_prologue()
    # tls is honored only on the mount.efs branch...
    assert "mount -t efs -o tls" in shell
    # ...and the nfs4 fallback emits a visible unencrypted-transit warning.
    assert "WITHOUT encryption" in shell
    _bash_ok(shell)


def test_efs_mount_prologue_no_tls_and_no_warning_when_tls_false():
    p = EfsProvider(
        "/mnt/gb-shared",
        EfsConfig(file_system_id="fs-0abc", region="us-east-1", tls=False),
    )
    shell = p.mount_prologue()
    assert "-o tls" not in shell
    assert "WITHOUT encryption" not in shell
    _bash_ok(shell)


def test_efs_cleanup_run_script_mounts_then_reaps():
    p = EfsProvider(
        "/mnt/gb-shared", EfsConfig(dns_name="fs-0abc.efs.eu-west-1.amazonaws.com")
    )
    script = p.cleanup_run_script("/mnt/gb-shared/builds/b1/runs/r1")
    assert "mountpoint -q /mnt/gb-shared" in script
    assert "rm -rf '/mnt/gb-shared/builds/b1/runs/r1'" in script
    assert "rmdir" in script  # parent reap
    _bash_ok(script)


def test_efs_transit_encryption_note_when_tls():
    """Regression (#389 review): a server-side note so an operator watching gbserver
    logs learns the mount may be cleartext (the step-log warning alone isn't seen)."""
    p = EfsProvider(
        "/mnt/gb-shared",
        EfsConfig(file_system_id="fs-0abc", region="us-east-1", tls=True),
    )
    note = p.transit_encryption_note()
    assert note is not None and "nfs4" in note.lower()


def test_efs_transit_encryption_note_none_when_tls_false():
    p = EfsProvider(
        "/mnt/gb-shared",
        EfsConfig(file_system_id="fs-0abc", region="us-east-1", tls=False),
    )
    assert p.transit_encryption_note() is None


def test_efs_cleanup_zone_from_config():
    p = EfsProvider(
        "/mnt/gb-shared",
        EfsConfig(file_system_id="fs-1", region="us-east-1", cleanup_zone="us-east-1a"),
    )
    assert p.cleanup_zone() == "us-east-1a"


def test_build_provider_none_when_absent():
    assert build_provider(_env({})) is None
    assert build_provider(_env({"shared_workdir": "/proj/x"})) is None
    assert build_provider(None) is None


def test_build_provider_returns_efs():
    prov = build_provider(
        _env(
            {
                "shared_filesystem": {
                    "provider": "efs",
                    "mount_point": "/mnt/gb-shared",
                    "efs": {"file_system_id": "fs-1", "region": "us-east-1"},
                }
            }
        )
    )
    assert isinstance(prov, EfsProvider)
    assert prov.mount_point == "/mnt/gb-shared"
