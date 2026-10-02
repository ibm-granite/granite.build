import asyncio
import pathlib
import subprocess
from types import SimpleNamespace
from unittest import mock

import pytest
import yaml

from gbserver.environment.shared_fs import build_providers
from gbserver.environment.shared_fs.base import (
    ProvisionedResources,
    resolve_local_scratch,
    resolve_shared_workdir,
    resolve_workdir_mount,
)
from gbserver.environment.shared_fs.config import (
    EfsConfig,
    SharedFilesystemConfig,
    parse_shared_filesystems,
)
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


def test_efs_cleanup_zone_rejected_for_ephemeral():
    """Ephemeral teardown deletes via boto3 and never launches the cleanup VM
    that consumes cleanup_zone, so accepting it would be a silent no-op; reject
    it at validation instead (PR #422 review)."""
    with pytest.raises(ValueError, match="cleanup_zone"):
        EfsConfig(provision="ephemeral", region="us-east-1", cleanup_zone="us-east-1a")


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


# --- Task 1: EfsConfig.provision ephemeral/BYO validation (#391) ---


def test_efs_byo_defaults_to_byo_and_validates_as_before():
    cfg = EfsConfig(file_system_id="fs-1", region="us-east-1")
    assert cfg.provision == "byo"
    assert cfg.derived_dns_name() == "fs-1.efs.us-east-1.amazonaws.com"


def test_efs_ephemeral_requires_region():
    with pytest.raises(ValueError, match="ephemeral.*region"):
        EfsConfig(provision="ephemeral")


def test_efs_ephemeral_forbids_file_system_id():
    with pytest.raises(ValueError, match="ephemeral.*must not set"):
        EfsConfig(provision="ephemeral", region="us-east-1", file_system_id="fs-1")


def test_efs_ephemeral_forbids_dns_name():
    with pytest.raises(ValueError, match="ephemeral.*must not set"):
        EfsConfig(provision="ephemeral", region="us-east-1", dns_name="x.example.com")


def test_efs_ephemeral_accepts_optional_networking():
    cfg = EfsConfig(
        provision="ephemeral",
        region="us-east-1",
        vpc_id="vpc-1",
        subnets=["subnet-a"],
        security_group_id="sg-1",
    )
    assert cfg.vpc_id == "vpc-1" and cfg.subnets == ["subnet-a"]
    assert cfg.derived_dns_name() is None  # no fsid yet


# --- Task 2: parse_shared_filesystems list coercion + cross-mount rules (#404) ---


def _byo(mp, fsid):
    return {
        "provider": "efs",
        "mount_point": mp,
        "efs": {"file_system_id": fsid, "region": "us-east-1"},
    }


def test_parse_none_returns_empty():
    assert parse_shared_filesystems(None) == []


def test_parse_lone_object_coerced_to_one_list():
    out = parse_shared_filesystems(_byo("/mnt/a", "fs-a"))
    assert len(out) == 1 and out[0].mount_point == "/mnt/a"


def test_parse_list_passthrough():
    out = parse_shared_filesystems([_byo("/mnt/a", "fs-a"), _byo("/mnt/b", "fs-b")])
    assert [m.mount_point for m in out] == ["/mnt/a", "/mnt/b"]


def test_parse_rejects_duplicate_mount_point():
    with pytest.raises(ValueError, match="unique"):
        parse_shared_filesystems([_byo("/mnt/a", "fs-a"), _byo("/mnt/a", "fs-b")])


def test_parse_rejects_nested_mount_point():
    with pytest.raises(ValueError, match="nested|under"):
        parse_shared_filesystems([_byo("/mnt/a", "fs-a"), _byo("/mnt/a/b", "fs-b")])


def test_parse_rejects_multiple_local_scratch():
    a = _byo("/mnt/a", "fs-a")
    a["local_scratch"] = "/tmp/s1"
    b = _byo("/mnt/b", "fs-b")
    b["local_scratch"] = "/tmp/s2"
    with pytest.raises(ValueError, match="local_scratch"):
        parse_shared_filesystems([a, b])


# --- Task 3: ProvisionedResources, workdir-mount resolver, list local_scratch ---
# NOTE: these reuse the module-level SimpleNamespace `_env` (resolve_* only read
# `.config`); the full EnvironmentConfig gate is exercised in test_environmentconfig.


def test_provisioned_resources_fields():
    pr = ProvisionedResources(
        region="us-east-1",
        file_system_id="fs-1",
        dns_name="fs-1.efs.us-east-1.amazonaws.com",
        mount_target_ids=["mt-1"],
        subnet_ids=["subnet-a"],
        security_group_id="sg-1",
        created_sg=True,
    )
    assert pr.file_system_id == "fs-1" and pr.created_sg is True


def test_resolve_workdir_mount_prefix_selects_mount():
    cfg = {
        "default_cloud": "aws",
        "shared_filesystem": [
            {
                "provider": "efs",
                "mount_point": "/mnt/a",
                "efs": {"file_system_id": "fs-a", "region": "us-east-1"},
            },
            {
                "provider": "efs",
                "mount_point": "/mnt/b",
                "efs": {"file_system_id": "fs-b", "region": "us-east-1"},
            },
        ],
        "shared_workdir": "/mnt/b/work",
    }
    m = resolve_workdir_mount(_env(cfg))
    assert m is not None and m.mount_point == "/mnt/b"


def test_resolve_workdir_mount_under_none_returns_none():
    cfg = {
        "default_cloud": "aws",
        "shared_filesystem": [
            {
                "provider": "efs",
                "mount_point": "/mnt/a",
                "efs": {"file_system_id": "fs-a", "region": "us-east-1"},
            }
        ],
        "shared_workdir": "/mnt/z/work",
    }
    assert resolve_workdir_mount(_env(cfg)) is None


def test_resolve_local_scratch_from_list():
    cfg = {
        "default_cloud": "aws",
        "shared_workdir": "/mnt/a/w",
        "shared_filesystem": [
            {
                "provider": "efs",
                "mount_point": "/mnt/a",
                "local_scratch": "/opt/nvme/scratch",
                "efs": {"file_system_id": "fs-a", "region": "us-east-1"},
            }
        ],
    }
    assert resolve_local_scratch(_env(cfg)) == "/opt/nvme/scratch"


def _env(cfg: dict):
    return SimpleNamespace(config=cfg)


def test_resolve_none_config():
    assert resolve_shared_workdir(None) is None


def test_resolve_returns_explicit_shared_workdir_with_shared_filesystem():
    env = _env(
        {
            "default_cloud": "aws",
            "shared_workdir": "/mnt/gb-shared/gbroot",
            "shared_filesystem": {
                "provider": "efs",
                "mount_point": "/mnt/gb-shared",
                "efs": {"file_system_id": "fs-0abc123", "region": "us-east-1"},
            },
        }
    )
    assert resolve_shared_workdir(env) == "/mnt/gb-shared/gbroot"


def test_resolve_returns_legacy_shared_workdir_without_shared_filesystem():
    env = _env({"shared_workdir": "/shared"})
    assert resolve_shared_workdir(env) == "/shared"


def test_resolve_none_when_neither_set():
    assert resolve_shared_workdir(_env({})) is None


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


# --- Task 5: mount_prologue dns_override + ephemeral 1777 root bootstrap (#391) ---


def test_ephemeral_prologue_chmods_root_and_uses_dns_override():
    p = EfsProvider("/mnt/e", EfsConfig(provision="ephemeral", region="us-east-1"))
    sh = p.mount_prologue(dns_override="fs-x.efs.us-east-1.amazonaws.com")
    assert "chmod 1777 /mnt/e" in sh
    assert "fs-x.efs.us-east-1.amazonaws.com:/" in sh
    _bash_ok(sh)


def test_byo_prologue_does_not_chmod_root():
    p = EfsProvider("/mnt/b", EfsConfig(file_system_id="fs-b", region="us-east-1"))
    sh = p.mount_prologue()
    assert "chmod 1777" not in sh
    assert "fs-b.efs.us-east-1.amazonaws.com:/" in sh
    _bash_ok(sh)


def test_dns_override_ignored_for_byo_uses_config():
    p = EfsProvider("/mnt/b", EfsConfig(file_system_id="fs-b", region="us-east-1"))
    sh = p.mount_prologue(dns_override="ignored.example.com")
    # BYO derives from config; override only matters when config has no dns (ephemeral)
    assert "fs-b.efs.us-east-1.amazonaws.com:/" in sh
    assert "ignored.example.com" not in sh


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


def test_ephemeral_prologue_honors_tls_via_runtime_fsid():
    """Regression (PR #422 review, security): ephemeral forbids a config
    file_system_id, so the mount.efs (`-o tls`) branch was never built and the
    mount silently fell back to cleartext nfs4 even where amazon-efs-utils is
    present -- defeating the tls=true default. The runtime DNS embeds the fsid
    (`<fsid>.efs.<region>...`), so the mount.efs branch must honor -o tls using
    the recovered fsid."""
    p = EfsProvider("/mnt/e", EfsConfig(provision="ephemeral", region="us-east-1"))
    sh = p.mount_prologue(dns_override="fs-x.efs.us-east-1.amazonaws.com")
    # mount.efs branch honors tls using the fsid recovered from the runtime DNS
    assert "mount -t efs -o tls fs-x:/" in sh
    # nfs4 fallback still present (stock images) with its unencrypted warning
    assert "mount -t nfs4" in sh
    assert "WITHOUT encryption" in sh
    _bash_ok(sh)


def test_ephemeral_prologue_no_tls_branch_uses_fsid_without_tls_flag():
    p = EfsProvider(
        "/mnt/e", EfsConfig(provision="ephemeral", region="us-east-1", tls=False)
    )
    sh = p.mount_prologue(dns_override="fs-x.efs.us-east-1.amazonaws.com")
    assert "mount -t efs fs-x:/" in sh  # efs path, no -o tls
    assert "-o tls" not in sh
    _bash_ok(sh)


def test_ephemeral_prologue_missing_runtime_dns_raises_clear_error():
    """Regression (PR #422 review): a stale/replayed setup_config (or a broken
    retry path) can leave an ephemeral mount with no runtime dns_override.
    Rather than let `shlex.quote(None + ':/')` raise an opaque TypeError, fail
    with a clear message naming the mount."""
    p = EfsProvider("/mnt/e", EfsConfig(provision="ephemeral", region="us-east-1"))
    with pytest.raises(ValueError, match="no runtime DNS"):
        p.mount_prologue(dns_override=None)


def test_efs_cleanup_run_script_mounts_then_reaps():
    p = EfsProvider(
        "/mnt/gb-shared", EfsConfig(dns_name="fs-0abc.efs.eu-west-1.amazonaws.com")
    )
    script = p.cleanup_run_script("/mnt/gb-shared/builds/b1/runs/r1")
    assert "mountpoint -q /mnt/gb-shared" in script
    assert "rm -rf '/mnt/gb-shared/builds/b1/runs/r1'" in script
    assert "rmdir" in script  # parent reap
    _bash_ok(script)


# --- Task 7: EfsProvider.provision/deprovision delegate to boto3 helper (#391) ---


def test_efs_provider_provision_byo_returns_none():
    p = EfsProvider("/mnt/b", EfsConfig(file_system_id="fs-b", region="us-east-1"))
    assert asyncio.run(p.provision({"gb-build-id": "b1"}, "gb-skypilot")) is None


def test_efs_provider_provision_ephemeral_delegates():
    p = EfsProvider(
        "/mnt/e",
        EfsConfig(
            provision="ephemeral",
            region="us-east-1",
            subnets=["subnet-a"],
            security_group_id="sg-1",
        ),
    )
    fake_pr = ProvisionedResources(
        region="us-east-1",
        file_system_id="fs-x",
        dns_name="fs-x.efs.us-east-1.amazonaws.com",
        mount_target_ids=["mt-1"],
        subnet_ids=["subnet-a"],
        security_group_id="sg-1",
        created_sg=False,
    )
    with (
        mock.patch.object(EfsProvider, "_session", return_value=object()) as sess,
        mock.patch(
            "gbserver.environment.shared_fs.efs.provision_efs", return_value=fake_pr
        ) as pv,
    ):
        out = asyncio.run(p.provision({"gb-build-id": "b1"}, "gb-skypilot"))
    assert out is fake_pr
    sess.assert_called_once_with("gb-skypilot")  # profile threaded through
    # region + optional networking forwarded to the helper
    args, kw = pv.call_args
    assert "us-east-1" in args or kw.get("region") == "us-east-1"


def test_efs_provider_deprovision_raises_on_failure():
    p = EfsProvider("/mnt/e", EfsConfig(provision="ephemeral", region="us-east-1"))
    pr = ProvisionedResources(
        region="us-east-1",
        file_system_id="fs-x",
        dns_name="d",
        mount_target_ids=[],
        subnet_ids=[],
        security_group_id=None,
        created_sg=False,
    )
    with (
        mock.patch.object(EfsProvider, "_session", return_value=object()),
        mock.patch(
            "gbserver.environment.shared_fs.efs.deprovision_efs",
            return_value=["delete_file_system fs-x: boom"],
        ),
    ):
        with pytest.raises(Exception) as ei:
            asyncio.run(p.deprovision(pr, "gb-skypilot"))
    assert "fs-x" in str(ei.value)


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


# --- Task 4: build_providers returns one provider per mount (#404) ---


def test_build_providers_empty_when_no_block():
    assert build_providers(_env({"default_cloud": "aws"})) == []
    assert build_providers(_env({})) == []
    assert build_providers(None) == []


def test_build_providers_one_per_mount_in_order():
    cfg = {
        "default_cloud": "aws",
        "shared_workdir": "/mnt/a/w",
        "shared_filesystem": [
            {
                "provider": "efs",
                "mount_point": "/mnt/a",
                "efs": {"file_system_id": "fs-a", "region": "us-east-1"},
            },
            {
                "provider": "efs",
                "mount_point": "/mnt/b",
                "efs": {"provision": "ephemeral", "region": "us-east-1"},
            },
        ],
    }
    ps = build_providers(_env(cfg))
    assert [p.mount_point for p in ps] == ["/mnt/a", "/mnt/b"]
    assert all(isinstance(p, EfsProvider) for p in ps)
    assert ps[1].cfg.provision == "ephemeral"


def test_build_providers_single_object_backcompat():
    prov = build_providers(
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
    assert len(prov) == 1 and isinstance(prov[0], EfsProvider)
    assert prov[0].mount_point == "/mnt/gb-shared"


# --- Task 13: example environment fixtures validate (ephemeral + multi-mount) ---

_ROOT = pathlib.Path(__file__).resolve().parents[3]  # repo root


@pytest.mark.parametrize(
    "rel",
    [
        "test-data/integration/ibm/buildrunner/skypilot/aws/ephemeral-efs/space/"
        "environments/skypilot/aws-ephemeral/environment.yaml",
        "test-data/integration/ibm/buildrunner/skypilot/aws/multi-efs/space/"
        "environments/skypilot/aws-multi/environment.yaml",
        "test-data/integration/ibm/buildrunner/skypilot/aws/multi-ephemeral-efs/"
        "space/environments/skypilot/aws-multi-ephemeral/environment.yaml",
    ],
)
def test_example_env_fixtures_validate(rel):
    from gbserver.types.environmentconfig import EnvironmentConfig

    data = yaml.safe_load((_ROOT / rel).read_text())
    EnvironmentConfig.model_validate(data)  # must not raise
