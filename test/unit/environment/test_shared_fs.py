import pytest

from gbserver.environment.shared_fs.config import EfsConfig, SharedFilesystemConfig


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


from types import SimpleNamespace

from gbserver.environment.shared_fs.base import resolve_shared_workdir


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
