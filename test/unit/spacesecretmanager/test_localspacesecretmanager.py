import pytest

from gbserver.spacesecretmanager.localspacesecretmanager import LocalSpaceSecretManager


def test_create_secret_creates_new_yaml_file(tmp_path):
    dir_path = tmp_path / "secrets"
    dir_path.mkdir(parents=True, exist_ok=True)
    manager = LocalSpaceSecretManager(uri="local", secrets_dir=dir_path)
    manager.create_secret(
        secret_name="API_KEY", secret_value="my-secret", secret_group_name="group1"
    )

    target_file = dir_path / "group1.yaml"
    assert target_file.exists()

    # File should contain encoded value
    import base64

    import yaml

    data = yaml.safe_load(target_file.read_text())
    assert data["API_KEY"] == base64.b64encode(b"my-secret").decode("utf-8")


def test_create_secret_overwrites_existing_secret(tmp_path, caplog):
    dir_path = tmp_path
    file = dir_path / "config.yaml"
    file.write_text("API_KEY: bXktb2xkLXNlY3JldA==")  # base64("my-old-secret")
    manager = LocalSpaceSecretManager(uri="local", secrets_dir=file)
    manager.create_secret("API_KEY", "new-value")
    assert "Overriding value" in caplog.text
    # verify encoded new value
    import base64

    import yaml

    updated = yaml.safe_load(file.read_text())
    assert updated["API_KEY"] == base64.b64encode(b"new-value").decode("utf-8")


def test_create_secret_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("")  # empty file

    manager = LocalSpaceSecretManager(uri="local", secrets_dir=env_file)
    manager.create_secret("TOKEN", "abc123")

    raw = env_file.read_text().strip()
    key, value = raw.split("=")

    assert key == "TOKEN"
    import base64

    assert value == base64.b64encode(b"abc123").decode("utf-8")


def test_get_secrets_returns_decoded_values(tmp_path):
    yaml_file = tmp_path / "secrets.yaml"

    yaml_file.write_text("""
SECRET1: c2VjcmV0MQ==
SECRET2: dGVzdDI=
""")

    manager = LocalSpaceSecretManager(uri="local", secrets_dir=yaml_file)
    secrets = manager.get_secrets()

    assert secrets["SECRET1"] == "secret1"
    assert secrets["SECRET2"] == "test2"


# ------------------------------------------------------------------ per-space isolation
#
# With no explicit secrets_dir, every space used to share <gb_home>/space_secrets/
# and reads loaded every file in it, so space A's builds and admins saw space B's
# secrets. The default is now <gb_home>/space_secrets/<space name>/.

from unittest.mock import patch  # noqa: E402

from gbserver.spacesecretmanager import (  # noqa: E402
    localspacesecretmanager as local_module,
)


@pytest.fixture
def gb_home(tmp_path):
    with patch.object(local_module, "get_gb_home_dir", return_value=str(tmp_path)):
        yield tmp_path


def _manager(space_name):
    return LocalSpaceSecretManager(uri="local", space_name=space_name)


def test_default_dir_is_per_space(gb_home):
    assert _manager("space-a").dir == gb_home / "space_secrets" / "space-a"


def test_spaces_do_not_see_each_others_secrets(gb_home):
    a, b = _manager("space-a"), _manager("space-b")
    a.create_secret("A_TOKEN", "a-value", secret_group_name="space-a")
    b.create_secret("B_TOKEN", "b-value", secret_group_name="space-b")

    assert a.get_secrets() == {"A_TOKEN": "a-value"}
    assert b.get_secrets() == {"B_TOKEN": "b-value"}
    assert b.get_secret("A_TOKEN") == {}
    assert a.list_secret_names() == ["A_TOKEN"]


def test_default_dir_requires_space_name(gb_home):
    with pytest.raises(ValueError, match="space_name"):
        LocalSpaceSecretManager(uri="local")


@pytest.mark.parametrize("name", ["", ".", "..", "../b", "a/b", "a\\b"])
def test_unsafe_space_name_rejected(gb_home, name):
    with pytest.raises(ValueError, match="space_name"):
        _manager(name)


def test_explicit_secrets_dir_ignores_space_name(tmp_path):
    manager = LocalSpaceSecretManager(
        uri="local", secrets_dir=tmp_path, space_name="space-a"
    )
    assert manager.dir == tmp_path


def test_legacy_flat_file_is_migrated(gb_home):
    flat = gb_home / "space_secrets"
    flat.mkdir()
    (flat / "space-a.yaml").write_text("A_TOKEN: YS12YWx1ZQ==\n")  # "a-value"
    (flat / "space-b.yaml").write_text("B_TOKEN: Yi12YWx1ZQ==\n")  # "b-value"

    a = _manager("space-a")

    assert a.get_secrets() == {"A_TOKEN": "a-value"}
    assert (flat / "space-a" / "space-a.yaml").is_file()
    assert not (flat / "space-a.yaml").exists()
    # Another space's legacy file is left alone (and never read by space-a).
    assert (flat / "space-b.yaml").is_file()


def test_legacy_migration_never_overwrites(gb_home):
    flat = gb_home / "space_secrets"
    (flat / "space-a").mkdir(parents=True)
    (flat / "space-a" / "space-a.yaml").write_text("A_TOKEN: bmV3\n")  # "new"
    (flat / "space-a.yaml").write_text("A_TOKEN: b2xk\n")  # "old"

    assert _manager("space-a").get_secrets() == {"A_TOKEN": "new"}
    assert (flat / "space-a.yaml").is_file()


def test_hybrid_forwards_space_name(gb_home):
    from gbserver.spacesecretmanager.hybridspacesecretmanager import (
        HybridSpaceSecretManager,
    )
    from gbserver.spacesecretmanager.spacesecretmanager import SpaceSecretManager

    SpaceSecretManager.load_spacesecretmanagers()
    hybrid = HybridSpaceSecretManager(
        uri="local", managers=[{"type": "local", "config": {}}], space_name="space-a"
    )
    assert hybrid.managers[0].dir == gb_home / "space_secrets" / "space-a"


def test_leftover_flat_files_are_ignored_with_a_warning(gb_home, caplog):
    # e.g. a hand-placed space_secrets/aws.json from before per-space isolation:
    # it is not read (it may belong to any space), but the user is told where it is.
    flat = gb_home / "space_secrets"
    flat.mkdir()
    (flat / "aws.json").write_text('{"GB_AWS_ACCESS_KEY_ID": "a2V5"}')
    local_module._warned_legacy_dirs.clear()

    a = _manager("space-a")
    _manager("space-a")  # warned once per directory, not per construction

    assert a.get_secrets() == {}
    warnings = [r for r in caplog.records if "aws.json" in r.getMessage()]
    assert len(warnings) == 1
    assert "space-a" in warnings[0].getMessage()
