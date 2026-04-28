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

    yaml_file.write_text(
        """
SECRET1: c2VjcmV0MQ==
SECRET2: dGVzdDI=
"""
    )

    manager = LocalSpaceSecretManager(uri="local", secrets_dir=yaml_file)
    secrets = manager.get_secrets()

    assert secrets["SECRET1"] == "secret1"
    assert secrets["SECRET2"] == "test2"
