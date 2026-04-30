import random
import string

import pytest

from gbserver.spacesecretmanager.ibmcloudspacesecretmanager import (
    IbmcloudSpaceSecretManagerAdmin,
)

pytestmark = pytest.mark.ibm


class TestIbmcloudSpaceSecretManagerAdmin:
    """
    Skipping this test by default because it uses actual access to IBM Secret Manager API. To run it,
        export IBM_CLOUD_SECRETS_MANAGER_SERVICE_URL=<url>
        export IBM_CLOUD_API_KEY=<key>
        pytest -s test/gbserver_test/spacesecretmanager/test_ibmcloudspacesecretmanager.py -m secret_manager
    """

    def test_secret_manager_admin_secret_groups(self):
        manager = IbmcloudSpaceSecretManagerAdmin()

        letters = string.ascii_lowercase
        secret_group_name = "test_secret_group" + "".join(random.choice(letters) for i in range(6))
        secret_groups = manager.get_all_secret_groups()
        secret_groups_before = list(filter(lambda x: x["name"] == secret_group_name, secret_groups))
        assert len(secret_groups_before) == 0

        description1 = "my test 1"
        manager.create_secret_group(secret_group_name, description1)
        secret_groups = manager.get_all_secret_groups()
        secret_groups_after = list(filter(lambda x: x["name"] == secret_group_name, secret_groups))
        assert len(secret_groups_after) == 1

        secret_group = secret_groups_after[0]
        assert secret_group["name"] == secret_group_name
        assert secret_group["description"] == description1

        description2 = "my test 2"
        manager.update_secret_group_description(secret_group_name, description2)
        secret_groups = manager.get_all_secret_groups()
        secret_groups_renamed = list(
            filter(lambda x: x["name"] == secret_group_name, secret_groups)
        )
        assert len(secret_groups_renamed) == 1
        secret_group = secret_groups_renamed[0]
        assert secret_group["name"] == secret_group_name
        assert secret_group["description"] == description2

        secret_group = manager.get_secret_group_by_name(secret_group_name)
        assert secret_group["name"] == secret_group_name
        assert secret_group["description"] == description2

        manager.delete_secret_group(secret_group_name)
        secret_groups = manager.get_all_secret_groups()
        secret_groups_final = list(filter(lambda x: x["name"] == secret_group_name, secret_groups))
        assert len(secret_groups_final) == 0

    def test_secret_manager_admin_secrets(self):
        manager = IbmcloudSpaceSecretManagerAdmin()

        letters = string.ascii_lowercase
        secret_group_name = "test_secret_group" + "".join(random.choice(letters) for i in range(6))
        description1 = "my test 1"
        manager.create_secret_group(secret_group_name, description1)

        secret_name = "mysecretABCDE"
        secret_value1 = "mysecretabcde"

        manager.create_secret(
            secret_group_name=secret_group_name,
            secret_name=secret_name,
            secret_value=secret_value1,
        )

        secrets = manager.list_secrets(secret_group_name)
        assert len(secrets) == 1

        secret_names = manager.list_secret_names(secret_group_name)
        assert len(secret_names) == 1

        assert manager.get_secret_value(secret_group_name, secret_name) == secret_value1

        secret_value2 = "mysecretxyzw"
        manager.update_secret_value(secret_group_name, secret_name, secret_value2)

        assert manager.get_secret_value(secret_group_name, secret_name) == secret_value2

        manager.delete_secret(secret_group_name, secret_name)
        secrets = manager.list_secrets(secret_group_name)
        assert len(secrets) == 0

        manager.delete_secret_group(secret_group_name)
