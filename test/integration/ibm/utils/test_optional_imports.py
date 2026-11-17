# SPDX-License-Identifier: Apache-2.0

"""Tests that optional IBM/RabbitMQ packages are present in the full IBM venv."""

import pytest

pytestmark = pytest.mark.ibm


class TestIBMOptionalImports:
    """Verify IBM-specific optional dependency flags are True in the full venv."""

    def test_ibm_secret_manager_has_flag(self):
        """Section 1.6: IBM SDK imports guarded with HAS_IBM_SDK flag."""
        from gbserver.utils.optional_imports import HAS_IBM_SDK

        # In test venv with all packages, flag should be True
        assert HAS_IBM_SDK is True

    def test_rabbitmq_base_has_flag(self):
        """Section 1.8: RabbitMQ imports guarded with HAS_RABBITMQ flag."""
        from gbserver.utils.optional_imports import HAS_RABBITMQ

        # In test venv with all packages, flag should be True
        assert HAS_RABBITMQ is True
