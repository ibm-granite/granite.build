# Copyright Granite.Build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
