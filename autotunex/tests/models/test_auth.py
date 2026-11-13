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

from __future__ import annotations

from uuid import uuid4

from autotunex.models.auth import Principal


def test_principal_defaults_impersonator_to_none() -> None:
    principal = Principal(email="admin@example.com", provider="session")

    assert principal.impersonator is None


def test_principal_carries_the_impersonator_when_impersonating() -> None:
    principal = Principal(
        email="target@example.com",
        provider="session",
        user_id=uuid4(),
        is_admin=True,
        impersonator="admin@example.com",
    )

    assert principal.impersonator == "admin@example.com"
    assert principal.is_admin is True
