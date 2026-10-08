#!/usr/bin/env python3

# Copyright LLM.build Authors
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

"""``gb artifact`` handles lakehouse's UnauthorizedException without importing
lakehouse (optional, and slow to import), with or without it installed."""

import sys
import types
from unittest.mock import patch

import pytest

from gbcli.commands import command_artifact

pytestmark = pytest.mark.standalone


def _raise_and_classify(exc: BaseException) -> str:
    try:
        raise exc
    except command_artifact._lakehouse_unauthorized():
        return "unauthorized"
    except Exception:
        return "other"


def test_without_lakehouse_loaded_nothing_matches():
    with patch.dict(sys.modules, {"lakehouse.core": None}):
        assert (
            command_artifact._lakehouse_unauthorized()
            is command_artifact._MissingLakehouseUnauthorizedException
        )
        assert _raise_and_classify(ValueError("x")) == "other"


def test_loaded_lakehouse_exception_is_caught():
    fake_core = types.ModuleType("lakehouse.core")

    class UnauthorizedException(Exception):
        pass

    fake_core.UnauthorizedException = UnauthorizedException
    with patch.dict(sys.modules, {"lakehouse.core": fake_core}):
        assert _raise_and_classify(UnauthorizedException("denied")) == "unauthorized"
        assert _raise_and_classify(ValueError("x")) == "other"


def test_real_lakehouse_exception_is_caught():
    lh_core = pytest.importorskip("lakehouse.core")
    assert _raise_and_classify(lh_core.UnauthorizedException("denied")) == (
        "unauthorized"
    )
