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

"""The LLM client registry: configured -> adapter, unset -> 503."""

from __future__ import annotations

import httpx
import pytest

from autotunex.core.exceptions import LlmNotConfiguredError
from autotunex.services.llm.openai_compatible import OpenAiCompatibleLlmClient
from autotunex.services.llm.registry import get_llm_client
from tests.conftest import make_settings


def test_returns_an_adapter_when_configured() -> None:
    settings = make_settings(llm_base_url="https://gw.example/v1", llm_api_key="sk", llm_model="m")

    client = get_llm_client(settings, httpx.AsyncClient())

    assert isinstance(client, OpenAiCompatibleLlmClient)


def test_raises_503_when_unconfigured() -> None:
    with pytest.raises(LlmNotConfiguredError):
        get_llm_client(make_settings(), httpx.AsyncClient())
