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

"""Tests for get_gb_log_reader selection.

``tests/conftest.py``'s ``make_settings`` does not accept ``job_backend`` (or the
``job_runtime_image``/``job_trainer_repo``/``job_output_uri_root``/``gb_server_url``
that ``Settings._validate_job_backend`` requires alongside it for ``"llmb"``), and
this task does not extend that shared fixture. The disabled-path selection is
covered end-to-end through ``get_gb_log_reader`` using ``make_settings``'s
defaults; the gbcli-path is covered the same way, through a real ``Settings``
built inline here with the fields ``_validate_job_backend`` requires for
``"llmb"``, so the registry's selection branch is genuinely exercised rather
than assumed.

(See this directory's ``conftest.py`` for why an unrelated-looking
``os.environ`` restore fixture is needed for these tests to be
order-independent.)
"""

from __future__ import annotations

import pytest

from autotunex.core.config import Settings
from autotunex.services.gb_logs.disabled_reader import DisabledGbLogReader
from autotunex.services.gb_logs.gbcli_reader import GbcliLogReader
from autotunex.services.gb_logs.registry import get_gb_log_reader
from tests.conftest import make_settings


def test_returns_disabled_reader_when_backend_is_none() -> None:
    reader = get_gb_log_reader(make_settings())

    assert isinstance(reader, DisabledGbLogReader)


def test_returns_gbcli_reader_when_backend_is_llmb_and_token_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GB_TOKEN", "tok")
    settings = Settings(
        _env_file=None,
        environment="test",
        auto_create_schema=False,
        job_backend="llmb",
        job_runtime_image="example.registry/build-runtime:1",
        job_trainer_repo="github.example.com/org/trainer.git",
        job_output_uri_root="hf://huggingface.co/org",
        gb_server_url="https://gbserver.example.com",
    )

    reader = get_gb_log_reader(settings)

    assert isinstance(reader, GbcliLogReader)
