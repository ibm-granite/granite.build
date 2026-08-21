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

"""Tests for the shared identity-resolution helper — extracted from three
independent copies (api/chat.py's _resolve_identity, api/analytics.py's
get_current_author, api/ai.py's _rate_limit_analyze_logs) that all
implemented the same AuthMiddleware-trusted-user -> X-User-Email header ->
fallback precedence, with only the final fallback differing by caller.
"""

from __future__ import annotations

from types import SimpleNamespace

from gb_ui_backend.services.request_identity import resolve_identity


def _fake_request(
    user_email: str | None = None, header_email: str | None = None
) -> SimpleNamespace:
    state = SimpleNamespace(
        data={"user": SimpleNamespace(email=user_email)} if user_email else {}
    )
    headers = {"x-user-email": header_email} if header_email else {}
    return SimpleNamespace(state=state, headers=headers)


class TestResolveIdentity:
    def test_trusted_authmiddleware_user_wins_over_header(self):
        request = _fake_request(
            user_email="alice@example.com", header_email="bob@example.com"
        )
        assert resolve_identity(request) == "alice@example.com"

    def test_header_is_a_fallback_when_no_authmiddleware_user(self):
        request = _fake_request(header_email="alice@example.com")
        assert resolve_identity(request) == "alice@example.com"

    def test_default_fallback_is_standalone(self):
        request = _fake_request()
        assert resolve_identity(request) == "standalone"

    def test_caller_supplied_fallback_is_used_instead_of_standalone(self):
        request = _fake_request()
        assert resolve_identity(request, fallback="127.0.0.1") == "127.0.0.1"

    def test_trusted_user_wins_even_over_a_custom_fallback(self):
        request = _fake_request(user_email="alice@example.com")
        assert resolve_identity(request, fallback="127.0.0.1") == "alice@example.com"
