# coding=utf-8
# Copyright 2023-present International Business Machines Corporation
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

"""Opt-in gate for AutoTuneX bridge logging.

Bridge logging is OFF by default. It is enabled only when the user passes a
non-empty ``--autotunex_server_url``. Keeping this decision in a tiny importable
helper lets it be unit-tested without executing main.py (which triggers Ray).
"""

from typing import Optional, Tuple


def resolve_bridge_settings(autotunex_server_url: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Return ``(bridge_enabled, base_url)``.

    The bridge is enabled only when ``autotunex_server_url`` is a non-empty
    string. An absent (``None``) or empty value means the run executes fully
    offline with no bridge calls.
    """
    if autotunex_server_url:
        return True, autotunex_server_url
    return False, None
