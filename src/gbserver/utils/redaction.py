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

"""Key-name based redaction of secret-looking values before they leave the server.

Used when emitting step ``config``/``metadata`` into build lineage, which is readable
by any space member. Redaction is by *key name* (not value inspection), so a secret
stored under a non-secret-looking key is not masked — an accepted defense-in-depth
tradeoff, since step metadata is operational data (e.g. a git commit SHA).
"""

import re
from typing import Any

__all__ = ["SENSITIVE_KEY_RE", "REDACTED", "redact_sensitive"]

# Case-insensitive, ``-``/``_`` tolerant match on common secret-bearing key names.
# Verbose form (whitespace/comments ignored) for readability; each alternative is a
# substring test against the key name. ``[_-]?`` also tolerates camelCase because the
# separator is optional and the following letter is matched case-insensitively (so
# ``apiKey``/``sshKey`` match too).
SENSITIVE_KEY_RE = re.compile(
    r"""
      password | passwd | pwd
    | secret
    | token
    | credential
    | cookie
    | bearer
    | api[_-]?key
    | access[_-]?key
    | private[_-]?key
    | ssh[_-]?key
    | authorization
    | auth(?!or|en)          # 'auth', 'auth_token', 'authToken' — but NOT 'author'/'authentic…'
                             # Exclusion is on the word stem (case-insensitive via re.IGNORECASE),
                             # so all-caps keys like AUTHOR / AUTHORED_DATE / GIT_AUTHOR are also
                             # left unmasked (the old (?-i:[a-z]) guard only skipped lowercase).
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Placeholder substituted for the value of any secret-looking key.
REDACTED = "<redacted>"


def redact_sensitive(value: Any) -> Any:
    """Recursively copy a mapping/list, masking values under secret-looking keys.

    :param value: any value; dicts and lists are copied and recursed into, other
        types are returned unchanged (no mutation of the input).
    :returns: a redacted deep-ish copy where any dict key matching
        ``SENSITIVE_KEY_RE`` has its value replaced with ``REDACTED``; nested dicts
        and lists are redacted in turn.
    """
    if isinstance(value, dict):
        return {
            key: (
                REDACTED
                if isinstance(key, str) and SENSITIVE_KEY_RE.search(key)
                else redact_sensitive(val)
            )
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    return value
