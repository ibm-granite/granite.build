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

"""Redaction of secret-looking values before they leave the server.

Used when emitting step ``config``/``metadata`` into build lineage, which is readable
by any space member. Two complementary passes run:

* **Key-name masking** (:data:`SENSITIVE_KEY_RE`) — replaces the value of any key whose
  *name* looks secret (``password``, ``token``, ...). A secret under a non-secret-looking
  key is not caught by this pass alone — an accepted defense-in-depth tradeoff, since step
  metadata is largely operational data (e.g. a git commit SHA).
* **Value scrubbing** (:func:`scrub_url_credentials`) — masks ``userinfo@`` credentials
  embedded in URL-shaped strings regardless of their key name. This closes the specific
  leak of a credentialed clone URL (e.g. ``https://x-access-token:ghp_x@github.com/org/repo``)
  riding in a byoc/BYOS step config or ``step_uri`` under an innocuous key like ``repo``.
"""

import re
from typing import Any

__all__ = [
    "SENSITIVE_KEY_RE",
    "URL_USERINFO_RE",
    "REDACTED",
    "redact_sensitive",
    "scrub_url_credentials",
]

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

# Matches the ``userinfo`` (credential) component of a URL: a scheme, ``://``, then a
# ``user:secret`` pair up to the first ``@``. The embedded ``:`` is required, so only a
# credential *pair* is masked — a bare ``user@``/``git@`` (a username with no password)
# is left intact. Neither side may contain ``/`` or whitespace, so the match stays within
# the authority component and never crosses into the path (``https://github.com/a@b`` is
# left alone). Captures the ``scheme://`` prefix so it can be preserved while the
# credentials are replaced. Works standalone or embedded in a larger command string.
# Note: this does NOT catch a token-as-username with no ``:`` (``https://TOKEN@host``);
# such secrets should be stored under a secret-*named* key so key-name masking covers them.
URL_USERINFO_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.\-]*://)[^/@\s]*:[^/@\s]*@")


def scrub_url_credentials(text: str) -> str:
    """Mask ``userinfo@`` credentials embedded in any URL-shaped substrings.

    Value-level companion to the key-name redaction: catches a secret that rides
    inside a URL *value* (e.g. a credentialed git clone URL) under a key whose name
    does not look sensitive. The scheme, host, and path are preserved so the entry
    stays useful in lineage; only the credential portion is replaced.

    Only a ``user:secret`` pair is masked; a bare ``user@``/``git@`` (username with
    no password) is preserved. A token-as-username with no ``:`` is therefore not
    caught here — rely on a secret-named key for that case.

    :param text: an arbitrary string, possibly containing one or more URLs
        (standalone or embedded in a larger command line).
    :returns: the string with each URL's ``user:secret`` credential pair replaced by
        ``REDACTED`` (``scheme://<redacted>@host/...``); a string with no
        credential-pair URL is returned unchanged.
    """
    return URL_USERINFO_RE.sub(rf"\1{REDACTED}@", text)


def redact_sensitive(value: Any) -> Any:
    """Recursively copy a mapping/list, masking values under secret-looking keys.

    :param value: any value; dicts and lists are copied and recursed into, other
        types are returned unchanged (no mutation of the input).
    :returns: a redacted deep-ish copy where any dict key matching
        ``SENSITIVE_KEY_RE`` has its value replaced with ``REDACTED``; every surviving
        string value has its URL credentials scrubbed via
        :func:`scrub_url_credentials`; nested dicts and lists are redacted in turn.
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
    if isinstance(value, str):
        return scrub_url_credentials(value)
    return value
