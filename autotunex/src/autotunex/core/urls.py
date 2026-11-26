# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""URL helpers shared by settings validation and the auth router."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import SplitResult, urlsplit

# RFC 3986 reg-name (unreserved / sub-delims), checked after lowercasing.
# Percent-escapes are left out: a browser decodes them before it serializes
# an origin, so a host spelled with one never matches.
_REG_NAME = re.compile(r"[a-z0-9\-._~!$&'()*+,;=]+")


def split_origin(url: str) -> tuple[str, SplitResult] | None:
    """Return ``url``'s origin as a browser serializes it, plus the parsed URL.

    The origin is ``http(s)://host[:port]`` with the host lowercased, an IPv6
    host compressed (``[0:0::1]`` becomes ``[::1]``) and a default port
    dropped, which is the form of a browser's ``Origin`` header. The parsed
    URL is returned too, so a caller redirects to the URL it validated rather
    than parsing it again.

    ``None`` covers anything that is not an absolute http(s) URL with a host,
    a URL that does not parse (``https://[``, a port over 65535), a URL
    carrying userinfo, which a browser never puts in an origin, and a host
    outside the reg-name or IPv6 forms (a space, a backslash, a
    percent-escape), which no browser origin can equal.
    """
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    host = parts.hostname
    if parts.scheme not in ("http", "https") or not host:
        return None
    if parts.username is not None or parts.password is not None:
        return None
    if parts.netloc.startswith("["):
        # An IP literal (userinfo is already ruled out, so the netloc starts
        # with the host). `%` would be an IPv6 zone id, which browsers do not
        # accept, and `[v1.fe]`-style IPvFuture fails to parse here too.
        if "%" in host:
            return None
        try:
            address = ipaddress.IPv6Address(host)
        except ValueError:
            return None
        # Python 3.13+ writes an IPv4-mapped address as `::ffff:127.0.0.1`;
        # a browser writes `::ffff:7f00:1`.
        if address.ipv4_mapped is not None:
            high, low = divmod(int(address) & 0xFFFFFFFF, 0x10000)
            host = f"[::ffff:{high:x}:{low:x}]"
        else:
            host = f"[{address.compressed}]"
    elif not _REG_NAME.fullmatch(host):
        return None
    origin = f"{parts.scheme}://{host}"
    if port is not None and port != (443 if parts.scheme == "https" else 80):
        origin += f":{port}"
    return origin, parts
