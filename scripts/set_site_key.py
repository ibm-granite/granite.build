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

"""Set one top-level key of a flat YAML site file, changing no other byte.

    set_site_key.py check|apply SITE KEY VALUE

The key's line is replaced in place (a trailing `# comment` on it is kept), or
the line is appended when the key is absent. SITE is followed through symlinks:
the real file is edited, in its own directory, so a link into
GB_HOME/local-env from a private repo stays a link.

Refused, with nothing written:

- the key's existing value spans more than one line (a block scalar, a block
  collection, or a continuation line), which a one-line edit would corrupt;
- the key appears more than once at top level;
- the edited text does not parse with yaml.safe_load, or parses to a different
  value for the key than VALUE.

`check` validates only, so a caller can refuse before writing anything else;
`apply` validates again and writes atomically, keeping the file's mode.

Exit codes: 0 done (or nothing to change); 1 bad arguments or I/O error; 2
refused.
"""

import os
import re
import sys
import tempfile

import yaml


class Refused(Exception):
    """The edit would not be safe; the message says why."""


def _comment(rest: bytes) -> bytes:
    """The trailing ` # comment` of a one-line value, with its leading space,
    or b"". A `#` inside quotes is not a comment, and one must follow a space."""
    quote = None
    i = 0
    while i < len(rest):
        c = rest[i : i + 1]
        if quote:
            if c == b"\\" and quote == b'"':
                i += 2
                continue
            if c == quote:
                if quote == b"'" and rest[i + 1 : i + 2] == b"'":
                    i += 2
                    continue
                quote = None
        elif c in (b"'", b'"'):
            quote = c
        elif c == b"#" and (i == 0 or rest[i - 1 : i] in (b" ", b"\t")):
            j = i
            while j > 0 and rest[j - 1 : j] in (b" ", b"\t"):
                j -= 1
            return rest[j:]
        i += 1
    return b""


def edit(data: bytes, key: str, value: str) -> bytes:
    """data with KEY set to VALUE; raises Refused."""
    # VALUE goes in single quotes; a quote in it is doubled.
    quoted = "'" + value.replace("'", "''") + "'"
    line = f"{key}: {quoted}".encode()
    pattern = re.compile(rb"^" + re.escape(key.encode()) + rb"[ \t]*:(?=[ \t\r\n]|$)")
    lines = data.splitlines(keepends=True)
    hits = [i for i, old in enumerate(lines) if pattern.match(old)]
    if len(hits) > 1:
        raise Refused(
            f"{key} appears more than once at top level (lines "
            f"{', '.join(str(i + 1) for i in hits)}); remove all but one"
        )
    if hits:
        (i,) = hits
        old = lines[i]
        body = old.rstrip(b"\r\n")
        rest = body[pattern.match(body).end() :]
        comment = _comment(rest)
        scalar = rest[: len(rest) - len(comment)].strip()
        # Nothing after the colon, or a block scalar indicator: the value is on
        # the lines below. Otherwise the next indented line that is not blank
        # or a comment continues it.
        multiline = scalar == b"" or re.fullmatch(rb"[|>][-+0-9]*", scalar)
        for nxt in lines[i + 1 :]:
            if not nxt.strip() or nxt.lstrip().startswith(b"#"):
                continue
            multiline = multiline or nxt[:1] in (b" ", b"\t")
            break
        if multiline:
            raise Refused(
                f"{key} has a multi-line value (line {i + 1}); make it one line "
                "and re-run"
            )
        lines[i] = line + comment + old[len(body) :]
    else:
        if lines and not lines[-1].endswith((b"\n", b"\r")):
            lines[-1] += b"\n"
        lines.append(line + b"\n")
    new = b"".join(lines)
    try:
        parsed = yaml.safe_load(new)
    except yaml.YAMLError as e:
        raise Refused(f"setting {key} would leave the file unparseable: {e}") from e
    if not isinstance(parsed, dict) or parsed.get(key) != value:
        raise Refused(
            f"setting {key} by a one-line edit does not give it the new value; "
            f"make its value a single line and re-run"
        )
    return new


def write_atomic(path: str, data: bytes) -> None:
    """Replace path with data via a temporary file beside it, keeping its mode."""
    mode = os.stat(path).st_mode & 0o7777
    fd, tmp = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(path)
    )
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def main(argv: list) -> int:
    if len(argv) != 4 or argv[0] not in ("check", "apply"):
        print("usage: set_site_key.py check|apply SITE KEY VALUE", file=sys.stderr)
        return 1
    mode, site, key, value = argv
    path = os.path.realpath(site)
    try:
        with open(path, "rb") as f:
            data = f.read()
        new = edit(data, key, value)
        if mode == "apply" and new != data:
            write_atomic(path, new)
    except Refused as e:
        print(f"{site}: {e}; nothing written", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"{site}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
