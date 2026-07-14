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


"""
Reference a file/folder in the remote filesystem.
"""

from pathlib import Path
from typing import Dict, List, Optional, Self
from urllib.parse import ParseResult, urlparse

from gbcommon.uri.uri import URI
from gbserver.types.constants import ENV_URI_SCHEME


class EnvURI(URI):
    """Reference a file/folder in the remote filesystem."""

    def __init__(
        self: Self, uri: ParseResult, context: Optional[str] = None, **kwargs: Dict
    ) -> None:
        self.context = context
        if (
            not Path(uri.path).is_absolute()
            and self.context is not None
            and self.context != ""
        ):
            uri = urlparse(ENV_URI_SCHEME + ":///" + self.context + "/" + uri.path)
        super().__init__(uri=uri, context=context, **kwargs)

    @staticmethod
    def get_supported_schemes() -> List[str]:
        """Return supported uri schemes as list"""
        return [ENV_URI_SCHEME]

    def exists(self: Self, force: bool = False) -> bool:
        return True  # TODO: fix this (ssh for LSF, rclone for s3, etc.)
        # return Path(self.uri.path).exists()

    def is_accessible(self: Self) -> bool:
        return self.exists()

    def pull(self: Self, dest: Path, force: bool = False) -> bool:
        return True  # TODO: fix this (scp for LSF, rclone for s3, etc.)

    def delete(self: Self) -> bool:
        raise NotImplementedError("EnvURI delete is not implemented")
        # return sync_or_copy(self.uri.path, dest)

    def custom_str(self: Self) -> str:
        """
        Custom stringification.
        Need this to get the triple slash
        """
        if not self.uri:
            return ""
        t1 = self.uri.geturl()
        t2 = t1.removeprefix(ENV_URI_SCHEME + ":")
        t3 = Path("/////" + t2)
        t4 = Path(self.context) / t3 if self.context else t3
        return ENV_URI_SCHEME + "://" + str(t4)


def is_relative_env_uri(uri_str: str) -> bool:
    """Return True if ``uri_str`` is a relative ``env:`` URI.

    Unlike ``file:``, an ``env:`` URI moves no data — it is a bare reference to a
    path the environment can already reach (its ``exists()``/``pull()`` are no-op
    stubs). A relative path therefore has no defined resolution root and is
    rejected; authors must use an absolute ``env:///…`` (or rely on a build
    ``context``). An ``env:`` URI is considered relative iff the text after the
    scheme does not start with ``/`` (e.g. ``env:outputs/x`` is relative, while
    ``env:///abs``, ``env://host/path`` and the templated ``env://{{ binding.path }}``
    are not). The prefix test is deliberately string-based so it also flags a
    relative literal that embeds a Jinja template (e.g.
    ``env:outputs/{{ binding.path | short_hash }}``) at config-load time, before
    the template is filled.

    :param uri_str: The raw URI string from a build config input/output.
    :returns: True if the string is a relative ``env:`` URI; False otherwise
        (including non-``env:`` schemes and absolute ``env:`` URIs).
    """
    prefix = ENV_URI_SCHEME + ":"
    return uri_str.startswith(prefix) and not uri_str.startswith(prefix + "/")
