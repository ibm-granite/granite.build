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

"""Unit tests for the shared SSH private-key writer."""

import pytest

from gbserver.utils.ssh_keys import write_private_key_file


class TestWritePrivateKeyFile:
    def test_writes_0600_with_normalized_newline(self, tmp_path):
        dest = tmp_path / "nested" / "id_key"
        returned = write_private_key_file("-----KEY-----\nbody", dest)
        assert returned == dest
        # Parent dir created, single trailing newline, mode 0600.
        assert dest.read_text(encoding="utf-8") == "-----KEY-----\nbody\n"
        assert (dest.stat().st_mode & 0o777) == 0o600

    def test_strips_extra_trailing_newlines(self, tmp_path):
        dest = tmp_path / "id_key"
        write_private_key_file("body\n\n\n", dest)
        assert dest.read_text(encoding="utf-8") == "body\n"

    def test_overwrites_existing(self, tmp_path):
        dest = tmp_path / "id_key"
        write_private_key_file("first", dest)
        write_private_key_file("second", dest)
        assert dest.read_text(encoding="utf-8") == "second\n"

    @pytest.mark.parametrize("bad", ["", "   ", "\n\n"])
    def test_empty_contents_raise(self, tmp_path, bad):
        with pytest.raises(ValueError):
            write_private_key_file(bad, tmp_path / "id_key")
