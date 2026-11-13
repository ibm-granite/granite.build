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

import pytest

from gbcommon.uri.lh import DEFAULT_FILESET_VERSION, DEFAULT_MODEL_REVISION
from gbserver.buildrunner.buildrunner import BuildRunner


class TestBuildRunnerStatics:

    def test_get_normalized_uri(self):
        namespace = "mynamespace"
        table_name = "mytable"
        name = "myfileset"
        rev = "1"

        uri = f"lh://anything/{namespace}/filesets/{table_name}/{name}"
        new_uri = BuildRunner._get_normalized_uri(uri)
        assert new_uri.endswith(DEFAULT_FILESET_VERSION)
        uri = f"lh://anything/{namespace}/filesets/{table_name}/{name}/{rev}"
        new_uri = BuildRunner._get_normalized_uri(uri)
        assert new_uri.endswith(rev)

        uri = f"lh://anything/{namespace}/models/{table_name}/{name}"
        new_uri = BuildRunner._get_normalized_uri(uri)
        assert new_uri.endswith(DEFAULT_MODEL_REVISION)
        uri = f"lh://anything/{namespace}/models/{table_name}/{name}/{rev}"
        new_uri = BuildRunner._get_normalized_uri(uri)
        assert new_uri.endswith(rev)
