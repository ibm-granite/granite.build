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
For storing and accessing file/folders in a remote environment
"""

from typing import Union

from gbcommon.uri.env import EnvURI
from gbcommon.uri.uri import URI
from gbserver.asset.assetstore import Assetstore


class Envstore(Assetstore):
    """A class for storing and accessing file/folders in a remote environment"""

    def __init__(self, uri: Union[URI, str], **kwargs):
        super().__init__(uri, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def get_supported_uri_classes(self):
        return [EnvURI]
