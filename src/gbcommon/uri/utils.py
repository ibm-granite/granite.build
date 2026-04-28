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

"""Helper functions for URIs"""

from gbcommon.uri.uri import URI
from gbserver.types.artifact import ArtifactType


def get_artifact_type(uri: str) -> ArtifactType:
    """Return the artifact type for the given URI string.

    Delegates to URI.get_artifact_type(), which is overridden by LhURI and HfURI.
    Returns UNDEFINED if the URI scheme is unrecognised or the type cannot be determined.

    Args:
        uri (str): URI string to inspect.

    Returns:
        ArtifactType: The artifact type, or UNDEFINED if not determinable.
    """
    return URI.get_uri(uri).get_artifact_type()
