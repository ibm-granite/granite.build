#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

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
