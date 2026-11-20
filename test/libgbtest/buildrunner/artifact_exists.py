# SPDX-License-Identifier: Apache-2.0

"""Check a build's artifacts against the store their target environment used.

A bare ``URI.exists()`` can't check store-backed asset URIs reliably: an
``s3://local/...`` URI's endpoint, real bucket and credential names live on the
assetstore, not the URI, and ``hf://`` honors only the default token name. So
the build-test harness routes each artifact the way the build did: through the
assetstores its target's ``environment.yaml`` declares, selected with the same
longest-prefix rule pull/push dispatch uses (``asset_uri_exists``).
"""

import logging
from typing import Any, Dict, Mapping, Optional, Self

from gbcommon.uri.git import GitURI
from gbcommon.uri.uri import URI
from gbserver.environment.environment import Environment, asset_uri_exists

logger = logging.getLogger(__name__)


class ArtifactExistenceChecker:
    """Answer "does this artifact exist?" via its build's target environment.

    Create one per verification pass: it caches the declared stores per
    environment URI and sets up each build's space (base URIs and secrets) on
    the current thread, which ``space://`` store URIs resolve against.
    """

    def __init__(self: Self, storage: Any) -> None:
        """
        :param storage: the test's storage (artifact, target, build and space
            registries).
        """
        self._storage = storage
        self._stores: Dict[str, Mapping[Any, Any]] = {}
        self._space_uri: Optional[str] = None

    def exists(self: Self, artifact: Any, uri: URI) -> bool:
        """Return whether ``artifact`` (parsed as ``uri``) exists in its store.

        Falls back to ``uri.exists()`` (logged) when the artifact's target or
        environment can't be resolved.

        :param artifact: a stored artifact record.
        :param uri: the artifact's URI object.
        :returns: whether it exists.
        """
        try:
            stores = self._stores_for(artifact)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                "could not resolve the stores for %s; using uri.exists(): %s",
                artifact.uri,
                e,
            )
            return uri.exists()
        return asset_uri_exists(stores, uri)

    def _stores_for(self: Self, artifact: Any) -> Mapping[Any, Any]:
        """The assetstores declared by the environment of ``artifact``'s target.

        :param artifact: a stored artifact record.
        :returns: store -> environment.yaml entry.
        :raises Exception: if the target, build, space or environment can't load.
        """
        target = self._storage.target_storage.get_by_uuid(artifact.created_by_target_id)
        build = self._storage.build_storage.get_by_uuid(artifact.created_by_build_id)
        self._use_space(build.space_name, build.username)
        if target.environment_uri not in self._stores:
            self._stores[target.environment_uri] = (
                Environment.load_declared_assetstores(target.environment_uri)
            )
        return self._stores[target.environment_uri]

    def _use_space(self: Self, space_name: str, username: Optional[str]) -> None:
        """Set the build's space up on this thread (once per space).

        :param space_name: the build's space name.
        :param username: the build's user, for per-user secrets.
        """
        from gbserver.build.space import (  # pylint: disable=import-outside-toplevel
            Space,
        )

        space = self._storage.space_storage.get_by_name(name=space_name)
        space_uri = GitURI.get_gb_space_config_uri(uri=space.git_repo_uri)
        if space_uri != self._space_uri:
            Space(space_uri, username=username)
            self._space_uri = space_uri
