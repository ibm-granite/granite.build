# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the build-test harness's ArtifactExistenceChecker."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from libgbtest.buildrunner.artifact_exists import ArtifactExistenceChecker

from gbcommon.uri.uri import URI

ENV_URI = "file:///spaces/x/environments/skypilot/slurm/local"
CHECKER = "libgbtest.buildrunner.artifact_exists"


def _storage():
    """Fake storage holding one build, its target run and its file:// space."""
    storage = MagicMock()
    storage.target_storage.get_by_uuid.return_value = SimpleNamespace(
        environment_uri=ENV_URI
    )
    storage.build_storage.get_by_uuid.return_value = SimpleNamespace(
        space_name="public", username="u"
    )
    storage.space_storage.get_by_name.return_value = SimpleNamespace(
        git_repo_uri="file:///spaces/x"
    )
    return storage


def _artifact(uri: str = "s3://local/out/o.txt"):
    return SimpleNamespace(uri=uri, created_by_target_id="t1", created_by_build_id="b1")


class TestArtifactExistenceChecker:
    def test_checks_via_the_target_environment_stores(self):
        """The artifact is checked by the store its target's env routes it to,
        with the build's space set up first; stores load once per env."""
        store = MagicMock()
        store.can_handle_len.return_value = 10
        store.uri_exists.return_value = True
        with (
            patch(f"{CHECKER}.Environment.load_declared_assetstores") as load,
            patch("gbserver.build.space.Space") as space,
        ):
            load.return_value = {store: "entry"}
            checker = ArtifactExistenceChecker(_storage())
            uri = URI.get_uri("s3://local/out/o.txt")
            assert checker.exists(_artifact(), uri)
            assert checker.exists(_artifact(), uri)
        load.assert_called_once_with(ENV_URI)
        space.assert_called_once_with("file:///spaces/x", username="u")
        assert store.uri_exists.call_count == 2

    def test_falls_back_to_uri_exists_when_unresolvable(self):
        """If the target/env can't be resolved, the URI checks itself."""
        storage = _storage()
        storage.target_storage.get_by_uuid.side_effect = KeyError("no target")
        uri = MagicMock()
        uri.exists.return_value = True
        assert ArtifactExistenceChecker(storage).exists(_artifact(), uri)
        uri.exists.assert_called_once_with()
