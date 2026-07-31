import tempfile
import threading
from pathlib import Path

import pytest

from gbserver.buildrunner.buildrunner import BuildRunner
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.status import Status


def _make_stored_build_with_config(
    build_config_yaml: str,
    status: Status = Status.FAILED,
    retry_count: int = 0,
    retry_of_build_id: str = None,
) -> StoredBuild:
    """Create a StoredBuild whose build_archive encodes the given build.yaml content."""
    with tempfile.TemporaryDirectory() as tmp:
        build_dir = Path(tmp)
        (build_dir / "build.yaml").write_text(build_config_yaml)
        build = StoredBuild.create(
            name="test-build",
            space_name="test-space",
            source_uri="",
            username="test-user",
            build_yaml_path=build_dir / "build.yaml",
            status=status,
        )
        build.retry_count = retry_count
        build.retry_of_build_id = retry_of_build_id
        return build


_BUILD_YAML_NO_RETRY = """\
llm.build:
  name: test
  targets:
    mytarget:
      environment_uri: space://environments/cpu
      steps:
        - step_uri: space://steps/download
"""

_BUILD_YAML_MAX_RETRIES_2 = """\
llm.build:
  name: test
  retries:
    max_retries: 2
  targets:
    mytarget:
      environment_uri: space://environments/cpu
      steps:
        - step_uri: space://steps/download
"""


class _FakeBuildStorage:
    """Minimal in-memory build storage for exercising __prepare_retry."""

    def __init__(self, build: StoredBuild):
        self._by_uuid = {build.uuid: build}
        self.added: list[StoredBuild] = []

    def get_by_uuid(self, uuid):
        return self._by_uuid.get(uuid)

    def add(self, build):
        self._by_uuid[build.uuid] = build
        self.added.append(build)

    def update(self, build):
        self._by_uuid[build.uuid] = build


class _FakeStorage:
    def __init__(self, build: StoredBuild):
        self.build_storage = _FakeBuildStorage(build)


class TestShouldRetry:
    """Unit tests for the BuildRunner._should_retry() instance method."""

    def _runner(self) -> BuildRunner:
        """Create a minimal BuildRunner instance without triggering __init__."""
        return object.__new__(BuildRunner)

    def test_non_failed_build_not_retried(self):
        runner = self._runner()
        for st in (
            Status.SUCCESS,
            Status.PENDING,
            Status.RUNNING,
            Status.CANCELLED,
            Status.INVALID,
        ):
            build = _make_stored_build_with_config(_BUILD_YAML_MAX_RETRIES_2, status=st)
            assert not runner._should_retry(build), f"Expected no retry for status {st}"

    def test_failed_no_max_retries(self):
        build = _make_stored_build_with_config(
            _BUILD_YAML_NO_RETRY, status=Status.FAILED
        )
        assert not self._runner()._should_retry(build)

    def test_failed_with_max_retries_first_attempt(self):
        build = _make_stored_build_with_config(
            _BUILD_YAML_MAX_RETRIES_2, status=Status.FAILED, retry_count=0
        )
        assert self._runner()._should_retry(build)

    def test_failed_with_max_retries_second_attempt(self):
        build = _make_stored_build_with_config(
            _BUILD_YAML_MAX_RETRIES_2, status=Status.FAILED, retry_count=1
        )
        assert self._runner()._should_retry(build)

    def test_failed_exhausted_retries(self):
        build = _make_stored_build_with_config(
            _BUILD_YAML_MAX_RETRIES_2, status=Status.FAILED, retry_count=2
        )
        assert not self._runner()._should_retry(build)

    def test_retry_build_preserves_lineage(self):
        """The retry_of_build_id field should point to the original build UUID."""
        original_id = "original-build-uuid"
        build = _make_stored_build_with_config(
            _BUILD_YAML_MAX_RETRIES_2,
            status=Status.FAILED,
            retry_count=1,
            retry_of_build_id=original_id,
        )
        assert build.retry_of_build_id == original_id
        assert build.retry_count == 1
        assert self._runner()._should_retry(build)

    def test_retry_copies_owner_and_space_from_the_original(self):
        """The status and lineage endpoints authorize a whole retry chain by
        checking only the queried build. That is safe only because every retry
        shares the original's owner and space. If __prepare_retry ever stops
        copying space_name/username, this must fail rather than the shortcut
        silently becoming an access-control hole.
        """
        original = _make_stored_build_with_config(
            _BUILD_YAML_MAX_RETRIES_2, status=Status.FAILED, retry_count=0
        )
        runner = self._runner()
        runner.storage = _FakeStorage(original)
        runner.stored_build = original
        runner._retry_chain_lock = threading.Lock()
        runner._retry_chain_build_ids = []

        retry = runner._BuildRunner__prepare_retry()

        assert retry is not None
        assert retry.space_name == original.space_name
        assert retry.username == original.username
        # And it is a genuine new chain member, not the original returned back.
        assert retry.uuid != original.uuid
        assert retry.retry_of_build_id == original.uuid
