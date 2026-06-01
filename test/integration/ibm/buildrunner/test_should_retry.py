import tempfile
from pathlib import Path

import pytest

from gbserver.buildrunner.buildrunner import BuildRunner
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.status import Status

pytestmark = pytest.mark.ibm


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
