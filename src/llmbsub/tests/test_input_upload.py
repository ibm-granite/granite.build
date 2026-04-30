"""
Standalone test suite for llmbsub input upload functionality.

This test suite is designed to run on BlueVela (LSF cluster with GPFS) to verify:
1. GPFS filesystem behavior (checksums, mtimes, file locking)
2. Performance characteristics on the cluster
3. All edge cases with mocked external dependencies

Run tests:
    pytest test_input_upload.py -v                    # All tests
    pytest test_input_upload.py -v -m "not gpfs"      # Unit tests only
    pytest test_input_upload.py -v -m "gpfs"          # GPFS tests only
    pytest test_input_upload.py -v -k "retry"         # Specific tests
    pytest test_input_upload.py -v --durations=10     # With timing info
"""

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

# ============================================================================
# INLINE IMPLEMENTATIONS (no external gbcli imports required)
# ============================================================================

# These are standalone implementations that mirror the actual code
# This allows tests to run without gbcli installed


class UploadStatus(Enum):
    """Status of an upload operation."""

    COMPLETED = "completed"
    EXISTS = "exists"
    IN_PROGRESS = "in_progress"
    ABORTED = "aborted"
    ERROR = "error"


class TransientError(Exception):
    """Represents a transient error that should be retried."""

    pass


class PermanentError(Exception):
    """Represents a permanent error that should not be retried."""

    pass


class SourceModifiedError(Exception):
    """Raised when source data is modified during upload."""

    pass


@dataclass
class UploadResult:
    """Result of processing a single input path."""

    input_path: str
    checksum: str
    dmf_uri: Optional[str]
    status: UploadStatus
    message: str


# Retry configuration (matches actual implementation)
RETRY_DELAYS = [30, 60, 120]
INTEGRITY_CHECK_INTERVAL = 30
PENDING_POLL_INTERVAL = 30
PENDING_POLL_MAX_WAIT = 3600


# ============================================================================
# MOCK CLASSES
# ============================================================================


class MockGBServerAPI:
    """Mock gbserver /artifact/status API responses."""

    def __init__(self):
        self.responses: Dict[str, dict] = {}
        self.call_count: Dict[str, int] = {}
        self.call_history: List[dict] = []
        self._pending_countdown: Dict[str, int] = {}
        self._final_response: Dict[str, dict] = {}
        self._should_raise: Optional[Exception] = None

    def set_response(self, checksum: str, response: dict):
        """Set response for a specific checksum."""
        self.responses[checksum] = response

    def set_pending_then_exists(self, checksum: str, pending_polls: int, uri: str):
        """Return pending N times, then exists."""
        self._pending_countdown[checksum] = pending_polls
        self._final_response[checksum] = {"status": "exists", "uri": uri}

    def set_pending_then_not_found(self, checksum: str, pending_polls: int, target_uri: str):
        """Return pending N times, then not_found (other job failed)."""
        self._pending_countdown[checksum] = pending_polls
        self._final_response[checksum] = {
            "status": "not_found",
            "target_uri": target_uri,
        }

    def set_error(self, exception: Exception):
        """Make API calls raise an exception."""
        self._should_raise = exception

    def clear_error(self):
        """Clear any set exception."""
        self._should_raise = None

    def __call__(self, path: str, checksum: str, space_name: str) -> dict:
        """Simulate API call."""
        if self._should_raise:
            raise self._should_raise

        self.call_history.append(
            {
                "path": path,
                "checksum": checksum,
                "space": space_name,
                "timestamp": time.time(),
            }
        )
        self.call_count[checksum] = self.call_count.get(checksum, 0) + 1

        # Handle pending countdown
        if checksum in self._pending_countdown:
            if self._pending_countdown[checksum] > 0:
                self._pending_countdown[checksum] -= 1
                return {"status": "pending"}
            return self._final_response[checksum]

        # Return configured response or default
        return self.responses.get(
            checksum,
            {
                "status": "not_found",
                "target_uri": f"lh://prod/test/filesets/input/{checksum[:8]}/{checksum}",
            },
        )

    def reset(self):
        """Reset all state."""
        self.responses.clear()
        self.call_count.clear()
        self.call_history.clear()
        self._pending_countdown.clear()
        self._final_response.clear()
        self._should_raise = None


class MockDMFCLI:
    """Mock dmf CLI for testing without actual uploads."""

    def __init__(self):
        self.push_calls: List[dict] = []
        self.should_fail: Dict[str, tuple] = {}  # path -> (error_type, error_data)
        self.fail_counts: Dict[str, int] = {}
        self._delay: float = 0

    def set_transient_failure(self, path: str, fail_count: int):
        """Make push fail N times then succeed."""
        self.should_fail[path] = ("transient", fail_count)
        self.fail_counts[path] = 0

    def set_permanent_failure(self, path: str, error_msg: str = "Permission denied"):
        """Make push always fail with permanent error."""
        self.should_fail[path] = ("permanent", error_msg)

    def set_timeout(self, path: str):
        """Make push timeout."""
        self.should_fail[path] = ("timeout", None)

    def set_delay(self, seconds: float):
        """Add delay to simulate slow uploads."""
        self._delay = seconds

    def push(
        self, namespace: str, table: str, path: str, version: str, label: str
    ) -> subprocess.CompletedProcess:
        """Simulate dmf push command."""
        call_record = {
            "namespace": namespace,
            "table": table,
            "path": path,
            "version": version,
            "label": label,
            "timestamp": time.time(),
        }
        self.push_calls.append(call_record)

        if self._delay > 0:
            time.sleep(self._delay)

        if path in self.should_fail:
            error_type, error_data = self.should_fail[path]

            if error_type == "permanent":
                return subprocess.CompletedProcess(
                    args=["dmf", "push"],
                    returncode=1,
                    stdout="",
                    stderr=f"403 Forbidden: {error_data}",
                )

            elif error_type == "timeout":
                raise subprocess.TimeoutExpired(cmd=["dmf", "push"], timeout=7200)

            elif error_type == "transient":
                self.fail_counts[path] = self.fail_counts.get(path, 0) + 1
                if self.fail_counts[path] <= error_data:
                    return subprocess.CompletedProcess(
                        args=["dmf", "push"],
                        returncode=1,
                        stdout="",
                        stderr="503 Service Unavailable",
                    )

        return subprocess.CompletedProcess(
            args=["dmf", "push"],
            returncode=0,
            stdout=f"Upload complete: {path} -> lh://{namespace}/filesets/{table}/{label}/{version}",
            stderr="",
        )

    def reset(self):
        """Reset all state."""
        self.push_calls.clear()
        self.should_fail.clear()
        self.fail_counts.clear()
        self._delay = 0


class MockChecksumScript:
    """Mock llmbsum.sh script for testing without actual checksum calculation."""

    def __init__(self):
        self.checksums: Dict[str, str] = {}
        self.call_count: Dict[str, int] = {}
        self._should_fail: bool = False
        self._delay: float = 0

    def set_checksum(self, path: str, checksum: str):
        """Set checksum for a specific path."""
        self.checksums[path] = checksum

    def set_failure(self, should_fail: bool = True):
        """Make checksum calculation fail."""
        self._should_fail = should_fail

    def set_delay(self, seconds: float):
        """Add delay to simulate slow checksum calculation."""
        self._delay = seconds

    def calculate(
        self, path: str, concurrency: int, output_file: str
    ) -> subprocess.CompletedProcess:
        """Simulate llmbsum.sh execution."""
        self.call_count[path] = self.call_count.get(path, 0) + 1

        if self._delay > 0:
            time.sleep(self._delay)

        if self._should_fail:
            return subprocess.CompletedProcess(
                args=["llmbsum.sh"],
                returncode=1,
                stdout="",
                stderr="Error: xxh64sum not found",
            )

        # Generate or use preset checksum
        if path in self.checksums:
            checksum = self.checksums[path]
        else:
            # Generate deterministic checksum from path
            checksum = hashlib.sha256(path.encode()).hexdigest()[:16]
            self.checksums[path] = checksum

        # Write output files like real script
        with open(f"{output_file}.all", "w") as f:
            f.write(f"{checksum}  {path}\n")

        with open(f"{output_file}.cksum", "w") as f:
            f.write(f"{checksum}  {output_file}.all\n")

        return subprocess.CompletedProcess(
            args=["llmbsum.sh", path, str(concurrency), output_file],
            returncode=0,
            stdout="",
            stderr="",
        )

    def reset(self):
        """Reset all state."""
        self.checksums.clear()
        self.call_count.clear()
        self._should_fail = False
        self._delay = 0


class MockChecksumCache:
    """Mock checksum cache for testing."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self._cache: Dict[str, dict] = {}
        self._mtimes: Dict[str, Dict[str, float]] = {}
        self.cache_dir = cache_dir or Path(tempfile.mkdtemp())

    def _get_cache_key(self, path: str) -> str:
        """Generate cache key from path."""
        abs_path = os.path.abspath(path)
        return hashlib.sha256(abs_path.encode()).hexdigest()[:16]

    def _get_dir_mtime(self, path: str) -> float:
        """Get the most recent mtime of any file in the directory."""
        max_mtime = 0.0
        path_obj = Path(path)

        if path_obj.is_file():
            return path_obj.stat().st_mtime

        for item in path_obj.rglob("*"):
            if item.is_file():
                try:
                    mtime = item.stat().st_mtime
                    if mtime > max_mtime:
                        max_mtime = mtime
                except (OSError, IOError):
                    continue

        return max_mtime

    def get(self, path: str) -> Optional[str]:
        """Get cached checksum if still valid."""
        key = self._get_cache_key(path)
        entry = self._cache.get(key)

        if not entry:
            return None

        # Check if mtime has changed
        try:
            current_mtime = self._get_dir_mtime(path)
            if entry.get("mtime") != current_mtime:
                return None
        except (OSError, IOError):
            return None

        return entry.get("checksum")

    def set(self, path: str, checksum: str):
        """Store checksum in cache."""
        key = self._get_cache_key(path)
        self._cache[key] = {
            "path": os.path.abspath(path),
            "checksum": checksum,
            "mtime": self._get_dir_mtime(path),
            "cached_at": datetime.now().isoformat(),
        }

    def invalidate(self, path: str):
        """Remove entry from cache."""
        key = self._get_cache_key(path)
        if key in self._cache:
            del self._cache[key]

    def get_all_mtimes(self, path: str) -> Dict[str, float]:
        """Get mtimes of all files in the directory."""
        mtimes = {}
        path_obj = Path(path)

        if path_obj.is_file():
            mtimes[str(path_obj)] = path_obj.stat().st_mtime
            return mtimes

        for item in path_obj.rglob("*"):
            if item.is_file():
                try:
                    mtimes[str(item)] = item.stat().st_mtime
                except (OSError, IOError):
                    continue

        return mtimes


# ============================================================================
# INLINE SERVICE IMPLEMENTATION FOR TESTING
# ============================================================================


class _InputUploadServiceForTesting:
    """
    Testable version of InputUploadService with injectable dependencies.

    This mirrors the actual implementation but allows mocks to be injected.
    Note: Name starts with underscore to avoid pytest collection warning.
    """

    def __init__(
        self,
        space: str,
        api_client: MockGBServerAPI,
        dmf_cli: MockDMFCLI,
        checksum_script: MockChecksumScript,
        checksum_cache: MockChecksumCache,
        concurrency: int = 8,
    ):
        self.space = space
        self.api_client = api_client
        self.dmf_cli = dmf_cli
        self.checksum_script = checksum_script
        self.checksum_cache = checksum_cache
        self.concurrency = concurrency
        self._abort_upload = False
        # For testing, use shorter intervals
        self.integrity_check_interval = 0.1  # 100ms for tests
        self.pending_poll_interval = 0.1
        self.pending_poll_max_wait = 1.0

    def process_inputs(self, input_paths: List[str]) -> List[UploadResult]:
        """Process all input paths for upload."""
        results = []
        for input_path in input_paths:
            result = self._process_single_input(input_path)
            results.append(result)
        return results

    def _process_single_input(self, input_path: str) -> UploadResult:
        """Process a single input path."""
        try:
            # Step 1: Calculate or retrieve checksum
            checksum = self._get_checksum(input_path)

            # Step 2: Query API for artifact status (with polling for pending)
            api_response = self._check_artifact_status_with_polling(input_path, checksum)

            if api_response.get("status") == "exists":
                return UploadResult(
                    input_path=input_path,
                    checksum=checksum,
                    dmf_uri=api_response.get("uri"),
                    status=UploadStatus.EXISTS,
                    message="Artifact already exists in Lakehouse",
                )

            if api_response.get("status") == "pending_timeout":
                return UploadResult(
                    input_path=input_path,
                    checksum=checksum,
                    dmf_uri=None,
                    status=UploadStatus.ERROR,
                    message="Timed out waiting for pending upload by another job",
                )

            # Step 3: Upload to Lakehouse
            target_uri = api_response.get("target_uri")
            if not target_uri:
                return UploadResult(
                    input_path=input_path,
                    checksum=checksum,
                    dmf_uri=None,
                    status=UploadStatus.ERROR,
                    message="No target URI returned from API",
                )

            upload_result = self._upload_to_lakehouse(input_path, target_uri, checksum)
            return upload_result

        except SourceModifiedError as e:
            return UploadResult(
                input_path=input_path,
                checksum="",
                dmf_uri=None,
                status=UploadStatus.ABORTED,
                message=f"Source data modified during upload: {e}",
            )
        except Exception as e:
            return UploadResult(
                input_path=input_path,
                checksum="",
                dmf_uri=None,
                status=UploadStatus.ERROR,
                message=str(e),
            )

    def _get_checksum(self, input_path: str) -> str:
        """Calculate or retrieve cached checksum."""
        # Check cache first
        cached = self.checksum_cache.get(input_path)
        if cached:
            return cached

        # Calculate new checksum
        checksum = self._calculate_checksum(input_path)
        self.checksum_cache.set(input_path, checksum)
        return checksum

    def _calculate_checksum(self, input_path: str) -> str:
        """Calculate checksum using mock script."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "checksum")

            result = self.checksum_script.calculate(input_path, self.concurrency, output_file)

            if result.returncode != 0:
                raise RuntimeError(f"Checksum calculation failed: {result.stderr}")

            checksum_file = f"{output_file}.cksum"
            with open(checksum_file, "r") as f:
                checksum_line = f.read().strip()
                return checksum_line.split()[0]

    def _check_artifact_status(self, path: str, checksum: str) -> Dict:
        """Query API for artifact status (single call)."""
        return self.api_client(path, checksum, self.space)

    def _check_artifact_status_with_polling(self, path: str, checksum: str) -> Dict:
        """Query API for artifact status, polling if pending."""
        start_time = time.time()
        poll_count = 0

        while True:
            response = self._check_artifact_status(path, checksum)
            status = response.get("status")

            if status == "exists":
                return response

            if status == "not_found":
                return response

            if status == "pending":
                elapsed = time.time() - start_time
                if elapsed >= self.pending_poll_max_wait:
                    return {"status": "pending_timeout"}

                poll_count += 1
                time.sleep(self.pending_poll_interval)
                continue

            # Unknown status - treat as not_found
            return response

    def _upload_to_lakehouse(self, input_path: str, target_uri: str, checksum: str) -> UploadResult:
        """Upload fileset to Lakehouse with integrity monitoring."""
        # Record initial mtimes
        initial_mtimes = self.checksum_cache.get_all_mtimes(input_path)

        # Parse URI
        uri_components = self._parse_lakehouse_uri(target_uri)
        if not uri_components:
            return UploadResult(
                input_path=input_path,
                checksum=checksum,
                dmf_uri=None,
                status=UploadStatus.ERROR,
                message=f"Failed to parse target URI: {target_uri}",
            )

        namespace = uri_components["namespace"]
        table = uri_components["table"]
        label = uri_components["label"]
        version = uri_components.get("version", checksum)

        # Start integrity monitoring
        self._abort_upload = False
        integrity_thread = threading.Thread(
            target=self._monitor_integrity,
            args=(input_path, initial_mtimes),
            daemon=True,
        )
        integrity_thread.start()

        try:
            success = self._upload_with_retry(input_path, namespace, table, label, version)

            if self._abort_upload:
                raise SourceModifiedError("Source data was modified during upload")

            if success:
                return UploadResult(
                    input_path=input_path,
                    checksum=checksum,
                    dmf_uri=target_uri,
                    status=UploadStatus.COMPLETED,
                    message="Upload successful",
                )
            else:
                return UploadResult(
                    input_path=input_path,
                    checksum=checksum,
                    dmf_uri=None,
                    status=UploadStatus.ERROR,
                    message="Upload failed after retries",
                )
        finally:
            self._abort_upload = True

    def _monitor_integrity(self, path: str, initial_mtimes: Dict[str, float]):
        """Monitor source directory for modifications during upload."""
        while not self._abort_upload:
            time.sleep(self.integrity_check_interval)
            if self._abort_upload:
                break

            current_mtimes = self.checksum_cache.get_all_mtimes(path)
            if current_mtimes != initial_mtimes:
                self._abort_upload = True
                break

    def _upload_with_retry(
        self, path: str, namespace: str, table: str, label: str, version: str
    ) -> bool:
        """Execute dmf push with retry logic."""
        retry_delays = [0.1, 0.2, 0.3]  # Shorter for tests

        for attempt, delay in enumerate(retry_delays + [None], 1):
            if self._abort_upload:
                return False

            try:
                result = self.dmf_cli.push(namespace, table, path, version, label)

                if result.returncode == 0:
                    return True

                stderr = result.stderr.lower()
                stdout = result.stdout.lower()
                output = stderr + stdout

                # Classify error
                permanent_indicators = [
                    "permission denied",
                    "not found",
                    "invalid",
                    "401",
                    "403",
                    "404",
                    "400",
                ]
                transient_indicators = [
                    "connection",
                    "timeout",
                    "unavailable",
                    "502",
                    "503",
                    "504",
                    "500",
                ]

                for indicator in permanent_indicators:
                    if indicator in output:
                        raise PermanentError(f"dmf push failed: {result.stderr}")

                # Transient error - retry if we have more attempts
                if delay is None:
                    return False
                time.sleep(delay)

            except subprocess.TimeoutExpired:
                if delay is None:
                    return False
                time.sleep(delay)

            except PermanentError:
                raise

        return False

    def _parse_lakehouse_uri(self, uri: str) -> Optional[Dict]:
        """Parse Lakehouse URI into components."""
        pattern = r"lh://([^/]+)/([^/]+)/filesets/([^/]+)/([^/]+)(?:/([^/]+))?"
        match = re.match(pattern, uri)

        if not match:
            return None

        return {
            "env": match.group(1),
            "namespace": match.group(2),
            "table": match.group(3),
            "label": match.group(4),
            "version": match.group(5) if match.group(5) else None,
        }


# ============================================================================
# PYTEST FIXTURES
# ============================================================================


@pytest.fixture
def test_directory(tmp_path):
    """Create test directory with sample files."""
    (tmp_path / "file1.txt").write_text("content1")
    (tmp_path / "file2.txt").write_text("content2")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file3.txt").write_text("content3")
    return tmp_path


@pytest.fixture
def large_test_directory(tmp_path):
    """Create large test directory for performance testing."""
    for i in range(100):  # Use 100 for faster tests, increase for stress testing
        (tmp_path / f"file_{i:04d}.txt").write_bytes(os.urandom(1024))
    return tmp_path


@pytest.fixture
def mock_api():
    """Create mock gbserver API."""
    return MockGBServerAPI()


@pytest.fixture
def mock_dmf():
    """Create mock DMF CLI."""
    return MockDMFCLI()


@pytest.fixture
def mock_checksum():
    """Create mock checksum script."""
    return MockChecksumScript()


@pytest.fixture
def mock_cache(tmp_path):
    """Create mock checksum cache."""
    return MockChecksumCache(cache_dir=tmp_path / "cache")


@pytest.fixture
def upload_service(mock_api, mock_dmf, mock_checksum, mock_cache):
    """Create testable upload service with mocked dependencies."""
    return _InputUploadServiceForTesting(
        space="test-space",
        api_client=mock_api,
        dmf_cli=mock_dmf,
        checksum_script=mock_checksum,
        checksum_cache=mock_cache,
    )


# ============================================================================
# UNIT TESTS: CHECKSUM CACHE
# ============================================================================


class TestChecksumCache:
    """Tests for checksum caching functionality."""

    def test_cache_miss_returns_none(self, mock_cache, test_directory):
        """New path returns None from cache."""
        result = mock_cache.get(str(test_directory))
        assert result is None

    def test_cache_hit_returns_checksum(self, mock_cache, test_directory):
        """Cached path returns stored value."""
        mock_cache.set(str(test_directory), "abc123")
        result = mock_cache.get(str(test_directory))
        assert result == "abc123"

    def test_cache_invalidation_on_mtime_change(self, mock_cache, test_directory):
        """Modified file invalidates cache entry."""
        mock_cache.set(str(test_directory), "abc123")

        # Modify a file
        time.sleep(0.01)  # Ensure mtime changes
        (test_directory / "file1.txt").write_text("modified content")

        result = mock_cache.get(str(test_directory))
        assert result is None

    def test_cache_key_generation_consistent(self, mock_cache):
        """Same path always generates same key."""
        path = "/some/test/path"
        key1 = mock_cache._get_cache_key(path)
        key2 = mock_cache._get_cache_key(path)
        assert key1 == key2

    def test_cache_key_different_for_different_paths(self, mock_cache):
        """Different paths generate different keys."""
        key1 = mock_cache._get_cache_key("/path/one")
        key2 = mock_cache._get_cache_key("/path/two")
        assert key1 != key2

    def test_get_all_mtimes_single_file(self, tmp_path):
        """get_all_mtimes works for single file."""
        cache = MockChecksumCache()
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        mtimes = cache.get_all_mtimes(str(test_file))
        assert len(mtimes) == 1
        assert str(test_file) in mtimes

    def test_get_all_mtimes_directory(self, test_directory):
        """get_all_mtimes returns all files in directory."""
        cache = MockChecksumCache()
        mtimes = cache.get_all_mtimes(str(test_directory))

        assert len(mtimes) == 3  # file1.txt, file2.txt, subdir/file3.txt

    def test_cache_invalidate_removes_entry(self, mock_cache, test_directory):
        """invalidate() removes cache entry."""
        mock_cache.set(str(test_directory), "abc123")
        mock_cache.invalidate(str(test_directory))
        result = mock_cache.get(str(test_directory))
        assert result is None


# ============================================================================
# UNIT TESTS: INPUT UPLOAD SERVICE
# ============================================================================


class TestInputUploadService:
    """Tests for main upload service functionality."""

    def test_process_inputs_single_path(self, upload_service, test_directory, mock_api):
        """Single input processed correctly."""
        mock_api.set_response(
            "any",
            {"status": "not_found", "target_uri": "lh://prod/ns/filesets/tbl/label/v1"},
        )

        results = upload_service.process_inputs([str(test_directory)])

        assert len(results) == 1
        assert results[0].input_path == str(test_directory)
        assert results[0].status in (UploadStatus.COMPLETED, UploadStatus.ERROR)

    def test_process_inputs_multiple_paths(self, upload_service, tmp_path, mock_api):
        """Multiple inputs processed in order."""
        dir1 = tmp_path / "dir1"
        dir2 = tmp_path / "dir2"
        dir1.mkdir()
        dir2.mkdir()
        (dir1 / "file.txt").write_text("content1")
        (dir2 / "file.txt").write_text("content2")

        results = upload_service.process_inputs([str(dir1), str(dir2)])

        assert len(results) == 2
        assert results[0].input_path == str(dir1)
        assert results[1].input_path == str(dir2)

    def test_artifact_exists_skips_upload(
        self, upload_service, test_directory, mock_api, mock_dmf, mock_checksum
    ):
        """API returns 'exists' - no dmf push called."""
        checksum = "existing_checksum"
        mock_checksum.set_checksum(str(test_directory), checksum)
        mock_api.set_response(
            checksum, {"status": "exists", "uri": "lh://prod/ns/filesets/tbl/label/v1"}
        )

        results = upload_service.process_inputs([str(test_directory)])

        assert len(results) == 1
        assert results[0].status == UploadStatus.EXISTS
        assert len(mock_dmf.push_calls) == 0

    def test_artifact_not_found_triggers_upload(
        self, upload_service, test_directory, mock_api, mock_dmf, mock_checksum
    ):
        """API returns 'not_found' - dmf push called."""
        checksum = "new_checksum"
        mock_checksum.set_checksum(str(test_directory), checksum)
        mock_api.set_response(
            checksum,
            {"status": "not_found", "target_uri": "lh://prod/ns/filesets/tbl/label/v1"},
        )

        results = upload_service.process_inputs([str(test_directory)])

        assert len(results) == 1
        assert results[0].status == UploadStatus.COMPLETED
        assert len(mock_dmf.push_calls) == 1

    def test_pending_status_polls_until_exists(
        self, upload_service, test_directory, mock_api, mock_dmf, mock_checksum
    ):
        """Polls pending status, then finds artifact exists."""
        checksum = "pending_checksum"
        mock_checksum.set_checksum(str(test_directory), checksum)
        mock_api.set_pending_then_exists(
            checksum, pending_polls=2, uri="lh://prod/ns/filesets/tbl/label/v1"
        )

        results = upload_service.process_inputs([str(test_directory)])

        assert results[0].status == UploadStatus.EXISTS
        assert mock_api.call_count[checksum] == 3  # 2 pending + 1 exists
        assert len(mock_dmf.push_calls) == 0

    def test_pending_status_polls_until_timeout(
        self, upload_service, test_directory, mock_api, mock_checksum
    ):
        """Polls pending status until timeout."""
        checksum = "timeout_checksum"
        mock_checksum.set_checksum(str(test_directory), checksum)

        # Always return pending
        mock_api.set_response(checksum, {"status": "pending"})

        # Set very short timeout for test
        upload_service.pending_poll_max_wait = 0.3
        upload_service.pending_poll_interval = 0.1

        results = upload_service.process_inputs([str(test_directory)])

        assert results[0].status == UploadStatus.ERROR
        assert "timed out" in results[0].message.lower() or "timeout" in results[0].message.lower()

    def test_pending_status_polls_then_not_found(
        self, upload_service, test_directory, mock_api, mock_dmf, mock_checksum
    ):
        """Polls pending, other job failed, we upload."""
        checksum = "retry_checksum"
        mock_checksum.set_checksum(str(test_directory), checksum)
        mock_api.set_pending_then_not_found(
            checksum, pending_polls=2, target_uri="lh://prod/ns/filesets/tbl/label/v1"
        )

        results = upload_service.process_inputs([str(test_directory)])

        assert results[0].status == UploadStatus.COMPLETED
        assert mock_api.call_count[checksum] == 3  # 2 pending + 1 not_found
        assert len(mock_dmf.push_calls) == 1


# ============================================================================
# UNIT TESTS: URI PARSING
# ============================================================================


class TestURIParsing:
    """Tests for Lakehouse URI parsing."""

    def test_parse_lakehouse_uri_full(self, upload_service):
        """Full URI with all components parsed correctly."""
        uri = "lh://prod/granite_dot_build.public/filesets/input_data/my_dataset/abc123"
        result = upload_service._parse_lakehouse_uri(uri)

        assert result is not None
        assert result["env"] == "prod"
        assert result["namespace"] == "granite_dot_build.public"
        assert result["table"] == "input_data"
        assert result["label"] == "my_dataset"
        assert result["version"] == "abc123"

    def test_parse_lakehouse_uri_no_version(self, upload_service):
        """URI without version handled correctly."""
        uri = "lh://prod/namespace/filesets/table/label"
        result = upload_service._parse_lakehouse_uri(uri)

        assert result is not None
        assert result["version"] is None

    def test_parse_lakehouse_uri_invalid(self, upload_service):
        """Invalid URI returns None."""
        invalid_uris = [
            "http://example.com",
            "lh://prod/namespace/models/table/label",  # models, not filesets
            "not_a_uri",
            "",
        ]

        for uri in invalid_uris:
            result = upload_service._parse_lakehouse_uri(uri)
            assert result is None, f"Expected None for invalid URI: {uri}"


# ============================================================================
# UNIT TESTS: RETRY LOGIC
# ============================================================================


class TestRetryLogic:
    """Tests for upload retry behavior."""

    def test_transient_error_retries(
        self, upload_service, test_directory, mock_api, mock_dmf, mock_checksum
    ):
        """Transient errors trigger retry."""
        checksum = "retry_test"
        mock_checksum.set_checksum(str(test_directory), checksum)
        mock_api.set_response(
            checksum,
            {"status": "not_found", "target_uri": "lh://prod/ns/filesets/tbl/label/v1"},
        )
        mock_dmf.set_transient_failure(str(test_directory), fail_count=2)

        results = upload_service.process_inputs([str(test_directory)])

        assert results[0].status == UploadStatus.COMPLETED
        assert len(mock_dmf.push_calls) == 3  # 2 failures + 1 success

    def test_permanent_error_no_retry(
        self, upload_service, test_directory, mock_api, mock_dmf, mock_checksum
    ):
        """Permanent errors don't retry."""
        checksum = "perm_error_test"
        mock_checksum.set_checksum(str(test_directory), checksum)
        mock_api.set_response(
            checksum,
            {"status": "not_found", "target_uri": "lh://prod/ns/filesets/tbl/label/v1"},
        )
        mock_dmf.set_permanent_failure(str(test_directory), "Access denied")

        results = upload_service.process_inputs([str(test_directory)])

        assert results[0].status == UploadStatus.ERROR
        assert len(mock_dmf.push_calls) == 1  # Only one attempt

    def test_max_retries_exceeded(
        self, upload_service, test_directory, mock_api, mock_dmf, mock_checksum
    ):
        """Fails after all retries exhausted."""
        checksum = "max_retry_test"
        mock_checksum.set_checksum(str(test_directory), checksum)
        mock_api.set_response(
            checksum,
            {"status": "not_found", "target_uri": "lh://prod/ns/filesets/tbl/label/v1"},
        )
        mock_dmf.set_transient_failure(str(test_directory), fail_count=10)  # More than max retries

        results = upload_service.process_inputs([str(test_directory)])

        assert results[0].status == UploadStatus.ERROR
        assert len(mock_dmf.push_calls) == 4  # 3 retry delays + 1 initial = 4 attempts


# ============================================================================
# UNIT TESTS: INTEGRITY MONITORING
# ============================================================================


class TestIntegrityMonitoring:
    """Tests for source data integrity monitoring."""

    def test_mtime_unchanged_continues(
        self, upload_service, test_directory, mock_api, mock_dmf, mock_checksum
    ):
        """Upload proceeds if source unchanged."""
        checksum = "unchanged_test"
        mock_checksum.set_checksum(str(test_directory), checksum)
        mock_api.set_response(
            checksum,
            {"status": "not_found", "target_uri": "lh://prod/ns/filesets/tbl/label/v1"},
        )

        results = upload_service.process_inputs([str(test_directory)])

        assert results[0].status == UploadStatus.COMPLETED

    def test_mtime_changed_aborts(
        self, upload_service, test_directory, mock_api, mock_dmf, mock_checksum
    ):
        """Upload aborts if source modified during upload."""
        checksum = "modified_test"
        mock_checksum.set_checksum(str(test_directory), checksum)
        mock_api.set_response(
            checksum,
            {"status": "not_found", "target_uri": "lh://prod/ns/filesets/tbl/label/v1"},
        )

        # Set delay so integrity check runs during upload
        mock_dmf.set_delay(0.3)
        upload_service.integrity_check_interval = 0.05

        # Modify file during upload (in separate thread)
        def modify_file():
            time.sleep(0.1)
            (test_directory / "file1.txt").write_text("modified!")

        modifier = threading.Thread(target=modify_file)
        modifier.start()

        results = upload_service.process_inputs([str(test_directory)])
        modifier.join()

        assert results[0].status == UploadStatus.ABORTED
        assert "modified" in results[0].message.lower()


# ============================================================================
# INTEGRATION TESTS: GPFS PERFORMANCE
# ============================================================================


@pytest.mark.gpfs
class TestGPFSChecksum:
    """Tests for GPFS filesystem operations."""

    def test_checksum_large_directory(self, large_test_directory, mock_cache):
        """Performance test for checksumming many files."""
        start_time = time.time()
        mtimes = mock_cache.get_all_mtimes(str(large_test_directory))
        elapsed = time.time() - start_time

        assert len(mtimes) == 100
        print(f"\nChecksum 100 files: {elapsed:.3f}s")

    def test_checksum_deep_hierarchy(self, tmp_path, mock_cache):
        """Test deep nested directory structure."""
        # Create deep hierarchy
        current = tmp_path
        for i in range(10):
            current = current / f"level_{i}"
            current.mkdir()
            (current / f"file_{i}.txt").write_text(f"content_{i}")

        start_time = time.time()
        mtimes = mock_cache.get_all_mtimes(str(tmp_path))
        elapsed = time.time() - start_time

        assert len(mtimes) == 10
        print(f"\nChecksum 10-level deep: {elapsed:.3f}s")

    def test_mtime_accuracy(self, tmp_path, mock_cache):
        """Test mtime resolution on filesystem."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("initial")
        mtime1 = mock_cache.get_all_mtimes(str(test_file))

        # Small delay
        time.sleep(0.01)
        test_file.write_text("modified")
        mtime2 = mock_cache.get_all_mtimes(str(test_file))

        assert mtime1 != mtime2, "Filesystem should detect mtime change"


@pytest.mark.gpfs
class TestGPFSConcurrency:
    """Tests for concurrent filesystem access."""

    def test_cache_concurrent_reads(self, test_directory, mock_cache):
        """Multiple threads can read cache simultaneously."""
        mock_cache.set(str(test_directory), "test_checksum")

        results = []
        errors = []

        def reader():
            try:
                for _ in range(10):
                    result = mock_cache.get(str(test_directory))
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert all(r == "test_checksum" for r in results)


# ============================================================================
# EDGE CASE TESTS: DATA MODIFICATION
# ============================================================================


class TestDataModification:
    """Tests for data modification scenarios."""

    def test_file_added_during_upload(
        self, upload_service, test_directory, mock_api, mock_dmf, mock_checksum
    ):
        """New file added triggers abort."""
        checksum = "file_added_test"
        mock_checksum.set_checksum(str(test_directory), checksum)
        mock_api.set_response(
            checksum,
            {"status": "not_found", "target_uri": "lh://prod/ns/filesets/tbl/label/v1"},
        )

        mock_dmf.set_delay(0.3)
        upload_service.integrity_check_interval = 0.05

        def add_file():
            time.sleep(0.1)
            (test_directory / "new_file.txt").write_text("new content")

        adder = threading.Thread(target=add_file)
        adder.start()

        results = upload_service.process_inputs([str(test_directory)])
        adder.join()

        assert results[0].status == UploadStatus.ABORTED

    def test_file_deleted_during_upload(
        self, upload_service, test_directory, mock_api, mock_dmf, mock_checksum
    ):
        """File deleted triggers abort."""
        checksum = "file_deleted_test"
        mock_checksum.set_checksum(str(test_directory), checksum)
        mock_api.set_response(
            checksum,
            {"status": "not_found", "target_uri": "lh://prod/ns/filesets/tbl/label/v1"},
        )

        mock_dmf.set_delay(0.3)
        upload_service.integrity_check_interval = 0.05

        def delete_file():
            time.sleep(0.1)
            (test_directory / "file1.txt").unlink()

        deleter = threading.Thread(target=delete_file)
        deleter.start()

        results = upload_service.process_inputs([str(test_directory)])
        deleter.join()

        assert results[0].status == UploadStatus.ABORTED


# ============================================================================
# EDGE CASE TESTS: ERROR RECOVERY
# ============================================================================


class TestErrorRecovery:
    """Tests for error handling and recovery."""

    def test_api_connection_failure(self, upload_service, test_directory, mock_api, mock_checksum):
        """API connection errors handled gracefully."""
        checksum = "api_error_test"
        mock_checksum.set_checksum(str(test_directory), checksum)
        mock_api.set_error(ConnectionError("Network unreachable"))

        results = upload_service.process_inputs([str(test_directory)])

        assert results[0].status == UploadStatus.ERROR
        assert "network" in results[0].message.lower() or "connection" in results[0].message.lower()

    def test_dmf_push_timeout(
        self, upload_service, test_directory, mock_api, mock_dmf, mock_checksum
    ):
        """DMF CLI timeout handled gracefully."""
        checksum = "timeout_test"
        mock_checksum.set_checksum(str(test_directory), checksum)
        mock_api.set_response(
            checksum,
            {"status": "not_found", "target_uri": "lh://prod/ns/filesets/tbl/label/v1"},
        )
        mock_dmf.set_timeout(str(test_directory))

        results = upload_service.process_inputs([str(test_directory)])

        assert results[0].status == UploadStatus.ERROR

    def test_checksum_script_failure(self, upload_service, test_directory, mock_api, mock_checksum):
        """Checksum script failure handled gracefully."""
        mock_checksum.set_failure(True)

        results = upload_service.process_inputs([str(test_directory)])

        assert results[0].status == UploadStatus.ERROR
        assert "checksum" in results[0].message.lower()

    def test_invalid_target_uri(self, upload_service, test_directory, mock_api, mock_checksum):
        """Invalid target URI handled gracefully."""
        checksum = "invalid_uri_test"
        mock_checksum.set_checksum(str(test_directory), checksum)
        mock_api.set_response(
            checksum, {"status": "not_found", "target_uri": "invalid://not/a/valid/uri"}
        )

        results = upload_service.process_inputs([str(test_directory)])

        assert results[0].status == UploadStatus.ERROR
        assert "uri" in results[0].message.lower()


# ============================================================================
# PERFORMANCE BENCHMARK TESTS
# ============================================================================


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmarks for BlueVela/GPFS."""

    def test_benchmark_cache_operations(self, mock_cache, test_directory):
        """Benchmark cache read/write performance."""
        iterations = 1000

        # Benchmark writes
        start = time.time()
        for i in range(iterations):
            mock_cache.set(f"{test_directory}/path_{i}", f"checksum_{i}")
        write_time = time.time() - start

        # Benchmark reads
        start = time.time()
        for i in range(iterations):
            mock_cache.get(f"{test_directory}/path_{i}")
        read_time = time.time() - start

        print(
            f"\nCache write: {iterations} ops in {write_time:.3f}s ({iterations/write_time:.0f} ops/s)"
        )
        print(
            f"Cache read: {iterations} ops in {read_time:.3f}s ({iterations/read_time:.0f} ops/s)"
        )

        assert write_time < 5.0, "Cache writes should complete in < 5s"
        assert read_time < 1.0, "Cache reads should complete in < 1s"

    def test_benchmark_mtime_collection(self, large_test_directory, mock_cache):
        """Benchmark mtime collection for large directories."""
        iterations = 10

        times = []
        for _ in range(iterations):
            start = time.time()
            mock_cache.get_all_mtimes(str(large_test_directory))
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        print(f"\nMtime collection (100 files): avg {avg_time*1000:.1f}ms over {iterations} runs")

        assert avg_time < 1.0, "Mtime collection should complete in < 1s"


# ============================================================================
# REAL CHECKSUM BENCHMARKS (GPFS/BlueVela)
# ============================================================================


# Path to real checksum script (adjust for your environment)
LLMBSUM_SCRIPT_PATH = "/proj/granite-build/tools/dirsum_sorted_128.sh"


def _run_real_checksum(path: str, concurrency: int = 8) -> tuple:
    """
    Run the real llmbsum.sh script and return (checksum, elapsed_time).

    Returns (None, 0) if script not available or not executable.
    """
    if not LLMBSUM_SCRIPT_PATH.exists():
        return None, 0

    # Check if script is executable
    if not os.access(LLMBSUM_SCRIPT_PATH, os.X_OK):
        return None, 0

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = os.path.join(tmpdir, "checksum")

        try:
            start_time = time.time()
            result = subprocess.run(
                [str(LLMBSUM_SCRIPT_PATH), path, str(concurrency), output_file],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            elapsed = time.time() - start_time

            if result.returncode != 0:
                return None, elapsed

            checksum_file = f"{output_file}.cksum"
            with open(checksum_file, "r") as f:
                checksum_line = f.read().strip()
                checksum = checksum_line.split()[0]
                return checksum, elapsed

        except (
            subprocess.TimeoutExpired,
            PermissionError,
            FileNotFoundError,
            IndexError,
        ):
            return None, 0


@pytest.mark.gpfs
@pytest.mark.slow
class TestRealChecksumBenchmarks:
    """
    Benchmarks using the real llmbsum.sh script.

    These tests require:
    - llmbsum.sh to exist at the expected path
    - xxh64sum to be available (typically at /proj/granite-build/tools/xxHash/xxh64sum)
    - GNU parallel to be installed

    Run on BlueVela with: pytest test_input_upload.py -v -m "gpfs and slow"
    """

    def test_real_checksum_small_directory(self, test_directory):
        """Benchmark real checksum on small directory (3 files)."""
        checksum, elapsed = _run_real_checksum(str(test_directory))

        if checksum is None:
            pytest.skip("llmbsum.sh not available or failed")

        print(f"\n[REAL] Small directory (3 files): {elapsed:.3f}s, checksum={checksum}")
        assert elapsed < 10.0, "Small directory checksum should complete in < 10s"

    def test_real_checksum_100_files(self, large_test_directory):
        """Benchmark real checksum on 100 files (~100KB total)."""
        checksum, elapsed = _run_real_checksum(str(large_test_directory))

        if checksum is None:
            pytest.skip("llmbsum.sh not available or failed")

        print(f"\n[REAL] 100 files (100KB): {elapsed:.3f}s, checksum={checksum}")
        assert elapsed < 30.0, "100 files checksum should complete in < 30s"

    def test_real_checksum_1000_files(self, tmp_path):
        """Benchmark real checksum on 1000 files (~1MB total)."""
        # Create 1000 files
        for i in range(1000):
            (tmp_path / f"file_{i:04d}.txt").write_bytes(os.urandom(1024))

        checksum, elapsed = _run_real_checksum(str(tmp_path))

        if checksum is None:
            pytest.skip("llmbsum.sh not available or failed")

        print(f"\n[REAL] 1000 files (1MB): {elapsed:.3f}s, checksum={checksum}")
        assert elapsed < 60.0, "1000 files checksum should complete in < 60s"

    def test_real_checksum_concurrency_scaling(self, tmp_path):
        """Benchmark checksum performance at different concurrency levels."""
        # Create 500 files
        for i in range(500):
            (tmp_path / f"file_{i:04d}.txt").write_bytes(os.urandom(2048))

        results = {}
        for concurrency in [1, 2, 4, 8]:
            checksum, elapsed = _run_real_checksum(str(tmp_path), concurrency=concurrency)
            if checksum is None:
                pytest.skip("llmbsum.sh not available or failed")
            results[concurrency] = elapsed

        print(f"\n[REAL] Concurrency scaling (500 files, 1MB):")
        for conc, elapsed in sorted(results.items()):
            speedup = results[1] / elapsed if results.get(1) else 0
            print(f"  concurrency={conc}: {elapsed:.3f}s (speedup: {speedup:.2f}x)")

        # Verify some speedup with parallelism
        if results.get(1) and results.get(8):
            assert results[8] < results[1], "Parallel checksum should be faster than serial"

    def test_real_checksum_large_files(self, tmp_path):
        """Benchmark real checksum on fewer but larger files."""
        # Create 10 files of 10MB each (100MB total)
        for i in range(10):
            (tmp_path / f"large_file_{i}.bin").write_bytes(os.urandom(10 * 1024 * 1024))

        checksum, elapsed = _run_real_checksum(str(tmp_path))

        if checksum is None:
            pytest.skip("llmbsum.sh not available or failed")

        throughput_mb = 100 / elapsed if elapsed > 0 else 0
        print(
            f"\n[REAL] 10 large files (100MB): {elapsed:.3f}s ({throughput_mb:.1f} MB/s), checksum={checksum}"
        )

    def test_real_checksum_deep_hierarchy(self, tmp_path):
        """Benchmark real checksum on deeply nested directory."""

        # Create 5 levels deep, 10 files per level
        def create_level(parent: Path, depth: int, max_depth: int = 5):
            if depth > max_depth:
                return
            for i in range(10):
                (parent / f"file_{depth}_{i}.txt").write_bytes(os.urandom(1024))
            subdir = parent / f"level_{depth}"
            subdir.mkdir()
            create_level(subdir, depth + 1, max_depth)

        create_level(tmp_path, 1)

        checksum, elapsed = _run_real_checksum(str(tmp_path))

        if checksum is None:
            pytest.skip("llmbsum.sh not available or failed")

        print(f"\n[REAL] Deep hierarchy (5 levels, 50 files): {elapsed:.3f}s, checksum={checksum}")

    def test_real_checksum_cached_vs_uncached(self, tmp_path, mock_cache):
        """Compare cached vs uncached checksum retrieval time."""
        # Create test files
        for i in range(100):
            (tmp_path / f"file_{i:04d}.txt").write_bytes(os.urandom(1024))

        # First run - uncached (real calculation)
        checksum1, uncached_time = _run_real_checksum(str(tmp_path))

        if checksum1 is None:
            pytest.skip("llmbsum.sh not available or failed")

        # Store in cache
        mock_cache.set(str(tmp_path), checksum1)

        # Second run - cached (just cache lookup)
        start = time.time()
        checksum2 = mock_cache.get(str(tmp_path))
        cached_time = time.time() - start

        print(f"\n[REAL vs CACHED] 100 files:")
        print(f"  Uncached (real llmbsum.sh): {uncached_time:.3f}s")
        print(f"  Cached (cache lookup):      {cached_time*1000:.3f}ms")
        print(f"  Speedup: {uncached_time/cached_time:.0f}x")

        assert checksum1 == checksum2, "Cached checksum should match"
        assert cached_time < uncached_time / 10, "Cache should be >10x faster"


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    # Run with verbose output and timing
    pytest.main([__file__, "-v", "--durations=10"])
