"""
Input upload service for llmbsub.

Handles uploading input datasets to Lakehouse (DMF) for input lineage tracking.
This service:
- Calculates checksums for input paths using dirsum_sorted_128.sh
- Uploads filesets to DMF using the dmf CLI
- Monitors source integrity during upload
- Implements retry logic for transient failures

Note: Registration is now handled by ArtifactRegistrationService.
This service focuses on checksum calculation and DMF upload only.
"""

import json
import logging
import multiprocessing
import os
import re
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from gbcli.client.client import GBClient
from llmbsub.utils.checksum_cache import ChecksumCache, DEFAULT_CACHE_BASE_PATH

logger = logging.getLogger(__name__)

# Retry configuration
RETRY_DELAYS = [30, 60, 120]  # seconds between retries

# Integrity check interval during upload
INTEGRITY_CHECK_INTERVAL = 30  # seconds

# Path to checksum script
LLMBSUM_SCRIPT_PATH = "/proj/granite-build/tools/dirsum_sorted_128.sh"

# Default concurrency for checksum calculation
DEFAULT_CHECKSUM_CONCURRENCY = 8


class UploadStatus(Enum):
    """Status of an upload operation."""

    COMPLETED = "completed"
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
    """Result of uploading a single input path."""

    input_path: str
    target_uri: str
    status: UploadStatus
    message: str


class InputUploadService:
    """Service for uploading llmbsub inputs to Lakehouse.

    This service provides:
    1. Checksum calculation (used by ArtifactRegistrationService)
    2. DMF upload for already-registered artifacts

    Usage:
        service = InputUploadService(space="my_space")

        # For checksum calculation (used internally by ArtifactRegistrationService)
        checksum = service._get_checksum("/path/to/data")

        # For uploading to a registered URI
        result = service.upload_to_uri("/path/to/data", "lh://prod/ns/filesets/table/label/version")
    """

    def __init__(
        self,
        space: str,
        concurrency: int = DEFAULT_CHECKSUM_CONCURRENCY,
        cache_base_path: str = DEFAULT_CACHE_BASE_PATH,
    ):
        self.space = space
        self.concurrency = concurrency
        self.checksum_cache = ChecksumCache(cache_base_path)
        self._abort_upload = False

    def _get_checksum(self, input_path: str) -> str:
        """Calculate or retrieve cached checksum.

        This method is used by ArtifactRegistrationService for registration.

        Args:
            input_path: Path to the input file or directory

        Returns:
            Checksum string
        """
        # Check cache first
        cached = self.checksum_cache.get(input_path)
        if cached:
            logger.info(f"Using cached checksum for {input_path}")
            return cached

        # Calculate new checksum
        checksum = self._calculate_checksum(input_path)
        self.checksum_cache.set(input_path, checksum)
        return checksum

    def _calculate_checksum(self, input_path: str) -> str:
        """Calculate checksum using dirsum_sorted_128.sh."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "checksum")

            result = subprocess.run(
                [
                    str(LLMBSUM_SCRIPT_PATH),
                    input_path,
                    str(self.concurrency),
                    output_file,
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                raise RuntimeError(f"Checksum calculation failed: {result.stderr}")

            checksum_file = f"{output_file}.cksum"
            with open(checksum_file, "r") as f:
                checksum_line = f.read().strip()
                # xxh64sum output format: "checksum  filename"
                return checksum_line.split()[0]

    def upload_to_uri(
        self, artifact_id: str, input_path: str, target_uri: str
    ) -> UploadResult:
        """
        Upload a file/directory to an already-registered Lakehouse URI.

        This method is called after registration to perform the actual DMF upload.

        Args:
            input_path: Local path to upload
            target_uri: Lakehouse URI from registration (e.g., lh://prod/ns/filesets/table/label/version)

        Returns:
            UploadResult with status and message
        """
        try:
            # Parse URI to extract components for dmf command
            uri_components = self._parse_lakehouse_uri(target_uri)
            if not uri_components:
                return UploadResult(
                    input_path=input_path,
                    target_uri=target_uri,
                    status=UploadStatus.ERROR,
                    message=f"Failed to parse target URI: {target_uri}",
                )

            type = uri_components["type"]
            namespace = uri_components["namespace"]
            table = uri_components["table"]
            label = uri_components["label"]
            version = uri_components["version"]

            # Record initial mtimes for integrity checking
            initial_mtimes = self.checksum_cache.get_all_mtimes(input_path)

            # Start integrity monitoring in background
            self._abort_upload = False
            integrity_thread = threading.Thread(
                target=self._monitor_integrity,
                args=(input_path, initial_mtimes),
                daemon=True,
            )
            integrity_thread.start()

            try:
                # Upload with retry
                success = self._upload_with_retry(
                    artifact_id, input_path, type, namespace, table, label, version
                )

                if self._abort_upload:
                    raise SourceModifiedError("Source data was modified during upload")

                if success:
                    return UploadResult(
                        input_path=input_path,
                        target_uri=target_uri,
                        status=UploadStatus.COMPLETED,
                        message="Upload successful",
                    )
                else:
                    return UploadResult(
                        input_path=input_path,
                        target_uri=target_uri,
                        status=UploadStatus.ERROR,
                        message="Upload failed after retries",
                    )

            finally:
                # Signal integrity thread to stop
                self._abort_upload = True

        except SourceModifiedError as e:
            logger.warning(f"Source modified during upload: {e}")
            return UploadResult(
                input_path=input_path,
                target_uri=target_uri,
                status=UploadStatus.ABORTED,
                message=f"Source data modified during upload: {e}",
            )
        except Exception as e:
            logger.error(f"Error uploading {input_path}: {e}")
            return UploadResult(
                input_path=input_path,
                target_uri=target_uri,
                status=UploadStatus.ERROR,
                message=str(e),
            )

    def upload_batch(self, uploads: List[Dict]) -> List[UploadResult]:
        """
        Upload multiple inputs to their registered URIs.

        Args:
            uploads: List of dicts with 'path' and 'target_uri' keys

        Returns:
            List of UploadResult for each input
        """
        results = []
        for upload in uploads:
            artifact_id = upload["artifact_id"]
            path = upload["path"]
            target_uri = upload["target_uri"]
            result = self.upload_to_uri(artifact_id, path, target_uri)
            results.append(result)
            logger.info(f"Uploaded {path}: {result.status.value} - {result.message}")
        return results

    def _monitor_integrity(self, path: str, initial_mtimes: Dict[str, float]):
        """
        Monitor source directory for modifications during upload.
        Sets _abort_upload flag if modifications detected.
        """
        while not self._abort_upload:
            time.sleep(INTEGRITY_CHECK_INTERVAL)
            if self._abort_upload:
                break

            current_mtimes = self.checksum_cache.get_all_mtimes(path)
            if current_mtimes != initial_mtimes:
                logger.warning(f"Source data modified during upload: {path}")
                self._abort_upload = True
                break

    def _upload_with_retry(
        self,
        artifact_id: str,
        path: str,
        input_type: str,
        namespace: str,
        table: str,
        label: str,
        version: str,
    ) -> bool:
        """
        Execute dmf push with retry logic for transient failures.

        Returns True if upload succeeded, False otherwise.
        """
        for attempt, delay in enumerate(RETRY_DELAYS + [None], 1):
            if self._abort_upload:
                logger.warning("Upload aborted due to source modification")
                return False

            try:
                self._run_dmf_push(
                    artifact_id, path, input_type, namespace, table, label, version
                )
                return True

            except TransientError as e:
                if delay is None:
                    logger.error(f"Upload failed after {attempt} attempts: {e}")
                    break
                logger.warning(
                    f"Upload attempt {attempt} failed: {e}. Retrying in {delay}s"
                )
                time.sleep(delay)

            except PermanentError as e:
                logger.error(f"Permanent error during upload: {e}")
                raise

        return False

    def _run_dmf_push(
        self,
        artifact_id: str,
        path: str,
        input_type: str,
        namespace: str,
        table: str,
        label: str,
        version: str,
    ):
        """
        Execute dmf push via GBClient in a separate process.

        Runs the push operation in a multiprocessing.Process to handle:
        - Artifact client initialization
        - DMF push with retryable error handling

        Args:
            path: Local filesystem path to upload
            type: Artifact type
            namespace: Lakehouse namespace
            table: Table name
            label: Fileset label (usually checksum)
            version: Version string

        Raises:
            TransientError: For retryable errors (network, 5xx)
            PermanentError: For non-retryable errors (4xx, permissions)
        """
        result_queue = multiprocessing.Queue()
        lh_token = GBClient.Auth.lakehouse_token_for_space(space=self.space)
        artifact_client = GBClient.Artifact()

        process = multiprocessing.Process(
            target=self._push_worker,
            args=(
                lh_token,
                artifact_client,
                path,
                input_type,
                namespace,
                table,
                label,
                version,
                self.space,
                result_queue,
            ),
        )

        process.start()

        try:
            # Wait for process with timeout (2 hours for large uploads)
            process.join(timeout=7200)

            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
                    process.join()
                raise TransientError("dmf push timed out")

            # Get result from queue
            if not result_queue.empty():
                result = result_queue.get()
                if result["success"]:
                    logger.info(
                        f"dmf push succeeded: {result.get('artifact_entry', '')}"
                    )

                    artifact_client.update_artifact(
                        artifact_id=artifact_id,
                        status="success",
                    )

                    return

                error_msg = result.get("error", "Unknown error")
                error_type = result.get("error_type", "transient")

                if error_type == "permanent":
                    artifact_client.update_artifact(
                        artifact_id=artifact_id,
                        status="failure",
                    )
                    raise PermanentError(f"dmf push failed: {error_msg}")
                else:
                    artifact_client.update_artifact(
                        artifact_id=artifact_id,
                        status="failure",
                    )
                    raise TransientError(f"dmf push failed: {error_msg}")
            else:
                raise TransientError("dmf push process completed without result")

        except Exception as e:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    process.kill()
            raise

    @staticmethod
    def _push_worker(
        lh_token: str,
        artifact_client: Any,
        path: str,
        input_type: str,
        namespace: str,
        table: str,
        label: str,
        version: str,
        space: str,
        result_queue: multiprocessing.Queue,
    ):
        """
        Worker function that runs in a separate process.

        Handles artifact push using GBClient.

        Args:
            path: Local filesystem path to upload
            namespace: Lakehouse namespace
            table: Table name
            label: Fileset label (usually checksum)
            version: Version string
            space: Lakehouse space
            result_queue: Queue to communicate result back to parent process
        """
        try:
            if input_type == "table":
                response_push = artifact_client.push(
                    lh_token,
                    path,
                    input_type,
                    None,
                    table,
                    "",
                    "",
                    "",
                    space,
                    table,
                    namespace,
                )

            if input_type in ("fileset", "model"):
                response_push = artifact_client.push(
                    lh_token,
                    path,
                    "fileset",
                    label,
                    label,
                    "",
                    "",
                    "",
                    space,
                    table,
                    namespace,
                    version,
                )
            # else:  # model TODO
            #   response_push = artifact_client.push(
            #       lh_token,
            #       input_path,
            #       type,
            #       label,
            #       label,
            #       '',
            #       '',
            #       '',
            #       space_name,
            #       table_name,
            #       namespace,
            #       version,
            #   )

            if type(response_push) is dict:
                result_queue.put({"success": True, "artifact_entry": response_push})

        except Exception as e:
            error_str = str(e).lower()

            # Classify error as transient or permanent
            permanent_indicators = [
                "permission denied",
                "not found",
                "invalid",
                "401",
                "403",
                "404",
                "400",
            ]

            error_type = "transient"
            for indicator in permanent_indicators:
                if indicator in error_str:
                    error_type = "permanent"
                    break

            result_queue.put(
                {"success": False, "error": str(e), "error_type": error_type}
            )

    def _parse_lakehouse_uri(self, uri: str) -> Optional[Dict]:
        """
        Parse Lakehouse URI into components for dmf command.

        URI format: lh://<env>/<namespace>/filesets/<table>/<label>/<version>
        Example: lh://prod/granite_dot_build.bv_data_eng/filesets/inputs/25d4e5b9b6607be316624367bbc42c1d/granite-dot-build

        Returns dict with:
            - env: Environment (e.g., "prod")
            - namespace: Full namespace (e.g., "granite_dot_build.bv_data_eng")
            - table: Table name (e.g., "inputs")
            - label: Fileset label (e.g., checksum "25d4e5b9b6607be316624367bbc42c1d")
            - version: Version string (e.g., "granite-dot-build")

        Returns None if URI is invalid.
        """
        # Pattern: lh://env/namespace/filesets/table/label/version
        pattern = r"lh://([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)/([^/]+)"
        match = re.match(pattern, uri)

        if not match:
            logger.error(f"Failed to parse Lakehouse URI: {uri}")
            return None

        return {
            "env": match.group(1),
            "namespace": match.group(2),
            "type": match.group(3).removesuffix("s"),
            "table": match.group(4),
            "label": match.group(5),
            "version": match.group(6),
        }


# Run DMF artifacts upload in bsub
if __name__ == "__main__":
    import sys

    inputs_file = sys.argv[1]
    space_name = sys.argv[2]
    log_file = sys.argv[3]

    inputs_list = []

    try:
        with open(inputs_file, "r") as file:
            for line in file:
                inputs_list.append(json.loads(line))

        service = InputUploadService(space=space_name)
        upload_response = service.upload_batch(inputs_list)

    except Exception as e:
        logger.error(f"Failed to upload artifacts: {e}")
        sys.exit(1)
