"""
Artifact registration service for llmbsub.

Handles registering input artifacts with gbserver for lineage tracking.
This service:
- Calculates checksums for input paths (reusing InputUploadService logic)
- Registers artifacts via gbserver /api/v1/artifacts/lh/fileset API
- Returns registration result with checksum, Lakehouse URI, and upload status
"""

import logging
from dataclasses import dataclass
from typing import Optional

from fastapi import HTTPException

from gbcli.utils.gbconstants import GBSERVER_INSTANCE
from gbcli.utils.gbcredentials import GBCredentials
from gbcli.utils.gbserver import gb_server_request
from llmbsub.services.input_upload_service import InputUploadService
from llmbsub.utils.checksum_cache import DEFAULT_CACHE_BASE_PATH

logger = logging.getLogger(__name__)

# Default Lakehouse configuration
DEFAULT_LH_ENV = "prod"
DEFAULT_TABLE_NAME = "fileset"


@dataclass
class RegistrationResult:
    """Result of artifact registration."""

    artifact_id: str
    checksum: str
    lh_uri: str
    upload_needed: bool
    message: str = ""


class ArtifactRegistrationService:
    """Service for artifact registration with gbserver.

    This service registers input artifacts for lineage tracking by:
    1. Calculating checksum of the input path
    2. Registering via gbserver /api/v1/artifacts/lh/fileset API
    3. Returning the Lakehouse URI and whether upload is needed

    Usage:
        service = ArtifactRegistrationService(space="my_space")
        result = service.register_input("/path/to/data")
        if result.upload_needed:
            # Start DMF upload to result.lh_uri
            pass
    """

    def __init__(
        self,
        space: str,
        namespace: Optional[str] = None,
        table_name: str = DEFAULT_TABLE_NAME,
        lh_env: str = DEFAULT_LH_ENV,
        cache_base_path: str = DEFAULT_CACHE_BASE_PATH,
        concurrency: int = 8,
        verbose: bool = False,
    ):
        """
        Initialize the registration service.

        Args:
            space: LLM.build space name
            namespace: Lakehouse namespace (defaults to granite_dot_build.{space})
            table_name: Lakehouse table name for inputs (default: "inputs")
            lh_env: Lakehouse environment (default: "prod")
            cache_base_path: Base path for checksum cache
            concurrency: Concurrency for checksum calculation
            verbose: Enable verbose logging of API calls and operations
        """
        self.space = space
        # Replace dashes with underscores in space name for namespace (Lakehouse requirement)
        space_normalized = space.replace("-", "_")
        self.namespace = namespace or f"granite_dot_build.{space_normalized}"
        self.table_name = table_name
        self.lh_env = lh_env
        self.verbose = verbose
        self.credentials = GBCredentials()
        self.user_token = self.credentials.get("token", section="user.github")
        self.username = self.credentials.get("login", section="user.github")
        # Reuse InputUploadService for checksum calculation
        self._upload_service = InputUploadService(
            space=space,
            concurrency=concurrency,
            cache_base_path=cache_base_path,
        )

        if self.verbose:
            self._log_verbose(f"Initialized ArtifactRegistrationService for space: {space}")
            self._log_verbose(f"Namespace: {self.namespace}")
            self._log_verbose(f"Table: {self.table_name}, LH env: {self.lh_env}")
            self._log_verbose(f"Cache base path: {cache_base_path}")
            self._log_verbose(f"Checksum concurrency: {concurrency}")

    def _log_verbose(self, message: str):
        """Log a message in verbose mode."""
        print(f"      [registration] {message}")

    def register_input(
        self, path: str, type: str, name: Optional[str] = None
    ) -> RegistrationResult:
        """
        Register an input artifact for lineage tracking.

        This method:
        1. Calculates the checksum of the input path
        2. Registers via gbserver /api/v1/artifacts/lh/fileset API
        3. If artifact already exists (409 conflict), returns existing URI
        4. Returns the result with checksum, Lakehouse URI, and upload status

        Args:
            path: Path to the input file or directory
            type: Input type
            name: Optional name for the artifact (defaults to checksum)

        Returns:
            RegistrationResult with:
            - artifact_id: Artifact UUID string
            - checksum: Calculated checksum of the input
            - lh_uri: Lakehouse URI for the artifact
            - upload_needed: True if artifact needs to be uploaded
            - message: Status message

        Raises:
            RuntimeError: If API registration fails
        """
        # Calculate checksum (reuses caching logic from InputUploadService)
        logger.info(f"Calculating checksum for {path}")
        if self.verbose:
            self._log_verbose(f"Starting checksum calculation for: {path}")

        checksum = self._upload_service._get_checksum(path)
        logger.info(f"Checksum for {path}: {checksum}")
        if self.verbose:
            self._log_verbose(f"Checksum result: {checksum}")

        # Use checksum as the artifact name/label if not provided
        artifact_name = name or checksum

        # Register artifact - API handles "already exists" case with 409 conflict
        logger.info(f"Registering artifact for {path}")
        if self.verbose:
            self._log_verbose(f"Calling gbserver /api/v1/artifacts/lh/{type} API...")

        try:
            response = self._register_input(path, checksum, type, artifact_name)

            # Check response for success - the API returns {"registered": {...artifact...}}
            registered = response.get("registered", {})
            lh_uri = registered.get("uri", "")
            artifact_id = registered.get("uuid" "")

            logger.info(f"Registration response for {path}: uri={lh_uri}")
            if self.verbose:
                self._log_verbose(f"API response: {response}")

            if lh_uri:
                return RegistrationResult(
                    artifact_id=artifact_id,
                    checksum=checksum,
                    lh_uri=lh_uri,
                    upload_needed=True,
                    message="Artifact registered successfully",
                )

            raise RuntimeError(f"Registration failed for {path}: {response}")

        except HTTPException as e:
            # 409 Conflict means artifact with this checksum already exists
            # The detail contains the existing artifact info including URI
            if e.status_code == 409:
                detail = e.detail
                if isinstance(detail, dict):
                    lh_uri = detail.get("uri", "")
                    artifact_id = detail.get("uuid" "")
                    if lh_uri:
                        logger.info(f"Artifact already exists: {lh_uri}")
                        if self.verbose:
                            self._log_verbose(f"Found existing artifact (409 conflict): {detail}")
                        return RegistrationResult(
                            artifact_id=artifact_id,
                            checksum=checksum,
                            lh_uri=lh_uri,
                            upload_needed=False,
                            message="Artifact already exists in Lakehouse",
                        )
                # 409 but no URI in detail - re-raise
                raise RuntimeError(f"Checksum conflict but no URI returned: {detail}")
            # Other HTTP errors - re-raise
            raise

    def _register_input(self, path: str, checksum: str, type: str, name: str) -> dict:
        f"""
        Register an input via gbserver /api/v1/artifacts/lh/{type} API.

        Args:
            path: Path to the input (used as origin_uri)
            checksum: Checksum of the input
            type: Input type
            name: Name/label for the fileset

        Returns:
            API response dict with status and URI information
        """
        url = f"{GBSERVER_INSTANCE}/api/v1/artifacts/lh/{type}"
        body = {
            "space_name": self.space,
            "username": self.username,
            "namespace": self.namespace,
            "table_name": self.table_name,
            "name": name,
            "lh_env": self.lh_env,
            "tags": [],
            "certified_no_restrictions": True,  # New input with no prior artifacts
            "origin_uris": [f"env://{path}"],
            "description": f"Input artifact from {path}",
            "checksum": checksum,
            "status": "pending",
        }

        if type == "fileset":
            body.update(
                {
                    "fileset_label": name,
                    "fileset_version": checksum,
                }
            )

        if type == "model":
            body.update({"model_label": name, "model_revision": checksum})

        logger.debug(f"Calling lh/{type} API: {url} with checksum {checksum[:16]}...")

        if self.verbose:
            self._log_verbose(f"REST API Request:")
            self._log_verbose(f"  URL: POST {url}")
            self._log_verbose(f"  Body: {body}")

        response = gb_server_request(
            user_token=self.user_token,
            url=url,
            http_method="post",
            body=body,
            params=None,
        )

        if self.verbose:
            self._log_verbose(f"REST API Response: {response}")

        return response
