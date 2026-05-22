"""Hugging Face Hub implementation of artifact registry."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

from gbcli.models.artifact import ArtifactMetadata
from gbcommon.utils.hf_utils import get_hf_artifact_uri

logger = logging.getLogger(__name__)


class HFRegistry:
    """Hugging Face Hub implementation of artifact registry.

    Uses huggingface_hub.HfApi to upload models and datasets.
    Supports auto-creation of repositories and private repos.
    """

    def __init__(
        self,
        hf_token: str,
        resource_group_id: str | None = None,
        organization: str | None = None,
    ) -> None:
        """Initialize Hugging Face registry client.

        Args:
            hf_token: Hugging Face API token for authentication.
            resource_group_id: Optional resource group ID for organizing artifacts.
            organization: Optional HuggingFace organization for namespace.

        Raises:
            ValueError: If token is empty.
        """
        if not hf_token:
            raise ValueError("HF_TOKEN must be set")
        self.api = HfApi(token=hf_token)
        self.token = hf_token
        self.resource_group_id = resource_group_id
        self.organization = organization
        logger.info(
            "HFRegistry initialized: resource_group_id=%s organization=%s",
            resource_group_id,
            organization,
        )

    def upload_artifact(
        self,
        local_path: str | Path,
        repo_id: str,
        artifact_type: Literal["model", "dataset", "space", "bucket"],
        private: bool = False,
        commit_message: str | None = None,
        revision: str | None = None,
        exist_ok: bool = True,
    ) -> ArtifactMetadata:
        """Upload an artifact to Hugging Face Hub.

        Creates the repository if it doesn't exist. Uploads all files from
        the local path. Returns canonical HF URL.

        Args:
            local_path: Local file or directory path to upload.
            repo_id: Repository ID in format 'organization/repo-name'.
            artifact_type: Type of artifact (model, dataset, or space).
            private: Whether to make the repository private.
            commit_message: Optional commit message.
            revision: Optional git revision/branch name (default: "main").
            exist_ok: If True, don't raise error if repository already exists. If False, raise error.

        Returns:
            ArtifactMetadata with HF Hub URI and details.

        Raises:
            FileNotFoundError: If local_path doesn't exist.
            ValueError: If repo_id format is invalid.
            RuntimeError: If upload fails or repository exists (when exist_ok=False).
        """
        local = Path(local_path)
        if not local.exists():
            raise FileNotFoundError(f"Local path does not exist: {local_path}")

        if "/" not in repo_id:
            raise ValueError(
                f"repo_id must be in format 'organization/repo-name', got: {repo_id}"
            )
        try:
            logger.info(
                "uploading_to_hf: repo_id=%s artifact_type=%s local_path=%s exist_ok=%s",
                repo_id,
                artifact_type,
                str(local),
                exist_ok,
            )

            if artifact_type == "bucket":
                self.api.create_bucket(
                    bucket_id=repo_id,
                    private=private,
                    resource_group_id=self.resource_group_id,
                    exist_ok=exist_ok,
                )
                logger.info(
                    "bucket_created_or_exists: bucket_id=%s exist_ok=%s",
                    repo_id,
                    exist_ok,
                )

                if local.is_file():
                    self.api.batch_bucket_files(
                        bucket_id=repo_id, add=[(str(local), local.name)]
                    )
                    logger.info("bucket_file_uploaded: file=%s", local.name)
                else:
                    bucket_hf_path = f"hf://buckets/{repo_id}"
                    self.api.sync_bucket(source=str(local), dest=bucket_hf_path)
                    logger.info("bucket_folder_uploaded: folder=%s", str(local))
            else:
                # Create repo if it doesn't exist
                repo_url = self.api.create_repo(
                    repo_id=repo_id,
                    repo_type=artifact_type,
                    private=private,
                    exist_ok=exist_ok,
                    resource_group_id=self.resource_group_id,
                )
                logger.info(
                    "repo_created_or_exists: repo_url=%s exist_ok=%s",
                    repo_url,
                    exist_ok,
                )

                # Ensure model card exists before uploading (for models only)
                if artifact_type == "model" and local.is_dir():
                    self._ensure_model_card(local, repo_id)

                # Upload files
                if local.is_file():
                    # Single file upload
                    self.api.upload_file(
                        path_or_fileobj=str(local),
                        path_in_repo=local.name,
                        repo_id=repo_id,
                        repo_type=artifact_type,
                        commit_message=commit_message or f"Upload {local.name}",
                        revision=revision,
                    )
                    logger.info(
                        "file_uploaded: file=%s revision=%s", local.name, revision
                    )
                else:
                    # Directory upload with exclusions for .DS_Store and hidden files
                    self.api.upload_folder(
                        folder_path=str(local),
                        repo_id=repo_id,
                        repo_type=artifact_type,
                        commit_message=commit_message or f"Upload from {local.name}",
                        revision=revision,
                        ignore_patterns=[
                            ".DS_Store",
                            "*/.DS_Store",
                            "__pycache__",
                            "*.pyc",
                            ".git",
                            ".git/*",
                            ".env",
                            ".env.*",
                            "*/.ipynb_checkpoints",
                            "*/.*",
                        ],
                    )
                    logger.info("folder_uploaded", folder=str(local), revision=revision)

            # Construct canonical URI
            uri = get_hf_artifact_uri(repo_id, artifact_type)

            # Compute hash for reference
            artifact_hash = self._compute_hash(local)

            metadata = ArtifactMetadata(
                uri=uri,
                artifact_type=artifact_type,
                registry="huggingface",
                repo_id=repo_id,
                artifact_hash=artifact_hash,
                is_private=private,
            )

            logger.info("artifact_uploaded_successfully", uri=uri)
            return metadata

        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code == 403:
                logger.error(
                    "upload_failed_permission_denied", repo_id=repo_id, error=str(e)
                )
                raise RuntimeError(
                    f"Your HuggingFace token does not have write access to '{repo_id}'. "
                    "Please check your token permissions at https://huggingface.co/settings/tokens "
                    "and ensure it has the 'write' role."
                ) from e
            logger.error("upload_failed", error=str(e), repo_id=repo_id)
            raise RuntimeError(f"Failed to upload to Hugging Face: {e}") from e
        except Exception as e:
            logger.error("upload_failed", error=str(e), repo_id=repo_id)
            raise RuntimeError(f"Failed to upload to Hugging Face: {e}") from e

    def _ensure_model_card(self, local_dir: Path, repo_id: str) -> None:
        """Ensure a model card (README.md) exists with proper YAML metadata.

        If README.md exists but lacks YAML frontmatter, enhances it.
        If no README.md exists, creates a minimal one with YAML metadata.

        Args:
            local_dir: Directory containing model files.
            repo_id: Repository ID for context in model card.
        """
        readme_path = local_dir / "README.md"

        # Check if README exists and has YAML
        if readme_path.exists():
            content = readme_path.read_text()
            if content.startswith("---"):
                # Already has YAML frontmatter
                logger.info("model_card_already_has_yaml", path=readme_path)
                return
            else:
                # Add YAML frontmatter to existing README
                logger.info("adding_yaml_to_existing_model_card", path=readme_path)
                yaml_header = self._generate_yaml_header(local_dir, repo_id)
                enhanced_content = yaml_header + "\n\n" + content
                readme_path.write_text(enhanced_content)
                return

        # Create minimal model card with YAML
        logger.info("creating_model_card_with_yaml", path=readme_path)
        model_card_content = self._generate_model_card(local_dir, repo_id)
        readme_path.write_text(model_card_content)

    def _generate_yaml_header(self, local_dir: Path, repo_id: str) -> str:
        """Generate YAML frontmatter for model card.

        Args:
            local_dir: Directory containing model files.
            repo_id: Repository ID.

        Returns:
            YAML frontmatter string.
        """
        # Try to infer model type from config.json
        model_type = "transformer"
        config_path = local_dir / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text())
                model_type = config.get("model_type", "transformer")
            except Exception:
                pass

        yaml = f"""---
language:
  - en
license: mit
library_name: transformers
tags:
  - {model_type}
  - pytorch
model_name: "Model from {repo_id}"
model_id: "{repo_id}"
---"""
        return yaml

    def _generate_model_card(self, local_dir: Path, repo_id: str) -> str:
        """Generate a minimal model card with YAML metadata.

        Args:
            local_dir: Directory containing model files.
            repo_id: Repository ID.

        Returns:
            Complete model card content with YAML frontmatter.
        """
        yaml_header = self._generate_yaml_header(local_dir, repo_id)

        # Try to extract model details from config.json
        model_info = {}
        config_path = local_dir / "config.json"
        if config_path.exists():
            try:
                model_info = json.loads(config_path.read_text())
            except Exception:
                pass

        model_type = model_info.get("model_type", "transformer").upper()
        hidden_size = model_info.get("hidden_size", "N/A")
        num_layers = model_info.get("num_hidden_layers", "N/A")
        vocab_size = model_info.get(
            "vocab_size", model_info.get("vocabulary_size", "N/A")
        )

        model_card = f"""{yaml_header}

# {model_type} Model

This is a model uploaded via dmf-ng (Data Lineage Management for Hugging Face).

## Model Details

### Model Description

- **Model type:** {model_type}
- **Repository:** {repo_id}
- **Upload timestamp:** {datetime.now().isoformat()}

### Architecture

- **Hidden size:** {hidden_size}
- **Number of layers:** {num_layers}
- **Vocabulary size:** {vocab_size}

## Files

This repository contains:
"""
        # List files in directory
        for file in sorted(local_dir.glob("*")):
            if file.is_file() and file.name != "README.md":
                size = file.stat().st_size
                model_card += f"- **{file.name}** ({size} bytes)\n"

        model_card += f"""
## Upload Information

- **Uploaded via:** dmf-ng
- **Repository ID:** {repo_id}
- **Upload date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

For information about this model, visit the [repository](https://huggingface.co/{repo_id}).
"""
        return model_card

    def download_artifact(
        self,
        repo_id: str,
        artifact_type: Literal["model", "dataset", "space", "bucket"],
        download_dir: str | Path,
        revision: str = "main",
    ) -> dict:
        """Download an artifact from Hugging Face Hub.

        Downloads all files from a repository to a local directory.
        Creates the directory if it doesn't exist.

        Args:
            repo_id: Repository ID in format 'organization/repo-name'.
            artifact_type: Type of artifact (model, dataset, or space).
            download_dir: Local directory where files will be downloaded.
            revision: Git revision/branch name (default: "main").

        Returns:
            Dictionary with download information including:
            - repo_id: Repository ID
            - artifact_type: Type of artifact
            - download_dir: Where files were downloaded
            - revision: Revision used
            - file_count: Number of files downloaded
            - total_size: Total size in bytes

        Raises:
            FileNotFoundError: If download_dir cannot be created.
            ValueError: If repo_id format is invalid.
            RuntimeError: If download fails.
        """
        local_dir = Path(download_dir)
        if "/" not in repo_id:
            raise ValueError(
                f"repo_id must be in format 'organization/repo-name', got: {repo_id}"
            )

        try:
            logger.info(
                "downloading_from_hf",
                repo_id=repo_id,
                artifact_type=artifact_type,
                download_dir=str(local_dir),
                revision=revision,
            )

            # Create download directory if it doesn't exist
            local_dir.mkdir(parents=True, exist_ok=True)

            if artifact_type == "bucket":
                bucket_hf_path = f"hf://buckets/{repo_id}"
                self.api.sync_bucket(source=bucket_hf_path, dest=str(local_dir))
                snapshot_path = str(local_dir)
            else:
                snapshot_path = snapshot_download(
                    repo_id=repo_id,
                    repo_type=artifact_type,
                    revision=revision,
                    cache_dir=str(local_dir),
                    force_download=False,
                )

            logger.info(
                "download_completed",
                repo_id=repo_id,
                local_path=snapshot_path,
            )

            # Gather file information for the response
            file_count = 0
            total_size = 0
            if Path(snapshot_path).exists():
                for file_path in Path(snapshot_path).rglob("*"):
                    if file_path.is_file():
                        file_count += 1
                        total_size += file_path.stat().st_size

            return {
                "repo_id": repo_id,
                "artifact_type": artifact_type,
                "download_dir": str(snapshot_path),
                "revision": revision,
                "file_count": file_count,
                "total_size": total_size,
            }

        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                logger.error("download_failed_not_found", repo_id=repo_id, error=str(e))
                raise RuntimeError(
                    f"Repository not found: '{repo_id}'. "
                    "Please check the repository ID and ensure it exists on Hugging Face Hub."
                ) from e
            elif e.response is not None and e.response.status_code == 403:
                logger.error(
                    "download_failed_permission_denied", repo_id=repo_id, error=str(e)
                )
                raise RuntimeError(
                    f"Access denied to '{repo_id}'. "
                    "The repository may be private. "
                    "Please ensure your HuggingFace token has appropriate permissions."
                ) from e
            logger.error("download_failed", error=str(e), repo_id=repo_id)
            raise RuntimeError(f"Failed to download from Hugging Face: {e}") from e
        except Exception as e:
            logger.error("download_failed", error=str(e), repo_id=repo_id)
            raise RuntimeError(f"Failed to download artifact: {e}") from e

    def _compute_hash(self, path: Path) -> str:
        """Compute SHA256 hash of file or directory.

        Args:
            path: File or directory path.

        Returns:
            Hex string of SHA256 hash.
        """
        hash_sha256 = hashlib.sha256()

        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        else:
            # For directories, hash all files in sorted order
            for file in sorted(path.rglob("*")):
                if file.is_file():
                    with open(file, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_sha256.update(chunk)

        return hash_sha256.hexdigest()
