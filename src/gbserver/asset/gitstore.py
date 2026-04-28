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

"""Access assets in local/remote git repos."""

import os
from pathlib import Path
from typing import Dict, Optional, Union

from git import Repo

from gbcommon.uri.git import GitURI
from gbcommon.uri.uri import URI
from gbserver.asset.assetstore import Assetstore
from gbserver.utils.git_retry import git_clone_retry
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class Gitstore(Assetstore):
    """Git-based Assetstore with https/ssh secret-based auth"""

    def __init__(self, uri: Union[URI, str], **kwargs):
        super().__init__(uri, **kwargs)

    @classmethod
    def get_supported_uri_classes(self):
        return [GitURI]

    def get_metadata(self, uri: URI) -> Dict:
        """
        Return metadata describing which secret names are required,
        based on the scheme.
        """
        if uri.scheme == "git+https":
            token_key = (
                self.config.config["token_secretname"]
                if self.config
                and isinstance(self.config.config, dict)
                and "token_secretname" in self.config.config
                else "GITHUB_PAT_TUNING"
            )
            return {"token_secretname": token_key}

        if uri.scheme == "git+ssh":
            ssh_key_key = (
                self.config.config["ssh_key_secretname"]
                if self.config
                and isinstance(self.config.config, dict)
                and "ssh_key_secretname" in self.config.config
                else "GIT_SSH_KEY"
            )
            return {"ssh_key_secretname": ssh_key_key}

        return {}

    def _resolve_secret(self, uri) -> str:
        """
        Resolve Git secret (HTTPS token or SSH key) for the given URI.
        Looks first in `self.secrets`, then falls back to environment variables.
        Returns None if no secret found.
        """
        metadata = self.get_metadata(uri)

        # Determine which secret to fetch
        if uri.scheme == "git+https":
            secret_name = metadata.get("token_secretname")
        elif uri.scheme == "git+ssh":
            secret_name = metadata.get("ssh_key_secretname")
        else:
            raise ValueError(f"Unsupported URI scheme: {uri.scheme}")

        secret = self.secrets.get(secret_name) if self.secrets else None
        if secret is None:
            secret = os.getenv(secret_name, None)

        if secret is not None and secret != "" and secret.strip() == "":
            secret = None

        return secret

    # def pull(self, uri, dest: str, ref: Optional[str] = None):
    #     """
    #     Clone from Git repo (supports https+token or ssh+key).
    #     Resolves secrets based on metadata and environment.
    #     """
    #     secrets_dict = {}
    #     metadata = self.get_metadata(uri)
    #     dest_path = Path(dest)

    #     # If destination already exists, skip download
    #     if dest_path.exists() and any(dest_path.iterdir()):
    #         logger.info("Destination %s already exists. Skipping git clone.", dest)
    #         return dest_path

    #     if uri.scheme == "git+https":
    #         token = self._resolve_secret(uri)
    #         if not token:
    #             raise ValueError(
    #                 f"Failed to retrieve HTTPS token for git from URI {uri}"
    #             )
    #         token_key = self.get_metadata(uri)["token_secretname"]
    #         secrets_dict[token_key] = token

    #     elif uri.scheme == "git+ssh":
    #         ssh_key = self._resolve_secret(uri)
    #         ssh_key_key = self.get_metadata(uri)["ssh_key_secretname"]
    #         secrets_dict[ssh_key_key] = ssh_key

    #         # For SSH, we now:
    #         # Write the key secret to a temp file.
    #         # Use GIT_SSH_COMMAND so Git uses it.
    #         # Disable StrictHostKeyChecking (otherwise we hang on unknown hosts).

    #         import stat
    #         import tempfile

    #         key_file = tempfile.NamedTemporaryFile(
    #             delete=False, mode="w", prefix="git_ssh_key_", suffix=".pem"
    #         )
    #         key_file.write(ssh_key.strip() + "\n")
    #         key_file.flush()
    #         key_file.close()
    #         os.chmod(key_file.name, stat.S_IRUSR | stat.S_IWUSR)

    #         git_env = os.environ.copy()
    #         git_env["GIT_SSH_COMMAND"] = (
    #             f"ssh -i {key_file.name} -o StrictHostKeyChecking=no"
    #         )
    #         # The below commented line is to pass no key to the git ssh command
    #         # git_env['GIT_SSH_COMMAND'] = 'ssh -i /dev/null -o StrictHostKeyChecking=no'

    #     git_uri = GitURI(uri)
    #     repo_url = git_uri.get_repo_url()

    #     logger.info("Cloning from repo %s into %s", repo_url, dest)

    #     clone_kwargs = {"depth": 1}
    #     if ref:
    #         clone_kwargs.update({"branch": ref, "single_branch": True})

    #     if uri.scheme == "git+ssh":
    #         repo = self._clone_with_retry(repo_url, dest, env=git_env, **clone_kwargs)
    #     else:
    #         repo = self._clone_with_retry(repo_url, dest, **clone_kwargs)

    #     return repo

    # @git_clone_retry
    # def _clone_with_retry(
    #     self, repo_url: str, dest: str, env: dict = None, **clone_kwargs
    # ) -> Repo:
    #     """Clone repository with retry logic for transient failures."""
    #     if env:
    #         return Repo.clone_from(repo_url, dest, env=env, **clone_kwargs)
    #     return Repo.clone_from(repo_url, dest, **clone_kwargs)


# Below lines can be removed after PR approval - this is intended for testing purposes

# uri_str = "git+ssh://github.ibm.com/granite-dot-build/gbspace-tuning.git"
# git_uri = GitURI.parse(uri_str)

# Load  SSH key from ~/.ssh/id_ed25519
# ssh_key_path = Path.home() / ".ssh/id_ed25519"
# with open(ssh_key_path, "r") as f:
#     test_ssh_key = f.read()

# secrets = {"GIT_SSH_KEY": test_ssh_key}

# # Create Gitstore
# store = Gitstore(git_uri, secrets=secrets)
# resolved_token = store._resolve_secret(git_uri)
# print("Resolved secret:", resolved_token)
# # Destination path
# dest = "/tmp/test-git-ssh"

# print("Cloning via SSH into", dest)
# repo = store.pull(git_uri,dest=dest)
# print("Repo cloned successfully:", repo)
