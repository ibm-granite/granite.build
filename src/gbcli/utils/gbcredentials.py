"""Gbcredentials module."""

import logging
import os
import sys

import portalocker
import requests
import toml
from requests import HTTPError
from requests.exceptions import ConnectionError
from toml import TomlDecodeError

from gbcli.utils.cli_config import get_local_gb_config
from gbcli.utils.gbconstants import USER_NOT_LOGGED_IN_ERROR_MESSAGE, is_standalone

logger = logging.getLogger(__name__)


class ConfigLockException(Exception):
    """Custom exception for locking errors."""

    pass


class GBTomlConfig:
    """
    The utility class that supports read/write of a config file in TOML format.
    It can be a base class for any config file in this format
    """

    def __init__(self, config_path):
        self._config = {}
        self._config_path = config_path
        self._config_dir = os.path.dirname(config_path)
        self.load()

    def load(self):
        """Parse ."""
        if os.path.isfile(self._config_path):
            self._config = toml.load(self._config_path)

    def save(self):
        """Save ."""
        os.makedirs(self._config_dir, exist_ok=True)
        try:
            with portalocker.Lock(self._config_path, "w", timeout=3) as configfile:
                toml.dump(self._config, configfile)

                if self._config_path.endswith("credentials"):
                    os.chmod(
                        self._config_path, 0o600
                    )  # setting permission to -rw------ for only credentials

        except portalocker.AlreadyLocked as e:
            raise ConfigLockException(
                f"Error: {self._config_path} is being modified by another process: {str(e)}"
            )

    def get(self, key, section="default"):
        # TODO: replace with more advanced toml parser to handle nested section names
        """Get the ."""
        s = self._config
        try:
            for x in section.split("."):
                s = s[x]
            return s[key]
        except KeyError:
            return None

    def get_section(self, section):
        """
        return a dictionary for all values defined in a section
        """
        # TODO: replace with more advanced toml parser to handle nested section names
        s = self._config
        try:
            for x in section.split("."):
                s = s[x]
            return s
        except KeyError:
            return None

    def set(self, key, value, section="default"):
        # TODO: replace with more advanced toml parser to handle nested section names
        """Set the ."""
        s = self._config
        for x in section.split("."):
            if not x in s.keys():
                s[x] = {}
            s = s[x]
        s[key] = value


class GBCredentials(GBTomlConfig):
    """
    The utility class that supports read/write of the per-user GB_CONFIG credential config file (default i.e. ~/.gbcli/credentials).
    """

    def __init__(self):
        credential_path = os.path.abspath(os.path.join(get_local_gb_config(), "credentials"))
        try:
            super().__init__(credential_path)
        except TomlDecodeError as e:
            sys.exit(
                f"❌ Error: {credential_path} can't be parsed: {str(e)}\nPlease run 'llmb cleanup --credentials' and try again."
            )

    def check_permissions(self):
        """Verify permissions."""
        if sys.platform.startswith("linux"):
            st = os.stat(self._config_path)
            oct_perm = oct(st.st_mode)[-3:]

            if int(oct_perm) > 600:
                logger.warning(
                    f"Warning: the permissions of the LLM.build credentials file are too open. Run 'chmod 600 {self._config_path}', or redo 'llmb login auth'."
                )

    def check_gbserver_values(self):
        """Verify gbserver values."""
        credentials = [
            self.get("api_key", section="user.gbserver"),
            self.get("login", section="user.gbserver"),
        ]
        return all(val is not None for val in credentials)

    def check_ibmid_values(self):
        """Return True if IBMid credentials are present."""
        credentials = [
            self.get("access_token", section="user.ibmid"),
            self.get("id_token", section="user.ibmid"),
            self.get("login", section="user.ibmid"),
            self.get("email", section="user.ibmid"),
        ]
        return all(val is not None for val in credentials)

    def check_values(self):
        """Verify values."""
        credentials = [
            self.get("token", section="user.github"),
            self.get("login", section="user.github"),
            self.get("email", section="user.github"),
        ]

        if any(val is None for val in credentials):
            return False
        else:
            # Validate token
            headers = {"Authorization": f'token {self.get("token", section="user.github")}'}

            try:
                response = requests.get("https://api.github.ibm.com/user", headers=headers)
                response.raise_for_status()
                return True

            except ConnectionError as e:
                logger.error(
                    f"Error: Unable to connect to network. Please check network connection."
                )
                sys.exit(1)

            except HTTPError as e:
                if e.response.status_code == 401:
                    logger.error(
                        f"Error: GitHub token is invalid. Please reauthenticate by obtaining a new token with auth login."
                    )
                else:
                    logger.error(str(e))
                sys.exit(1)

            except Exception as e:
                logger.error(str(e))
                sys.exit(1)


def get_user_token() -> str:
    """Get the user token for API authentication.

    In standalone mode (GB_ENVIRONMENT=STANDALONE), checks [user.gbserver] api_key
    in credentials first, then falls back to GBSERVER_API_KEY env var.
    Otherwise, checks the ``default_provider`` setting to return the
    appropriate token (IBMid or GitHub).
    """
    if is_standalone():
        creds = GBCredentials()
        api_key = creds.get("api_key", section="user.gbserver")
        if api_key:
            return api_key
        return os.environ.get("GBSERVER_API_KEY", "")

    creds = GBCredentials()
    default_provider = creds.get("default_provider", section="user")

    if default_provider == "ibmid":
        if creds.check_ibmid_values():
            # Check token expiry before making any API call
            expires_at = creds.get("expires_at", section="user.ibmid")
            if expires_at is not None:
                import time as _time

                remaining = int(expires_at) - int(_time.time())
                if remaining < 0:
                    sys.exit(
                        "❌ Error: IBMid token has expired. "
                        "Please re-authenticate with 'auth login --sso ibm'."
                    )
                elif remaining < 300:
                    logger.warning(
                        "IBMid access token expires in %d seconds. "
                        "Consider re-authenticating with 'auth login --sso ibm'.",
                        remaining,
                    )

            creds.check_permissions()
            token = creds.get("id_token", section="user.ibmid")
            return token or ""

    if default_provider == "apikey":
        if creds.check_gbserver_values():
            creds.check_permissions()
            token = creds.get("api_key", section="user.gbserver")
            return token or ""

    # Default: GitHub
    if not creds.check_values():
        sys.exit(f"❌ {USER_NOT_LOGGED_IN_ERROR_MESSAGE}")

    creds.check_permissions()

    token = creds.get("token", section="user.github")
    return token or ""


class GBConfig(GBTomlConfig):
    """
    The utility class that supports read/write of the per-user GB_CONFIG config file (i.e. defaults to ~/.gbcli/config).

    seperate from credentials, this stores user-specifc data such as user spaces
    """

    def __init__(self):
        credential_path = os.path.abspath(os.path.join(get_local_gb_config(), "config"))
        try:
            super().__init__(credential_path)
        except TomlDecodeError as e:
            sys.exit(
                f"❌ Error: {credential_path} can't be parsed: {str(e)}\nPlease run 'llmb cleanup --config' and try again."
            )
