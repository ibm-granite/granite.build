import os
import secrets
from pathlib import Path
from typing import Any, ClassVar, Optional
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from libgbtest.constants import (
    GBTEST_ADMIN_GITHUB_TOKEN,
    GBTEST_NON_ADMIN_GITHUB_TOKEN,
)
from libgbtest.mode import is_mock_mode
from libgbtest.utils import AbstractSingletonStorageUsingPreloadedSpaceTest

from gbserver.api.auth import _make_synthetic_user, get_gh_user
from gbserver.api.builds import BuildStatusResponse
from gbserver.api.root_api import root_api
from gbserver.lineage.jobstats import reset_lineage_store
from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory
from gbserver.storage.storage_factory import StorageFactory
from gbserver.storage.stored_space import StoredSpace
from gbserver.storage.stored_space_user import StoredSpaceUser
from gbserver.types.auth import User
from gbserver.types.constants import (
    ENV_VAR_PREFIX,
    GBSERVER_GITHUB_TOKEN,
    PUBLIC_SPACE_LH_NAMESPACE,
    PUBLIC_SPACE_NAME,
)

# The repo's shared local space; the standalone API tests' "public" space points
# at it so space:// URIs resolve without cloning from GitHub.
_LOCAL_SPACE_DIR = Path(__file__).resolve().parents[3] / "configurations/spaces/local"


class AbstractAPITest(AbstractSingletonStorageUsingPreloadedSpaceTest):
    """Base for REST API tests, authenticating with GitHub tokens.

    Tests name callers through the identity hooks (``default_identity``,
    ``admin_identity``, ``non_admin_identity``) and resolve them with
    ``get_test_client`` / ``get_caller_*``, so the same test body runs under any
    auth mode.  Here an identity is a GitHub token; :class:`AbstractStandaloneAPITest`
    makes it an apikey username instead.
    """

    auth_mode: ClassVar[str] = "github"
    """The ``GBSERVER_AUTH_MODE`` these tests authenticate with."""

    @pytest.fixture(autouse=True)
    def _apply_server_env(self):
        """Apply :meth:`_server_env` to the environment for the duration of each test.

        The lineage store is a singleton that caches the provider it was created
        with, so it is reset on entry and exit. Otherwise a store built before this
        fixture ran (at session start, or by an earlier fixture) would ignore the
        provider set here, e.g. leave a standalone test on a WandB store or an IBM
        test on the no-op store, whose lineage assertions are skipped.
        """
        with patch.dict(os.environ, self._server_env()):
            reset_lineage_store()
            yield
        reset_lineage_store()

    def _server_env(self) -> dict[str, str]:
        """Env vars that configure the server (auth, lineage) for these tests.

        Read by the server at request time. None here: use the environment's.
        """
        return {}

    def default_identity(self) -> Optional[str]:
        """Identity of the default caller (the GBSERVER_GITHUB_TOKEN user)."""
        return GBSERVER_GITHUB_TOKEN

    def admin_identity(self) -> Optional[str]:
        """Identity of a caller that tests make a super-admin, or None if unavailable."""
        return GBTEST_ADMIN_GITHUB_TOKEN

    def non_admin_identity(self) -> Optional[str]:
        """Identity of a caller that is never an admin, or None if unavailable."""
        return GBTEST_NON_ADMIN_GITHUB_TOKEN

    def require_admin_and_non_admin(self) -> tuple[str, str]:
        """Return (admin, non-admin) identities, skipping the test if they can't be distinct.

        In mock mode the server runs apikey auth and maps every GitHub token to one
        synthetic user, so GitHub-auth tests can't tell admin from non-admin there.

        Returns:
            The admin and non-admin identities.

        Raises:
            pytest.skip.Exception: if the two identities are unavailable or not distinct.
        """
        if self.auth_mode == "github" and is_mock_mode():
            pytest.skip(
                reason="apikey auth maps every token to one synthetic user; "
                "admin/non-admin authorization requires live GitHub identities"
            )
        admin, non_admin = self.admin_identity(), self.non_admin_identity()
        if admin is None or non_admin is None:
            pytest.skip(
                reason="No admin/non-admin identities available in the environment"
            )
        return admin, non_admin

    def get_caller_user(self, identity: Optional[str] = None) -> User:
        """Resolve the user the server sees for ``identity`` (default: :meth:`default_identity`).

        Raises:
            AssertionError: if the user cannot be resolved from the GitHub token.
        """
        __tracebackhide__ = True  # Hide token during stack traces
        user, _ = get_gh_user(identity or self.default_identity() or "")
        assert user is not None and isinstance(
            user, User
        ), "Could not get username/login from git token"
        return user

    def get_caller_username(self, identity: Optional[str] = None) -> str:
        """Login of the user the server sees for ``identity``."""
        return self.get_caller_user(identity).login

    def get_caller_email(self, identity: Optional[str] = None) -> str:
        """Email (the space-access identity) of the user the server sees for ``identity``."""
        return self.get_caller_user(identity).email

    def grant_super_admin(self, identity: Optional[str] = None) -> None:
        """Register ``identity``'s caller as an admin of the public space.

        Under the storage-backed ``StorageSpaceAccessManager``, read/write
        endpoints require the caller to own the item or be a space admin, and
        list/count endpoints only return rows from the caller's spaces. An admin of
        the public space is a super-admin, unrestricted in any (including
        synthetic) space, so tests that work across arbitrary spaces call this.

        The identity registered must match whatever the auth middleware attaches
        to the request, which :meth:`get_caller_email` resolves for the test's
        auth mode (the synthetic apikey user, or the GitHub token's user).

        Args:
            identity: The caller to register; defaults to the identity used by
                :meth:`get_test_client`.

        Raises:
            AssertionError: in GitHub auth, if the user cannot be resolved from
                the token.
        """
        self.storage.space_user_storage.add(
            StoredSpaceUser(
                space_name=PUBLIC_SPACE_NAME,
                username=self.get_caller_email(identity),
                role="admin",
            )
        )

    def get_test_client(self, identity: Optional[str] = None) -> TestClient:
        """Get a TestClient that authenticates to the server as ``identity``.

        Args:
            identity: GitHub token; defaults to :meth:`default_identity`, then $GITHUB_TOKEN.

        Returns:
            TestClient: a client sending the token as a Bearer credential.
        """
        return github_test_client(identity or self.default_identity())


def github_test_client(token: Optional[str] = GBSERVER_GITHUB_TOKEN) -> TestClient:
    """Get a TestClient that sends ``token`` (or $GITHUB_TOKEN) as a Bearer credential.

    Raises:
        AssertionError: if neither ``token`` nor $GITHUB_TOKEN is set.
    """
    __tracebackhide__ = True  # Hide token during stack traces
    if token == None or token == "":
        token = os.environ.get("GITHUB_TOKEN")
        assert (
            token != None
        ), "GBSERVER_GITHUB_TOKEN or GITHUB_TOKEN env var must be set to enable server authentication"
    return TestClient(root_api, headers={"authorization": "Bearer " + token})


class _ApiKeyUserClient(TestClient):
    """TestClient that authenticates with the API key as a chosen apikey user.

    The apikey middleware reads ``GBSERVER_API_USER`` on every request, so each
    request sets it to this client's user — letting one test act as several users.
    """

    def __init__(self, api_key: str, user: str) -> None:
        super().__init__(root_api, headers={"authorization": f"Bearer {api_key}"})
        self._user = user

    def request(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        """Send the request with ``GBSERVER_API_USER`` set to this client's user."""
        with patch.dict(os.environ, {"GBSERVER_API_USER": self._user}):
            return super().request(*args, **kwargs)


# apikey auth never contacts GitHub, so opt out of conftest's mock-mode GitHub
# fixtures: they also bypass space-access checks (is_super_admin -> True), which
# would defeat the admin/non-admin authorization assertions.
@pytest.mark.live("github")
class AbstractStandaloneAPITest(AbstractAPITest):
    """Base for REST API tests on apikey auth, with no GitHub or IBM dependency.

    Identities are apikey usernames; storage is SQLite; the preloaded "public"
    space is the repo's local space; no lineage provider is used.
    """

    auth_mode: ClassVar[str] = "apikey"
    _API_KEY: ClassVar[str] = secrets.token_urlsafe(32)

    @classmethod
    def _get_storage_factory(cls) -> StorageFactory:
        return SqliteStorageFactory()

    def _get_preloaded_spaces(self) -> list[StoredSpace]:
        return [
            StoredSpace(
                name=PUBLIC_SPACE_NAME,
                git_repo_uri=f"file://{_LOCAL_SPACE_DIR}",
                lakehouse_namespace=PUBLIC_SPACE_LH_NAMESPACE,
            )
        ]

    def _server_env(self) -> dict[str, str]:
        # No lineage provider: outside STANDALONE it would default to WandB.
        return {
            "GBSERVER_AUTH_MODE": "apikey",
            "GBSERVER_API_KEY": self._API_KEY,
            f"{ENV_VAR_PREFIX}_LINEAGE_PROVIDER": "none",
        }

    def default_identity(self) -> Optional[str]:
        return "standalone"

    def admin_identity(self) -> Optional[str]:
        return "gbtest-admin"

    def non_admin_identity(self) -> Optional[str]:
        return "gbtest-nonadmin"

    def get_caller_user(self, identity: Optional[str] = None) -> User:
        """The synthetic user the apikey middleware builds for ``identity``."""
        return _make_synthetic_user(identity or self.default_identity() or "")

    def get_test_client(self, identity: Optional[str] = None) -> TestClient:
        """A client sending the API key and acting as apikey user ``identity``."""
        return _ApiKeyUserClient(
            self._API_KEY, identity or self.default_identity() or ""
        )


if __name__ == "__main__":
    client = github_test_client()
    id = "39bbdc33-cfb2-4113-accc-c180aa3cd483"
    url = f"api/v1/builds/{id}/status"
    resp = client.get(url)
    print(f"\nurl={url}")
    resp_json = resp.json()
    # print(f"\nresp.content={resp.content}")
    print(f"\njson resp={resp_json}")
    resp: BuildStatusResponse = BuildStatusResponse.model_validate(resp_json)
    print(f"\n\nbuild status={resp}")
