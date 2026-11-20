# SPDX-License-Identifier: Apache-2.0

"""The admin storage isolation in test/unit/conftest.py.

CI runs integration and e2e tests on the same xdist workers as the unit tests,
and one of them can leave the admin storage singleton built on its own db,
possibly the developer's real one. Each unit test must still build one on its
own temp db.
"""

import pytest

from gbserver.storage import singleton_storage


@pytest.fixture(scope="module")
def leaked_admin_storage(tmp_path_factory):
    """Stand in for a singleton an earlier, non-unit test left built on its db."""
    # Import root_api first, as an earlier test on the worker has: its first
    # import runs the gbserver CLI init, which replaces the storage factory
    # (dropping a cached singleton). Not at module level: imported during
    # collection, that init's configure_logging() would no longer remove
    # pytest's live-logging handler, which then swaps sys.stdout out from under
    # CliRunner in later tests.
    # pylint: disable-next=unused-import,import-outside-toplevel
    import gbserver.api.root_api  # noqa: F401

    saved = getattr(singleton_storage, "__STORAGE")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GB_HOME_DIR", str(tmp_path_factory.mktemp("leaked_gb_home")))
        singleton_storage.reset_admin_storage()
        leaked = singleton_storage.get_admin_storage()
    yield leaked
    setattr(singleton_storage, "__STORAGE", saved)


def test_a_unit_test_does_not_reuse_a_leaked_admin_storage(leaked_admin_storage):
    assert singleton_storage.get_admin_storage() is not leaked_admin_storage
