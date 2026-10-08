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

"""The default local secret store keys each space's directory on its space.yaml
``name`` (space_secrets/<name>/). Registered names are unique but space.yaml names
are not, so registering a second, different space with the same space.yaml name
would make the two share secrets. Registration must refuse it.
"""

import logging
from pathlib import Path

import pytest

from gbserver.spaces.local_secrets_isolation import (
    LocalSecretsCollisionError,
    check_local_secrets_collision,
    default_local_secrets_key,
    load_space_config,
)
from gbserver.storage.stored_space import StoredSpace

LOCAL_DEFAULT = "secret_manager:\n  type: local\n  config: {}\n"


def _space(tmp_path: Path, dirname: str, yaml_name: str, sm: str = LOCAL_DEFAULT):
    space_dir = tmp_path / dirname
    space_dir.mkdir()
    (space_dir / "space.yaml").write_text(f"name: {yaml_name}\n{sm}")
    return space_dir.as_uri()


def _stored(name: str, uri: str) -> StoredSpace:
    return StoredSpace(name=name, git_repo_uri=uri, lakehouse_namespace="")


def test_same_yaml_name_different_space_rejected(tmp_path):
    a = _space(tmp_path, "a", "local")
    b = _space(tmp_path, "b", "local")
    with pytest.raises(LocalSecretsCollisionError, match="team-a"):
        check_local_secrets_collision(_stored("team-b", b), [_stored("team-a", a)])


@pytest.mark.parametrize("yaml_name", ['""', "''"])
def test_default_local_secrets_without_name_rejected(tmp_path, yaml_name):
    # space.yaml's name defaults to "" -- there is no directory to key on, so
    # every build in the space would fail; refuse it once, at registration.
    uri = _space(tmp_path, "a", yaml_name)
    with pytest.raises(LocalSecretsCollisionError, match="no `name:`"):
        check_local_secrets_collision(_stored("team-a", uri), [])


def test_explicit_secrets_dir_without_name_allowed(tmp_path):
    explicit = "secret_manager:\n  type: local\n  config:\n    secrets_dir: /x\n"
    uri = _space(tmp_path, "a", '""', explicit)
    check_local_secrets_collision(_stored("team-a", uri), [])


def test_aliases_of_the_same_space_allowed(tmp_path):
    uri = _space(tmp_path, "a", "public")
    check_local_secrets_collision(_stored("local", uri), [_stored("public", uri)])


def test_different_yaml_names_allowed(tmp_path):
    a = _space(tmp_path, "a", "alpha")
    b = _space(tmp_path, "b", "beta")
    check_local_secrets_collision(_stored("b", b), [_stored("a", a)])


def test_explicit_secrets_dir_is_not_shared(tmp_path):
    a = _space(tmp_path, "a", "local")
    explicit = "secret_manager:\n  type: local\n  config:\n    secrets_dir: /x\n"
    b = _space(tmp_path, "b", "local", explicit)
    check_local_secrets_collision(_stored("b", b), [_stored("a", a)])
    check_local_secrets_collision(_stored("a", a), [_stored("b", b)])


def test_hybrid_wrapping_default_local_counts(tmp_path):
    hybrid = (
        "secret_manager:\n  type: hybrid\n  config:\n    managers:\n"
        "      - type: env\n        config: {}\n"
        "      - type: local\n        config: {}\n"
    )
    a = _space(tmp_path, "a", "local", hybrid)
    b = _space(tmp_path, "b", "local")
    with pytest.raises(LocalSecretsCollisionError):
        check_local_secrets_collision(_stored("b", b), [_stored("a", a)])


def test_non_local_manager_has_no_key(tmp_path):
    uri = _space(tmp_path, "a", "local", "secret_manager:\n  type: env\n  config: {}\n")
    assert default_local_secrets_key(load_space_config(uri)) is None


def test_unreadable_registered_space_is_skipped_with_warning(tmp_path, caplog):
    b = _space(tmp_path, "b", "local")
    gone = (tmp_path / "deleted").as_uri()
    with caplog.at_level(logging.WARNING):
        check_local_secrets_collision(_stored("b", b), [_stored("old", gone)])
    assert any("old" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ registration paths


class _FakeSpaceStorage:
    """The slice of the admin space storage the registration paths use."""

    def __init__(self, spaces):
        self.rows = {s.name: s for s in spaces}

    def get_by_uuid(self, uuids):
        assert uuids is None
        return list(self.rows.values())

    def get_by_name(self, name):
        return self.rows.get(name)

    def add(self, items):
        for item in items if isinstance(items, list) else [items]:
            self.rows[item.name] = item

    def update(self, item, create_if_not_exist=False):
        self.rows[item.name] = item


def test_standalone_registration_rejects_colliding_space(tmp_path):
    from gbserver.commands.utils import register_standalone_space

    other = _space(tmp_path, "other", "public")
    standalone_dir = Path(_space(tmp_path, "standalone", "public")[len("file://") :])
    storage = _FakeSpaceStorage([_stored("team-x", other)])

    with pytest.raises(LocalSecretsCollisionError, match="team-x"):
        register_standalone_space(storage, str(standalone_dir))
    assert set(storage.rows) == {"team-x"}  # nothing registered


def test_standalone_registration_repoints_its_own_aliases(tmp_path):
    from gbserver.commands.utils import register_standalone_space

    old = _space(tmp_path, "old", "public")
    new_dir = Path(_space(tmp_path, "new", "public")[len("file://") :])
    storage = _FakeSpaceStorage(
        [_stored(n, old) for n in ("public", "standalone", "local")]
    )

    register_standalone_space(storage, str(new_dir))  # not a collision
    assert {s.git_repo_uri for s in storage.rows.values()} == {f"file://{new_dir}"}


def test_create_spaces_rejects_collisions_within_and_across_runs(tmp_path):
    from gbserver.commands.command_create_spaces import (
        _check_local_secrets_collisions,
    )

    a = _space(tmp_path, "a", "local")
    b = _space(tmp_path, "b", "local")
    c = _space(tmp_path, "c", "local")

    with pytest.raises(LocalSecretsCollisionError):  # against a registered space
        _check_local_secrets_collisions(
            _FakeSpaceStorage([_stored("team-a", a)]), [_stored("team-b", b)]
        )
    with pytest.raises(LocalSecretsCollisionError):  # two new spaces in one run
        _check_local_secrets_collisions(
            _FakeSpaceStorage([]), [_stored("team-b", b), _stored("team-c", c)]
        )
    # Re-creating a space under its own name (--replace) is not a collision.
    _check_local_secrets_collisions(
        _FakeSpaceStorage([_stored("team-a", a)]), [_stored("team-a", c)]
    )
