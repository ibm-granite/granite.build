import os
import shutil
from typing import Optional

import click

from gbcommon.types.gbenvconfig import is_standalone
from gbserver.storage import singleton_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.constants import STANDALONE_ENV_DEFAULTS
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class MutexOption(click.Option):
    """Enables options that are mutually exclusive.
    Borrowed from https://stackoverflow.com/questions/44247099/click-command-line-interfaces-make-options-required-if-other-optional-option-is
    Usage, for example:
        @click.option("--username", prompt=True, cls=MutexOption, not_required_if=["token"])
        @click.option("--password", prompt=True, hide_input=True, cls=MutexOption, not_required_if=["token"])
        @click.option("--token", cls=MutexOption, not_required_if=["username","password"])
    """

    def __init__(self, *args, **kwargs):
        self.not_required_if: list = kwargs.pop("not_required_if")

        assert self.not_required_if, "'not_required_if' parameter required"
        kwargs["help"] = (
            kwargs.get("help", "")
            + "  Option is mutually exclusive with "
            + ", ".join(self.not_required_if)
            + "."
        ).strip()
        super(MutexOption, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        current_opt: bool = self.name in opts
        for mutex_opt in self.not_required_if:
            if mutex_opt in opts:
                if current_opt:
                    raise click.UsageError(
                        "Illegal usage: '"
                        + str(self.name)
                        + "' is mutually exclusive with "
                        + str(mutex_opt)
                        + "."
                    )
                else:
                    self.prompt = None
        return super(MutexOption, self).handle_parse_result(ctx, opts, args)


def set_failed_build_status(build_id: str):
    # DEPRECATED in favor of finalize_build_status()
    # Don't update targets or steps if already FAILED or SUCCESS or CANCELED w/o updating the update_time field
    _set_build_status(
        build_id, status=Status.FAILED, unfinished_targets_and_steps_only=True
    )


def _set_build_status(
    build_id: str, status: Status, unfinished_targets_and_steps_only: bool = False
):
    """Set the build, target, and step status. w/o updating the update_time field

    Args:
        build_id (str): _description_
        status (Status): _description_
    """
    admin_storage = singleton_storage.get_admin_storage()
    build_storage = admin_storage.build_storage
    target_storage = admin_storage.target_storage
    step_storage = admin_storage.step_storage

    # Update the targets and steps first so that the build's status is cleared last.

    for target in target_storage.get_by_where({"build_id": build_id}):
        assert isinstance(target, StoredTargetRun)
        if not unfinished_targets_and_steps_only or not target.status.is_finished():
            target.status = status
            target_storage.update(target, update_updated_time=False)
            print(f"Updated status of target with id {target.uuid} to {status}")

    for step in step_storage.get_by_where({"build_id": build_id}):
        assert isinstance(step, StoredStepRun)
        if not unfinished_targets_and_steps_only or not step.status.is_finished():
            step.status = status
            step_storage.update(step, update_updated_time=False)
            print(f"Updated status of step with id {step.uuid} to {status}")

    # Do this last so that the build's status is the trigger that brings us here on
    # the next run of a run that was interrupted above.
    build = build_storage.get_by_uuid(build_id)
    assert build is not None, f"Build with id {build_id} not found in build storage"
    assert isinstance(build, StoredBuild)
    build.status = status
    build_storage.update(build, update_updated_time=False)
    print(f"Build with id {build_id} status updated to {status}")


def _migrate_legacy_sqlite_db() -> None:
    """One-time migration of the standalone SQLite db from ~/.llmb to the GB home dir.

    Earlier versions stored the standalone metadata db at ``~/.llmb/llmb-server.db``.
    State now lives under the consolidated GB home dir (default ~/.granite.build).
    If the new db does not yet exist but a legacy one does, copy it across so existing
    standalone deployments keep their builds/spaces. The legacy file is left in place as
    a backup, and an existing new db is never overwritten. Safe to call repeatedly.

    A copy failure is raised rather than swallowed: continuing would let the storage
    factory create a fresh empty db, silently abandoning the user's migrated history.
    """
    from gbcommon.types.constants import get_gb_home_dir
    from gbserver.storage.sqlite.sqlite_storage import (
        LEGACY_LLMB_DIR_NAME,
        SQLITE_DB_FILE_NAME,
    )

    gb_home_dir = get_gb_home_dir()
    new_db = os.path.join(gb_home_dir, SQLITE_DB_FILE_NAME)
    legacy_db = os.path.join(
        os.path.expanduser("~"), LEGACY_LLMB_DIR_NAME, SQLITE_DB_FILE_NAME
    )
    if os.path.exists(new_db):
        return  # New db is the source of truth; never overwrite.
    if not os.path.exists(legacy_db):
        return  # Nothing to migrate.
    os.makedirs(gb_home_dir, exist_ok=True)
    shutil.copy2(legacy_db, new_db)
    logger.info(
        "Migrated legacy standalone db %s -> %s (legacy file left as backup)",
        legacy_db,
        new_db,
    )


def check_and_init_for_standalone(space_dir: Optional[str] = None) -> None:
    """Initialize the process for standalone mode; no-op outside standalone.

    Safe to call near the top of every gbserver command: it returns immediately
    unless ``GB_ENVIRONMENT=STANDALONE``, so non-standalone (cloud) runs are
    unaffected. In standalone mode it performs the one-time process setup the
    command needs before doing any storage work:

    1. Apply standalone-friendly env-var defaults (only where not already set, so
       the user can override them).
    2. Reload ``gbserver.types.constants`` so values captured at import time pick
       up the just-applied defaults.
    3. Migrate any legacy SQLite db before the storage factory opens it, then
       install the SQLite storage factory.
    4. Install the standalone space access manager (bypasses Lakehouse auth).
    5. If ``space_dir`` is given, register the standalone space under its current
       name and legacy aliases (only the standalone *server* knows the space dir;
       other commands pass nothing and rely on the already-registered space).

    The storage/space imports are deliberately function-local so they run *after*
    the constants reload (those modules capture constants at import time).

    Args:
        space_dir: Optional path to the space directory (contains space.yaml,
            environments/, steps/). When provided, the space is registered under
            'public' and the legacy aliases 'standalone' and 'local'. When None,
            space registration is skipped.

    Returns:
        None.
    """
    # Only the standalone environment needs this setup; everywhere else this is a
    # no-op so the call can sit unconditionally at the top of each command.
    if not is_standalone():
        return

    # 1. Apply the shared standalone env-var defaults (single source of truth in
    #    constants.STANDALONE_ENV_DEFAULTS), only where not already set so the
    #    user can override. GB_ENVIRONMENT is not in that dict — it's the trigger
    #    (the is_standalone() guard above), already set by the time we get here.
    for key, value in STANDALONE_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)

    # 2. Re-evaluate constants captured at import time before our env-var
    #    defaults were applied (e.g. GB_METADATA_STORAGE, GB_ENVIRONMENT).
    import importlib

    import gbserver.types.constants

    importlib.reload(gbserver.types.constants)

    # 3. Force SQLite storage — standalone always uses SQLite.
    from gbserver.storage.sqlite.storage_factory import SqliteStorageFactory

    # Migrate any legacy ~/.llmb db into GB_HOME_DIR before the factory opens it.
    _migrate_legacy_sqlite_db()

    singleton_storage.set_storage_factory(SqliteStorageFactory())

    # 4. Use standalone space access manager — bypasses Lakehouse authorization.
    from gbserver.spaces.space_access_manager import set_space_access_manager
    from gbserver.spaces.standalone_space_access_manager import (
        StandaloneSpaceAccessManager,
    )

    set_space_access_manager(StandaloneSpaceAccessManager())

    # 5. Space registration is only possible when the caller supplies the space
    #    directory (the standalone server). Other commands skip it.
    if space_dir is None:
        return

    from gbserver.storage.stored_space import StoredSpace

    # Backward compatibility:  the standalone space.yaml's `name:` field used
    # to be `standalone`, then `public`.  The space directory has since moved
    # to configurations/spaces/local, but the `name:` field is still `public`.
    # Existing deployments, bookmarks, scripts, and database rows reference the
    # older names, so we register several rows pointing at the exact same
    # directory:
    #
    #   - 'public'     — matches the current space.yaml name field.
    #   - 'standalone' — legacy alias kept so old build configs and tooling
    #                    that still say `space_name: standalone` continue to
    #                    resolve.
    #   - 'local'      — alias matching the current directory name
    #                    (configurations/spaces/local) for configs and tooling
    #                    that reference the space by its directory name.
    #
    # All rows share the same `git_repo_uri`.  This is allowed because the
    # `git_repo_uri` column is not unique (see SQLSpaceStorage); only `name`
    # is unique, so the rows coexist cleanly.
    storage = singleton_storage.get_admin_storage()
    abs_dir = os.path.abspath(space_dir)
    space_uri = f"file://{abs_dir}"
    space_aliases = [
        ("public", space_uri),
        ("standalone", space_uri),
        ("local", space_uri),
    ]
    for name, uri in space_aliases:
        existing = storage.space_storage.get_by_name(name)
        stored_space = StoredSpace(
            name=name,
            git_repo_uri=uri,
            lakehouse_namespace="",
        )
        if existing is None:
            storage.space_storage.add(stored_space)
            logger.info("Created '%s' space with URI %s", name, uri)
        elif existing.git_repo_uri != uri:
            # Update the existing row to point at the current --space-dir.
            # Without this, re-launching standalone against a different
            # directory would silently keep using the stale URI from the
            # prior run.
            stored_space.uuid = existing.uuid
            storage.space_storage.update(stored_space, create_if_not_exist=False)
            logger.info(
                "Updated '%s' space (uuid=%s) from %s to %s",
                name,
                existing.uuid,
                existing.git_repo_uri,
                uri,
            )
        else:
            logger.info("'%s' space already exists (uuid=%s)", name, existing.uuid)
