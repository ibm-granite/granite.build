import click

from gbserver.storage import singleton_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.storage.stored_step_run import StoredStepRun
from gbserver.storage.stored_target_run import StoredTargetRun
from gbserver.types.status import Status


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
