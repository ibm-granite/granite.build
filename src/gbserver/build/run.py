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

"""
Abstract class for all runs.
"""

import asyncio
import traceback
from abc import ABC, abstractmethod
from asyncio import Event, Queue, Task, TaskGroup
from pathlib import Path
from typing import List, Optional, Self

from gbserver.build.buildentity import BuildEntity
from gbserver.types.buildevent import (
    BuildEvent,
    BuildEventStatusPayload,
    BuildEventType,
    EntityRunMetadata,
)
from gbserver.types.constants import truncate
from gbserver.types.status import STATUS_TO_ICON, Status
from gbserver.utils.logger import get_logger
from gbserver.utils.unwrap_errors import get_readable_error_message
from gbserver.utils.utils import get_uuid

logger = get_logger(__name__)


class RunFailed(RuntimeError):
    """Indicates that the Run has failed."""

    def __init__(
        self: Self,
        *args,
        status_updated: bool = False,
        exceptions: Optional[List[BaseException]] = None,
    ) -> None:
        if exceptions is None:
            super().__init__(*args)
        else:
            aggregated_message = "Exception Details:"
            aggregated_message += "\n".join(
                [
                    line
                    for e in exceptions
                    for line in traceback.format_exception(type(e), e, e.__traceback__)
                ]
            )
            super().__init__(aggregated_message)
        self.status_updated = status_updated
        self.exceptions = exceptions


class Run(ABC):
    """Runtime equivalent of an entity."""

    id: str
    entity: BuildEntity
    event_q: Queue[Event]
    build_id: str
    task: Optional[Task]
    metadata: dict
    status: Status
    base_dir: Path
    dry_run: bool = False

    def __init__(
        self: Self,
        entity: BuildEntity,
        event_q: Optional[Queue] = None,
        base_dir: Optional[Path] = None,
        id: Optional[str] = None,
        dry_run: bool = False,
    ) -> None:
        if id is None or id == "":
            id = get_uuid()
        self.id = id
        self.dry_run = dry_run
        if self.dry_run:
            logger.warning("dry_run flag is set for the run: %s", self.id)
        logger.debug("Run.__init__ %s start", self.id)
        self.entity = entity
        if event_q is None:
            event_q = Queue()
        self.event_q = event_q
        self.build_id = entity.build_id
        self.task = None
        self.metadata = {}
        self.status = Status.INVALID
        self.update_status(Status.PENDING)
        assert base_dir is not None, f"base_dir {base_dir} cannot be empty"
        self.dir = base_dir / self.id
        logger.debug("Run.__init__ %s end", self.id)

    @abstractmethod
    async def _run(self: Self, tg: Optional[TaskGroup] = None, **kwargs) -> None:
        """Implemention of core run logic"""

    async def run(self: Self, tg: Optional[TaskGroup] = None, **kwargs) -> None:
        """Run the entity."""
        logger.debug("Run.run %s start", self.id)
        try:
            logger.debug("Running the entity %s", self.id)
            self._add_to_run_kwargs(kwargs)
            self.update_status(Status.RUNNING)
            await self._run(tg, **kwargs)
            self.update_status(Status.SUCCESS)
        except asyncio.CancelledError as e:
            logger.error(
                "Run.run: [%s : %s] the run was cancelled exception: %s",
                type(self),
                self.id,
                e,
            )
            self.update_status(Status.CANCELLED)
            raise e
        except BaseExceptionGroup as eg:
            # Python 3.11+ TaskGroup raises BaseExceptionGroup (rather than
            # ExceptionGroup) when the group contains a mix of Exception and
            # BaseException subclasses — e.g. a monitor task fails with a
            # RuntimeError and the launch task is subsequently cancelled
            # (CancelledError). BaseExceptionGroup is NOT caught by
            # `except Exception`, so we handle it explicitly here.
            failures = [
                e for e in eg.exceptions if not isinstance(e, asyncio.CancelledError)
            ]
            if failures:
                primary = failures[0]
                err_stack = "".join(
                    traceback.format_exception(type(primary), primary, primary.__traceback__)
                )
                body = get_readable_error_message(e=primary, err_stack=err_stack)  # type: ignore[arg-type]
                self.update_status(Status.FAILED, extra_msg=body)
                raise RunFailed(status_updated=True, exceptions=failures) from eg
            else:
                self.update_status(Status.CANCELLED)
                raise asyncio.CancelledError() from eg
        except Exception as e:
            err_stack = traceback.format_exc()
            body = get_readable_error_message(e=e, err_stack=err_stack)
            # logger.error("%s", body) # TODO: is this necessary?
            self.update_status(Status.FAILED, extra_msg=body)
            raise RunFailed(status_updated=True) from e
        finally:
            logger.debug("Run.run: do some cleanup after the run is done")
            cleanup_task = asyncio.ensure_future(self._cleanup(tg=tg))
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                # The outer task was cancelled, but we must wait for cleanup
                # to finish (e.g. helm uninstall) before propagating.
                await cleanup_task
        logger.debug("Run.run %s end", self.id)

    def _add_to_run_kwargs(self: Self, kwargs: dict) -> None:
        """Add additional kwargs like run metadata: build_id, etc."""
        kwargs["runmetadata"] = self.get_runmetadata()

    async def _cleanup(self: Self, tg: Optional[TaskGroup] = None) -> None:
        """Check status and cleanup after the job is done."""
        logger.debug("Run._cleanup %s start", self.id)
        logger.debug("Run._cleanup %s end", self.id)

    def dispatch_event(self: Self, event: Event) -> None:
        """Dispatch an event if the event queue exists."""
        logger.debug("Run.dispatch_event %s start", self.id)
        logger.info("run %s dispatching event: %s", self.id, event)
        if self.event_q is None:
            logger.debug("Run.dispatch_event no event_q end")
            return
        self.event_q.put_nowait(event)
        logger.debug("Run.dispatch_event %s end", self.id)

    def create_message(self: Self, extra_msg: str = "") -> str:
        """Create a user-readable message."""
        icon = STATUS_TO_ICON[self.status]
        status = self.status.name
        build_id = self.build_id
        target_id = self.id
        step_id = self.id
        step_uri = ""
        entity_type = type(self).__name__
        target_name = entity_type
        if entity_type == "TargetRun":
            target_name = getattr(self.entity, "name", target_name)
        elif entity_type == "TargetStepRun":
            target_name = getattr(self.entity, "target_name", target_name)
            step_uri = getattr(self.entity, "step_uri", target_name)
        msg = f"""```
Status      : {icon} {status}
Build ID    : {build_id}
```
"""
        if entity_type == "TargetRun":
            msg = f"""```
Status      : {icon} {status}
Target Name : {target_name}
Type        : target
Target ID   : {target_id}
Build ID    : {build_id}
```
"""
        elif entity_type == "TargetStepRun":
            msg = f"""```
Status      : {icon} {status}
Target Name : {target_name}
Type        : step
Step URI    : {step_uri}
Step ID     : {step_id}
Build ID    : {build_id}
```
"""
        if extra_msg != "":
            msg += "\n" + extra_msg + "\n"
        return msg

    def update_status(self: Self, status: Status, extra_msg: str = ""):
        """Update status and send events"""
        logger.debug("Run.update_status %s start", self.id)
        self.status = status
        msg = self.create_message(extra_msg=extra_msg)
        logger.info("msg: %s", truncate(msg))
        event = BuildEvent(
            run_metadata=self.get_runmetadata(),
            type=BuildEventType.STATUS_EVENT,
            payload=BuildEventStatusPayload(
                status=status, msg=msg, metadata=self.metadata
            ),
        )
        logger.debug("event: %s", event)
        self.dispatch_event(event)
        logger.debug("Run.update_status %s end", self.id)

    def check_exceptions(self: Self, t: asyncio.Task):
        """Necessary to capture uncaught exceptions from tasks."""
        logger.debug("Run.check_exceptions %s start", self.id)
        try:
            _ = t.result()
        except asyncio.CancelledError:
            logger.info(
                "Task was cancelled for build_id=%s task id=%s", self.build_id, self.id
            )
        except Exception as e:
            # logger.error("%s", traceback.format_exc()) # TODO: is this necessary?
            logger.error("job task failed with exception: %s", e)
        logger.debug("Run.check_exceptions %s end", self.id)

    def async_run(self: Self, tg: Optional[asyncio.TaskGroup] = None) -> asyncio.Task:
        """Run the job."""
        logger.debug("Run.async_run %s start", self.id)
        if self.status is not Status.PENDING:
            raise ValueError(f"the run is not in a pending state: {self}")
        assert self.task is None, "task already exists"
        self.status = Status.RUNNING
        if tg:
            self.task = tg.create_task(self.run(tg))
        else:
            self.task = asyncio.create_task(self.run())
        self.task.add_done_callback(self.check_exceptions)
        logger.debug("Run.async_run %s end", self.id)
        return self.task

    def cancel(self: Self) -> bool:
        """Cancel the job."""
        if self.task is None:
            logger.error("no task to cancel!")
            return False
        if self.task.done():
            return True
        return self.task.cancel()

    async def wait(self: Self) -> None:
        """Wait for the run to finish."""
        logger.debug("Run.wait %s start", self.id)
        if self.status is not Status.RUNNING and self.status is not Status.CANCELLED:
            if self.status is Status.PENDING:
                raise ValueError("the run has not been started")
            logger.debug("the run is not in a running state: %s", self.status)
            return
        if self.task is None:
            logger.debug("the run was not started, nothing to wait for")
            return
        if self.status is Status.CANCELLED:
            try:
                logger.debug("Run.wait wait for the cancelled run")
                await self.task
            except asyncio.CancelledError:
                pass
        else:
            logger.debug("Run.wait wait for the running run")
            await self.task
        logger.debug("Run.wait %s end", self.id)

    async def run_and_wait(self: Self, tg: Optional[TaskGroup] = None):
        """Run and wait for it to finish."""
        logger.debug("Run.run_and_wait %s start", self.id)
        self.async_run(tg)
        await self.wait()
        logger.debug("Run.run_and_wait %s end", self.id)

    @abstractmethod
    def get_runmetadata(self: Self) -> EntityRunMetadata:
        """Get run metadata."""
