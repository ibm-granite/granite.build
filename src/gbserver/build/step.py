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
The step.
"""

import glob
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional, Self, Union

from gbcommon.uri.uri import URI
from gbserver.asset.asset import Asset
from gbserver.build.entity import Entity
from gbserver.types.constants import is_debug_mode
from gbserver.types.stepconfig import StepConfig
from gbserver.utils.logger import get_logger

STEP_FILE_NAME = "step.yaml"
STEP_DEFAULT_FILE_NAME = "step_default.yaml"

logger = get_logger(__name__)


class Step(Entity):
    """A single step in a target of a build."""

    _thread_local = threading.local()

    def __init__(
        self: Self,
        stepuri: Union[URI, str],
        context: Optional[str] = None,
        force_fetch: bool = False,
        **kwargs: dict,
    ):
        self.stepasset = Asset(stepuri, context=context)

        if not hasattr(self._thread_local, "stepcache_dir"):
            self._thread_local.stepcache_dir = Path(tempfile.mkdtemp())

        # We make 2 calls to the Step() in the targetstep, we want to
        # persist the information of whether step fallback to basestep was
        # used or not across these 2 calls.

        # Default is false because we dont use fallback if there is a step.yaml
        # in a non-empty step uri
        if not hasattr(self._thread_local, "step_fallback_used"):
            self._thread_local.step_fallback_used = False

        self.step_fallback_used = self._thread_local.step_fallback_used

        th_stepcache_dir = self._thread_local.stepcache_dir
        assert isinstance(th_stepcache_dir, Path)
        stepasset_dir = th_stepcache_dir / self.urihash()
        self.stepasset.sync(dest=stepasset_dir, force=force_fetch)
        files = glob.glob(str(stepasset_dir / "**" / STEP_FILE_NAME), recursive=True)

        # Scenario 1: if step uri is empty, it wont have step.yaml -> thus get
        # base step_default.yaml and
        # rename it to step.yaml so that it works seamlessly
        #  with the existing flow.
        if len(files) == 0:
            default_files = glob.glob(
                str(stepasset_dir / "**" / STEP_DEFAULT_FILE_NAME), recursive=True
            )

            # found step_default.yaml from the base step
            if len(default_files) > 0:
                self._thread_local.step_fallback_used = True
                self.step_fallback_used = True
                logger.warning(
                    "=== Handling case: step uri is empty, using step_default.yaml from base step ==="
                )
                actual_default = Path(default_files[0])

                target_dir = stepasset_dir / "gbstep"
                target_dir.mkdir(parents=True, exist_ok=True)

                # name to rename into
                dest = target_dir / STEP_FILE_NAME

                # renaming step_default.yaml to step.yaml
                actual_default.replace(dest)

                # Use the newly created path as the step file path
                files = [str(dest)]

        # Scenario 2: step_uri is non-empty (which means by default it is not the builtin base
        # step) but NO step.yaml in it
        # We copy the step_default.yaml from default step in base
        if not files:
            # we are here means we did not get default step from the step uri meaning that the step uri was not empty
            logger.warning(
                f"=== Handling case: NO step.yaml present in the step uri {stepuri}..using step_default.yaml from base step ==="
            )
            self._thread_local.step_fallback_used = True
            self.step_fallback_used = True
            default_path = (
                Path(os.path.abspath(__file__)).parent.parent / "builtins/steps/gbstep"
            )

            builtin_default = default_path / STEP_DEFAULT_FILE_NAME
            assert (
                builtin_default.exists()
            ), f"Builtin default step not found at {builtin_default}"

            for p in stepasset_dir.iterdir():
                if p.is_dir() and p.name != "gbstep":
                    target_step_dir = p
                    break

            target_dir = target_step_dir  # step dir NOT containing the step.yaml
            dest = target_dir / STEP_FILE_NAME

            shutil.copy(builtin_default, dest)

            files = [str(dest)]

        # if both step.yaml and step_default.yaml not found -> raise Assertion Error
        assert len(files) > 0, (
            f"failed to find a {STEP_FILE_NAME} or {STEP_DEFAULT_FILE_NAME} "
            f"for uri {URI.get_uristr(stepuri)} in {stepasset_dir}"
        )

        step_yaml_path = Path(files[0])
        assert step_yaml_path.is_file(), f"expected {step_yaml_path} to be a file"
        self.step_yaml_path = step_yaml_path
        try:
            config = StepConfig.from_yaml(step_yaml_path, context=context)
            super().__init__(
                type="step",
                config=config,
                dir=stepasset_dir,
                force_fetch=force_fetch,
                **kwargs,
            )
        except Exception as e:
            raise ValueError(f"file {step_yaml_path} is invalid") from e

        if not is_debug_mode():
            # TODO: The step may be coming from a space repo, which might have a large experiments directory.
            # TODO: for now only remove the experiments directory to be safe, but longterm we should be removing the whole stepasset_dir
            exp_dirs = glob.glob(
                str(stepasset_dir / "**" / "experiments"), recursive=True
            )
            for dir in exp_dirs:
                if not "step" in str(
                    dir
                ):  # Make sure not to remove anything step-related though
                    shutil.rmtree(dir)

    def urihash(self: Self) -> str:
        """Urihash."""
        return self.stepasset.urihash()

    def assimilate(self: Self) -> None:
        pass
