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

"""Thin ClearML wrapper around a NeMo-RL example driver.

NeMo-RL has no native ClearML logger, but it logs scalars to TensorBoard and
ClearML auto-mirrors TensorBoard + stdout + GPU metrics IN THE PROCESS where
Task.init() runs. So this wrapper arms Task.init() and then runs the selected
examples/run_<algo>.py in the SAME process (via runpy), rather than spawning it
— the Ray driver is what emits the scalars; Ray workers need no ClearML.

When ClearML is disabled, clearml is never imported: the wrapper is a pure
passthrough that runs the driver.

CONTRACT
  clearml_launch.py --clearml-mode <auto|on|off> \
      [--clearml-project P] [--clearml-task T] \
      --driver <path/to/run_algo.py> -- <driver argv...>
"""

import argparse
import os
import runpy
import sys
from collections.abc import Mapping

_CLEARML_ENV = (
    "CLEARML_API_HOST",
    "CLEARML_API_ACCESS_KEY",
    "CLEARML_API_SECRET_KEY",
)


def clearml_enabled(mode: str, env: Mapping) -> bool:
    """True iff ClearML tracking should be armed for this run.

    off => never; on => always; auto => only when all server creds are present.
    nemo_rl_run.sh mirrors this exact rule to decide `uv run --with clearml`, so
    the install and the import never disagree.
    """
    if mode == "off":
        return False
    if mode == "on":
        return True
    return all(env.get(k) for k in _CLEARML_ENV)


def _parse_args(argv: list[str]):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--clearml-mode", default="auto", choices=("auto", "on", "off"))
    parser.add_argument("--clearml-project", default=None)
    parser.add_argument("--clearml-task", default=None)
    parser.add_argument("--driver", required=True)
    args, rest = parser.parse_known_args(argv)
    # Everything after the `--` separator is the driver's own argv. argparse may
    # or may not keep a leading `--` in `rest` across versions; strip one if so.
    if rest and rest[0] == "--":
        rest = rest[1:]
    return args, rest


def _run_driver(driver: str, driver_argv: list[str]) -> None:
    # run_<algo>.py reads sys.argv via argparse.parse_known_args, so hand it
    # exactly argv[0]=driver path then its flags, and execute it as __main__.
    sys.argv = [driver, *driver_argv]
    runpy.run_path(driver, run_name="__main__")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    args, driver_argv = _parse_args(argv)

    if not clearml_enabled(args.clearml_mode, os.environ):
        _run_driver(args.driver, driver_argv)
        return

    from clearml import Task

    task = Task.init(
        project_name=args.clearml_project or "nemo-rl",
        task_name=args.clearml_task or os.path.basename(args.driver),
        output_uri=False,
    )
    try:
        _run_driver(args.driver, driver_argv)
    finally:
        task.close()


if __name__ == "__main__":
    main()
