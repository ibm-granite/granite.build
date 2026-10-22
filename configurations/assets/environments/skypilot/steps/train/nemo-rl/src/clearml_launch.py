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

"""In-process launcher for a NeMo-RL example driver (ClearML + V2 save format).

NeMo-RL has no native ClearML logger, but it logs scalars to TensorBoard and
ClearML auto-mirrors TensorBoard + stdout + GPU metrics IN THE PROCESS where
Task.init() runs. So this wrapper arms Task.init() and then runs the selected
examples/run_<algo>.py in the SAME process (via runpy), rather than spawning it
— the Ray driver is what emits the scalars; Ray workers need no ClearML.

With --v2-save-format it also resolves the driver's config exactly as the driver
will (NeMo-RL's loader: `defaults:` inheritance plus the Hydra overrides) and,
for the V2 DTensor worker, appends ++checkpointing.model_save_format=torch_save so
the checkpoint is DCP (convert_dcp_to_hf.py needs its .metadata). Doing it here,
in the process that is about to import nemo_rl anyway, avoids a separate
interpreter start just to read one flag, and a bad override fails with the
loader's own traceback before the driver starts.

When ClearML is disabled, clearml is never imported: the wrapper is a
passthrough that runs the driver.

CONTRACT
  clearml_launch.py --clearml-mode <auto|on|off> \
      [--clearml-project P] [--clearml-task T] [--v2-save-format] \
      --driver <path/to/run_algo.py> -- --config <cfg.yaml> [hydra overrides...]
"""

import argparse
import os
import runpy
import sys
from collections.abc import Mapping

TORCH_SAVE_OVERRIDE = "++checkpointing.model_save_format=torch_save"

_CLEARML_ENV = (
    "CLEARML_API_HOST",
    "CLEARML_API_ACCESS_KEY",
    "CLEARML_API_SECRET_KEY",
)


def clearml_enabled(mode: str, env: Mapping) -> bool:
    """Whether ClearML tracking should be armed for this run.

    nemo_rl_run.sh applies the same rule to decide `uv run --with clearml`, so
    the install and the import agree; test_clearml_rule_parity.py pins that.

    :param mode: ``off`` (never), ``on`` (always) or ``auto`` (only when every
        ClearML server credential is present and non-empty).
    :param env: The environment to read the credentials from.
    :returns: True if ``Task.init()`` should run.
    """
    if mode == "off":
        return False
    if mode == "on":
        return True
    return all(env.get(k) for k in _CLEARML_ENV)


def uses_v2(cfg: Mapping) -> bool:
    """Whether a resolved NeMo-RL config trains with the V2 DTensor worker.

    :param cfg: The resolved config (after inheritance and overrides).
    :returns: ``policy.dtensor_cfg._v2``, the flag lm_policy selects the worker by.
    """
    policy = cfg.get("policy") or {}
    dtensor = policy.get("dtensor_cfg") or {}
    return bool(dtensor.get("_v2", False))


def _with_v2_save_format(driver_argv: list[str]) -> list[str]:
    """Append the torch_save override when the driver's config is V2.

    The config is resolved the way examples/run_<algo>.py resolves it, from the
    same argv: ``--config <path>`` plus the remaining Hydra overrides.

    :param driver_argv: The driver's argv (``--config <path> [overrides...]``).
    :returns: ``driver_argv``, with :data:`TORCH_SAVE_OVERRIDE` appended for V2.
    :raises SystemExit: If ``--config`` is missing.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args(driver_argv)

    from nemo_rl.utils import config as nrl_config

    nrl_config.register_omegaconf_resolvers()
    cfg = nrl_config.load_config(args.config)
    if overrides:
        cfg = nrl_config.parse_hydra_overrides(cfg, overrides)
    if uses_v2(cfg):
        return [*driver_argv, TORCH_SAVE_OVERRIDE]
    return list(driver_argv)


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Split the wrapper's own options from the driver's argv.

    :param argv: The wrapper's argv, with the driver's argv after ``--``.
    :returns: The parsed wrapper options and the driver's argv.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--clearml-mode", default="auto", choices=("auto", "on", "off"))
    parser.add_argument("--clearml-project", default=None)
    parser.add_argument("--clearml-task", default=None)
    parser.add_argument("--v2-save-format", action="store_true")
    parser.add_argument("--driver", required=True)
    args, rest = parser.parse_known_args(argv)
    # Everything after the `--` separator is the driver's own argv. argparse may
    # or may not keep a leading `--` in `rest` across versions; strip one if so.
    if rest and rest[0] == "--":
        rest = rest[1:]
    return args, rest


def _run_driver(driver: str, driver_argv: list[str]) -> None:
    """Run ``driver`` as ``__main__`` in this process, as ``python driver`` would.

    The driver reads ``sys.argv`` (argparse), so it gets ``[driver, *argv]``.
    ``runpy.run_path`` does not put the driver's directory at ``sys.path[0]`` the
    way a direct launch does (here it would be this wrapper's ``src/``), so it is
    inserted explicitly; otherwise a driver importing a sibling module under
    ``examples/`` fails with ``ModuleNotFoundError``.

    :param driver: Path to ``examples/run_<algorithm>.py``.
    :param driver_argv: The driver's own arguments.
    """
    sys.argv = [driver, *driver_argv]
    sys.path.insert(0, os.path.dirname(os.path.abspath(driver)))
    runpy.run_path(driver, run_name="__main__")


def main(argv: list[str] | None = None) -> None:
    """Run the driver in-process, under ClearML when enabled.

    :param argv: The wrapper's argv; defaults to ``sys.argv[1:]``.
    """
    argv = sys.argv[1:] if argv is None else argv
    args, driver_argv = _parse_args(argv)
    if args.v2_save_format:
        driver_argv = _with_v2_save_format(driver_argv)

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
