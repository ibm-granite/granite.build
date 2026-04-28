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
Run user provided bash scripts in the local filesystem.
"""

import asyncio
import os
from asyncio.subprocess import Process
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Tuple, Union
from urllib.parse import urlparse

from gbcommon.uri.uri import URI
from gbserver.environment.local_assets import get_hf_cache_dir, pull_asset_hfstore
from gbserver.environment.environment import (
    BINDING_KEY,
    Environment,
    EventLogLineParserConfig,
)
from gbserver.types.buildconfig import BuildTargetStepConfig
from gbserver.types.buildevent import EntityRunMetadata
from gbserver.types.environmentconfig import EnvironmentConfig
from gbserver.utils.filesystem import sync_or_copy
from gbserver.utils.launch import launch_command_and_raise_errors
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)
DEFAULT_OUTPUT_DIR = "~/.gbcli/logs"
BASH_SCRIPTS = "bash_scripts"
JOB_SUB_SH = "llmb_bash_jobsub.sh"


class Bash(Environment):
    """
    The local filesystem environment.
    Used to run bash scripts.
    """

    launched_processes: Dict[str, Process]
    env: dict[str, Any]

    def __init__(self: Self, event_q: asyncio.Queue, **kwargs) -> None:
        self.launched_processes = {}
        self.env = {}
        super().__init__(event_q=event_q, **kwargs)

    async def setup_nohup(self: Self, **kwargs):
        space_secrets = kwargs.get("space_secrets", {})
        for key, value in space_secrets.items():
            key_str = str(key).strip()
            value_str = (
                str(value).encode("utf-8", "ignore").decode("unicode_escape")
            )  # Decode escaped sequences safely
            self.env[key_str] = value_str

    async def launch_nohup(
        self: Self,
        launch_id: str,
        targetsteprun_asset_dir: Optional[Path] = None,
        environment_config: Optional[EnvironmentConfig] = None,
        **kwargs,
    ):
        """
        Launches a Bash-based process asynchronously using a no-hang-up (nohup-style) execution model.

        This method prepares the runtime environment, sets up working directories, and executes
        the corresponding Bash job submission script for the specified step or target. It ensures that
        all logs and outputs are organized under the standardized output hierarchy for
        better traceability and log management across builds.

        Args:
            launch_id (str):
                Unique identifier for the current launch instance.
            targetsteprun_asset_dir (Optional[Path]):
                Path to the asset directory associated with the target step run.
            environment_config (Optional[EnvironmentConfig]):
                Configuration object containing environment variables and workspace information.
            **kwargs:
                Additional keyword arguments.
        """
        try:
            assert environment_config is not None, "environment_config is None"
            config_env = environment_config.get("env", {}) or {}
            for key, value in config_env.items():
                key_str = str(key).strip()
                expanded_value = os.path.expandvars(str(value))
                self.env[key_str] = str(expanded_value)
            launcher_config = kwargs.get("launcher_config", {}) or {}
            cwd = None
            working_dir = launcher_config.get("working_dir")
            if working_dir:
                cwd = Path(working_dir).expanduser().resolve()
                logger.info("Using working_dir from launcher_config: %s", cwd)
            elif targetsteprun_asset_dir:
                parsed_url = urlparse(str(targetsteprun_asset_dir))
                targetsteprun_asset_dir_path = Path(parsed_url.path)
                if targetsteprun_asset_dir_path.exists():
                    cwd = targetsteprun_asset_dir_path
                    logger.info(
                        "Inferred working_dir from targetsteprun_asset_dir: %s", cwd
                    )
                else:
                    logger.warning(
                        "The targetsteprun_asset_dir path does not exist: %s",
                        targetsteprun_asset_dir_path,
                    )

            if not cwd or not cwd.exists():
                cwd = Path(".").resolve()
                logger.info("Falling back to current working directory: %s", cwd)

            env = {**self.env, **launcher_config.get("env", {})}
            logger.debug(f"launch_nohup() called with launch_id={launch_id}")
            logger.debug(f"launcher_config = {launcher_config}")
            step_name = kwargs.get("step", {}).get("name", "")
            if step_name == "":
                step_name = kwargs.get("run_metadata", {}).get("target_name")
                assert step_name != "", "step_name and target_name is empty"
            command_list = [f"./{BASH_SCRIPTS}/{step_name}/{JOB_SUB_SH}"]
            logger.info(f"Launching {launch_id}: {command_list} in {cwd}")
            logger.debug(f"computed cwd = '{cwd}' (exists={os.path.exists(cwd)})")
            logger.debug(f"env vars = {env}")
            env["LLMB_BASH_LAUNCH_ID"] = launch_id
            env["LLMB_BASH_ASSET_DIR"] = str(targetsteprun_asset_dir)
            self.output_dir = (environment_config.get("workspace") or {}).get(
                "output_dir", ""
            )
            if self.output_dir:
                self.output_dir = Path(
                    os.path.expandvars(os.path.expanduser(self.output_dir))
                )
            else:
                self.output_dir = Path(DEFAULT_OUTPUT_DIR).expanduser()
            run_metadata = kwargs.get("run_metadata")
            assert isinstance(
                run_metadata, dict
            ), f"invalid run_metadata: {run_metadata}"
            final_asset_dir = await self._copy_assets(
                launch_id=launch_id,
                asset_dir=targetsteprun_asset_dir,
                **kwargs,
            )
            final_asset_output_dir = Path(final_asset_dir) / "outputs"
            logger.info("final_asset_output_dir: %s", final_asset_output_dir)
            self.log_paths = {launch_id: str(Path(final_asset_output_dir) / "job.log")}
            env["LLMB_BASH_OUTPUT_DIR"] = str(final_asset_output_dir)
            process, _, _ = await launch_command_and_raise_errors(
                command_list=command_list,
                launch_id=launch_id,
                start_new_session=True,
                cwd=cwd,
                env=env,
            )
            self.launched_processes[launch_id] = process
            self._release_monitors(launch_id)
        except FileNotFoundError as fe:
            # logger.error("Command not found: %s", command_list)
            raise ValueError(f"Command not found: {command_list}") from fe
        except Exception as e:
            # logger.error("Error launching process: %s", e)
            raise ValueError("Error launching process") from e

    async def monitor_log_monitor(
        self: Self,
        launch_id: str,
        event_q: Optional[asyncio.Queue] = None,
        entityrun_metadata: Optional[EntityRunMetadata] = None,
        event_configs: Optional[List] = None,
        build_id: str = "",
        **kwargs,
    ) -> None:
        event_log_parser_configs = []
        if event_configs is not None:
            event_log_parser_configs = [
                EventLogLineParserConfig.model_validate(config)
                for config in event_configs
            ]
        assert event_q is not None, "the event_q is None"
        assert entityrun_metadata is not None, "the entityrun_metadata is None"
        await self._monitor_logs_of_async_subprocess_all(
            self.launched_processes[launch_id],
            event_q,
            event_log_parser_configs,
            entityrun_metadata,
        )

    async def pullasset_filestore(
        self: Self,
        uri: URI,
        binding: Optional[Any] = None,
        # storeload_config: Optional[StoreLoad] = None,
        # assetstore: Optional[Assetstore] = None,
        # secrets: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[Dict, Optional[BuildTargetStepConfig]]:
        if isinstance(uri, str):
            uri = URI.get_uri(uri)
        assert uri.uri is not None, "the URI is None"
        if binding is None:
            binding_config = {BINDING_KEY: {"path": uri.uri.path}}
            return binding_config, None
        else:
            sync_or_copy(uri.uri.path, binding)
            return binding, None

    async def pullasset_hfstore(
        self: Self,
        uri: URI,
        binding: Optional[Any] = None,
        storeload_config=None,
        assetstore=None,
        secrets: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[Dict, Optional[BuildTargetStepConfig]]:
        """Download an HF model snapshot and bind as a local path."""
        local_path = pull_asset_hfstore(uri, assetstore, storeload_config)
        binding_config = {BINDING_KEY: {"path": str(local_path)}}
        return binding_config, None

    @staticmethod
    def _get_hf_cache_dir(storeload_config) -> str:
        """Resolve HF model cache directory from config or default."""
        return get_hf_cache_dir(storeload_config)

    async def pushasset_filestore(
        self: Self,
        binding: Any,
        uri: Optional[URI] = None,
        base_uri: Optional[URI] = None,
        **kwargs,
    ) -> Any:
        if uri is None and base_uri is None:
            return None
        if uri is not None:
            uriobj = uri
            if isinstance(uri, str):
                uriobj = URI.get_uri(uri)
            assert uriobj.uri is not None, "the URI is None"
            sync_or_copy(binding, uriobj.uri.path)
            return uri
        elif base_uri is not None:
            uriobj = base_uri
            if isinstance(base_uri, str):
                uriobj = URI.get_uri(base_uri)
            assert uriobj.uri is not None, "the URI is None"
            sync_or_copy(binding, uriobj.uri.path)
            return URI.get_uristr(base_uri) + "/" + os.path.basename(binding)
        else:
            return None

    def _get_job_name(self: Self, launch_id: str) -> str:
        return "launch-" + launch_id

    def _get_workspace_sub_dir(
        self: Self,
        build_id: str,
        target_name: str,
        targetrun_id: str,
        step_name: str,
        targetsteprun_id: str,
        launch_id: str,
    ) -> Path:
        return (
            Path(f"llm-build-{build_id}")
            / f"target-{target_name}"
            / f"target-run-{targetrun_id}"
            / f"step-{step_name}"
            / f"step-run-{targetsteprun_id}"
            / self._get_job_name(launch_id=launch_id)
        )

    async def _copy_assets(
        self: Self,
        launch_id: str,
        asset_dir: Path,
        **kwargs: Dict,
    ) -> Path:
        """Returns final_asset_dir"""
        run_metadata = kwargs.get("run_metadata")
        assert isinstance(run_metadata, dict), f"invalid run_metadata: {run_metadata}"
        step_name = kwargs.get("step", {}).get("name", "")
        final_asset_dir = self._get_final_asset_dir(
            asset_dir=asset_dir,
            launch_id=launch_id,
            run_metadata=run_metadata,
            step_name=step_name,
        )
        logger.info("final_asset_dir: %s", final_asset_dir)
        logger.info("copying %s to %s", asset_dir, final_asset_dir)
        sync_or_copy(
            src=str(asset_dir) + "/",
            dest=final_asset_dir,
            delete=False,
            raise_errors=True,
        )
        return final_asset_dir

    def _get_final_asset_dir(
        self: Self,
        asset_dir: Path,
        launch_id: str,
        run_metadata: Union[Dict, EntityRunMetadata],
        step_name: str = "",
    ) -> Path:
        """run_metadata is a serialized EntityRunMetadata"""
        if isinstance(run_metadata, dict):
            run_metadata = EntityRunMetadata.from_dict(run_metadata)
        build_id = run_metadata.build_id
        assert build_id, f"invalid build_id: {run_metadata}"
        target_name = run_metadata.target_name
        assert target_name, f"invalid target_name: {run_metadata}"
        targetrun_id = run_metadata.targetrun_id
        assert targetrun_id, f"invalid targetrun_id: {run_metadata}"
        targetsteprun_id = run_metadata.targetsteprun_id
        assert targetsteprun_id, f"invalid targetsteprun_id: {run_metadata}"
        sub_dir = self._get_workspace_sub_dir(
            build_id=build_id,
            target_name=target_name,
            targetrun_id=targetrun_id,
            step_name=step_name,
            targetsteprun_id=targetsteprun_id,
            launch_id=launch_id,
        )
        logger.info("asset_dir: %s sub_dir: %s", asset_dir, sub_dir)
        final_asset_dir = sub_dir
        if self.output_dir:
            final_asset_dir = Path(self.output_dir) / sub_dir
            final_asset_dir = final_asset_dir.resolve()
        return final_asset_dir
