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
LSF based environments.
"""

import asyncio
import os
import random
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Tuple, Union

from pydantic import BaseModel

from gbcommon.uri.cos import CosURI
from gbcommon.uri.env import EnvURI
from gbcommon.uri.hf import HfURI
from gbcommon.uri.lh import LhURI
from gbcommon.uri.uri import URI
from gbserver.asset.asset import Asset
from gbserver.asset.assetstore import Assetstore
from gbserver.asset.cosstore import Cosstore
from gbserver.asset.hfstore import Hfstore
from gbserver.asset.lhstore import Lhstore
from gbserver.environment.environment import (
    BINDING_KEY,
    Environment,
    EventLogLineParserConfig,
)
from gbserver.monitoring.logfile_monitor import LogFileMonitor
from gbserver.monitoring.lsf_bsub_monitor import LSFBsubMonitor
from gbserver.monitoring.streams.log_stream_base import LogStreamSource
from gbserver.monitoring.streams.stream_factory import make_stream
from gbserver.resilience.strategies.aspera_failure import AsperaRetryStrategy
from gbserver.types.buildconfig import BuildTargetOutputConfig, BuildTargetStepConfig
from gbserver.types.buildevent import (
    EntityRunMetadata,
)
from gbserver.types.constants import (
    CODE_GBSERVER_BUILTINS_STEPS_DIR,
    DEFAULT_ROOT_WORKSPACE_DIR,
    ENABLE_SSH_HOST_KEY_VERIFICATION,
    LSF_USE_ASPERA,
    STEP_FILE_NAME,
)
from gbserver.types.environment.environment import StepConfigSection
from gbserver.types.environmentconfig import (
    EnvironmentConfig,
    StoreLoad,
    StorePush,
)
from gbserver.types.stepconfig import StepConfig
from gbserver.utils.filesystem import sync_or_copy
from gbserver.utils.launch import (
    launch_command_and_raise_errors,
    launch_command_and_retry_or_raise_errors,
)
from gbserver.utils.logger import get_logger
from gbserver.utils.ssh_tunnel import SshTunnel
from gbserver.utils.utils import cmd_safe_join, get_uuid, short_alphanumeric_lower_hash

logger = get_logger(__name__)

JOB_LOG_STDOUT_FILENAME = "job_log.out"
JOB_LOG_STDERR_FILENAME = "job_log.err"
LSF_SCRIPTS = "lsf_scripts"
JOB_SUB_SH = "llmb_lsf_jobsub.sh"
REPLACE_THIS_PREFIX = "LLMB_LSF_REPLACE_THIS_"

# LSF-specific copies of the auto-injected builtin steps live under
# `<builtins>/steps/lsf/<name>/`.  This module reads their step.yaml directly
# (filesystem read, not a SpaceURI lookup) to extract the LSF launcher config
# section.  At runtime the step itself is referenced via a `space://steps/<name>`
# URI that the SpaceURI resolver routes to the correct env-keyed split.
_LSF_BUILTIN_STEPS_DIR = CODE_GBSERVER_BUILTINS_STEPS_DIR / "lsf"
_LSF_HFPULL_DIR = _LSF_BUILTIN_STEPS_DIR / "hfpull"
_LSF_HFPUSH_DIR = _LSF_BUILTIN_STEPS_DIR / "hfpush"
_LSF_LHPULL_DIR = _LSF_BUILTIN_STEPS_DIR / "lhpull"
_LSF_LHPUSH_DIR = _LSF_BUILTIN_STEPS_DIR / "lhpush"
_LSF_COSRCLONE_DIR = _LSF_BUILTIN_STEPS_DIR / "cosrclone"


class BJobRecord(BaseModel):
    """A bjob record."""

    JOBID: str
    STAT: str
    EXIT_CODE: str
    EXIT_REASON: str


class BJobOutput(BaseModel):
    """Output of bjob."""

    COMMAND: str
    JOBS: int
    RECORDS: list[BJobRecord]


class ExistingBsubJobs(BaseModel):
    """
    Jobs launched by the user outside of our control.
    E.g. LLMB Lite launched jobs that we passively monitor.
    """

    job_id: str = ""


class Lsf(Environment):
    """
    Load Sharing Facility (LSF) based environments
    that use bsub to submit jobs.
    """

    _log_paths: Dict[str, str]  # launch_id -> output directory

    def __init__(
        self: Self,
        event_q: asyncio.Queue,
        environment_config: Optional[EnvironmentConfig] = None,
        secrets: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        assert environment_config is not None, "environment_config is None"
        env_config = environment_config.config
        assert env_config is not None, "environment_config.config is None"
        Environment.__init__(
            self,
            event_q=event_q,
            environment_config=environment_config,
            secrets=secrets,
            **kwargs,
        )

        self._launched_jobs: Dict[str, str] = {}
        # self.created_setup_files: Dict[str, str] = {}
        self._key_file_path: Optional[str] = None
        self._log_paths: Dict[str, str] = {}
        # for existing llmb-lite jobs, launch id -> {'jobid': 'xxxx'}
        self._existing_jobids: Dict[str, ExistingBsubJobs] = {}
        # Store launch kwargs for retry capability
        self._launch_kwargs: Dict[str, Dict] = {}
        # Coordination for RetryHandler-driven retries: signal monitor_bsub_monitor
        self._lsf_retry_complete_events: Dict[str, asyncio.Event] = {}
        # Launch asset dirs to delete at teardown, after all pipeline steps finish.
        # Populated in _copy_assets (at creation) so all dirs are captured even for retries.
        self._pending_cleanup_dirs: set[Path] = set()
        # Temporary storage for event_configs forwarded by the deprecated monitor_logfile_monitor stub
        self._logfile_event_configs: Dict[str, List] = {}
        self._ssh_tunnel: Optional[SshTunnel] = None
        # Load env-specific config
        # Workspace
        lsf_workspace: Dict = env_config.get("workspace", {})
        self.lsf_workspace_local_dir: str = lsf_workspace.get("local_dir", "")
        if not self.lsf_workspace_local_dir:
            self.lsf_workspace_local_dir = str(
                Path(DEFAULT_ROOT_WORKSPACE_DIR) / "env_lsf"
            )
        self.lsf_workspace_remote_dir: str = lsf_workspace.get("remote_dir", "")
        # Authentication
        authentication: Dict = env_config.get("authentication", {})
        self.use_ssh: bool = authentication.get("use_ssh", True)
        self.ssh_port = int(authentication.get("ssh_port", "22"))
        self.ssh_max_sessions = int(authentication.get("ssh_max_sessions", "10"))
        if isinstance(self.use_ssh, str):
            self.use_ssh = self.use_ssh.lower() == "true"
        assert isinstance(
            self.use_ssh, bool
        ), f"invalid use_ssh type: {type(self.use_ssh).__name__} (expected 'bool' type)"
        self.login_nodes: List[str] = authentication.get("login_nodes", [])
        assert isinstance(
            self.login_nodes, list
        ), f"invalid login_nodes type: {type(self.login_nodes).__name__} (expected 'list' type)"
        login_node_count = len(self.login_nodes)
        if login_node_count == 0:
            assert (
                not self.use_ssh
            ), "At least one login node must be provided when using ssh"
        else:
            self.login_node_idx = random.randrange(
                0, login_node_count
            )  # Randomly load balance
            for v in self.login_nodes:
                assert isinstance(v, str), f"invalid login node: {v}"
        self.username: str = authentication.get("login_node_username", "")
        self.ssh_key_secret_name: str = authentication.get("login_node_ssh_key", "")
        self.ssh_key = ""
        # -------------------------
        self.ssh_host_key_verification = (
            str(
                authentication.get(
                    "ssh_host_key_verification", ENABLE_SSH_HOST_KEY_VERIFICATION
                )
            ).lower()
            == "true"
        )
        if authentication.get("login_node_rotation"):
            logger.warning(
                "login_node_rotation is deprecated (we now always rotate through login nodes)"
            )

        self.copy_method: str = authentication.get("copy_method", "scp")
        assert self.copy_method in (
            "scp",
            "rsync",
        ), f"invalid copy_method: {self.copy_method} (expected 'scp' or 'rsync')"

        self.ssh_timeout = int(authentication.get("ssh_timeout", "5"))
        self.node_search_lock = asyncio.Lock()
        self.unreachable_ssh_nodes = []  # type: ignore[var-annotated]
        if self.use_ssh:
            assert (
                self.ssh_key_secret_name
            ), f"invalid ssh_key_secret_name: {self.ssh_key_secret_name}"
            logger.info(
                f"Created Lsf environment with ssh enabled, username={self.username}, keyname={self.ssh_key_secret_name}, login nodes are {self.login_nodes}"
            )

        else:
            logger.info(f"creating Lsf environment with ssh disabled")

    async def __preload_unreachable_ssh_nodes(self: Self) -> None:
        # Loop through all our nodes and identify the ureachable ones
        # Raises exception if no reachable nodes are found
        for _ in range(0, len(self.login_nodes)):
            await self._get_reachable_ssh_node()
        reachable = [
            item for item in self.login_nodes if item not in self.unreachable_ssh_nodes
        ]
        logger.info(
            f"Reachable ssh nodes: {reachable}, unreachable ssh nodes: {self.unreachable_ssh_nodes}"
        )

    def _get_default_retry_strategies(self: Self):
        from gbserver.resilience.strategies.lsf_transient_error import (
            LsfTransientErrorRetryStrategy,
        )

        return [LsfTransientErrorRetryStrategy(), AsperaRetryStrategy()]

    def _get_retry_test_scenario(self: Self) -> Optional[str]:
        return "lsf_transient_error"

    async def retry_workload(
        self: Self,
        launch_id: str,
        nodes_to_avoid: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Retry an LSF workload after a transient error.

        Called by RetryHandler when LsfTransientErrorRetryStrategy triggers.
        Re-launches the job and signals monitor_bsub_monitor via the coordination event.
        """
        original_kwargs = self._launch_kwargs.get(launch_id, {})
        job_id = self._launched_jobs.get(launch_id, launch_id)

        msg = f"⚠️ LSF error: JobID={job_id}. Retrying..."
        self._send_message(msg=msg, **original_kwargs)

        # Signal the LSFBsubMonitor to exit its polling loop cleanly before we bkill
        # the job. This prevents the monitor from treating the bkill as a real failure.
        self._get_launch_stopped_event(launch_id).set()

        try:
            task = self.cleanup_bsub(launch_id=launch_id)
            assert task
            await task
        except Exception as cleanup_error:
            logger.warning(
                "Failed to cleanup job %s during retry: %s",
                job_id,
                cleanup_error,
            )

        # Clear the stop event so the next monitor loop iteration starts fresh.
        self._get_launch_stopped_event(launch_id).clear()

        try:
            task = self.launch_bsub(launch_id, **original_kwargs)
            assert task
            await task
        except Exception as launch_error:
            logger.error(
                "Could not retry launch launch_id=%s: %s", launch_id, launch_error
            )
            raise launch_error

        # Signal monitor_bsub_monitor to loop for the next iteration
        retry_event = self._lsf_retry_complete_events.get(launch_id)
        if retry_event is not None:
            retry_event.set()

    def __get_ssh_node(self: Self) -> Optional[str]:
        """
        Get a candidate ssh login node.
        ONLY to be called from get_reachable_ssh_node() - these two work together
        on the use of self.unreachable_ssh_nodes.
        """
        l = len(self.login_nodes)
        assert (
            l > 0
        ), "Should only be using ssh when at least one ssh login node is provided"
        login_node = None
        if len(self.login_nodes) == len(self.unreachable_ssh_nodes):
            # We should not get here since get_reachable_ssh_node() clear this, but just in case.
            self.unreachable_ssh_nodes.clear()
        tries = 0
        while not login_node and tries < len(self.login_nodes):
            tries += 1
            self.login_node_idx = (self.login_node_idx + 1) % l
            login_node = self.login_nodes[self.login_node_idx]
            if login_node in self.unreachable_ssh_nodes:
                login_node = None  # Reset if using node rotation

        logger.info("login node is: %s", login_node)
        return login_node

    async def _get_reachable_ssh_destination(self: Self) -> str:
        """Get the node or user@node destination for ssh command"""
        node = await self._get_reachable_ssh_node()
        return self.__get_ssh_destination(node=node)

    async def _get_reachable_ssh_node(self: Self) -> str:
        """Try all of our nodes until we find one that we can ssh to."""
        assert self._key_file_path, "Must be provided"
        tried_nodes = []
        reset_enabled = True
        async with self.node_search_lock:
            node = self.__get_ssh_node()
            while not node in tried_nodes:
                if node is None:
                    break  # and raise exception
                launch_id = get_uuid()
                if node not in self.unreachable_ssh_nodes:
                    if await self.__is_ssh_node_reachable(
                        node=node, launch_id=launch_id
                    ):
                        logger.info("Found reachable ssh node %s", node)
                        return node
                    self.unreachable_ssh_nodes.append(node)
                tried_nodes.append(node)
                if reset_enabled and len(self.unreachable_ssh_nodes) == len(
                    self.login_nodes
                ):
                    # IF there are no reachable nodes, retry them all again only once though.
                    self.unreachable_ssh_nodes.clear()
                    tried_nodes.clear()
                    reset_enabled = False
                node = self.__get_ssh_node()
        raise RuntimeError(f"Could not find a reachable ssh node among {tried_nodes}")

    async def __is_ssh_node_reachable(self: Self, node: str, launch_id: str) -> bool:
        """"""
        assert node, "Node must be provided, otherwise we have an infinite loop here"
        cmds = await self.create_ssh_base_cmd(node=node)
        cmds.append("-o")
        cmds.append(f"ConnectTimeout={self.ssh_timeout}")
        cmds.append("echo")
        cmds.append("testing node availability")
        try:
            await launch_command_and_raise_errors(
                command_list=cmds, launch_id=launch_id
            )
            return True
        except:
            return False

    def _prepare_assets_replace_vars(
        self: Self,
        jobsub_path: Path,
        replace_vars: Dict[str, str],
        output_path: Path,
    ) -> str:
        """
        Reads the template at jobsub_path, applies variable replacements, and writes
        the result to output_path. The template file is never modified.
        Returns the resulting script content.
        """
        with open(jobsub_path, "r", encoding="utf-8") as f:
            jobsub_data = f.read()
        logger.info("before replacing jobsub_data: %s", jobsub_data)
        for k, v in replace_vars.items():
            kk = REPLACE_THIS_PREFIX + k.upper()
            logger.info("replacing key: %s with value: %s", kk, v)
            jobsub_data = jobsub_data.replace(kk, v)
            logger.info("after replacing jobsub_data: %s", jobsub_data)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(jobsub_data)
        os.chmod(output_path, os.stat(jobsub_path).st_mode)
        return jobsub_data

    def _build_cmd_to_run_with_ssh(
        self: Self,
        jobsub_path: Path,
        env_vars: Optional[Dict] = None,
    ) -> Tuple[str, str]:
        """
        Build ``[KEY=val ...] /path/jobsub.sh`` as a remote command string.

        Returns:
            Tuple of (command_str, redacted_command_str) where secret values
            in env_vars are replaced with ``<redacted>`` in the redacted version.
        """
        parts: List[str] = []
        to_redact: set[str] = set()
        if env_vars:
            logger.info("injecting env_vars: %s", env_vars.keys())
            for k, v in env_vars.items():
                parts.append(f"{k}={v}")
                to_redact.add(str(v))
        parts.append(str(jobsub_path))
        cmd = " ".join(parts)
        redacted = cmd
        for v in to_redact:
            redacted = redacted.replace(v, "'<redacted>'")
        return cmd, redacted

    def _get_local_bsub_command(
        self: Self,
        launch_id: str,
        jobsub_path: Path,
        env_vars: Optional[Dict] = None,
    ) -> Tuple[List[str], str]:
        logger.info("jobsub_path: %s", jobsub_path)
        # if self.use_ssh:
        #     logger.info("using ssh to launch")
        #     ssh_cmd = self.create_ssh_base_cmd(launch_id=launch_id)
        #     env_cmd, env_cmd_redacted = self._build_cmd_to_run_with_ssh(jobsub_path, env_vars)
        #     command: List[str] = ssh_cmd + [env_cmd]
        #     redacted_command_str = cmd_safe_join(ssh_cmd) + " " + env_cmd_redacted
        # else:
        command = [str(jobsub_path)]
        redacted_command_str = cmd_safe_join(command)
        return command, redacted_command_str

    @staticmethod
    def _get_rel_jobsub_path(step_name: str) -> Path:
        return Path(LSF_SCRIPTS) / step_name / JOB_SUB_SH

    def _prepare_assets(
        self: Self,
        launch_id: str,
        asset_dir: Path,
        final_asset_dir: Path,
        **kwargs,
    ) -> Tuple[Path, Path, str]:
        """Returns (jobsub_path, final_jobsub_path, jobsub_data)"""
        step_config_dict = kwargs.get("config", {})
        assert isinstance(
            step_config_dict, dict
        ), f"invalid config type: {type(step_config_dict).__name__} (expected 'dict' type)"
        step_config_section = StepConfigSection.model_validate(step_config_dict)
        self._set_log_path(
            launch_id=launch_id,
            step_config_section=step_config_section,
            final_asset_dir=final_asset_dir,
        )
        step_name = kwargs.get("step", {}).get("name", "")
        if step_name == "":
            step_name = kwargs.get("run_metadata", {}).get("target_name")
            assert step_name != "", "step_name and target_name is empty"
        assert isinstance(step_name, str), f"step_name is not a string: {step_name}"
        rel_jobsub_path = self._get_rel_jobsub_path(step_name=step_name)
        jobsub_path = asset_dir / rel_jobsub_path
        assert jobsub_path.is_file(), f"expected '{jobsub_path}' to be a file"
        logger.info("jobsub_path: %s", jobsub_path)
        if self.use_ssh:
            # For SSH: write replaced script to a launch-specific local file so the
            # template stays untouched. rsync will transfer it to the remote host.
            output_path = (
                jobsub_path.parent
                / f"{jobsub_path.stem}_{launch_id}{jobsub_path.suffix}"
            )
            final_jobsub_path = final_asset_dir / output_path.relative_to(asset_dir)
        else:
            # For non-SSH: the copy to final_asset_dir has already happened, so write
            # the replaced script directly to the copy. The template in asset_dir is
            # never modified, so retries always find the original placeholders intact.
            output_path = final_asset_dir / rel_jobsub_path
            final_jobsub_path = output_path
        replace_vars = {
            "launch_id": launch_id,
            "asset_dir": str(final_asset_dir),
        }
        jobsub_data = self._prepare_assets_replace_vars(
            jobsub_path=jobsub_path,
            replace_vars=replace_vars,
            output_path=output_path,
        )
        return (jobsub_path, final_jobsub_path, jobsub_data)

    async def _copy_assets(
        self: Self,
        launch_id: str,
        asset_dir: Path,
        **kwargs,
    ) -> Tuple[Path, Path, Path, str]:
        """Returns (final_asset_dir, jobsub_path, final_jobsub_path, jobsub_data)"""
        run_metadata = kwargs.get("run_metadata")
        assert isinstance(
            run_metadata, dict
        ), f"invalid run_metadata type: {type(run_metadata).__name__} (expected 'dict' type)"
        step_name = kwargs.get("step", {}).get("name", "")
        final_asset_dir = self._get_final_asset_dir(
            asset_dir=asset_dir,
            use_ssh=self.use_ssh,
            launch_id=launch_id,
            run_metadata=run_metadata,
            step_name=step_name,
        )
        self._pending_cleanup_dirs.add(final_asset_dir)
        logger.info("final_asset_dir: %s", final_asset_dir)
        if self.use_ssh:
            # For SSH: prepare assets BEFORE rsyncing so the replaced file exists locally
            # and rsync can transfer it to the remote host alongside the other assets.
            jobsub_path, final_jobsub_path, jobsub_data = self._prepare_assets(
                launch_id=launch_id,
                asset_dir=asset_dir,
                final_asset_dir=final_asset_dir,
                **kwargs,
            )
            logger.info("using ssh, copying the asset into the env")
            ssh_tunnel = self._ssh_tunnel
            assert ssh_tunnel
            logger.info("copying %s to %s", asset_dir, final_asset_dir)
            # Create remote destination directory via tunnel
            logger.info("creating remote directory %s via tunnel", final_asset_dir)
            await ssh_tunnel.run_remote_with_retries(
                f"mkdir -p {final_asset_dir} || true"
            )
            # Copy assets to remote host (local-to-remote transfer)
            if self.copy_method == "rsync":
                copy_cmd = await self._create_rsync_cmd(
                    launch_id=launch_id,
                    src=str(asset_dir),
                    dest=str(final_asset_dir),
                )
            else:
                copy_cmd = await self._create_scp_cmd(
                    launch_id=launch_id,
                    src=str(asset_dir),
                    dest=str(final_asset_dir),
                )
            logger.info("copy command (%s): %s", self.copy_method, copy_cmd)
            returncode, stdout, stderr = await ssh_tunnel.run_local_with_retries(
                command=copy_cmd,
                launch_id=launch_id,
            )
            logger.info(
                "copy command returncode %s stdout %s stderr %s",
                returncode,
                stdout,
                stderr,
            )
        else:
            # For non-SSH: copy assets FIRST (template copied as-is to final_asset_dir),
            # then apply replacements to the copy. The template in asset_dir is never
            # modified, so retries always find the original placeholders intact.
            # (final_asset_dir is unique per launch_id, so each retry gets its own copy.)
            logger.info("not using ssh, make a non-temp copy of the assets")
            logger.info("copying %s to %s", asset_dir, final_asset_dir)
            sync_or_copy(
                src=str(asset_dir) + "/",
                dest=final_asset_dir,
                delete=False,
                raise_errors=True,
            )
            jobsub_path, final_jobsub_path, jobsub_data = self._prepare_assets(
                launch_id=launch_id,
                asset_dir=asset_dir,
                final_asset_dir=final_asset_dir,
                **kwargs,
            )
        return (final_asset_dir, jobsub_path, final_jobsub_path, jobsub_data)

    async def launch_bsub(
        self: Self,
        launch_id: str,
        targetsteprun_asset_dir: Optional[Path] = None,
        **kwargs,
    ) -> None:
        """Launch a job (via SSH) in the LSF login node using a script (via call to _synchronized_launch method).
        Called by reflection.
        """
        step_config_dict = kwargs.get("config", {})
        assert isinstance(
            step_config_dict, dict
        ), f"invalid config type: {type(step_config_dict).__name__} (expected 'dict' type)"
        step_config_section = StepConfigSection.model_validate(step_config_dict)
        config_lsf_bsub = step_config_section.lsf.bsub
        existing_jobid = config_lsf_bsub.jobid
        if existing_jobid != "":
            logger.warning("there is an existing LSF job: %s", config_lsf_bsub)
            self._existing_jobids[launch_id] = ExistingBsubJobs(job_id=existing_jobid)
            log_path = config_lsf_bsub.log_path
            assert (
                log_path != ""
            ), f"invalid config_lsf_bsub.log_path: {step_config_section}"
            self._set_log_path(launch_id=launch_id, log_path=log_path)
            self._launched_jobs[launch_id] = existing_jobid
            self._release_monitors(launch_id)
            msg = f"⚡ LSF job has already been launched with JobID={existing_jobid}"
            self._send_message(msg=msg, **kwargs)
            return
        assert targetsteprun_asset_dir is not None, "targetsteprun_asset_dir is None"
        final_asset_dir, jobsub_path, final_jobsub_path, jobsub_data = (
            await self._copy_assets(
                launch_id=launch_id,
                asset_dir=targetsteprun_asset_dir,
                **kwargs,
            )
        )
        # Get useful env vars to inject for LhPull and LhPush
        env_vars = {}
        secrets_to_inject = (
            kwargs.get("config", {})
            .get("lsf", {})
            .get("secrets", {})
            .get("secret_names_to_use_as_env_variable", [])
        )
        # {
        #     env_name: Optional[str] = None
        #     secret_name: Optional[str] = None
        # }
        assert isinstance(
            secrets_to_inject, list
        ), f"invalid secrets_to_inject type: {type(secrets_to_inject).__name__} (expected 'list')"
        if len(secrets_to_inject) > 0:
            space_secrets = kwargs.get("setup_config", {}).get("space_secrets", {})
            assert isinstance(
                space_secrets, dict
            ), f"invalid space_secrets class: {type(space_secrets).__name__} (expected 'dict')"
            assert len(space_secrets) > 0, "empty space_secrets"
            all_keys = list(space_secrets.keys())
            for secret_to_inject in secrets_to_inject:
                assert isinstance(
                    secret_to_inject, dict
                ), f"invalid secret_to_inject class: {type(secret_to_inject).__name__} (expected 'dict')"
                env_var_name = secret_to_inject["env_name"]
                secret_name = secret_to_inject["secret_name"]
                logger.info(
                    "looking up secret %s for env var %s", secret_name, env_var_name
                )
                assert (
                    secret_name in space_secrets
                ), f"failed to find the secret {secret_name} in {all_keys}"
                env_vars[env_var_name] = space_secrets[secret_name]
        try:
            if self.use_ssh:
                ssh_tunnel = self._ssh_tunnel
                assert ssh_tunnel
                remote_cmd, redacted_cmd = self._build_cmd_to_run_with_ssh(
                    final_jobsub_path, env_vars
                )
                msg = f"⚡ Launching LSF job with command:\n```\n{redacted_cmd}\n```"
                self._send_message(msg=msg, **kwargs)
                _, stdout, stderr = await ssh_tunnel.run_remote_with_retries(
                    command=remote_cmd, redacted_command=redacted_cmd
                )
            else:
                command, redacted_command_str = self._get_local_bsub_command(
                    launch_id=launch_id,
                    jobsub_path=final_jobsub_path,
                    env_vars=env_vars,
                )
                msg = f"⚡ Launching LSF job with command:\n```\n{redacted_command_str}\n```"
                self._send_message(msg=msg, **kwargs)
                _, stdout, stderr = await launch_command_and_retry_or_raise_errors(
                    command_list=command,
                    launch_id=launch_id,
                    redacted_command_str=redacted_command_str,
                )
            output = f"stdout:\n{stdout}\nstderr:\n{stderr}"
            msg = f"⚡ Output of job submission:\n{output}"
            self._send_message(msg=msg, **kwargs)
            # store job ID if "Job <xxxx> is submitted" appears in stdout
            assert isinstance(stdout, str), f"invalid stdout: {stdout}"
            job_id_match = re.search(r"Job <(\d+)> is submitted", stdout)
            if job_id_match:
                job_id = job_id_match.group(1)
                self._launched_jobs[launch_id] = job_id
                self._release_monitors(launch_id)
                # Store launch kwargs for retry capability
                self._launch_kwargs[launch_id] = {
                    "targetsteprun_asset_dir": targetsteprun_asset_dir,
                    **kwargs,
                }
                msg = f"⚡ LSF job submitted with JobID={job_id}"
                self._send_message(msg=msg, **kwargs)
            else:
                raise RuntimeError(
                    f"failed to parse JobID from the submission output:\n{output}"
                )
        except Exception as e:
            raise RuntimeError("LSF job launch failed") from e

    async def setup_bsub(
        self: Self,
        setup_id: str,
        space_secrets: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict:
        """One time instance setup of the ssh keys and tunnel (via _synchronized_setup method)
        Called by reflection.
        """
        if not space_secrets:
            raise ValueError(f"setup_id: {setup_id} space_secrets should not be empty")

        if not self.use_ssh:
            logger.warning("setup_id: %s we are not using ssh", setup_id)
            return {
                "space_secrets": space_secrets,
                "space": {"secret": ""},
            }
        else:
            self.__setup_ssh(setup_id=setup_id, space_secrets=space_secrets)
            assert self._key_file_path
            # await self.__preload_unreachable_ssh_nodes()
            await self._open_ssh_tunnel(setup_id)
            return {
                "ssh_key_file": self._key_file_path,
                "space_secrets": space_secrets,
                "space": {"secret": ""},
            }

    def __setup_ssh(self: Self, setup_id: str, space_secrets: Dict[str, str]) -> None:
        ssh_key_secret_name = self.ssh_key_secret_name
        logger.info(
            "setup_id: %s getting ssh_key from space_secrets using name: %s",
            setup_id,
            ssh_key_secret_name,
        )
        self.ssh_key = space_secrets.get(ssh_key_secret_name, "")
        if not self.ssh_key:
            raise ValueError(
                f"setup_id: {setup_id} invalid ssh_key named {ssh_key_secret_name} in space_secrets"
            )
        self._key_file_path = self._create_ssh_key_file(setup_id)
        # self.created_setup_files[self._get_ssh_key(setup_id)] = self._key_file_path # file removed in teardown method
        logger.info("setup_id: %s SSH key written to %s", setup_id, self._key_file_path)

    async def _open_ssh_tunnel(
        self: Self,
        setup_id: str,
    ) -> None:
        """
        Resolve the SSH key from space_secrets, write it to a temp file,
        and open a persistent SshTunnel.  Returns the key file path.
        """
        login_node = await self._get_reachable_ssh_node()
        self._ssh_tunnel = SshTunnel(
            host=login_node,
            username=self.username,
            key_file=self._key_file_path,
            host_key_verification=self.ssh_host_key_verification,
            port_forwards=[(0, login_node, self.ssh_port)],
            max_sessions=self.ssh_max_sessions,
        )
        await self._ssh_tunnel.open()
        logger.info("setup_id: %s SSH tunnel opened to %s", setup_id, login_node)

    async def _cleanup_asset_dirs(self) -> None:
        # Delete all launch dirs now that all pipeline steps are done.
        # Must happen before the SSH tunnel is closed.
        dirs_to_delete = set(self._pending_cleanup_dirs)
        self._pending_cleanup_dirs.clear()
        ssh_tunnel = self._ssh_tunnel
        for dir_path in dirs_to_delete:
            if self.use_ssh:
                if ssh_tunnel is not None:
                    logger.info("removing remote asset dir %s", dir_path)
                    await ssh_tunnel.run_remote_with_retries(
                        f"rm -rf {dir_path} || true"
                    )
            else:
                logger.info("removing local asset dir %s", dir_path)
                shutil.rmtree(dir_path, ignore_errors=True)

    async def teardown_bsub(self: Self, setup_id: str) -> None:
        """
        Remove the SSH key file created in setup_bsub (see _synchronized_teardown method.
        Called via reflection.
        """
        # TODO: Don't clean these up because there is some ETE cron job running
        # that is pulling the logs from these asset dirs.  Until this is replaced
        # with a local push of the logs, we can't delete these.
        # self._cleanup_asset_dirs()

        ssh_tunnel = self._ssh_tunnel
        if ssh_tunnel is not None:
            await ssh_tunnel.close()
            self._ssh_tunnel = None
        key_file_path = self._key_file_path
        self._key_file_path = None
        if key_file_path and os.path.exists(key_file_path):
            temp_dir = os.path.dirname(key_file_path)
            try:
                os.unlink(key_file_path)
                os.rmdir(temp_dir)
                logger.info("SSH key file %s deleted", key_file_path)
            except Exception as e:
                logger.warning(
                    "failed to delete the SSH key at %s , error: %s", key_file_path, e
                )

    def monitor(self, type: str, launch_id: str, task_group, **kwargs):  # type: ignore[override]
        """Pre-populate _logfile_event_configs for the deprecated logfile_monitor.

        When step.yaml still declares a separate logfile_monitor, targetsteprun
        calls environment.monitor() synchronously for each monitor in a for-loop
        before any tasks run.  We capture the event_configs here (synchronously,
        race-free) so that monitor_bsub_monitor can read them without needing a
        scheduling yield.
        """
        if type == "logfile_monitor":
            event_configs = kwargs.get("event_configs")
            if event_configs:
                self._logfile_event_configs[launch_id] = event_configs
        return super().monitor(type, launch_id, task_group, **kwargs)

    async def monitor_bsub_monitor(
        self: Self,
        launch_id: str,
        event_q: Optional[asyncio.Queue] = None,
        entityrun_metadata: Optional[EntityRunMetadata] = None,
        event_configs: Optional[List] = None,
        **kwargs,
    ) -> None:
        """Monitor the bsub job status with retry for transient LSF errors.

        Also creates and manages a LogFileMonitor for the job's log file,
        running it concurrently with the bsub monitor within each retry iteration.
        """
        build_id = entityrun_metadata.build_id if entityrun_metadata else launch_id
        retry_complete_event = asyncio.Event()
        self._lsf_retry_complete_events[launch_id] = retry_complete_event
        step_id = (
            (entityrun_metadata.targetsteprun_id or "") if entityrun_metadata else ""
        )

        enabled, retry_transparently = self._get_step_retry_config(
            self._launch_kwargs.get(launch_id, {}),
        )
        async with self._with_retry_handler(
            launch_id,
            event_q,
            build_id,
            enabled=enabled,
            entityrun_metadata=entityrun_metadata,
            retry_transparently=retry_transparently,
        ) as monitor_queue:
            # _logfile_event_configs is pre-populated synchronously by monitor()
            # before any tasks run, so no scheduling yield is needed here.
            resolved_event_configs = event_configs or self._logfile_event_configs.pop(
                launch_id, None
            )
            event_log_parser_configs = []
            if resolved_event_configs is not None:
                event_log_parser_configs = [
                    EventLogLineParserConfig.model_validate(ec)
                    for ec in resolved_event_configs
                ]
            current_launch_id = launch_id
            try:
                while True:
                    stop_event = self._get_launch_stopped_event(
                        launch_id=current_launch_id
                    )
                    job_id = self._launched_jobs[current_launch_id]
                    log_file_path = self.get_log_path(launch_id=current_launch_id)

                    log_stream_source: Optional[LogStreamSource] = None
                    if self.use_ssh:
                        ssh_opts = self.ssh_no_verification_flags()
                        key_file_path = self._key_file_path
                        if key_file_path:
                            ssh_opts.extend(["-i", key_file_path])
                        host = await self._get_reachable_ssh_node()
                        log_stream_source = make_stream(
                            use_ssh=True,
                            host=host,
                            user=self.username,
                            ssh_opts=ssh_opts,
                            path=log_file_path,
                        )
                    else:
                        log_stream_source = make_stream(
                            use_ssh=False,
                            path=log_file_path,
                        )

                    logger.info(
                        "Starting LSF logfile monitoring for launch_id %s, using stream %s",
                        current_launch_id,
                        log_stream_source,
                    )
                    logfile_monitor = LogFileMonitor(
                        step_id=step_id,
                        stream_source=log_stream_source,
                        event_configs=event_log_parser_configs,
                        launch_id=current_launch_id,
                        entityrun_metadata=entityrun_metadata,
                        event_queue=monitor_queue,
                        stop_event=stop_event,
                    )

                    logger.info(
                        "Starting LSF bsub monitoring for job_id %s launch_id %s",
                        job_id,
                        current_launch_id,
                    )
                    lsf_bsub_monitor = LSFBsubMonitor(
                        lsf=self,
                        job_id=job_id,
                        launch_id=current_launch_id,
                        entityrun_metadata=entityrun_metadata,
                        event_queue=monitor_queue,  # RetryHandler reading this looking for the need to retry
                        stop_event=stop_event,
                    )
                    retry_complete_event.clear()
                    await asyncio.gather(
                        lsf_bsub_monitor.monitor(),  # Detects end sets stop_event
                        logfile_monitor.monitor(),  # Stops when stop_event is set
                    )
                    is_retry = retry_complete_event.is_set()
                    if is_retry:
                        # LSFBsubMonitor emitted a transient error event; RetryHandler
                        # triggered retry_workload which set this event. Loop for the
                        # next iteration with the same launch_id.
                        continue
                    logger.info(
                        "LSF bsub monitoring finished for job_id %s launch_id %s",
                        job_id,
                        current_launch_id,
                    )
                    return  # Success
            finally:
                self._lsf_retry_complete_events.pop(launch_id, None)

    def ssh_no_verification_flags(self: Self) -> List[str]:
        """Flags to disable SSH Host key verification."""
        if self.ssh_host_key_verification:
            logger.info("ssh host key verification is enabled")
            return []
        logger.warning("ssh host key verification is disabled")
        return [
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "StrictHostKeyChecking=no",
        ]

    async def monitor_logfile_monitor(
        self: Self,
        launch_id: str,
        event_configs: Optional[List] = None,
        **kwargs,
    ) -> None:
        """Deprecated: log file monitoring is now handled by monitor_bsub_monitor.

        Forwards event_configs to monitor_bsub_monitor for backward compatibility
        with step.yaml files that still declare a separate logfile_monitor.
        """
        logger.warning(
            "[Lsf] monitor_logfile_monitor is deprecated; "
            "log file monitoring is now handled by monitor_bsub_monitor. "
            "Move event_configs under bsub_monitor and remove logfile_monitor from your step.yaml."
        )
        # if event_configs:
        #     self._logfile_event_configs[launch_id] = event_configs

    async def pullasset_envstore(
        self: Self,
        uri: URI,
        binding: Optional[Any] = None,
        storeload_config: Optional[StoreLoad] = None,
        assetstore: Optional[Assetstore] = None,
        secrets: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[Dict, Optional[BuildTargetStepConfig]]:
        """Load an asset from the env asset store"""
        envuri = uri if isinstance(uri, URI) else URI.get_uri(uri)
        assert isinstance(envuri, EnvURI), f"invalid envuri: {envuri}"
        assert envuri.uri, f"invalid envuri: {envuri}"
        final_binding_path = envuri.uri.path
        assert final_binding_path, f"invalid envuri: {envuri}"
        binding_config = {BINDING_KEY: {"path": final_binding_path}}
        logger.info("loaded env uri: %s at binding: %s", uri, binding_config)
        return (binding_config, None)

    def _load_builtin_lh_lsf_section(
        self: Self, step_dir: Path, lh_metadata: dict
    ) -> Tuple[dict, dict]:
        """Read a builtin LH step YAML and return (lsf_section_dict, workload_section_dict)
        with the LAKEHOUSE_TOKEN secret injected into the LSF section.

        Args:
            step_dir: Directory containing the builtin step YAML (e.g. lhpull or lhpush dir).
            lh_metadata: Lakehouse metadata dict; must contain 'token_secretname'.
        Returns:
            Tuple of (lsf_dict, workload_dict) ready for BuildTargetStepConfig config.
        Raises:
            AssertionError: if the step YAML is missing or 'token_secretname' is absent.
        """
        step_path = step_dir / STEP_FILE_NAME
        assert step_path.is_file(), f"step yaml is missing: {step_path}"
        step_config = StepConfig.from_yaml(path=step_path)
        lhpc = step_config.config
        assert isinstance(
            lhpc, dict
        ), f"invalid step config type: {type(lhpc).__name__}"
        step_config_section = StepConfigSection.model_validate(lhpc)
        logger.info("step_config_section: %s", step_config_section)
        lsf_dict = step_config_section.lsf.model_dump(exclude_unset=True)
        workload_dict = step_config_section.workload.model_dump(exclude_unset=True)
        assert isinstance(
            lh_metadata, dict
        ), f"invalid lh_metadata type: {type(lh_metadata).__name__}"
        assert (
            "token_secretname" in lh_metadata
        ), "token_secretname is missing in lh_metadata"
        lsf_dict["secrets"] = {
            "secret_names_to_use_as_env_variable": [
                {
                    "env_name": "LAKEHOUSE_TOKEN",
                    "secret_name": lh_metadata["token_secretname"],
                }
            ]
        }
        lsf_dict["skip_finding_output_artifacts"] = True
        return lsf_dict, workload_dict

    def _load_builtin_hf_lsf_section(
        self: Self, step_dir: Path, hf_metadata: dict
    ) -> Tuple[dict, dict]:
        """Read a builtin HF step YAML and return (lsf_section_dict, workload_section_dict)
        with the HF_TOKEN secret injected into the LSF section.

        Args:
            step_dir: Directory containing the builtin step YAML (e.g. hfpull or hfpush dir).
            hf_metadata: HF metadata dict; must contain 'token_secretname'.
        Returns:
            Tuple of (lsf_dict, workload_dict) ready for BuildTargetStepConfig config.
        Raises:
            AssertionError: if the step YAML is missing or 'token_secretname' is absent.
        """
        step_path = step_dir / STEP_FILE_NAME
        assert step_path.is_file(), f"step yaml is missing: {step_path}"
        step_config = StepConfig.from_yaml(path=step_path)
        hfpc = step_config.config
        assert isinstance(
            hfpc, dict
        ), f"invalid step config type: {type(hfpc).__name__}"
        step_config_section = StepConfigSection.model_validate(hfpc)
        logger.info("step_config_section: %s", step_config_section)
        lsf_dict = step_config_section.lsf.model_dump(exclude_unset=True)
        workload_dict = step_config_section.workload.model_dump(exclude_unset=True)
        assert isinstance(
            hf_metadata, dict
        ), f"invalid hf_metadata type: {type(hf_metadata).__name__}"
        assert (
            "token_secretname" in hf_metadata
        ), "token_secretname is missing in hf_metadata"
        lsf_dict["secrets"] = {
            "secret_names_to_use_as_env_variable": [
                {
                    "env_name": "HF_TOKEN",
                    "secret_name": hf_metadata["token_secretname"],
                }
            ]
        }
        lsf_dict["skip_finding_output_artifacts"] = True
        return lsf_dict, workload_dict

    async def pullasset_lhstore(
        self: Self,
        uri: URI,
        binding: Optional[Any] = None,
        storeload_config: Optional[StoreLoad] = None,
        assetstore: Optional[Assetstore] = None,
        secrets: Optional[dict] = None,
        **kwargs: Dict,
    ) -> Tuple[Dict, Optional[BuildTargetStepConfig]]:
        """
        Allow for a asset from lh to be made available in k8s
        """
        assert isinstance(
            assetstore, Lhstore
        ), f"invalid type assetstore: {type(assetstore).__name__} (expected 'Lhstore')"
        assert storeload_config is not None, "storeload_config is None"
        assert (
            storeload_config.mode == "dmf_pull"
        ), f"Only 'dmf_pull' mode is supported for Lsf, mode: {storeload_config.mode} uri: {uri}"
        cache_path = storeload_config.config.get("cache_path", None)
        assert isinstance(cache_path, str), f"invalid cache_path: {cache_path}"
        assert cache_path != "", f"invalid cache_path: {cache_path}"
        lhuri = uri if isinstance(uri, URI) else URI.get_uri(uri)
        assert isinstance(lhuri, LhURI), f"invalid lhuri: {lhuri}"
        binding_path = Path(cache_path) / lhuri.hash()
        lhuristr = URI.get_uristr(lhuri)
        lh_metadata = Asset(uri=lhuri).get_metadata()
        logger.info("lh_metadata: %s", lh_metadata)
        logger.info("LSF_USE_ASPERA %s", LSF_USE_ASPERA)
        lhpull_config = {
            "use_aspera": LSF_USE_ASPERA,
            "path": str(binding_path),
            "uri": lhuristr,
            "lh": lh_metadata,
        }
        logger.info("lhpull_config: %s", lhpull_config)
        lhpull_stepuri = "space://steps/lhpull"
        if (
            storeload_config is not None
            and storeload_config.config is not None
            and "step_uri" in storeload_config.config
        ):
            lhpull_stepuri = storeload_config.config["step_uri"]
            assert isinstance(
                lhpull_stepuri, str
            ), f"invalid lhpull_stepuri: {lhpull_stepuri}"
        final_binding_path = binding_path / assetstore.get_subdir(uri)
        binding_config = {BINDING_KEY: {"path": str(final_binding_path)}}
        lsf_dict, workload_dict = self._load_builtin_lh_lsf_section(
            _LSF_LHPULL_DIR, lh_metadata
        )
        return binding_config, BuildTargetStepConfig(
            step_uri=lhpull_stepuri,
            config={
                "lsf": lsf_dict,
                "workload": workload_dict,
                "lhpull_config": lhpull_config,
            },
        )

    async def pushasset_lhstore(
        self: Self,
        binding: Any,
        binding_id: Optional[str] = "",
        storepush_config: Optional[StorePush] = None,
        uri: Optional[Union[str, URI]] = None,
        assetstore: Optional[Assetstore] = None,
        **kwargs: Dict,
    ) -> BuildTargetStepConfig:
        """
        Allow for a random folder/file to be copied from any mounted storage in the cluster to a lh bucket
        """
        if uri is None or uri == "":
            raise ValueError(f"Empty uri received to pushasset {binding}")
        lhuri = uri if isinstance(uri, URI) else URI.get_uri(uri)
        assert isinstance(lhuri, LhURI), f"invalid lhuri: {lhuri}"
        logger.info("binding type %s value %s", type(binding), binding)
        assert isinstance(
            binding, dict
        ), f"expected binding to be a dict, actual: {type(binding).__name__} {binding}"
        assert (
            "path" in binding
        ), f"expected 'path' to be in the binding, actual: {binding}"
        binding_path = binding["path"]
        logger.info("binding_path: %s", binding_path)
        binding_path_path = Path(binding_path)
        logger.info("binding_path_path: %s", binding_path_path)
        assert (
            len(binding_path_path.parts) >= 2
        ), f"expected at least 2 parts to the path: {binding_path_path}"
        volume = str(binding_path_path.parts[1])
        logger.info("volume: %s", volume)
        lhuristr = URI.get_uristr(lhuri)
        lh_metadata = Asset(lhuri).get_metadata()
        logger.info("lh_metadata: %s", lh_metadata)
        logger.info("LSF_USE_ASPERA %s", LSF_USE_ASPERA)
        lhpush_config = {
            "use_aspera": LSF_USE_ASPERA,
            "path": binding_path,
            "volume": volume,
            "uri": lhuristr,
            "binding_id": binding_id,
            "lh": lh_metadata,
        }
        logger.info("lhpush_config: %s", lhpush_config)
        lhpush_stepuri = "space://steps/lhpush"
        if (
            storepush_config is not None
            and storepush_config.config is not None
            and "step_uri" in storepush_config.config
        ):
            lhpush_stepuri = storepush_config.config["step_uri"]
            assert isinstance(
                lhpush_stepuri, str
            ), f"invalid lhpush_stepuri: {lhpush_stepuri}"
        lsf_dict, workload_dict = self._load_builtin_lh_lsf_section(
            _LSF_LHPUSH_DIR, lh_metadata
        )
        return BuildTargetStepConfig(
            step_uri=lhpush_stepuri,
            config={
                "lsf": lsf_dict,
                "workload": workload_dict,
                "lhpush_config": lhpush_config,
            },
        )

    async def pullasset_hfstore(
        self: Self,
        uri: URI,
        binding: Optional[Any] = None,
        storeload_config: Optional[StoreLoad] = None,
        assetstore: Optional[Assetstore] = None,
        secrets: Optional[dict] = None,
        **kwargs: Dict,
    ) -> Tuple[Dict, Optional[BuildTargetStepConfig]]:
        """Pull an asset from Hugging Face Hub to LSF cluster storage.

        Args:
            uri: HF URI to pull (e.g. hf://models/org/repo).
            binding: Unused for hfpull.
            storeload_config: Must have mode 'hf_pull' and config with 'cache_path'.
            assetstore: Hfstore instance.
            secrets: Optional secrets dict.
        Returns:
            Tuple of (binding_config, BuildTargetStepConfig).
        Raises:
            AssertionError: If assetstore type or mode is invalid.
        """
        assert isinstance(
            assetstore, Hfstore
        ), f"invalid type assetstore: {type(assetstore).__name__} (expected 'Hfstore')"
        assert storeload_config is not None, "storeload_config is None"
        assert (
            storeload_config.mode == "hf_pull"
        ), f"Only 'hf_pull' mode is supported for Lsf, mode: {storeload_config.mode} uri: {uri}"
        cache_path = storeload_config.config.get("cache_path", None)
        assert isinstance(cache_path, str), f"invalid cache_path: {cache_path}"
        assert cache_path != "", f"invalid cache_path: {cache_path}"
        hfuri = uri if isinstance(uri, HfURI) else HfURI.parse(uri)  # type: ignore[arg-type]
        binding_path = (
            Path(cache_path) / hfuri.get_owner() / hfuri.get_repo() / hfuri.hash()
        )
        hf_metadata = Asset(uri=hfuri).get_metadata()
        logger.info("hf_metadata: %s", hf_metadata)
        hfpull_config = Hfstore.build_hfpull_step_config(
            hfuri=hfuri,
            binding_path=str(binding_path),
        )
        logger.info("hfpull_config: %s", hfpull_config)
        hfpull_stepuri = "space://steps/hfpull"
        if (
            storeload_config is not None
            and storeload_config.config is not None
            and "step_uri" in storeload_config.config
        ):
            hfpull_stepuri = storeload_config.config["step_uri"]
            assert isinstance(
                hfpull_stepuri, str
            ), f"invalid hfpull_stepuri: {hfpull_stepuri}"
        binding_config = {BINDING_KEY: {"path": str(binding_path)}}
        lsf_dict, workload_dict = self._load_builtin_hf_lsf_section(
            _LSF_HFPULL_DIR, hf_metadata
        )
        return binding_config, BuildTargetStepConfig(
            step_uri=hfpull_stepuri,
            config={
                "lsf": lsf_dict,
                "workload": workload_dict,
                "hfpull_config": hfpull_config,
            },
        )

    async def pushasset_hfstore(
        self: Self,
        binding: Any,
        binding_id: Optional[str] = "",
        storepush_config: Optional[StorePush] = None,
        uri: Optional[Union[str, URI]] = None,
        assetstore: Optional[Assetstore] = None,
        output_config: Optional[BuildTargetOutputConfig] = None,
        **kwargs: Dict,
    ) -> BuildTargetStepConfig:
        """Push an artifact from LSF cluster storage to Hugging Face Hub.

        Pre-creates the target HF repo server-side (with the correct
        resource_group_id resolved from the space name) so the LSF compute
        node's ``huggingface-cli upload`` only has to push files to a repo
        that already exists.

        Args:
            binding: Dict with a 'path' key pointing to the artifact on cluster.
            binding_id: Output binding name for artifact tracking.
            storepush_config: Environment-level push configuration.
            uri: Target HF URI string or object.
            assetstore: Hfstore instance.
            output_config: Per-output config from build.yaml; ``space_name`` is
                used to derive the HF Enterprise resource group.
        Returns:
            BuildTargetStepConfig for the hfpush step.
        Raises:
            ValueError: If uri is empty or the resource group cannot be resolved.
            AssertionError: If binding has no 'path'.
        """
        if uri is None or uri == "":
            raise ValueError(f"Empty uri received to pushasset {binding}")
        hfuri = uri if isinstance(uri, HfURI) else HfURI.parse(uri)  # type: ignore[arg-type]
        logger.info("binding type %s value %s", type(binding), binding)
        assert isinstance(
            binding, dict
        ), f"expected binding to be a dict, actual: {type(binding).__name__} {binding}"
        assert (
            "path" in binding
        ), f"expected 'path' to be in the binding, actual: {binding}"
        binding_path = binding["path"]
        logger.info("binding_path: %s", binding_path)
        hf_metadata = Asset(uri=hfuri).get_metadata()
        logger.info("hf_metadata: %s", hf_metadata)

        hf_resource_group_id: Optional[str] = None
        hf_resource_group_name: Optional[str] = None
        hf_private = True
        # Environment-level storepush_config (lower priority).
        if storepush_config is not None and storepush_config.config is not None:
            hf_cfg = storepush_config.config.get("hf", {})
            hf_resource_group_id = hf_cfg.get("resource_group_id", hf_resource_group_id)
            hf_resource_group_name = hf_cfg.get(
                "resource_group_name", hf_resource_group_name
            )
            hf_private = hf_cfg.get("private", hf_private)
        # build.yaml output store_push (higher priority, overrides).
        if output_config is not None and output_config.store_push is not None:
            hf_cfg = output_config.store_push.config.get("hf", {})
            hf_resource_group_id = hf_cfg.get("resource_group_id", hf_resource_group_id)
            hf_resource_group_name = hf_cfg.get(
                "resource_group_name", hf_resource_group_name
            )
            hf_private = hf_cfg.get("private", hf_private)

        space_name = output_config.space_name if output_config else None

        assert isinstance(
            assetstore, Hfstore
        ), f"invalid assetstore: {type(assetstore).__name__} (expected 'Hfstore')"
        if hf_resource_group_id:
            resource_group_id: Optional[str] = hf_resource_group_id
        else:
            resource_group_id = hfuri.resolve_resource_group_id(
                token=assetstore.resolve_token(hfuri),
                resource_group_name=hf_resource_group_name,
                space_name=space_name,
            )

        hfpush_config = Hfstore.build_hfpush_step_config(
            hfuri=hfuri,
            binding_path=binding_path,
            binding_id=binding_id or "",
            hf_private=hf_private,
            hf_resource_group_id=resource_group_id,
        )
        logger.info("hfpush_config: %s", hfpush_config)
        hfpush_stepuri = "space://steps/hfpush"
        if (
            storepush_config is not None
            and storepush_config.config is not None
            and "step_uri" in storepush_config.config
        ):
            hfpush_stepuri = storepush_config.config["step_uri"]
            assert isinstance(
                hfpush_stepuri, str
            ), f"invalid hfpush_stepuri: {hfpush_stepuri}"
        lsf_dict, workload_dict = self._load_builtin_hf_lsf_section(
            _LSF_HFPUSH_DIR, hf_metadata
        )
        return BuildTargetStepConfig(
            step_uri=hfpush_stepuri,
            config={
                "lsf": lsf_dict,
                "workload": workload_dict,
                "hfpush_config": hfpush_config,
            },
        )

    def _load_builtin_cos_lsf_section(self: Self, cos_metadata: dict) -> dict:
        """Read the cosrclone builtin step YAML and return an lsf_section_dict
        with COS credentials injected as environment secrets.

        Args:
            cos_metadata: COS asset metadata; must contain
                'cos_secret_access_key_secret_name' and 'cos_access_key_id_secret_name'.
        Returns:
            lsf_dict ready to merge into BuildTargetStepConfig config.
        Raises:
            AssertionError: if the cosrclone step YAML is missing or required
                secret name keys are absent from cos_metadata.
        """
        cosrclone_step_path = (
            _LSF_COSRCLONE_DIR / STEP_FILE_NAME
        )
        assert cosrclone_step_path.is_file(), "cosrclone step is missing"
        step_section = StepConfigSection.model_validate(
            StepConfig.from_yaml(path=cosrclone_step_path).config
        )
        lsf_dict = step_section.lsf.model_dump(exclude_unset=True)
        assert isinstance(cos_metadata, dict), f"invalid cos_metadata: {cos_metadata}"
        assert (
            "cos_secret_access_key_secret_name" in cos_metadata
        ), f"cos_secret_access_key_secret_name is missing: {cos_metadata}"
        assert (
            "cos_access_key_id_secret_name" in cos_metadata
        ), f"cos_access_key_id_secret_name is missing: {cos_metadata}"
        lsf_dict["secrets"] = {
            "secret_names_to_use_as_env_variable": [
                {
                    "env_name": "COS_SECRET_ACCESS_KEY",
                    "secret_name": cos_metadata["cos_secret_access_key_secret_name"],
                },
                {
                    "env_name": "COS_ACCESS_KEY_ID",
                    "secret_name": cos_metadata["cos_access_key_id_secret_name"],
                },
            ]
        }
        return lsf_dict

    async def pullasset_cosstore(
        self: Self,
        uri: URI,
        binding: Optional[Any] = None,
        storeload_config: Optional[StoreLoad] = None,
        assetstore: Optional[Assetstore] = None,
        secrets: Optional[dict] = None,
        **kwargs: Dict,
    ) -> Tuple[Dict, Optional[BuildTargetStepConfig]]:
        """
        Load data from COS bucket into the cluster filesystem.
        """
        assert isinstance(
            assetstore, Cosstore
        ), f"invalid type assetstore: {assetstore}"
        assert storeload_config is not None, "storeload_config is None"
        assert (
            storeload_config.mode == "cos_pull"
        ), f"Only 'cos_pull' mode is supported for COS in LSF. Got: {storeload_config.mode}"

        cosuri = uri if isinstance(uri, URI) else URI.get_uri(uri)
        assert isinstance(cosuri, CosURI), f"invalid cosuri: {cosuri}"

        cache_path = storeload_config.config.get("cache_path", None)
        assert (
            isinstance(cache_path, str) and cache_path != ""
        ), f"invalid cache_path: {cache_path}"

        binding_path = Path(cache_path) / cosuri.hash()
        bucket_path = cosuri.get_metadata()["bucket_path"]
        cos_metadata = Asset(uri=cosuri).get_metadata()
        assert isinstance(cos_metadata, dict), f"invalid cos_metadata: {cos_metadata}"

        cospull_config = {
            "path": str(binding_path),
            "uri_bucket_path": bucket_path,
            "push": False,
            "cos": cos_metadata,
        }
        cosrclone_stepuri = "space://steps/cosrclone"
        if (
            storeload_config.config is not None
            and "step_uri" in storeload_config.config
        ):
            cosrclone_stepuri = storeload_config.config["step_uri"]

        lsf_dict = self._load_builtin_cos_lsf_section(cos_metadata)
        binding_config = {BINDING_KEY: {"path": str(binding_path)}}
        return binding_config, BuildTargetStepConfig(
            step_uri=cosrclone_stepuri,
            config={
                "lsf": lsf_dict,
                "cos_config": cospull_config,
            },
        )

    async def pushasset_cosstore(
        self: Self,
        binding: Any,
        binding_id: Optional[str] = "",
        storepush_config: Optional[StorePush] = None,
        uri: Optional[Union[str, URI]] = None,
        assetstore: Optional[Assetstore] = None,
        **kwargs: Dict,
    ) -> BuildTargetStepConfig:
        """
        Copy folder/file from cluster filesystem to a COS bucket.
        """
        if uri is None or uri == "":
            raise ValueError(f"Empty uri received to pushasset {binding}")

        cosuri = uri if isinstance(uri, URI) else URI.get_uri(uri)
        assert isinstance(cosuri, CosURI), f"invalid cosuri: {cosuri}"

        assert isinstance(
            binding, dict
        ), f"expected binding to be a dict, got {type(binding)}"
        assert "path" in binding, f"missing 'path' in binding: {binding}"

        binding_path = binding["path"]
        binding_path_path = Path(binding_path)
        assert (
            len(binding_path_path.parts) >= 2
        ), f"invalid binding path: {binding_path_path}"

        volume = str(binding_path_path.parts[1])
        logger.info("volume: %s", volume)
        uri_bucket_path = cosuri.get_metadata()["bucket_path"]
        uri_bucket_name = cosuri.get_metadata()["bucket_name"]
        cos_metadata = Asset(cosuri).get_metadata()
        assert isinstance(cos_metadata, dict), f"invalid cos_metadata: {cos_metadata}"
        bucket_name = cos_metadata.get("cos_bucket_name") or uri_bucket_name
        if not uri_bucket_path.startswith(f"{bucket_name}/"):
            uri_bucket_path = (
                f"{bucket_name}/{uri_bucket_path}" if uri_bucket_path else bucket_name
            )

        cospush_config = {
            "path": binding_path,
            "push": True,
            "uri_bucket_path": uri_bucket_path,
            "bucket_name": bucket_name,
            "binding_id": binding_id,
            "cos": cos_metadata,
        }
        cosrclone_stepuri = "space://steps/cosrclone"
        if (
            storepush_config is not None
            and storepush_config.config is not None
            and "step_uri" in storepush_config.config
        ):
            cosrclone_stepuri = storepush_config.config["step_uri"]

        lsf_dict = self._load_builtin_cos_lsf_section(cos_metadata)
        return BuildTargetStepConfig(
            step_uri=cosrclone_stepuri,
            config={
                "lsf": lsf_dict,
                "cos_config": cospush_config,
            },
        )

    def __get_ssh_destination(self: Self, node: Optional[str] = None) -> str:
        """Get the user@host string defined by this instance and the optional node argument.

        Args:
            self (Self): _description_
            node (Optional[str], optional):

        Returns:
            str: one of node or user@node
        """
        u = self.username
        assert node, "Node must be provided"
        return f"{u + '@' if u else ''}{node}"

    async def create_ssh_base_cmd(self: Self, node: Optional[str] = None) -> List[str]:
        """Create the beginnings of the ssh command based on the optional arguments below.
        For example, ssh -i <a key file> user@node

        Args:
            'node' - if present will be used to create the target user@node destination otherwise
            a reachable node will be searched for and used.  If none are reachable, an exception is raised.

        Returns:
            List[str]: list of command tokens (beginning with ssh) to execute the ssh command.
                For example, ssh -i <key file path> someuser@somehost.
        """
        key_file_path = self._key_file_path
        assert key_file_path
        ssh_cmd = ["ssh"]
        ssh_cmd.extend(["-p", str(self.ssh_port)])
        ssh_cmd.extend(["-i", key_file_path])

        ssh_cmd.extend(self.ssh_no_verification_flags())

        if node:
            ssh_destination = self.__get_ssh_destination(node)
        else:
            ssh_destination = await self._get_reachable_ssh_destination()
        ssh_cmd.append(ssh_destination)

        return ssh_cmd

    async def _create_scp_cmd(
        self: Self, launch_id: str, src: str, dest: str, add_slashes: bool = True
    ) -> List[str]:
        """Create an SCP command for copying assets to the remote LSF node.

        Args:
            launch_id: The launch identifier.
            src: Source directory path.
            dest: Destination directory path on the remote host.
            add_slashes: Whether to ensure trailing slashes on src/dest.

        Returns:
            List of command tokens for the SCP invocation.
        """
        scp_cmd = ["scp", "-r"]
        key_file_path = self._key_file_path
        assert key_file_path, "SSH key file path is not set"
        scp_cmd.extend(["-i", key_file_path])
        scp_cmd.extend(self.ssh_no_verification_flags())
        if self.use_ssh:
            ssh_tunnel = self._ssh_tunnel
            assert ssh_tunnel
            ssh_dest = self.__get_ssh_destination(node="localhost")
            local_port = ssh_tunnel.get_local_port(ssh_tunnel.host, self.ssh_port)
            assert (
                local_port is not None
            ), "SSH tunnel has no local port forward for SSH"
            scp_cmd.extend(["-P", str(local_port)])
        else:
            ssh_dest = await self._get_reachable_ssh_destination()
        if add_slashes:
            # SCP uses /. to copy directory contents (rsync uses trailing /)
            src = src.rstrip("/") + "/."
            dest = dest if dest.endswith("/") else dest + "/"
            logger.info("after adding slashes src: %s dest: %s", src, dest)
        scp_cmd.append(src)
        scp_cmd.append(f"{ssh_dest}:{dest}")
        return scp_cmd

    async def _create_rsync_cmd(
        self: Self, launch_id: str, src: str, dest: str, add_slashes: bool = True
    ) -> List[str]:
        scp_cmd = [
            "rsync",
            "-chavzP",
            "--stats",
        ]
        key_file_path = self._key_file_path
        ssh_t1 = self.ssh_no_verification_flags()
        ssh_t2 = cmd_safe_join(ssh_t1)
        rsync_ssh = f"ssh -i {key_file_path} {ssh_t2}"
        if self.use_ssh:
            ssh_tunnel = self._ssh_tunnel
            assert ssh_tunnel
            ssh_dest = self.__get_ssh_destination(
                node="localhost"
            )  # localhost because of the port forwarding
            local_port = ssh_tunnel.get_local_port(ssh_tunnel.host, self.ssh_port)
            assert (
                local_port is not None
            ), "SSH tunnel has no local port forward for SSH"
            rsync_ssh = f"{rsync_ssh} -p {local_port}"
        else:
            ssh_dest = await self._get_reachable_ssh_destination()
        scp_cmd.extend(
            [
                "-e",
                rsync_ssh,
            ]
        )
        if add_slashes:
            src = src if src.endswith("/") else src + "/"
            dest = dest if dest.endswith("/") else dest + "/"
            logger.info("after adding slashes src: %s dest: %s", src, dest)
        scp_cmd.append(src)
        scp_cmd.append(f"{ssh_dest}:{dest}")
        return scp_cmd

    # def _create_copy_assets_cmd(
    #     self: Self, launch_id: str, src: str, dest: str
    # ) -> List[str]:
    #     # scp_cmd_copy = self._create_scp_cmd(launch_id=launch_id, src=src, dest=dest)
    #     scp_cmd_copy = self._create_rsync_cmd(launch_id=launch_id, src=src, dest=dest)
    #     scp_cmd_copy_str = cmd_safe_join(scp_cmd_copy)

    #     logger.info("scp_cmd_copy_str: %s", scp_cmd_copy_str)
    #     ssh_cmd_prefix = self.create_ssh_base_cmd(launch_id=launch_id)
    #     ssh_cmd_mkdir = ssh_cmd_prefix + [f"mkdir -p {dest}"]
    #     ssh_cmd_mkdir_str = cmd_safe_join(ssh_cmd_mkdir)
    #     logger.info("ssh_cmd_mkdir_str: %s", ssh_cmd_mkdir_str)
    #     copy_assets_cmd = [
    #         "bash",
    #         "-c",
    #         ssh_cmd_mkdir_str + " && " + scp_cmd_copy_str,
    #     ]
    #     return copy_assets_cmd

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

    def _get_final_asset_dir(
        self: Self,
        asset_dir: Path,
        use_ssh: bool,
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
        logger.info(
            "asset_dir: %s sub_dir: %s use_ssh: %s", asset_dir, sub_dir, use_ssh
        )
        final_asset_dir = sub_dir
        if self.lsf_workspace_remote_dir:
            final_asset_dir = Path(self.lsf_workspace_remote_dir) / sub_dir
        if not use_ssh:
            final_asset_dir = Path(self.lsf_workspace_local_dir) / sub_dir
            final_asset_dir = final_asset_dir.resolve()
        logger.info("final_asset_dir: %s", final_asset_dir)
        return final_asset_dir

    def _set_log_path(
        self: Self,
        launch_id: str = "",
        log_path: str = "",
        step_config_section: StepConfigSection = StepConfigSection(),
        final_asset_dir: Union[str, Path] = "",
    ) -> str:
        """Set the output directory for a specific launch ID"""
        if log_path != "":
            logger.info("setting launch_id %s -> log path %s", launch_id, log_path)
            self._log_paths[launch_id] = log_path
            return log_path
        logger.info(
            "computing log_path from config: %s and final_asset_dir: %s",
            step_config_section,
            final_asset_dir,
        )
        workspace_dir = step_config_section.workload.workspace_dir or str(
            final_asset_dir
        )
        assert isinstance(workspace_dir, str), f"invalid workspace_dir: {workspace_dir}"
        assert workspace_dir != "", f"invalid workspace_dir: {workspace_dir}"
        output_dir = (
            step_config_section.workload.output_dir or f"{workspace_dir}/outputs"
        )
        assert isinstance(output_dir, str), f"invalid output_dir: {output_dir}"
        assert output_dir != "", f"invalid output_dir: {output_dir}"
        log_path = str(Path(output_dir) / "job.log")
        logger.info("setting launch_id %s -> computed log path %s", launch_id, log_path)
        self._log_paths[launch_id] = log_path
        return log_path

    def get_log_path(self: Self, launch_id: str, default: Optional[str] = None) -> str:
        """Get the log path for a specific launch ID.

        Args:
            launch_id: The launch ID to look up.
            default: Value to return if launch_id is not present. If None
                (the default), KeyError is raised on a missing launch_id.

        Returns:
            The log path string, or `default` when launch_id is not present.

        Raises:
            KeyError: If launch_id is not present and `default` is None.
        """
        if default is None:
            return self._log_paths[launch_id]
        return self._log_paths.get(launch_id, default)

    async def _bkill(self: Self, launch_id: str, **kwargs) -> None:
        """Kill the LSF job associated with launch_id via bkill."""
        job_id = self._launched_jobs.get(launch_id)
        if not job_id:
            logger.warning("no jobid found for launch_id %s", launch_id)
            return

        if launch_id in self._existing_jobids:
            t1 = self._existing_jobids[launch_id]
            assert (
                job_id == t1.job_id
            ), f"launch_id {launch_id}: job id mismatch, current {job_id} stored {t1.job_id}"
            logger.warning(
                "launch_id %s job %s was launched outside our framework, skipping clean up",
                launch_id,
                job_id,
            )
            return

        try:
            if self.use_ssh:
                ssh_tunnel = self._ssh_tunnel
                assert ssh_tunnel
                logger.info("running cleanup command via tunnel: bkill %s", job_id)
                max_attempts = 3
                for attempt in range(1, max_attempts + 1):
                    try:
                        rc, stdout, stderr = await ssh_tunnel.run_remote(
                            f"bkill {job_id}", raise_on_error=False
                        )
                        break
                    except TimeoutError:
                        if attempt < max_attempts:
                            logger.warning(
                                "bkill %s timed out (attempt %d/%d), retrying",
                                job_id,
                                attempt,
                                max_attempts,
                            )
                            await asyncio.sleep(attempt * 2)
                        else:
                            logger.error(
                                "bkill %s timed out after %d attempts",
                                job_id,
                                max_attempts,
                            )
                            raise
            else:
                command = ["bkill", job_id]
                logger.info("running cleanup command: %s", cmd_safe_join(command))
                process, stdout, stderr = (
                    await launch_command_and_retry_or_raise_errors(
                        command_list=command,
                        launch_id=launch_id,
                        raise_error=False,
                    )
                )
                rc = process.returncode if process.returncode is not None else -1
            stderr_str = (
                stderr
                if isinstance(stderr, str)
                else stderr.decode("utf-8", errors="replace")
            )
            if "Job has already finished" in stderr_str:
                logger.info("Job %s is already finished, skipping kill.", job_id)
            elif rc != 0:
                msg = f"⚠️ Failed to kill LSF job {job_id}.\nstdout:\n{stdout}\nstderr:\n{stderr}"
                logger.error(
                    "bkill failed for job %s with rc=%s: %s", job_id, rc, stderr_str
                )
                self._send_message(msg=msg, **kwargs)
            else:
                msg = (
                    f"⚡ Killed LSF job {job_id}.\nstdout:\n{stdout}\nstderr:\n{stderr}"
                )
                self._send_message(msg=msg, **kwargs)
        except Exception as e:
            raise RuntimeError("LSF job cleanup failed") from e

    async def cleanup_bsub(self: Self, launch_id: str, **kwargs) -> None:
        """Clean up the launched job."""
        await self._bkill(launch_id, **kwargs)

    def _create_ssh_key_file(self: Self, launch_id: str) -> str:
        ssh_key = self.ssh_key
        assert ssh_key, "failed to find an ssh_key"
        temp_dir = tempfile.mkdtemp()
        suffix_hash = short_alphanumeric_lower_hash(launch_id)
        key_file_path = os.path.join(temp_dir, f"ssh_key_{suffix_hash}")
        with open(key_file_path, "w", encoding="utf-8") as f:
            f.write(ssh_key.rstrip("\n") + "\n")
        os.chmod(key_file_path, 0o600)
        logger.info(
            "SSH key file created at %s for launch_id %s", key_file_path, launch_id
        )
        return key_file_path

    def get_ssh_tunnel(self: Self) -> Optional[SshTunnel]:
        """Get the tunnel, if open and usable.

        Args:
            self (Self): _description_

        Returns:
            Optional[SshTunnel]: _description_
        """
        return self._ssh_tunnel
