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
Sidecar container based monitoring in K8s clusters.
"""

import argparse
import asyncio
import json
import os
import signal
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Union

import yaml

from gbserver.environment.environment import EventLogLineParserConfig
from gbserver.messaging.messaging_base import MessagingBase
from gbserver.monitoring.logfile_monitor import LogFileMonitor

try:
    from gbserver.messaging.rabbitmq_base import RabbitMQBase
except ImportError:
    RabbitMQBase = None  # type: ignore[assignment,misc]
from gbserver.monitoring.process_cmdline_monitor import CmdlineMonitor
from gbserver.monitoring.streams.local_file_stream import LocalFileStream
from gbserver.monitoring.streams.log_stream_base import LogStreamSource
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


class SidecarOrchestrator:
    """
    Orchestrates multiple sidecar instances for monitoring user processes.

    The orchestrator:
    - Loads configuration from a YAML file
    - Creates RabbitMQ messaging connection shared by all sidecars
    - Launches one or more Sidecar instances, each monitoring a log file
    - Waits for all sidecars to complete
    """

    def __init__(
        self: Self,
        exchange_name: str,
        queue_name: str,
        routing_key: Optional[str] = None,
        config_file_path: str = "monitor_config.yaml",
        custom_log_paths: Optional[str] = None,
    ):
        """Initialize the SidecarOrchestrator.

        Args:
            exchange_name: RabbitMQ exchange name for publishing events
            queue_name: RabbitMQ queue name for message routing
            routing_key: Optional routing key (format: "prefix.targetsteprun_id.launch_id")
            config_file_path: Full path to the YAML configuration file
            custom_log_paths: Optional comma-separated paths to user process log files.
                            If not provided, defaults to /logs/output.log or /logs/output-{idx}.log
        """

        # initialized through the NUM_USER_PROCESSES environment variable
        self.num_user_processes = int(os.getenv("NUM_USER_PROCESSES", "1"))
        assert (
            self.num_user_processes >= 1
        ), f"expected num_user_processes >= 1, got instead {self.num_user_processes}"

        # set the path of the configuration file
        self.config_file_path = Path(config_file_path)

        # set the list of monitored log files
        self.log_path_list = None

        if custom_log_paths:
            self.log_path_list = [
                Path(log_path.strip()) for log_path in custom_log_paths.split(",")
            ]
        else:
            if self.num_user_processes == 1:
                self.log_path_list = [Path("/logs/output.log")]
            else:
                self.log_path_list = [
                    Path(f"/logs/output-{idx}.log")
                    for idx in range(self.num_user_processes)
                ]

        # setup the RabbitMQ instance that will be used by all launched sidecars
        messaging_secret_str = os.getenv("MESSAGING_AUTHENTICATION")
        messaging_secret = None
        if messaging_secret_str:
            messaging_secret = json.loads(messaging_secret_str)

        self.rabbitmq_stop_event = asyncio.Event()
        self.msg = RabbitMQBase.from_env_and_args(
            exchange_name=exchange_name,
            queue_name=queue_name,
            routing_key=routing_key,
            messaging_secret=messaging_secret,
            stop_evt=self.rabbitmq_stop_event,
        )

        # calculate targetsteprun_id and launch_id from routing key
        self.targetsteprun_id = "n/a"
        self.launch_id = "n/a"
        if routing_key:
            tokens = routing_key.split(".")
            if len(tokens) == 3:
                self.targetsteprun_id = tokens[1]
                self.launch_id = tokens[2]

        # list of sidecars that will be launched by this orchestrator
        self.sidecars: List[Sidecar] = []

    def _extract_event_configs(
        self: Self, data: Any
    ) -> List[List[EventLogLineParserConfig]]:
        """Extract log parsing event configurations from YAML data.

        Handles two configuration patterns:

        Case 1: Multiple processes with different configs (multi_process_event_configs):
                sidecar_monitor:
                - config:
                    multi_process_event_configs:
                    - event_configs:
                        - event_fields: ...
                    - event_configs:
                        - event_fields: ...
                  name: log_monitor

        Case 2: Single config or multiple processes with same config (event_configs):
                sidecar_monitor:
                - config:
                    event_configs:
                    - event_fields: ...
                  name: log_monitor

        Args:
            data: Parsed YAML configuration dictionary

        Returns:
            List of event config lists, one per user process. Each inner list contains
            EventLogLineParserConfig objects for that process's log parsing rules.

        Raises:
            ValueError: If neither 'sidecar_monitor' nor 'event_monitor' key exists in YAML
        """
        event_configs_list: List[List[EventLogLineParserConfig]] = []

        # Navigate to sidecar_monitor or event monitor
        sidecar_monitor_cfgs = data.get("sidecar_monitor", [])
        event_monitor_cfgs = data.get("event_monitor", [])
        monitor_config = sidecar_monitor_cfgs + event_monitor_cfgs

        if not monitor_config:
            raise ValueError(
                "Missing 'sidecar_monitor' or 'event_monitor' key in YAML."
            )

        # Assume only one monitor config for simplicity
        config = monitor_config[0].get("config", {})

        # Case 1: multi_process_event_configs exists
        if "multi_process_event_configs" in config:
            for mp_config in config["multi_process_event_configs"]:
                event_configs = [
                    EventLogLineParserConfig(**ec)
                    for ec in mp_config.get("event_configs", [])
                ]
                event_configs_list.append(event_configs)
            logger.info(
                "[SidecarOrchestrator %s] Using multiple (%d) log parsing event configs",
                self.targetsteprun_id,
                len(event_configs_list),
            )
        else:
            # Case 2: single event_configs subtree
            event_configs = [
                EventLogLineParserConfig(**ec) for ec in config.get("event_configs", [])
            ]
            # Duplicate the same event_configs for each process
            event_configs_list = [event_configs for _ in range(self.num_user_processes)]
            logger.info(
                "[SidecarOrchestrator %s] Using one parsing event config for %d user processes",
                self.targetsteprun_id,
                len(event_configs_list),
            )

        return event_configs_list

    def _load_config(self: Self) -> Dict[Path, List[EventLogLineParserConfig]]:
        """Load and parse log monitor configuration from YAML file.

        Reads the configuration file, extracts event configs for each user process,
        and creates a mapping from log file paths to their corresponding event configs.

        Returns:
            Dictionary mapping log file Path objects to lists of EventLogLineParserConfig objects.
            Each key is a log file to be monitored, and the value is the list of event parsing
            rules for that log file.

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            AssertionError: If the number of event configs doesn't match the number of log paths
        """
        # Check that the config file exists
        if not self.config_file_path.exists():
            raise FileNotFoundError(f"Config file {self.config_file_path} not found")

        # Load YAML into python dictionary
        with self.config_file_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        event_configs_list: List[List[EventLogLineParserConfig]] = (
            self._extract_event_configs(
                data=cfg,
            )
        )

        assert len(event_configs_list) == len(self.log_path_list), (
            f"length of event_configs_list ({len(event_configs_list)}) does not match "
            f"length of log path list ({len(self.log_path_list)})"
        )

        sidecar_configs: Dict[Path, List[EventLogLineParserConfig]] = {
            logpath: evt_cfgs
            for logpath, evt_cfgs in zip(self.log_path_list, event_configs_list)
        }

        return sidecar_configs

    async def run_sidecars(self: Self):
        """Run all sidecars concurrently until completion.

        This method:
        1. Sets up the RabbitMQ messaging connection
        2. Loads configuration and creates Sidecar instances
        3. Launches all sidecars as concurrent async tasks
        4. Waits for all sidecars to complete
        5. Closes the messaging connection

        Each sidecar runs independently and monitors its assigned log file,
        with its own LogFileMonitor and CmdlineMonitor.
        """
        await self.msg.setup()

        logger.info(
            "[SidecarOrchestrator %s] Reading configuration from file %s",
            self.targetsteprun_id,
            self.config_file_path,
        )

        sidecar_configs = self._load_config()

        logger.info(
            "[SidecarOrchestrator %s] Starting %d sidecars",
            self.targetsteprun_id,
            len(sidecar_configs),
        )

        # Create all sidecars
        for idx, (logfile_path, event_configs) in enumerate(sidecar_configs.items()):
            self.sidecars.append(
                Sidecar(
                    messenger=self.msg,
                    log_path=logfile_path,
                    event_configs=event_configs,
                    idx=idx,
                    targetsteprun_id=self.targetsteprun_id,
                    launch_id=self.launch_id,
                )
            )

        # Run all sidecars as concurrent tasks
        sidecar_tasks = [
            asyncio.create_task(sidecar.run(), name=f"sidecar_{i}")
            for i, sidecar in enumerate(self.sidecars)
        ]

        # Wait for all sidecars to complete
        results = await asyncio.gather(*sidecar_tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "[SidecarOrchestrator] Sidecar %s exited with error: %s", i, result
                )
            else:
                logger.info("[SidecarOrchestrator] Sidecar %s exited cleanly.", i)

        logger.info(
            "[SidecarOrchestrator %s] All sidecars have finished.",
            self.targetsteprun_id,
        )
        await self.msg.close()
        logger.info("[SidecarOrchestrator %s] Shutdown complete", self.targetsteprun_id)


class Sidecar:
    """
    Monitors a single log file and detects when the associated process completes.

    Each Sidecar instance:
    - Creates a LogFileMonitor to tail a log file and publish events when patterns match
    - Creates a CmdlineMonitor to detect when the process writing to the log file exits
    - Terminates when the CmdlineMonitor detects the process has finished
    - Shares a MessagingBase connection with other sidecars for publishing events

    This runs alongside user processes to monitor their log output and lifecycle.
    """

    def __init__(
        self: Self,
        messenger: MessagingBase,
        log_path: Path,
        event_configs: List[EventLogLineParserConfig],
        idx: int,
        targetsteprun_id: str = "n/a",
        launch_id: str = "n/a",
    ) -> None:
        """Initialize a Sidecar instance.

        Args:
            messenger: Shared MessagingBase instance for publishing events
            log_path: Path to the log file to monitor
            event_configs: List of event parsing configurations for log line pattern matching
            idx: Index of this sidecar (for logging and identification)
            targetsteprun_id: Unique ID for the associated build step (default: "n/a")
            launch_id: Unique ID for this specific launch/run (default: "n/a")
        """
        self.stop_event = asyncio.Event()
        self.msg = messenger
        self._event_configs = event_configs
        self.targetsteprun_id = targetsteprun_id
        self.launch_id = launch_id
        self.log_path: Path = log_path
        self.idx = idx

        logger.info(
            "[Sidecar %s/%d]: __init__() completed", self.targetsteprun_id, self.idx
        )

    # -------------- public entry ---------------------
    async def run(self: Self):
        """Run the sidecar monitoring process.

        This method:
        1. Creates a LogFileMonitor to tail the log file and detect event patterns
        2. Creates a CmdlineMonitor to watch for process termination (looks for "tee <log_path>")
        3. Waits for the CmdlineMonitor to set stop_event when the process exits
        4. Ensures both monitors shut down gracefully

        The method blocks until the monitored process completes and all cleanup is done.
        """
        stream_source: LogStreamSource = LocalFileStream(
            path=self.log_path,
            targetsteprun_id=self.targetsteprun_id,
            launch_id=self.launch_id,
        )
        log_monitor: LogFileMonitor = LogFileMonitor(
            step_id=f"{self.targetsteprun_id}/{self.idx}",
            stream_source=stream_source,
            event_configs=self._event_configs,
            messenger=self.msg,
            stop_event=self.stop_event,
        )

        # Start log monitor task
        log_task = asyncio.create_task(
            log_monitor.monitor(), name=f"log_monitor_{self.idx}"
        )

        termination_monitor = CmdlineMonitor(
            cmd_substring=f"tee {self.log_path}",
            log_path=str(self.log_path),
            stop_event=self.stop_event,
            targetsteprun_id=self.targetsteprun_id,
        )
        termination_monitor_task = asyncio.create_task(termination_monitor.monitor())

        # block until stop_event is triggered (set by termination monitor)
        await self.stop_event.wait()

        # wait for both monitors to exit gracefully
        results = await asyncio.gather(
            log_task, termination_monitor_task, return_exceptions=True
        )

        for result in results:
            if isinstance(result, Exception):
                logger.error(
                    "[Sidecar %s/%d] Monitor exited with error: %s",
                    self.targetsteprun_id,
                    self.idx,
                    result,
                )
            else:
                logger.info(
                    "[Sidecar %s/%d] Monitor exited cleanly.",
                    self.targetsteprun_id,
                    self.idx,
                )
        logger.info(
            "[Sidecar %s/%d] All monitors have finished.",
            self.targetsteprun_id,
            self.idx,
        )


async def main():
    """Entrypoint  of the sidecar container CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--routing-key")
    parser.add_argument("--log")
    parser.add_argument("--config-file", default="monitor_config.yaml")
    parser.add_argument("--polling-interval", type=int, default=10)
    parser.add_argument("--termination-delay", type=int, default=30)
    args = parser.parse_args()

    so = SidecarOrchestrator(
        exchange_name=args.exchange,
        queue_name=args.queue,
        routing_key=args.routing_key,
        config_file_path=args.config_file,
        custom_log_paths=args.log,
    )

    # Setup signal handler for graceful shutdown
    def signal_handler():
        logger.info("Received SIGINT, initiating shutdown...")
        so.rabbitmq_stop_event.set()

    asyncio.get_running_loop().add_signal_handler(signal.SIGINT, signal_handler)

    await so.run_sidecars()


if __name__ == "__main__":
    asyncio.run(main())
