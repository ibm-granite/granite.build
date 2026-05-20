import argparse
import asyncio
from typing import List

import yaml

from gbserver.environment.environment import Environment, EventLogLineParserConfig
from gbserver.types.buildevent import EntityRunMetadata
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


async def log_parse_driver(log_config_file: str, log_line_file: str):
    """Test driver for pod logs parsing.  Add the log monitor configuration in a yaml file
    (sample provided in `log_config.yaml' and a log message line text file
    (sample provided in `log_line.txt`). The driver will run the message through the parser
    with the given configuration and will either collect a `BuildEvent`, meaning that the log
    parsing was successful, or will timeout waiting for `BuildEvent`s, meaning that log parsing
    failed.

    Args:
        log_config_file: name of the file with log configuration (sample in log_config.yaml)
        log_line_file: name of the file that contains a test log line (sample in log_line.txt)
    """
    log_queue = asyncio.Queue()
    with open(log_config_file) as stream:
        try:
            log_monitor_configs = yaml.safe_load(stream)
        except yaml.YAMLError as ex:
            logger.error(ex)
    try:
        with open(log_line_file, "r", encoding="utf-8") as file:
            test_log_line = (
                file.readline().strip()
            )  # Read the first line and remove trailing whitespace
    except FileNotFoundError:
        logger.error(f"Error: File '{log_line_file}' not found.")
        test_log_line = ""
    except Exception as e:  # Catch other potential errors
        logger.error(f"An error occurred: {e}")
        test_log_line = ""

    logger.info(f"test_log_line = {test_log_line}")
    event_configs: List[EventLogLineParserConfig] = []
    for log_monitor_config in log_monitor_configs.get("event_configs", []):
        event_configs.append(EventLogLineParserConfig(**log_monitor_config))

    await Environment.get_events_from_log_line(
        log_line=test_log_line,
        event_configs=event_configs,
        event_q=log_queue,
        entityrun_metadata=EntityRunMetadata(),
    )

    try:
        build_event = await asyncio.wait_for(log_queue.get(), timeout=10)
        logger.info(f"build_event = {build_event}")
    except asyncio.TimeoutError:
        logger.error(f"Could not get a build event for log line {test_log_line}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test regex for log monitoring")
    parser.add_argument(
        "-c",
        "--log_config_file",
        default="log_config.yaml",
        help="Location of the configuration file for monitoring",
    )
    parser.add_argument(
        "-l",
        "--log_line_file",
        default="log_line.txt",
        help="Location of the file that contains a test log line",
    )
    args = parser.parse_args()
    asyncio.run(log_parse_driver(args.log_config_file, args.log_line_file))
