#!/usr/bin/env python3
"""
AutoTune Wrapper for Granite.build (granite_build_wrapper.py)

This wrapper bridges Granite.build's YAML configuration with fm-tune's AutoTune functionality.
It accepts an AutoTune configuration as a JSON string, generates the required YAML config file,
and executes fm-tune's main.py with the appropriate arguments.

IMPORTANT: This file should be located in the fm-tune repository at:
    fm-tune/tools/granite_build_wrapper.py

This reference copy in the assets repo is for documentation purposes only.
The production version lives in the fm-tune repository.

Usage:
    python autotune_wrapper.py \
        --autotune_config '{"tune_config": {...}, "training_config": {...}, "tuners_config": {...}}' \
        --train_file <path> \
        --validation_file <path> \
        --model_name_or_path <path> \
        --tuning_type <lora|sft|etc> \
        --output_dir <path> \
        --run_name <name> \
        --output_model_name <name> \
        [--save_history] \
        [--do_checkpoint] \
        [--no_autotune]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FM-Tune AutoTune wrapper for Granite.build",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # AutoTune config (embedded from build.yaml)
    parser.add_argument(
        "--autotune_config",
        type=str,
        required=True,
        help="JSON string containing AutoTune configuration (tune_config, training_config, tuners_config)",
    )

    # FM-Tune required arguments
    parser.add_argument(
        "--train_file", type=str, required=True, help="Path to training data file"
    )
    parser.add_argument(
        "--validation_file",
        type=str,
        required=True,
        help="Path to validation data file",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to model or HuggingFace model name",
    )
    parser.add_argument(
        "--tuning_type", type=str, required=True, help="Tuning type (e.g., lora, sft)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory for results"
    )
    parser.add_argument(
        "--run_name", type=str, required=True, help="Name for this AutoTune run"
    )
    parser.add_argument(
        "--output_model_name", type=str, required=True, help="Name for the output model"
    )

    # FM-Tune optional flags
    parser.add_argument(
        "--save_history", action="store_true", help="Save trial history"
    )
    parser.add_argument(
        "--do_checkpoint", action="store_true", help="Save model checkpoints"
    )
    parser.add_argument(
        "--no_autotune",
        action="store_true",
        help="Disable AutoTune (just run single training)",
    )

    return parser.parse_args()


def validate_autotune_config(config_dict):
    """
    Validate the structure of the AutoTune config.

    Args:
        config_dict: Parsed config dictionary

    Returns:
        bool: True if valid, raises exception otherwise
    """
    required_sections = ["tune_config", "training_config", "tuners_config"]
    missing_sections = [s for s in required_sections if s not in config_dict]

    if missing_sections:
        raise ValueError(
            f"AutoTune config is missing required sections: {', '.join(missing_sections)}\n"
            f"Expected sections: {', '.join(required_sections)}"
        )

    # Validate tune_config
    tune_config = config_dict["tune_config"]
    if "search_alg" not in tune_config:
        raise ValueError("tune_config must contain 'search_alg' field")
    if "num_samples" not in tune_config:
        raise ValueError("tune_config must contain 'num_samples' field")

    # Validate training_config
    training_config = config_dict["training_config"]
    if "num_train_epochs" not in training_config and "max_steps" not in training_config:
        raise ValueError(
            "training_config must contain either 'num_train_epochs' or 'max_steps'"
        )

    # Validate tuners_config
    tuners_config = config_dict["tuners_config"]
    if not tuners_config or len(tuners_config) == 0:
        raise ValueError("tuners_config must contain at least one tuner configuration")

    logger.info("AutoTune config validation passed")
    return True


def write_config_file(config_dict, config_path):
    """
    Write the AutoTune config to a YAML file.

    Args:
        config_dict: Configuration dictionary
        config_path: Path where to write the YAML file
    """
    try:
        with open(config_path, "w") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Written AutoTune config to: {config_path}")

        # Log config summary for debugging
        logger.info("Config summary:")
        logger.info(
            f"  - Search algorithm: {config_dict['tune_config'].get('search_alg')}"
        )
        logger.info(
            f"  - Number of trials: {config_dict['tune_config'].get('num_samples')}"
        )
        logger.info(f"  - Tuners: {', '.join(config_dict['tuners_config'].keys())}")

    except Exception as e:
        raise RuntimeError(f"Failed to write config file: {str(e)}")


def build_fm_tune_command(args, config_file_path):
    """
    Build the command to execute fm-tune's main.py.

    Args:
        args: Parsed command-line arguments
        config_file_path: Path to the generated config YAML file

    Returns:
        list: Command as list of strings for subprocess
    """
    cmd = [
        "python",
        "main.py",
        "--config_file",
        config_file_path,
        "--train_file",
        args.train_file,
        "--validation_file",
        args.validation_file,
        "--model_name_or_path",
        args.model_name_or_path,
        "--tuning_type",
        args.tuning_type,
        "--output_dir",
        args.output_dir,
        "--run_name",
        args.run_name,
        "--output_model_name",
        args.output_model_name,
    ]

    # Add optional flags
    if args.save_history:
        cmd.append("--save_history")
    if args.do_checkpoint:
        cmd.append("--do_checkpoint")
    if args.no_autotune:
        cmd.append("--no_autotune")

    return cmd


def execute_fm_tune(cmd):
    """
    Execute the fm-tune command.

    Args:
        cmd: Command as list of strings

    Returns:
        int: Return code from subprocess
    """
    logger.info("=" * 80)
    logger.info("Executing FM-Tune AutoTune")
    logger.info("=" * 80)
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("=" * 80)

    try:
        # Execute with real-time output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # Stream output in real-time
        for line in process.stdout:
            print(line, end="")

        # Wait for completion
        return_code = process.wait()

        if return_code == 0:
            logger.info("=" * 80)
            logger.info("FM-Tune AutoTune completed successfully")
            logger.info("=" * 80)
        else:
            logger.error("=" * 80)
            logger.error(f"FM-Tune AutoTune failed with return code: {return_code}")
            logger.error("=" * 80)

        return return_code

    except Exception as e:
        logger.error(f"Error executing FM-Tune: {str(e)}")
        raise


def main():
    """Main execution function."""
    logger.info("Starting FM-Tune AutoTune Wrapper for Granite.build")

    # Parse arguments
    args = parse_arguments()

    # Parse AutoTune config from JSON string
    try:
        config_dict = json.loads(args.autotune_config)
        logger.info("Successfully parsed AutoTune config from JSON")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse autotune_config JSON: {str(e)}")
        logger.error(f"Received config string: {args.autotune_config[:200]}...")
        sys.exit(1)

    # Validate config structure
    try:
        validate_autotune_config(config_dict)
    except ValueError as e:
        logger.error(f"Config validation failed: {str(e)}")
        sys.exit(1)

    # Create temporary config file
    config_file = None
    try:
        # Create a named temporary file that won't be auto-deleted
        config_fd, config_file = tempfile.mkstemp(
            suffix=".yaml", prefix="autotune_config_"
        )
        os.close(config_fd)  # Close the file descriptor

        logger.info(f"Created temporary config file: {config_file}")

        # Write config to file
        write_config_file(config_dict, config_file)

        # Build fm-tune command
        cmd = build_fm_tune_command(args, config_file)

        # Execute fm-tune
        return_code = execute_fm_tune(cmd)

        # Exit with same code as fm-tune
        sys.exit(return_code)

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

    finally:
        # Cleanup: remove temporary config file
        if config_file and os.path.exists(config_file):
            try:
                os.unlink(config_file)
                logger.info(f"Cleaned up temporary config file: {config_file}")
            except Exception as e:
                logger.warning(
                    f"Failed to clean up temporary file {config_file}: {str(e)}"
                )


if __name__ == "__main__":
    main()
