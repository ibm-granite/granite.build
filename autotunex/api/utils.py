# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0

import os
import re
import json
from typing import List, Dict, Any
from typing import Optional, Tuple
import uuid
import shutil
import subprocess
import ast
from datetime import datetime, timezone
import pytz
import math
import asyncio
import logging

logger = logging.getLogger(__name__)


def transform_db_results(db_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Transform database query results into a structured hyperparameters format.

    Args:
        db_results: List of dictionaries containing database query results

    Returns:
        Dictionary with restructured hyperparameters
    """
    # Initialize the structure
    result = {"hyperparameters": {"parameters": {}}}

    # Group items by config_type
    for item in db_results:
        config_type = item["config_type"]

        # Convert string representations of lists/values to Python objects
        values = json.loads(item["values"])

        # Try to convert default_value to int if possible
        try:
            default_value = int(item["default_value"])
        except ValueError:
            default_value = item["default_value"]

        parameter_info = {
            "id": item["id"],
            "parameter_name": item["parameter_name"],
            "strategy": item["strategy"],
            "values": values,
            "default_value": default_value,
        }

        # Add to the result structure
        if config_type not in result["hyperparameters"]["parameters"]:
            result["hyperparameters"]["parameters"][config_type] = []

        result["hyperparameters"]["parameters"][config_type].append(parameter_info)

    return result


def get_alora_inference_script(model_id: str, base_model: str):
    return f"""
import sys, torch
from alora.peft_model_alora import aLoRAPeftModelForCausalLM
from alora.config import aLoraConfig
from alora.tokenize_alora import tokenize_alora
from transformers import AutoModelForCausalLM, AutoTokenizer

# Get input from command line argument
input_text = sys.argv[1]

BASE_MODEL="{base_model}"
ALORA_NAME="{model_id}"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map = 'auto')
model_alora = aLoRAPeftModelForCausalLM.from_pretrained(model_base, ALORA_NAME)

INVOCATION_SEQUENCE = model_alora.peft_config["default"].invocation_string
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

inputs, alora_offsets = tokenize_alora(tokenizer, input_text + "\\n", INVOCATION_SEQUENCE)
outputs = model_alora.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), max_new_tokens=200, alora_offsets=alora_offsets)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("generated_text", generated_text)
"""


def get_inference_script(model_id: str):
    return f"""
# pip install peft transformers accelerate
import sys
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import infer_auto_device_map, init_empty_weights

# Get input from command line argument
input_text = sys.argv[1]

peft_model_id = "{model_id}"  # Path to the PEFT model

# Load the configuration
config = PeftConfig.from_pretrained(peft_model_id)

# Load the base model with appropriate settings for memory efficiency
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype='auto',
    device_map='auto',
    offload_folder="offload",
    offload_state_dict=True
)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id)

# Example inference
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
"""


def write_inference_script(
    base_model: str, model_id: str, output_path, for_alora: bool = False
):
    """
    Generate a Python script for running inference with a PEFT-adapted model.

    Args:
        model_id (str): The identifier/path of the PEFT model
        output_path: Path where script need to be generated

    Returns:
        str: The complete inference script as a string
    """
    inference_code = ""
    if for_alora:
        inference_code = get_alora_inference_script(
            model_id=model_id, base_model=base_model
        )
    else:
        inference_code = get_inference_script(model_id=model_id)

    with open(os.path.join(output_path, "inference.py"), "w") as file:
        inference_code = file.write(inference_code)

    print("Inference script generated and saved to 'inference.py'")


def generate_bash_script(output_path):
    """
    Generate a bash script that runs a Python inference script with an input argument
    and save it to the specified location.

    Args:
        output_path (str): Path where the bash script should be saved
    """
    # Define the bash script content
    bash_script = """#!/bin/bash

# Check if an input argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No input text provided."
    echo "Usage: $0 \\"Your text prompt here\\""
    exit 1
fi

# Store the input text
INPUT_TEXT="$1"
# Run the modified inference script with the input text
python inference.py "$INPUT_TEXT"
"""

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Write the bash script to the specified location
    with open(os.path.join(output_path, "run_model.sh"), "w") as file:
        file.write(bash_script)

    # Make the script executable
    os.chmod(os.path.join(output_path, "run_model.sh"), 0o755)

    print(f"Bash script successfully generated and saved to: {output_path}")
    print("The script has been made executable.")


def generate_install_bash_script(output_path):
    """
    Generate a bash script that runs a Python inference script with an input argument
    and save it to the specified location.

    Args:
        output_path (str): Path where the bash script should be saved
    """
    # Define the bash script content
    bash_script = """#!/bin/bash

# Run installation script
pip install torch transformers peft accelerate alora
"""

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Write the bash script to the specified location
    with open(os.path.join(output_path, "install.sh"), "w") as file:
        file.write(bash_script)

    # Make the script executable
    os.chmod(os.path.join(output_path, "install.sh"), 0o755)

    print(f"Install script successfully generated and saved to: {output_path}")
    print("The script has been made executable.")


def generate_readme(output_path):
    """
    Generate a README file with instructions for running the inference script

    Args:
        output_path (str): Path where the README file should be saved
    """
    readme_content = """# Inference Script Runner

## Overview
This repository contains a bash script that helps run the inference script with text prompts.

## Prerequisites
- Python 3.10 or higher
- The `inference.py` script in the same directory

## Usage

### Installation of dependencies
1. Run the install script:
   ```
   ./install.sh
   ```

### Running the script
1. Make sure the bash script is executable:
   ```
   chmod +x run_model.sh
   ```

2. Run the script with your text prompt:
   ```
   ./run_model.sh "Your text prompt here"
   ```

### Examples
```
./run_model.sh "Generate a story about dragons"
```

### Error Handling
If you run the script without providing a text prompt, you'll see an error message:
```
Error: No input text provided.
Usage: ./run_model.sh "Your text prompt here"
```

## How It Works
The bash script takes your input text and passes it to the `inference.py` Python script, which processes the prompt and generates the output.

## Troubleshooting
- Ensure `inference.py` exists in the same directory as the bash script
- Make sure Python is properly installed and accessible from your command line
- Check that the bash script has execution permissions
"""

    # Create directories if they don't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Write the bash script to the specified location
    with open(os.path.join(output_path, "README.md"), "w") as file:
        file.write(readme_content)

    print(f"Readme successfully generated and saved to: {output_path}")


def extract_parameter_length(model_string):
    """
    Extracts the parameter length (e.g., '8b', '125m') from a model identifier string.

    It looks for patterns like digits followed by 'b' or 'm'. If multiple are
    found, it assumes the last one is the main parameter size.

    Args:
      model_string: The input string (e.g., 'ibm-granite/granite-3.2-8b-instruct').

    Returns:
      The extracted parameter length string (like '8b', '125m'),
      or None if no such pattern is found.
    """
    # Regex pattern: matches one or more digits (\d+) followed by 'b' or 'm' ([bm])
    # Using \b ensures we match whole words/segments to avoid partial matches within longer strings if needed,
    # although for simple cases \d+[bm] is often sufficient. Let's stick to the simpler one first.
    pattern = r"\d+[bm]"

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, model_string)

    if matches:
        # If one or more matches are found, return the last one
        return matches[-1]
    else:
        # If no match is found, return None
        return None


def is_flash_attn_2_available():
    import torch
    import importlib.metadata
    import importlib.util
    from packaging import version
    from transformers.utils import is_torch_available

    if not is_torch_available():
        return False

    if importlib.util.find_spec("flash_attn") is None:
        return False

    # Let's add an extra check to see if CUDA is available
    if not torch.cuda.is_available():
        return False

    try:
        flash_attn_version = version.parse(importlib.metadata.version("flash_attn"))
        if torch.version.cuda:
            return flash_attn_version >= version.parse("2.1.0")
        elif torch.version.hip:
            return flash_attn_version >= version.parse("2.0.4")
        else:
            return False
    except importlib.metadata.PackageNotFoundError:
        return False


# The Granite Build CLI ships under two interchangeable names: `gb` and its
# alias `llmb`. Either one satisfies the runtime requirement.
GB_CLI_NAMES = ("gb", "llmb")


# Cache for get_gb_binary(). _GB_BINARY_RESOLVED tracks whether the PATH lookup
# has run, so an absent binary (cached as None) isn't looked up on every call.
_GB_BINARY_RESOLVED = False
_GB_BINARY_CACHE: Optional[str] = None


def get_gb_binary() -> Optional[str]:
    """Return the resolved path to the Granite Build CLI, or None if absent.

    Checks each known CLI name (`gb`, then the `llmb` alias) on PATH. The
    result is cached because is_gb_enabled() is called on hot paths
    (per-request registry/backend selection); PATH does not change at runtime.
    """
    global _GB_BINARY_RESOLVED, _GB_BINARY_CACHE
    if not _GB_BINARY_RESOLVED:
        _GB_BINARY_CACHE = next(
            (path for name in GB_CLI_NAMES if (path := shutil.which(name))), None
        )
        _GB_BINARY_RESOLVED = True
    return _GB_BINARY_CACHE


def is_gb_enabled():
    """
    Whether Granite Build integration is usable.

    Requires both a configured GB_TOKEN and the GB CLI (`gb` or its `llmb`
    alias) on PATH. Gating on the binary too means every GB code path degrades
    gracefully when the optional `granite.build` dependency is not installed,
    instead of failing later with a FileNotFoundError from subprocess.

    Returns:
        bool: True if GB_TOKEN is set AND the gb/llmb CLI is available.
    """
    return bool(os.getenv("GB_TOKEN")) and get_gb_binary() is not None


def get_gb_token():
    """
    Returns the GB_TOKEN from environment variables.

    Returns:
        string: GB_TOKEN.
    """
    return os.getenv("GB_TOKEN")


def extract_uuid_uri(log_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts UUID and URI from the provided log text.

    Args:
        log_text (str): The log text to process.

    Returns:
        Tuple containing (uuid, uri) or (None, None) if not found.
    """
    # UUID regex pattern
    uuid_pattern = r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    # URI regex (starts with 'lh://')
    uri_pattern = r"hf://[^\s]+"

    uuid_match = re.search(uuid_pattern, log_text)
    uri_match = re.search(uri_pattern, log_text)

    uuid = uuid_match.group(0) if uuid_match else None
    uri = uri_match.group(0) if uri_match else None

    return uuid, uri


def extract_github_url(log_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts URL and UUID from the provided log text.

    Args:
        log_text (str): The log text to process.

    Returns:
        Tuple containing (url, uuid) or (None, None) if not found.
    """
    github_host = os.getenv("GITHUB_HOST", "github.com").strip().rstrip("/")
    url_pattern = rf"https://{re.escape(github_host)}/granite-dot-build[^\s]+"
    url_match = re.search(url_pattern, log_text)
    url = url_match.group(0) if url_match else None

    # UUID pattern - matches standard UUID format (8-4-4-4-12 hex characters)
    uuid_pattern = r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
    uuid_match = re.search(uuid_pattern, log_text)
    uuid = uuid_match.group(0) if uuid_match else None

    return url, uuid


def is_valid_uuid(s: str) -> bool:
    try:
        # Try to create a UUID object from the string
        uuid.UUID(s)
        return True
    except ValueError:
        return False


def execute_command(command: List[str]):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout.strip())
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(e)
        return {"status": "error", "error": f"Command failed: {e}", "stderr": e.stderr}
    except Exception as e:
        return {"status": "error", "error": f"Unexpected error: {str(e)}"}


def extract_artifact_identifier(url):
    """
    Extract identifier and revision using regex pattern matching.

    Supports two artifact_uri schemes:
    - DMF/Granite Build: lh://.../model_shared/<model_id>/<revision>
    - Hugging Face: hf://huggingface.co/models/<org>/<name>
      (no separate revision segment, so revision is None)
    """
    model_shared_pattern = r"model_shared/([^/]+)/([^/]+)"
    match = re.search(model_shared_pattern, url)
    if match:
        return (match.group(1), match.group(2))

    hf_pattern = r"hf://huggingface\.co/models/([^/]+)/([^/]+)"
    match = re.search(hf_pattern, url)
    if match:
        return (f"{match.group(1)}/{match.group(2)}", None)

    return (None, None)


def build_dmf_url(model_id, revision) -> Optional[str]:
    """Build a DMF model-detail UI link, or None if DMF_UI_URL is not configured.

    The DMF UI is IBM-internal infrastructure with no public default, so the base
    URL must be provided via the DMF_UI_URL env var (e.g.
    https://ui.dmf.vpc-int.res.ibm.com). When unset, no link is produced.
    """
    base = os.getenv("DMF_UI_URL", "").strip().rstrip("/")
    if not base or model_id is None or revision is None:
        return None
    return f"{base}/v2/models/detail/granite_dot_build.public/model_shared/{model_id}/{revision}"


def parse_gb_message(message):
    """Parse GB_PR_MESSAGE and fix the URL protocol"""

    # Check if it's a GB_PR_MESSAGE
    if not message.startswith("##"):
        return None

    # Extract the dictionary part
    dict_match = re.search(r"\{.*\}", message)
    if not dict_match:
        return None

    try:
        dict_str = dict_match.group()
        model_info = ast.literal_eval(dict_str)

        # Parse the result
        endpoint = list(model_info.keys())[0]
        model_name = list(model_info.values())[0]

        # Fix the endpoint URL - replace 'fits//' with 'https://'
        fixed_endpoint = endpoint.replace("rits//", "https://")

        return {
            "message_type": "GB_PR_MESSAGE",
            "endpoint": endpoint,
            "model_name": model_name,
            "full_url": fixed_endpoint,
            "original_endpoint": endpoint,
            "raw_dict": model_info,
        }
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing dictionary: {e}")
        return None


def utc_now_string():
    return datetime.now(timezone.utc).isoformat()


def is_valid_timestamp(timestamp_str):
    try:
        datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return True
    except (ValueError, AttributeError):
        return False


def time_elapsed(timestamp_str: str):
    if not is_valid_timestamp(timestamp_str):
        return None

    past_timestamp = datetime.fromisoformat(timestamp_str)

    current_utc = datetime.now(timezone.utc)
    difference = current_utc - past_timestamp

    return round(difference.total_seconds() / 60)  # Returning difference in minutes


def str_to_bool(value):
    """Convert string to boolean"""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes", "on"):
        return True
    elif value.lower() in ("false", "0", "no", "off"):
        return False
    else:
        return False


# def get_utc_timestamp(date:str):
#     if date:
#         return pytz.UTC.localize(date).isoformat()


def get_utc_timestamp(date_input):
    if not date_input:
        return None
    # If it's a string, parse it to datetime first
    if isinstance(date_input, str):
        # Parse the string to datetime (assuming format: 'YYYY-MM-DD HH:MM:SS')
        date_obj = datetime.strptime(date_input, "%Y-%m-%d %H:%M:%S")
    else:
        date_obj = date_input

    # Only localize if the datetime is naive (no timezone info)
    if date_obj.tzinfo is None:
        return pytz.UTC.localize(date_obj).isoformat()
    else:
        # If it already has timezone info, convert to UTC
        return date_obj.astimezone(pytz.UTC).isoformat()


def extract_chars(text):
    """
    Extracts up to 7 characters from a string, ensuring the output
    doesn't end with a special character.

    Args:
        text (str): Input string

    Returns:
        str: String with up to 7 characters, not ending with special character
    """
    if not text:
        return ""

    # Take first 7 characters
    result = text[:7]

    # Remove trailing special characters (keep only alphanumeric)
    # Strip from right while the last character is not alphanumeric
    while result and not result[-1].isalnum():
        result = result[:-1]

    return result


def parse_result(data):
    result = {}
    if data.get("metric") == "loss":
        result = {
            "loss": (
                None
                if data.get("loss") is None or math.isnan(data.get("loss"))
                else data.get("loss")
            ),
            "train_loss": (
                None
                if data.get("train_loss") is None or math.isnan(data.get("train_loss"))
                else data.get("train_loss")
            ),
            "total_time": (
                None
                if data.get("time_total_s") is None
                or math.isnan(data.get("time_total_s"))
                else data.get("time_total_s")
            ),
        }
        return result
    else:
        return json.dumps({"error": "Unsupported metric"}, indent=4)


def get_granite_model_params(model_name: str) -> float:
    if model_name is None:
        return

    if not model_name.startswith("ibm-granite/granite-4.0"):
        return

    granite_hybrid_models = {
        "ibm-granite/granite-4.0-micro": 3.0,
        "ibm-granite/granite-4.0-h-micro": 3.0,
        "ibm-granite/granite-4.0-h-tiny": 7.0,
        "ibm-granite/granite-4.0-tiny": 7.0,
        "ibm-granite/granite-4.0-h-small": 32.0,
    }

    match = next((k for k in granite_hybrid_models if k in model_name), None)
    if match:
        return granite_hybrid_models[match]


async def run_command(command: str):
    logger.debug("start run_command function")
    logger.debug(f"executing command: {command}")
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout, stderr = await process.communicate()
    logger.debug("end of run_command completed")
    logger.debug(f"stdout: {stdout.decode()}")
    logger.debug(f"stderr: {stderr.decode()}")
    return {
        "stdout": stdout.decode(),
        "stderr": stderr.decode(),
        "code": process.returncode,
    }
