# Copyright IBM Corp. 2024-2026
# SPDX-License-Identifier: Apache-2.0
"""Artifact-packaging helpers for a completed local (in-process) tuning run.

Ports the 2025 ``api/utils.py`` and ``api/services/file/reads.py`` packaging
helpers as pure standard-library functions: none of them touch a database, a
logger, or ``print`` — they only read the ``config_data``/metrics passed in and
write files to the caller-supplied directory. The ``local`` runner uses them to
turn a best-trial checkpoint into a self-contained, runnable artifact bundle
(``inference.py`` + ``run_model.sh`` + ``install.sh`` + ``README.md``) and to
zip a folder deterministically.
"""

from __future__ import annotations

import math
import zipfile
from pathlib import Path
from typing import Any

_RUN_MODEL_SH = """#!/bin/bash

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

_INSTALL_SH = """#!/bin/bash

# Run installation script
pip install torch transformers peft accelerate alora
"""

_README = """# Inference Script Runner

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
The bash script takes your input text and passes it to the `inference.py` Python
script, which processes the prompt and generates the output.

## Troubleshooting
- Ensure `inference.py` exists in the same directory as the bash script
- Make sure Python is properly installed and accessible from your command line
- Check that the bash script has execution permissions
"""


def _metric_or_none(value: float | None) -> float | None:
    """Return ``value`` unless it is missing or ``NaN``, in which case ``None``.

    The ``is None`` check must short-circuit before ``math.isnan`` so a missing
    metric never reaches ``math.isnan`` (which would raise on ``None``).
    """
    if value is None or math.isnan(value):
        return None
    return value


def parse_result(data: dict[str, Any]) -> dict[str, float | None]:
    """Map a Ray-trial result blob to the trio persisted for a trial.

    Only the ``"loss"`` metric is understood today (mirroring 2025). For it,
    ``loss``/``train_loss``/``time_total_s`` are projected onto
    ``loss``/``train_loss``/``total_time``, with any missing or ``NaN`` value
    collapsed to ``None`` so the persistence layer stores a clean absence rather
    than a ``NaN`` float. Any other metric yields an empty mapping.

    Args:
        data: A trial result dict; expected keys include ``metric`` and the
            metric values (``loss``, ``train_loss``, ``time_total_s``).

    Returns:
        ``{"loss", "train_loss", "total_time"}`` for the loss metric, else ``{}``.
    """
    if data.get("metric") != "loss":
        return {}
    return {
        "loss": _metric_or_none(data.get("loss")),
        "train_loss": _metric_or_none(data.get("train_loss")),
        "total_time": _metric_or_none(data.get("time_total_s")),
    }


def _inference_script(model_id: str) -> str:
    """Return the PEFT/LoRA ``inference.py`` source for ``model_id``."""
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


def _alora_inference_script(model_id: str, base_model: str) -> str:
    """Return the activated-LoRA ``inference.py`` source for ``model_id``."""
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
outputs = model_alora.generate(
    inputs["input_ids"].to(device),
    attention_mask=inputs["attention_mask"].to(device),
    max_new_tokens=200,
    alora_offsets=alora_offsets,
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("generated_text", generated_text)
"""


def write_inference_script(
    base_model: str,
    model_id: str,
    output_path: Path,
    *,
    for_alora: bool,
) -> None:
    """Write ``inference.py`` for running the tuned adapter into ``output_path``.

    Args:
        base_model: The base model the adapter was trained on (used only by the
            activated-LoRA branch, which loads the base explicitly).
        model_id: The identifier/path of the PEFT adapter to load.
        output_path: Directory the ``inference.py`` file is written into.
        for_alora: When ``True`` emit the activated-LoRA script, else the plain
            PEFT/LoRA one.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    code = (
        _alora_inference_script(model_id=model_id, base_model=base_model)
        if for_alora
        else _inference_script(model_id=model_id)
    )
    (output_path / "inference.py").write_text(code)


def generate_bash_script(output_path: Path) -> None:
    """Write an executable ``run_model.sh`` runner into ``output_path``.

    The script forwards its first argument to ``python inference.py`` and is
    marked executable (``0o755``) so the artifact bundle is runnable as-is.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    script = output_path / "run_model.sh"
    script.write_text(_RUN_MODEL_SH)
    script.chmod(0o755)


def generate_install_bash_script(output_path: Path) -> None:
    """Write an executable ``install.sh`` dependency installer into ``output_path``.

    Marked executable (``0o755``) so the bundle installs its own runtime deps.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    script = output_path / "install.sh"
    script.write_text(_INSTALL_SH)
    script.chmod(0o755)


def generate_readme(output_path: Path) -> None:
    """Write the artifact-bundle ``README.md`` (usage instructions) into ``output_path``."""
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "README.md").write_text(_README)


def zip_folder(folder: Path, output_zip: str, output_dir: Path) -> Path:
    """Zip ``folder`` into ``output_dir/output_zip`` and return the archive path.

    Names are stored relative to ``folder``'s parent, so the folder name is the
    top-level entry inside the archive (matching the 2025 behaviour). The caller
    supplies both the archive name and its directory — no timestamped default is
    generated — so the result is deterministic.

    Args:
        folder: The directory whose contents are archived. Must exist.
        output_zip: Archive filename; a ``.zip`` suffix is appended if absent.
        output_dir: Directory the archive is written into (created if needed).

    Returns:
        The path to the created ``.zip`` archive.

    Raises:
        FileNotFoundError: If ``folder`` does not exist.
    """
    if not folder.exists():
        raise FileNotFoundError(f"The folder '{folder}' does not exist")

    if not output_zip.endswith(".zip"):
        output_zip += ".zip"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / output_zip

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                zipf.write(path, path.relative_to(folder.parent))

    return archive
