#!/usr/bin/env python3
"""Validate model-task compatibility before running evaluation."""

import re
import sys

# Compatibility matrix
MODEL_TASK_COMPATIBILITY = {
    "OLMo-1.x": {
        "compatible_regimes": ["OLMES-v0.1", "OLMo-v1"],
        "incompatible_regimes": ["TÜLU 3", "OLMo 2"],
        "revision_pattern": r"step\d+-tokens\d+[BM]",
    },
    "OLMo-2.x": {
        "compatible_regimes": ["OLMES-v0.1", "OLMo 2"],
        "incompatible_regimes": [],
        "revision_pattern": r".*",
    },
    "TÜLU-3": {
        "compatible_regimes": ["TÜLU 3", "tulu_3_dev", "tulu_3_unseen"],
        "incompatible_regimes": ["OLMo-v1"],
    },
}

TASK_REGIME_REQUIREMENTS = {
    "core_9mcqa::olmes": {
        "required_regime": "OLMES-v0.1",
        "compatible_model_families": ["OLMo-1.x", "OLMo-2.x"],
    },
    "main_suite::olmo1": {
        "required_regime": "OLMo-v1",
        "compatible_model_families": ["OLMo-1.x"],
    },
    "mmlu::olmes": {
        "required_regime": "OLMES-v0.1",
        "compatible_model_families": ["OLMo-1.x", "OLMo-2.x"],
    },
    "tulu_3_dev": {
        "required_regime": "TÜLU 3",
        "compatible_model_families": ["TÜLU-3"],
    },
    "olmo_2_generative::olmes": {
        "required_regime": "OLMo 2",
        "compatible_model_families": ["OLMo-2.x"],
    },
}


def detect_model_family(model, revision):
    """Detect model family from model name and revision."""
    model_lower = model.lower()

    # OLMo 2.x detection
    if "olmo-2" in model_lower or "olmo2" in model_lower:
        return "OLMo-2.x"

    # OLMo 1.x detection
    if "olmo" in model_lower:
        if revision and re.match(r"step\d+-tokens\d+[BM]", revision):
            return "OLMo-1.x"
        # Default to 2.x if no clear 1.x revision pattern
        return "OLMo-2.x"

    # TÜLU detection
    if "tulu" in model_lower or "tülu" in model_lower:
        return "TÜLU-3"

    return None


def get_task_regime(task):
    """Extract regime from task name."""
    # Direct lookup
    if task in TASK_REGIME_REQUIREMENTS:
        return TASK_REGIME_REQUIREMENTS[task]["required_regime"]

    # Infer from task name
    if "::olmes" in task:
        return "OLMES-v0.1"
    if "::olmo1" in task:
        return "OLMo-v1"
    if "tulu_3" in task:
        return "TÜLU 3"
    if "olmo_2" in task:
        return "OLMo 2"

    return None


def validate_compatibility(model, model_revision, tasks, evaluation_regime):
    """
    Validate that model version is compatible with task suite.

    Returns:
        (bool, str): (is_valid, error_message)
    """
    # Detect model family
    model_family = detect_model_family(model, model_revision)

    if not model_family:
        return False, f"Could not determine model family for: {model}"

    print(f"Detected model family: {model_family}")

    # Check each task
    for task in tasks:
        task_regime = get_task_regime(task)

        if task in TASK_REGIME_REQUIREMENTS:
            req = TASK_REGIME_REQUIREMENTS[task]

            # Check regime compatibility
            if evaluation_regime and evaluation_regime != req["required_regime"]:
                return False, (
                    f"Task {task} requires regime '{req['required_regime']}', "
                    f"but got '{evaluation_regime}'"
                )

            # Check model family compatibility
            if model_family not in req["compatible_model_families"]:
                return False, (
                    f"Task {task} is not compatible with model family '{model_family}'. "
                    f"Compatible families: {req['compatible_model_families']}"
                )

        # Check model-level incompatibilities
        if model_family in MODEL_TASK_COMPATIBILITY:
            compat = MODEL_TASK_COMPATIBILITY[model_family]

            if task_regime in compat.get("incompatible_regimes", []):
                return False, (
                    f"Model family '{model_family}' is incompatible with "
                    f"task regime '{task_regime}' (task: {task})"
                )

    return True, "Compatibility validation passed"


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: validate_compatibility.py <model> <revision> <tasks_comma_separated> [regime]"
        )
        sys.exit(1)

    model = sys.argv[1]
    model_revision = sys.argv[2] if sys.argv[2] != "null" else None
    tasks = sys.argv[3].split(",") if sys.argv[3] else []
    regime = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "null" else None

    valid, message = validate_compatibility(model, model_revision, tasks, regime)
    print(message)
    sys.exit(0 if valid else 1)
