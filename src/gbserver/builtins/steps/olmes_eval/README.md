# OLMES Evaluation Steps

This directory contains gbserver steps for running OLMES (Open Language Model Evaluation System) evaluations on language models.

## Overview

The OLMES integration consists of two steps that work together in a pipeline:

1. **hf_model_pull**: Downloads models from HuggingFace Hub with version tracking
2. **olmes_eval**: Runs OLMES evaluation suite with model-task compatibility validation

These steps can be combined with the existing `lhpush` step to create complete evaluation pipelines.

## Steps

### hf_model_pull

Downloads models from HuggingFace Hub to a local directory for evaluation.

**Key Features:**
- Downloads models with specific revisions (tags, branches, commits)
- Caches models to avoid redundant downloads
- Supports trust_remote_code for custom model code
- Outputs model path for binding to downstream steps

**Configuration:**

```yaml
hf_model_pull_config:
  model_id: "allenai/OLMo-7B"            # HuggingFace model ID
  revision: "step1000-tokens4B"          # Specific revision (optional)
  output_dir: "/workspace/models"        # Where to save model
  cache_dir: "/root/.cache/huggingface" # HF cache directory
  trust_remote_code: false               # Allow custom model code
  token: null                            # HF token (for private models)
```

**Outputs:**
- `model_path`: Path to downloaded model directory (for binding)

### olmes_eval

Runs OLMES evaluation suite on a language model with automatic compatibility validation.

**Key Features:**
- Pre-flight validation of model-task compatibility
- Prevents dimension mismatches between model versions and evaluation regimes
- Supports multiple evaluation regimes (OLMES-v0.1, OLMo-v1, TÜLU 3, OLMo 2)
- Tracks model hash and tokenizer version for reproducibility
- Saves raw evaluation requests for debugging

**Configuration:**

```yaml
olmes_eval_config:
  # Model specification
  model: "allenai/OLMo-7B"              # Model name or HF path
  model_revision: "step1000-tokens4B"   # HF revision (for validation)
  model_commit: null                    # Exact commit hash (optional)
  model_family: "OLMo-1.x"             # Model family (optional, helps validation)
  model_path: null                      # Pre-downloaded model path (optional)

  # Task configuration
  tasks:                                # List of tasks to evaluate
    - "arc_challenge::olmes"
    - "mmlu::olmes"
  evaluation_regime: "OLMES-v0.1"      # Evaluation regime

  # Validation
  validate_compatibility: true          # Run pre-flight validation
  fail_on_incompatibility: true        # Fail vs. warn on incompatibility

  # Execution
  output_dir: "/workspace/results"      # Where to save results
  batch_size: 16                        # Batch size for inference
  num_shots: null                       # Override number of shots (optional)
  limit: null                           # Limit instances per task (for testing)
  gpus: 1                              # Number of GPUs

  # Reproducibility
  track_model_hash: true               # Store model file checksums
  track_tokenizer_version: true        # Track tokenizer version
  save_raw_requests: true              # Save raw evaluation requests
```

**Outputs:**
- `results_path`: Path to results.json file
- `metrics_path`: Path to metrics.json file (if generated)

## Model-Task Compatibility

**Critical:** Different model versions may have incompatible dimensions with certain evaluation tasks. The `validate_compatibility.py` script automatically checks compatibility before running evaluations.

### Model Families

- **OLMo-1.x**: Models with revision format `step1000-tokens4B`
  - Compatible: OLMES-v0.1, OLMo-v1
  - Incompatible: TÜLU 3, OLMo 2

- **OLMo-2.x**: Second-generation OLMo models
  - Compatible: OLMES-v0.1, OLMo 2
  - Incompatible: (none specified)

- **TÜLU-3**: TÜLU 3 instruction-tuned models
  - Compatible: TÜLU 3, tulu_3_dev, tulu_3_unseen
  - Incompatible: OLMo-v1

### Automatic Detection

The validation script automatically detects model families based on:
- Model name (e.g., "olmo-2" → OLMo-2.x, "tulu" → TÜLU-3)
- Revision pattern (e.g., "step1000-tokens4B" → OLMo-1.x)

You can also explicitly specify `model_family` to override automatic detection.

### Task Regime Requirements

Certain tasks require specific evaluation regimes:

- `core_9mcqa::olmes` → OLMES-v0.1 (OLMo-1.x, OLMo-2.x only)
- `main_suite::olmo1` → OLMo-v1 (OLMo-1.x only)
- `tulu_3_dev` → TÜLU 3 (TÜLU-3 only)
- `olmo_2_generative::olmes` → OLMo 2 (OLMo-2.x only)

## Usage Examples

### Example 1: Basic Pipeline (Pull + Eval + Push)

```yaml
targets:
  olmes_evaluation:
    steps:
      # Pull model from HuggingFace
      - step_uri: "space://steps/hf_model_pull"
        name: pull_model
        config:
          hf_model_pull_config:
            model_id: "allenai/OLMo-7B"
            revision: "step1000-tokens4B"

      # Run OLMES evaluation
      - step_uri: "space://steps/olmes_eval"
        name: evaluate
        config:
          olmes_eval_config:
            model: "allenai/OLMo-7B"
            model_revision: "step1000-tokens4B"
            model_family: "OLMo-1.x"
            evaluation_regime: "OLMES-v0.1"
            tasks:
              - "arc_challenge::olmes"
              - "mmlu::olmes"
        inputs:
          model_path:
            binding: pull_model.model_path

      # Push results to Lakehouse
      - step_uri: "space://steps/lhpush"
        name: push_results
        config:
          lhpush_config:
            source: "/workspace/results"
            target: "lh://evaluations/olmo-7B-step1000"
        inputs:
          results_path:
            binding: evaluate.results_path
```

### Example 2: Using Pre-Downloaded Model

If you already have a model downloaded, skip the pull step:

```yaml
targets:
  eval_local_model:
    steps:
      - step_uri: "space://steps/olmes_eval"
        name: evaluate
        config:
          olmes_eval_config:
            model_path: "/data/models/olmo-2-7b"
            model_family: "OLMo-2.x"
            evaluation_regime: "OLMo 2"
            tasks:
              - "olmo_2_generative::olmes"
```

### Example 3: TÜLU 3 Evaluation

```yaml
targets:
  tulu3_eval:
    steps:
      - step_uri: "space://steps/hf_model_pull"
        name: pull_tulu
        config:
          hf_model_pull_config:
            model_id: "allenai/tulu-3-7b"

      - step_uri: "space://steps/olmes_eval"
        name: evaluate_tulu
        config:
          olmes_eval_config:
            model: "allenai/tulu-3-7b"
            model_family: "TÜLU-3"
            evaluation_regime: "TÜLU 3"
            tasks:
              - "tulu_3_dev"
              - "tulu_3_unseen"
        inputs:
          model_path:
            binding: pull_tulu.model_path
```

## Validation Errors

If you see validation errors like:

```
Task mmlu::olmes is not compatible with model family 'TÜLU-3'.
Compatible families: ['OLMo-1.x', 'OLMo-2.x']
```

This means the model version doesn't support the requested task. Check the compatibility matrix above and adjust either:
- The model/revision to match the task requirements
- The tasks to match the model capabilities

## Testing with Limited Data

For testing, use the `limit` parameter to reduce evaluation time:

```yaml
olmes_eval_config:
  limit: 10  # Only evaluate 10 instances per task
```

## Reproducibility

For reproducible results, specify:
- `model_commit`: Exact commit hash (instead of branch/tag)
- `track_model_hash: true`: Store checksums of model files
- `track_tokenizer_version: true`: Record tokenizer version
- `save_raw_requests: true`: Save all evaluation requests

Example:

```yaml
olmes_eval_config:
  model: "allenai/OLMo-7B"
  model_commit: "abc123..."  # Exact commit
  track_model_hash: true
  track_tokenizer_version: true
  save_raw_requests: true
```

## See Also

- Full example: `example_build.yaml`
- Design document: `OLMES_INTEGRATION_DESIGN.md` (in project root)
- OLMES documentation: https://github.com/allenai/olmes
