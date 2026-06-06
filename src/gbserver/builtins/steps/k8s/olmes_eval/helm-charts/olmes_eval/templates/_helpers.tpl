{{- define "olmes_eval_command" -}}
{{- $config := .Values.olmes_eval_config -}}
python3 -c '
import os
import sys
import subprocess
import json
from pathlib import Path

# Configuration
model = """{{ $config.model }}"""
model_revision = {{ if $config.model_revision }}"{{ $config.model_revision }}"{{ else }}None{{ end }}
model_commit = {{ if $config.model_commit }}"{{ $config.model_commit }}"{{ else }}None{{ end }}
model_family = {{ if $config.model_family }}"{{ $config.model_family }}"{{ else }}None{{ end }}
model_path = {{ if $config.model_path }}"{{ $config.model_path }}"{{ else }}None{{ end }}
trust_remote_code = {{ $config.trust_remote_code }}
tasks = {{ $config.tasks | toJson }}
evaluation_regime = {{ if $config.evaluation_regime }}"{{ $config.evaluation_regime }}"{{ else }}None{{ end }}
validate_compatibility = {{ $config.validate_compatibility }}
fail_on_incompatibility = {{ $config.fail_on_incompatibility }}
output_dir = "{{ $config.output_dir }}"
batch_size = {{ $config.batch_size }}
num_shots = {{ if $config.num_shots }}{{ $config.num_shots }}{{ else }}None{{ end }}
limit = {{ if $config.limit }}{{ $config.limit }}{{ else }}None{{ end }}
gpus = {{ $config.gpus }}
track_model_hash = {{ $config.track_model_hash }}
track_tokenizer_version = {{ $config.track_tokenizer_version }}
save_raw_requests = {{ $config.save_raw_requests }}

print(f"OLMES Evaluation Configuration:")
print(f"  Model: {model}")
print(f"  Model Revision: {model_revision}")
print(f"  Model Commit: {model_commit}")
print(f"  Model Family: {model_family}")
print(f"  Model Path: {model_path}")
print(f"  Trust Remote Code: {trust_remote_code}")
print(f"  Tasks: {tasks}")
print(f"  Evaluation Regime: {evaluation_regime}")
print(f"  Output Directory: {output_dir}")
print(f"  Batch Size: {batch_size}")
print(f"  GPUs: {gpus}")
print()

# Validate compatibility if enabled
if validate_compatibility and tasks:
    print("Running compatibility validation...")
    validation_script = Path("/workspace/scripts/validate_compatibility.py")

    tasks_str = ",".join(tasks)
    regime_arg = evaluation_regime if evaluation_regime else "null"
    revision_arg = model_revision if model_revision else "null"

    cmd = [
        sys.executable,
        str(validation_script),
        model,
        revision_arg,
        tasks_str,
        regime_arg
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    if result.returncode != 0:
        print(f"Compatibility validation failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        if fail_on_incompatibility:
            sys.exit(1)
        else:
            print("Warning: Continuing despite compatibility issues")
    else:
        print("Compatibility validation passed")
    print()

# Build OLMES command
cmd = [sys.executable, "-m", "oe_eval.launch"]

# Model specification
if model_path:
    cmd.extend(["--model_path", model_path])
else:
    cmd.extend(["--model", model])
    if model_revision:
        cmd.extend(["--model_revision", model_revision])

# Trust remote code
if trust_remote_code:
    cmd.append("--trust_remote_code")

# Tasks
if tasks:
    for task in tasks:
        cmd.extend(["--task", task])

# Evaluation regime
if evaluation_regime:
    cmd.extend(["--regime", evaluation_regime])

# Output directory
cmd.extend(["--output_dir", output_dir])

# Batch size
cmd.extend(["--batch_size", str(batch_size)])

# Optional parameters
if num_shots is not None:
    cmd.extend(["--num_shots", str(num_shots)])

if limit is not None:
    cmd.extend(["--limit", str(limit)])

# GPU configuration
if gpus > 0:
    cmd.extend(["--gpus", str(gpus)])

# Metadata tracking
if track_model_hash:
    cmd.append("--track_model_hash")

if track_tokenizer_version:
    cmd.append("--track_tokenizer_version")

if save_raw_requests:
    cmd.append("--save_raw_requests")

print(f"Running OLMES evaluation:")
print(f"Command: {\" \".join(cmd)}")
print()

# Run evaluation
result = subprocess.run(cmd, check=False)

if result.returncode != 0:
    print(f"OLMES evaluation failed with return code {result.returncode}", file=sys.stderr)
    sys.exit(result.returncode)

print()
print("OLMES evaluation completed successfully")

# Write output paths for gbserver bindings
output_path = Path(output_dir)
results_file = output_path / "results.json"
metrics_file = output_path / "metrics.json"

# Write binding files
Path("/workspace/results_path.txt").write_text(str(results_file))
if metrics_file.exists():
    Path("/workspace/metrics_path.txt").write_text(str(metrics_file))

print(f"Results written to: {results_file}")
if metrics_file.exists():
    print(f"Metrics written to: {metrics_file}")
'
{{- end -}}
