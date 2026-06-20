# Design: Real-Time Event Streaming & Inline HF Pull for SkyPilot

## 1. Real-Time Artifact Event Streaming

### Problem

Previously, the SkyPilot environment monitor only parsed job logs **after** the job reached terminal status (SUCCEEDED/FAILED). It called `sky.download_logs()` to rsync the full log directory, then iterated line-by-line applying regex-based event configs. This meant artifact events (checkpoint saves, progress updates) were never emitted during training — only post-mortem.

Other environments (LSF, Docker, K8s) already stream logs in real-time using the `LogFileMonitor` + `LogStreamSource` pattern.

### Solution

A new `SkyPilotLogStreamSource` wraps `sky.tail_logs(follow=True, preload_content=False)` — the same API behind `sky logs <cluster>` — and implements the `LogStreamSource` protocol. It runs as a concurrent task alongside the existing status polling loop.

### Architecture

```
_poll_skypilot_job (status polling every 15-30s)
    │
    ├── On RUNNING transition:
    │     └── Launch log_stream_task (LogFileMonitor + SkyPilotLogStreamSource)
    │           └── sky.tail_logs() → async iterator → get_events_from_log_line()
    │                                                    → event_q.put(BuildEvent)
    │
    ├── On stream task failure (restartable):
    │     └── Restart with start_line=logfile_monitor.line_num (skip already-processed lines)
    │
    └── On terminal status:
          ├── Stop log stream task
          └── _download_and_parse_logs(start_line_num=lines_already_processed)
              (fallback: only processes lines the live stream didn't reach)
```

### Key Files

- `src/gbserver/monitoring/streams/skypilot_log_stream.py` — `SkyPilotLogStreamSource` class
- `src/gbserver/environment/skypilot.py` — `_poll_skypilot_job` (concurrent streaming), `_start_log_stream_task`, `_download_and_parse_logs` (with `start_line_num`)
- `src/gbserver/monitoring/logfile_monitor.py` — Reused as-is

### Resume Strategy

`sky.tail_logs()` always replays from the beginning (no offset parameter). On reconnection after a transient failure:

1. Read `logfile_monitor.line_num` — the count of lines successfully processed
2. Create a new `SkyPilotLogStreamSource(cluster, job_id, start_line=N)`
3. The source skips the first N lines internally before yielding to the monitor

On terminal status, `_download_and_parse_logs` accepts `start_line_num` and skips lines already emitted by the live stream, avoiding duplicates.

### Event Config Example (step.yaml)

```yaml
monitors:
  skypilot_monitor:
    type: skypilot_monitor
    config:
      poll_interval_seconds: 30
      event_configs:
      - event_type: WORKLOAD_STATUS_EVENT
        line_regex: "Step:\\s\\d+"
        event_fields:
          - field_name: status
            field_value_template: "RUNNING"
          - field_name: message
            field_regex: "Step:.+"
            is_data: True    # <-- required: puts message in payload.data, not as a kwarg
      - event_type: NEWARTIFACT_IN_ENVIRONMENT_EVENT
        line_regex: "Special\\stokens\\sfile\\ssaved\\sin\\s.*"
        event_fields:
          - field_name: binding_id
            field_value_template: checkpoint
          - field_name: path
            field_regex: '/.+(?=/[^/]+/?$)'
            is_data: True
          - field_name: binding
            field_value_template: "{ \"path\": \"{{ fields.data.path }}\" }"
            is_json: True
```

---

## 2. Inline HF Pull (No Separate Cluster)

### Problem

On LSF/Slurm, the hfpull builtin step downloads models to a **shared filesystem** (`/shared/hf_cache`), so the training step on the same FS can access it. On AWS SkyPilot, each cluster has isolated storage — the hfpull cluster's downloads are lost when it's torn down.

For provenance tracking, inputs must still be declared via `inputs:` with `hf:///` URIs so the build metadata records the source artifact.

### Solution

Add `inline: true` to the HF assetstore's load config. When `pullasset_hfstore` sees this flag, it:
1. Returns `(binding_config, None)` — no separate hfpull cluster launched
2. Stashes download metadata in `self._pending_hfpulls`
3. When the main step launches, `_launch_skypilot_inner` prepends `hf download` commands to the setup script

The model downloads on the **same cluster** that runs training.

### Configuration

**Environment YAML** (`configurations/assets/environments/skypilot/aws/environment.yaml`):

```yaml
assetstores:
  - store_uri: space://assetstores/hf
    load:
      - mode: hf_pull
        config:
          cache_path: /tmp/hf_cache
          inline: true          # <-- download on the training cluster, not a separate one
    push:
      - mode: hf_push
        config: {}
```

**Build YAML** (`recipes/granite4-350m/aws/sft/build.yaml`):

```yaml
targets:
  sft-training:
    inputs:
      model:
        uri: "hf:///ibm-granite/granite-4.0-350m-base"
        type: model
    steps:
      - step_uri: space://steps/openinstruct-sft
        config:
          sft_config:
            model_path: "{{ bindings.model.binding.path }}"
```

**Step YAML** (optional `inputs` declaration for validation):

```yaml
inputs:
  optional:
    model:
      type: model
      accept: [uri, binding]
```

### How It Works

```
1. build.yaml declares:  inputs.model.uri = "hf:///ibm-granite/granite-4.0-350m-base"

2. TargetRun.run() calls pull_assets() → dispatches to pullasset_hfstore()

3. pullasset_hfstore() sees inline: true in storeload_config:
   - Computes binding_path = /tmp/hf_cache/ibm-granite/granite-4.0-350m-base/main
   - Stashes in self._pending_hfpulls["model"] = {repo, path, revision, type, token}
   - Returns ({"binding": {"path": "/tmp/hf_cache/.../main"}}, None)
                                                                  ^^^^ no separate step

4. TargetRun sets: bindings["model"] = {"binding": {"path": "/tmp/hf_cache/.../main"}}

5. Template rendering resolves:
   {{ bindings.model.binding.path }} → /tmp/hf_cache/ibm-granite/granite-4.0-350m-base/main

6. _launch_skypilot_inner() sees self._pending_hfpulls is non-empty:
   - Prepends to setup script:
     pip install --no-cache-dir 'huggingface_hub[cli]' 2>/dev/null || true
     hf download "ibm-granite/granite-4.0-350m-base" \
       --local-dir "/tmp/hf_cache/ibm-granite/granite-4.0-350m-base/main" \
       --repo-type model

7. On the cluster:
   - Setup runs hf download → model at /tmp/hf_cache/.../main
   - Step's own setup sees -d "${MODEL_PATH}" is true → skips snapshot_download
   - Training uses --model_name_or_path /tmp/hf_cache/.../main
```

### Binding Path Template

The binding structure returned by `pullasset_hfstore` is:
```python
{"binding": {"path": "/tmp/hf_cache/ibm-granite/granite-4.0-350m-base/main"}}
```

In templates, access it as: `{{ bindings.<input_name>.binding.path }}`

### Multiple Inputs

Works for any number of inputs — each gets its own `hf download` line in the injected setup:

```yaml
inputs:
  model:
    uri: "hf:///ibm-granite/granite-4.0-350m-base"
    type: model
  dataset:
    uri: "hf:///org/my-training-data"
    type: dataset
```

All downloads are injected into a single setup block before the step's own setup runs.

### When NOT to Use `inline: true`

- Environments with **shared filesystems** (LSF, Slurm with `shared_workdir`) don't need it — the default hfpull step downloads to a shared path visible to all jobs
- K8s with PVCs — the hfpull pod and training pod share the same PVC

### Provenance

The build metadata records `inputs.model = hf:///ibm-granite/granite-4.0-350m-base` regardless of whether the download was inline or via a separate cluster. The binding path is tracked in the build's runtime state.
