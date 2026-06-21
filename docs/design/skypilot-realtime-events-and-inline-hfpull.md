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
        line_regex: "Special\\stokens\\sfile\\ssaved\\sin\\s.*-hf.*"
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

### How `line_regex` and `field_regex` Interact

The event log parser (`get_events_from_log_line`) processes each log line in two stages:

**Stage 1: `line_regex`** — decides whether this line produces an event at all.
- Runs `re.search(line_regex, log_line)` against the full log line
- If no match → line is skipped entirely
- If match → the **matched substring** (from `match.group(0)`) becomes the input for field extraction

**Stage 2: `field_regex`** — extracts field values from the matched substring.
- Each `event_field` with a `field_regex` runs `re.search(field_regex, matched_text)`
- The result (`match.group(0)`) becomes the field's value

**Example walkthrough:**

Log line:
```
(gb-xxx, pid=3344) Special tokens file saved in /output/test1/v0-hf/epoch_hf_0/special_tokens_map.json
```

1. `line_regex: "Special\\stokens\\sfile\\ssaved\\sin\\s.*-hf.*"` matches.
   Matched text = `Special tokens file saved in /output/test1/v0-hf/epoch_hf_0/special_tokens_map.json`

2. `field_regex: '/.+(?=/[^/]+/?$)'` runs against the matched text.
   This lookahead regex captures from the first `/` up to (but not including) the last path segment.
   Result = `/output/test1/v0-hf/epoch_hf_0`

**Key points:**
- `line_regex` controls **which lines trigger events** — add `-hf` to filter out raw checkpoints
- `field_regex` controls **what value is extracted** — independent of the line filter
- `field_value_template` provides a static value (no regex needed) — used for `binding_id: checkpoint`
- `is_data: True` puts the field into `payload.data` dict (nested), not as a top-level payload kwarg
- `is_json: True` parses the field value as JSON — used for the `binding` field

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

---

## 3. Retry Behavior for SkyPilot Steps

### Two Levels of Retry

SkyPilot steps have two independent retry mechanisms:

**1. Provision retry (`_provision_with_retry`)** — retries the `sky.launch()` call when cloud resources are unavailable (e.g., spot instances exhausted across all zones).

- Controlled by `retry.max_retries` and `retry.delay_seconds` in environment.yaml
- Uses exponential backoff: starts at `multiplier` seconds (default: 30s), doubles each attempt, capped at `delay_seconds`
- Example sequence with `delay_seconds: 1800`: 30s, 60s, 120s, 240s, 480s, 960s, 1800s, 1800s...
- Fires when `_is_transient_provision_error()` returns True (ResourcesUnavailableError, "failed to provision", "failed to acquire resources")
- Each retry tears down the partial cluster before relaunching
- Operates **before** the job starts — no monitor is running yet

**2. RetryHandler (post-launch)** — retries after a job was running and then failed (preemption, NCCL error, crash).

- Same `retry.max_retries` controls the budget
- `AnyFailureRetryStrategy` fires on WORKLOAD_STATUS_EVENT(FAILED) or MESSAGE_EVENT with state=Failed
- `retry.delay_seconds` sets the backoff between handler retries
- Operates through the monitor — only fires after the cluster was provisioned and the job started

### Configuration

```yaml
# configurations/assets/environments/skypilot/aws/environment.yaml
config:
  retry:
    max_retries: 10       # max attempts for both provision and handler retries
    delay_seconds: 1800   # max backoff cap (30 minutes)
```

### Per-Step Retry Control

Steps opt into retry via build.yaml:

```yaml
steps:
  - step_uri: space://steps/sage-eval-bcb
    retry_enabled: true           # enable RetryHandler for this step
    retry_transparently: true     # don't emit intermediate FAILED events to downstream
```

Without `retry_enabled: true`, the RetryHandler is created with `max_retries=0` — provision retries still fire (they're unconditional for transient errors), but post-launch failures are fatal.

### Why Provision Retries Matter for Spot Instances

Spot instance availability fluctuates. The original behavior was 4 rapid retries (1s, 2s, 4s backoff, ~30s total) before giving up. For L40S spot instances that may be unavailable for minutes or hours, this was insufficient. The fix reads `retry.max_retries` and `retry.delay_seconds` from the environment config, allowing up to 10 attempts over ~2.5 hours (with 30-minute max backoff).

### Log Messages

```
# Provision retry (inner) — cluster never launched
Transient provision failure for gb-531c9193-665 (attempt 3): Failed to provision...

# Handler retry (outer) — job ran then failed
[RetryHandler launch_id xxx] Waiting 300.0 seconds before retry (backoff from AnyFailureRetryStrategy)

# Successful launch after retries
SkyPilot cluster gb-531c9193-665 launched: job_id=1 launch_id=...
```

### Monitoring Retries

The `build_status.sh` script shows per-cluster retry status with eval type identification:

```
┌─ Retries ───────────────────────────────────────────────────
│  ⟳ bigcodebench (gb-531c9193-665): attempt 5/10
│  ⟳ bigcodebench (gb-ac2615d8-ac0): attempt 7/10
└────────────────────────────────────────────────────────────
```

Cluster-to-target mapping comes from MESSAGE_EVENT logs ("SkyPilot job on gb-xxx") which include `target_name` in the run_metadata.
