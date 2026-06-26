# Granite 4.0 Nano — SFT + Eval on AWS

End-to-end workflow for fine-tuning Granite 4.0 350M on AWS and running the full evaluation suite using granite.build with SkyPilot.

## Prerequisites

1. AWS credentials configured (`~/.aws/credentials`) with `us-east-2` access
2. SkyPilot installed and verified:
   ```bash
   pip install "skypilot-nightly[aws]"
   sky check aws
   ```
3. ECR docker access (all images are in `022767362696.dkr.ecr.us-east-2.amazonaws.com`)
4. S3 buckets:
   - `s3://granite-build-datasets/tokenized/8192/data_filtered` — tokenized training data
   - `s3://granite-build-checkpoints` — checkpoint storage
   - `s3://granite-build-eval-results` — eval output
5. vCPU quota in us-east-2:
   - SFT training (A100): P-family >= 96 vCPUs (1x p4d.24xlarge)
   - SFT training (L40S fallback): G-family >= 192 vCPUs (1x g6e.48xlarge)
   - Full eval suite: G-family >= 88 vCPUs (22x g6e.xlarge spot instances)

**Important:** `GBSERVER_SECRET_SKYPILOT_DOCKER_PASSWORD` must be set in the
**gbserver process's environment** before the server starts — not in the client
shell where you run `gb build start`. The server reads secrets from its own
`os.environ` via `EnvSpaceSecretManager` at build execution time. If you're
running gbserver as a standalone process, export the variable in the same shell
before launching it. Note that ECR tokens expire after 12 hours, so you'll need
to restart gbserver (or refresh the variable) for long-running sessions.

## Step 1: SFT Training

Fine-tune `granite-4.0-350m-base` using Open-Instruct on 8x A100 GPUs (or L40S as fallback).

```bash
export GBSERVER_SECRET_SKYPILOT_DOCKER_PASSWORD=$(aws ecr get-login-password --region us-east-2)
gb build start -f recipes/granite4-350m/aws/sft/build.yaml --param NAME=oisft001
```

**What this does:**
- Provisions a `p4d.24xlarge` (8x A100 40GB, 96 vCPUs) via SkyPilot
- Uses a custom host AMI (`ami-0cc52c70f2edeb6c6`) with NVIDIA driver 550+
- Runs FSDP-distributed SFT with Open-Instruct
- Saves checkpoints to `s3://granite-build-checkpoints/sft/<NAME>_<timestamp>-hf/`
- Saves every 2500 steps, keeps last 5 checkpoints

**Monitor:**
```bash
gb build list
gb build logs <build-id>
sky status   # cluster state
sky logs granite-sft-l40s  # raw SkyPilot logs
```

**Hyperparameters** (defaults in `openinstruct-sft/step.yaml`):
| Parameter | Default | Description |
|-----------|---------|-------------|
| num_epochs | 3 | Training epochs |
| per_device_train_batch_size | 2 | Batch size per GPU |
| gradient_accumulation_steps | 8 | Effective batch = 2 * 8 * 8 GPUs = 128 |
| learning_rate | 1e-5 | Peak learning rate |
| max_seq_len | 8192 | Sequence length |
| lr_scheduler_type | linear | LR schedule |
| warmup_ratio | 0.03 | Warmup fraction |
| checkpointing_steps | 2500 | Save every N steps |
| mixed_precision | bf16 | BFloat16 training |

**After training completes**, tear down the SFT cluster to free vCPU quota for evals:
```bash
sky down granite-sft-l40s
```

## Step 2: Run All Evaluations (Spot Instances)

Launch the full eval suite (26 evals across 22 spot instances) against a checkpoint.

```bash
export GBSERVER_SECRET_SKYPILOT_DOCKER_PASSWORD=$(aws ecr get-login-password --region us-east-2)
gb build start -f recipes/granite4-350m/aws/full-eval/build.yaml \
  --param NAME=eval-l40s-350m-s7500 \
  --param MODEL_S3=s3://granite-build-checkpoints/sft/v0-20260614_093520-hf/step_hf_7500
```

**What this does:**
- Launches 22 targets in parallel (granite.build targets run concurrently)
- Each target provisions a spot `g6e.xlarge` (1x L40S, 4 vCPUs) via SkyPilot
- All eval steps default to `use_spot: true` for cost savings
- Total vCPU footprint: 88 vCPUs (within 192 quota)

**Eval breakdown (22 clusters):**

| Category | Evals | Image | Clusters |
|----------|-------|-------|----------|
| OLMES (general, math, cruxeval) | 11 | sage-py311-olmes:0.025 | 11 |
| CODE (evalplus, multiple_*) | 7 | sage-py311-code:0.025 | 7 |
| SAFETY (attaq, salad-bench) | 2 | sage-py311-safety:0.025 | 2 |
| MULTILINGUAL (grouped) | 5 | sage-py311-multilingual:0.025 | 1 |
| BFCL | 1 | bfcl-py311:0.02 | 1 |
| **Total** | **26** | | **22** |

**Monitor:**
```bash
gb build list
gb build logs <build-id>
sky status  # see all eval clusters
```

**Results** are written to:
```
s3://granite-build-eval-results/sage/<experiment>/
s3://granite-build-eval-results/bfcl/<experiment>/code-bfclv3/
```

## Running All Evals — Grouped Mode (5 instances)

The cost-optimized approach groups evals by category onto single instances instead of
launching one per eval. Uses 5 instances instead of 22 (77% fewer), trading parallelism
for cost savings:

```bash
export GBSERVER_SECRET_SKYPILOT_DOCKER_PASSWORD=$(aws ecr get-login-password --region us-east-2)
gb build start -f recipes/granite4-350m/aws/full-eval-grouped/build.yaml \
  --param NAME=eval-l40s-350m-s7500 \
  --param MODEL_S3=s3://granite-build-checkpoints/sft/v0-20260614_093520-hf/step_hf_7500
```

| Target | Step | Image | Evals | Instance |
|--------|------|-------|-------|----------|
| olmes-grouped | sage-eval-olmes-grouped | sage-py311-olmes:0.025 | 11 | 1x L40S |
| code-grouped | sage-eval-code-grouped | sage-py311-code:0.025 | 7 | 1x L40S |
| safety-grouped | sage-eval-safety-grouped | sage-py311-safety:0.025 | 2 | 1x L40S |
| multilingual-grouped | sage-eval-multilingual-grouped | sage-py311-multilingual:0.025 | 5 | 1x L40S |
| bfcl | bfcl-eval | bfcl-py311:0.02 | 1 | 1x L40S |

**Cost comparison:**

| Mode | Instances | vCPUs | Spot cost/hr |
|------|-----------|-------|--------------|
| Ungrouped (`run-all-evals`) | 22 | 88 | ~$7.50 |
| **Grouped** (`run-all-evals-grouped`) | 5 | 20 | ~$1.70 |

**Note:** BigCodeBench is NOT included in either mode — launch it separately (see below).

## Running Individual Eval Groups

### OLMES only (11 evals, 1 instance)

```bash
export GBSERVER_SECRET_SKYPILOT_DOCKER_PASSWORD=$(aws ecr get-login-password --region us-east-2)
gb build start -f recipes/granite4-350m/aws/olmes-eval/build.yaml \
  --param NAME=eval-l40s-350m-s7500 \
  --param MODEL_S3=s3://granite-build-checkpoints/sft/v0-20260614_093520-hf/step_hf_7500
```

Runs all 11 OLMES evals sequentially on a single spot instance (sage-py311-olmes:0.025):
- code-olmes-cruxeval
- general-olmes-agi-eval
- general-olmes-bbh
- general-olmes-mmlu-pro
- general-olmes-ifeval
- general-olmes-mmlu-mc
- math-olmes-deepmind-math
- math-olmes-gpqa
- math-olmes-gsm8k
- math-olmes-gsm-symbolic
- math-olmes-minerva-math

### CODE only (7 evals, 1 instance)

```bash
export GBSERVER_SECRET_SKYPILOT_DOCKER_PASSWORD=$(aws ecr get-login-password --region us-east-2)
gb build start -f recipes/granite4-350m/aws/code-eval/build.yaml \
  --param NAME=eval-l40s-350m-s7500 \
  --param MODEL_S3=s3://granite-build-checkpoints/sft/v0-20260614_093520-hf/step_hf_7500
```

Runs all 7 CODE evals sequentially on a single spot instance (sage-py311-code:0.025):
- code-evalplus-humaneval (max_length=4096)
- code-evalplus-mbpp (max_length=4096)
- code-multiple-sh (max_length=512)
- code-multiple-cpp (max_length=512)
- code-multiple-java (max_length=512)
- code-multiple-js (max_length=512)
- code-multiple-rs (max_length=512)

The `multiple_*` evals use `MULTIPLE_LANG` env var to select the target language and a
shorter `max_length=512` for code generation.

### SAFETY only (2 evals, 1 instance)

```bash
export GBSERVER_SECRET_SKYPILOT_DOCKER_PASSWORD=$(aws ecr get-login-password --region us-east-2)
gb build start -f recipes/granite4-350m/aws/safety-eval/build.yaml \
  --param NAME=eval-l40s-350m-s7500 \
  --param MODEL_S3=s3://granite-build-checkpoints/sft/v0-20260614_093520-hf/step_hf_7500
```

Runs both SAFETY evals sequentially on a single spot instance (sage-py311-safety:0.025):
- safety-attaq
- safety-salad-bench

### Multilingual only (5 evals, 1 instance)

```bash
export GBSERVER_SECRET_SKYPILOT_DOCKER_PASSWORD=$(aws ecr get-login-password --region us-east-2)
gb build start -f recipes/granite4-350m/aws/multilingual-eval/build.yaml \
  --param NAME=eval-l40s-350m-s7500 \
  --param MODEL_S3=s3://granite-build-checkpoints/sft/v0-20260614_093520-hf/step_hf_7500
```

Runs all 5 multilingual evals sequentially on a single spot instance (sage-py311-multilingual:0.025):
- multilingual-global-mmlu
- multilingual-mgsm
- multilingual-include-ar-de-es-fr
- multilingual-include-hi-bn-ta-te
- multilingual-include-it-ja-ko-nl-pt-zh

### BigCodeBench (automated sidecar)

BigCodeBench requires an external evaluator sidecar container and more memory than other evals.
granite.build automatically starts the sidecar via a post-launch task on the host VM:

```bash
export GBSERVER_SECRET_SKYPILOT_DOCKER_PASSWORD=$(aws ecr get-login-password --region us-east-2)
gb build start -f recipes/granite4-350m/aws/bcb-eval/build.yaml \
  --param NAME=eval-l40s-350m-s7500 \
  --param MODEL_S3=s3://granite-build-checkpoints/sft/v0-20260614_093520-hf/step_hf_7500
```

**Provisioning Sequence:**

The BCB eval follows a precise asynchronous sequence to coordinate the main container with the sidecar:

```
Time 0:00 — gb build start
  └─ granite.build submits the build target to gbserver

Time 0:30-2:00 — EC2 Instance Provisioning (SkyPilot)
  ├─ AWS provisions g6e.8xlarge spot instance
  ├─ Instance boots, installs Docker, pulls sage-py311-olmes:0.025 image
  └─ SkyPilot marks cluster as UP once provisioned and healthy

Time 2:00-2:30 — File Mounts & Setup (SkyPilot, concurrent)
  ├─ S3 model checkpoint synced to /model (COPY)
  ├─ S3 output bucket mounted to /output (MOUNT)
  └─ Main eval container setup phase runs (HF login, env check, etc.)

Time 2:30 — Main Container Starts Running (Job ID 1)
  ├─ The `run:` section of sage-eval-bcb/step.yaml begins executing
  ├─ IMMEDIATELY enters the BCB evaluator health check loop:
  │  └─ Polls http://localhost:7860/health every 2 seconds
  │  └─ Timeout: 10 minutes (300 attempts × 2 sec = 600 sec)
  └─ Blocks here waiting for sidecar to become healthy

Time 2:30-3:00 — Post-Launch Task Starts (Asynchronous)
  ├─ granite.build detects that cluster is fully provisioned
  ├─ Extracts host IP and SSH key from ~/.sky/generated/ssh/{cluster_name}
  ├─ SSHes directly to ubuntu@HOST_IP:22 (NOT through SkyPilot's container proxy)
  ├─ Executes post_launch_task run script on the HOST VM:
  │  ├─ Authenticates with ECR (docker login)
  │  ├─ Pulls oe-eval-bcb-lite-evaluator:0.01 image (~1-2 minutes)
  │  ├─ Starts sidecar with: docker run -d --network host bcb-evaluator
  │  └─ Waits for sidecar health check (localhost:7860/health, up to 30 seconds)
  └─ Post-launch task completes and exits

Time 3:00-5:00 — Race Condition Window
  ├─ Main container continues polling for sidecar health
  ├─ Sidecar startup happens concurrently (takes 2-3 minutes total)
  └─ Health check succeeds once sidecar is ready

Time 5:00+ — Evaluation Proceeds
  ├─ Main container detects sidecar health check success
  ├─ Proceeds with BigCodeBench evaluation
  └─ Communicates with sidecar via http://localhost:7860/evaluate/
```

**Key Points:**
- The post-launch task runs **concurrently** with the main container, not before it
- The main container must wait long enough for the entire sequence (provisioning + image pull + container start)
- Both run on the **same host** with `--network host`, allowing direct localhost communication
- The 10-minute timeout allows time for:
  - ECR authentication (~10 sec)
  - Docker image pull (~90-120 sec)
  - Container startup (~10 sec)
  - Health check polling (~30 sec max)

No manual SSH or cluster-specific steps needed — everything is automated.

## Running Evals with Completion Tracking (run_all_evals.sh)

The `scripts/run_all_evals.sh` script checks S3 for `.done` markers and only launches
evals that haven't completed yet. This makes it safe to re-run after spot preemptions
or partial failures — completed evals are skipped automatically.

### Usage

```bash
# First, ensure gbserver is running with ECR credentials:
export GBSERVER_SECRET_SKYPILOT_DOCKER_PASSWORD=$(aws ecr get-login-password --region us-east-2)
gbserver standalone --space-dir configurations/spaces/local 2>&1 | tee /tmp/standalone.log

# In another terminal:

# Check what's missing (dry run — no launches):
DRY_RUN=1 ./scripts/run_all_evals.sh sft/v0-20260614_093520-hf/step_hf_7500 eval-l40s-350m-s7500

# Launch missing evals in grouped mode (5 instances, cost-optimized):
GROUPED=1 ./scripts/run_all_evals.sh sft/v0-20260614_093520-hf/step_hf_7500 eval-l40s-350m-s7500

# Launch missing evals in individual mode (22 instances, fast):
./scripts/run_all_evals.sh sft/v0-20260614_093520-hf/step_hf_7500 eval-l40s-350m-s7500

# Force re-run all evals regardless of completion state:
FORCE=1 GROUPED=1 ./scripts/run_all_evals.sh sft/v0-20260614_093520-hf/step_hf_7500 eval-l40s-350m-s7500
```

### Arguments

| Argument | Description |
|----------|-------------|
| `<s3_checkpoint_subpath>` | Path under `s3://granite-build-checkpoints/` (e.g. `sft/v0-20260614_093520-hf/step_hf_7500`) |
| `<experiment>` | Experiment name used for output paths and tracking (e.g. `eval-l40s-350m-s7500`) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DRY_RUN` | `0` | Set to `1` to only check state, don't launch |
| `GROUPED` | `0` | Set to `1` to use grouped mode (5 instances vs 22) |
| `FORCE` | `0` | Set to `1` to re-run all evals regardless of state |

### Completion Tracking

The script tracks eval completion in two ways:

1. **S3 `.done` markers** — Each eval writes a `.done` file to S3 on successful completion:
   - Sage evals: `s3://granite-build-eval-results/sage/<experiment>/<eval-name>.done`
   - BFCL: `s3://granite-build-eval-results/bfcl/<experiment>/code-bfclv3/bfcl.done`

2. **Local state file** — `.eval_runs/<experiment>.completed` caches which evals are done
   to avoid repeated S3 checks. Delete this file to force S3 re-verification.

### Status Detection

For each eval, the script determines status as:

| Status | Meaning |
|--------|---------|
| `completed (local)` | Found in local state file |
| `completed (S3)` | `.done` marker found in S3 |
| `incomplete (preempted)` | `.log` exists but no `.done` — spot instance was preempted |
| `pending` | No evidence of prior run |
| `pending (forced)` | FORCE=1 set, will re-run |

### Output Example

```
============================================================
 granite.build Eval Suite — 26 evals
 Checkpoint: s3://granite-build-checkpoints/sft/v0-20260614_093520-hf/step_hf_7500
 Experiment: eval-l40s-350m-s7500
 Mode:       GROUPED (5 instances)
 DRY RUN — no launches
============================================================

============================================================
 Eval Status Summary — eval-l40s-350m-s7500
============================================================
  EVAL                                          STATUS
  ----                                          ------
  code-evalplus-humaneval                       completed (S3)
  code-evalplus-mbpp                            completed (S3)
  code-multiple-cpp                             incomplete (preempted)
  code-multiple-java                            pending
  ...
  safety-attaq                                  completed (S3)
  safety-salad-bench                            pending
------------------------------------------------------------
  Completed: 18 | Running: 0 | Incomplete: 3 | Pending: 5 | Total: 26
============================================================
```

### BigCodeBench (separate launch)

BigCodeBench requires a sidecar evaluator and is always launched separately:

```bash
gb build start -f recipes/granite4-350m/aws/bcb-eval/build.yaml \
  --param NAME=eval-l40s-350m-s7500 \
  --param MODEL_S3=s3://granite-build-checkpoints/sft/v0-20260614_093520-hf/step_hf_7500
```

## A100 GPU Support — Host AMI Override

### Problem

SkyPilot's default GPU AMI for AWS (`skypilot:custom-gpu-ubuntu`) ships NVIDIA driver
535.216.01, which is broken on kernel 6.8.0-1015-aws. The driver's management API works
(`nvidia-smi` succeeds) but `cuInit()` returns error 802 — making all CUDA computation
impossible. This affects A100 instances (p4d.24xlarge) running the default AMI.

### Solution

granite.build supports overriding the host VM's AMI in Docker mode via the
`docker.host_image_id` field in build.yaml's `launcher_config`:

```yaml
steps:
  - step_uri: space://steps/openinstruct-sft
    config:
      launcher_config:
        resources:
          accelerators: "A100:8"
        docker:
          host_image_id: "ami-0cc52c70f2edeb6c6"
```

This sets the EC2 instance's base AMI to the AWS Deep Learning Base AMI (2026-06-19)
which ships NVIDIA driver 550+, while still running the training code inside the Docker
container specified in the step's `image_id`.

### How It Works

The override flows through three layers:

1. **gbserver** (`src/gbserver/environment/skypilot.py`) merges docker config from the
   step's launcher_config and the build.yaml's config.launcher_config (build.yaml wins),
   then passes it as `_cluster_config_overrides` on `sky.Resources`:
   ```python
   docker_config = {
       **launcher_config.get("docker", {}),
       **config.get("launcher_config", {}).get("docker", {}),
   }
   cluster_config_overrides["docker"] = docker_config
   resources = sky.Resources(..., _cluster_config_overrides=cluster_config_overrides)
   ```

2. **SkyPilot constants** (`sky/skylet/constants.py`) — the key `('docker', 'host_image_id')`
   must be in `OVERRIDEABLE_CONFIG_KEYS_IN_TASK` for the value to survive the
   `Resources.copy()` filtering that occurs during task setup. (This is a local patch
   to SkyPilot; without it the override is silently dropped.)

3. **SkyPilot AWS backend** (`sky/clouds/aws.py`) — in Docker mode, when
   `resources.extract_docker_image()` is not None, the code checks
   `cluster_config_overrides` for `docker.host_image_id`. If present, it uses that AMI
   as the host instance image instead of defaulting to None (SkyPilot's GPU catalog AMI):
   ```python
   if resources.extract_docker_image() is not None:
       overrides = resources.cluster_config_overrides or {}
       host_image_override = (overrides.get('host_image_id') or
                              overrides.get('docker', {}).get('host_image_id'))
       if host_image_override:
           image_id_to_use = {None: host_image_override}
       else:
           image_id_to_use = None
   ```

### Required SkyPilot Patches

Two local patches to the SkyPilot installation (`.venv/lib/python3.13/site-packages/sky/`)
are required until upstream support is added:

1. **`sky/skylet/constants.py`** — Add `('docker', 'host_image_id')` to
   `OVERRIDEABLE_CONFIG_KEYS_IN_TASK`
2. **`sky/clouds/aws.py`** — In `make_deploy_resources_variables()`, read
   `host_image_id` from `cluster_config_overrides` when in Docker mode

After patching, restart the SkyPilot API server:
```bash
sky api stop && sky api start
```

### Finding a Compatible AMI

```bash
aws ec2 describe-images \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
  --query 'Images | sort_by(@, &CreationDate) | [-5:].[ImageId,Name,CreationDate]' \
  --output table \
  --region us-east-2
```

Look for recent AMIs (2026+) which ship driver 550+. Verify the driver version in the
AMI description or by launching a test instance.

### Batch Size for A100 40GB

With 8192 sequence length on A100 40GB GPUs, use:
- `per_device_train_batch_size: 2` (batch_size=8 causes OOM at backward pass)
- `gradient_accumulation_steps: 8` (effective batch = 2 * 8 * 8 GPUs = 128)

## Docker SSH Fix (sage/bfcl images)

The sage and bfcl Docker images set `ENV HOME=/workspace` in their Dockerfile, but
`/etc/passwd` still lists root's home as `/root`. When SkyPilot starts sshd inside
the container, it resolves the user's home directory from `/etc/passwd` — so it
looks for authorized_keys at `/root/.ssh/`, while SkyPilot wrote them to
`/workspace/.ssh/` (following `$HOME`). This mismatch causes "Permission denied
(publickey,password)" during provisioning.

All eval step definitions include the fix:
```yaml
docker:
  run_options:
    - "-e HOME=/root"
```

This overrides `HOME` at `docker run` time so SkyPilot places SSH keys where sshd
expects them. If you create custom steps using these Docker images, include this
`docker` section in your launcher config.

## Post-Launch Tasks (Advanced)

granite.build supports `post_launch_task` sections in launcher configs to run commands
on the host VM during the provisioning phase. This is useful for:

- Starting sidecar containers (like BCB evaluator)
- Setting up host-level dependencies
- Configuring networking or storage

Example in `step.yaml`:

```yaml
launchers:
  my-launcher:
    type: skypilot
    config:
      image_id: "docker:my-image:latest"
      run: |
        # Main job runs in container
        python train.py
      # Runs on host during setup phase (before container starts)
      post_launch_task:
        run: |
          set -e
          # Pull and start sidecar on host
          docker pull my-sidecar:latest
          docker run -d --name my-sidecar --network host my-sidecar:latest

          # Wait for health
          for i in $(seq 1 30); do
            curl -sf http://localhost:9000/health && exit 0
            sleep 1
          done
          exit 1
```

**Implementation detail:** The post-launch task runs on the provisioned cluster's **host VM**
(not in the container) via direct SSH to the EC2 instance's public IP. granite.build extracts
the host IP and SSH key from SkyPilot's generated SSH config (`~/.sky/generated/ssh/{cluster_name}`)
and SSHes directly to `ubuntu@HOST_IP:22`, following the approach used in gbansible's
`run_bcb_eval.sh`. This approach:

- Runs on the host where Docker and other system tools are available (not in the container)
- Executes immediately after provisioning, before the main job starts
- Gives full access to host networking and resources
- Supports starting containers with `--network host` for direct port access
- Works reliably since it uses direct SSH to the host, not SkyPilot's container proxy

The main eval container and sidecar containers can communicate via the host network
(e.g., BCB evaluator on `localhost:7860`, accessible from the eval container via localhost).

## Implementation Details: Post-Launch Task Execution

Post-launch tasks are implemented in `src/gbserver/environment/skypilot.py` and execute asynchronously after the cluster reaches UP status. Here's how it works:

### Code Flow

1. **Cluster Provisioning** (`skypilot.py:_run_workload()`)
   - After `sky.stream_and_get()` returns with the cluster provisioned
   - Extract `post_launch_config` from the launcher configuration

2. **SSH Info Extraction** (`skypilot.py:_extract_host_ssh_info()`)
   ```python
   @retry(stop=stop_after_attempt(30), wait=wait_exponential(multiplier=1, max=10))
   def _extract_host_ssh_info(cluster_name):
       # Read SkyPilot-generated SSH config: ~/.sky/generated/ssh/{cluster_name}
       # Extract HOST_IP from ProxyCommand regex: r"ProxyCommand.*?(\d+\.\d+\.\d+\.\d+)"
       # Extract SSH_KEY from IdentityFile regex: r"IdentityFile\s+(.+)"
       # Retry up to 30 times with exponential backoff (1s → 10s max)
   ```
   - Reads `~/.sky/generated/ssh/{cluster_name}` file (may not exist immediately)
   - Extracts host IP address from SkyPilot's ProxyCommand line
   - Extracts SSH key path from IdentityFile line
   - Retry logic handles timing where SSH config is generated asynchronously

3. **Remote Command Execution** (`skypilot.py:_execute_on_host_via_ssh()`)
   ```python
   def _execute_on_host_via_ssh(host_ip, ssh_key, commands, env_vars=None):
       # Build SSH command: ssh -i {ssh_key} -p 22 ubuntu@{host_ip} bash
       # Inject environment variables into bash script
       # Execute via subprocess.run() with 300-second timeout
       # Log stdout/stderr, raise RuntimeError on failure
   ```
   - SSH directly to `ubuntu@HOST_IP:22` (NOT through SkyPilot's container proxy on port 10022)
   - Runs commands in bash on the host VM where Docker daemon is available
   - Passes environment variables (e.g., `SKYPILOT_DOCKER_PASSWORD`) to the remote script
   - Enforces 300-second timeout to prevent hanging

4. **Error Handling**
   - SSH config file missing → Retry with exponential backoff
   - SSH connection fails → Propagate error and fail the target
   - Post-launch script fails → Propagate error and fail the target
   - Main container will timeout waiting for sidecar if post-launch fails

### Why Direct SSH to Host?

SkyPilot provides a container proxy on port 10022 (`sky ssh`), but post-launch tasks need to:
- Execute on the **host VM** (not in container) where Docker daemon runs
- Access host networking and ports directly
- Start containers with `--network host` flag

Direct SSH to `ubuntu@HOST_IP:22` bypasses the container proxy and reaches the actual EC2 instance.

### Configuration

Define post-launch tasks in launcher config:
```yaml
launchers:
  sage-eval-bcb:
    type: skypilot
    config:
      # Main job configuration
      run: |
        cd /workspace/sage
        sage set hf-token "${HF_TOKEN}"
        bash eval.sh

      # Post-launch task: runs on host VM via direct SSH
      post_launch_task:
        run: |
          set -e
          # All commands execute on ubuntu@HOST_IP:22
          docker login -u AWS --password-stdin ${ECR_SERVER} << EOF
          ${SKYPILOT_DOCKER_PASSWORD}
          EOF
          docker pull ${IMAGE}:${TAG}
          docker run -d --network host --name evaluator ${IMAGE}:${TAG}
```

Environment variables in `post_launch_env` are passed to the remote script via bash exports.

## Spot Instance Behavior

All eval steps default to `use_spot: true`. Spot instances are significantly cheaper
but can be preempted by AWS. When preemption occurs:

- SkyPilot detects the preemption and marks the cluster as terminated
- granite.build detects the step failure and marks the target as failed
- Partial results (`.log` files) may exist in S3 but no `.done` marker is written
- Re-run the build to retry failed evals

To force on-demand instances for a specific eval that keeps getting preempted, override
the step's resources in your build.yaml or edit the step definition directly.

## Cancellation

To stop a running build (tears down all SkyPilot clusters):
```bash
gb build cancel <build-id>
```

### How Cancellation Works

When `gb build cancel` is called, the following sequence occurs:

1. **Cancel signal received** — The REST API marks the build as `cancel_requested`.
   The buildwatcher picks this up and calls `BuildRunner.__cancel_build_run()` which
   invokes `task.cancel()` on the BuildRun's asyncio task.

2. **Propagation down the hierarchy** — CancelledError propagates through the TaskGroup
   chain: `BuildRun.run()` → `TargetRun.run()` → `TargetStepRun.run()`.

3. **Waiting for sky.launch to finish** — If the cluster is still in INIT state
   (provisioning), the TargetStepRun is blocked inside `asyncio.to_thread(sky.launch)`
   which runs in a real OS thread. **OS threads cannot be interrupted by asyncio
   cancellation.** The cancel only takes effect once provisioning completes and the
   `await` returns. This means cancel latency equals the remaining provisioning time
   (typically 2-5 minutes for spot instances).

4. **Cleanup runs** — Once the TargetStepRun's `_run` finishes (either normally or via
   cancellation), the `finally` block in `Run.run` executes `_cleanup()`. For SkyPilot
   steps, this calls `cleanup_skypilot()` which runs `sky.down(cluster_name, purge=True)`
   to terminate the cluster.

5. **sky.down completes** — The cluster is terminated (~12 seconds). Build is marked
   as fully cancelled.

### Cancellation Challenges & Implementation

The cancellation cleanup is non-trivial because Python's asyncio cancellation semantics
conflict with the requirement that `sky.down` must run to completion:

**Problem:** In a cancelled asyncio coroutine, every `await` immediately raises
`CancelledError` without actually waiting. This means standard patterns for running
cleanup code after cancellation fail:

- `asyncio.shield(task)` — Protects the inner task from being cancelled, but the
  outer `await` still raises `CancelledError` immediately without blocking.
- `TaskGroup` + `create_task` — `TaskGroup.__aexit__` cancels all child tasks during
  cancellation, killing the cleanup task before `sky.down` runs.
- `asyncio.ensure_future` + `await` — The `await` raises `CancelledError` without
  blocking, so the cleanup runs as fire-and-forget but the build finishes and the
  event loop moves on before `sky.down` completes, leaking the cluster.

**Solution:** Two-part approach implemented in `src/gbserver/build/run.py` and
`src/gbserver/build/targetsteprun.py`:

1. **`Run.run` finally block** uses `Task.uncancel()` (Python 3.11+) to temporarily
   suppress the pending cancellation, allowing `await cleanup_task` to block normally
   until `_cleanup` finishes. After cleanup completes, it re-cancels the task to
   propagate the cancellation.

2. **`TargetStepRun._cleanup`** calls `cleanup_skypilot()` directly (bypassing
   `environment.cleanup()` which uses `ensure_future`). The `cleanup_skypilot` →
   `_teardown` → `asyncio.to_thread(sky.down)` path is un-cancellable because
   `sky.down` runs in a real OS thread that completes regardless of asyncio state.

### Timeline Example

```
T+0:00   gb build start → cluster provisioning begins (INIT)
T+1:30   gb build cancel → cancel signal received
         └─ BuildRun.run() catches CancelledError, enters finally block
         └─ TargetStepRun still blocked in asyncio.to_thread(sky.launch)
T+4:00   sky.launch completes → cluster is UP
         └─ CancelledError propagates to TargetStepRun.run()
         └─ Run.run finally block calls uncancel() + await _cleanup()
         └─ _cleanup calls cleanup_skypilot() → sky.down(cluster)
T+4:12   sky.down completes → cluster terminated
         └─ Build marked as fully cancelled
```

## Directory Structure

```
recipes/granite4-350m/aws/
  sft/build.yaml                           # SFT training (8x L40S)
  full-eval/build.yaml                     # Full 26-eval suite, 22 instances (spot)
  full-eval-grouped/build.yaml             # Full 26-eval suite, 5 instances (grouped)
  olmes-eval/build.yaml                    # 11 OLMES evals on 1 instance
  code-eval/build.yaml                     # 7 CODE evals on 1 instance
  safety-eval/build.yaml                   # 2 SAFETY evals on 1 instance
  multilingual-eval/build.yaml             # 5 multilingual evals on 1 instance
  bcb-eval/build.yaml                      # BigCodeBench with sidecar (8x L40S)
  bfcl-eval/build.yaml                     # BFCL eval

configurations/assets/environments/skypilot/aws/steps/
  openinstruct-sft/step.yaml               # SFT step definition
  sage-eval/step.yaml                      # Single sage eval (configurable image/script)
  sage-eval-olmes-grouped/step.yaml        # 11 OLMES evals grouped
  sage-eval-code-grouped/step.yaml         # 7 CODE evals grouped
  sage-eval-safety-grouped/step.yaml       # 2 SAFETY evals grouped
  sage-eval-multilingual-grouped/step.yaml # 5 multilingual evals grouped
  sage-eval-bcb/step.yaml                  # BigCodeBench with sidecar
  bfcl-eval/step.yaml                      # BFCL eval

scripts/
  run_all_evals.sh                         # Check S3 and launch missing evals
  ssh-host.sh                              # SSH to host VM (bypass container proxy)
```
