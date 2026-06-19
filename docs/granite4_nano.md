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
5. vCPU quota for G-family instances in us-east-2:
   - SFT training: 192 vCPUs (1x g6e.48xlarge)
   - Full eval suite: 88 vCPUs (22x g6e.xlarge spot instances)

**Important:** `GBSERVER_SECRET_SKYPILOT_DOCKER_PASSWORD` must be set in the
**gbserver process's environment** before the server starts — not in the client
shell where you run `gb build start`. The server reads secrets from its own
`os.environ` via `EnvSpaceSecretManager` at build execution time. If you're
running gbserver as a standalone process, export the variable in the same shell
before launching it. Note that ECR tokens expire after 12 hours, so you'll need
to restart gbserver (or refresh the variable) for long-running sessions.

## Step 1: SFT Training

Fine-tune `granite-4.0-350m-base` using Open-Instruct on 8x L40S GPUs.

```bash
export GBSERVER_SECRET_SKYPILOT_DOCKER_PASSWORD=$(aws ecr get-login-password --region us-east-2)
gb build start -f samples/standalone/openinstruct-sft/build.yaml --param NAME=oisft001
```

**What this does:**
- Provisions a `g6e.48xlarge` (8x L40S 48GB, 192 vCPUs) via SkyPilot
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
gb build start -f samples/standalone/run-all-evals/build.yaml \
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
s3://granite-build-eval-results/bfcl/<experiment>/
```

## Running Individual Eval Groups

### Multilingual only (5 evals, 1 instance)

```bash
export GBSERVER_SECRET_SKYPILOT_DOCKER_PASSWORD=$(aws ecr get-login-password --region us-east-2)
gb build start -f samples/standalone/sage-eval-multilingual/build.yaml \
  --param NAME=eval-l40s-350m-s7500 \
  --param MODEL_S3=s3://granite-build-checkpoints/sft/v0-20260614_093520-hf/step_hf_7500
```

Runs all 5 multilingual evals sequentially on a single spot instance:
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
gb build start -f samples/standalone/sage-eval-bcb/build.yaml \
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

This propagates cancellation through the full task hierarchy and runs `sky down` on
each provisioned cluster.

## Directory Structure

```
samples/standalone/
  openinstruct-sft/build.yaml         # SFT training
  run-all-evals/build.yaml            # Full 26-eval suite (spot)
  sage-eval-multilingual/build.yaml   # Multilingual evals only

configurations/assets/environments/skypilot/aws/steps/
  openinstruct-sft/step.yaml          # SFT step definition
  sage-eval/step.yaml                 # Single sage eval (configurable image)
  sage-eval-bcb/step.yaml             # BigCodeBench with sidecar
  sage-eval-multilingual-grouped/step.yaml  # 5 multilingual evals grouped
  bfcl-eval/step.yaml                 # BFCL eval
```
