# Build-test cluster catalogue

Offer these to the user as a multi-select. Each row is one `test/<cluster>/` dir.
For a variant on the same cluster, suffix the dir: `slurm-pii`, `aws-tok`.

## SkyPilot steps (`steps/<path>/skypilot/`)

| Cluster dir | `environment_uri` | `space_uri` | `pytestmark` | Skip gate | Setup | Notes |
|---|---|---|---|---|---|---|
| `slurm` | `space://environments/skypilot/slurm` | `../../space` | `pytest.mark.skypilot_integration` | `_slurm_cluster_reachable()` from `integration.environment.test_skypilot_slurm_e2e` | `test-setup` → `$(MAKE) -C $(REPO_ROOT) slurm-setup` | Local Docker SLURM. **No Pyxis**: custom-image steps can't run. `hf://` input is a separate hfpull step (+1 `step_count`). Each step is its own node, so cross-step paths must be shared (`env:///shared/...`). |
| `aws` | `space://environments/skypilot/aws` | `../../space` | `pytest.mark.skypilot_integration` | `aws_credentials_present()` from `gbserver.environment.skypilot` | export AWS creds | Real EC2 (costs money). `hf://` pull is **inline** (no extra step). Custom-image steps OK (image must be in a pullable registry). |
| `kube` | `space://environments/skypilot/kubernetes/local` | `../../space` | `pytest.mark.skypilot_integration` | `kind_cluster_reachable()`, reason `kind_skip_reason()`, plus `@KUBE_XDIST_GROUP` (all from `libgbtest.kube`) | `test-setup-kube` → `$(MAKE) -C $(REPO_ROOT) kube-setup` | Local kind cluster. Shared volume at `/shared`. hfpull is separate. |
| `lsf-bluevela` | `space://environments/skypilot/lsf/ibm-bluevela` | `../../space` | `pytest.mark.ibm` | `RUNNING_IN_CICD` skip (see below) + `@pytest.mark.xdist_group(name="buildtest_bv")` | `~/.ssh/ibm-bluevela.key` locally | IBM infra through the local space, so still `ibm`. Enroot for images. |
| `slurm-bluevela` | `space://environments/skypilot/slurm/bluevela` | `git+ssh://github.ibm.com/granite-dot-build/gb-test.git@gbspace-config` | `pytest.mark.ibm` | same as `lsf-bluevela` | `GBTEST_SPS_IBMCLOUD_API_KEY` set | **IBM space only**: the env exists only in gb-test. The step resolves from gb-test's assets, **not your source**: it must be published there first. Shared partition queueing, so use a long `timeout_minutes` and the timeout override. hfpull is separate. |

**`hf://` outputs, on every cluster.** Each one is pushed by its own hfpush step, so the
run needs an `HF_TOKEN` that can write `GBTEST_HF_PUSH_ORG` (default `ibm-research`;
locally set it to your own HF user, plus `GBTEST_SPS_ENABLE_ENV_VAR_OVERRIDE=true` so the
secret manager's token doesn't replace yours). On aws, no step fixture in this repo
pushes `hf://` yet (dpk's aws fixtures use `env://`), so treat aws push counts as
unverified until calibrated.

Gate imports, exactly as the existing tests write them:

```python
from integration.environment.test_skypilot_slurm_e2e import _slurm_cluster_reachable  # slurm
from gbserver.environment.skypilot import aws_credentials_present                     # aws
from libgbtest.kube import KUBE_XDIST_GROUP, kind_cluster_reachable, kind_skip_reason  # kube
```

**BlueVela memory.** gbserver drops `compute_config.total_memory_per_node` on slurm and
lsf, and BlueVela enforces per-job memory (the partition default is 2 CPUs / 2G). Request
memory in the fixture's step config: `launcher_config: {resources: {memory: "16"}}` (bare GiB, as a string).

The BlueVela CI skip used by existing tests:

```python
@pytest.mark.skipif(
    os.environ.get("RUNNING_IN_CICD", "False").lower() == "true",
    reason="Skip in SPS CI/CD until the bluevela environment.yaml references BV_SSH_PRIVATE_KEY in the space repos",
)
```

## Docker steps (`steps/<path>/docker/`)

| Cluster dir | `environment_uri` | `space_uri` | `pytestmark` | Skip gate |
|---|---|---|---|---|
| `docker` | `space://environments/docker` | `../../space` | `pytest.mark.docker_required` | `RUNNING_IN_CICD` skip (image pulling isn't automatic in CI) |

## Bash steps (`steps/<path>/bash/`)

| Cluster dir | `environment_uri` | `space_uri` | `pytestmark` | Skip gate |
|---|---|---|---|---|
| `bash` | `space://environments/bash` | `../../space` | none needed | none (runs locally) |

## Any other IBM space

If the user gives a different `github.ibm.com` space for a cluster, use it as the
`space_uri` and apply the same rules as `slurm-bluevela`: mark it `ibm`, and say the
step resolves from that space.
