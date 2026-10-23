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
| `slurm-bluevela` | `space://environments/skypilot/slurm/bluevela` | `.` (custom test space: `test-data/slurm-bluevela/space.yaml`, see below) | `pytest.mark.ibm` | same as `lsf-bluevela`, plus an `IBM_CLOUD_API_KEY` skip | `GBTEST_SPS_IBMCLOUD_API_KEY` set (`make ibm-venv`) | The env exists only in gb-test, and its SSH key is an `ibmcloud` secret, so the fixture dir carries its own `space.yaml`. The step still resolves from **your source**. Mode-1 only: not published. CPU-only steps need `zone: cpu` + `launcher_config.sbatch_options.constraint: null` (gpu-mid rejects GPU-less jobs). Shared partition queueing, so use a long `timeout_minutes` and the timeout override. hfpull is separate. |

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

## Custom test space (`slurm-bluevela`, or any cluster that needs one)

Write `test-data/<cluster>/space.yaml` and set the fixture's `space_uri: .`. `make space`
links `test-data/<cluster>/steps` to the generated `space/steps`, so the step resolves
from the local source; `make publish-step` skips the cluster. `base_uris` are not
inherited, so list each one. Never put a secret in it.

For BlueVela, the space must match `configurations/spaces/ibm/space.yaml` (the IBM
space), except that its `file://` base is the generated space, `file://../../space`.
Copy `steps/misc/byoc/skypilot/test-data/slurm-bluevela/space.yaml`:

```yaml
name: public
secret_manager:
  type: ibmcloud
  config:
    service_url: https://3a634e1e-2591-4a19-baa8-34cf8f12defb.us-east.secrets-manager.appdomain.cloud
base_uris:
  - git+ssh://github.ibm.com/granite-dot-build/gb-test.git@gbspace-config
  - git+ssh://github.ibm.com/granite-dot-build/assets.git@gbspace-config-dev
  - file://../../space
```

`steps/conftest.py` loads the repo's `test/conftest.py`, so `GBTEST_SPS_IBMCLOUD_API_KEY`
fills in `IBM_CLOUD_API_KEY` from SPS. Add this skip so a run without it skips instead of
failing:

```python
@pytest.mark.skipif(
    not os.environ.get("IBM_CLOUD_API_KEY"),
    reason="IBM_CLOUD_API_KEY (set GBTEST_SPS_IBMCLOUD_API_KEY) is required by the custom test space's ibmcloud secret manager",
)
```

## Any other IBM space

If the user gives a different `github.ibm.com` space for a cluster, use it as the
`space_uri` and mark it `ibm` and say the step resolves from that space (not your source).
To test your source there instead, chain to it from a custom test space (above).
