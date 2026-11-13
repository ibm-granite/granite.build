# SkyPilot on Kubernetes

> **Audience:** operators configuring a `Skypilot` environment whose `default_cloud` is `kubernetes`.
> Read [skypilot.md](skypilot.md) first for the compute model and config common to all clouds; this
> page covers only what is Kubernetes-specific. For the *native* Kubernetes backend (gbserver submits
> via Helm + AppWrapper), see [k8s.md](k8s.md) instead.

## Compute environment

With `default_cloud: kubernetes` (alias `k8s`), SkyPilot launches each step as a pod on an **existing
Kubernetes cluster**. SkyPilot uses the cluster reachable through your kube context — there is **no
SSH config** to materialize and no SkyPilot-managed cluster provisioning; the cluster already exists.

This differs from the native [K8s environment](k8s.md): that path submits an AppWrapper via Helm and
can stream live RabbitMQ events; the SkyPilot path launches a SkyPilot pod and uses the polling
`skypilot_monitor`. Choose SkyPilot-on-Kubernetes when you want one environment definition that targets
Kubernetes alongside other clouds with a uniform launcher.

## Kubernetes-specific configuration

### Credentials: your kubeconfig, or `kube_configs`

SkyPilot reads kube contexts from `KUBECONFIG` (or `~/.kube/config`) on the gbserver host. There are two
ways to supply them:

- **Out of band:** your kubeconfig already has the context, e.g. written by `sky local up` for the local
  kind cluster. For a full setup walkthrough, including the SkyPilot SSH-node-pool / RBAC requirements, see
  [setup/skypilot-kubernetes-setup.md](setup/skypilot-kubernetes-setup.md).
- **From the environment, with `kube_configs`:** each entry names a context and a space secret holding a
  kubeconfig that defines it.

  ```yaml
  config:
    default_cloud: kubernetes
    kube_configs:
      - context: my-cluster
        kubeconfig: my-cluster-kubeconfig   # space secret: a kubeconfig defining my-cluster
    cloud_config:
      kubernetes:
        allowed_contexts: [my-cluster]
  ```

  `kubeconfig` is the **name** of a space secret whose value is the whole kubeconfig (YAML); no credentials
  appear in `environment.yaml`. It must name an existing secret (matched exactly, including case): a
  missing one fails setup, and a pasted kubeconfig is refused rather than used. Each context must also be
  pinned in the environment's own `allowed_contexts` (top level or the active workspace's), or
  `allowed_contexts` must be `all`.

#### How `kube_configs` stays isolated

Each environment *instance* (one per space, environment and build thread) gets its own kubeconfig file under
`~/.sky/gbserver-kube/` (mode 0600), holding only its contexts. Only that instance's SkyPilot requests carry
it: each request's `KUBECONFIG` is your own kubeconfig files followed by that file, and the local SkyPilot
API server applies it to that request alone. gbserver never changes its process `KUBECONFIG`, so nothing
else sees the file: not other environments, not other spaces, not the native Kubernetes (`K8s`)
environment or the `kubectl`/`helm` commands it runs. Two spaces can use the same context name with
different credentials, and `allowed_contexts: all` only ever means your contexts plus the instance's own.
SkyPilot's API server caches its Kubernetes clients by context name; gbserver's `sky_patches` server plugin
keys those caches by `KUBECONFIG` too, so requests the server runs side by side on shared threads (launch
validation, `sky logs`) never get each other's client.

- **Static credentials only.** A secret's `user` may hold only `token`, `client-certificate-data`,
  `client-key-data`, or `username`/`password`; its `cluster` only `server`, `certificate-authority-data`
  and TLS options; its context only `cluster`, `user` and `namespace`. `exec` and `auth-provider` plugins
  are refused (they'd run a command on the gbserver host), and so are file paths (they'd resolve on the
  host). Values must be strings (`true`/`false` for the two flags), and `*-data` must be base64 (line-wrapped
  is fine). `server` and `proxy-url` aren't restricted: whoever can write the secret chooses the endpoint.
- **Names.** Context names may use letters, digits and `. _ : @ / + = -`, and a context's `namespace` must be
  a valid Kubernetes namespace name. `ssh-*` names, SkyPilot's in-cluster context name and
  `gbserver-no-default-context` are reserved. A context, cluster or user of the
  same name in your own kubeconfig is refused, as is one your kubeconfig's `current-context` names. The
  cluster and user are renamed `gbserver/<context>`.
- **No secret context becomes a default.** If your kubeconfig sets no usable `current-context`, the
  instance file's default is an unreachable, credential-less placeholder (`gbserver-no-default-context`).
- **Rotation.** An instance reads its secrets once, so a rotated secret takes effect with the next build.
- **Lifetime.** A file is removed when its environment instance is released (normally when its build
  finishes) or gbserver exits; files left by a gbserver that crashed are removed when gbserver next starts
  a SkyPilot environment. Files belonging to another running gbserver sharing the home directory, or to
  another host, are kept.

Limits:
- **Local API server only.** `kube_configs` is refused unless gbserver's SkyPilot API server is local,
  bound to loopback and not deployed: only such a server applies a forwarded `KUBECONFIG` (gbserver's
  `sky_patches` plugin reports this). Elsewhere the credentials would be silently ignored.
- **Not for managed jobs.** `Skypilot_managed` refuses `kube_configs`: its jobs controller is shared by all
  of your managed jobs and keeps the credentials it's provisioned with.
- **No `SKYPILOT_KUBECONFIG_REFRESH_INTERVAL_SECONDS`.** `kube_configs` is refused while it's set: SkyPilot's
  periodic client refresh rebuilds a client from whatever `KUBECONFIG` the refreshing thread sees, which can be
  the server's own.
- **Background server work** (e.g. the API server's periodic status refresh) uses the server's own
  environment, so it doesn't see these contexts; gbserver's own requests do.
- **No `remote_identity: LOCAL_CREDENTIALS`.** It's refused for `kube_configs` environments (in the
  environment's `cloud_config` or the host's `~/.sky/config.yaml`): SkyPilot would flatten each request's
  kubeconfig, credentials included, into a temporary file it never removes. The Kubernetes default,
  `SERVICE_ACCOUNT`, uploads no credentials.
- **No exec-based auth** (e.g. `aws eks get-token`): use a static token or client certificate.

### No `cluster_ssh_configs` or `aws_credentials`

The Kubernetes backend uses neither inline block. If you need to tune SkyPilot's Kubernetes behaviour,
use a `cloud_config` with a `kubernetes:` block (deep-merged into `~/.sky/config.yaml`); otherwise the
`config:` block is minimal.

### Idle timeout (autodown, not autostop)

A pod cannot be stopped, only deleted, so SkyPilot supports **autodown** on Kubernetes but not autostop —
a plain autostop request fails every launch with "Auto-stop is not supported on Kubernetes". gbserver
therefore applies `idle_minutes_to_autostop` on Kubernetes as autodown (`down=True`): after that many idle
minutes the pod is deleted.

Per-step `cleanup_skypilot()` already runs `sky down` after each step, so the idle timeout (default 10) is
only a safety net that reaps a pod orphaned by a crashed gbserver. **Keep it well above the step's monitor
poll interval** (`poll_interval_seconds`, default 300s): a job that finishes just after a poll would
otherwise be autodowned before the next one, and the poller then sees a missing pod and marks the step
`FAILED`. gbserver enforces a floor of 2× the step's poll interval on Kubernetes, raising a shorter window
and logging a warning; the shipped env uses 15. Set `null` to disable it entirely.

> **`sbatch_options` is a no-op on Kubernetes.** The per-step `sbatch_options`
> field ([skypilot.md](skypilot.md#config-overrides-docker-sbatch_options)) is a
> **SLURM-only** knob; SkyPilot exposes no per-task equivalent on Kubernetes, so
> a value set here is ignored (a WARNING is logged). Bound job runtime inside the
> `run:` command or via a Kubernetes-level policy instead.

### Shared storage: `shared_workdir` on a ReadWriteMany PVC

Every step runs in **its own pod**, which is deleted when the step finishes. So without a volume every pod
mounts, nothing written by one pod is visible to the next: an `hf://` input downloaded by the separate
`hfpull` step never reaches the step that consumes it, and an `env://` output never reaches another target.

The fix is the same model as SLURM's `/shared`: mount one **ReadWriteMany** PersistentVolumeClaim at the
same path in every pod and point `shared_workdir` at it. Each run then gets its per-run workdir under
`${shared_workdir}/builds/…`, the hf assetstore caches to `${shared_workdir}/hf_cache` (a separate `hfpull`
step, as on SLURM), and targets can hand files to each other through `env://<shared path>`.

- **It must be ReadWriteMany, even on a single node.** SkyPilot refuses to start a pod on a ReadWriteOnce
  claim that another pod still uses (`check_pvc_usage_for_pod`), and pods overlap: the `hfpull` pod and
  its consumer, or one step tearing down while the next launches.
- **Attach it with `kubernetes.context_configs.<context>.pod_config`** in the env's `cloud_config`, scoped
  to your context. `cloud_config` is deep-merged into `~/.sky/config.yaml`, and lists there are
  *replaced*, not merged, so a top-level `kubernetes.pod_config` would overwrite any `pod_config` you keep
  for other clusters.
- **Pin the context** with `kubernetes.allowed_contexts: [<context>]`. SkyPilot then launches only there,
  whatever kube context is current. Like all `cloud_config`, it is written into the global
  `~/.sky/config.yaml`, last writer wins: gbserver records what it wrote and, before each launch, removes
  the previous environment's keys and writes the launching environment's, so a pin never reaches another
  environment's launches (keys gbserver never wrote, and keys you edited since, are kept). Between builds it
  does limit plain `sky` use on that machine to the pinned context; `make kube-teardown` removes it.

Kubernetes has no host/container split to worry about: the PVC is attached as a pod volume, so it is the
container's filesystem too. There is no separate `workdir` mount to configure (unlike SLURM), and image
steps see `/shared` like bare ones.

## The two shipped environments

**`skypilot/kubernetes`** works out of the box for steps without `hf://` inputs or cross-target file
handoff. It ships the shared-storage wiring above as a commented-out, documented block to fill in with your
context and RWX claim. Its hf pull is a separate `hfpull` step (no `inline: true`), so `hf://` inputs need
that storage enabled.

**`skypilot/kubernetes/local`** is that setup, working, for a local kind cluster created by `make kube-setup`
(`scripts/kube/setup-kind.sh`). The script creates the `gb-shared` ReadWriteMany claim (a static hostPath
volume on the single kind node, since kind's `local-path` class only provisions ReadWriteOnce) and restores
your previous kube context afterwards. The skypilot/kubernetes build tests run against it.

```yaml
name: sky-kube-local
type: Skypilot
subtype: kubernetes
config:
  default_cloud: kubernetes
  idle_minutes_to_autostop: 15  # applied as autodown on Kubernetes; keep well above the poll interval
  shared_workdir: /shared
  steps:  # the hf steps default to 10Gi; concurrent ones wouldn't fit on one node
    hfpull: { launcher_config: { resources: { memory: "1+" } } }
    hfpush: { launcher_config: { resources: { memory: "1+" } } }
  cloud_config:
    kubernetes:
      allowed_contexts: [kind-skypilot]
      context_configs:
        kind-skypilot:
          pod_config:
            spec:
              volumes:
                - name: gb-shared
                  persistentVolumeClaim:
                    claimName: gb-shared
              containers:
                - name: ray-node
                  imagePullPolicy: IfNotPresent
                  volumeMounts:
                    - name: gb-shared
                      mountPath: /shared
assetstores:
  - store_uri: space://assetstores/hf
    pull:
      - mode: default
        config:
          cache_path: /shared/hf_cache
    push:
      - mode: default
        config: {}
```

The `env://` store is registered implicitly for **every** environment, so it needs no `assetstores` entry.

`imagePullPolicy: IfNotPresent` replaces SkyPilot's `Always` on its `ray-node` container, so an image
already on the kind node (the nightly preloads them with `scripts/ci-docker-images.sh kind-load`) isn't
re-pulled. A mutable tag (`:latest`) then isn't refreshed either: pin a version, or remove the stale
copy with `docker exec skypilot-control-plane crictl rmi <image>`.

Steps may set `image_id` (Kubernetes runs containers natively — no Pyxis/enroot caveat), and
`resources.accelerators` / `resources.memory` map onto the pod's resource requests.

## Custom images

Two requirements, both checked on a local kind cluster:

- **The image must be Debian/Ubuntu-based, or already contain SkyPilot's bootstrap packages.** Before a
  step runs, SkyPilot's pod startup (`sky/templates/kubernetes-ray.yml.j2`) needs `rsync curl wget netcat
  gcc patch pciutils openssh-server`; it installs any that are missing with **`apt-get`**, and the pod
  exits if it cannot. Debian/Ubuntu images (including the `python:*-slim` family) work as-is; a Fedora,
  RHEL or Alpine image works only if all of those are preinstalled. `quay.io/fedora/fedora-minimal`, for
  example, fails at startup. To avoid Docker Hub pull-rate limits, `public.ecr.aws/docker/library/<image>`
  mirrors the Docker official images.
- **With shared storage, the image needs nothing to fetch `hf://` inputs.** They are downloaded by the
  separate `hfpull` step into the shared volume and read from there. Inline pulls (`inline: true`) are
  legacy, kept only for existing environments; don't use them on Kubernetes.

## See also

- [SkyPilot overview](skypilot.md) — compute model, launcher fields, inline-config rules
- [Native Kubernetes (`K8s`) environment](k8s.md) — Helm + AppWrapper, live RabbitMQ events
- [SkyPilot on Kubernetes setup](setup/skypilot-kubernetes-setup.md) — cluster prerequisites
