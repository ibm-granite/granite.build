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

### Credentials: the host kubeconfig, or `kube_configs`

SkyPilot reads kube contexts from `KUBECONFIG` (or `~/.kube/config`) on the gbserver host. There are two
ways to supply them:

- **Out of band:** the host's kubeconfig already has the context, e.g. written by `sky local up` for the
  local kind cluster. For a full setup walkthrough, including the SkyPilot SSH-node-pool / RBAC
  requirements, see [setup/skypilot-kubernetes-setup.md](setup/skypilot-kubernetes-setup.md).
- **From the environment, with `kube_configs`:** each entry names a context and a space secret holding a
  kubeconfig that defines it. The context must also be pinned in the environment's own `cloud_config`, as
  `kubernetes.allowed_contexts` or the active workspace's `workspaces.<name>.kubernetes.allowed_contexts`;
  the environment's first setup or launch fails otherwise. Only the environment's `cloud_config` is checked:
  a host `~/.sky/config.yaml` that overrides the workspace can still narrow what SkyPilot uses, which fails
  safe.

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

  `kubeconfig` is the **name** of a space secret whose value is the whole kubeconfig (YAML); no
  credentials appear in `environment.yaml`. Unlike `cluster_ssh_configs` and `aws_credentials` values,
  which fall back to the literal text when no secret of that name exists, `kubeconfig` must name an
  existing secret: a missing one fails setup ("not in the space's secrets"). A kubeconfig is all
  credentials, so a pasted kubeconfig or a mistyped name is refused rather than used. Secret names are
  matched exactly, including case. `context` is a plain context name and is never looked up in secrets.

  gbserver merges each context, with the cluster and user it names, into its own
  `~/.sky/gbserver-kubeconfig` (mode 0600) and adds that file, last, to `KUBECONFIG` in its process. The
  cluster and user are renamed `gbserver/<context>`, so a secret's generic names (`default`, say) can never
  shadow a same-named cluster or user in your kubeconfig. Your own kubeconfig is never edited, and its
  contexts stay usable.

  - **Collisions are refused.** A context, cluster or user of the same name in your own kubeconfig, a
    `current-context` in your kubeconfig that names one of the environment's contexts, or a context already
    written differently by another environment, is refused (`SkypilotConfigCollisionError`). An identical
    context is shared: each environment instance using it is recorded as an owner (gbserver creates one
    instance per space, environment and build thread, so concurrent builds normally get their own). An
    environment re-writing its own entry replaces it, e.g. a rotated token, even while other builds of the
    same environment use it (it's the same secret, and the old token is likely revoked). A build whose
    secrets were read before the rotation keeps using the newer definition rather than putting the old one
    back. A *different*
    environment sharing the context can't change it until that environment's builds finish. Same-named
    environments in two spaces never replace each other's credentials: ownership uses the space's
    *registered* name, not the `name:` in its own `space.yaml`, and a build whose space can't be identified
    never shares a context with another build. Each owner also records the gbserver
    process that wrote it: an owner left behind by a gbserver that stopped mid-build is dropped on the next
    write on that host.
  - **No environment's cluster becomes the default.** gbserver's file comes last on `KUBECONFIG`, so your
    own entries win any name clash. If your kubeconfig sets a `current-context`, gbserver's file sets none,
    and both `kubectl` and the Python client use yours. If it sets none, gbserver's file names an
    unreachable, credential-less placeholder context (`gbserver-no-default-context`) as the default, so a
    request that relies on the default (an environment without `allowed_contexts`, or a `kubectl` call
    without `--context`) fails, as it would without `kube_configs`. With several files on `KUBECONFIG`, both
    the first `current-context` set (which `kubectl` uses) and the last (which the Python client uses) must
    name one of your own contexts; otherwise the placeholder is used. When it is, an environment with
    `allowed_contexts: all` also sees it, and SkyPilot spends about 5 seconds timing out on it when probing
    contexts (pinning `allowed_contexts` avoids that).
  - **Don't make an environment's context your default.** gbserver adds its kubeconfig to `KUBECONFIG`
    only in its own process; to reach an environment's context from your shell, add
    `~/.sky/gbserver-kubeconfig` to your `KUBECONFIG` and use `kubectl --context <context> ...`. Don't
    `kubectl config use-context <context>`: that writes it into your own kubeconfig as its
    `current-context`, so the environment's next setup is refused (`SkypilotConfigCollisionError`),
    because every tool would default to its credentials. Keep your `current-context` on one of your own
    contexts.
  - **Static credentials only.** The secret's `user` may hold only `token`, `client-certificate-data`,
    `client-key-data`, or `username`/`password`; its `cluster` only `server`, `certificate-authority-data`
    and TLS options; the context only `cluster`, `user` and `namespace`. Values must be strings (`true`/
    `false` for `insecure-skip-tls-verify` and `disable-compression`), and `*-data` fields must be valid
    base64: kubectl refuses the whole merged `KUBECONFIG` over a single bad value. Context names may use only
    letters, digits and `. _ : @ / + = -`, starting with a letter or digit. `exec` and `auth-provider` plugins are
    refused: they would make every use of the
    context run a command on the gbserver host, so anyone able to write a space secret could run code
    there. File paths (`client-key`, `client-certificate`, `tokenFile`, `certificate-authority`) are refused
    too: they would resolve on the gbserver host. Embed credentials as `*-data` fields or a token. `server`
    and `proxy-url` are not restricted: gbserver connects to whatever endpoint the secret names, so
    whoever can write a space secret chooses where gbserver sends requests (and that cluster's credentials).
  - **Contexts are released when unused.** Each context records the instances using it. When an instance's
    last target finishes and none of its clusters is still tracked (a SERVICE outlives its target, for
    example), it drops out, and a context nobody else uses is removed from `~/.sky/gbserver-kubeconfig`;
    the next setup merges it again. A context shared by two environments (identical definitions) stays
    while either uses it.
  - **Pins must name a context you or the environment supply.** Every name in an environment's
    `allowed_contexts` must be defined by your own kubeconfig or by that environment's `kube_configs`; any
    other name is refused at setup and launch. Another environment's context is on the process's
    `KUBECONFIG`, and SkyPilot resolves a context when the request runs, so a name that isn't defined yet
    could be merged by another environment in between. To use the same cluster, list it under the
    environment's own `kube_configs` with its own secret. SkyPilot's in-cluster context name and `ssh-*`
    names are always allowed (in a pod or not), and `kube_configs` may not define them. A pin set only
    in the host's `~/.sky/config.yaml` is refused only while gbserver holds that context for an environment,
    so a stale host entry doesn't stop every environment.
  - **"All contexts" is refused while other environments' contexts exist:** `allowed_contexts: all` in the
    active workspace or at the top level, or no `allowed_contexts` while
    `SKYPILOT_ALLOW_ALL_KUBERNETES_CONTEXTS` is true (read from gbserver's environment, which a local API
    server it starts inherits). Pin the contexts explicitly. Contexts left by a gbserver that stopped
    mid-build count until they're pruned.

    Both checks apply to every environment, whatever its `default_cloud` (a step can target Kubernetes on
    its own). They read the host's own `~/.sky/config.yaml` too, where the environment's `cloud_config`
    doesn't set a value. They run on every setup and again just before each launch is submitted. With a
    remote API server they're skipped: it uses its own kubeconfig, never gbserver's.
  - **Managed jobs (`Skypilot_managed`) wait for other environments' contexts.** That environment type has
    no `kube_configs` or pin checks, so while gbserver holds any environment's contexts its launches are
    refused (its request would carry them); they work again once those builds finish.
  - **The native Kubernetes environment (`K8s`) never reads gbserver's kubeconfig:** neither its Python
    client nor its `kubectl`, `oc` and `helm` commands.

  Limits:
  - **"All contexts" timing gap.** An environment allowed all contexts at a moment when no other
    environment held one can still see a context another environment merges before SkyPilot runs its
    request. Pin contexts explicitly to avoid it. Tracked in
    [#571](https://github.com/ibm-granite/granite.build/issues/571), together with the next item.
  - **The credentials are on the process's `KUBECONFIG`.** The `allowed_contexts` check stops another
    environment from *configuring* a managed context, but code in the same process that reads `KUBECONFIG`
    directly (a `kubectl` call by a step running on the gbserver host, say) can still see it. Run environments that must not share a host's credentials in separate
    gbserver processes; per-environment kubeconfig isolation is a follow-up.
  - **Names:** SkyPilot ignores contexts whose names start with `ssh-`, and the placeholder name
    `gbserver-no-default-context` is reserved.
  - **Local API server only.** The SkyPilot client forwards `KUBECONFIG` to a local API server with each
    request; a remote one uses its own kubeconfig. The API server's own background jobs (e.g. its periodic
    status refresh) use the environment the server was *started* with, so they don't see these contexts;
    gbserver's own requests do.
  - **`remote_identity: LOCAL_CREDENTIALS`** uploads the whole merged kubeconfig to the pods, gbserver's
    entries and your own. The Kubernetes default, `SERVICE_ACCOUNT`, uploads none.
  - **No exec-based auth** (e.g. `aws eks get-token`): use a static token or client certificate instead
    (see "Static credentials only").

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
