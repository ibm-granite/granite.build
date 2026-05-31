# Troubleshooting

Concrete failure modes and where to look. If your problem isn't here, the
fastest debug path is usually `gb build status <id>` followed by
`gb build log <id>`, then drilling into the relevant runner's logs.

> **Audience:** anyone running gbserver or submitting builds against one. The
> fixes here assume you can shell into the runner's environment and read
> server logs.

## Build is stuck in `pending`

The BuildWatcher hasn't picked it up. Most common causes:

1. **No watcher running.** `gb build list` shows the build but no runner is
   advancing it.
   - Standalone mode: confirm `gbserver standalone` is still alive in its
     terminal.
   - Production: confirm the buildwatcher pod/process is running.
2. **`GBSERVER_DEFAULT_BUILDRUNNER_TYPE` mismatch.** If the watcher is
   configured for `job` (Kubernetes) but you're running standalone, jobs
   never get dispatched. Set `GBSERVER_DEFAULT_BUILDRUNNER_TYPE=thread` for
   local development.
3. **Filtering by space or label.** The watcher polls a specific space or
   label. Check that your build's space matches what the watcher is
   subscribed to.

## Step fails to start

The runner picks the build up, but the step never produces logs.

- **Image pull failure.** In Kubernetes/Docker, look for `ImagePullBackOff`
  or `Error: pulling image`. Confirm the image exists and the runner has
  pull credentials. See
  [`bring-your-own-image.md`](../users/bring-your-own-image.md) for the
  pull-secret flow.
- **Missing secret.** Steps that reference an env-mounted secret fail
  immediately if the secret manager doesn't have it. Check
  [`local-secrets-manager.md`](local-secrets-manager.md) (standalone) or
  the SPS configuration (production).
- **Bad `environment_uri`.** Spelling or template-resolution errors here
  surface as "environment not found." `gb build describe -f build.yaml`
  shows the resolved URI.

## HuggingFace push returns 403

- The token doesn't have write access to the target org/repo, or
- The resource group isn't resolvable.

See [`users/hf-push.md`](../users/hf-push.md) for the URI format and
resource-group resolution rules. Sanity-check with:

```bash
huggingface-cli login --token <your-token>
huggingface-cli whoami
```

## Kubernetes `env` rejected with "must be a STRING"

`BuildTargetStepConfig.validate_k8s_env_section` rejects unquoted integers
in `step.config.k8s.env`. Quote the values:

```yaml
config:
  k8s:
    env:
      NCCL_TIMEOUT:
        value: "10800000"   # not 10800000
```

## SkyPilot doesn't see the SLURM cluster

```bash
sky check slurm
```

If it reports "not configured":

- For the local Docker SLURM cluster, did `make slurm-setup` complete?
  Re-run it. See
  [`skypilot-local-infrastructure.md`](skypilot-local-infrastructure.md).
- For a remote cluster, confirm SSH keys and the cluster is in
  `~/.sky/config.yaml`.

## Step retry isn't happening

Step-level retries are gated by `GBSERVER_ENABLE_STEP_RETRY` *and* by the
step's `retry_enabled` flag. Both must be true. See
[`step-retry-configuration.md`](../features/step-retry-configuration.md) for
the priority chain (build.yaml override → step.yaml default → env-var gate).

Currently only K8s (helm) and LSF (bsub) environments support step-level
retry. Other environments will silently skip retry attempts.

## Standalone mode 401s on remote requests

Localhost-only mode is the default when `GBSERVER_API_KEY` isn't set —
remote requests get 401. Set the same key on both server and client:

```bash
# server
export GBSERVER_API_KEY="my-secret-key"
gbserver standalone --space-dir ...

# client
export GBSERVER_API_KEY="my-secret-key"
gb build start ...
```

## VSCode pytest debugger hangs on a subprocess

Known issue with the VSCode pytest plugin and blocking `subprocess.wait()`
calls in fixtures: the Test Explorer's "Debug Test" button hangs.

Workaround: use a "pytest as module" launch configuration in
`launch.json` (calls `python -m pytest <file>::<test>`) instead of the
plugin's debugger button.

## Capturing state when a build test fails

`test/conftest.py` has a `pytest_runtest_makereport` hook that, on
assertion failures matching `[Build: <id>]` in the message, automatically
appends a JSON dump of the build state to the report. If you're writing
buildtest assertions, format failure messages with `[Build: <id>]` to opt
into this.

The dump is also useful when reproducing a customer's failure: it captures
target/step state at the moment the assertion fired.

## Where to look when nothing else works

- `gb build status <id>` — high-level state, errors per step.
- `gb build log <id>` — per-step stdout/stderr.
- `gb admin log <module>` — server-side logs for the rest server, build
  watcher, build runner. Requires admin auth.
- For SQL-storage debugging: connect to the gbserver Postgres directly and
  read `gb_builds`, `gb_targets`, `gb_steps`, `gb_artifacts`.
- For SQLite (standalone): the database file is at `<space-dir>/.gb/state.sqlite`.
- Runner-environment specifics:
  - K8s: `kubectl logs -n <ns> <pod>` for the launcher pod.
  - SkyPilot: `sky logs <cluster>`.
  - RunPod: the RunPod web console — pods are short-lived.

If you've exhausted the above, file an issue at
<https://github.com/ibm-granite/granite.build/issues> with the build ID,
the relevant `gb build status` output, and any logs that mention errors.
