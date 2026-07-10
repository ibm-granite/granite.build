---
name: run-gbserver
description: Clone, set up, and run the Granite.build standalone gbserver. Use when asked to set up, install, start, or run gbserver / the Granite.build standalone server and when running builds.
argument-hint: "[path-to-existing-checkout]"
allowed-tools: Bash(git clone *) Bash(git -C *) Bash(make *) Bash(source *) Bash(gbserver *) Bash(curl *) Bash(ls *) Bash(test *) Bash(python3 *) Bash(python3.13 *) Bash(lsof *) Bash(sleep *)
---

# Run the Granite.build standalone gbserver

Set up and launch the standalone `gbserver` from the `granite.build` repo. Standalone is the **default way to run Granite.build in this environment**: SQLite storage, the **bash** compute backend, no cloud credentials. A running gbserver is a prerequisite for everything else — you cannot create, validate, or run a build without one. If another skill (e.g. `create-build`) needs a server and none is up, run this skill first.

Repo: https://github.com/ibm-granite/granite.build
Clone URL (HTTPS): `https://github.com/ibm-granite/granite.build.git`

## 0. Locate or clone the repo

Find an existing checkout before cloning — do not re-clone a repo that already exists.

1. If `$ARGUMENTS` is set, treat it as the path to an existing `granite.build` checkout. Verify it contains a `Makefile` and `samples/standalone/`.
2. Otherwise look for a checkout in likely spots: the current directory, `./granite.build`, `~/granite.build`. A valid checkout has both `Makefile` and `samples/standalone/standalone-quickstart/build.yaml`.
3. If none is found, clone into `./granite.build`:
   ```
   git clone https://github.com/ibm-granite/granite.build.git
   ```

Let `$REPO` be the resolved repo directory for the rest of these steps. Run all commands from inside `$REPO`.

## 1. Create the virtualenv and install

Requires **Python ≥ 3.13**. Pick an interpreter: prefer `python3.13` if present, otherwise use `python3` when `python3 --version` reports ≥ 3.13. If neither is ≥ 3.13, stop and tell the user to install Python 3.13. Use whichever you found as `PYTHON=` below.

```
make standalone-venv PYTHON=python3.13   # or PYTHON=python3 if that is your >=3.13 interpreter
```

This creates `.venv/` and installs the standalone dependencies (the gbserver/`gb` CLI). If `.venv/` already exists and the `gbserver` entry point is present, you may skip this step.

> This venv is for gbserver and the `gb` CLI **only**. Do NOT install workload/training dependencies (torch, trl, peft, …) into it — each step's script creates and manages its own venv at run time. Keep this venv clean.

## 2. Start the server (it must survive across tool calls)

gbserver is a long-running service. Launch it with the Bash tool's own parameters, not shell backgrounding. Invoke the Bash tool with BOTH set:
- `run_in_background: true` — runs it as a tracked daemon that survives across tool calls and is stopped cleanly when the session ends. (A foreground `gbserver` never returns, so without this the launching call just blocks until it times out.)
- `dangerouslyDisableSandbox: true` — in environments where the Bash tool sandboxes each call in its own network namespace, this runs the server in the shared host namespace so your later calls can reach its port. In some environments (e.g. inside an unprivileged container) calls already share one namespace and this flag is a harmless no-op — set it anyway; it's the safe default either way.

Do NOT use `&` / `nohup` / `setsid` / `disown` — the tool parameters handle lifetime. Command:

```
source .venv/bin/activate && gbserver standalone --port ${GBSERVER_PORT:-8080} --space-dir configurations/spaces/local > /tmp/gbserver.log 2>&1
```

Use port `${GBSERVER_PORT:-8080}`. This host may be shared with other builds, so the environment sets `GBSERVER_PORT` to a per-job value to keep two jobs from colliding on the same port — always honor it; don't hardcode 8080.

**Don't double-start:** if a gbserver for this port is already running, reuse it — do not start a second (it fails "address already in use"). Check by process (the PID namespace is shared, so this works from any call without needing the sandbox flag):

```
pgrep -f "gbserver standalone --port ${GBSERVER_PORT:-8080}" >/dev/null && echo "already running — reuse it"
```

## 3. Confirm it's ready, then report

Poll until the gbserver process is up, then confirm it answers. The process check needs no sandbox flag (shared PID namespace); the readiness `curl` should run **with `dangerouslyDisableSandbox: true`** so it shares the server's namespace:

```
until pgrep -f "gbserver standalone --port ${GBSERVER_PORT:-8080}" >/dev/null; do sleep 0.5; done   # process alive
curl -s -o /dev/null "http://127.0.0.1:${GBSERVER_PORT:-8080}/" && echo "gbserver ready"             # accepting connections
```

Once it's ready, report to the user:
- The repo path used (and whether it was cloned or reused).
- That the server is running and on which port (`${GBSERVER_PORT:-8080}`).
- How to stop it (scoped to THIS job's port via its command line, so it never touches a co-located job's server):
  ```
  pkill -f "gbserver standalone --port ${GBSERVER_PORT:-8080}"
  ```
  If launched as a tracked background task it also stops when this Claude session exits.

If startup fails, show the relevant lines from `/tmp/gbserver.log` rather than guessing. For setup or startup issues, invoke the `gb-docs` skill — it reads the checkout's docs (`docs/getting-started.md`, `docs/help/troubleshooting.md`).

## Steps are picked up dynamically — you do NOT need to restart

Once the server is running, adding or editing a **step** under the space's assets (`configurations/.../environments/bash/steps/<name>/`) is picked up on the next build automatically. Do **not** restart gbserver after authoring or changing a step — it resolves the current on-disk step at build time. Only restart if a change genuinely isn't being seen after you've ruled out everything else (wrong path, wrong step name, syntax error).

## Submitting a build (only if asked)

The **client** needs `GB_ENVIRONMENT=STANDALONE`, and — because the server lives in the shared host namespace — every `gb` command must run in a Bash call **with `dangerouslyDisableSandbox: true`** (otherwise it lands in a fresh sandbox namespace and can't reach the server). Activate the venv and set the env in that same call:

Because the server may be on a non-default port (the environment sets `GBSERVER_PORT`), point the client at it in the same call. The client reads the server base URL from **`GBSERVER_HOST`** (a full `scheme://host:port`; its standalone default is `http://localhost:8080`), so set it to match the port:

```
# each of these is a separate Bash tool call with dangerouslyDisableSandbox: true
source .venv/bin/activate && export GB_ENVIRONMENT=STANDALONE GBSERVER_HOST="http://127.0.0.1:${GBSERVER_PORT:-8080}" && gb build validate -f <path>/build.yaml   # schema + URI check
source .venv/bin/activate && export GB_ENVIRONMENT=STANDALONE GBSERVER_HOST="http://127.0.0.1:${GBSERVER_PORT:-8080}" && gb build start    -f <path>/build.yaml   # prints a build id
source .venv/bin/activate && export GB_ENVIRONMENT=STANDALONE GBSERVER_HOST="http://127.0.0.1:${GBSERVER_PORT:-8080}" && gb build status   <build-id>             # PENDING → RUNNING → SUCCESS/FAILED
```

`GBSERVER_API_KEY` is not needed for localhost — the server allows unauthenticated access from `127.0.0.1`/`::1`. The client defaults to `http://localhost:8080`, so whenever `GBSERVER_PORT` is a non-default port you MUST export `GBSERVER_HOST` to match, as above, or the client can't reach the server.

**If a `gb` command warns the CLI is out of date** (a client/server version mismatch), add **`--skip-version-check`** — a per-command flag, so it goes *after* the subcommand: `gb build start -f <path>/build.yaml --skip-version-check`. The version check is advisory; skipping it is safe here.

### Where a build's real output lives

`gb build status`/`gb build log` mostly show status events, not your script's stdout. The actual stdout/stderr (prints, tracebacks, training progress) is on disk:

```
find ~/.granite.build/workdir/llm-build-<build-id> -name job.log
```

Read that `job.log` to see what the workload actually did. It is the primary debugging artifact when a build fails or "succeeds" without doing anything.

## Server life policy

Shut down the server when it's not in use (no builds running). The server should default to being down unless it's actively being used — start it up again for future builds or any other granite.build task. Manage the server lifecycle yourself; don't assume anything else will clean it up (a background daemon left running keeps holding its port and resources until you stop it — e.g. with `pkill -f "gbserver standalone --port ${GBSERVER_PORT:-8080}"`). If you want to keep it running across several back-to-back tasks you may, but don't leave it up once you're done. No need to ask the user.
