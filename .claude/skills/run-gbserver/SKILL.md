---
name: run-gbserver
description: Ensure the Granite.build standalone gbserver backend is running so the gbmcp build tools work. Use when a build tool reports the backend is unreachable, or when asked to start / set up gbserver.
allowed-tools: mcp__gbmcp__gbserver_status mcp__gbmcp__gbserver_start mcp__gbmcp__gbserver_stop mcp__gbmcp__build_start mcp__gbmcp__build_status mcp__gbmcp__build_log mcp__gbmcp__build_list mcp__gbmcp__build_describe mcp__gbmcp__build_job_log
---

# Ensure the gbserver backend is up

The `mcp__gbmcp__*` tools run as a local stdio process that Claude Code launches, so they're always available. But the build tools (`build_start`, …) call a **separate gbserver backend** over REST — that process is what needs to be running. Manage it entirely through tools:

- **Check:** `gbserver_status()` → `ready: true` means the backend is up; go run builds.
- **Start:** if not ready, `gbserver_start()` — launches `gbserver standalone` and blocks until reachable (idempotent; safe to call anytime). Returns `ready: true`, or an `error` / `log_tail` if it couldn't start.
- **Stop:** `gbserver_stop()` — only when the user asks (see Lifecycle below).

The loop: `gbserver_status` → `gbserver_start` if needed → drive builds. No server to start by hand, and no endpoint to register.

## Lifecycle — a long-lived dev service; leave it running

`gbserver standalone` is more than a build backend — the **same process serves the web dashboard** (`http://127.0.0.1:<port>`, the port from `gbserver_status`) and the REST/analytics API. Treat it like a dev server, not a per-build worker:

- **Leave it running after a build finishes.** The user likely wants to view the build and its result in the dashboard, and the next build reuses the warm server. After starting it, tell the user the dashboard URL so they can browse builds between runs.
- **When you wrap up, hand the stop decision to the user.** Tell them gbserver is **still running** and give the dashboard URL (from `gbserver_status`), and ask them to **let you know when they'd like it stopped**. Don't phrase it as "should I stop it now?" (that frames stopping as expected), and never `gbserver_stop` unprompted — leave it up until they ask.
- It's a detached daemon, so it persists across sessions until stopped or a reboot (fine for a dev service). If the user wants a clean slate, `gbserver_stop`.

## Port already in use

If `gbserver_status` / `gbserver_start` reports the port is held by another (non-gbserver) process, gbserver can't bind it. The port is fixed at MCP-server launch, so it can't change mid-session — the user must **exit and relaunch** with a free port set, e.g. `GBSERVER_PORT=<free> claude`. Don't suggest `! … claude`: the `!` prefix runs *inside* the session, so it can't relaunch it.

## One-time install (prerequisite)

`gbserver_start` launches the `gbserver` that ships alongside gbmcp, so gbmcp must already be installed — one of:
- **pip:** `pip install 'granite.build[standalone]'` (Python ≥3.11; the default 3.9 fails on `sqlite_database`).
- **Repo checkout:** `make standalone-venv PYTHON=python3.13` — builds gbserver + gbmcp from the current source.

If `gbserver_start` returns an error that no gbserver executable was found, this install hasn't happened — tell the user to run one of the above. (`GBSERVER_BIN` overrides the binary; `GBSERVER_PORT` the port.)

## Debugging

- A build's real stdout: `build_job_log(build_id)` — the on-disk `job.log` tail; prefer it over `build_log` / `build_status` (status events only).
- gbserver failed to start: read `gbserver_start`'s `log_tail`, or `/tmp/gbserver-<port>.log`. For setup / troubleshooting docs, use the `gb-docs` skill.
