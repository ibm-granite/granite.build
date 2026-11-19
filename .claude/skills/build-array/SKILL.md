---
name: build-array
description: Run and manage a large array of Granite.build builds — a hyperparameter sweep, a multi-model/multi-dataset fan-out, a per-item pipeline batch, seed-repro runs, an ablation, or a retry/heal campaign. Plans the array with the user, submits every build, monitors them to completion, records each outcome to a durable file that never goes stale, delegates failures to investigation subagents, and synthesizes results into next steps. Use when the user wants to run an array / batch / sweep of builds or tasks (more than a handful at once).
argument-hint: "[array-name]"
allowed-tools: Agent Monitor Read Write Bash(curl *) Bash(jq *) Bash(date *) Bash(mkdir *) Bash(ls *) Bash(cat *) Bash(test *) Bash(python3 *) mcp__gbmcp__gbserver_status mcp__gbmcp__gbserver_start mcp__gbmcp__build_start mcp__gbmcp__build_status mcp__gbmcp__build_log mcp__gbmcp__build_list mcp__gbmcp__build_describe mcp__gbmcp__build_job_log mcp__gbmcp__build_cancel
---

# Orchestrate an array of Granite.build builds

Use this when the work is **not one build but many** — a sweep, a batch, a fan-out. The skill gives that work a repeatable spine: **plan → submit → monitor → record → synthesize**. It does **not** re-teach build authoring — a single build is authored with **`create-build`** (inline `command`) or **`create-step`** (packaged step); this skill is the layer *above* that, which takes a build template + a set of parameter points and runs the whole array well.

Granite.build abstracts **submission** entirely: you call `build_start` and gb runs the build in whatever environment the build.yaml declares (bash, LSF, docker, …) — the skill never submits jobs to a scheduler itself and never reasons about queues or nodes. Its job is *organization*: deciding the array, running it under control, and turning a pile of build outcomes into an answer.

## The model — one orchestrator, one monitor, investigation subagents

- **You are the orchestrator** (main loop). You plan the array with the user, ensure the backend is up, **submit every build yourself** (a `build_start` loop — mechanical and fast, nothing to fan out), record each outcome, and synthesize.
- **A single background monitor** watches all in-flight builds and notifies you as each reaches a terminal state (surfacing failures). You never poll builds in-context — the monitor is a background process, so the poll loop costs no tokens.
- **Subagents are for investigation, not submission or polling.** When a build fails or flags anomalous, delegate a focused root-cause task to a subagent (it has the build_id, the point's params, and log access). That is where a subagent's judgment pays for itself.

### Recovery is on disk, not in context

The builds keep running under gbserver whether or not this session is alive — gb owns them. What a dropped session loses is the *record*, not the builds. So the load-bearing durability primitive is cheap: **write each `build_id` to disk the instant `build_start` returns it.** If CC drops and a fresh session resumes, it reads the run-state file, re-queries `build_status` for each `build_id`, rebuilds the record, and restarts the monitor for anything still in flight. The array-wide `array:<name>` tag is the belt-and-suspenders backstop: a resume can list every build via `GET /api/v1/builds/?tag=array:<name>` even if a `build_id` never made it to disk. Do not rely on in-context memory to resume — after a real drop it is gone.

## Phase 0 — backend up

`gbserver_status()`; if not `ready`, `gbserver_start()` (defer to **`run-gbserver`** for lifecycle, ports, install). The dashboard URL it returns is where the user watches the array live.

## Phase 1 — PLAN (recommend, decide together, write it down)

**Nothing launches in this phase.** Turn the user's intent into a concrete array and record it in `PLAN.md` *before* execution.

1. **Establish the build template — authored elsewhere.** One base build.yaml that every point instantiates. **This skill does not author it:** a build.yaml (targets, steps, inputs, outputs, workload) is authored with **`create-build`** (inline `command`, or a shipped step like `space://steps/lora-finetune`) or **`create-step`**. If none exists, route there first, then return with the working build.yaml text. Make it **parameterized** so you hold one template, not N copies (see **Parameterize once**, below).
2. **Establish the array shape (the matrix).** The axes and their values, plus any **exclusions**. Write axes as named lists; a "point" is one choice from each axis. Recommend sensible defaults and *prune known-bad combos up front* (e.g. a learning rate known to diverge, a config known to OOM) so the array doesn't burn compute rediscovering failures.
3. **Recommend an analysis grouping, let the user decide.** Inspect the matrix shape and propose how results should be *grouped for comparison* (see **Grouping** below) — this is a **tag** recorded on each build, read at synthesis, **not** a way to split execution. State your recommendation and why; the user confirms or overrides. Record the choice in `PLAN.md`.
4. **Declare the health check (workload-specific).** The built-in anomaly flags only verify artifact *integrity*, not whether the result is any good (see **Anomaly flags**). If a "successful" build can still be a bad result — a diverged training run, an eval that scores at the floor, an empty-but-well-formed output — name the concrete signal now: the log line or metric to parse and the threshold that means "distrust this" (e.g. "loss NaN or rising over the last 5 steps", "grad-norm > 100", "eval output empty or non-alphanumeric"). `none` is a valid answer for a pure pass/fail batch — say so explicitly rather than leaving it implicit.
5. **Pick the run directory and write `PLAN.md`.** Default `./<array-name>/` under the user's cwd (or a path they give). Write `PLAN.md` from the template in `references/templates.md`: template, full point list (with resolved params), exclusions with reasons, grouping, the health check from step 4, output-location convention, submit mode (fan-out vs. max-in-flight — see **Local-bash arrays**), and **stop conditions** (e.g. "abort the array if the first N builds all fail").
6. **Get explicit approval** of `PLAN.md` before Phase 2.

### Parameterize once — one template, N submits

The answer to "do I write N build.yamls?" is **no**. Author **one** base build.yaml with each swept knob as a **`$${NAME}`** placeholder, then submit that same text N times — each call carries only that point's overrides:

```python
build_start(file_content=<the one base yaml>, params=["LEARNING_RATE=2e-4", "LORA_RANK=16", "OUT=v1_x"], tags=["array:sweep1", "group:3b_lora"])
```

- **How to pass a parameter — the exact contract.** `params` is a `list[str]`, each entry `"KEY=value"`. gb splits on the **first** `=`; the key and value are each stripped of surrounding whitespace; the value is everything after (spaces and further `=` are allowed inside it). Dotted keys nest into the YAML tree: `config.bash.env.LR=2e-4`. In the template you mark where each value lands with **`$${KEY}`** (double-dollar, single closing brace) for a value and `<% … %>` for logic — delimiters chosen so they don't collide with gb's own `{{ binding }}` syntax.
- **The value is spliced in as raw text, before YAML is parsed — it is not quoted or escaped for you.** Put the quotes in the *template*, around the placeholder, and prefer double quotes: `SUBJECT: "$${SUBJECT}"`. A value carrying a YAML-special character (an apostrophe, a `:`, a leading `#`) that lands in a *single*-quoted or unquoted spot yields invalid YAML and the submit fails (as `"null"`, below).
- **`StrictUndefined`:** a `$${NAME}` with no matching param is a hard error *at submit* — a typo fails fast instead of running blank. There's no `build_validate` in standalone, so **canary the first point** and confirm it starts before submitting the rest.
- **A `build_start` that returns the literal string `"null"` (instead of a build_id) is a *swallowed* render/validate error, not a mechanism failure.** gbmcp calls the client with no error callback, so every internal failure — bad param, template syntax, invalid YAML, unknown space, missing file — collapses to the same `null` with its reason discarded (a gbmcp defect, not just yours). Common causes, in order: an unmatched `$${NAME}` (missing/typo'd param — `StrictUndefined`); **a stray `$${`, `<% %>`, or `{# #}` anywhere in the raw text, including a YAML `#` comment** — the renderer runs *before* YAML, so comments do not hide it, and a comment that documents the syntax as `$${...}` is itself parsed and dies on the `.`; or a value that renders into invalid YAML. This is what the canary is for: it turns a fleet-wide `null` into one legible failure before fan-out.
- You keep the base text in context **once**; each point is a short param list, not a rendered document.
- Give each point a **distinct output** via a param (e.g. `OUT=…`) so artifacts don't collide.
- **Step-agnostic — works with a shipped step or the inline `command` step alike.** Substitution runs on the raw build.yaml *text* before parsing, so it doesn't care which step you use. The clean spot for a swept knob is the same either way: a `config.bash.env` value written as `$${NAME}`, which the step (or the inline command, via `os.environ["NAME"]`) reads as an env var. The `$${…}` delimiter is double-dollar, so it never collides with an inline command's own shell `${VAR}` expansion (`${LLMB_BASH_OUTPUT_DIR:?}` stays literal). But because the renderer runs on the raw text, the param delimiters (`$${`, `<% %>`, `{# #}`) must not appear literally anywhere they aren't a real placeholder — comments and heredoc bodies included (see the `"null"` note above).

*Where* a knob lives in the template is a `create-build` concern; this skill only fills it.

## Phase 2 — SUBMIT (the orchestrator submits every build)

1. Expand the matrix (cross product minus exclusions) into the point list — each point is a coord, its **param list** (the `key=value` overrides, including a distinct `OUT=…`), and its **tags** (an array-wide `array:<name>` plus the point's group tag from Phase 1).
2. **Write the recovery anchor first.** Create `<run-dir>/run-state.json` listing every point (coord + params + output + empty `build_id`) before submitting anything — this is what a resumed session reads.
3. **Idempotent skip:** if a point's declared output already exists **and is non-empty**, mark it `skipped` and don't relaunch. Makes reruns cheap.
4. **Validate one build first — unless the template is already proven.** Before waiting, check: did an equivalent build already complete **this session** (a prior run of this template, even from the `create-build` step that authored it; the same step built in an earlier build; or a resume where its artifacts exist)? If so, the template is proven and — on bash — the shared venv is resolved, so **skip to the fan-out (step 5)** and reuse that build's wall-time and `device` as the baseline. Otherwise submit the first point alone and let it run **to completion**: one clean build proves the template end-to-end — rendered (a build_id, not `"null"`), succeeded, plausible-sized artifact — catching a broken template once instead of fleet-wide. Record its wall-time and `device` in `run-state.json` as the slow/stuck baseline (Phase 2.5); on bash it also resolves the shared venv (see **Why validate one build before the fan-out** below).
5. **Send the rest off — in one batched message.** Fan out the remaining points with a `build_start(file_content=<base yaml>, params=<point's list>, tags=<point's tags>)` per point, issued as **parallel calls in a single message**, not one per turn. Each `build_start` is a synchronous ~3-round-trip call to gbserver (space lookup, static validation, submit), so a sequential loop pays that latency N times end-to-end while batched calls overlap. **Immediately** write each returned `build_id` into its `run-state.json` entry so a crash mid-submit still leaves a resumable record.
6. Once all are submitted, start **one** monitor over the array (Phase 2.5).

**Tag every build natively — it is the array's handle, not just a synthesis label.** `build_start` takes a `tags=` list, and gbserver's list/count endpoints filter by tag (`GET /api/v1/builds/?tag=array:<name>`). Passing `array:<name>` on every submit buys three things a private `group` field in `run-state.json` cannot: (a) the monitor polls the whole array in **one** request per tick instead of one `curl` per `build_id` (Phase 2.5); (b) a resumed session can rediscover every build by tag even if `run-state.json` lost a `build_id` mid-submit; (c) grouping at synthesis reads gb's own tags. Keep recording `group` in the record too, but let the native tag be the load-bearing handle.

### Why validate one build before the fan-out

The rule is one line — **validate one build to completion, then send the array** — but *why* the wait matters depends on the `environment_uri`:

- **Scheduler backends (LSF, Slurm, docker, skypilot):** builds are isolated jobs with no shared local state. The first build is pure validation — prove the template before you queue N. Then fan out; the scheduler handles concurrency.
- **Local bash:** builds share **one unlocked venv per step** (`~/.granite.build/.gb-venvs/<step>`). Fire N at a *cold* step and their `pip install`s race to rewrite the same `site-packages`; a build hitting `torch.cuda.is_available()` mid-rewrite gets a silent `dlopen` failure → trains on **CPU** with the GPU idle (looks hung, since the mamba CPU path is ~1000× slower). The first build **resolves the venv**, so every sibling after it finds `pip install` a no-op — nothing left to race. Hence: wait for it to *finish*, not just start.

**Caveat (resource, not correctness):** if genuinely heavy builds outnumber the host's GPUs they'll contend or OOM. Then cap concurrency with a **max-in-flight window** (submit K, submit the next as each finishes), keeping `run-state.json` fresh on every transition so a resume sees the true frontier. LoRA-scale jobs co-reside fine — fan out.

## Phase 2.5 — MONITOR + INVESTIGATE

Start a single background monitor over the whole array. It emits terminal events (and `slow` events) as notifications while you keep working — and must surface failures, not only successes. **Poll the array's native tag in one request per tick, not one `curl` per `build_id`.** `build_start` tagged every point with `array:<name>` (Phase 2), so a single `GET /api/v1/builds/?tag=array:<name>` returns every build's `uuid` + `status` — one request whether the array is 5 builds or 500. A per-build loop would fire N requests per tick against gbserver, the poll storm this skill exists to avoid.

Two facts the loop depends on, both verified against gbserver: the build's id is the **`uuid`** field of each list entry (not `build_id`), and `status` is a **lowercase** `StrEnum` value (`success`/`failed`/`invalid`/`cancelled` — the same casing the filter uses; ignore any docs showing `RUNNING`). Parse with `python3`, per the same reasoning as the Phase 3 fold — the datetime math for `slow` is a jq trap:

```bash
PORT=${GBSERVER_PORT:-8080}; TAG="array:<array-name>"
# SLOW_SECS ≈ 2.5× the canary baseline wall_seconds (run-state.json → canary_baseline).
python3 - "http://127.0.0.1:$PORT/api/v1/builds/" "$TAG" "<SLOW_SECS>" <<'PY'
import json, sys, time, urllib.parse, urllib.request
from datetime import datetime, timezone
base, tag, slow = sys.argv[1], sys.argv[2], float(sys.argv[3])
TERMINAL = {"success", "failed", "invalid", "cancelled"}
url = base + "?" + urllib.parse.urlencode({"tag": tag})
seen, flagged = set(), set()
while True:
    try:
        builds = json.loads(urllib.request.urlopen(url, timeout=10).read()).get("builds", [])
    except Exception:
        time.sleep(30); continue          # transient error must not kill the monitor
    now = datetime.now(timezone.utc)
    for b in builds:
        bid, st = b.get("uuid"), (b.get("status") or "").lower()
        if st in TERMINAL:
            if bid not in seen:
                seen.add(bid); print(f"terminal {bid} {st}", flush=True)
        elif bid not in flagged:          # alive but maybe pathological → slow
            try:
                started = datetime.fromisoformat((b.get("created_time") or "").replace("Z", "+00:00"))
                if (now - started).total_seconds() > slow:
                    flagged.add(bid); print(f"slow {bid} {int((now-started).total_seconds())}s", flush=True)
            except Exception:
                pass
    # exit only once the array is non-empty AND every build is terminal
    if builds and all((b.get("status") or "").lower() in TERMINAL for b in builds):
        print("all-terminal", flush=True); break
    time.sleep(30)
PY
```

Run it as a **Monitor** (or a `run_in_background` bash loop) so events arrive as notifications while you keep working. The `slow` branch is **not** optional: a terminal-only monitor is blind to a build that is *alive but pathological* (e.g. silently training on CPU past ~2–3× the canary baseline), which then gives no signal until it finishes or hangs forever. **For a very large array, print one *summary* line per tick** (the batch of newly-terminal `uuid`s) rather than one line per build, so a burst of near-simultaneous completions stays under the Monitor event-rate cap. As each event arrives:

1. Read the outcome with `build_status(build_id)` and `build_describe(build_id)`; compute the **anomaly flags** (below).
2. **Append one JSON line** to `<run-dir>/builds.jsonl` (schema in `references/templates.md`) and update the point's status in `run-state.json`. Appending — not rewriting — is crash-safe and preserves the trajectory for synthesis.
3. **On `failed`, a non-`ok` flag, a failed PLAN health check, or a `slow` event, investigate — and pull evidence *before* hypothesizing.** First read `build_job_log` (and device/utilization: is it actually on the GPU? is the GPU busy?), *then* form a theory — do not assert "contention" or "a race" from status alone (in the run, two wrong guesses preceded the first look at the log). For anything beyond a trivial read, spawn an investigation subagent (Agent tool) with the build_id, the point's params, and log access (`build_log`, `build_job_log`); its job is root cause + proposed fix (bad hyperparameter, OOM → needs other parallelism, full fileset → 0-byte artifact, cold-venv CPU fallback → warm-then-fan-out, transient infra → safe to retry). Fold the finding into the record's `note`. For a clear transient-infra failure, retry once or twice; do **not** blindly retry application failures.
4. **Enforce the plan's stop conditions.** When one trips (e.g. "first N builds all failed"), stop launching new points and `build_cancel(build_id)` the still-running ones before surfacing the abort to the user — don't let a doomed array burn the rest of its compute.

### Anomaly flags (compute at record time — this is where the value is)

A build's status flag alone lies: Granite.build (and the guardian sweeps that motivated this skill) can report **`success` while writing a 0-byte artifact** when a fileset is full. So each record carries computed flags, not just the raw status. The **built-in flags check artifact integrity only** — is a well-formed, plausibly-sized artifact actually on disk:

- `zero_byte_artifact` — a registered artifact path exists but is 0 bytes (classic full-disk checkpoint).
- `empty_success` — status `success` but no artifact registered / output dir empty.
- `missing_artifact` — an output declared in the target was never bound.
- `ok` — terminal `success` with a non-empty artifact of plausible size **and no other flag set** (`ok` is *exclusive*: it never co-occurs with an anomaly flag).

**These flags do not judge whether the artifact is any *good*.** A training run that diverges writes a normal-sized, well-formed, useless adapter and passes every integrity check as `ok` — this is the single most important failure mode for a training sweep, and artifact flags are blind to it. (Observed: two diverged LoRA adapters — grad-norm ~900, output collapsed to comma spam — were normal-sized and flagged `ok`; the divergence was caught only by hand.) **Quality/divergence is workload-specific, so the skill cannot hard-code it — the array declares its own health check in `PLAN.md`** (Phase 1): the log line or metric that signals a bad-but-well-formed result (loss NaN or non-monotonic, grad-norm spike, garbled/empty eval output, an accuracy floor). Compute that check at record time alongside the integrity flags, add its verdict to the point's `flags` (e.g. `diverged`, `unstable`), and record the supporting numbers in the record's `metrics` object. A failed health check fires the same investigation trigger as a failed integrity flag (Phase 2.5 step 3).

These flags — integrity **and** the declared health check — are the raw material for synthesis and the thing a bare status board (like a hand-kept tracker) can never give you.

## Phase 3 — RECORD (durable, generated, never hand-typed)

Fold `builds.jsonl` into the canonical record and regenerate the human view. The record is **generated from gb's actual build state** every time — never hand-edited — so it cannot drift the way a manually maintained tracker does.

- `<run-dir>/results.json` — canonical machine record: one entry per point (coordinates, build_id, status, timestamps, artifacts + sizes, flags, retries, investigation note).
- `<run-dir>/RESULTS.md` — human table rendered *from* `results.json` (status + key flags per point). Regenerate it, don't type it.

Regenerate both any time the user asks "where are we" mid-run — fold whatever `builds.jsonl` currently holds. For a definitive answer, or after a resume, re-query `build_status` for each `build_id` in `run-state.json` first.

**Fold with `python3`, not a multi-clause `jq` pipeline.** The records carry nested objects (`artifacts`, `metrics`) and conditional flags; `jq` precedence around `|`, `//`, and comma-args is a trap that silently mis-sorts or drops fields (hit in the run). A small `python3` script reading `builds.jsonl` line-by-line is clearer and safer.

## Phase 4 — SYNTHESIZE (the answer, not just a status board)

When the array is terminal, read `results.json` and write `<run-dir>/SUMMARY.md` (template in `references/templates.md`):

- **Outcomes:** counts by status; which points succeeded/failed/were skipped.
- **Patterns across the array:** e.g. "learning rate 5e-4 diverged in N of M points," "every 30b point OOM'd," "results collapse after epoch 1." Group by the analysis tag from Phase 1 so comparisons are clean. This is exactly what per-build monitoring can't see and central synthesis can.
- **Anomalies to distrust:** every integrity flag (`zero_byte_artifact` / `empty_success` / `missing_artifact`) **and every failed health check** (`diverged` / `unstable` / whatever the array declared) — call these out explicitly with their supporting `metrics`. A green status on any of these is not a real result; a normal-sized artifact from a diverged run is the most dangerous, because nothing but the health check flags it.
- **Investigation findings:** the root causes the subagents found, and what each implies.
- **Recommended next steps:** concrete follow-ups (re-run the collapsed points at the good epoch; the OOM points need the other parallelism strategy; the winning config is X — promote it).

**Treat a single-seed result as provisional, and say so.** A win you are about to *recommend*, or an anomaly you are about to *flag* as real (a run that diverged, contaminated a control, or OOM'd non-deterministically), rests on one sample — one run cannot separate a real effect from seed variance, and a non-monotonic pattern across the matrix ("breaks at these two cells but not their neighbours") is the classic tell of seed noise rather than the swept axis. Don't silently promote or condemn it — but don't quietly launch more runs either. Name the candidate and the anomaly as **provisional** and **suggest to the user** a seed-repeat of exactly those points (the *per-aggregate seed grouping* in **Grouping** — fix every axis, vary only the seed, compare mean/variance), then let them decide whether to run it before you conclude.

Then hand the stop decision for gbserver back to the user (per **`run-gbserver`** — leave it running for the dashboard).

## Grouping — an analysis tag, not a way to split execution

Every build is submitted by the one orchestrator and watched by the one monitor regardless of grouping. Grouping only decides **how results are compared at synthesis** — so record it as a `group:<value>` tag on each point (via `build_start(tags=…)`, alongside the array-wide `array:<name>` tag) and use it in Phase 4.

| Situation | Recommended grouping | Compare within a group by |
|---|---|---|
| No comparison axis (multi-model / multi-dataset eval fan-out, data shards) | none (flat) | n/a — just tally outcomes |
| A swept "question" axis (LR, epochs) with per-group winners wanted | **Fibers** — fix the structural axes, sweep the question axis inside a group | the swept axis |
| Each point is a multi-step chain (tokenize→train→eval) | per-item | the item's stages |
| An axis you aggregate over (seeds) | per aggregate group | mean/variance over seeds |
| Vary one component from a baseline | per ablation dimension | vs. the baseline |
| Re-running prior failures | by failure mode | the applied fix |

**Beware the matrix trap.** "Group by a hyperparameter" is ambiguous. Grouping by one *value* of an axis (a hyperplane — "everything at lr=2e-4") mixes unrelated points and yields no conclusion. Grouping by *all axes except the swept one* (a fiber — "gc/3b/LoRA, vary lr") yields a clean comparison. Prefer fibers when the intent is analytical.

## Resource-mindfulness

- **The orchestrator submits directly; no submission or polling subagents.** Subagents are spawned only to investigate failures — where judgment adds value.
- **One monitor for the whole array** — not one per build, and **one request per tick** (list-by-tag), not one `curl` per `build_id`. Parallel watchers buy nothing because builds already run concurrently under gbserver, and per-build polling scales the request count with the array size.
- **Idempotent skip** (Phase 2) avoids recomputing points that already have valid output.
- **Poll, never spin or tail** — the single background monitor is the only wait; 30 s is a sane default interval. Never poll a build in-context.
- **Route heavy work to the right environment in the build.yaml, not here.** Large fan-outs and multi-node compute belong on an LSF/skypilot environment (gb handles submission and packing); the bash environment runs builds as local processes on **one** host, so a bash array is bounded by that host's cores/GPUs and must follow the rule in **Why validate one build before the fan-out** (validate one build to completion, which also resolves the shared venv; cap concurrency only for heavy builds). This is a property of the **template's** `environment_uri`, decided in `create-build` — this skill just runs whatever the template declares.

## When unsure

- Authoring/parameterizing the single build → **`create-build`** (inline) or **`create-step`** (packaged).
- Backend lifecycle, ports, install → **`run-gbserver`**.
- build.yaml schema / field questions → **`gb-docs`**.
- Exact record/plan/summary skeletons and the JSON schema → **`references/templates.md`** bundled with this skill.
