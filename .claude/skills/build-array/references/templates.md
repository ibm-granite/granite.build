# build-array — record & document templates

The artifacts the skill writes into the run directory (`<run-dir>/`), plus the
per-build record schema. `PLAN.md` and `SUMMARY.md` are prose the orchestrator writes;
`run-state.json` is the recovery anchor; `builds.jsonl` is the append-only event log;
`results.json` / `RESULTS.md` are **generated** by folding `builds.jsonl` (and re-querying
gb build state) — never hand-edited.

```
<run-dir>/
├── PLAN.md            # written in Phase 1, approved before any launch
├── run-state.json     # recovery anchor: every point + its build_id, written at submit
├── builds.jsonl       # append-only, one line per build reaching terminal
├── results.json       # canonical record, folded from builds.jsonl
├── RESULTS.md         # human table rendered from results.json
└── SUMMARY.md         # synthesis, written in Phase 4
```

Single writer (the orchestrator), so a single file per role is safe — no per-lane files.
`builds.jsonl` is appended (crash-safe, keeps the trajectory); `run-state.json` is the small
mutable index a resumed session reads first.

---

## `PLAN.md` (Phase 1 — written before execution, user-approved)

```markdown
# Build array: <array-name>

Created: <YYYY-MM-DD HH:MM>   ·   Run dir: <abs path>   ·   gbserver: <dashboard url>

## Intent
<one paragraph: what question this array answers / what batch it runs>

## Template
- Build: <path to build.yaml, or space://steps/<name>>
- Environment: <bash | lsf | skypilot | …>   ← decides where gb submits; see create-build
- Fixed settings: <anything constant across all points>

## Axes
| Axis | Values |
|------|--------|
| <axis-a> | v1, v2, v3 |
| <axis-b> | x, y |

Total points (full cross product): <N>

## Exclusions (pruned up front, with reason)
- <axis-a=v3> — <reason, e.g. "known to diverge (prior sweep)">
- <combo> — <reason, e.g. "OOMs at this model size">

Points after exclusions: <M>

## Analysis grouping (a tag on each point; read at synthesis, not a way to split execution)
- Grouping: <none/flat | fibers | per-item | per-aggregate | per-ablation | by-failure-mode>
- Rationale: <why this fits the matrix shape>
- Compare within a group by: <the swept axis / vs baseline / mean over seeds / …>

## Health check (workload-specific — the built-in flags only check artifact integrity)
- Signal: <none | the log line/metric that means "bad but well-formed", e.g. "train loss NaN or non-monotonic over last 5 steps", "grad-norm > 100", "eval output empty/non-alphanumeric">
- Source: <where to read it — e.g. build_job_log, a metrics file the artifact emits>
- Verdict flag(s): <e.g. diverged, unstable>   ·   Metrics to record: <e.g. final_loss, grad_norm_peak>
- (`none` is valid for a pure pass/fail batch — state it explicitly.)

## Output convention
- Each point writes to: <file:///…/{axis-a}_{axis-b}/>   (must resolve to a fileset with headroom)

## Submit mode
- <fan-out (default) | max-in-flight=K | serial>   — always validate one build to completion first, then send the array (that build also resolves the shared bash venv). Use max-in-flight/serial only when genuinely heavy builds outnumber the host's GPUs. See SKILL.md → Why validate one build before the fan-out.

## Stop conditions
- <e.g. "abort if the first 5 builds all fail"; "cap total retries at 20">

## Sign-off
- [ ] User approved this plan
```

---

## `run-state.json` (Phase 2 — recovery anchor, written before/at submit)

The small mutable index. Every point is listed **before** any build is submitted; each
`build_id` is filled in the instant `build_start` returns it, so a crash mid-submit still
leaves a resumable record. A resumed session reads this, re-queries `build_status` for each
non-terminal `build_id`, and restarts the monitor.

```json
{
  "array": "<array-name>",
  "array_tag": "array:<array-name>",
  "run_dir": "/abs/path",
  "gbserver_port": 8080,
  "submit_mode": "fan-out | max-in-flight=K | serial",
  "canary_baseline": {"build_id": "llm-build-…", "wall_seconds": 68, "device": "cuda"},
  "points": [
    {
      "coord": {"<axis-a>": "v1", "<axis-b>": "x"},
      "group": "<fiber tag or null>",
      "tags": ["array:<array-name>", "group:<fiber>"],
      "params": {"lr": "2e-4", "model": "3b"},
      "output": "/proj/…/v1_x/",
      "build_id": "llm-build-…",
      "prev_build_ids": [],
      "status": "submitted"
    }
  ]
}
```

- `status` here is the last-known coarse state (`pending | submitted | running | success | failed | invalid | cancelled | skipped`). The authoritative per-build detail lives in `builds.jsonl`. **Refresh this on every transition** — including in serial/max-in-flight mode when the monitor isn't the thing advancing the array — or a resumed session reads a stale frontier.
- `canary_baseline` — the first point's terminal wall-time and device, recorded once the canary finishes; the monitor uses `wall_seconds` as the yardstick for `slow` events (a running build past ~2–3× is pathological).
- `prev_build_ids` — build_ids of prior attempts at this point that were cancelled/superseded (e.g. a CPU-fallback build cancelled and resubmitted). Keeps resubmission lineage instead of overwriting `build_id` silently.
- `array_tag` / `points[].tags` — the tags passed to `build_start(tags=…)`. The array-wide `array:<name>` is the monitor's single-request filter (`GET /api/v1/builds/?tag=array:<name>`) and a resume's backstop for rediscovering builds whose `build_id` never reached disk; per-point `group:<value>` carries the analysis grouping in gb's own tag store.

---

## Per-build record — one JSON line per build in `builds.jsonl`

Appended by the orchestrator the moment a build reaches a terminal state. Coordinates
identify the matrix point; flags are computed from artifacts, not copied from status.

```json
{
  "coord": {"<axis-a>": "v1", "<axis-b>": "x"},
  "group": "<fiber tag or null>",
  "build_id": "llm-build-…",
  "status": "success",
  "flags": ["ok"],
  "device": "cuda",
  "submitted_at": "2026-07-27T14:03:11Z",
  "finished_at": "2026-07-27T14:41:52Z",
  "wall_seconds": 68,
  "artifacts": [
    {"id": "adapter", "path": "/proj/…/v1_x/", "bytes": 134217728, "suspect": false}
  ],
  "metrics": {},
  "retries": 0,
  "note": ""
}
```

Field notes:
- `status` — gb terminal state, lowercase: `success | failed | invalid | cancelled` (or `skipped` for an idempotent skip).
- `flags` — computed at record time. Integrity flags: `ok`, `zero_byte_artifact`, `empty_success`, `missing_artifact`; plus any **health-check flags declared in `PLAN.md`** (e.g. `diverged`, `unstable`). **`ok` is exclusive** — it appears iff the flag list is otherwise empty, so a build never carries both `ok` and an anomaly flag (the run produced incoherent `["ok","diverged"]` records; don't). A `success` status with any non-`ok` flag is **not** a trustworthy result. Do not add a parallel top-level boolean (e.g. `diverged: true`) that duplicates a flag — the `flags` list is the single source.
- `device` — where the build actually ran (`cuda` / `cpu` / …), read from the log. Catches silent CPU fallback (SKILL.md → Local-bash arrays) that a status board never shows.
- `wall_seconds` — terminal wall-time; compared against `canary_baseline.wall_seconds` to spot the pathologically-slow build.
- `artifacts[].bytes` / `suspect` — from a `stat` of the registered path; `suspect: true` if 0 bytes or implausibly small. This is what catches the full-disk 0-byte-checkpoint failure that a status board misses.
- `metrics` — free-form, workload-specific object holding the numbers behind the health check and any per-point domain result (e.g. `{"final_loss": 0.28, "grad_norm_peak": 8.1, "eval": {"fact_learned": true, "control_contaminated": false}}`). The *shape* is declared per-array in `PLAN.md`; the skill just carries it. This is the per-point signal a binary pass/fail board can't hold.
- `retries` — count of infrastructure retries used (application failures are not blindly retried).
- `note` — root cause + proposed fix from the investigation subagent, when one ran.

---

## `results.json` (Phase 3 — folded from builds.jsonl)

```json
{
  "array": "<array-name>",
  "generated_at": "2026-07-27T14:50:00Z",
  "totals": {"success": 18, "failed": 3, "skipped": 2, "suspect": 1, "pending": 0, "diverged": 2},
  "builds": [ { …per-build record… }, … ]
}
```

`totals` is **extensible** — beyond the status/integrity counts, add a tally for each declared
health-check flag (e.g. `diverged`) and any array-specific rollup, so the count of bad-but-well-formed
results is as visible as the failure count.

Regenerate on demand by folding `builds.jsonl` (last line wins per `build_id`) — prefer a small
`python3` fold over a multi-clause `jq` pipeline (nested `metrics`/`artifacts` + conditional flags
make `jq` precedence a trap). For a definitive answer or after a resume, re-query `build_status`
for each `build_id` first.

---

## `RESULTS.md` (Phase 3 — rendered from results.json, never typed by hand)

```markdown
# Results: <array-name>   (generated <YYYY-MM-DD HH:MM>)

Totals: 18 success · 3 failed · 2 skipped · **1 suspect · 2 diverged**

| <axis-a> | <axis-b> | group | status | flags | device | key artifact (bytes) | key metric | build_id |
|----------|----------|-------|--------|-------|--------|----------------------|-----------|----------|
| v1 | x | 3b/LoRA | success | ok | cuda | adapter (128 MB) | loss 0.28 | …a1b2 |
| v3 | y | 8b/LoRA | failed  | —  | — | —                    | — | …c3d4 |
| v2 | x | 3b/LoRA | success | **zero_byte_artifact** | cuda | adapter (0 B) | — | …e5f6 |
| v1 | y | 3b/LoRA | success | **diverged** | cuda | adapter (36 MB) | grad-norm ~900 | …7g8h |
```

Sort suspect / failed / health-flagged rows to the top so problems are visible first — a
`diverged` row with a normal-sized artifact is the easiest to miss and the most important to see.

---

## `SUMMARY.md` (Phase 4 — synthesis)

```markdown
# Summary: <array-name>

## Outcomes
<counts; which points succeeded / failed / were skipped>

## Patterns across the array (grouped by the analysis tag)
- <e.g. "lr=5e-4 diverged in 3 of 4 fibers">
- <e.g. "all 30b points OOM'd — model-parallel needed">

## Distrust these (green status, not a real result)
- <point> — zero_byte_artifact (fileset was full when the checkpoint was written)
- <point> — diverged / unstable (well-formed, normal-sized artifact from a bad run — flagged only by the health check, e.g. grad-norm ~900, output collapsed) → <supporting metrics>

## Investigation findings
- <point> — <root cause the subagent found> → <implication / fix>

## Recommended next steps
1. <concrete follow-up>
2. <concrete follow-up>

## Best configuration (if the array was a search)
- <winning point + its metric>
```
