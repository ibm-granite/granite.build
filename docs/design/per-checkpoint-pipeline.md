# Design: Per-Checkpoint Eval+Extract Pipeline

## Problem Statement

The `gbansible/watch_and_eval_bluevela.sh` script implements a per-checkpoint pipeline:
- Training produces checkpoints incrementally (step_2500, step_5000, step_7500...)
- For each checkpoint: launch N parallel evals → wait for all evals → run export/extract
- Multiple checkpoint pipelines can run concurrently (throttled by `--max-concurrent`)

## Current Architecture (Updated June 2026)

The core binding propagation mechanism **already supports one output triggering multiple downstream dispatches**:

1. Each `NEWARTIFACT_IN_ENVIRONMENT_EVENT` processed by `BuildRun._process_event()` independently calls `pushasset()` → `__propagate_binding()` → `__dispatch_target()`
2. `__propagate_binding()` calls `__dispatch_target()` every time it's invoked — there is **no guard** preventing a target from being dispatched multiple times
3. Each dispatch creates a **new `TargetRun`** instance — so N artifact events produce N independent eval runs

**For `env://` outputs (no push needed):**
- `pushasset_envstore` returns immediately (no-op)
- `__propagate_binding` fires inside the same `_process_event` task
- Downstream targets are dispatched **as soon as each artifact event arrives** during training

**Prerequisites for this to work:**
1. The environment must have `space://assetstores/env-local` configured (so `pushasset` can route `env://` URIs)
2. The step's monitor must emit `NEWARTIFACT_IN_ENVIRONMENT_EVENT` during execution (real-time log streaming — implemented via `SkyPilotLogStreamSource`)
3. The output URI template `env://{{ binding.path }}` must be resolvable with the artifact's binding data

**What still can't be expressed:**
1. Per-artifact scoping — if evals for checkpoint N and checkpoint N+1 run concurrently, there's no way to say "extract target waits for evals of *its specific* checkpoint"
2. Throttling — no `max_concurrent` limit on how many downstream instances run in parallel
3. Aggregate — no way to collect results from all per-checkpoint evals into a single summary target

## Binding Propagation Flow (per artifact event)

```
Training step emits "Special tokens file saved in /output/.../epoch_hf_0/..."
    │
    ▼
SkyPilotLogStreamSource (real-time) → LogFileMonitor → get_events_from_log_line()
    │
    ▼
NEWARTIFACT_IN_ENVIRONMENT_EVENT(binding_id="checkpoint", binding={path: "/output/.../epoch_hf_0"})
    │
    ▼
BuildRun._process_event()
    ├── dispatch_event() → buildrunner (stores event, ignores for propagation)
    └── pushasset(uri="env://{{ binding.path }}", binding={path: "/output/.../epoch_hf_0"})
        │   fill_template → "env:///output/.../epoch_hf_0"
        │   route to pushasset_envstore → returns URI immediately (no-op)
        │
        └── __propagate_binding(binding_info, [uri])
            ├── marks binding.available = True, appends URI
            └── for each downstream target with all inputs satisfied:
                └── __dispatch_target() → creates new TargetRun → launches eval
```

Each subsequent artifact (epoch_hf_1, epoch_hf_2) repeats this flow independently, dispatching a new eval instance each time.

## Design Options

### Option A: Sub-Build Per Checkpoint (Recommended)

**Concept:** Training emits checkpoint events as they complete. A "checkpoint handler" in the build framework spawns a new child build (eval suite + extract) for each checkpoint. Each child build is independent and runs the full eval+extract pipeline.

**Build YAML syntax:**
```yaml
granite.build:
  name: $${NAME}
  targets:
    sft-training:
      environment_uri: space://environments/skypilot/aws
      steps:
        - step_uri: space://steps/openinstruct-sft
      outputs:
        checkpoint:
          type: dataset
          # NEW: marks this output as streaming (multiple artifacts over time)
          streaming: true

    # NEW: on_each block — defines a sub-pipeline triggered per artifact
    on_each:
      source: sft-training.checkpoint
      max_concurrent: 2
      build_template: samples/standalone/run-all-evals/build.yaml
      params:
        NAME: "$${NAME}-{{ artifact.basename }}"
        MODEL_S3: "{{ artifact.path }}"
      # Optional: run another template after evals complete
      then:
        build_template: samples/standalone/extract/build.yaml
        params:
          NAME: "$${NAME}-{{ artifact.basename }}"
          EVAL_RESULTS: "{{ outputs.eval_results }}"
```

**Implementation:**
- New `on_each` top-level key in build.yaml (parallel to `targets`)
- BuildRun gets a new event handler: when a `streaming: true` output emits an artifact, instead of propagating via bindings, it triggers child build creation
- Child builds are tracked in BuildRun state, counted toward build completion
- `max_concurrent` throttles how many checkpoint pipelines run simultaneously
- Child build failure doesn't fail the parent (training continues)

**Pros:**
- Clean separation: child builds are independent (own targets, own cleanup)
- Reuses existing build templates (run-all-evals, extract)
- Natural throttling via `max_concurrent`
- Parent build tracks child builds and reports aggregate status
- Easy to implement incrementally (first: just spawn child builds; later: add `then` chaining)

**Cons:**
- New concept (sub-builds) — adds complexity to the build model
- Child build lifecycle management (cancel parent = cancel all children?)
- Serialization/storage: many builds for one training run

---

### Option B: `for_each` Target Mode

**Concept:** A target attribute `for_each: true` makes it re-trigger for every new artifact from its input binding. Each instance gets a scoped run with the specific artifact URI.

**Build YAML syntax:**
```yaml
granite.build:
  name: $${NAME}
  targets:
    sft-training:
      outputs:
        checkpoint:
          type: dataset
          streaming: true

    eval-suite:
      for_each: true          # NEW: re-trigger per artifact
      max_concurrent: 2       # NEW: throttle parallel instances
      inputs:
        model_checkpoint:
          binding: sft-training.checkpoint
      steps:
        - step_uri: space://steps/sage-eval-multilingual-grouped
      outputs:
        eval_results:
          type: dataset

    extract:
      for_each: true
      inputs:
        results:
          binding: eval-suite.eval_results
          # NEW: scope matching — only trigger when results come from
          # the same checkpoint instance that produced this artifact
          scope: per_instance
```

**Implementation:**
- `for_each: true` on a target makes `__propagate_binding()` dispatch a new TargetRun each time (not just the first time)
- Each TargetRun instance is tagged with a `scope_id` (e.g., the artifact URI hash)
- Downstream `for_each` targets with `scope: per_instance` only trigger when their input comes from the same scope
- `max_concurrent` limits parallel TargetRun instances

**Pros:**
- Stays within the existing build model (no sub-builds)
- Familiar target/step semantics
- Scoping naturally chains: training → eval → extract all share a checkpoint scope

**Cons:**
- Significant changes to BuildRun's propagation logic (currently assumes each target runs once)
- Scoping semantics are complex (what if eval-suite has 22 sub-targets? how does extract know when all 22 finished for one checkpoint?)
- Harder to express the "eval fan-out + merge" pattern within a single target

---

### Option C: Event-Driven Trigger with Checkpoint Groups

**Concept:** Keep targets running once, but add a "checkpoint group" that dynamically creates target instances based on events. The group acts as a template that's instantiated per checkpoint.

**Build YAML syntax:**
```yaml
granite.build:
  name: $${NAME}
  targets:
    sft-training:
      outputs:
        checkpoint:
          type: dataset
          streaming: true

  # NEW: checkpoint_groups define per-artifact pipelines
  checkpoint_groups:
    eval-pipeline:
      trigger: sft-training.checkpoint
      max_concurrent: 2
      targets:
        multilingual-evals:
          inputs:
            model: "{{ trigger.path }}"
          steps:
            - step_uri: space://steps/sage-eval-multilingual-grouped
        bcb-eval:
          inputs:
            model: "{{ trigger.path }}"
          steps:
            - step_uri: space://steps/sage-eval-bcb
        extract:
          inputs:
            ml_results:
              binding: multilingual-evals.results
            bcb_results:
              binding: bcb-eval.results
          steps:
            - step_uri: space://steps/extract-results
```

**Implementation:**
- `checkpoint_groups` defines a template DAG that gets instantiated per trigger artifact
- Each instance is a mini-build with its own target runs, scoped to one checkpoint
- The group's internal targets use standard binding semantics (extract waits for both evals)
- `max_concurrent` at the group level throttles checkpoint parallelism

**Pros:**
- Cleanest expression of the per-checkpoint pipeline pattern
- Internal targets use normal binding semantics (fan-out + merge works naturally)
- No changes to core target execution logic — groups are syntactic sugar over sub-builds

**Cons:**
- New top-level concept (checkpoint_groups)
- Overlaps with Option A's sub-build approach
- More complex YAML schema

---

## Comparison Matrix

| Aspect | Option A (Sub-Build) | Option B (for_each) | Option C (Groups) |
|--------|---------------------|--------------------|--------------------|
| Implementation complexity | Medium | High | Medium |
| Changes to core BuildRun | Small (add child build tracking) | Large (re-trigger logic, scoping) | Small (template instantiation) |
| Reuses existing build templates | Yes | No | Partially |
| Fan-out + merge per checkpoint | Natural (child build has its own DAG) | Complex (needs scope matching) | Natural (group-internal DAG) |
| Throttling | Simple (max_concurrent children) | Complex (per-target instances) | Simple (max_concurrent groups) |
| Cancel semantics | Clear (cancel parent = cancel children) | Unclear (cancel one instance?) | Clear (cancel group = cancel instances) |
| Monitoring/UI | Each child is a build (existing UI works) | Many TargetRuns per target (UI changes) | Need new group view |

## Recommendation

**Option A (Sub-Build Per Checkpoint)** for v1:
- Lowest implementation risk — child builds reuse 100% of existing infrastructure
- Training step modification is minimal (emit one `LLMB_ARTIFACT_ID` per checkpoint instead of only the final one)
- Existing `run-all-evals/build.yaml` template works unchanged as the child build
- Cancel semantics are straightforward
- Monitoring: `gb build list` shows parent + children naturally

**Future enhancement:** Option C (checkpoint_groups) as syntactic sugar over sub-builds once the pattern proves out.

## Prerequisites (Independent of Option Chosen)

1. **Training step must emit checkpoint artifacts as they happen** — modify the `run:` script in `openinstruct-sft/step.yaml` to emit `LLMB_ARTIFACT_ID:checkpoint` after each `checkpointing_steps` save, not just at the end.

2. **The `streaming: true` output attribute** — tells the framework this binding will produce multiple artifacts over time (semantically: "don't wait for training to finish before processing downstream").

3. **Monitor must parse artifacts during execution** — currently the skypilot monitor polls job logs periodically. For streaming artifacts, the monitor must forward NEWARTIFACT events while the job is still running (it already does this for WORKLOAD_STATUS_EVENT, so the infrastructure exists).

## Confirmed Approach: Option A with Aggregate Target

### Execution Model

```
┌─────────────────────────────────────────────────────────────────┐
│ Parent Build                                                     │
│                                                                  │
│  ┌──────────────┐     streaming: true                           │
│  │ sft-training │──── checkpoint artifacts ──┐                  │
│  └──────────────┘                            │                  │
│                                              ▼                  │
│                                      ┌─────────────┐           │
│                                      │   on_each   │           │
│                                      └──────┬──────┘           │
│                              ┌───────────────┼───────────────┐  │
│                              ▼               ▼               ▼  │
│                     ┌─────────────┐  ┌─────────────┐  ┌──────┐ │
│                     │ Child Build │  │ Child Build │  │ ...  │ │
│                     │ (ckpt 2500) │  │ (ckpt 5000) │  │      │ │
│                     │             │  │             │  │      │ │
│                     │ eval→extract│  │ eval→extract│  │      │ │
│                     └──────┬──────┘  └──────┬──────┘  └──┬───┘ │
│                            │               │             │      │
│                            └───────────────┼─────────────┘      │
│                                            ▼                    │
│                                   ┌────────────────┐            │
│                                   │ aggregate-report│            │
│                                   │ (runs once after│            │
│                                   │  all children)  │            │
│                                   └────────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### Build YAML (Full Example)

```yaml
granite.build:
  name: $${NAME}
  targets:
    sft-training:
      environment_uri: space://environments/skypilot/aws
      steps:
        - step_uri: space://steps/openinstruct-sft
      outputs:
        checkpoint:
          type: dataset
          streaming: true   # emits multiple artifacts over time

  on_each:
    source: sft-training.checkpoint
    max_concurrent: 2
    build_template: samples/standalone/run-all-evals-and-extract/build.yaml
    params:
      NAME: "$${NAME}-{{ artifact.basename }}"
      MODEL_S3: "{{ artifact.path }}"

  # Optional: runs once after ALL child builds complete
  targets:
    aggregate-report:
      inputs:
        all_results:
          binding: on_each.results   # collected from all children
      steps:
        - step_uri: space://steps/aggregate-scores
```

### Child Build Template (self-contained per checkpoint)

```yaml
# samples/standalone/run-all-evals-and-extract/build.yaml
granite.build:
  name: $${NAME}
  targets:
    multilingual-evals:
      environment_uri: space://environments/skypilot/aws
      inputs:
        model: { uri: "$${MODEL_S3}" }
      steps:
        - step_uri: space://steps/sage-eval-multilingual-grouped
      outputs:
        eval_results: { type: dataset }

    bcb-eval:
      environment_uri: space://environments/skypilot/aws
      inputs:
        model: { uri: "$${MODEL_S3}" }
      steps:
        - step_uri: space://steps/sage-eval-bcb
      outputs:
        eval_results: { type: dataset }

    # Fan-in: waits for all evals in THIS child to finish
    extract:
      inputs:
        ml_results: { binding: multilingual-evals.eval_results }
        bcb_results: { binding: bcb-eval.eval_results }
      steps:
        - step_uri: space://steps/extract-results
      outputs:
        results: { type: dataset }
```

### Semantics

| Concept | Behavior |
|---------|----------|
| `streaming: true` | Output produces multiple artifacts over time; each triggers `on_each` |
| `on_each` | Spawns one child build per artifact; child is fully independent |
| `max_concurrent` | Limits how many child builds run in parallel |
| `on_each.results` | Aggregated output binding; becomes available when ALL children complete |
| Child build failure | Logged but does not fail parent (training continues) |
| Parent cancel | Cancels training + all active child builds |
| Aggregate target | Standard binding semantics — waits for `on_each.results` to be available |

### Implementation Plan

**Phase 1: Streaming artifact emission**
- Modify `openinstruct-sft/step.yaml` to emit `LLMB_ARTIFACT_ID:checkpoint` after each save (not just final)
- Add `streaming: true` to `BuildTargetOutputConfig` schema
- Verify skypilot monitor forwards NEWARTIFACT events while job is still running

**Phase 2: `on_each` handler**
- Add `on_each` schema to `BuildConfig` in `buildconfig.py`
- In `BuildRun.__propagate_binding()`: if binding has `streaming: true` and `on_each` is configured, spawn child build instead of triggering downstream targets
- Child build creation: reuse `BuildRunner.start_and_wait()` with template + params
- Track child builds in `BuildRun.child_builds: Dict[str, BuildRun]`

**Phase 3: Aggregate binding**
- When all child builds complete: collect their output artifact URIs
- Mark `on_each.results` binding as available with the aggregated URI list
- Dispatch any targets waiting on that binding (e.g., `aggregate-report`)

**Phase 4: Lifecycle management**
- Parent cancel → cancel all active child builds (reuse existing cancel mechanism)
- Child build failure → log warning, mark that checkpoint as failed, continue
- `gb build list` shows parent with child build IDs
- `gb build logs <parent-id>` shows aggregate; `gb build logs <child-id>` shows per-checkpoint

## Key Code Paths

- Build YAML parsing: `src/gbserver/types/buildconfig.py`
- Target dispatch & binding propagation: `src/gbserver/build/buildrun.py` (lines 259-289, 373-429)
- Artifact event processing: `src/gbserver/build/buildrun.py` (lines 556-697)
- Target input tracking: `src/gbserver/build/target.py` (`BindingInfo` dataclass)
- Event log parsing: `src/gbserver/environment/environment.py` (`get_events_from_log_line`)
- Existing multi-target build example: `recipes/granite4-350m/lsf/sft-eval-full-dataset/build.yaml`
