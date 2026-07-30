---
name: create-build
description: Use a new Granite.build build.yaml for any compute workload — model training (SFT/LoRA/EPT), inference/serving, data generation/processing, evaluation, or arbitrary compute. Targets, inputs/outputs (artifacts), compute resources, and how they resolve within a space. The workload runs inline in the build.yaml via the built-in `command` step (no custom step directory to author). Use when asked to create, write, or scaffold a new build, build.yaml, or pipeline.
argument-hint: "[build-name]"
allowed-tools: Bash(ls *) Bash(test *) Bash(cat *) Bash(grep *) Bash(find *) Bash(curl *) mcp__gbmcp__gbserver_status mcp__gbmcp__gbserver_start mcp__gbmcp__build_start mcp__gbmcp__build_status mcp__gbmcp__build_log mcp__gbmcp__build_list mcp__gbmcp__build_describe mcp__gbmcp__build_job_log
---

# Author a Granite.build build

This skill makes a workload (train a model, run inference, generate/process data, evaluate, or any script) run on Granite.build. In the standalone bash environment you produce **one** artifact:

- A **build.yaml** — the run definition. It references the built-in **`command`** step by URI, wires inputs/outputs, passes per-run settings, and carries the workload itself **inline** in the command step's `config.command_config.command`. For anything more than a one-liner, the command writes its own script via a shell **heredoc** and runs it. Keep the build.yaml with the user's project (e.g. a `lora-granite/build.yaml` folder).

You do **not** author a step directory (`step.yaml` + `bash_scripts/`) for the common case — the generic `command` step already exists and runs whatever command you give it. This keeps the whole workload **self-contained in one file**.

Granite.build is **workload-agnostic**: training, inference, data-gen, and evaluation are all "a command that runs." What changes between workloads is the command string and the settings in build.yaml — not the structure.

> **Scope — this skill covers ONLY the inline `command` + heredoc approach, and delegates everything more complex to `create-step`.** The heredoc approach is **bash-specific** (it relies on the bash environment's runtime contract — see below) and is **not portable across environments**: Docker/LSF/skypilot provision Python + dependencies and wire inputs differently. For the standalone-bash default it's the simplest, fastest path. The moment you need a *reusable, packaged, or owned-code* step — its own `step.yaml` + `bash_scripts/`, referenced from the build by a **`file://`** URI — stop and use the **`create-step`** skill. This skill does not describe step authoring at all; it routes you to an existing step (below) or hands the case off to `create-step`.

## First: is there already a step for your workload? (prefer it over inline)

The standalone space ships **tested, purpose-built steps**. If one matches your workload, **use it** — it's battle-tested, and reinventing it inline is slower and more error-prone. Check these **before** reaching for a heredoc:

| `space://steps/<name>` | What it does | Inputs → outputs (key `config.bash.env`) |
|---|---|---|
| `inference` | single-prompt generation with any causal LM | `model` → `generation` (`PROMPT`, `MAX_NEW_TOKENS`) |
| `lora-finetune` | LoRA supervised fine-tune | `model` (+ optional `dataset`) → `adapter` (`MAX_STEPS`, `LEARNING_RATE`, `LORA_RANK`, …) |
| `inference-lora` | generation from a base model + a LoRA adapter | `model` + `adapter` → `generation` |
| `command` | run an arbitrary shell command (this skill's heredoc uses this) | whatever the command emits |
| `hello` | minimal echo — smoke/reference | — |

**Decision rule — in this order:**
1. **A shipped step fits (inference, LoRA fine-tune, adapter inference, …)? → use it.** Wire `space://steps/<name>` into your build.yaml and set its inputs/outputs + per-run knobs in `config.bash.env`. **Do not re-implement it inline** — e.g. a LoRA fine-tune is `space://steps/lora-finetune`, not a hand-written heredoc.
2. **Own reusable/multi-file code with no shipped step? →** author one with the **`create-step`** skill (`file://` URI).
3. **A genuinely one-off, self-contained workload with no shipped step? →** the inline `command` + heredoc below.

**Full contracts + runnable examples are bundled with this skill:** read **`references/steps.md`** for each shipped step's inputs/outputs/env and the decision order, and **`references/samples/`** for known-good build.yamls (`quickstart.build.yaml`, `inference.build.yaml`, `lora-finetune.build.yaml` — the last a two-target train→infer pipeline). Consult the **`gb-docs`** skill for the authoritative build.yaml schema.

## Operating assumptions (read first)

These reflect how this environment actually behaves. Follow them unless the user says otherwise.

- **Default to standalone.** Assume `GB_ENVIRONMENT=STANDALONE`, the **bash** compute backend, and SQLite. No cloud creds, no Kubernetes. Everything below targets that backend.
- **All build actions go through the `mcp__gbmcp__*` tools, not the `gb` CLI.** The tools are always attached; if a build call reports the backend is unreachable, `gbserver_start()` brings it up (see **`run-gbserver`**).
- **The workload lives inline in the build.yaml.** Put the command (and any script it needs, via a heredoc) in the `command` step's `config.command_config.command`. Do **not** create a step directory or a separate script file for the common case.
- **Compose with targets as the workload needs.** A workload is one or more **targets**, each with one or more **steps** (here, `command` steps). Steps within a target run sequentially and share a filesystem; targets can be wired with `binding` (a downstream target's input bound to an upstream target's output). A single target with one `command` step is the simplest shape.
- **The command sets up its own runtime.** The bash env hands the command a clean environment (no PATH/HOME). The command establishes its own PATH and, if it needs Python deps (torch/trl/peft/…), builds its own venv inline (idempotently) before importing them — see "The command step + heredoc."
- **build.yaml edits are read per build.** There's no step to register — a new/edited build.yaml is picked up on the next `build_start`. No restart needed.

## Before you start

Check the backend with `gbserver_status()`; if it isn't `ready`, call `gbserver_start()` (see **`run-gbserver`**). In standalone **every bundled tool is available** — nothing is pruned to work around.

## Steps and URIs (use the references, don't invent)

A build.yaml refers to steps and environments by URI — you don't inspect anything on disk to find them. The shipped steps and their exact contracts live in **`references/steps.md`**, with runnable build.yamls in **`references/samples/`**. Use those names.

- **`space://steps/command`** — the generic command step this skill's heredoc approach uses; the only step you need for the inline approach.
- **`space://steps/<name>`** — purpose-built steps (`inference`, `lora-finetune`, `inference-lora`, `hello`); see `references/steps.md` for each one's inputs/outputs/env.
- **`space://environments/<name>`** — default to `bash` in standalone.

Use the exact names from `references/steps.md`; don't invent URIs.

## The `command` step + heredoc (the core skill)

The built-in **`command`** step runs an arbitrary shell command supplied in `config.command_config.command`, and its exit status becomes the step's status (so `command: "exit 1"` hard-FAILS the target). It already carries the artifact-capture monitor. You author **nothing** on disk — you just fill in the command.

For a Python workload, the command is a small shell script that (1) sets up PATH + venv, (2) **writes `run.py` via a heredoc** into the output dir, (3) runs it. The workload logic is the heredoc body.

### The bash runtime contract the command can rely on

The command runs with a **clean environment** — no `PATH`, no `HOME`, no inherited `os.environ`. gbserver injects:

- `LLMB_BASH_PYTHON_DIR` — dir of a pinned Python (≥3.11). **Lead your PATH with it**, e.g. `export PATH="$LLMB_BASH_PYTHON_DIR:/usr/local/bin:/usr/bin:/bin"`. There is no HOME, so point caches somewhere writable (e.g. `export HF_HOME="$VENV_BASE/hf-cache"`).
- `LLMB_BASH_OUTPUT_DIR` — write outputs here; this dir (or a path you register) becomes the artifact.
- `LLMB_BASH_INPUT_<NAME>` — the resolved **local path** of each target input named `<name>` (uppercased). This works with the `command` step even though it declares no input schema — gbserver auto-exports every resolved input. E.g. input `model` → `$LLMB_BASH_INPUT_MODEL`.
- Anything under `config.bash.env` in build.yaml — exported as an env var (per-run params like `PROMPT`, `MAX_STEPS`). Read them with defaults in the script.

### Registering outputs (artifacts)

Print a line **at the start of a line** of the form:

```
LLMB_ARTIFACT_ID:<name> LLMB_ARTIFACT_PATH:<abs-dir>
```

The `command` step's monitor is anchored to the start of the line, so put this `print(...)` on its own line (not indented) and it registers `<abs-dir>` as the `<name>` artifact. Declare a matching `outputs.<name>` in the target (it validates; the command step allows unknown outputs).

### Heredoc mechanics that matter

- gbserver renders the command **per value** (not by re-parsing YAML text), so a multi-line command with quotes survives intact — write it as a YAML block scalar (`command: |`). It reaches gbmcp as build.yaml text via `build_start(file_content=…)`, which preserves it.
- Use a **quoted** heredoc delimiter (`<<'PYEOF'`) so `$`, backticks, etc. in the Python stay literal.
- The embedded Python must use **single-brace f-strings** only (`f"{x}"`). Do **not** use `{{ }}` — that is Jinja and will be rendered away.
- **Cache the venv (and model download) OUTSIDE the per-run output dir** so reruns are fast — derive a stable base from `LLMB_BASH_OUTPUT_DIR`.

### Worked example command

```bash
set -eu
# Clean env: establish PATH from the launcher's pinned python dir.
export PATH="${LLMB_BASH_PYTHON_DIR:?need LLMB_BASH_PYTHON_DIR}:/usr/local/bin:/usr/bin:/bin"
OUT="${LLMB_BASH_OUTPUT_DIR:?need LLMB_BASH_OUTPUT_DIR}"

# Cached venv outside the per-run output dir so reruns are fast.
case "$OUT" in
  */workdir/*) VENV_BASE="${OUT%%/workdir/*}/.gb-venvs" ;;
  *)           VENV_BASE="${TMPDIR:-/tmp}/.gb-venvs" ;;
esac
mkdir -p "$VENV_BASE"
export HF_HOME="$VENV_BASE/hf-cache"      # child env has no HOME
PY="$LLMB_BASH_PYTHON_DIR/python3"; [ -x "$PY" ] || PY=python3
VENV="$VENV_BASE/myworkload"
[ -x "$VENV/bin/python" ] || { "$PY" -m venv "$VENV"; "$VENV/bin/pip" install --quiet --upgrade pip; }
"$VENV/bin/pip" install --quiet "torch==2.12.0" "transformers>=4.56,<5" accelerate

# Write the workload into the workspace (self-contained; no external files).
cat > "$OUT/run.py" <<'PYEOF'
import json, os, sys
model_path = os.environ.get("LLMB_BASH_INPUT_MODEL", "")   # auto-exported input
out = os.environ["LLMB_BASH_OUTPUT_DIR"]
prompt = os.environ.get("PROMPT", "hello")                  # from config.bash.env
if not model_path or not os.path.isdir(model_path):
    sys.exit(f"ERROR: bad model path: {model_path!r}")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).eval()
enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
ids = model.generate(**enc, max_new_tokens=64, do_sample=False, pad_token_id=tok.eos_token_id)
resp = tok.decode(ids[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "response.txt"), "w").write(resp + "\n")
json.dump({"prompt": prompt, "response": resp}, open(os.path.join(out, "result.json"), "w"), indent=2)
print(f"LLMB_ARTIFACT_ID:response LLMB_ARTIFACT_PATH:{out}")
print("WORKLOAD_SUCCESS")
PYEOF

exec "$VENV/bin/python" "$OUT/run.py"
```

For a **pure shell** workload (no Python), skip the venv/heredoc entirely — put the shell directly in `command` and, if it produces an output, `echo "LLMB_ARTIFACT_ID:… LLMB_ARTIFACT_PATH:…"` at line start.

## The build.yaml — targets and the command step

Keep it with the user's project (not under assets). Single target, one `command` step is the simplest shape. Per-run settings go in `config.bash.env`. A base model can be an `hf:///owner/repo` input — the bash env's HF assetstore pulls it automatically (no separate pull step) and exposes it as `$LLMB_BASH_INPUT_MODEL`.

```yaml
granite.build:
  name: <build-name>
  version: 0.0.1
  targets:
    run:                                       # one target
      environment_uri: space://environments/bash
      inputs:
        model:
          uri: hf:///ibm-granite/granite-4.0-h-350m   # triple slash; host defaults to huggingface.co
      outputs:
        response:                              # registered by the LLMB_ARTIFACT_ID line
          uri: file:outputs/run_{{ binding.path | short_hash }}/
      steps:
        - step_uri: space://steps/command      # the built-in generic step — nothing to author
          config:
            command_config:
              command: |                       # the whole workload, inline (see example above)
                set -eu
                export PATH="${LLMB_BASH_PYTHON_DIR:?}:/usr/local/bin:/usr/bin:/bin"
                OUT="${LLMB_BASH_OUTPUT_DIR:?}"
                # ... venv setup, heredoc that writes run.py, then: exec .../python "$OUT/run.py"
            bash:
              env:                             # per-run settings -> env vars in the command
                PROMPT: "what is the best ibm office location"
            compute_config:
              num_nodes: 1
```

Notes:
- **Inputs** carry exactly one of `uri:` (fixed location: `hf:///…`, `file:…`) or `binding:` (wire to another target's output for multi-target workloads).
  - **A `file:` input MUST use an absolute path** — `file:///abs/path/`. A *relative* `file:foo/` resolves against the build's **runtime working dir, not your project dir**, so the input can **silently resolve to nothing** (e.g. an `adapter`/`dataset` that never loads) while the build still reports **SUCCESS**. When the file is produced by another target in the *same* build, prefer a `binding:` (no path at all); use an absolute `file:///…` URI to point at a fixed location from a *previous* build. (Confirm it actually loaded in `build_job_log` — a missing `file:` input is a `WARNING … ignoring`, not a failure.)
- **Produced outputs** declare a `uri:`/`base_uri:` push destination; the command registers them via the `LLMB_ARTIFACT_ID` line.
- Multiple `command` steps in one target run sequentially and share the filesystem. Use separate bound targets when phases have distinct inputs/outputs.

## Submit, then monitor

Submit the authored build.yaml by passing its **text** (not a path) to `build_start`:

```
build_start(file_content=<the build.yaml as a string>[, space=..., params=...])   # returns a build_id
build_list()                                     # builds still running? lists builds + statuses
build_status(build_id)                           # monitor a specific build's status
build_log(build_id)                              # a build's log output
build_describe(...)                              # inspect a build's structure/results
```

**Monitor by polling `build_status(build_id)`** — done when `details.status` is `success` / `failed` / `cancelled` (lowercase). Then `build_job_log(build_id)` for the real output.

### Long, unattended build — one "done" ping (optional)

`build_status` is the source of truth, but a shell can't call MCP tools, so there's no clean `sleep`-and-loop — you re-poll `build_status` turn by turn. For a build that runs for minutes, get a **single** completion notification instead by background-polling gbserver's REST status endpoint. It reports the **same** status `build_status` does (that tool's `GBClient` hits this very endpoint) — REST is just the door a shell can reach so it can sleep between checks.

Run it with **`run_in_background: true`**; it exits at the first terminal state and notifies you once:

```bash
ID="<build-id>"; URL="http://127.0.0.1:${GBSERVER_PORT:-8080}/api/v1/builds/$ID/status"
until curl -s "$URL" | jq -r '.status.build.status' | grep -qE '^(success|failed|invalid|cancelled)$'; do
  sleep 15
done
echo "build $ID reached a terminal state"
```

Overall status is at `.status.build.status` (lowercase). No auth header — standalone gbserver/gbmcp is unauthenticated on localhost. When it exits, read `build_status(build_id)` for the outcome and `build_job_log(build_id)` for the output. (No `jq`? Swap in `python3 -c 'import sys,json;print(json.load(sys.stdin)["status"]["build"]["status"])'`.)

There is **no `build_validate` tool** in gbmcp, and validation never proved a build runs anyway. Always do a real `build_start` and watch it reach SUCCESS (via `build_list` / `build_status`) with the command actually executing.

## Debugging — where the REAL output is

`build_log(build_id)` and `build_status(build_id)` mostly show **status events**, not your command's stdout. The actual output (prints, tracebacks, the `LLMB_ARTIFACT_ID` line) is on the same host at:

```
~/.granite.build/workdir/llm-build-<build-id>/target-<t>/target-run-*/step-<step-name>/step-run-*/launch-*/outputs/job.log
```

Fetch it via MCP with `build_job_log(build_id)` — it returns that `job.log`'s tail directly, no shell needed. Note that `command.sh` echoes the full command back at the top (`command step start: …`), so you'll see your heredoc'd Python source before the real execution output — the actual run begins after that. This `job.log` is your primary debugging artifact; if a step "succeeded" but did nothing, read it first.

## Authoring procedure (checklist)

1. `gbserver_status()`; if it isn't `ready`, `gbserver_start()` to bring the backend up (see **`run-gbserver`**).
2. Use the `command` step (`space://steps/command`) in the `bash` environment — see `references/steps.md`.
3. Write the build.yaml with the user's project: one target, one `command` step. Put the workload in `config.command_config.command` (shell for simple cases; a heredoc that writes+runs `run.py` for Python). Declare inputs (`hf:///…` model), outputs (registered by the `LLMB_ARTIFACT_ID` line), and per-run params in `config.bash.env`. No step directory to author.
4. Submit with **`build_start(file_content=<yaml text>)`**. Monitor with `build_status(build_id)`; done once it leaves `build_list()`. Read `build_job_log(build_id)` to confirm the workload actually ran (not just that status flipped to SUCCESS).
5. If anything is unclear about a field/option/error, consult the **`gb-docs`** skill.

## When unsure

Invoke **`gb-docs`** for the authoritative schema/CLI/troubleshooting docs. If the docs and this skill disagree on a documented field, trust the docs — but note the docs are k8s/LSF-centric, and several documented conveniences (`config.workload.commands`, `gb.files_to_create`) are **k8s/LSF-only and do not apply to the bash backend**. For bash behavior, the source of truth is this skill's `references/steps.md` plus a working build's `job.log` (via `build_job_log`).
