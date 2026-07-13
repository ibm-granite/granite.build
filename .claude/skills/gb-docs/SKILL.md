---
name: gb-docs
description: Look up authoritative Granite.build documentation that ships inside the granite.build repo (build.yaml schema, steps, CLI, troubleshooting, glossary). Use when unsure about a Granite.build field, option, command, concept, or error, or when another gb skill says to consult the docs.
argument-hint: "[topic or question]"
---

# Granite.build docs lookup

The granite.build repo ships docs under `docs/`. They track the installed version. **Read** the relevant file(s) with the file tools, then answer grounded in their content, citing the doc path.

## Important caveat: the docs are k8s/LSF-centric

This environment runs the **standalone bash backend** by default, but most docs are written for the Kubernetes/LSF backends. Several documented conveniences **do not apply to bash**, e.g.:
- `config.workload.commands` (inline command list) — k8s/LSF only.
- `config.gb.files_to_create` / `additional_files` — k8s/LSF only.
- The generic `gbstep` step — has no bash launcher; using it on bash fails with `KeyError: 'helm'`.

So: use the docs for the **schema and concepts** (URIs, inputs/outputs, the field reference), but for **how a bash step actually launches and behaves**, the authoritative source is the on-disk reality, not the prose:
- The on-disk steps under `<assets>/environments/bash/steps/` — `hello`/`command` are the minimal correct bash steps (`step.yaml` with a `Bash`/`nohup` launcher + a `bash_scripts/<step>/command.sh`), while `inference`/`inference-lora`/`lora-finetune` are the reference for a step that captures outputs as artifacts (they carry the `NEWARTIFACT_IN_ENVIRONMENT_EVENT` monitor).
- A *prior successful build's* artifacts under `~/.granite.build/workdir/llm-build-<id>/.../launch-*/` — the copied `step.yaml`, the script, and especially `job.log` (the real stdout). Reverse-engineering a working build beats guessing from docs.

When the docs and on-disk bash reality disagree about bash behavior, trust the on-disk reality and say so.

## Locate the docs

Find the granite.build checkout, in this order, then use its `docs/` directory:
`./granite.build`, `~/granite.build`. If no checkout exists, the `run-gbserver` skill clones it. As a fallback, the docs are browsable at github.com/ibm-granite/granite.build under `docs/`.

## Index — pick by topic

**Authoring builds**
- `docs/builds/build-yaml-reference.md` — **authoritative `build.yaml` schema** (targets, steps, inputs/outputs, URIs, all fields/options).
- `docs/builds/README.md` — build features overview; `docs/builds/hf-push.md` — push artifacts to HuggingFace; `docs/builds/event-notifications.md`.
- `docs/steps/bring-your-own-step.md`, `docs/steps/custom-code-steps.md`, `docs/steps/bring-your-own-image.md` — author/use custom steps (examples lean k8s/docker; the `commands`/`files_to_create` features are k8s/LSF only). `docs/steps/README.md` — steps overview.

**Using the system**
- `docs/getting-started.md` — end-to-end walkthrough.
- `docs/cli/gb-cli-reference.md` — the `gb` CLI; `docs/cli/gbserver-cli-reference.md` — the server CLI; `docs/cli/gbtest-cli-reference.md` — gbtest.
- `docs/help/faq.md` — common questions; `docs/glossary.md` — terminology (build, target, step, artifact, space, …).
- `docs/demos/` — demo walkthroughs (`docker-demo.md`, `granite4_nano.md`, `skypilot-slurm-demo.md`).

**Concepts / architecture**
- `docs/architecture/arch-diagram.md`, `docs/architecture/environment-classes.md`.
- `docs/steps/README.md`, `docs/templates/README.md`, `docs/spaces/README.md`, `docs/asset-stores/README.md`.

**Build features**
- `docs/builds/lineage.md`, `docs/builds/build-retry.md`, `docs/builds/retry.md`, `docs/builds/step-retry-configuration.md`, `docs/builds/target-reuse.md`.

**Environments / config / secrets / troubleshooting**
- `docs/help/troubleshooting.md` — **diagnosing failures** (server or build).
- `docs/environments/bash.md` — the **standalone bash backend** (most relevant here); also `docker.md`, `k8s.md`, `lsf.md`, `runpod.md`, `skypilot*.md`, `step-resolution.md`, and `setup/`.
- `docs/configuration/` — `config-files.md`, `environment-variables.md`, `gb-environment.md`.
- `docs/secrets/` — `local-secrets-manager.md`, `env-secrets-manager.md`, `ibmcloud-secrets-manager.md`; `docs/rest-api/multi-provider-authentication.md`.

`docs/README.md` is the repo's own index if you need the full catalog.

## How to use

1. Resolve the checkout's `docs/` dir.
2. From `$ARGUMENTS` (the topic/question) pick the most relevant file(s); if unsure, skim `docs/README.md` or grep `docs/` for keywords.
3. **Read** the file(s) and answer grounded in their content, citing the doc path.
4. If the question is about **bash-backend execution** specifically, also check the on-disk `hello` step and (if available) a working build's `job.log`/`step.yaml` — and prefer those over doc prose when they conflict.
