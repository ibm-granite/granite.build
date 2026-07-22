---
name: gb-docs
description: Look up authoritative Granite.build documentation that ships inside the granite.build repo (build.yaml schema, steps, CLI, troubleshooting, glossary). Use when unsure about a Granite.build field, option, command, concept, or error, or when another gb skill says to consult the docs.
argument-hint: "[topic or question]"
---

# Granite.build docs lookup

Granite.build's docs ship as Markdown in **`references/`** next to this skill. **Read** the relevant file(s) with the file tools, then answer grounded in their content, citing the doc path.

> Pick a file from the topic index below and read it directly with the file tools.

## Important caveat: the docs are k8s/LSF-centric

This environment runs the **standalone bash backend** by default, but most docs are written for the Kubernetes/LSF backends. Several documented conveniences **do not apply to bash**, e.g.:
- `config.workload.commands` (inline command list) — k8s/LSF only.
- `config.gb.files_to_create` / `additional_files` — k8s/LSF only.
- The generic `gbstep` step — has no bash launcher; using it on bash fails with `KeyError: 'helm'`.

So: use the docs for the **schema and concepts** (URIs, inputs/outputs, the field reference), but for **how a bash step actually launches and behaves**, prefer these over the doc prose:
- The **`create-build`** skill's `references/steps.md` — tested contracts (inputs/outputs/env, the artifact monitor) for the shipped bash steps (`inference`, `lora-finetune`, `inference-lora`, `hello`, `command`).
- A *prior successful build's* `job.log`, via **`build_job_log`** — the real stdout of a working build. Reverse-engineering a working build beats guessing from docs.

When the docs and actual bash behavior disagree, trust the observed behavior and say so.

## Where the docs are

Everything is in **`references/`** next to this `SKILL.md` (the index below lists `references/<x>` paths). If a file is ever missing locally, the same docs are online in the `ibm-granite/granite.build` repository.

## Index — pick by topic

**Authoring builds**
- `references/builds/build-yaml-reference.md` — **authoritative `build.yaml` schema** (targets, steps, inputs/outputs, URIs, all fields/options).
- `references/builds/README.md` — build features overview; `references/builds/hf-push.md` — push artifacts to HuggingFace; `references/builds/event-notifications.md`.
- `references/steps/bring-your-own-step.md`, `references/steps/custom-code-steps.md`, `references/steps/bring-your-own-image.md` — author/use custom steps (examples lean k8s/docker; the `commands`/`files_to_create` features are k8s/LSF only). `references/steps/README.md` — steps overview.

**Using the system**
- `references/getting-started.md` — end-to-end walkthrough.
- `references/cli/gb-cli-reference.md` — the `gb` CLI; `references/cli/gbserver-cli-reference.md` — the server CLI; `references/cli/gbtest-cli-reference.md` — gbtest.
- `references/help/faq.md` — common questions; `references/glossary.md` — terminology (build, target, step, artifact, space, …).
- `references/demos/` — demo walkthroughs (`docker-demo.md`, `granite4_nano.md`, `skypilot-slurm-demo.md`).

**Concepts / architecture**
- `references/architecture/arch-diagram.md`, `references/architecture/environment-classes.md`.
- `references/steps/README.md`, `references/templates/README.md`, `references/spaces/README.md`, `references/asset-stores/README.md`.

**Build features**
- `references/builds/lineage.md`, `references/builds/build-retry.md`, `references/builds/retry.md`, `references/builds/step-retry-configuration.md`, `references/builds/target-reuse.md`.

**Environments / config / secrets / troubleshooting**
- `references/help/troubleshooting.md` — **diagnosing failures** (server or build).
- `references/environments/bash.md` — the **standalone bash backend** (most relevant here); also `docker.md`, `k8s.md`, `lsf.md`, `runpod.md`, `skypilot*.md`, `step-resolution.md`, and `setup/`.
- `references/configuration/` — `config-files.md`, `environment-variables.md`, `gb-environment.md`.
- `references/secrets/` — `local-secrets-manager.md`, `env-secrets-manager.md`, `ibmcloud-secrets-manager.md`; `references/rest-api/multi-provider-authentication.md`.

`references/README.md` is the repo's own index if you need the full catalog.

## How to use

1. From `$ARGUMENTS` (the topic/question) pick the most relevant file(s) in `references/`; if unsure, skim `references/README.md` or grep `references/` for keywords.
2. **Read** the file(s) and answer grounded in their content, citing the doc path.
3. If the question is about **bash-backend execution** specifically, also check the `create-build` skill's `references/steps.md` and (if available) a working build's `job.log` via `build_job_log` — and prefer those over doc prose when they conflict.
