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
- The `hello` step under `<assets>/environments/bash/steps/hello/` — the minimal correct bash step (`step.yaml` with a `Bash`/`nohup` launcher + its `bash_scripts/hello/command.sh`).
- A *prior successful build's* artifacts under `~/.granite.build/workdir/llm-build-<id>/.../launch-*/` — the copied `step.yaml`, the script, and especially `job.log` (the real stdout). Reverse-engineering a working build beats guessing from docs.

When the docs and on-disk bash reality disagree about bash behavior, trust the on-disk reality and say so.

## Locate the docs

Find the granite.build checkout, in this order, then use its `docs/` directory:
`./granite.build`, `~/granite.build`. If no checkout exists, the `run-gbserver` skill clones it. As a fallback, the docs are browsable at github.com/ibm-granite/granite.build under `docs/`.

## Index — pick by topic

**Authoring builds**
- `docs/users/build-yaml-reference.md` — **authoritative `build.yaml` schema** (targets, steps, inputs/outputs, URIs, all fields/options).
- `docs/users/bring-your-own-step.md` — use/author custom steps (note: examples lean k8s/docker).
- `docs/users/custom-code-steps.md` — running custom code (the `commands`/`files_to_create` features here are k8s/LSF only).
- `docs/users/bring-your-own-image.md` — custom container images.
- `docs/users/hf-push.md` — pushing artifacts to HuggingFace.

**Using the system**
- `docs/getting-started.md` — end-to-end walkthrough.
- `docs/users/cli-reference.md` — the `gb` CLI commands.
- `docs/users/faq.md` — common questions.
- `docs/glossary.md` — terminology (build, target, step, artifact, space, …).
- `docs/demos.md` — demo walkthroughs.

**Concepts / architecture**
- `docs/architecture/arch-diagram.md`, `docs/architecture/environment-classes.md`.
- `docs/steps/README.md`, `docs/templates/README.md`.

**Features**
- `docs/features/lineage.md`, `build-retry.md`, `retry.md`, `step-retry-configuration.md`, `target-reuse.md`, `gbtest.md`.

**Operators / setup / troubleshooting**
- `docs/operators/troubleshooting.md` — **diagnosing failures** (server or build).
- `docs/operators/environment-yaml-config.md`, `local-secrets-manager.md`, `multi-provider-authentication.md`, `runpod-orchestrator.md`, `skypilot-local-infrastructure.md`, `setup/skypilot-kubernetes-setup.md`.

`docs/README.md` is the repo's own index if you need the full catalog.

## How to use

1. Resolve the checkout's `docs/` dir.
2. From `$ARGUMENTS` (the topic/question) pick the most relevant file(s); if unsure, skim `docs/README.md` or grep `docs/` for keywords.
3. **Read** the file(s) and answer grounded in their content, citing the doc path.
4. If the question is about **bash-backend execution** specifically, also check the on-disk `hello` step and (if available) a working build's `job.log`/`step.yaml` — and prefer those over doc prose when they conflict.
