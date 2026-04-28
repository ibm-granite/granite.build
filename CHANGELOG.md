# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **gbcli** — CLI client added to the monorepo under `src/gbcli/`, with entry points `gb`, `gbcli`, `llmbuild`, `llmb`, `lamb`
- Standalone mode — all-in-one server with SQLite storage and thread-based execution
- Docker environment — run build steps in containers with GPU support
- Bash environment — run build steps as local processes (macOS/Linux)
- Kubernetes environment — run build steps as K8s jobs
- RunPod environment (beta) — run build steps on RunPod GPU instances
- SkyPilot/AWS environment (beta) — run build steps on cloud instances via SkyPilot
- HuggingFace Hub integration — download models and datasets via `hf://` URIs
- REST API — FastAPI-based build management at `/api/v1`
- Pipeline orchestration — multi-step builds defined in `build.yaml`
- Built-in steps: `gbstep`, `hfpull`, `hfpush`, `lhpull`, `lhpush`, `cosrclone`
