# Examples

Build configurations for gbserver / gbcli.

All examples can be run using the `gb` CLI (or any equivalent entry point: `gbcli`, `llmbuild`, `llmb`, `lamb`).

## 1. Granite model: TRL fine-tuning and unitxt evaluation (Docker + GPU)

Real-world pipeline using `ibm-granite/granite-4.0-350m` from HuggingFace, TRL fine-tuning in Docker, and unitxt evaluation.

Build configs are in `test-data/standalone-environments/builds/`. The test orchestrator is at `test/gbserver_test/test_standalone_environments_e2e.py`.

### Run via gbcli

Start the standalone server with the test space:

```bash
pip install -e ".[standalone,docker,dev]"
gbserver standalone --space-dir test-data/standalone-environments
```

Submit builds from another terminal:

```bash
# TRL fine-tuning (downloads granite-4.0-350m, fine-tunes in Docker)
gb build start test-data/standalone-environments/builds/docker-trl.yaml

# unitxt evaluation (downloads granite-4.0-350m, evaluates in Docker)
gb build start test-data/standalone-environments/builds/docker-unitxt.yaml

# Monitor
gb build list
gb build get <build-id>
```

### Run via pytest

```bash
# TRL fine-tuning with Docker
pytest -s test/gbserver_test/test_standalone_environments_e2e.py::TestStandaloneEnvironmentsE2E::test_docker_trl_finetune

# unitxt evaluation with Docker
pytest -s test/gbserver_test/test_standalone_environments_e2e.py::TestStandaloneEnvironmentsE2E::test_docker_unitxt_eval

# Or both
pytest -s test/gbserver_test/test_standalone_environments_e2e.py -k "docker_trl or docker_unitxt"
```

### Available build configs

| Config | Environment | Description |
|--------|-------------|-------------|
| `bash-inference.yaml` | Bash | Download model + run inference |
| `docker-inference.yaml` | Docker | Download model + run inference |
| `bash-trl.yaml` | Bash | TRL fine-tuning (granite-4.0-350m) |
| `docker-trl.yaml` | Docker | TRL fine-tuning (granite-4.0-350m) |
| `bash-unitxt.yaml` | Bash | unitxt evaluation (granite-4.0-350m) |
| `docker-unitxt.yaml` | Docker | unitxt evaluation (granite-4.0-350m) |
| `bash-gpu.yaml` | Bash | GPU availability check |
| `docker-gpu.yaml` | Docker | GPU passthrough check |

## 2. Standalone quickstart (multiple backends)

The [standalone quickstart](../samples/standalone/standalone-quickstart/) supports bash, Docker, RunPod, and SkyPilot backends.

```bash
gbserver standalone --space-dir samples/standalone/standalone-quickstart
gb build start samples/standalone/standalone-quickstart/build.yaml
```

## 3. Minimal hello-world (local bash, no GPU)

A single-step build that cats a text file. Good for verifying your install.

```bash
# Direct execution (no server needed). The positional argument is a build
# DIRECTORY (one containing build.yaml), not a path to the YAML file.
# --space-config-uri tells gbserver where to resolve space:// URIs from —
# here we point it at the same directory because the example is self-contained.
gbserver build run \
  --space-config-uri "file://$(pwd)/examples/minimal-build" \
  examples/minimal-build

# Via gbcli (requires a running gbserver)
gbserver standalone --space-dir examples/minimal-build &
gb build start -f examples/minimal-build/build.yaml
gb build list
```

See [examples/minimal-build/](minimal-build/) for details. For a fully
self-contained example with all the space / environment / step files
already wired up, see
[`samples/tests/local_hello_world_full/`](../samples/tests/local_hello_world_full/).
