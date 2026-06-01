# Minimal Build Example

A minimal build configuration that runs a single step (cat a text file) in the local bash environment. Use this to verify your installation works end-to-end.

`build.yaml` references shared environment, step, and data definitions under [`samples/`](../../samples/). For a fully self-contained example with its own `environments/`, `steps/`, and `assetstores/` directories, see [`samples/tests/local_hello_world_full/`](../../samples/tests/local_hello_world_full/).

## Prerequisites

```bash
pip install -e ".[standalone]"
```

## Run with gbserver directly (no server needed)

```bash
gbserver build run \
  --space-config-uri "file://$(pwd)/examples/minimal-build" \
  examples/minimal-build
```

## Run with gbcli (requires a running server)

Start the server in one terminal:

```bash
export GBSERVER_API_KEY="my-secret-key"
gbserver standalone --space-dir examples/minimal-build
```

Submit the build from another terminal:

```bash
export GB_ENVIRONMENT=STANDALONE
export GBSERVER_API_KEY="my-secret-key"
gb build start -f examples/minimal-build/build.yaml
gb build list
```
