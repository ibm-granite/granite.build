# Minimal Build Example

A minimal build configuration that runs a single step (cat a text file) in the local bash environment. Use this to verify your installation works end-to-end.

## Prerequisites

```bash
pip install -e ".[standalone]"
```

## Run with gbserver directly (no server needed)

```bash
gbserver build run --build-dir examples/minimal-build
```

## Run with gbcli (requires a running server)

Start the server in one terminal:

```bash
gbserver standalone --space-dir /tmp/gb-space
```

Submit the build from another terminal:

```bash
gb build start examples/minimal-build/build.yaml
gb build list
```
