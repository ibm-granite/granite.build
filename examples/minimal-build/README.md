# Minimal Build Example

A single-step `build.yaml` that cats a text file, intended as a "smallest possible config" reference.

> **Status:** the config parses and validates, but does not yet run end-to-end. Its `environment_uri` points at [`samples/environments/local/`](../../samples/environments/local/), which has no assetstore declared, so the build fails resolving the input URI at execution time. Tracked in #59.
>
> For a working example with environment + step + assetstore files all wired up, see [`samples/tests/local_hello_world_full/`](../../samples/tests/local_hello_world_full/).

## Prerequisites

```bash
pip install -e ".[standalone]"
```
