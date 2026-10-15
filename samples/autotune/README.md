# AutoTune build.yaml references

- `build.bash.yaml` — local run on the **bash** environment via
  `space://steps/autotune` (`BACKEND: torch`). `FM_TUNE_ROOT` points at the
  fm-tune copy vendored in this repo at `autotunex/src/fm-tune`, so no external
  clone is needed. The inline `autotune-config` is materialized by the step's
  `command.sh`.
- `build.bash.test.yaml` — the same bash shape with a fuller `autotune-config`
  and the `mlx` backend.
- `build.k8s.yaml` — **production** run using the shipped `space://steps/gbstep`.
  The same inline `autotune-config` block is written to `/tmp/autotunex.yaml` by
  `config.gb.files_to_create`, so no bespoke k8s step is needed.

**Note:** all three are structural references, not runnable as-is. Replace every
`<YOUR_*>` placeholder first — `<YOUR_GRANITE_BUILD_CHECKOUT>` and
`<YOUR_OUTPUT_DIR>` (both **absolute**: the step runs with its CWD in the build
workdir, and gbserver resolves a relative `file:` URI against its own CWD),
`<YOUR_DATASET_DIR>`, `<YOUR_ORG>`, `<YOUR_DATASET>`, `<YOUR_GIT_HOST>`, and
`<YOUR_REGISTRY>`. For the bash samples that is:

```sh
sed -i '' -e "s|<YOUR_GRANITE_BUILD_CHECKOUT>|$PWD|" \
          -e "s|<YOUR_OUTPUT_DIR>|$PWD/outputs|" samples/autotune/build.bash.yaml
```

**Docker is deferred.** A docker variant of this step exists on the
`feat/autotune-step-docker` branch, along with its build test and a `FINDINGS.md`
recording an unresolved artifact-delivery issue (a successful build can register an
empty model artifact). It is intentionally not part of this change.
