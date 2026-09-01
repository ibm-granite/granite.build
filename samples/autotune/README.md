# AutoTune build.yaml references

- `build.bash.yaml` — local run on the **bash** environment via
  `space://steps/autotune` (`BACKEND: torch`). `FM_TUNE_ROOT` points at the
  fm-tune copy vendored in this repo at `autotunex/src/fm-tune`, so no external
  clone is needed. The inline `autotune-config` is materialized by the step's
  `command.sh`.
- `build.bash.test.yaml` — the same bash shape with a fuller `autotune-config`
  and the `mlx` backend.
- `build.k8s.yaml` — **production** run using the shipped `space://steps/custom_code`.
  The same inline `autotune-config` block is written to `/tmp/autotunex.yaml` by
  `config.gb.files_to_create`, so no bespoke k8s step is needed.

**Note:** all three are structural references; none is submitted by the plan.
Replace every `<YOUR_*>` placeholder before use —
`<YOUR_GRANITE_BUILD_CHECKOUT>` (an **absolute** path to your clone of this
repo; the step runs with its CWD in the build workdir, not the repo root),
`<YOUR_DATASET_DIR>`, `<YOUR_ORG>`, `<YOUR_DATASET>`, `<YOUR_GIT_HOST>`, and
`<YOUR_REGISTRY>`. For the bash samples that is just:

```sh
sed -i '' "s|<YOUR_GRANITE_BUILD_CHECKOUT>|$PWD|" samples/autotune/build.bash.yaml
```
