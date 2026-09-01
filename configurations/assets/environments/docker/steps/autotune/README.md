# autotune (docker) — AutoTune / fm-tune HPO + training step

The Docker copy of `space://steps/autotune`. Same `command.sh` / `run.py` as the
bash copy (kept byte-identical); the differences live in `step.yaml`:

- `docker` launcher on the fm-tune runtime image; `command` runs the shared
  `command.sh` from the `/gb-workspace` bind-mount.
- `launcher.config.image` is a placeholder (`<YOUR_REGISTRY>/autotunex/build-runtime:21`)
  — replace it with your fm-tune runtime image. It can also be overridden
  per-build via `config.docker.image`.
- Inputs are wired via `launcher.config.env` (Docker does not auto-export them).
- `BASH_BUILD_VENV=false` — deps come from the image, not a venv.
- `BACKEND=torch`; set `FM_TUNE_ROOT` (the image's fm-tune path) per-build via
  `config.docker.env`.

Materialization of `config.autotune-config` is identical to the bash copy
(handled inside `command.sh`).
