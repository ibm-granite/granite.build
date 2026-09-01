# autotune (docker) — AutoTune / fm-tune HPO + training step

The Docker copy of `space://steps/autotune`. Same `command.sh` / `run.py` as the
bash copy (kept byte-identical); the differences live in `step.yaml`:

- `docker` launcher; `command` runs the shared `command.sh` from the
  `/gb-workspace` bind-mount.
- **The image is not set in `step.yaml`** — supply your fm-tune runtime image via
  `config.docker.image` in the build. `Docker._resolve_image` checks the launcher
  config *first*, so an image pinned in `step.yaml` could not be overridden.
- Inputs are wired via `launcher.config.env` (Docker does not auto-export them),
  including the optional `hpo_config`, which is guarded so an unbound input renders
  to an empty string rather than failing the strict template fill.
- `BASH_BUILD_VENV=false` — deps come from the image, not a venv. This one is
  intentionally in the launcher env because it is intrinsic to running in a
  container.
- Set `FM_TUNE_ROOT` (the image's fm-tune path) and, if you want anything other
  than torch, `BACKEND`, per-build via `config.docker.env`.

**Precedence gotcha:** launcher env *wins* over `config.docker.env`
(`Docker.get_launch_env_vars` merges `config.docker.env` first, then updates with
the launcher's) — the inverse of the Bash environment. Anything a build needs to
tune must therefore stay out of `launcher.config.env`.

Materialization of `config.autotune-config` is identical to the bash copy
(handled inside `command.sh`).
