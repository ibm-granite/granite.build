# autotune / docker — development notes

Authoring home for the `autotune` step's **docker** implementation. Dev-oriented and
**not** published; `USAGE.md` ships as the released step's `README.md`.

Released to `configurations/assets/environments/docker/steps/autotune/` by
`make publish-step`.

`command.sh` and `run.py` are kept **byte-identical** to the bash copy (a unit test
enforces it). Only `step-template.yaml` differs. See
[`../bash/README.md`](../bash/README.md) for the `SRC_DIR := bash_scripts` deviation,
which applies here identically.

## No build test

There is deliberately no `test/` tree, so `publish-step` warns — accurately. A real
docker build test needs an fm-tune runtime image (CUDA + fm-tune baked in) that this
repo does not publish and a checkout cannot build, so any test here would skip
unconditionally. The bash variant's build test covers the shared `command.sh` /
`run.py` behavior; what is docker-specific is `step-template.yaml`, which the unit
suite asserts.

## Docker config precedence — why step-template.yaml looks sparse

`launcher_config` **outranks** `config.docker.*` in both directions:

- `Docker._resolve_image` (`docker.py:181-183`) checks `launcher_config.image` first,
  so the template pins **no** image — a build supplies it via `config.docker.image`.
- `Docker.get_launch_env_vars` (`docker.py:416-420`) merges `config.docker.env` and
  then `env.update(launcher_env)`, so launcher env wins.

Only values intrinsic to running in a container belong in the launcher env
(`BASH_BUILD_VENV=false`, the input wiring, the output dir). Anything a build should
tune — `BACKEND`, `FM_TUNE_ROOT` — must stay out or it becomes unoverridable. This is
the inverse of the bash environment, and it is the easiest thing to get wrong here.
