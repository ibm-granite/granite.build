# `file_mounts` steps fail on BlueVela SLURM (sudo-less symlink wrap)

**Status:** fixed in the SkyPilot fork by
[cmadam/skypilot PR 3](https://github.com/cmadam/skypilot/pull/3), green on BlueVela
SLURM 2026-09-28; **pending** that PR's merge and a re-pin of `gb-sky-v1-stable`. Diagnosed 2026-09-18. See [Resolution](#resolution).
**Affects:** any **bare (non-containerized)** step with a *relative* `file_mounts`
destination, on any `skypilot`/`slurm` environment that sets `shared_workdir` and whose
SSH user is not root. **Containerized steps are NOT affected** — see
[Why containerized file_mounts work](#why-containerized-file_mounts-work-and-bare-ones-do-not),
which is why PR #401's containerized filemount test passes on this same environment.
Today the exposed steps are
**two steps, both with `src: src`** — `dpk` and **`byoc`**
(`configurations/assets/environments/skypilot/steps/{dpk,byoc}/step.yaml`, and their
`step-template.yaml` sources under `steps/`). `byoc` has the same exposure and has simply
not been run on BlueVela SLURM yet.
**First hit by:** `test/integration/ibm/buildrunner/skypilot/slurm_bluevela/test_dpk_pii.py`
(`TestDPKBlueVelaSlurm`), the `dpk` step running `pii_redactor` on
`space://environments/skypilot/slurm/bluevela`.
**Fix lives in:** the pinned SkyPilot fork, not this repo — Option A below, as implemented in
[Resolution](#resolution).

---

## TL;DR

The `dpk` step ships its bash scripts through `file_mounts`. gbserver rewrites that
relative destination onto the shared `/proj` tree; SkyPilot then tries to
`sudo mkdir -p` the target, and the BlueVela account has no passwordless sudo, so the
build dies before the transform runs.

`LsfCommandRunner` already declares an exemption for exactly this case.
`SlurmCommandRunner` does not. That asymmetry is the whole bug.

Nothing in `granite.build` can express the exemption — the SkyPilot backend asks only
the *runner object*, and the SLURM runner has no channel for it.

---

## Reproduction and evidence

```
pytest -s test/integration/ibm/buildrunner/skypilot/slurm_bluevela/test_dpk_pii.py::TestDPKBlueVelaSlurm::test_runner
```

(Requires `source ./setup.sh` first — without `GB_ENVIRONMENT` the harness refuses
early with `Refusing to run storage tests with GB_ENVIRONMENT=PROD`, which is a
separate, unrelated env issue.)

Observed run: build `ca7265ef-a134-4fc7-ac34-f115c2de4bfd`, `PENDING → RUNNING → FAILED`
in about 6 minutes. Build-level error:

```
Command { [ "$(whoami)" == "root" ] && function sudo() { "$@"; } || true; } &&
  sudo mkdir -p /proj/data-eng/... failed with return code 1.
Failed to create symlinks. The target destination may already exist.
```

SkyPilot truncates the path. Its own `file_mounts.log` carries the real cause — note
that the step's files **transfer fine**; only the destination directory fails:

```
Transfer starting: 5 files
dpk_guard.sh              12951 100%
dpk_run.sh                 9309 100%
dpk_setup.sh               5168 100%
validate_tokenization2arrow.py  24162 100%
...
sudo: a terminal is required to read the password; either use the -S option to read
      from standard input or configure an askpass helper
sudo: a password is required
srun: error: p1-r08-n1: task 0: Exited with exit code 1
```

---

## The chain

Four behaviors compose into the failure. None is individually a bug.

### 1. The step must ship files, at a destination it cannot move

`configurations/assets/environments/skypilot/steps/dpk/step.yaml:189` declares:

```yaml
file_mounts:
  src: src
```

The destination is **relative** and **load-bearing** — both the `setup` and `run`
blocks invoke `bash ./src/dpk_guard.sh`, `./src/dpk_setup.sh`, `./src/dpk_run.sh`, and
the step.yaml states the files "land at `./src`, relative to the step's working
directory — the same directory setup and run start in."

So the payload has to be at `./src` relative to the step's CWD. It cannot be
redirected to `/tmp` or anywhere else without breaking the step's own scripts.

### 2. gbserver rewrites that destination onto the shared filesystem

`src/gbserver/environment/skypilot.py:780` `_remap_relative_dest`, ending at line 829:

```python
if not build_workdir:
    return dst   # no shared workdir: leave to SkyPilot's ~/sky_workdir default
return os.path.normpath(os.path.join(build_workdir, dst))
```

On BlueVela this yields
`/proj/data-eng/llmb-read-write/builds/<build_id>/runs/<targetrun_id>/src`.

This function is **cloud-agnostic**: it remaps for any env defining `shared_workdir`,
so SLURM gets it exactly as LSF does. Its docstring already anticipates the
consequence and names LSF's mitigation:

> On the LSF/enroot backend the shared `/proj` tree is bind-mounted identity into the
> step container, so a payload written to `${build_workdir}` on the (sudo-less) login
> node is visible to the job at the same path; the SkyPilot backend's symlink-wrap is
> exempted for these shared roots (see the fork's `sky/provision/lsf` runner hook).

### 3. SkyPilot sudo-wraps absolute destinations unless the runner exempts them

`sky/backends/cloud_vm_ray_backend.py:6252` and `:6294`:

```python
unwrapped_prefixes = runners[0].get_unwrapped_mount_prefixes()
...
if (not dst.startswith('~/') and not dst.startswith('/tmp/') and
        not dst_is_shared):
    wrapped_dst = backend_utils.FileMountHelper.wrap_file_mount(dst)
    cmd = backend_utils.FileMountHelper.make_safe_symlink_command(
        source=dst, target=wrapped_dst)     # defaults to sudo_cmd='sudo'
```

`make_safe_symlink_command` (`sky/backends/backend_utils.py:496`) takes a `sudo_cmd`
argument and documents passing `''` to suppress it, but this call site does not — so
the sudo path is unconditional once the destination is not exempt.

Note `~/sky_workdir` — SkyPilot's own default for relative destinations — starts with
`~/` and *would* have been exempt. It is gbserver's remap in step 2 that moves the
destination out of the exempt set.

### 4. `SlurmCommandRunner` never overrides the exemption hook

| | LSF | SLURM |
|---|---|---|
| Runner class | `command_runner.py:1940` | `command_runner.py:2202` |
| `get_unwrapped_mount_prefixes()` | overridden, `:2002` | **not overridden** → inherits base `:341`, returns `[]` |
| `shared_fs_roots` ctor arg | yes | **absent** |
| Wired in provisioner | `provision/lsf/instance.py:1360` | `provision/slurm/instance.py:1155` — not passed |
| Roots source | `_SHARED_FS_ROOTS = ['/proj', '/opt/share']`, `provision/lsf/instance.py:49`, plus `_derive_shared_fs_roots(enroot_mounts)` | none |

`LsfCommandRunner.get_unwrapped_mount_prefixes()`'s docstring describes our failure
verbatim: the wrap "both fails on the sudo-less login node and redirects the payload
to `~/.sky/file_mounts/...` — breaking that identity mapping."

So the mechanism designed for this exact problem exists, and was never extended from
LSF to SLURM.

---

## Why containerized `file_mounts` work and bare ones do not

This is the single most important qualifier on this report, and it is why PR #401
("Add filemount build test for skypilot/slurm/bluevela") passes while the dpk test
fails on the same cluster, same partition, same fork.

`SlurmCommandRunner` routes every command either into the enroot container or onto the
host, switched purely on whether the step declared an image
(`sky/utils/command_runner.py`, `SlurmCommandRunner.run`):

```python
def run(self, cmd, ...):
    in_container = self.container_args is not None
    return self._run_via_srun(cmd, in_container=in_container, **kwargs)
```

`rsync()` does the same ("Default: run in container if container_args set, otherwise on
host"). Combined with the root-sudo alias that the backend prepends to the symlink
command (`cloud_vm_ray_backend.py:6367`):

| Step kind | `container_args` | file_mounts run as | `sudo` | Outcome |
|---|---|---|---|---|
| Containerized (`image`/`dpk_image` set) | not `None` | **root**, inside enroot | aliased to a no-op | wrap succeeds ✅ |
| Bare (no image) | `None` | `granitebuild`, on the host | real, interactive | `a password is required` ❌ |

So the symlink wrap is not *wrong* on BlueVela — it is merely survivable whenever the
command happens to run as root. Containerization and the local Docker cluster's
`User: root` are two different ways of accidentally satisfying that, which is why this
went unnoticed.

**Confirmed directly, 2026-09-28.** The containerized column is no longer an inference
from PR #401: a dpk image-mode run on this environment synced its file_mounts
successfully, and the log shows the destination was the *wrapped* path, i.e. the
`sudo`-dependent symlink step ran and succeeded:

```
Syncing (to 1 node): .../src -> ~/.sky/file_mounts/proj/data-eng/llmb-read-write/builds/builds/<build>/runs/<run>/src
✓ Synced file_mounts.
```

The `~/.sky/file_mounts/...` prefix is what `wrap_file_mount()` produces. (Note the
doubled `builds/builds` segment — the env's `shared_workdir` already ends in `builds/`
and gbserver appends `builds/<build_id>/...`. Cosmetic, pre-existing, but confusing.)

**Corollary:** setting `dpk_config.dpk_image` would likely make the dpk test pass, but
it is not a usable workaround today. The step's config documentation states that an
image "is expected to already provide DPK... A base image WITHOUT DPK is therefore not
supported — there is no install to add it," and no prebaked DPK image exists for this
transform. It would also change what the test covers, moving it onto the Pyxis path
that `1step` and PR #401 already exercise, and away from the bare path that is broken.

## Why every existing test missed it

Two independent blind spots. Both matter for planning verification.

**The sibling BlueVela SLURM fixtures cannot reach the code.** `1step` and `2target`
use the builtin `command` step, which has no asset directory and therefore no
`file_mounts` at all. Nothing on BlueVela SLURM had ever exercised the remap. (The
dpk step.yaml notes this directly: the builtin command step has no asset dir, "which
is why a template cannot ship files this way.")

**The local Docker SLURM fixture passes for a reason that does not generalize.**
`configurations/assets/environments/skypilot/slurm/environment.yaml:25` sets
`User: root`, and SkyPilot's `ALIAS_SUDO_TO_EMPTY_FOR_ROOT_CMD`
(`sky/utils/command_runner.py:113`) rewrites `sudo` to a no-op function for root:

```sh
{ [ "$(whoami)" == "root" ] && function sudo() { "$@"; } || true; }
```

So the wrap *succeeds* locally. BlueVela's env uses `User: granitebuild`, where sudo
is real and interactive.

> **Consequence:** this class of bug is **not reproducible on the local Docker SLURM
> cluster**. `steps/dpk/skypilot/test/slurm-pii` passing is not evidence that dpk
> works on a sudo-less SLURM cluster. Any fix must be validated against BlueVela.

---

## Options

| # | Change | Owner | Verdict |
|---|---|---|---|
| **A** | `get_unwrapped_mount_prefixes()` on `SlurmCommandRunner` | skypilot fork | **Recommended** |
| B | Inline dpk's scripts; drop `file_mounts` | assets / dpk step | Fallback, with real costs |
| C | Change bluevela env `workdir` | gb-test | Reject |
| D | Passwordless sudo on BV compute nodes | BlueVela admins | Not ours; wrong lever |
| E | Decouple gbserver's remap from its `cd` | this repo | Reject |

### Option A — the SLURM runner hook (recommended)

This ports a finished, documented pattern rather than designing anything, **and the
precondition is already satisfied**. The SLURM provisioner already computes the
shared root and already identity-mounts it:

```python
# sky/provision/slurm/instance.py:410
workdir = skypilot_config.get_effective_region_config(
    cloud='slurm', keys=('workdir',), ...)

# sky/provision/slurm/instance.py:559
if workdir is not None and workdir != remote_home_dir:
    mount_paths.append(f'{workdir}:{workdir}')     # identity mount
```

That `workdir` is exactly what the bluevela env sets under
`cloud_config.slurm.cluster_configs.bluevela.workdir` (`/proj/data-eng/llmb-read-write`).
Identity-mounting is precisely the property that makes LSF's exemption sound: a write
on the driver side is visible to the job at the identical path. The value is known at
the point the runner is constructed and simply never handed over.

Sketch:

```python
# provision/slurm/instance.py, beside the existing mount logic
shared_fs_roots = ([workdir]
                   if workdir is not None and workdir != remote_home_dir
                   else [])
command_runner.SlurmCommandRunner(..., shared_fs_roots=shared_fs_roots, ...)
```

plus `shared_fs_roots` on `SlurmCommandRunner.__init__`, a
`get_unwrapped_mount_prefixes()` override copied from `LsfCommandRunner`, and the
`utils/command_runner.pyi:403` stub.

The `workdir != remote_home_dir` guard aligns with the existing documented rule in
[`skypilot-slurm.md`](skypilot-slurm.md) that a `workdir` equal to the account home
makes the mount inert.

**The complexity is not in the code.** It is in three places:

1. **The pin is a moving alias.** `pyproject.toml:94` (and `:110`) pin
   `git+https://github.com/cmadam/skypilot.git@gb-sky-v1-stable`, described there as
   "a moving alias for the recommended v1-line commit (re-pointed as the fork
   advances, like a docker minor tag)... cached clones need `git fetch --tags --force`."
   Changing it affects every consumer, so this needs the fork owner's coordination,
   not a drive-by commit.
2. **One real semantic trade-off.** Un-wrapping also opts the root out of
   `make_safe_symlink_command`'s clobber guard, which errors when the destination
   already exists as a real file or dir. LSF documents this deliberately: un-wrapped
   roots join `~/` and `/tmp/` in the "delegated to rsync" category. Low risk here
   because `$GB_BUILD_WORKDIR` is per-run, but it is a behavior change for *every*
   SLURM env with a `workdir`.
3. **SLURM runs file_mounts on the compute node, not the login node — verified, and the
   exemption holds anyway.** Every `SlurmCommandRunner` command goes through
   `_run_via_srun` (`srun --jobid=… --nodelist=<node> … bash -c …`), which is why the
   failure named `p1-r08-n1`. Re-derived for SLURM rather than borrowed from LSF: on the
   bare path, `workdir` is `sky_base_dir` — the directory SkyPilot itself creates the
   cluster home under — so the login user can write it by construction; on the container
   path it is identity-mounted (`{workdir}:{workdir}`), so the rsync writes through to the
   host path. Both were confirmed green on BlueVela.

Preserve LSF's stated **fail-closed** property: a root not recognized as shared gets
symlink-wrapped (a loud sudo failure), never silently redirected. Scoping SLURM to
`workdir` alone keeps that. LSF additionally needed a *node-local denylist* (e.g.
`/dev/shm`, `/home`) so identity mounts that are not actually shared across the
driver/compute split are not wrongly exempted; SLURM may need the same if more than
`workdir` is ever exempted.

### Option B — remove dpk's need for `file_mounts` (fallback)

Inline the three scripts as heredocs, the way the builtin `command` step does. Lives
entirely in repos we control, fixes dpk on every sudo-less SLURM env immediately, and
needs no fork coordination.

**But it regresses real quality.** The step deliberately keeps its shell in `src/` so
it is shellcheck'd, `bash -n`'d and unit tested — there are dedicated suites
(`test_dpk_run_sh.py`, `test_dpk_guard_sh.py`, `test_dpk_setup_sh.py`). Inlining
trades that for Jinja-in-YAML. It also fixes only dpk, leaving the next `file_mounts`
step to rediscover this. Reasonable as a stopgap if A is blocked for a long time; not
the answer.

### Option C — change the bluevela env `workdir`

Reject. `/proj` is *required* for the containerized `1step`/`2target` cross-node
handoff, so this degrades what currently works in order to accommodate a bare step
that needs no `/proj`. It is also a gb-test change, boxed in by the documented
constraints (must be an ancestor of `shared_workdir`, must not equal home), and fixes
nothing for the next step.

### Option D — passwordless sudo on BlueVela compute nodes

Not ours to grant, and the wrong lever: it would paper over a wrap that should not be
happening on a shared identity-mounted root in the first place.

### Option E — fix it in `granite.build`

Rejected after investigation, recorded so it is not re-proposed. Skipping the remap on
SLURM sends `src` to `~/sky_workdir/src` (exempt, no sudo), but gbserver still `cd`s
the step into `$GB_BUILD_WORKDIR`, so `./src` breaks again. The remap and the `cd` are
emitted together and would have to be decoupled per-cloud — a larger, more invasive
change in the more load-bearing codebase than Option A.

Also ruled out, for the record:

- **Overriding `file_mounts` from build.yaml.** `merge_dicts`
  (`src/gbserver/utils/filesystem.py:183`) is a recursive merge, so adding a key *adds
  a second mount* and leaves `src: src` in place — the `/proj` mount still runs. And
  the destination is the dict *key*, so it cannot be replaced, only added to.
- **Retargeting the destination to `/tmp/...`** (which SkyPilot exempts). Breaks the
  step's hardcoded `./src/...` invocations.
- **Unsetting `shared_workdir` per step.** It is read off the environment
  (`src/gbserver/environment/skypilot.py:1295`), not step config, so no build can
  override it.

---

## Resolution

Implemented in [cmadam/skypilot PR 3](https://github.com/cmadam/skypilot/pull/3)
(based on `granite-build` at `5f18669`), with unit tests beside the LSF ones:

- `SlurmCommandRunner.__init__` takes `shared_fs_roots`; `get_unwrapped_mount_prefixes()`
  returns it (docstring covers both execution modes and the clobber-guard trade-off).
- `provision/slurm/instance.py::get_command_runners` passes `[workdir]` when a SLURM
  `workdir` is configured, `[]` otherwise — so a home-based cluster keeps the wrap
  exactly as before. **Deviation from the Option A sketch:** there is no
  `workdir != remote_home_dir` guard. That guard governs whether the *container mount*
  is added, not whether the path is writable; a configured `workdir` is where SkyPilot
  creates the cluster home, so it is writable by the login user in either case.
- `command_runner.pyi` stub and the `_execute_file_mounts` comment updated.
- `tests/unit_tests/slurm/test_file_mount_wrap_exemption.py`: 6 tests (bare and
  containerized exemption, a prefix-sharing sibling stays wrapped, no roots wraps
  everything, and `get_command_runners` deriving the root from `workdir`). All 338
  SLURM + LSF unit tests pass.

**Validated on BlueVela SLURM, 2026-09-28** (patched fork installed editable, SkyPilot API
server restarted on it): `test_dpk_pii.py`, `test_dpk_tok_image.py`, `test_1step.py`
(runner + cancellation) and `test_2target.py` — `5 passed, 3 skipped` (the skips are the
fixtures' own `runner_cancellation` opt-outs). The pii fixture also needed an explicit
`launcher_config.resources.memory` (see below).

**Remaining:** merge that PR and re-point `gb-sky-v1-stable`. Until then only a venv with
the patched fork installed can run the bare-mode test (`test_dpk_pii.py`).
`byoc` has the same `src: src` exposure and is covered by the same fix, but has not been
run on BlueVela SLURM.

## Image mode: the "sibling blocker" was a stale API server and a mismatched image

An earlier version of this doc recorded an unexplained image-mode failure
(`/bin/bash: line 1: Usage:: command not found`, exit 127, reported as
`Job 1's setup failed`). Its diagnosis — double shlex-quoting in
`SlurmCommandRunner._run_via_srun`, possibly truncation — was **wrong**. What actually
happened:

1. **The setup script never ran.** Every saved `run.log` fails right after
   `Waiting for task resources` and never prints `Job started. Streaming logs…`. In
   `SlurmCodeGen` (`backends/task_codegen.py`) that is the *run-reservation* `srun`
   dying before it creates the allocation signal file — a branch that prints the same
   "setup failed" message and sets `FAILED_SETUP`. Setup is detached by default and is
   not marshalled through `_run_via_srun` at all.
2. **The failing command was SkyPilot's own.** That srun runs
   `/bin/bash -c '<SKY_SLURM_PYTHON_CMD> -m sky.skylet.executor.slurm …'` inside the
   container. Before fork commit `5f18669`, `SKY_SLURM_PYTHON_CMD` resolved `env` with
   `$(which env …)`; `srun --export=ALL` re-imports the host's exported `which` shell
   function into the container, a minimal image's `/usr/bin/which` rejects its GNU
   flags and prints `Usage: …` to stdout, and the command bash runs begins with
   `Usage:`. `5f18669` fixed exactly this — **but the local SkyPilot API server**, which
   generates that code for every launch, had been started weeks earlier from a
   *different checkout's* venv with an older fork. Restarting it from this checkout's
   venv removed the failure. `sky api info` reports the commit the server runs.
3. **The image did not provide public DPK.** With the server fixed, the run reached the
   transform and failed `No module named 'dpk_tokenization2arrow'`: `gb_v1` was built
   from IBM-internal DPK, which packages the transform as
   `tokenization2arrow_transform_python`. The `dpk-1.1.8` tag, built from the Dockerfile
   beside `dpk-tok-image/build.yaml` (public `data-prep-toolkit-transforms[tokenization2arrow]==1.1.8`
   plus SkyPilot's container prerequisites), passes with `validate: true`.

Still true from the earlier write-up: BlueVela's enroot already holds credentials for
`cil15-shared-registry` (granite.build supplies none on this path), and any image used
here should pre-install `curl fuse git rsync wget openssh-client` and be Debian/apt-based.

## Related findings from the same validation

- **Memory is enforced per job on BlueVela, but not requested.** gbserver drops
  `compute_config.total_memory_per_node` on slurm/lsf, so the pii job got the partition
  default (`AllocCPUS=2 ReqMem=2G`, from `sacct`) and was OOM-killed loading
  `flair/ner-english-large`. The fixture now sets `launcher_config.resources.memory: "16"`,
  which gbserver passes through.
- **`teardown_skypilot` leaked an allocation per target on SLURM/LSF.** Its `td-`
  cleanup cluster was launched with `down=True`, i.e. SkyPilot autodown, which these
  schedulers do not support, so each one held a `gpu-mid` allocation until downed by
  hand. Fixed in this repo (`src/gbserver/environment/skypilot.py`): on slurm/lsf it now
  downs the `td-` cluster explicitly, even if the `rm -rf` fails. In the validation batch
  all six `td-` jobs ended `COMPLETED` with nothing left in `squeue`.

---

## Appendix: artifacts from the diagnosed run

| Item | Location |
|---|---|
| Build ID | `ca7265ef-a134-4fc7-ac34-f115c2de4bfd` |
| Failing compute node | `p1-r08-n1` |
| Partition | `gpu-mid` (from the env's `zone`) |
| SSH user | `granitebuild` (BlueVela) vs `root` (local Docker SLURM) |

Line numbers reference `granite.build` at commit `b9df4dd` and the installed
`skypilot 1.0.0.dev0`, whose `direct_url.json` records:

```json
{"url": "https://github.com/cmadam/skypilot.git",
 "vcs_info": {"commit_id": "5f18669dc9985f0649147dbcc6bb79d89aeb428d",
              "requested_revision": "gb-sky-v1-stable", "vcs": "git"}}
```

**There is exactly one `sky` install in the venv.** Every SkyPilot test in this repo —
local Docker SLURM, AWS, BlueVela LSF, BlueVela SLURM — runs against that same forked
package. In particular the local `steps/dpk/skypilot/test/slurm-pii` fixture, which
passes, exercises the *identical* `_execute_file_mounts` code path and the *identical*
missing `SlurmCommandRunner` hook. The only variable that differs is the SSH user
(`root` locally vs `granitebuild` on BlueVela), which isolates the cause to the sudo
aliasing described in [Why every existing test missed it](#why-every-existing-test-missed-it).
