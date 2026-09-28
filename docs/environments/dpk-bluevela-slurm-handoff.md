# HANDOFF: dpk step coverage on BlueVela SLURM — resolved, pending a fork PR

**Written for:** an engineer or agent picking this up cold. Assumes familiarity with Python
and pytest, but **not** with granite.build's internals.
**Updated:** 2026-09-28. Supersedes the earlier "two open blockers" version of this file,
whose diagnosis of the image-mode failure was wrong (see
[What actually broke](#3-what-actually-broke)).
**Deep dive:** [skypilot-slurm-file-mounts-sudo.md](skypilot-slurm-file-mounts-sudo.md).

---

## 1. Status

Both dpk fixtures pass on the BlueVela SLURM environment
(`space://environments/skypilot/slurm/bluevela`), alongside the existing sibling tests.
Final validation batch, 2026-09-28: **`5 passed, 3 skipped`** (the skips are the fixtures'
own `runner_cancellation` opt-outs), with no SLURM allocation left behind.

| Test | Path | Result |
|---|---|---|
| `test_dpk_tok_image.py` | image mode (`dpk-1.1.8`, `validate: true`) | PASSED |
| `test_dpk_pii.py` | bare node | PASSED — **needs the fork fix** |
| `test_1step.py` (runner + cancellation) | `command` step, containerized | PASSED |
| `test_2target.py` | `command` step, containerized | PASSED |

**Open:** the bare-mode fix lives in the SkyPilot fork this repo pins. It is committed on a
local branch and must still go to `cmadam/skypilot` and be re-pinned — see
[Remaining work](#5-remaining-work). Until then `test_dpk_pii.py` passes only from a venv
with the patched fork installed.

---

## 2. Background you need

**granite.build pins a FORK of SkyPilot** (`pyproject.toml`):
`skypilot[kubernetes,slurm,aws] @ git+https://github.com/cmadam/skypilot.git@gb-sky-v1-stable`.
Upstream SkyPilot has no LSF cloud, so `LsfCommandRunner` and `get_unwrapped_mount_prefixes`
are granite-build additions in that fork. `gb-sky-v1-stable` is a moving alias, so
re-pointing it affects every consumer and needs the fork owner's coordination.

**The SkyPilot API server is shared per OS user, not per checkout.** Every `sky launch`
goes through one long-lived local daemon, and it generates the code that runs on the
cluster. If it was started from another checkout's venv, launches run *that* venv's fork,
whatever this checkout has installed. Check with `sky api info` (it prints the commit);
fix with `sky api stop` and `sky api start` from this checkout's venv. This was the whole
cause of the image-mode `Usage:` failure below.

**Step and environment resolution.** `space://steps/dpk` and the environment both resolve
from the gb-test space (`space_uri: git+ssh://…/gb-test.git@gbspace-config`); the step comes
from gb-test's base_uri (`assets.git@gbspace-config-dev`, `steps/skypilot/dpk`, byte-identical
to this repo's published copy). The environment sets
`shared_workdir: /proj/data-eng/llmb-read-write/builds/` and, under
`cloud_config.slurm.cluster_configs.bluevela.workdir`, `/proj/data-eng/llmb-read-write` —
the SLURM `workdir` the fork fix exempts.

**Bare vs containerized is the pivotal distinction.** A step with an image
(`dpk_config.dpk_image` / `command_config.image`) runs inside the enroot container as root;
without one it runs on the host as `granitebuild`, which has no passwordless sudo.

---

## 3. What actually broke

Five independent problems, each confirmed from a run log or `sacct`, not inferred:

1. **Stale SkyPilot API server → `Usage:: command not found`, exit 127 (image mode).** The
   server had been running since Aug 25 from `gb_open/.venv` (fork `c8df959`), whose
   container command resolved `env` via `$(which env …)`. `srun --export=ALL` carries the
   host's exported `which` function into the container, where a minimal `/usr/bin/which`
   prints `Usage: …` to stdout, so bash tried to execute `Usage:`. Fork commit `5f18669`
   already fixed this; restarting the server from this checkout's venv applied it. The
   log never showed `Job started`, because it was the run-reservation `srun` failing —
   dpk's setup never ran, so the earlier bisection of the setup script examined code that
   never executed.
2. **Wrong image contents → `No module named 'dpk_tokenization2arrow'`.** `gb_v1` was built
   from IBM-internal DPK (module `tokenization2arrow_transform_python`). Replaced by
   `tokenization-python:dpk-1.1.8`, built from the Dockerfile beside
   `test-data/…/slurm_bluevela/dpk-tok-image/build.yaml` on public DPK 1.1.8.
3. **Bare-mode `file_mounts` sudo wrap → `sudo: a password is required`.** `SlurmCommandRunner`
   lacked the `get_unwrapped_mount_prefixes()` exemption `LsfCommandRunner` has, so the
   remapped `/proj/…/src` destination was `sudo mkdir`'d on the compute node. Fixed in the
   fork: see the deep dive's Resolution section. It also covers `byoc`, the other step
   declaring `src: src`.
4. **Memory not requested → OOM (bare mode).** gbserver drops
   `compute_config.total_memory_per_node` on slurm/lsf, so the pii job got
   `AllocCPUS=2 ReqMem=2G` and was OOM-killed loading `flair/ner-english-large`. The fixture
   now sets `launcher_config.resources.memory: "16"`.
5. **`td-` teardown clusters leaked an allocation per target.** `teardown_skypilot` launched
   its `td-` cleanup cluster with `down=True`, i.e. SkyPilot autodown, which SLURM/LSF do not
   support (the launch path already forces autostop to `None` on those clouds). So every
   target — failed or successful — left a `gpu-mid` job running. A sweep on 2026-09-28
   cancelled 11 of them, the oldest 3 days old. An earlier guess attributed this to a
   trailing dash in the derived cluster name; the code shows the autodown dependency is the
   cause. Fixed in `src/gbserver/environment/skypilot.py`: on slurm/lsf the `td-` cluster is
   now downed explicitly, even if its `rm -rf` fails. In the validation batch all six `td-`
   jobs ended `COMPLETED`.

Also fixed: both dpk tests override pytest-timeout's global `timeout = 1200`
(`pyproject.toml`), which would otherwise kill them below their 60/45-minute budgets.

**Image lessons that still apply.** An earlier bullseye-based `:latest` failed in
`container-init` because `bullseye-security` packages 404'd. Any image used here should be
Debian/apt-based, run as root or grant passwordless sudo, and pre-install SkyPilot's
prerequisites (`curl fuse git rsync wget openssh-client`) so its bootstrap has nothing to
fetch. BlueVela's enroot already holds credentials for `cil15-shared-registry`;
granite.build supplies none on this path.

---

## 4. How to run the tests

```bash
source ./setup.sh     # REQUIRED: exports GB_ENVIRONMENT=STAGING + GBTEST_SPS_IBMCLOUD_API_KEY
sky api info          # the commit must be this checkout's fork (the patched one for pii)
D=test/integration/ibm/buildrunner/skypilot/slurm_bluevela
python -m pytest -s -rA $D/test_dpk_tok_image.py $D/test_dpk_pii.py $D/test_1step.py $D/test_2target.py
```

- `GB_ENVIRONMENT` is read at import time (`src/gbserver/types/constants.py`) and defaults
  to `PROD`, which the harness refuses (`Refusing to run storage tests with
  GB_ENVIRONMENT=PROD.`). Unit tests should also run under STAGING. The VSCode test runner
  does not inherit shell exports and has no `python.envFile`, so IDE-launched tests hit
  this guard.
- `-s` is required: under pytest capture a second `sky launch` in one process dies with
  `OSError: Bad file descriptor`.
- Cost: about 48 minutes of `gpu-mid` for the four files above.

**Why the local Docker SLURM fixtures could not have caught 3–5:** its env logs in as
`User: root`, where SkyPilot aliases `sudo` away; it has no Pyxis, so image mode cannot run
there; and it neither enforces memory nor is shared.

---

## 5. Remaining work

1. **Open the fork PR.** Branch `gb/dpk-bluevela-fixes` of the local fork clone
   (`gb_open_latest/skypilot-fork`, base `5f18669`) carries the `SlurmCommandRunner` hook
   and its tests. PR it to `cmadam/skypilot` (the LSF shared-root work references
   `dawood`, who will know the mount invariants), re-point `gb-sky-v1-stable`, then
   reinstall the pin here and restart the API server.
2. **Run `byoc` on BlueVela SLURM.** Same `src: src` exposure, covered by the same fix, not
   yet run there.
3. **Close the test gap.** A local Docker SLURM variant with a non-root SSH user would make
   this bug class catchable without BlueVela.

---

## 6. Practical notes for whoever continues

- **Read the logs, don't just grep them.** A pytest run here is ~1.1 MB, dominated by a few
  45,000-char Jinja dumps; `cut -c1-400 run.txt > slim.txt` makes it readable. Grepping
  alone caused wrong turns in this investigation.
- **The real workload log is saved locally** — SkyPilot logs
  `Saved job logs to /tmp/sky-logs/<cluster>/job-1`. Its `run.log` is short and is the most
  informative artifact. A failure *before* `Job started. Streaming logs…` is SkyPilot's own
  command, not the step, whatever the message says about "setup".
- **A SUCCEEDED job is not necessarily the dpk step.** The hfpull step runs first on a
  cluster also named after the target (`…-tokenize-…`, `…-redact-…`). Check which step's
  `full_config` preceded the event before reading anything into it.
- **`sacct` on login1 is the ground truth for resources**
  (`sacct -j <id> --format=JobID,JobName,AllocCPUS,ReqMem,MaxRSS,State,ExitCode`).
- **`ugrep` (aliased as grep here) rejects bounded patterns** like `.{0,90}` with "exceeds
  complexity limits". Use a small Python script instead.
- **Check the queue after runs, and identify ownership before cancelling.** Jobs run as the
  shared `granitebuild` account, so the queue mixes everyone's work. Two reliable signals:
  the SkyPilot client-id suffix in the job name (this machine's is `71e892c7`, also shown
  by `sky api info`; a different suffix such as `5170adfa` is another session or machine)
  and the target name from our fixtures (`tokenize` = dpk-tok-image, `redact` = dpk-pii,
  `slurm-run`/`first`/`second` = the siblings). Anything else is somebody else's — e.g. a
  real training job `gdpval-granite42_30b_…` on `gpu-high` was deliberately left alone.
  Never `scancel -u`. When in doubt, report rather than cancel.

  SSH needs no separate credential: `~/.ssh/config` has `Include ~/.sky/generated/ssh/*`,
  whose aliases carry a materialized key.

  ```bash
  ALIAS=$(ls ~/.sky/generated/ssh/ | grep bluevela | head -1)
  ssh "$ALIAS" 'squeue -u granitebuild -o "%.8i %.10P %.12M %.60j %R"'
  ssh "$ALIAS" 'scancel <ids>'
  ```

  A colleague reports `sky down` is not reliable on BlueVela and recommends `scancel`;
  `sky down` / `sky.down(purge=True)` did release every allocation during this validation.
  After a `scancel`, run `sky status --refresh <cluster>` so SkyPilot's local state agrees.
- **Artifacts from this investigation are in `/tmp` and are transient** (`/tmp/dpk-*.txt`,
  `/tmp/bv-final.txt`, `/tmp/sky-logs/`). Green builds: `39d28601`, `bf7d1a76`
  (dpk-tok-image), `deda8eac`, `b6459799` (dpk-pii). Historical failures: `ca7265ef…` (sudo
  wrap), `79ea6f07…` / `4d7c5adb…` (stale API server), `d8cd80c0…` (internal-DPK image),
  `c82e576e…` (OOM).
