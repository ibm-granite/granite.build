# Sourced by the GOLD launchers. Defines require_hub_kernels.
#
# THE PROBLEM THIS SOLVES. gold.py pre-populates transformers' hub-kernel mapping at
# module scope with local_files_only=True and, on a cache miss, deliberately falls
# through to the ONLINE lazy_load_kernel path -- its own comment says "preflight will
# repopulate the cache". The recorded scripts satisfied that with an unconditional Hub
# download inside every GPU allocation. This repo made the download opt-in
# (GOLD_PREFETCH_KERNELS), which removed the download and, unnoticed, the invariant with
# it. A direct measurement therefore reached the GPU, started vLLM, fanned out to the trainer, and
# only then died ~90s in with a 40-line traceback ending:
#
#   huggingface_hub.errors.OfflineModeIsEnabled: Cannot reach
#   https://huggingface.co/api/models/kernels-community/causal-conv1d/refs
#
# A missing cache entry is a preflight-detectable condition, so it belongs in preflight,
# where it costs seconds and names its own fix. Three-way gate:
#
#   cached                        -> proceed silently
#   missing, GOLD_PREFETCH_KERNELS=1 -> download now (needs egress), then proceed
#   missing, otherwise            -> FATAL, naming prefetch-kernels.sh
#
# WHICH KERNELS ARE REQUIRED. causal-conv1d and mamba-ssm only. These are the Mamba2
# mixer kernels a Granite 4 hybrid model actually needs, and they are exactly the two
# gold.py's preamble preloads. The recorded prefetch also pulled
# kernels-community/vllm-flash-attn3, which this deliberately does NOT require: FA3 is
# reachable only through the vendored GraniteSWAAttention path, which is gated behind
# GOLD_SWA_ARM (see src/gb_steps_post_training/distillation/_swa_arm.py). Requiring it
# would make every run depend on fetching a kernel it never calls.
#
# EXTRA REPOS, per entrypoint. The two Mamba2 kernels above are required by every
# entrypoint in this collection, so they are the defaults. distill-sft-baseline adds a third,
# and the reason is worth stating because it is invisible from the config: sft.py's
# verify_optimization_stack REQUIRES flash_attention_2 on a non-SWA student and aborts every
# rank otherwise -- and transformers 5.x, when the `flash_attn` PACKAGE is absent, serves FA2
# out of the kernels hub instead (modeling_utils.py:1898 -> FLASH_ATTN_KERNEL_FALLBACK
# ['flash_attention_2'] == 'kernels-community/flash-attn2'). This account has no flash_attn
# package, so that fallback is the only FA2 there is here, and a cold cache means the same
# OfflineModeIsEnabled failure a direct measurement paid for -- this time after the corpus is
# tokenized.
#
# Usage: require_hub_kernels "$PYBIN" [extra/repo ...] || exit 1

require_hub_kernels() {
  local pybin="$1"; shift || true
  local out rc
  # Passed to the probe through the environment, not by interpolating into the heredoc: the
  # heredocs below are QUOTED ('PY'), which is what keeps the probe from being rewritten by
  # the calling shell.
  local repos=("kernels-community/causal-conv1d" "kernels-community/mamba-ssm" "$@")
  local repo_list; repo_list="${repos[*]}"

  # Probe with local_files_only so the probe itself can never reach the network --
  # otherwise a warm-looking probe would just be the failure we are trying to detect,
  # relocated.
  # 2>/dev/null, not 2>&1: the probe reports missing repos on stdout, so folding stderr
  # in only prepends unrelated site-packages warnings to the message.
  out=$(HF_HUB_OFFLINE=1 REQUIRED_KERNEL_REPOS="$repo_list" "$pybin" - <<'PY' 2>/dev/null
import json, os, sys
from pathlib import Path
# EXIT 2 MEANS "THE PROBE COULD NOT ASK", and it is kept distinct from exit 1 ("asked, and repos
# are missing") because conflating the two produced a message that named the wrong fix. See the
# shell below.
try:
    from kernels.utils import install_kernel
except Exception as e:
    print("PROBE-BROKEN %s: %s" % (type(e).__name__, e))
    sys.exit(2)

missing = []
for repo in os.environ["REQUIRED_KERNEL_REPOS"].split():
    try:
        _, path = install_kernel(repo, revision="main", local_files_only=True)
        json.loads((Path(path) / "metadata.json").read_text())
    except Exception as e:
        missing.append(f"{repo} ({type(e).__name__})")
print(" ".join(missing))
sys.exit(1 if missing else 0)
PY
  )
  rc=$?

  if (( rc == 0 )); then
    echo "hub kernels: cached (${repos[*]})"
    return 0
  fi

  # A CRASHED PROBE IS NOT A COLD CACHE, and telling them apart is worth these lines.
  #
  # How this was found: a companion drift check renders the step command and runs it,
  # and its run ended in
  #
  #     hub kernels: NOT cached ->
  #     FATAL: required hub kernels are not in this account's HF cache:
  #
  # with NOTHING after either colon. The probe reports missing repos on stdout, and this function
  # runs it with 2>/dev/null for the good reason above -- so a probe that died before printing
  # anything (no `kernels` module, wrong interpreter, broken install) arrived here as rc!=0 with an
  # empty list, and was reported as a cache miss OF NOTHING, answered with a prefetch recipe that
  # cannot possibly fix it. Verified cause in that case: the launcher was exec'd under
  # `uv run --no-project --with jinja2 --with pyyaml`, PYBIN defaults to a bare `python`, and there
  # that resolved to an interpreter with jinja2 and pyyaml and no `kernels` at all.
  #
  # The empty list was the tell. On a real GPU job that message would have sent whoever read it to
  # run a download inside an allocation -- exactly the cost this preflight exists to avoid paying.
  # Same class as the two other fixes in this tree tonight: a condition of the ENVIRONMENT must not
  # be reported as a defect in the SUBJECT. See the companion require-gpu.sh helper.
  if (( rc == 2 )) || [[ -z "${out//[[:space:]]/}" ]]; then
    cat >&2 <<MSG
FATAL: the hub-kernel PROBE could not run, so whether the cache is warm is UNKNOWN.
  ${out:-(the probe produced no output at all)}

  This is not a cache miss, and prefetching will not fix it: the probe failed before it could ask.
  It ran as: $pybin
  Check that this interpreter has the \`kernels\` package. PYBIN defaults to a bare \`python\`,
  which resolves to whatever is first on PATH -- under \`uv run --no-project --with ...\` that is an
  ephemeral environment holding only what was asked for. Pass PYBIN=<the image's python>, or run the
  launcher under the interpreter that has torch.
MSG
    return 1
  fi

  echo "hub kernels: NOT cached -> $out"

  if [[ "${GOLD_PREFETCH_KERNELS:-0}" == "1" ]]; then
    echo "GOLD_PREFETCH_KERNELS=1: downloading now (needs egress to huggingface.co)..."
    HF_HUB_OFFLINE=0 REQUIRED_KERNEL_REPOS="$repo_list" "$pybin" - <<'PY'
import json, os, sys
from pathlib import Path
from kernels.utils import install_kernel
rc = 0
for repo in os.environ["REQUIRED_KERNEL_REPOS"].split():
    try:
        _, path = install_kernel(repo, revision="main")
        json.loads((Path(path) / "metadata.json").read_text())
        print("  ok", repo, path)
    except Exception as e:
        print(f"  FAIL {repo}: {type(e).__name__}: {e}")
        rc = 1
sys.exit(rc)
PY
    rc=$?
    if (( rc != 0 )); then
      echo "FATAL: in-job kernel prefetch failed (rc=$rc)."
      return 1
    fi
    echo "hub kernels: downloaded"
    return 0
  fi

  cat <<MSG
FATAL: required hub kernels are not in this account's HF cache:
  $out

  The Mamba2 pair is resolved at gold.py import time. With HF_HUB_OFFLINE=1 and a cold
  cache that raises OfflineModeIsEnabled *after* the allocation, with the vLLM server and
  the trainer fan-out already up -- which is what a direct measurement spent ~90 GPU-seconds
  proving. kernels-community/flash-attn2, when listed, is transformers' FA2 fallback for
  an environment with no flash_attn package, and sft.py aborts every rank without it.

  Fix once, out of band (idempotent, and the cache is shared across all future runs):

    <submit one GPU job through your scheduler> \
         -J prefetch-kernels -o data/logs/prefetch-kernels.%J.out \\
         -e data/logs/prefetch-kernels.%J.err \\
         bash prefetch-kernels.sh

  A companion prefetch script (not included in this repo) fetches the same list this check
  requires, including any extra repos passed via KERNEL_REPOS -- for the FA2 fallback:

    KERNEL_REPOS=kernels-community/flash-attn2 bash prefetch-kernels.sh

  Or set GOLD_PREFETCH_KERNELS=1 to download inside this job instead. That works, but
  it spends allocated GPU time on a network fetch and re-does it on every run.
MSG
  return 1
}
