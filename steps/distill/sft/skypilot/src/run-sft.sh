#!/bin/bash
#
# PORTED, not authored here. Upstream source of truth:
#   repo   github.ibm.com/Herbert-Woisetschlaeger/gb-steps-collection-post-training
#   path   steps/distill-sft/src/run-sft.sh
#   commit 70c1550a171aa8e09a9ad9047a5bf763c39e8579
#
# Verbatim apart from `black`/`isort` reflow where the file is Python, which CI requires
# repo-wide. Keep it that way so re-syncing upstream stays a three-way merge; behaviour
# changes belong upstream.
#
# The response template's trailing newline is decoded by the STEP TEMPLATE before this is
# called, not here: it cannot cross gbserver's config fill as a real newline, so it travels
# as a two-character escape. That is a property of granite.build's transport, so the
# compensation lives on granite.build's side rather than in this ported file.
#
# distill-sft step entrypoint, INSIDE the container.
#
# This is the CONTROL arm. Every distillation number is reported relative to what this step
# produces, so the failure mode that matters here is not "the run dies" -- it is "the run
# succeeds at something slightly different from the treatment". That is why this file is a
# deliberate near-twin of distill-gold-train/src/run-gold.sh: same part-location logic, same
# kernel preflight, same allocation assertion, same TRK_ convention, same multi-node refusal.
# Where the two differ, the difference is named in a comment. A control that drifts from its
# treatment in an unnamed way is not a control.
#
# WHAT IS DELIBERATELY ABSENT, versus run-gold.sh:
#   - the vLLM server, its ports and its health wait. This step never generates: there is no
#     student rollout, no sampling, no second role.
#   - the topology dispatch on lmbda. There is only one shape here, all-trainer.
#   - --teacher-model-path. The teacher enters this step, if at all, as a DIRECTORY OF
#     PRECOMPUTED LOGITS (--precomputed-logits-dir), never as a live model. With that flag
#     empty this step is plain SFT and the control; with it set it is forward-KL distillation
#     against a frozen teacher. render_sft_config.py refuses the four ways to set it and still
#     train no KD term.
set -uo pipefail

# ---------------------------------------------------------------- locating our own parts
# Identical to run-gold.sh, for the reason given there: STEP_HOME is derived from where THIS
# FILE lives so that the RENDERED step command is runnable against the checkout, which is the
# only way this step gets tested before there is an image (job 1138798).
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STEP_HOME="${STEP_HOME:-$HERE}"
CHECKOUT_ROOT="$(cd -- "$STEP_HOME/../../.." 2>/dev/null && pwd || echo "")"

first_dir() { local c; for c in "$@"; do [[ -d "$c" ]] && { (cd "$c" && pwd); return 0; }; done; return 1; }

# The distillation package: /opt/<step>/vendor/... in the image, src/gb_steps_post_training/...
# in the checkout. SFT_SRC, not GOLD_SRC -- the directory is the same one, but a step that reads
# another step's env var is a step you can misconfigure by exporting the wrong name.
if [[ -z "${SFT_SRC:-}" ]]; then
  SFT_SRC="$(first_dir \
    "$STEP_HOME/vendor/gb_steps_post_training/distillation" \
    "${CHECKOUT_ROOT:-/nonexistent}/src/gb_steps_post_training/distillation" \
  )" || SFT_SRC="$STEP_HOME/vendor/gb_steps_post_training/distillation"
fi
if [[ -z "${LIB_DIR:-}" ]]; then
  LIB_DIR="$(first_dir \
    "$STEP_HOME/lib" \
    "${CHECKOUT_ROOT:-/nonexistent}/scripts/bluevela/lib" \
  )" || LIB_DIR="$STEP_HOME/lib"
fi
# THE PACKAGE ROOT, and it is load-bearing twice over. render_sft_config.py imports
# gb_steps_post_training.distillation.render_common (shared with distill-gold-train so the two
# arms cannot validate by different rules), and sft.py's own import graph reaches
# gb_steps_post_training.distillation at MODULE scope. sft.py is launched FLAT (`cd $SFT_SRC`;
# `accelerate launch sft.py`), which puts the flat directory on sys.path and the package root
# nowhere -- so without this every rank dies during import, before argv is parsed. That is jobs
# 1162604 and 1162625 on the BlueVela path: 49 s of allocation each, zero optimizer steps.
# checks/import-layout.sh asserts this line against the real import graph.
SFT_PKG_ROOT="$(cd -- "$SFT_SRC/../.." 2>/dev/null && pwd || echo "")"
[[ -n "$SFT_PKG_ROOT" ]] && export PYTHONPATH="$SFT_PKG_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYBIN="${PYBIN:-python}"
ACCELERATE="${ACCELERATE:-accelerate}"

# Defaults mirror the step template; every one is overridable by a flag.
STUDENT="" ; CORPUS="" ; OUTPUT_DIR="output"
# The ChatML assistant opener. This is the FALLBACK boundary mechanism, not the primary one --
# see step-template.yaml's response_template comment and render_sft_config.py.
RESPONSE_TEMPLATE="<|im_start|>assistant
"
MAX_LENGTH="4096" ; PDTBS="1" ; GAS="8" ; LR="1e-6" ; EPOCHS="1" ; SAVE_STEPS="100" ; SEED="42"
# auto = sft.py's own presence-based resume, unchanged. See render_common.RESUME_MODES for why
# never/require exist: the trainer resumes on the mere PRESENCE of a checkpoint directory and
# offers no way to ask for anything else, so the only place to state intent is preflight.
RESUME="auto"
DS_CONFIG="" ; EXTRA_YAML="" ; GPN="8" ; NODES="1" ; HF_HOME_ARG=""
LIGER_MEM="0" ; LIGER_SWIGLU="0"
PRECOMPUTED_LOGITS_DIR="" ; KD_TOP_K="256" ; KD_WEIGHT="1.0" ; CE_WEIGHT="0.0"
KD_TEMPERATURE="1.0" ; MAX_DATASET_SIZE="-1"
# Tracking. TRK_ prefix for the reason spelled out in run-gold.sh: unprefixed, these would hold
# the flag values under the EXACT names ClearML and W&B read from the environment, and one stray
# `export` or `set -a` (this script exports HF_HOME below) would hand a library a raw,
# unvalidated project name, bypassing the partial-config abort in tracking.py that is the whole
# point of routing through it.
TRK_CLEARML_PROJECT="" ; TRK_CLEARML_RUN_NAME=""
TRK_WANDB_ENTITY="" ; TRK_WANDB_PROJECT="" ; TRK_WANDB_RUN_NAME=""

while (( $# )); do
  case "$1" in
    --student-model-path)   STUDENT="$2"; shift 2 ;;
    --corpus-path)          CORPUS="$2"; shift 2 ;;
    --output-dir)           OUTPUT_DIR="$2"; shift 2 ;;
    --response-template)    RESPONSE_TEMPLATE="$2"; shift 2 ;;
    --max-length)           MAX_LENGTH="$2"; shift 2 ;;
    --per-device-train-batch-size) PDTBS="$2"; shift 2 ;;
    --gradient-accumulation-steps) GAS="$2"; shift 2 ;;
    --learning-rate)        LR="$2"; shift 2 ;;
    --num-train-epochs)     EPOCHS="$2"; shift 2 ;;
    --save-steps)           SAVE_STEPS="$2"; shift 2 ;;
    --seed)                 SEED="$2"; shift 2 ;;
    --resume)               RESUME="$2"; shift 2 ;;
    --deepspeed-config)     DS_CONFIG="$2"; shift 2 ;;
    # Flag PAIRS, no value. step-template.yaml renders `{% if %}--flag{% else %}--no-flag{% endif %}`
    # because jinja writes a YAML boolean with PYTHON casing, so `--flag {{ x }}` arrives as the
    # literal string "False" and silently does nothing (job 1138651).
    --use-liger-memory-opt)    LIGER_MEM="1"; shift ;;
    --no-use-liger-memory-opt) LIGER_MEM="0"; shift ;;
    --use-liger-swiglu-mlp)    LIGER_SWIGLU="1"; shift ;;
    --no-use-liger-swiglu-mlp) LIGER_SWIGLU="0"; shift ;;
    --precomputed-logits-dir) PRECOMPUTED_LOGITS_DIR="$2"; shift 2 ;;
    --kd-top-k)             KD_TOP_K="$2"; shift 2 ;;
    --kd-weight)            KD_WEIGHT="$2"; shift 2 ;;
    --ce-weight)            CE_WEIGHT="$2"; shift 2 ;;
    --kd-temperature)       KD_TEMPERATURE="$2"; shift 2 ;;
    --max-dataset-size)     MAX_DATASET_SIZE="$2"; shift 2 ;;
    --extra-config-yaml)    EXTRA_YAML="$2"; shift 2 ;;
    --gpus-per-node)        GPN="$2"; shift 2 ;;
    --nodes)                NODES="$2"; shift 2 ;;
    --hf-home)              HF_HOME_ARG="$2"; shift 2 ;;
    --clearml-project)      TRK_CLEARML_PROJECT="$2"; shift 2 ;;
    --clearml-run-name)     TRK_CLEARML_RUN_NAME="$2"; shift 2 ;;
    --wandb-entity)         TRK_WANDB_ENTITY="$2"; shift 2 ;;
    --wandb-project)        TRK_WANDB_PROJECT="$2"; shift 2 ;;
    --wandb-run-name)       TRK_WANDB_RUN_NAME="$2"; shift 2 ;;
    *) echo "FATAL: unknown argument: $1" >&2; exit 2 ;;
  esac
done

echo "=== distill-sft ==="
date; hostname
echo "step home : $STEP_HOME"
echo "sft src   : $SFT_SRC"
echo "lib dir   : $LIB_DIR"
echo "pkg root  : ${SFT_PKG_ROOT:-<unresolved>}"
# WHICH ARM IS THIS. Printed here and not only in the rendered config, because the log is what
# a reader has when they are trying to work out whether the number in front of them is the
# baseline or a distilled run.
if [[ -n "$PRECOMPUTED_LOGITS_DIR" ]]; then
  echo "arm       : forward-KL distillation from PRECOMPUTED logits ($PRECOMPUTED_LOGITS_DIR)"
  echo "            kd_weight=$KD_WEIGHT ce_weight=$CE_WEIGHT top_k=$KD_TOP_K T=$KD_TEMPERATURE"
else
  echo "arm       : plain SFT -- THIS IS THE CONTROL (no teacher, no KD term)"
fi

# An empty hf_home means "leave the image's own default alone". Setting HF_HOME="" would
# instead point the cache at the process CWD.
if [[ -n "$HF_HOME_ARG" ]]; then
  export HF_HOME="$HF_HOME_ARG"
  echo "HF_HOME   : $HF_HOME"
fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# SFT_DRY_RUN=1 STOPS HERE: every argument parsed, nothing done.
#
# WHO ASKS FOR THIS. checks/sft-drift.sh exists to catch a flag the step template passes that this
# launcher no longer accepts -- the class of defect that shipped --hidden-loss-gamma to a config
# nothing read. To catch it, the check has to run the rendered command far enough to prove the arg
# loop consumed everything, and then NOT train. Until now it got its stop by accident: the kernels
# preflight below happened to fail under the check's `uv run` interpreter, so the check never
# reached `mkdir -p "$OUTPUT_DIR"`. One warm cache away from launching a trainer inside a CPU-only
# check suite. This is the deterministic version of that stop.
#
# WHY AN ENV VAR AND NOT A FLAG. The flag surface is a contract with step-template.yaml, and two
# checks assert its exact size; adding a flag would change the thing being measured in order to
# measure it. An env var is invisible to that contract.
#
# WHY 0 AND NOT 2. Unknown arguments already exit 2 above, and run-all.sh maps 2 to UNMEASURED --
# a dry run that succeeded must not be reportable as "could not measure".
#
# WHY BEFORE THE KERNEL PREFLIGHT AND NOT AFTER. The preflight asks about the ENVIRONMENT (is this
# account's HF cache warm), which has nothing to do with whether the arguments are still accepted.
# Downstream of it, this lever would be unreachable on precisely the machines where a drift check
# is cheapest to run -- a login node, a CPU-only suite -- and drift would go unwatched there.
if [[ "${SFT_DRY_RUN:-0}" == "1" ]]; then
  echo "SFT_DRY_RUN=1: arguments parsed, nothing executed."
  echo "dry-run: all arguments accepted"
  exit 0
fi

# Same precondition as run-gold.sh and the same reason: the student is a granitemoehybrid model
# and its Mamba2 hub kernels are resolved at IMPORT time, so with HF_HUB_OFFLINE=1 a cold cache
# is fatal after the GPUs are allocated (job 1136209). It applies identically here -- the
# control loads the same student architecture as the treatment.
#
# ONE REPO MORE THAN run-gold.sh, and the difference is real rather than an oversight there:
# this step ASKS for FA2 and the gold step does not. render_sft_config.py always emits
# `attn_implementation` (sft.py's verify_optimization_stack aborts every rank without it), while
# no gold config sets it and render_gold_config.py never emits it -- so gold loads on whatever
# transformers picks by default and never resolves an FA2 kernel at all.
#
# Asking for it costs a cache entry, because FA2 on this account comes from the kernels hub and
# not from the `flash_attn` package (absent from the venv, unbuildable here: system CUDA 13.1 vs
# torch's cu128). transformers resolves flash_attention_2 to kernels-community/flash-attn2 at
# MODEL-LOAD time, which is offline and after the allocation, so a cold cache is fatal ~30 s in.
# scripts/bluevela/checks/fa2-alias.sh asserts that repo id by reading it out of transformers
# instead of duplicating it here, and asserts it is cached.
if [[ -f "$LIB_DIR/gold-kernels.sh" ]]; then
  source "$LIB_DIR/gold-kernels.sh"
  require_hub_kernels "$PYBIN" kernels-community/flash-attn2 || exit 1
fi

if [[ ! -f "$STEP_HOME/render_sft_config.py" ]]; then
  echo "FATAL: render_sft_config.py not found under STEP_HOME=$STEP_HOME." >&2
  echo "       In the image it is COPYd there from steps/distill-sft/src/; in a" >&2
  echo "       checkout STEP_HOME is that directory. Set STEP_HOME explicitly if this" >&2
  echo "       script was moved away from it." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR" || { echo "FATAL: cannot create output_dir: $OUTPUT_DIR" >&2; exit 1; }

# WHERE THE PREPROCESSING CACHE GOES. sft.py hashes every input the map chain depends on and
# reuses `<root>/<hash>/` across runs -- which is worth real time (the control and the KD arm
# preprocess IDENTICALLY, so the second one is free), but only if the root outlives the process.
# Left to itself sft.py falls back to a path beside its own file: inside src/ in a checkout, and
# inside the container layer in an image, where it is discarded at exit and not shared between
# ranks on separate hosts. The launcher is the only party that knows a persistent location, so
# the launcher names one. OUTPUT_DIR's PARENT rather than OUTPUT_DIR itself, deliberately: two
# runs of this step write different output dirs under the same parent, and sharing the cache
# between them is the entire point -- the hash is what keeps runs that preprocess differently
# from colliding. Overridable, because a recipe may mount something better.
if [[ -z "${SFT_PREPROCESS_CACHE_ROOT:-}" ]]; then
  SFT_PREPROCESS_CACHE_ROOT="$(cd -- "$OUTPUT_DIR" && cd .. && pwd)/sft-preprocess-cache"
fi
export SFT_PREPROCESS_CACHE_ROOT
mkdir -p "$SFT_PREPROCESS_CACHE_ROOT" 2>/dev/null || {
  echo "WARNING: cannot create SFT_PREPROCESS_CACHE_ROOT=$SFT_PREPROCESS_CACHE_ROOT;" >&2
  echo "         sft.py will fall back to a path beside its own file, which is not" >&2
  echo "         persistent in a container. Preprocessing will re-run every time." >&2
}
echo "cache root: $SFT_PREPROCESS_CACHE_ROOT"
RENDERED="$OUTPUT_DIR/sft-config.rendered.yaml"

LIGER_MEM_FLAG=(--no-use-liger-memory-opt)
[[ "$LIGER_MEM" == "1" ]] && LIGER_MEM_FLAG=(--use-liger-memory-opt)
LIGER_SWIGLU_FLAG=(--no-use-liger-swiglu-mlp)
[[ "$LIGER_SWIGLU" == "1" ]] && LIGER_SWIGLU_FLAG=(--use-liger-swiglu-mlp)

"$PYBIN" "$STEP_HOME/render_sft_config.py" \
  --student-model-path "$STUDENT" \
  --corpus-path "$CORPUS" \
  --output-dir "$OUTPUT_DIR" \
  --response-template "$RESPONSE_TEMPLATE" \
  --max-length "$MAX_LENGTH" \
  --per-device-train-batch-size "$PDTBS" \
  --gradient-accumulation-steps "$GAS" \
  --learning-rate "$LR" --num-train-epochs "$EPOCHS" \
  --save-steps "$SAVE_STEPS" --seed "$SEED" \
  --resume "$RESUME" \
  --deepspeed-config "$DS_CONFIG" \
  "${LIGER_MEM_FLAG[@]}" \
  "${LIGER_SWIGLU_FLAG[@]}" \
  --precomputed-logits-dir "$PRECOMPUTED_LOGITS_DIR" \
  --kd-top-k "$KD_TOP_K" --kd-weight "$KD_WEIGHT" \
  --ce-weight "$CE_WEIGHT" --kd-temperature "$KD_TEMPERATURE" \
  --max-dataset-size "$MAX_DATASET_SIZE" \
  --extra-config-yaml "$EXTRA_YAML" \
  --gpus-per-node "$GPN" --nodes "$NODES" \
  --clearml-project "$TRK_CLEARML_PROJECT" \
  --clearml-run-name "$TRK_CLEARML_RUN_NAME" \
  --wandb-entity "$TRK_WANDB_ENTITY" \
  --wandb-project "$TRK_WANDB_PROJECT" \
  --wandb-run-name "$TRK_WANDB_RUN_NAME" \
  --out "$RENDERED" || exit $?

# ------------------------------------------------------------- allocation assertion
# Byte-for-byte the same rule as run-gold.sh, and here it is load-bearing for an extra reason:
# gpus_per_node is a factor of the effective batch size, so a control that silently trains at 2
# GPUs while the treatment trained at 8 differs from it by 4x in global batch -- and neither run
# fails. CUDA_VISIBLE_DEVICES wins over nvidia-smi when set, because a scheduler that grants 2
# of a host's 8 GPUs restricts it that way and `nvidia-smi -L` would report all eight.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  VISIBLE=$(awk -F, '{n=0; for(i=1;i<=NF;i++) if(length($i)) n++; print n}' <<<"$CUDA_VISIBLE_DEVICES")
  VISIBLE_SRC="CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
  VISIBLE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true)
  VISIBLE_SRC="nvidia-smi -L"
else
  VISIBLE=""
  VISIBLE_SRC=""
fi

if [[ -n "$VISIBLE" ]]; then
  echo "allocation: $VISIBLE GPU(s) visible (via $VISIBLE_SRC), gpus_per_node=$GPN"
  if (( VISIBLE != GPN )); then
    echo "FATAL: gpus_per_node=$GPN but $VISIBLE GPU(s) are visible (via $VISIBLE_SRC)." >&2
    echo "       Training at a different width than the config claims is silent: the loss" >&2
    echo "       curve looks plausible and the effective batch size is wrong by a factor" >&2
    echo "       of $GPN/$VISIBLE. For THIS step that also silently breaks the comparison" >&2
    echo "       every distillation arm is reported against. Fix the profile's ACCELERATORS" >&2
    echo "       or the recipe's GPUS_PER_NODE so the two agree." >&2
    exit 1
  fi
else
  echo "WARNING: cannot count GPUs (no CUDA_VISIBLE_DEVICES, no nvidia-smi)." >&2
  echo "         gpus_per_node=$GPN is UNVERIFIED against the real allocation." >&2
fi

# ------------------------------------------------------------------ launch
# One shape only. There is no lmbda dispatch here because there is no generation: whether this
# step is the control or the KD arm changes the LOSS, not the topology.
NUM_PROCESSES=$(( NODES * GPN ))
if (( NODES > 1 )); then
  # The same D3 gap run-gold.sh refuses on, reached by a shorter road: a step gets no host list,
  # so a multi-node accelerate launch has no main_process_ip, no port and no machine_rank per
  # host. Refusing beats launching one node while reporting NODES, which would produce a control
  # trained at 1/NODES of the declared global batch and report it as the declared one.
  echo "FATAL: nodes=$NODES needs a multi-node rendezvous (main_process_ip/port and a" >&2
  echo "       machine_rank per host) that this step is not handed. Use nodes=1 here, or run" >&2
  echo "       it inside an LSF allocation where LSB_HOSTS supplies the host list. Tracked" >&2
  echo "       as D3 -- see distill-gold-train/src/run-gold.sh for the full statement." >&2
  exit 1
fi
echo "topology  : all-trainer, 1 node x $GPN GPU(s)"
cd "$SFT_SRC" || exit 1
"$ACCELERATE" launch \
  --num_processes "$NUM_PROCESSES" \
  --num_machines 1 \
  --config_file "$DS_CONFIG" \
  sft.py --config "$RENDERED" --output_dir "$OUTPUT_DIR"
RC=$?

echo
echo "trainer rc=$RC  output=$OUTPUT_DIR"
date
echo "=== done (rc=$RC) ==="
exit "$RC"
