#!/usr/bin/env bash
#
# PORTED, not authored here. Upstream source of truth:
#   repo   github.ibm.com/Herbert-Woisetschlaeger/gb-steps-collection-post-training
#   path   steps/distill-tokenizer-align/src/run-align.sh
#   commit 70c1550a171aa8e09a9ad9047a5bf763c39e8579
#
# The ONLY intentional divergence from upstream is the artifact marker prefix:
# LLMB_ARTIFACT_* -> GB_ARTIFACT_*. granite.build's shipped monitor accepts both
# (its line_regex is "(?:GB_|LLMB_)ARTIFACT_ID:..."), so this is convention, not
# necessity — but GB_ is what every other step in this repo prints.
#
# Keep this file a near-verbatim copy so re-syncing upstream stays a three-way merge
# rather than an eyeball diff. Behaviour changes belong upstream, not here.
#
# Tokenizer alignment: build the teacher and student overlays, then retag the student
# onto the teacher's tokenizer.
#
# THE THREE STAGES ARE INDEPENDENT, and an earlier version of this header claimed
# otherwise. It said the order was load-bearing because the retag had to read the teacher's
# tokenizer from the overlay rather than from the teacher directory. That was wrong, and
# executing the rendered step command (LSF job 1137372) proved it: stage 3 died on a
# missing config.json, because an overlay is a tokenizer and retag_student needs a MODEL --
# vocab_size and the bos/eos/pad id scheme live in config.json and nowhere else.
#
# The reason the overlay bought stage 3 nothing is worth stating, because it is the same
# reason the fix is safe: retag_student.py never goes through AutoTokenizer. It reads
# tokenizer.json as raw JSON, copies three tokenizer files, and pins tokenizer_class on its
# own output. So it is immune BY CONSTRUCTION to the trap the overlay exists to dodge --
# a Granite dir's tokenizer_config.json declares `tokenizer_class: "GPT2Tokenizer"`, so
# AutoTokenizer builds THAT class, which rebuilds its backend from vocab+merges and
# installs a plain ByteLevel(use_regex=True) -- discarding whatever pre_tokenizer
# tokenizer.json stored. It does not error, and it still reports is_fast=True: CLASS
# IDENTITY is the mechanism, not fast-versus-slow. Stage 3 takes "$TEACHER_MODEL" for
# that reason, and the overlays exist for the DOWNSTREAM consumers that do load a
# tokenizer through transformers.
#
# Two things NOT to infer from that (both measured -- jobs 1136957/1137115/1137253, and
# see docs/tokenizer_mismatch.md):
#   - It is not the legacy vocab.json/merges.txt sidecars. Under transformers 5.8.0 those
#     are inert; the overlay excludes them for hygiene, not for protection.
#   - Deleting the key is not sufficient either. With a config.json present, the
#     `model_type: granite` fallback through TOKENIZER_MAPPING_NAMES revives the override
#     even with the key absent -- so build_overlay PINS it to PreTrainedTokenizerFast.
#
# Worth knowing while reading the two stages below: on THIS teacher the override happens
# to be a no-op, because granite-4.2's own tokenizer.json pre_tokenizer is already a plain
# ByteLevel. The bug bites the 4.1 base STUDENT, whose pre_tokenizer is
# Sequence[Split(regex), ByteLevel] -- the override discards that Split. Both overlays are
# built anyway: the teacher's costs nothing and stops the asymmetry from being load-bearing.
#
# WHAT THE PIN DOES NOT DO, because this is the trap one family up. Pinning the class stops
# the lookup from replacing the rule stored in tokenizer.json; it does NOT decide whether
# that stored rule is the one the model was TRAINED with. For the 4.1/4.2 family it is,
# which is why the pin is the whole fix for this pairing. For granite-5.0-20b-sft it is
# NOT: its stored Sequence[Split(regex), ByteLevel(use_regex=False)] is vestigial, and the
# model's own likelihood prefers the imposed plain ByteLevel by 17.0-19.8% of TOTAL NLL
# over 512 documents (jobs 1857118, 1857242), so the pin ALONE turned a working directory
# into one that cost an 8-GPU arm at 0/24 steps. Such a teacher needs the pin PLUS a
# pre_tokenizer transplant (scripts/bluevela/build-model-mirror.py --pre-tokenizer-from),
# and which rule it was trained with is a MEASUREMENT, not a reading of its files:
# scripts/bluevela/compare-tokenizer-nll.py, ranked on TOTAL NLL and never PPL/token --
# two pre-split rules emit different token counts, so a per-token mean is not comparable
# across them, and the older 26.1-vs-3.29 band does not transfer because that was a slow
# class rebuilding the MERGES.
#
# WHY ONE STEP AND NOT TWO. Overlay-building and retagging are both tokenizer alignment
# against the same teacher and they are always run as a pair -- every consumer that wants
# a retagged student also wants the overlays, because they have to agree about the
# tokenizer. (Not, as this comment previously said, because the retag consumes the overlay:
# it does not, see above.) Split apart they would be two steps that a recipe has to wire
# to the same teacher and that nothing would check had been.
set -euo pipefail

STUDENT_MODEL=""
TEACHER_MODEL=""
OUT_DIR=""
CHAT_TEMPLATE=""
COPY_MODE="copy"
VERIFY="true"
DRY_RUN="false"
# Whether the TEACHER and the retagged student must speak ChatML. Default true, which is
# what upstream hard-codes and what the granite-4.2 reference pair needs. It is a knob
# because the markup family is a property of the PAIR, not of alignment: granite 4.0/4.1
# carry <|start_of_role|>/<|end_of_role|> and no <|im_start|> at all, so a granite-4.1
# teacher fails stage [1/4] under the upstream default -- the run never starts.
REQUIRE_CHATML="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --student-model)  STUDENT_MODEL="$2"; shift 2 ;;
    --teacher-model)  TEACHER_MODEL="$2"; shift 2 ;;
    --out-dir)        OUT_DIR="$2"; shift 2 ;;
    --chat-template)  CHAT_TEMPLATE="$2"; shift 2 ;;
    --copy-mode)      COPY_MODE="$2"; shift 2 ;;
    --require-chatml)    REQUIRE_CHATML="true"; shift ;;
    --no-require-chatml) REQUIRE_CHATML="false"; shift ;;
    --verify)         VERIFY="true"; shift ;;
    --no-verify)      VERIFY="false"; shift ;;
    --dry-run)        DRY_RUN="true"; shift ;;
    --no-dry-run)     DRY_RUN="false"; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

for name in STUDENT_MODEL TEACHER_MODEL OUT_DIR; do
  if [[ -z "${!name}" ]]; then
    echo "ERROR: --${name,,} is required" | tr '_' '-' >&2
    exit 2
  fi
done

PYBIN="${PYBIN:-python}"
# Both entrypoints live in the shared distillation package, which the Dockerfile puts on
# PYTHONPATH -- the same arrangement steps/granite42-sft-train uses for
# megatron_bridge_granite. Invoked with -m so the package's own imports resolve.
PKG="gb_steps_post_training.distillation"

TEACHER_OVERLAY="${OUT_DIR}/teacher_overlay"
STUDENT_OVERLAY="${OUT_DIR}/student_overlay"
RETAGGED="${OUT_DIR}/retagged_student"

verify_flag() { [[ "$VERIFY" == "true" ]] && echo "--verify" || echo "--no-verify"; }
chatml_flag() { [[ "$REQUIRE_CHATML" == "true" ]] && echo "--require-chatml" || echo "--no-require-chatml"; }

echo "=== distill-tokenizer-align ==="
echo "  student : ${STUDENT_MODEL}"
echo "  teacher : ${TEACHER_MODEL}"
echo "  out     : ${OUT_DIR}"
echo "  verify  : ${VERIFY}   dry-run: ${DRY_RUN}   copy-mode: ${COPY_MODE}"
echo "  chatml  : ${REQUIRE_CHATML}"

# The three artifact lines are printed from ONE place, because a step that reports SKIP still has
# to hand its consumers the paths -- a resumed recipe whose step 2 says "already done" and then
# publishes nothing has broken step 3, which is a worse failure than re-doing the work.
publish_artifacts() {
  echo "GB_ARTIFACT_ID:retagged_student GB_ARTIFACT_PATH:${RETAGGED}"
  echo "GB_ARTIFACT_ID:teacher_overlay GB_ARTIFACT_PATH:${TEACHER_OVERLAY}"
  echo "GB_ARTIFACT_ID:student_overlay GB_ARTIFACT_PATH:${STUDENT_OVERLAY}"
}

# ------------------------------------------------------------------- resume
# A recipe on the preemptable queue is RESTARTED, not resumed: the restart re-runs every step
# from the top, so each step has to answer "has this already been done, and with what?" for
# itself. align_state.py is this step's answer -- see its docstring for which facts about the
# student, the teacher and the template are compared and why paths are not among them.
#
#   0  RUN     nothing recorded here; do the work
#  64  SKIP    recorded under this EXACT expectation; publish and exit
#  65  REFUSE  recorded under a different one, or a declared output is gone. The message names
#              the key. Nothing is overwritten and nothing is silently rebuilt: this step's
#              output is what four later steps compare their tokenizers against, so quietly
#              replacing it with something built from different inputs is the expensive mistake.
#
# The gate is SKIPPED entirely under --dry-run, and that is deliberate rather than incidental: a
# dry run writes no artifacts, so it must neither consult a marker (it would report SKIP for work
# it did not do) nor write one (it would claim outputs that do not exist).
STATE_ARGS=(
  --student-model "$STUDENT_MODEL"
  --teacher-model "$TEACHER_MODEL"
  --out-dir "$OUT_DIR"
  --copy-mode "$COPY_MODE"
)
[[ -n "$CHAT_TEMPLATE" ]] && STATE_ARGS+=(--chat-template "$CHAT_TEMPLATE")
[[ "$VERIFY" == "true" ]] && STATE_ARGS+=(--verify) || STATE_ARGS+=(--no-verify)

if [[ "$DRY_RUN" != "true" ]]; then
  echo
  echo "--- [0/3] resume check"
  # `|| RC=$?` rather than a bare call: `set -e` would abort on 64, which is not an error but
  # the whole point of asking.
  STATE_RC=0
  "$PYBIN" -m "${PKG}.align_state" check "${STATE_ARGS[@]}" || STATE_RC=$?
  case "$STATE_RC" in
    0)  ;;
    64) echo
        echo "nothing to do: this out-dir is already aligned under this expectation."
        publish_artifacts
        exit 0 ;;
    65) echo
        echo "ERROR: refusing to overwrite ${OUT_DIR}. See the key named above." >&2
        exit 1 ;;
    *)  echo
        echo "ERROR: the resume check itself failed (exit ${STATE_RC}). Fix that before" >&2
        echo "       running the step: a step that cannot tell whether it has already run" >&2
        echo "       must not guess." >&2
        exit 1 ;;
  esac
fi

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------- already-retagged input
# retag_student.py writes retag_manifest.json into its OWN --out directory (see its tail),
# naming the student and teacher it was given. So that file's presence in STUDENT_MODEL is
# not a guess -- it is retag_student.py's own signature -- and it means STUDENT_MODEL is
# already somebody's retagged_student output, not a raw base model.
#
# Retagging such a directory AGAIN is not idempotent: the second retag's chat_template
# install lands on a tokenizer that may already carry a different installed template from
# its first retag, and stage [4/4]'s masking-contract derivation can then disagree with
# itself about where assistant spans start (measured: probe "two assistant turns, mixed
# [thinking=True]" disagreed at [96] vs [96, 158] retagging an already-retagged
# granite-4.1-3b-base against granite-4.2-30b). The fix is not to make the second retag's
# verification pass -- it is to never do a second retag: reuse the recorded output as-is.
#
# [4/4] masking contract still RUNS here, unlike the retag -- it is not skipped. It only
# reads the tokenizer that is about to be reused and derives masking.json from whatever
# template is already installed; it writes nothing back into the tokenizer, so it carries
# none of the double-retag risk above. align_state.declared_outputs() also requires it
# unconditionally whenever a chat_template is given (see align_state.py), so a retagged
# student pre-dating this masking step (recorded via retag_manifest.json but built before
# masking.json existed) would otherwise fail mark() with a FileNotFoundError.
if [[ -f "${STUDENT_MODEL}/retag_manifest.json" ]]; then
  echo
  echo "--- [skip] ${STUDENT_MODEL} already carries retag_manifest.json"
  echo "  This student is already a distill-tokenizer-align output, not a raw base model."
  echo "  Skipping [3/4] retag -- re-retagging an already-retagged student is not"
  echo "  idempotent and can produce a masking contract that disagrees with itself."
  echo "  Reusing ${STUDENT_MODEL} as this run's retagged_student unchanged."
  case "$COPY_MODE" in
    copy)     cp -a "${STUDENT_MODEL}/." "${RETAGGED}/" ;;
    hardlink) mkdir -p "${RETAGGED}"; cp -al "${STUDENT_MODEL}/." "${RETAGGED}/" ;;
    *)        echo "ERROR: unknown --copy-mode ${COPY_MODE}" >&2; exit 2 ;;
  esac

  echo
  echo "--- [skip] teacher overlay -> ${TEACHER_OVERLAY}"
  "$PYBIN" -m "${PKG}.build_overlay" \
    --source "$TEACHER_MODEL" --out "$TEACHER_OVERLAY" \
    --copy-mode "$COPY_MODE" "$(verify_flag)" "$(chatml_flag)"

  echo
  echo "--- [skip] student overlay -> ${STUDENT_OVERLAY}"
  "$PYBIN" -m "${PKG}.build_overlay" \
    --source "$STUDENT_MODEL" --out "$STUDENT_OVERLAY" \
    --copy-mode "$COPY_MODE" "$(verify_flag)" --no-require-chatml

  # tokenizer_identity.write() was added to retag_student.py after some already-retagged
  # students (this one included) were produced, so retag_manifest.json's presence does not
  # guarantee tokenizer_identity.json's -- and align_state.declared_outputs() requires the
  # latter unconditionally (measured: FileNotFoundError on exactly this file). Backfilling
  # it needs no retag: the identity is a name derived from the TEACHER plus a hash of the
  # tokenizer already sitting in $RETAGGED, so this only records a fact about bytes that
  # are not changing here.
  if [[ ! -f "${RETAGGED}/tokenizer_identity.json" ]]; then
    echo
    echo "--- [skip] backfilling tokenizer_identity.json (absent on this older retag)"
    "$PYBIN" -c "
from pathlib import Path
from gb_steps_post_training.distillation import tokenizer_identity
identity = tokenizer_identity.derive_name(Path('${TEACHER_MODEL}'))
dest = tokenizer_identity.write(
    Path('${RETAGGED}'), identity,
    produced_by='distill-tokenizer-align/run-align.sh (skip path)',
    student='${STUDENT_MODEL}',
    teacher='${TEACHER_MODEL}',
)
print(f'  recorded tokenizer identity {identity!r} -> {dest}')
"
  fi

  if [[ -n "$CHAT_TEMPLATE" ]]; then
    echo
    echo "--- [4/4] masking contract -> ${RETAGGED}/masking.json"
    "$PYBIN" -m "${PKG}.masking" emit --tokenizer "$RETAGGED"
  else
    echo "--- [4/4] masking contract: SKIPPED, no chat template was installed" >&2
  fi

  echo
  echo "--- marking complete"
  "$PYBIN" -m "${PKG}.align_state" mark "${STATE_ARGS[@]}"

  publish_artifacts
  exit 0
fi

echo
echo "--- [1/4] teacher overlay -> ${TEACHER_OVERLAY}"
"$PYBIN" -m "${PKG}.build_overlay" \
  --source "$TEACHER_MODEL" --out "$TEACHER_OVERLAY" \
  --copy-mode "$COPY_MODE" "$(verify_flag)" "$(chatml_flag)"

# The student overlay is NOT consumed by the retag. It exists because the plan's coupling
# note is real and now measured (LSF job 1136957): the teacher's tokenizer.json carries a
# plain ByteLevel pre_tokenizer while the student's carries Sequence[Split(regex),ByteLevel],
# so the SAME TEXT segments differently, and corpus-prep needs a trustworthy student
# tokenizer to compare against when it asserts one-tokenizer-per-run.
#
# --no-require-chatml, and it MUST be that way round. This overlay is built from the
# PRE-RETAG base student, whose vocabulary contains no <|im_start|>/<|im_end|> at all --
# `<|im_start|>` comes back from the raw backend as [27, 91, 318, 5011, 91, 29]. That is
# not a defect for a base model to fix; it is the reason stage 3 exists. Demanding ChatML
# here fails a correct artifact. Demanding it on the teacher (stage 1) and on the retagged
# student (post-condition below) is what actually catches regressions.
echo
echo "--- [2/4] student overlay -> ${STUDENT_OVERLAY}"
"$PYBIN" -m "${PKG}.build_overlay" \
  --source "$STUDENT_MODEL" --out "$STUDENT_OVERLAY" \
  --copy-mode "$COPY_MODE" "$(verify_flag)" --no-require-chatml

echo
echo "--- [3/4] retag student -> ${RETAGGED}"
RETAG_ARGS=(
  --student "$STUDENT_MODEL"
  # The teacher MODEL directory, not "$TEACHER_OVERLAY" -- see the header. retag_student
  # needs vocab_size and the bos/eos/pad ids, which only a model config carries, and it
  # gains nothing from the overlay because it never loads a tokenizer through
  # transformers. It refuses with an instruction rather than a traceback if handed an
  # overlay anyway.
  --teacher "$TEACHER_MODEL"
  --out "$RETAGGED"
  --copy-mode "$COPY_MODE"
)
if [[ -n "$CHAT_TEMPLATE" ]]; then
  RETAG_ARGS+=(--chat-template "$CHAT_TEMPLATE")
else
  # Not defaulted silently: gold/ has no config knob for a chat template, sft.py reads it
  # off the tokenizer, and a template without {% generation %} markers makes
  # assistant_masks all-zero, which raises at sft.py:909. A student directory with no
  # template at all means apply_chat_template has nothing to apply. So an absent template
  # is a decision the caller must make explicitly.
  RETAG_ARGS+=(--no-chat-template)
  echo "  NOTE: no --chat-template given, passing --no-chat-template." >&2
  echo "  The resulting student has NO chat template. GOLD training will fail to build" >&2
  echo "  assistant_masks from it (sft.py:909). This is only correct if a later step" >&2
  echo "  installs one." >&2
fi
[[ "$DRY_RUN" == "true" ]] && RETAG_ARGS+=(--dry-run)

"$PYBIN" -m "${PKG}.retag_student" "${RETAG_ARGS[@]}"

# Post-condition on the step's PRIMARY output. The retag's whole purpose is to make the
# student able to represent the TEACHER'S turn boundaries, so assert it did: under
# --require-chatml that means <|im_start|> and <|im_end|> must now be single ids. Either
# way the pre_tokenizer must still be the trained one, which is the half of verify() that
# is family-independent and the half that catches the expensive mistake.
#
# Under --no-require-chatml the marker half of this check is not merely skipped, it moves:
# for a same-family pair (granite 4.0 student, granite 4.1 teacher) the vocabularies are
# already identical, so there are no new turn tokens to assert. What must hold instead is
# that the installed template's masking contract can be DERIVED, and stage [4/4] below is
# that assertion -- it runs unconditionally and fails the step if the marker cannot be
# recovered from the template. So do not read a false here as "the boundary is unchecked".
# This is the check that would have caught retag v1, where the mean-initialised EOS row
# left a model that trained fine and could not emit EOS.
#
# verify() is reused rather than reimplemented, and it is pointed at the retag OUTPUT, not
# at an overlay -- retag_student.py writes no sidecars and already sets
# tokenizer_class=PreTrainedTokenizerFast, so the output satisfies verify()'s preconditions
# directly (confirmed: job 1137161 section 5 had nothing left to force). This is also the
# only place in the step where the retag's output is loaded through transformers at all,
# which is precisely why it is worth doing here rather than trusting the pin.
if [[ "$VERIFY" == "true" && "$DRY_RUN" != "true" ]]; then
  echo
  if [[ "$REQUIRE_CHATML" == "true" ]]; then
    echo "--- post-condition: retagged student speaks ChatML"
  else
    echo "--- post-condition: retagged student tokenizer is intact (chatml not required)"
  fi
  "$PYBIN" - "$RETAGGED" "$REQUIRE_CHATML" <<'PYCHECK'
import sys
from pathlib import Path
from gb_steps_post_training.distillation.build_overlay import verify, OverlayError
try:
    for line in verify(Path(sys.argv[1]), require_chatml=sys.argv[2] == "true"):
        print(f"  verified: {line}")
except OverlayError as e:
    print(f"ERROR: retagged student failed its post-condition: {e}", file=sys.stderr)
    raise SystemExit(1)
PYCHECK
fi

echo
if [[ "$DRY_RUN" == "true" ]]; then
  echo "dry-run: no artifacts published"
  exit 0
fi

# --- [4/4] the masking contract, DERIVED from the template this step just installed.
#
# WHY IT LIVES HERE. Twenty-two gold configs each restate `response_template:
# "<|im_start|>assistant\n"` by hand. That is not a preference, it is a fact about the chat
# template -- and this is the step that installs the chat template. Pair the student with another
# family and every one of those configs is silently wrong: the collator's scan matches nothing,
# every label stays at ignore_index, and the run trains on NOTHING while producing a loss curve
# and checkpoints (utils.py:465-500 has no post-condition, unlike sft.py:896). Deriving it once,
# in the step that owns the template, is what "do not scatter one step across granite.build steps"
# means in practice.
#
# BEFORE the marker, so a template whose masking contract cannot be derived is never recorded as a
# completed align -- the whole reason the marker goes last.
if [[ -n "$CHAT_TEMPLATE" ]]; then
  echo "--- [4/4] masking contract -> ${RETAGGED}/masking.json"
  "$PYBIN" -m "${PKG}.masking" emit --tokenizer "$RETAGGED"
else
  # Nothing to derive: masking is a property OF a template. align_state.declared_outputs makes the
  # same call on the same argument, so the marker does not promise a file this branch never wrote.
  echo "--- [4/4] masking contract: SKIPPED, no chat template was installed" >&2
  echo "  The masking contract is derived from the template. Gold training against this" >&2
  echo "  student would fail at sft.py:909 before masking mattered." >&2
fi

# The marker goes LAST, after the post-condition, and that ordering is the reason it can be
# trusted: `.step-done.json` written before the ChatML assertion would mark an unverified -- and
# possibly broken -- student as complete, and the next restart would walk straight past it. It is
# also written atomically by step_state, so it cannot appear beside a half-copied shard.
echo "--- marking complete"
"$PYBIN" -m "${PKG}.align_state" mark "${STATE_ARGS[@]}"

publish_artifacts
