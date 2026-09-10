#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Required-config guards for the dpk step, invoked from the TOP of both the generated
# step.yaml's `setup:` and `run:` blocks and shipped to the node by
# `file_mounts: {src: src}`.
#
# WHY THESE LIVE HERE RATHER THAN IN JINJA
# Review feedback, and it agreed with a principle this step already claimed: shell
# embedded in a YAML scalar behind Jinja can only be RENDERED and pattern matched,
# while shell in a file can be executed, shellcheck'd, `bash -n`'d and unit tested.
# These guards were ~130 lines of Jinja duplicated across both blocks (Jinja macros
# are block-scoped, so the text could not be shared), which needed its own test just
# to police the two copies for drift. Here they are one script called twice.
#
# WHY THEY RUN IN BOTH PHASES
# `setup` is the expensive one: the launcher prepends `hf download` for every hf://
# input binding into it, then dpk_setup.sh bootstraps uv, creates the venv and installs
# the transform's extra — 125 packages including torch/flair/presidio for pii_redactor.
# Guarding only in `run` meant an invalid build paid all of that before being refused.
# `run` guards too because it can be reached on a warm cluster without a fresh setup.
#
# WHAT IS NOT HERE, AND WHY IT CANNOT BE
# Two guards stay in the template because they need information that does not survive
# rendering — the script cannot see it at all:
#
#   * the input-name COLLISION check. Two declared inputs `raw-docs` and `raw.docs`
#     both sanitize to $GB_INPUT_raw_docs, so by the time this script runs exactly one
#     variable exists. The collision is unobservable here; that it is invisible is the
#     whole bug it catches.
#   * the `args` KEY check. Keys arrive as already-rendered argv words, so a valid
#     `--tkn_chunk_size` and a typo'd `--tkn-chunk-size` are indistinguishable by then.
#
# CONTRACT
#   dpk_guard.sh --transform <t> --module <m> --dpk-image <i> \
#                --input <name> --output <name> [--] [declared input names...]
#
#   Every option is REQUIRED but may be EMPTY — that is what is being checked. The
#   declared input names after `--` are the target's inputs, passed as argv because the
#   script cannot enumerate bindings itself; zero of them is a real case (a target that
#   declares none) and is reported as such.
set -euo pipefail

transform=""
module=""
dpk_image=""
input=""
output=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --transform)  transform="$2"; shift 2 ;;
    --module)     module="$2";    shift 2 ;;
    --dpk-image)  dpk_image="$2"; shift 2 ;;
    --input)      input="$2";     shift 2 ;;
    --output)     output="$2";    shift 2 ;;
    --)           shift; break ;;
    *)            break ;;
  esac
done

# Remaining argv is the declared input names.
declared=("$@")

# --- transform ------------------------------------------------------------------
# `transform` supplies TWO things — a module name AND a pip extra — so whatever exempts
# a build from setting it has to supply both. Neither override does alone, and this
# guard was wrong in both directions before landing on the conjunction:
#
#   `transform or module`    — `module` gives a module but no dependencies, so it built
#                              a venv with NO DPK and died "No module named dpk_custom".
#   `transform or dpk_image` — `dpk_image` removes the install but gives no module, so
#                              it rendered `--module 'dpk_.runtime'` and died
#                              "No module named dpk_".
#
# Both are verbatim the illegible failure this guard exists to prevent, reached THROUGH
# the guard. So: an image to skip the install AND a module to run.
if [ -z "$transform" ] && ! { [ -n "$dpk_image" ] && [ -n "$module" ]; }; then
  echo "dpk: ERROR dpk_config.transform is required." >&2
  echo "dpk: it names the DPK transform to run, e.g. tokenization2arrow — the step" >&2
  echo "dpk: derives BOTH the python module and the pip extra from it, so it is what" >&2
  echo "dpk: makes the install know what to install." >&2
  echo "dpk: neither override replaces it alone: 'module' supplies a module but no" >&2
  echo "dpk: dependencies (empty venv), and 'dpk_image' skips the install but supplies" >&2
  echo "dpk: no module (leaving 'dpk_.runtime'). Set 'transform', or set BOTH" >&2
  echo "dpk: 'dpk_image' and 'module'." >&2
  exit 1
fi

# --- output ---------------------------------------------------------------------
# Only EMPTINESS is checkable, here or in Jinja: declared OUTPUTS are not in the
# runtime render context (targetstep.py's _get_validation_context vs the runtime
# bindings/run_metadata/setup_config), and the node never learns them either. Worth
# knowing what that leaves open, because it is worse than the empty case — a MISTYPED
# output emits GB_ARTIFACT_ID:<typo>, buildrun.py logs "failed to find output binding
# ... Ignoring" and continues, so the target goes GREEN having registered nothing and a
# downstream target then fails pointing elsewhere. Recorded in README.md's Known gaps.
if [ -z "$output" ]; then
  echo "dpk: ERROR dpk_config.output is required." >&2
  echo "dpk: it must name one of this target's declared outputs, and becomes the" >&2
  echo "dpk: artifact id the step registers." >&2
  exit 1
fi

# --- input ----------------------------------------------------------------------
# Unguarded, an empty or mistyped `input` renders $GB_INPUT_ or $GB_INPUT_<typo> and
# dies at `set -u` with "GB_INPUT_dcos: unbound variable" — before this script's own
# transform runs, and naming bash rather than the mistake. A typo in an input name is
# an ordinary authoring slip and deserves an ordinary message, with the valid names.
#
# The listing prints the names the AUTHOR wrote (the declared names as passed), not the
# sanitized $GB_INPUT_ forms: a build sets `input: raw-docs`, so reporting "raw_docs"
# sends them chasing a name they never typed. Printed with printf '%s' rather than
# interpolated into a double-quoted echo, so a backtick or $( ) inside an author's name
# cannot run — the template's collision guard reports only sanitized names for exactly
# that reason, and this one has to show raw ones to be useful.
_list_declared() {
  if [ "${#declared[@]}" -eq 0 ]; then
    echo "dpk:   (this target declares NO inputs at all)" >&2
    return
  fi
  for n in "${declared[@]}"; do
    printf 'dpk:   %s\n' "$n" >&2
  done
}

if [ -z "$input" ]; then
  echo "dpk: ERROR dpk_config.input is required." >&2
  echo "dpk: it must name one of this target's declared inputs:" >&2
  _list_declared
  exit 1
fi

found=""
for n in "${declared[@]+"${declared[@]}"}"; do
  if [ "$n" = "$input" ]; then
    found="yes"
    break
  fi
done
if [ -z "$found" ]; then
  echo "dpk: ERROR dpk_config.input names no declared input of this target." >&2
  echo "dpk: it must be one of:" >&2
  _list_declared
  exit 1
fi
