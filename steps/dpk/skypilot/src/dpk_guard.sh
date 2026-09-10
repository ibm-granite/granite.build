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
# WHY THE TEMPLATE CALLS THIS, AND NOT dpk_setup.sh / dpk_run.sh
# Asked in review, and the answer is that neither work script can do it. Two reasons:
#
#   1. There is no single script both phases pass through. `setup` and `run` are separate
#      SkyPilot lifecycle phases with separate entrypoints (dpk_setup.sh, dpk_run.sh), so
#      the guard needs two call sites either way; only the template sits above both.
#   2. Neither script is told what the transform rule needs. dpk_run.sh receives the
#      DERIVED module, not `transform`/`dpk_image` — and the derived value cannot
#      distinguish the valid config from the invalid one:
#
#        module: dpk_x.runtime, no image  ->  --module 'dpk_x.runtime'   INVALID (empty venv)
#        module: dpk_x.runtime + image    ->  --module 'dpk_x.runtime'   VALID
#
#      Same value, opposite verdicts; the difference is whether an install happened, which
#      dpk_run.sh is not told. It would catch the `dpk_.runtime` cases and miss exactly the
#      `module`-alone hole that a review round found. dpk_setup.sh is worse: it is not
#      invoked at all in image mode, so it can never see the image-set cases.
#
# Moving the call would therefore mean passing `transform`, `dpk_image` and `output` into
# dpk_run.sh — options it has no other use for — and all five into dpk_setup.sh. The Jinja
# does not shrink, it relocates, and guard ordering stops being visible at the call site,
# which is the bug this step already had once (marker guards after the transform, not
# before). Left in the template deliberately.
#
# WHAT IS NOT HERE, AND WHY IT CANNOT BE
# One guard stays in the template: the `args` KEY check. Keys arrive as
# already-rendered argv words, so a valid `--tkn_chunk_size` and a typo'd
# `--tkn-chunk-size` are indistinguishable by the time any script runs.
#
# There used to be a second — an input-name COLLISION check, for two declared inputs
# whose names sanitized to the same $GB_INPUT_ variable. It is gone along with the
# names: the step takes `input_path` as a PATH resolved by the build (the byoc
# pattern), so it never learns binding names and cannot be wrong about them.
#
# CONTRACT
#   dpk_guard.sh --transform <t> --module <m> --dpk-image <i> \
#                --output <name> --input-path <dir>
#
#   Every option is REQUIRED but may be EMPTY — that is what is being checked.
set -euo pipefail

transform=""
module=""
dpk_image=""
output=""
input_path=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --transform)  transform="$2"; shift 2 ;;
    --module)     module="$2";    shift 2 ;;
    --dpk-image)  dpk_image="$2"; shift 2 ;;
    --output)     output="$2";    shift 2 ;;
    --input-path) input_path="$2"; shift 2 ;;
    --)           shift; break ;;
    *)            break ;;
  esac
done

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

# --- input_path -----------------------------------------------------------------
# The step takes a PATH, resolved by the build from its own declared inputs (the byoc
# pattern), rather than the NAME of a binding it would have to resolve itself. So the
# name-shaped failures are gone: no sanitizing, no collision between two names that
# sanitize alike, and no `set -u` abort from a mistyped variable name.
#
# What replaces them is one failure the PATH form introduces, and it is quiet.
if [ -z "$input_path" ]; then
  echo "dpk: ERROR dpk_config.input_path is required." >&2
  echo "dpk: it is the directory the transform reads, resolved by the build from one" >&2
  echo "dpk: of its declared inputs:" >&2
  echo "dpk:   input_path: \"{{ bindings.<name>.binding.path }}\"" >&2
  exit 1
fi

# A MISTYPED binding name does not fail at render time. Step config is rendered with
# strict=False and PreserveUndefined (utils/template.py), so
# `{{ bindings.dcos.binding.path }}` comes through as the LITERAL text
# "{{ dcos.binding.path }}" rather than raising. Verified: it then reaches DPK as
# --data_local_config {'input_folder': '{{ dcos.binding.path }}'} and fails on the node,
# after the install, complaining about a path nobody wrote.
#
# So refuse anything that still looks like a template. `{{` cannot appear in a real
# staged path — those are assetstore-built (hf cache dirs, shared workdirs) — so this
# costs no legitimate input.
case $input_path in
  *'{{'*|*'{%'*)
    echo "dpk: ERROR dpk_config.input_path still contains an unrendered Jinja" >&2
    echo "dpk: expression: '${input_path}'" >&2
    echo "dpk: the binding name is probably misspelled — it must match one of the" >&2
    echo "dpk: target's declared inputs. Undefined names are preserved rather than" >&2
    echo "dpk: raised, so this is the first point it can be caught." >&2
    exit 1 ;;
esac
