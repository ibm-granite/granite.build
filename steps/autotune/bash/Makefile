# autotune (AutoTune / fm-tune HPO + training) step for the bash environment.
#
# autotune runs in a PUBLIC image (the bash runtime supplies the interpreter; the
# docker variant expects the fm-tune runtime image to be named per build via
# config.docker.image). It builds no custom image: there is no Dockerfile here, so
# common.mk auto-detects a non-image step and the image / publish-image targets are
# no-ops.
#
# Usage:
#   make space         # render a Space (space.yaml + steps/autotune/step.yaml) into space/
#   make publish-step  # promote into configurations/assets/environments/bash/steps/autotune
#   make check-published  # verify the committed release still matches this source
#   make clean         # remove space/
#   make help          # list targets; points to the shared steps/README.md

STEP_NAME := autotune

include ../../common.mk

# --- Framework deviation, deliberate -----------------------------------------
# common.mk defaults SRC_DIR to `src`, which publishes to <step>/src/. The bash and
# docker launchers do NOT look there: llmb_bash_jobsub.sh resolves a launcher's
# script_path relative to
#     LLMB_BASH_ASSET_BASH_SCRIPTS_DIR="${LLMB_BASH_ASSET_DIR}/bash_scripts/${STEP_FOLDER}"
# so a bash/docker step's scripts must land at <step>/bash_scripts/<name>/. autotune
# is the framework's first non-skypilot step, and `src` cannot satisfy that contract,
# so SRC_DIR is pointed at bash_scripts/ instead. Assigned AFTER the include because
# common.mk uses a plain `=`.
#
# Consequence: the .gbignore that publish-step writes lists `src/**`, which no longer
# matches, so these scripts ARE Jinja-rendered at build time. That is required here --
# command.sh materializes the inline config.autotune-config block via
# `{{ config['autotune-config'] | to_yaml | b64encode }}`. run.py deliberately contains
# no Jinja and a unit test enforces that.
#
# If the framework would rather keep src/ verbatim for every step, the alternative is
# to move materialization out of command.sh (into step-template.yaml's launcher env as
# a base64 blob) and teach common.mk the bash_scripts/ layout -- happy to switch.
SRC_DIR := bash_scripts
