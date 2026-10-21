# eval/livecodebench-score step for the SkyPilot environment.
#
# Scores LiveCodeBench completions written by eval/livecodebench-generate, on CPU:
# it installs LiveCodeBench (pinned) into a venv and runs LiveCodeBench's own
# scorer through src/lcb_score.py. It builds no custom image: there is no
# Dockerfile here, so common.mk auto-detects a non-image step and the image /
# publish-image targets are no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/eval/livecodebench-score/step.yaml) into space/
#   make test     # render the Space, then run test/
#   make clean    # remove space/
#   make help     # list targets; points to the shared steps/README.md

STEP_NAME := livecodebench-score

include ../../../common.mk
