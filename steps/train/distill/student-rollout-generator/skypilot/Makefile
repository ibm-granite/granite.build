# vllm-server step for the SkyPilot environment (LSF / BlueVela).
#
# Serves a model under vLLM as a long-lived SERVICE target, publishing its URL as
# a mem:// binding so a training target can reach it. Runs in the shared distillation
# GPU image (built by steps/train/distill/Makefile), which already carries
# run_vllm_serve.py; there is no Dockerfile here, so common.mk treats it as
# a non-image step and the image / publish-image targets are no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/train/distill/student-rollout-generator/step.yaml)
#   make test     # render, then run test/
#   make clean    # remove space/
#   make help     # list targets; see the shared steps/README.md

STEP_NAME := student-rollout-generator

include ../../../../common.mk
