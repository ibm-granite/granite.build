# vllm-server step for the SkyPilot environment (LSF / BlueVela).
#
# Serves a model under vLLM as a long-lived SERVICE target, publishing its URL as
# a mem:// binding so a training target can reach it. Runs in the same prebuilt
# kd-sandbox image distill-gold uses and takes run_vllm_serve.py from the live
# /proj checkout at runtime, so there is no Dockerfile here: common.mk auto-detects
# a non-image step and the image / publish-image targets are no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/distill/vllm-server/step.yaml)
#   make test     # render, then run test/
#   make clean    # remove space/
#   make help     # list targets; see the shared steps/README.md

STEP_NAME := vllm-server

include ../../../common.mk
