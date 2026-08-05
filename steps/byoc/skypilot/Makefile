# byoc (bring-your-own-code) step for the SkyPilot environment.
#
# byoc runs in a PUBLIC container image and brings its code at runtime by cloning
# a public git repo in the launcher's setup phase, then running a user-defined
# command. It builds no custom image: there is no Dockerfile here, so common.mk
# auto-detects a non-image step and the image / publish-image targets are no-ops.
#
# Usage:
#   make step     # render step-template.yaml -> step/step.yaml, bundle src/
#   make clean    # remove step/
#   make help     # show the shared steps/README.md

STEP_NAME := byoc

include ../../common.mk
