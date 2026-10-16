#!/bin/bash
# Optional helper functions for the byoc step.
#
# This directory is file-mounted into ./src at the workdir root on the cluster
# (see file_mounts in step-template.yaml). Both setup and run start in the
# workdir root, so the user command can source it from there, e.g.:
#
#   source "src/helpers.sh"
#   byoc_log "starting"
#
# (If your command cd's into the cloned repo first, adjust the path accordingly,
# e.g. "source ../src/helpers.sh".)
#
# It is intentionally minimal — byoc's real code comes from the cloned git repo.

# byoc_log: print a namespaced, timestamped log line.
# $1: message to log.
byoc_log() {
    printf '[byoc] %s\n' "$1"
}
