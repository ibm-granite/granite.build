#!/bin/bash
# Optional helper functions for the byoc step.
#
# This directory is file-mounted into $GB_BUILD_WORKDIR/src on the cluster (see
# file_mounts in step-template.yaml). The user command can source it, e.g.:
#
#   source "$GB_BUILD_WORKDIR/src/helpers.sh"
#   byoc_log "starting"
#
# It is intentionally minimal — byoc's real code comes from the cloned git repo.

# byoc_log: print a namespaced, timestamped log line.
# $1: message to log.
byoc_log() {
    printf '[byoc] %s\n' "$1"
}
