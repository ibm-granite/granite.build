#!/bin/bash

# Copyright Granite.Build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
