#!/usr/bin/env bash

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

# teardown-postgres.sh -- Stop the local test PostgreSQL started by setup-postgres.sh.
#
# Removes the container (and with it all test data) and the env file and CA
# certificate setup wrote. Only those files are deleted; their directories are
# removed only if left empty, so GBTEST_POSTGRES_DIR may safely point at a shared
# directory. Safe to run when nothing is running.
#
# Usage:
#   bash scripts/postgres/teardown-postgres.sh
#
# Environment variables:
#   GBTEST_POSTGRES_DIR  - Directory setup-postgres.sh wrote to
#                          (default: <repo>/.cache/gbtest-postgres)
#   DOCKER               - Container runtime: docker or podman (default: auto-detect)

set -euo pipefail

# PG_DIR, CONTAINER_NAME, detect_runtime, remove_generated_files.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUNTIME="$(detect_runtime)"
if [[ -n "${RUNTIME}" ]]; then
    "${RUNTIME}" rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
else
    echo "[postgres-teardown] neither docker nor podman found; no container to stop"
fi
remove_generated_files
echo "[postgres-teardown] Removed ${CONTAINER_NAME} and its generated files in ${PG_DIR}"
