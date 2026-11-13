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

# fm-tune step for the SkyPilot environment (slurm subtype): runs one fm-tune
# (autotunex/src/fm-tune) tuning job on a bare node; the run script builds its
# own venv. Public image (no Dockerfile).
#
# Usage:
#   make test-setup   # once: bring up the local podman/Docker Slurm cluster
#   make test         # render the Space, then run test/ (slurm test self-skips)
#   make publish-step

STEP_NAME := fm-tune

# This step defines its own test-setup target (below).
HAS_TEST_SETUP := true

include ../../../common.mk

test-setup:
	$(MAKE) -C $(REPO_ROOT) slurm-setup
