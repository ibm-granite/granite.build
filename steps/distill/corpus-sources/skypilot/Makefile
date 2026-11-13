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

# corpus-sources step for the SkyPilot environment (LSF / BlueVela).
#
# Builds ONE prep-ready corpus file from several raw SFT splits: renames
# `conversations` to `messages`, reservoir-samples each split in proportion to its
# size, and shuffles the result. Runs in the same prebuilt kd-sandbox image as the
# distill steps and needs only the standard library, so there is no Dockerfile here:
# common.mk auto-detects a non-image step and the image / publish-image targets are
# no-ops.
#
# Usage:
#   make space    # render a Space (space.yaml + steps/distill/corpus-sources/step.yaml)
#   make test     # render, then run test/
#   make clean    # remove space/
#   make help     # list targets; see the shared steps/README.md

STEP_NAME := corpus-sources

include ../../../common.mk
