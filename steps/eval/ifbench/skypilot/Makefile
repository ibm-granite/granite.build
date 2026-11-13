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

# ifbench step for the SkyPilot environment.
#
# Like byoc, this step builds no image here: no Dockerfile sits next to this
# Makefile, so common.mk auto-detects a non-image step and the image /
# publish-image targets are no-ops. The family image is built one level up by
# steps/eval/Makefile (from ../Dockerfile.gpu and ../Dockerfile.cpu); a run
# picks it from config.image.
#
# Usage:
#   make space        # render a Space into space/
#   make publish-step # render into the committed assets tree

STEP_NAME := ifbench

include ../../../common.mk
