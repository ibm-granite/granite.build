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

# hf-peft-sft step for SkyPilot: LoRA / QLoRA / aLoRA SFT with HF PEFT + TRL,
# ClearML tracking. Custom-image step: the Dockerfile bakes the hashed training
# stack and src/ into the image, pushed as
# $(REGISTRY)/steps/train/hf-peft-sft/skypilot:<IMAGE_VERSION>.
STEP_NAME := hf-peft-sft

# Public namespace; the step's repository there must be public (steps/README.md,
# "A custom image-based step's image repository").
REGISTRY ?= quay.io/granitebuild

# Stable tag, not the git SHA: BlueVela caches a .sqsh per tag. Bump the suffix
# when the Dockerfile, requirements.lock or src/ changes; the prefix with peft.
IMAGE_VERSION ?= peft-v0.21.2-6

# Pinned uv for the lock, same version as the image's uv stage.
UV_LOCK ?= uvx --from uv==0.13.6 uv

include ../../../common.mk

.PHONY: lock
lock: ## Recompile requirements.lock (hashed) from requirements.in for linux x86_64
	$(UV_LOCK) pip compile requirements.in --python-version 3.12 \
	  --python-platform x86_64-manylinux_2_28 --generate-hashes -o requirements.lock
