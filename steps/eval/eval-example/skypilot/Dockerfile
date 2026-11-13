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

# Image for the eval EXEMPLAR step. This is a CUSTOM-IMAGE step: the step's code
# is baked into the image and referenced from the generated step.yaml via image_id
# (contrast byoc, a public-image step that brings its code at runtime).
#
# The exemplar payload is a small placeholder SHELL script (src/eval.sh), so the
# image needs no Python/pip — a minimal base is enough. It MUST be Debian/Ubuntu
# based: SkyPilot's docker runtime runs `apt-get` inside the container to install
# its own dependencies, so a non-apt base (e.g. Fedora) fails cloud provisioning
# with "apt-get: command not found" (return code 127). debian:*-slim ships bash,
# which is all eval.sh needs. It is pulled from AWS's public Docker Hub mirror
# (public.ecr.aws/docker/library) rather than docker.io directly, to avoid Docker
# Hub's unauthenticated pull-rate limit on a hot path (CI). Swappable for another
# mirror, e.g. mirror.gcr.io/library/debian. When eval is implemented for real,
# base the image on a suitable Debian/Ubuntu runtime (e.g. a python:*-slim image)
# and install the evaluation dependencies here.
ARG DEBIAN_VERSION=12
FROM public.ecr.aws/docker/library/debian:${DEBIAN_VERSION}-slim

# Bake the eval script into the image and make it executable. The generated
# step.yaml run/command block invokes `bash /opt/eval/eval.sh`.
WORKDIR /opt/eval
COPY src/ /opt/eval/
RUN chmod +x /opt/eval/eval.sh
ENV PATH="/opt/eval:${PATH}"
