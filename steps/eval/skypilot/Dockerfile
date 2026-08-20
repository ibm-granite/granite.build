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
