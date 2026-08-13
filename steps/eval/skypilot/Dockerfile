# Image for the eval EXEMPLAR step. This is a CUSTOM-IMAGE step: the step's code
# is baked into the image and referenced from the generated step.yaml via image_id
# (contrast byoc, a public-image step that brings its code at runtime).
#
# The exemplar payload is a small placeholder SHELL script (src/eval.sh), so the
# image needs no Python/pip — a minimal Fedora base is enough. It lives on quay.io
# to match the rest of this repo (the byoc default and the buildrunner test-data)
# and to avoid Docker Hub's unauthenticated pull-rate limit. When eval is
# implemented for real, base the image on a suitable runtime (e.g. a Python image)
# and install the evaluation dependencies here.
ARG FEDORA_VERSION=42
FROM quay.io/fedora/fedora-minimal:${FEDORA_VERSION}

# Bake the eval script into the image and make it executable. The generated
# step.yaml run/command block invokes `bash /opt/eval/eval.sh`.
WORKDIR /opt/eval
COPY src/ /opt/eval/
RUN chmod +x /opt/eval/eval.sh
ENV PATH="/opt/eval:${PATH}"
