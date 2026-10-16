# Step image for the gated EFS access-point fixture (../containerized/build.yaml).
# An access-point mount needs amazon-efs-utils (mount.efs), which worker steps
# never auto-install, so the image must bundle it.
#
# Base: Debian, not UBI. SkyPilot's container init runs `apt-get install` inside
# the step image (sky/provision/docker_utils.py), so a non-apt image fails to
# launch. The container also runs as root (no USER): the in-container
# `mount -t efs` needs root, and the slim image has no sudo.
#
# Build and publish (linux/amd64, the EC2 worker arch) to the public
# quay.io/granitebuild/gb-efs-utils repo:
#   podman build --platform linux/amd64 -t quay.io/granitebuild/gb-efs-utils:v1.35.2 .
#   podman push --digestfile digest.txt quay.io/granitebuild/gb-efs-utils:v1.35.2
# then pin ../containerized/build.yaml to the digest in digest.txt.
ARG DEBIAN_IMAGE=public.ecr.aws/docker/library/debian:13-slim

# Builder: build the amazon-efs-utils .deb from source (Debian ships no
# package). Pinned to the same release and commit gbserver's teardown VM uses
# (src/gbserver/environment/shared_fs/efs.py): v1.35.2 is the last
# pure-Python/stunnel release; v2+ needs a Rust toolchain.
FROM ${DEBIAN_IMAGE} AS builder
ARG EFS_UTILS_VERSION=v1.35.2
ARG EFS_UTILS_COMMIT=0fdb5b0af469737f5730c5bdca91d9846f042247
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      git ca-certificates binutils build-essential debhelper dh-make python3 \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch "${EFS_UTILS_VERSION}" \
      https://github.com/aws/efs-utils /src/efs-utils \
 && test "$(git -C /src/efs-utils rev-parse HEAD)" = "${EFS_UTILS_COMMIT}" \
 && cd /src/efs-utils && ./build-deb.sh

# Runtime: only the .deb and its runtime deps (nfs-common, stunnel4, python3).
FROM ${DEBIAN_IMAGE}
COPY --from=builder /src/efs-utils/build/amazon-efs-utils*.deb /tmp/
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      /tmp/amazon-efs-utils*.deb \
 && rm -rf /var/lib/apt/lists/* /tmp/amazon-efs-utils*.deb
# Fail the image build, not a cluster run, if mount.efs is missing or broken.
RUN mount.efs --version
