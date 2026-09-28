#   Copyright IBM Corporation 2025
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# Builder image
FROM registry.access.redhat.com/ubi10/python-312-minimal:10.2-1789550733@sha256:0f71645815e5e9fa5cbbbba051047fd3a276a45171e591498802a76c156c3a7c AS builder
# Artifactory creds for installing dmf library
ARG ARTIFACTORY_USER
ARG ARTIFACTORY_API_KEY
USER root
# Working directory
WORKDIR /app
# ubi10-minimal drops git and the C toolchain the full python image had. git installs
# the git+https SkyPilot dep; gcc/make/python3.12-devel build any dep lacking a wheel.
# Builder-only stage, so this adds no runner size.
RUN microdnf install -y --nodocs --setopt=install_weak_deps=0 \
        git gcc gcc-c++ make python3.12-devel && \
    microdnf clean all
# Custom artifactory for DMF library. Taking this approach as fetching from IBM GitHub in Dockerfile is an extra pain...
RUN mkdir -p /opt/app-root/src/.pip
RUN echo "[global]" >> /opt/app-root/src/.pip/pip.conf
RUN echo "extra-index-url = https://${ARTIFACTORY_USER}:${ARTIFACTORY_API_KEY}@na.artifactory.swg-devops.com/artifactory/api/pypi/res-data-model-factory-team-pypi-local/simple" >> /opt/app-root/src/.pip/pip.conf

# Copy only what the editable install + running server need (the runner copies /app
# from here, so anything here ships). setuptools discovers src/, test/, and repo-root
# configurations/ (see [tool.setuptools.packages.find]); pyproject/README/constraints
# are read at install; k8s/ holds dep-build-runner.yaml the buildrunner loads at runtime
# (relative to /app); .git lets setuptools_scm derive the version (dropped just below).
# The built UI ships under src/gbserver/static/ui/, so frontend/ sources aren't needed.
COPY pyproject.toml README.md constraints.txt ./
COPY .git/ ./.git/
COPY src/ ./src/
COPY test/ ./test/
COPY configurations/ ./configurations/
COPY k8s/ ./k8s/
# PIP_CONSTRAINT pins the AWS SDK cluster (boto3/botocore/awscli/aiobotocore) to
# avoid multi-hour pip resolver backtracking. Applied to every pip invocation in
# this stage. See constraints.txt for details.
ENV PIP_CONSTRAINT=/app/constraints.txt
# --timeout/--retries harden large-wheel downloads (pyarrow, torch, …) against pip's
# 15s socket timeout tripping a ReadTimeoutError mid-download.
RUN pip install --upgrade --timeout 120 --retries 5 -e ".[all]"
# Drop .git now that setuptools_scm has derived the version. Removing it here (not in
# the runner, where it would only add a whiteout over a committed layer) actually
# shrinks the image; the server needs no git history in /app.
RUN rm -rf /app/.git
# Keeps Python from generating .pyc files in the container
# ENV PYTHONDONTWRITEBYTECODE=1

# Runner image
FROM registry.access.redhat.com/ubi10/python-312-minimal:10.2-1789550733@sha256:0f71645815e5e9fa5cbbbba051047fd3a276a45171e591498802a76c156c3a7c AS runner
# Non-root user
ARG USER=gbserver
# Current image tag
ARG GBSERVER_GIT_COMMIT=unset
ARG GBSERVER_IMAGE_TAG=latest
ARG GBSERVER_SIDECAR_MONITORING_IMAGE_TAG=latest
ENV GBSERVER_GIT_COMMIT=${GBSERVER_GIT_COMMIT}
ENV GBSERVER_IMAGE_TAG=${GBSERVER_IMAGE_TAG}
ENV GBSERVER_SIDECAR_MONITORING_IMAGE_TAG=${GBSERVER_SIDECAR_MONITORING_IMAGE_TAG}

USER root
# Port for REST API server
EXPOSE 8080
# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
# Working directory
WORKDIR /app
# ubi10-minimal uses microdnf (no dnf) and omits these. shadow-utils provides useradd
# (used next); openssl/tar/gzip are needed by the helm install script below.
RUN microdnf install -y --nodocs --setopt=install_weak_deps=0 \
        shadow-utils git vim-minimal rsync tar gzip openssl && \
    microdnf clean all
# Add the non-root user
RUN useradd -ms /bin/bash ${USER}
RUN chown ${USER}:root /app
RUN chmod 775 /app
# install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
RUN install -o ${USER} -g root -m 0775 kubectl /usr/local/bin/kubectl && rm kubectl
# install helm
RUN curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
# Copy installed package
COPY --from=builder --chown=${USER}:root --chmod=775 /opt/app-root/lib/python3.12/site-packages /opt/app-root/lib/python3.12/site-packages
# Copy our console scripts. Previously only gbserver and dmf were copied, so `gb`
# — the CLI, useful for in-pod debugging — was absent from every runner image even
# though pyproject declares it. Named explicitly rather than copying the whole
# bin/: that directory holds ~114 scripts (every dependency's, e.g. aws/black/
# pytest), and a directory COPY would also make bin/ itself group-writable,
# letting the app user add entries to its own PATH.
# Keep in sync with [project.scripts] in pyproject.toml (dmf comes from the
# Artifactory DMF package, not pyproject).
COPY --from=builder --chown=${USER}:root --chmod=775 \
    /opt/app-root/bin/gbserver \
    /opt/app-root/bin/gbcli \
    /opt/app-root/bin/gb \
    /opt/app-root/bin/llmbuild \
    /opt/app-root/bin/llmb \
    /opt/app-root/bin/lamb \
    /opt/app-root/bin/gbtest \
    /opt/app-root/bin/gbmcp \
    /opt/app-root/bin/gb-ui-backend \
    /opt/app-root/bin/dmf \
    /opt/app-root/bin/
# Copy the source code
COPY --from=builder --chown=${USER}:root --chmod=775 /app /app
# Switch to the non-root user
COPY letsencrypt-r13.pem /etc/pki/ca-trust/source/anchors/letsencrypt-r13.pem
RUN update-ca-trust
USER ${USER}:root
RUN git config --global --add safe.directory /app
# Entrypoint
ENTRYPOINT ["gbserver"]
CMD ["rest-server"]
