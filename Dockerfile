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
FROM registry.access.redhat.com/ubi9/python-312:9.7@sha256:a0a5885769d5a8c5123d3b15d5135b254541d4da8e7bc445d95e1c90595de470 AS builder
# Artifactory creds for installing dmf library
ARG ARTIFACTORY_USER
ARG ARTIFACTORY_API_KEY
USER root
# Working directory
WORKDIR /app
# Custom artifactory for DMF library. Taking this approach as fetching from IBM GitHub in Dockerfile is an extra pain...
RUN mkdir -p /opt/app-root/src/.pip
RUN echo "[global]" >> /opt/app-root/src/.pip/pip.conf
RUN echo "extra-index-url = https://${ARTIFACTORY_USER}:${ARTIFACTORY_API_KEY}@na.artifactory.swg-devops.com/artifactory/api/pypi/res-data-model-factory-team-pypi-local/simple" >> /opt/app-root/src/.pip/pip.conf
# Copy the pyproject.toml to install dependencies
COPY pyproject.toml pyproject.toml
RUN pip install ".[all]"
# Copy the source code and install in editable mode
COPY . .
RUN pip install --no-deps -e .
# Patch aiohttp and kubernetes_asyncio
RUN patch -i connector.py.patch /opt/app-root/lib/python3.12/site-packages/aiohttp/connector.py
RUN patch -i api_client.py.patch /opt/app-root/lib/python3.12/site-packages/kubernetes_asyncio/client/api_client.py
# Keeps Python from generating .pyc files in the container
# ENV PYTHONDONTWRITEBYTECODE=1

# Runner image
FROM registry.access.redhat.com/ubi9/python-312:9.7@sha256:a0a5885769d5a8c5123d3b15d5135b254541d4da8e7bc445d95e1c90595de470 AS runner
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
# Add the non-root user
RUN useradd -ms /bin/bash ${USER}
RUN chown ${USER}:root /app
RUN chmod 775 /app
# Install useful tools
RUN dnf install -y git vim rsync
# install kubectl
RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
RUN install -o ${USER} -g root -m 0775 kubectl /usr/local/bin/kubectl && rm kubectl
# install helm
RUN curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
# Copy installed package
COPY --from=builder --chown=${USER}:root --chmod=775 /opt/app-root/lib/python3.12/site-packages /opt/app-root/lib/python3.12/site-packages
# Copy the executable script
COPY --from=builder --chown=${USER}:root --chmod=775 /opt/app-root/bin/gbserver /opt/app-root/bin/gbserver
COPY --from=builder --chown=${USER}:root --chmod=775 /opt/app-root/bin/dmf /opt/app-root/bin/dmf
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
