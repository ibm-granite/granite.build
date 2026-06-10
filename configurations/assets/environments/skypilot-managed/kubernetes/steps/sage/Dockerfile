ARG UBUNTU_VERSION=22.04
FROM ubuntu:${UBUNTU_VERSION}

ARG ENABLE_CUDA=false
ARG CUDA_VERSION=12-1
ARG PYTHON_VERSION=3.12
ARG DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        wget \
        build-essential \
        ca-certificates \
        software-properties-common \
        gnupg \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && rm -rf /var/lib/apt/lists/*

# Conditional CUDA install
RUN if [ "$ENABLE_CUDA" = "true" ]; then \
        wget -qO /tmp/cuda-keyring.deb \
            https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
        && dpkg -i /tmp/cuda-keyring.deb \
        && rm /tmp/cuda-keyring.deb \
        && apt-get update \
        && apt-get install -y --no-install-recommends cuda-toolkit-${CUDA_VERSION} \
        && rm -rf /var/lib/apt/lists/*; \
    fi

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64"

# Python tooling
RUN python -m ensurepip --upgrade \
    && python -m pip install --no-cache-dir uv

# Install sage
COPY . /opt/sage
RUN python -m pip install --no-cache-dir /opt/sage

# CPU builds: tell uv to resolve CPU-only PyTorch variants so runtime
# dependency installs (via sage) pull torch+cpu (~200 MB) instead of the
# GPU variant (~4 GB of NVIDIA CUDA packages).
RUN if [ "$ENABLE_CUDA" != "true" ]; then \
        mkdir -p /root/.config/uv && \
        printf '[pip]\ntorch-backend = "cpu"\n' \
            > /root/.config/uv/uv.toml; \
    fi

# Runtime configuration
RUN mkdir -p /workspace/scripts /workspace/setup /workspace/logs
ENV SCRIPTS_DIR=/workspace/scripts
ENV SETUP_DIR=/workspace/setup
ENV LOGS_DIR=/workspace/logs

WORKDIR /workspace
ENTRYPOINT ["sage"]
