ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

# System dependencies (git enables pip installs from VCS refs).
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (requirements.txt lives beside the Dockerfile, not
# in src/, since it is a build-time-only file — it is never bundled into the
# deployable step/ and never referenced from step.yaml). Copying it separately
# also keeps the pip layer cached across code-only changes to src/.
WORKDIR /opt/eval
COPY requirements.txt /opt/eval/
RUN pip install --no-cache-dir -r requirements.txt

# Bake the eval code into the image.
COPY src/ /opt/eval/

# Make the eval entrypoint discoverable; the step.yaml `run` block invokes
# `python /opt/eval/eval.py`.
ENV PATH="/opt/eval:${PATH}"
