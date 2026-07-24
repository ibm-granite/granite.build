FROM python:3.11.13

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TZ=America/New_York

# Accept GitHub credentials as build arguments
ARG GH_USER
ARG GH_TOKEN

# Install system dependencies and Python 3.11
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip and install common packages
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

COPY . .
RUN pip install .

# Expose FastAPI port
EXPOSE 8000

CMD [ "python", "server.py" ]