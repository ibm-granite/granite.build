#!/usr/bin/env bash

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

set -euo pipefail
trap 'EC=$?; echo "${LLMB_LSF_JOB_NAME:-s3pull}: s3pull failed at line $LINENO, exit code: $EC" >&2; exit $EC' ERR

# ===============================================
echo 's3pull start'

{%- set s3p = config.s3pull_config %}
{%- set s3_uri = s3p.s3_uri %}
{%- set local_path = s3p.local_path %}
{%- set endpoint_url = s3p.endpoint_url | default("") %}

# --------------------------------------------------------------------------
echo "LLMB_LSF_LAUNCH_ID ${LLMB_LSF_LAUNCH_ID}"
echo "LLMB_LSF_ASSET_DIR ${LLMB_LSF_ASSET_DIR}"
echo "LLMB_LSF_OUTPUT_DIR ${LLMB_LSF_OUTPUT_DIR}"

# --------------------------------------------------------------------------

if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]]; then
    echo 'AWS_ACCESS_KEY_ID is not set'
    exit 1
fi

if [[ -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    echo 'AWS_SECRET_ACCESS_KEY is not set'
    exit 1
fi

echo "Pulling S3 URI: {{ s3_uri }} to local path: {{ local_path }}"

mkdir -p "{{ local_path }}"

{%- if endpoint_url != "" %}
ENDPOINT_FLAG="--endpoint-url {{ endpoint_url }}"
{%- else %}
ENDPOINT_FLAG=""
{%- endif %}

aws s3 sync "{{ s3_uri }}" "{{ local_path }}" $ENDPOINT_FLAG

echo "Pulled S3 URI: {{ s3_uri }} to local path: {{ local_path }}"

echo 's3pull end'
# ===============================================
