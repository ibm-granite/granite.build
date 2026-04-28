#!/usr/bin/env bash

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

if [[ -z "$AWS_ACCESS_KEY_ID" ]]; then
    echo 'AWS_ACCESS_KEY_ID is not set'
    exit 1
fi

if [[ -z "$AWS_SECRET_ACCESS_KEY" ]]; then
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

MY_EXIT_CODE=$?
if [[ "${MY_EXIT_CODE}" != '0' ]]; then
    echo "s3pull failed with exit code: ${MY_EXIT_CODE}"
    exit 1
fi

echo "Pulled S3 URI: {{ s3_uri }} to local path: {{ local_path }}"

echo 's3pull end'
# ===============================================
