#!/usr/bin/env bash

# ===============================================
echo 's3push start'

{%- set s3p = config.s3push_config %}
{%- set local_path = s3p.local_path %}
{%- set s3_uri = s3p.s3_uri %}
{%- set endpoint_url = s3p.endpoint_url | default("") %}

# --------------------------------------------------------------------------

if [[ -z "$AWS_ACCESS_KEY_ID" ]]; then
    echo 'AWS_ACCESS_KEY_ID is not set'
    exit 1
fi

if [[ -z "$AWS_SECRET_ACCESS_KEY" ]]; then
    echo 'AWS_SECRET_ACCESS_KEY is not set'
    exit 1
fi

echo "Pushing local path: {{ local_path }} to S3 URI: {{ s3_uri }}"

{%- if endpoint_url != "" %}
ENDPOINT_FLAG="--endpoint-url {{ endpoint_url }}"
{%- else %}
ENDPOINT_FLAG=""
{%- endif %}

aws s3 sync "{{ local_path }}" "{{ s3_uri }}" $ENDPOINT_FLAG

MY_EXIT_CODE=$?
if [[ "${MY_EXIT_CODE}" != '0' ]]; then
    echo "s3push failed with exit code: ${MY_EXIT_CODE}"
    exit 1
fi

echo "Pushed local path: {{ local_path }} to S3 URI: {{ s3_uri }}"

echo 's3push end'
# ===============================================
