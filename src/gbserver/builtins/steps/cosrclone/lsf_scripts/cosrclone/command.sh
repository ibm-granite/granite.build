#!/usr/bin/env bash

set -euo pipefail
trap 'EC=$?; rm -f "${tmpconf:-}"; if [[ $EC -ne 0 ]]; then echo "${LLMB_LSF_JOB_NAME:-cosrclone}: cosrclone failed at line $LINENO, exit code: $EC" >&2; fi; exit $EC' EXIT

# ===============================================
echo 'cosrclone start'

{%- set cfg = config.cos_config %}
{%- set path = cfg.path %}
{%- set bucket_name = cfg.bucket_name %}
{%- set cos_uri_bucket_path = cfg.uri_bucket_path %}
{%- set binding_id = cfg.binding_id %}
{%- set is_push = cfg.push | default(false) %}

tmpconf=$(mktemp /tmp/rclone-XXXXXX.conf)

echo "Writing temporary rclone config to ${tmpconf}"
cat <<EOF > ${tmpconf}
[{{ bucket_name }}]
type = s3
provider = IBMCOS
access_key_id = ${COS_ACCESS_KEY_ID:-}
secret_access_key = ${COS_SECRET_ACCESS_KEY:-}
region = {{ cfg.cos.config.cos_region | default("us-east") }}
endpoint = {{ cfg.cos.config.cos_endpoint | default("s3.us-east.cloud-object-storage.appdomain.cloud") }}
EOF

if [[ "{{ is_push | lower }}" == "True" ]]; then
    echo "Pushing from cluster to COS: {{ path }} -> {{ bucket_name }}:{{ cos_uri_bucket_path }}"
    rclone copy -P "{{ path }}" "{{ bucket_name }}:{{ cos_uri_bucket_path }}" --config "${tmpconf}"
else
    echo "Pulling from COS to cluster: {{ bucket_name }}:{{ cos_uri_bucket_path }} -> {{ path }}"
    rclone copy -P "{{ bucket_name }}:{{ cos_uri_bucket_path }}" "{{ path }}" --config "${tmpconf}"
fi

if [[ "{{ is_push | lower }}" == "true" ]]; then
    echo 'Pushed URI: {{ cos_uri_bucket_path }} for binding {{ binding_id }}'
else
    echo 'Pulled URI: {{ cos_uri_bucket_path }} for binding {{ binding_id }}'
fi

echo "Completed COS rclone operation successfully"
echo 'cosrclone end'
# ===============================================