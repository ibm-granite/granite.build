#!/usr/bin/env bash

set -euo pipefail
trap 'EC=$?; echo "${LLMB_LSF_JOB_NAME:-hfpush}: hfpush failed at line $LINENO, exit code: $EC" >&2; exit $EC' ERR

# ===============================================
echo 'hfpush start'

# --------------------------------------------------------------------------

{%- set hfp = config.hfpush_config %}

HF_SOURCE='{{ hfp.path }}'
HF_URI='{{ hfp.uri }}'
export HF_ENDPOINT='{{ hfp.endpoint }}'
HF_OWNER='{{ hfp.owner }}'
HF_REPO_NAME='{{ hfp.repo }}'
HF_REPO="${HF_OWNER}/${HF_REPO_NAME}"
HF_REVISION='{{ hfp.revision }}'
HF_PRIVATE='{{ hfp.private }}'
HF_TYPE='{{ hfp.hf.type }}'
HF_RESOURCE_GROUP_ID='{{ hfp.hf.resource_group_id }}'
BINDING_ID='{{ hfp.binding_id }}'

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo 'HF_TOKEN is not set'
    exit 1
fi

# --------------------------------------------------------------------------
# Environment variables

echo "LLMB_LSF_LAUNCH_ID ${LLMB_LSF_LAUNCH_ID}"
echo "LLMB_LSF_ASSET_DIR ${LLMB_LSF_ASSET_DIR}"
echo "LLMB_LSF_QUEUE ${LLMB_LSF_QUEUE}"
echo "LLMB_LSF_JOB_NAME ${LLMB_LSF_JOB_NAME}"
echo "LLMB_LSF_WORKSPACE_DIR ${LLMB_LSF_WORKSPACE_DIR}"
echo "LLMB_LSF_OUTPUT_DIR ${LLMB_LSF_OUTPUT_DIR}"
echo "LLMB_LSF_LOG_FILE_STDOUT ${LLMB_LSF_LOG_FILE_STDOUT}"
echo "LLMB_LSF_LOG_FILE_STDERR ${LLMB_LSF_LOG_FILE_STDERR}"
echo "LLMB_LSF_SCRIPT_PATH ${LLMB_LSF_SCRIPT_PATH}"
echo "LLMB_LSF_NUM_NODES ${LLMB_LSF_NUM_NODES}"
echo "LLMB_LSF_NUM_CPUS ${LLMB_LSF_NUM_CPUS}"
echo "LLMB_LSF_NUM_GPUS ${LLMB_LSF_NUM_GPUS}"
echo "LLMB_LSF_MEMORY_SIZE ${LLMB_LSF_MEMORY_SIZE}"
echo "LLMB_LSF_BUILD_ID ${LLMB_LSF_BUILD_ID}"
echo "LLMB_LSF_TARGET_RUN_ID ${LLMB_LSF_TARGET_RUN_ID}"
echo "LLMB_LSF_TARGET_STEP_RUN_ID ${LLMB_LSF_TARGET_STEP_RUN_ID}"
echo "LLMB_LSF_TARGET_NAME ${LLMB_LSF_TARGET_NAME}"

# --------------------------------------------------------------------------
# Create the HF repo (idempotent).  `hf upload` cannot attach a
# resource_group_id, so we POST to /api/repos/create ourselves with the
# resolved id.  HTTP 409 means the repo already exists, which is fine.

HF_VISIBILITY="public"
if [[ "${HF_PRIVATE}" == "True" ]]; then
    HF_VISIBILITY="private"
fi

if [[ -n "${HF_RESOURCE_GROUP_ID}" ]]; then
    CREATE_BODY=$(printf '{"name":"%s","organization":"%s","type":"%s","visibility":"%s","resourceGroupId":"%s"}' \
        "${HF_REPO_NAME}" "${HF_OWNER}" "${HF_TYPE}" "${HF_VISIBILITY}" "${HF_RESOURCE_GROUP_ID}")
else
    CREATE_BODY=$(printf '{"name":"%s","organization":"%s","type":"%s","visibility":"%s"}' \
        "${HF_REPO_NAME}" "${HF_OWNER}" "${HF_TYPE}" "${HF_VISIBILITY}")
fi

echo "Creating HF repo ${HF_REPO} (resource_group_id=${HF_RESOURCE_GROUP_ID:-<none>})"
CREATE_RESP=$(mktemp)
HTTP_CODE=$(curl -sS -o "${CREATE_RESP}" -w "%{http_code}" \
    -X POST \
    -H "Authorization: Bearer ${HF_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "${CREATE_BODY}" \
    "${HF_ENDPOINT}/api/repos/create")

if [[ "${HTTP_CODE}" != "200" && "${HTTP_CODE}" != "409" ]]; then
    echo "HF create_repo failed: HTTP ${HTTP_CODE}"
    cat "${CREATE_RESP}"
    rm -f "${CREATE_RESP}"
    exit 1
fi
rm -f "${CREATE_RESP}"

# --------------------------------------------------------------------------

echo "Pushing HF URI: ${HF_URI} from path ${HF_SOURCE}"

REVISION_FLAG=""
if [[ -n "${HF_REVISION}" ]]; then
    REVISION_FLAG="--revision ${HF_REVISION}"
fi

PRIVATE_FLAG=""
if [[ "${HF_PRIVATE}" == "True" ]]; then
    PRIVATE_FLAG="--private"
fi

REPO_TYPE_FLAG=""
if [[ -n "${HF_TYPE}" ]]; then
    REPO_TYPE_FLAG="--repo-type ${HF_TYPE}"
fi

echo hf upload "${HF_REPO}" "${HF_SOURCE}" ${REVISION_FLAG} ${PRIVATE_FLAG} ${REPO_TYPE_FLAG}
hf upload "${HF_REPO}" "${HF_SOURCE}" ${REVISION_FLAG} ${PRIVATE_FLAG} ${REPO_TYPE_FLAG}

# --------------------------------------------------------------------------

echo "Pushed HF URI: ${HF_URI} for binding ${BINDING_ID}"

echo 'hfpush end'
# ===============================================
