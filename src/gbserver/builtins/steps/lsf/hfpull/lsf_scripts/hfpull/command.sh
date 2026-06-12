#!/usr/bin/env bash

set -euo pipefail
trap 'EC=$?; echo "${LLMB_LSF_JOB_NAME:-hfpull}: hfpull failed at line $LINENO, exit code: $EC" >&2; exit $EC' ERR

# ===============================================
echo 'hfpull start'

# --------------------------------------------------------------------------

{%- set hfp = config.hfpull_config %}

HF_DEST='{{ hfp.path }}'
HF_URI='{{ hfp.uri }}'
export HF_ENDPOINT='{{ hfp.endpoint }}'
HF_REPO='{{ hfp.owner }}/{{ hfp.repo }}'
HF_REVISION='{{ hfp.revision }}'
HF_TYPE='{{ hfp.hf.type }}'

if [[ "${GBTEST_MOCK_HF_CALLS:-}" == "true" ]]; then
    echo "[GBTEST_MOCK_HF_CALLS] mocking hfpull — skipping real download"
    mkdir -p "${HF_DEST}"
    echo mock > "${HF_DEST}/.gbtest_mock_hfpull"
    echo "Pulled HF URI: ${HF_URI} to path ${HF_DEST}"
    echo 'hfpull end'
    exit 0
fi

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

echo "Pulling HF URI: ${HF_URI} to path ${HF_DEST}"

REVISION_FLAG=""
if [[ -n "${HF_REVISION}" ]]; then
    REVISION_FLAG="--revision ${HF_REVISION}"
fi

REPO_TYPE_FLAG=""
if [[ -n "${HF_TYPE}" ]]; then
    REPO_TYPE_FLAG="--repo-type ${HF_TYPE}"
fi

echo hf download "${HF_REPO}" --local-dir "${HF_DEST}" ${REVISION_FLAG} ${REPO_TYPE_FLAG}
hf download "${HF_REPO}" --local-dir "${HF_DEST}" ${REVISION_FLAG} ${REPO_TYPE_FLAG}

# --------------------------------------------------------------------------

echo "Pulled HF URI: ${HF_URI} to path ${HF_DEST}"

echo 'hfpull end'
# ===============================================
