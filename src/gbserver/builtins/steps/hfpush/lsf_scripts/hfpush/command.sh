#!/usr/bin/env bash

# ===============================================
echo 'hfpush start'

# --------------------------------------------------------------------------

{%- set hfp = config.hfpush_config %}

HF_SOURCE='{{ hfp.path }}'
HF_URI='{{ hfp.uri }}'
HF_REPO='{{ hfp.owner }}/{{ hfp.repo }}'
HF_REVISION='{{ hfp.revision }}'
HF_PRIVATE='{{ hfp.private }}'
BINDING_ID='{{ hfp.binding_id }}'

if [[ -z "$HF_TOKEN" ]]; then
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

echo "Pushing HF URI: ${HF_URI} from path ${HF_SOURCE}"

REVISION_FLAG=""
if [[ -n "${HF_REVISION}" ]]; then
    REVISION_FLAG="--revision ${HF_REVISION}"
fi

PRIVATE_FLAG=""
if [[ "${HF_PRIVATE}" == "True" ]]; then
    PRIVATE_FLAG="--private"
fi

echo huggingface-cli upload "${HF_REPO}" "${HF_SOURCE}" ${REVISION_FLAG} ${PRIVATE_FLAG}
huggingface-cli upload "${HF_REPO}" "${HF_SOURCE}" ${REVISION_FLAG} ${PRIVATE_FLAG}

# --------------------------------------------------------------------------

MY_EXIT_CODE=$?
if [[ "${MY_EXIT_CODE}" != '0' ]]; then
    echo "${LLMB_LSF_JOB_NAME}: hfpush failed, exit code: ${MY_EXIT_CODE}"
    exit 1
fi

echo "Pushed HF URI: ${HF_URI} for binding ${BINDING_ID}"

echo 'hfpush end'
# ===============================================
