#!/usr/bin/env bash

# ===============================================
echo 'lhpull start'

# --------------------------------------------------------------------------

{%- set lhp = config.lhpull_config %}
{%- set lh_path = lhp.path %}
{%- set lhpconf = lhp.lh %}
{%- set lh_env = lhpconf.env %}
{%- set lh_type = lhpconf.type %}
{%- set lh_namespace = lhpconf.namespace %}
{%- set lh_table_name = lhpconf.table_name %}
{%- set lh_model_revision = lhpconf.model_revision | default("granite-dot-build") %}
{%- set lh_fileset_version = lhpconf.fileset_version | default("granite-dot-build") %}
{%- set use_aspera_flag = '--use-aspera' if lhp.use_aspera is defined and lhp.use_aspera else '' %}

export LAKEHOUSE_ENVIRONMENT='{{ lh_env }}'
{%- if use_aspera_flag != '' %}
export LAKEHOUSE_REUSE_ASPERA_DAEMON=True
{%- endif %}

if [[ -z "$LAKEHOUSE_TOKEN" ]]; then
    echo 'LAKEHOUSE_TOKEN is not set'
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
echo "LLMB_LSF_VIRTUAL_ENV ${LLMB_LSF_VIRTUAL_ENV}"
echo "LLMB_LSF_CONDA_ENV ${LLMB_LSF_CONDA_ENV}"
echo "LLMB_LSF_NUM_NODES ${LLMB_LSF_NUM_NODES}"
echo "LLMB_LSF_NUM_CPUS ${LLMB_LSF_NUM_CPUS}"
echo "LLMB_LSF_NUM_GPUS ${LLMB_LSF_NUM_GPUS}"
echo "LLMB_LSF_MEMORY_SIZE ${LLMB_LSF_MEMORY_SIZE}"
echo "LLMB_LSF_BUILD_ID ${LLMB_LSF_BUILD_ID}"
echo "LLMB_LSF_TARGET_RUN_ID ${LLMB_LSF_TARGET_RUN_ID}"
echo "LLMB_LSF_TARGET_STEP_RUN_ID ${LLMB_LSF_TARGET_STEP_RUN_ID}"
echo "LLMB_LSF_TARGET_NAME ${LLMB_LSF_TARGET_NAME}"

echo "LAKEHOUSE_ENVIRONMENT ${LAKEHOUSE_ENVIRONMENT}"

# --------------------------------------------------------------------------
echo 'Pulling URI: {{ lhp.uri }} to path {{ lh_path }}'

{%- if lh_type == 'table' %}

echo dmf table pull --dir {{ lh_path }} --namespace {{ lh_namespace }} --table {{ lh_table_name }} --use-batch-reader
dmf table pull --dir {{ lh_path }} --namespace {{ lh_namespace }} --table {{ lh_table_name }} --use-batch-reader

{%- elif lh_type == 'model' %}

{%- set lh_model_label = lhpconf.model_label %}
echo dmf model pull --dir {{ lh_path }} --namespace {{ lh_namespace }} --table {{ lh_table_name }} --revision {{ lh_model_revision }} {{ use_aspera_flag }} {{ lh_model_label }}
dmf model pull --dir {{ lh_path }} --namespace {{ lh_namespace }} --table {{ lh_table_name }} --revision {{ lh_model_revision }} {{ use_aspera_flag }} {{ lh_model_label }}

{%- elif lh_type == 'fileset' %}

{%- set lh_fileset_label = lhpconf.fileset_label %}
echo dmf fileset pull --dir {{ lh_path }} --namespace {{ lh_namespace }} --table {{ lh_table_name }} --version {{ lh_fileset_version }} {{ use_aspera_flag }} {{ lh_fileset_label }}
dmf fileset pull --dir {{ lh_path }} --namespace {{ lh_namespace }} --table {{ lh_table_name }} --version {{ lh_fileset_version }} {{ use_aspera_flag }} {{ lh_fileset_label }}

{%- elif lh_type == 'dataset' %}

echo dmf dataset pull --dir {{ lh_path }} --namespace {{ lh_namespace }} --table {{ lh_table_name }} --use-batch-reader
dmf dataset pull --dir {{ lh_path }} --namespace {{ lh_namespace }} --table {{ lh_table_name }} --use-batch-reader

{%- else %}

echo 'Unsupported data type for lhpull: "{{ lh_type }}"'; exit 1

{%- endif %}

# --------------------------------------------------------------------------

MY_EXIT_CODE=$?
if [[ "${MY_EXIT_CODE}" != '0' ]]; then
    echo "${LLMB_LSF_JOB_NAME}: dmf pull failed, exit code: ${MY_EXIT_CODE}"
    exit 1
fi

echo 'Pulled URI: {{ lhp.uri }} to path {{ lh_path }}'

echo 'lhpull end'
# ===============================================
