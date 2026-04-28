#!/usr/bin/env bash

# ===============================================
# Script for launching jobs in local environment

# --------------------------------------------------------------------------
# echo \nLLMB_BASH_ASSET_DIR $LLMB_BASH_ASSET_DIR
# echo LLMB_BASH_LAUNCH_ID $LLMB_BASH_LAUNCH_ID
# --------------------------------------------------------------------------
{#- Set useful template variables #}
echo starting llmb_bash_jobsub.sh
{#- config.bash #}
{%- set cbash = config.bash if config.bash is defined else {} %}
{%- set job_id = '' %}
{%- set job_name = 'llmb-${LLMB_BASH_LAUNCH_ID}' %}

{#- config.workload #}
{%- set cwork = config.workload if config.workload is defined else {} %}
{#- workload script #}
{%- set rel_script_path = launcher_config.script_path if launcher_config.script_path is defined and launcher_config.script_path else 'command.sh' %}
{%- set rel_script_path = cwork.path if cwork.path is defined and cwork.path else rel_script_path %}
{%- set script_path = rel_script_path if rel_script_path.startswith('/') else ('${LLMB_BASH_ASSET_BASH_SCRIPTS_DIR}/' ~ rel_script_path) %}
{%- set script_args = launcher_config.script_args if launcher_config.script_args is defined else '' %}
{%- set script_args = cwork.args if cwork.args is defined else script_args %}
{#- workload directories #}
{%- set workspace_dir = cwork.workspace_dir if cwork.workspace_dir is defined and cwork.workspace_dir else '${LLMB_BASH_ASSET_DIR}' %}
{%- set stdout_log_path = '${LLMB_BASH_OUTPUT_DIR}/job_log.out' %}
{%- set stderr_log_path = '${LLMB_BASH_OUTPUT_DIR}/job_log.err' %}
{%- set combined_log_path = '${LLMB_BASH_OUTPUT_DIR}/job.log' %}
{#- config.workload.python_env #}
{%- set cwork_pyenv = cwork.python_env if cwork.python_env is defined else {} %}
{%- set conda_env = cwork_pyenv.conda if cwork_pyenv.conda is defined else '' %}
{%- set virtual_env = cwork_pyenv.venv if cwork_pyenv.venv is defined else '' %}
# --------------------------------------------------------------------------
{#- Environment variables #}

export LLMB_BASH_STEP_FOLDER_NAME='{{ step.name | default(run_metadata.target_name) }}'
export LLMB_TARGET_STEP_RUN_ASSET_DIR="$LLMB_BASH_ASSET_DIR"
export LLMB_BASH_ASSET_BASH_SCRIPTS_DIR="${LLMB_BASH_ASSET_DIR}/bash_scripts/${LLMB_BASH_STEP_FOLDER_NAME}"
export LLMB_BASH_WRAPPER_SCRIPT_PATH="${LLMB_BASH_ASSET_BASH_SCRIPTS_DIR}/llmb_bash_wrapper.sh"

export LLMB_BASH_JOB_NAME="{{ job_name }}"
export LLMB_BASH_WORKSPACE_DIR="{{ workspace_dir }}"
export LLMB_BASH_LOG_FILE_STDOUT="{{ stdout_log_path }}"
export LLMB_BASH_LOG_FILE_STDERR="{{ stderr_log_path }}"
export LLMB_BASH_LOG_FILE_COMBINED="{{ combined_log_path }}"
export LLMB_BASH_SCRIPT_PATH="{{ script_path }}"
export LLMB_BASH_VIRTUAL_ENV="{{ virtual_env }}"
export LLMB_BASH_CONDA_ENV="{{ conda_env }}"
export LLMB_BASH_BUILD_ID='{{ run_metadata.build_id }}'
export LLMB_BASH_TARGET_RUN_ID='{{ run_metadata.targetrun_id }}'
export LLMB_BASH_TARGET_STEP_RUN_ID='{{ run_metadata.targetsteprun_id }}'
export LLMB_BASH_TARGET_NAME='{{ run_metadata.target_name }}'

{#- Input paths as env vars #}
{%- if bindings is defined %}
{%- for b_input_name, b_input_details in bindings.items() %}
{%- set b_input_name_upper = b_input_name | upper %}
{%- set b_input_env = 'LLMB_BASH_INPUT_' ~ b_input_name_upper %}
{%- if b_input_details.binding is defined and b_input_details.binding.path is defined %}
export {{ b_input_env }}="{{ b_input_details.binding.path }}"
{%- endif %}
{%- endfor %}
{%- endif %}
# --------------------------------------------------------------------------
{#- Main code #}

if [[ ! -x "${LLMB_BASH_SCRIPT_PATH}" ]]; then
  echo "${LLMB_BASH_JOB_NAME}: the script path ${LLMB_BASH_SCRIPT_PATH} is not executable, exiting"
  exit 1
fi

if [[ ! -d "${LLMB_BASH_WORKSPACE_DIR}" ]]; then
  echo "${LLMB_BASH_JOB_NAME}: the workspace directory ${LLMB_BASH_WORKSPACE_DIR} does not exist, creating"
  mkdir -p "${LLMB_BASH_WORKSPACE_DIR}" || exit 1
fi

echo "${LLMB_BASH_JOB_NAME}: using the workspace directory ${LLMB_BASH_WORKSPACE_DIR}"

cd "${LLMB_BASH_WORKSPACE_DIR}" || exit 1

echo "${LLMB_BASH_JOB_NAME}: recreate the output directory at ${LLMB_BASH_OUTPUT_DIR}"

# umask 0002
# rm -rf "${LLMB_BASH_OUTPUT_DIR}"

if [[ ! -d "${LLMB_BASH_OUTPUT_DIR}" ]]; then
  echo "${LLMB_BASH_JOB_NAME}: the output directory ${LLMB_BASH_OUTPUT_DIR} does not exist, creating"
  mkdir -p "${LLMB_BASH_OUTPUT_DIR}" || exit 1
fi

# --------------------------------------------------------------------------
echo "${LLMB_BASH_JOB_NAME}: submitting job using wrapper around script ${LLMB_BASH_SCRIPT_PATH}"

nohup "${LLMB_BASH_WRAPPER_SCRIPT_PATH}" {% for arg in script_args %}"{{ arg }}"{% if not loop.last %} {% endif %}{% endfor %}

LLMB_BASH_JOB_EXIT_CODE=$?
if [[ "${LLMB_BASH_JOB_EXIT_CODE}" != "0" ]]; then
    echo "${LLMB_BASH_JOB_NAME}: failed to submit, exit code: ${LLMB_BASH_JOB_EXIT_CODE}"
    exit 1
fi

echo "${LLMB_BASH_JOB_NAME}: submitted job successfully"
# ===============================================
