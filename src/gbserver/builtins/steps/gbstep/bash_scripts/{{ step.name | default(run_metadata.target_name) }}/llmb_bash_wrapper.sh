#!/usr/bin/env bash

# ===============================================
# Wrapper script meant to run in a local machine
# surrounding the actual workload script

# echo \nLLMB_BASH_ASSET_DIR $LLMB_BASH_ASSET_DIR
# echo LLMB_BASH_LAUNCH_ID $LLMB_BASH_LAUNCH_ID

{#- config.workload #}
{%- set cwork = config.workload if config.workload is defined else {} %}
{#- config.workload.python_env #}
{%- set cwork_pyenv = cwork.python_env if cwork.python_env is defined else {} %}
{%- set conda_env = cwork_pyenv.conda if cwork_pyenv.conda is defined else '' %}
{%- set virtual_env = cwork_pyenv.venv if cwork_pyenv.venv is defined else '' %}

{#- config.bash #}
{%- set cbash = config.bash if config.bash is defined else {} %}
# --------------------------------------------------------------------------
# Create a metadata file next to the log file so it gets uploaded with the appropriate labels
LLMB_BASH_LOG_FILE_COMBINED_METADATA="${LLMB_BASH_LOG_FILE_COMBINED}.metadata"
echo -e "labels:\n  kubernetes.labels.granite-dot-build/build-id: ${LLMB_BASH_BUILD_ID}\n  kubernetes.labels.llmbuild/build-id: ${LLMB_BASH_BUILD_ID}" | tee -a "${LLMB_BASH_LOG_FILE_COMBINED_METADATA}"
# NOTE: llmbuild owns and monitors the log file LLMB_BASH_LOG_FILE_COMBINED
echo "${LLMB_BASH_JOB_NAME}: wrapper start" | tee -a "${LLMB_BASH_LOG_FILE_COMBINED}"

{%- if virtual_env or conda_env %}
# --------------------------------------------------------------------------
echo "${LLMB_BASH_JOB_NAME}: deactivate any venvs that are active" | tee -a "${LLMB_BASH_LOG_FILE_COMBINED}"

deactivate_venv ()
{
    if [ -n "${_OLD_VIRTUAL_PATH:-}" ]; then
        PATH="${_OLD_VIRTUAL_PATH:-}";
        export PATH;
        unset _OLD_VIRTUAL_PATH;
    fi;
    if [ -n "${_OLD_VIRTUAL_PYTHONHOME:-}" ]; then
        PYTHONHOME="${_OLD_VIRTUAL_PYTHONHOME:-}";
        export PYTHONHOME;
        unset _OLD_VIRTUAL_PYTHONHOME;
    fi;
    hash -r 2> /dev/null;
    if [ -n "${_OLD_VIRTUAL_PS1:-}" ]; then
        PS1="${_OLD_VIRTUAL_PS1:-}";
        export PS1;
        unset _OLD_VIRTUAL_PS1;
    fi;
    unset VIRTUAL_ENV;
    unset VIRTUAL_ENV_PROMPT;
    if [ ! "${1:-}" = "nondestructive" ]; then
        unset -f deactivate;
    fi
}

while [[ -n "${VIRTUAL_ENV}" ]];
do
    echo "deactivating VIRTUAL_ENV: ${VIRTUAL_ENV}" | tee -a "${LLMB_BASH_LOG_FILE_COMBINED}"
    deactivate_venv
done

# --------------------------------------------------------------------------
# Activate the appropriate environments

echo "${LLMB_BASH_JOB_NAME}: deactivate any conda envs that are active" | tee -a "${LLMB_BASH_LOG_FILE_COMBINED}"

if [ -f "/opt/share/miniconda/etc/profile.d/conda.sh" ]; then
    source /opt/share/miniconda/etc/profile.d/conda.sh

    if command -v conda >/dev/null 2>&1; then
        while [[ -n "${CONDA_PREFIX}" ]]; do
            echo "deactivating CONDA_PREFIX: ${CONDA_PREFIX}" | tee -a "${LLMB_BASH_LOG_FILE_COMBINED}"
            conda deactivate
        done
        if [[ -n "${LLMB_BASH_CONDA_ENV}" ]]; then
            echo "activating conda env ${LLMB_BASH_CONDA_ENV}" | tee -a "${LLMB_BASH_LOG_FILE_COMBINED}"
            conda activate "${LLMB_BASH_CONDA_ENV}" || exit 1
        fi
    else
        echo "Warning: conda command not found even after sourcing conda.sh" | tee -a "${LLMB_BASH_LOG_FILE_COMBINED}"
    fi
else
    echo "Warning: /opt/share/miniconda/etc/profile.d/conda.sh not found — skipping conda deactivation" | tee -a "${LLMB_BASH_LOG_FILE_COMBINED}"
fi

if [[ -n "${LLMB_BASH_VIRTUAL_ENV}" ]]; then
    echo "activating venv ${LLMB_BASH_VIRTUAL_ENV}" | tee -a "${LLMB_BASH_LOG_FILE_COMBINED}"
    source "${LLMB_BASH_VIRTUAL_ENV}/bin/activate" || exit 1
fi

{%- endif %}

# --------------------------------------------------------------------------
echo "${LLMB_BASH_JOB_NAME}: running the workload script ${LLMB_BASH_SCRIPT_PATH}" | tee -a "${LLMB_BASH_LOG_FILE_COMBINED}"

if [[ ! -x "${LLMB_BASH_SCRIPT_PATH}" ]]; then
  echo "${LLMB_BASH_JOB_NAME}: the script path ${LLMB_BASH_SCRIPT_PATH} is not executable, exiting" | tee -a "${LLMB_BASH_LOG_FILE_COMBINED}"
  exit 1
fi

{#- Create additional files #}
{%- set additional_files = config.gb.additional_files if config.gb is defined and config.gb.additional_files is defined else {} %}
{%- set additional_files = config.llmb.additional_files if config.llmb is defined and config.llmb.additional_files is defined else additional_files %}
{%- for filename, filecontents in additional_files.items() %}
echo "${LLMB_BASH_JOB_NAME}: creating the additional file named {{ filename }}"
echo '{{ filecontents | b64encode }}' | base64 -d > {{ filename }}
{%- endfor %}

"${LLMB_BASH_SCRIPT_PATH}" "$@" 2>&1 | tee -a "${LLMB_BASH_LOG_FILE_COMBINED}"

LLMB_BASH_JOB_EXIT_CODE="${PIPESTATUS[0]}"

{%- if cbash.skip_finding_output_artifacts is defined and cbash.skip_finding_output_artifacts %}

echo "${LLMB_BASH_JOB_NAME}: skip making artifacts out of ${LLMB_BASH_OUTPUT_DIR}"

{%- elif cbash.single_output_artifact is defined and cbash.single_output_artifact %}

echo "${LLMB_BASH_JOB_NAME}: making a single artifact out of ${LLMB_BASH_OUTPUT_DIR}"
echo "LLMB_ARTIFACT_ID:${LLMB_BASH_OUTPUT_DIR##*/} LLMB_ARTIFACT_PATH:${LLMB_BASH_OUTPUT_DIR}"

{%- else %}

echo "${LLMB_BASH_JOB_NAME}: making artifacts out of sub-directories of ${LLMB_BASH_OUTPUT_DIR}"
find "${LLMB_BASH_OUTPUT_DIR}" -depth -mindepth 1 -maxdepth 1 -type d -exec bash -c 'echo "LLMB_ARTIFACT_ID:${0##*/} LLMB_ARTIFACT_PATH:{}"' {} \; | tee -a "${LLMB_BASH_LOG_FILE_COMBINED}"

{%- endif %}

if [[ "${LLMB_BASH_JOB_EXIT_CODE}" != "0" ]]; then
    echo "${LLMB_BASH_JOB_NAME}: workload script failed, exit code: ${LLMB_BASH_JOB_EXIT_CODE}" | tee -a "${LLMB_BASH_LOG_FILE_COMBINED}"
    exit 1
fi

echo "${LLMB_BASH_JOB_NAME}: workload script finished successfully" | tee -a "${LLMB_BASH_LOG_FILE_COMBINED}"
# ===============================================
