#!/usr/bin/env bash

# ===============================================
# Wrapper script meant to run in a compute node
# surrounding the actual workload script

{#- config.workload #}
{%- set cwork = config.workload if config.workload is defined else {} %}
{#- config.workload.python_env #}
{%- set cwork_pyenv = cwork.python_env if cwork.python_env is defined else {} %}
{%- set conda_env = cwork_pyenv.conda if cwork_pyenv.conda is defined else '' %}
{%- set virtual_env = cwork_pyenv.venv if cwork_pyenv.venv is defined else '' %}

{#- config.lsf #}
{%- set clsf = config.lsf if config.lsf is defined else {} %}

# --------------------------------------------------------------------------
# Create a metadata file next to the log file so it gets uploaded with the appropriate labels

LLMB_LSF_LOG_FILE_COMBINED_METADATA="${LLMB_LSF_LOG_FILE_COMBINED}.metadata"
# Overwrite the metadata file instead of appending to it to maintain a valid yaml file
echo -e "labels:\n  kubernetes.labels.granite-dot-build/build-id: ${LLMB_LSF_BUILD_ID}\n  kubernetes.labels.llmbuild/build-id: ${LLMB_LSF_BUILD_ID}" | tee "${LLMB_LSF_LOG_FILE_COMBINED_METADATA}"
# NOTE: llmbuild owns and monitors the log file LLMB_LSF_LOG_FILE_COMBINED
echo "${LLMB_LSF_JOB_NAME}: wrapper start" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
echo "${LLMB_LSF_JOB_NAME}: LLMB_EVENT_WORKLOAD_STATUS:running" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
echo "${LLMB_LSF_JOB_NAME}: cd into ${LLMB_LSF_ASSET_LSF_SCRIPTS_DIR}" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
cd "$LLMB_LSF_ASSET_LSF_SCRIPTS_DIR" || exit 1

# --------------------------------------------------------------------------
# Make conda available

if command -v conda &>/dev/null; then
    echo "${LLMB_LSF_JOB_NAME}: conda is already available, we will source conda.sh anyway" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
else
    echo "${LLMB_LSF_JOB_NAME}: conda is not available, we will source conda.sh" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
fi
source /opt/share/miniconda/etc/profile.d/conda.sh
if command -v conda &>/dev/null; then
    echo "${LLMB_LSF_JOB_NAME}: conda was made available" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
    which conda
else
    echo "${LLMB_LSF_JOB_NAME}: failed to make conda available" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
    exit 1
fi

# --------------------------------------------------------------------------
# Emulate the Helm merge algorithm

if [[ -f "${LLMB_LSF_ASSET_LSF_SCRIPTS_DIR}/values-default.yaml" ]]; then
if [[ -f "${LLMB_LSF_ASSET_LSF_SCRIPTS_DIR}/values.yaml" ]]; then
if [[ -f "${LLMB_LSF_ASSET_LSF_SCRIPTS_DIR}/values-config.yaml" ]]; then

LLMB_LSF_HELM_MERGE_EMULATION_CONDA_ENV="${LLMB_LSF_HELM_MERGE_EMULATION_CONDA_ENV:-/u/granitebuild/llmb_python_envs/conda-llmbhelmmergeemulation}"
echo "${LLMB_LSF_JOB_NAME}: found some values.yaml files, emulating the helm values merge algo with the script ${LLMB_LSF_HELM_MERGE_SCRIPT_PATH} and the conda env ${LLMB_LSF_HELM_MERGE_EMULATION_CONDA_ENV}" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
conda activate "$LLMB_LSF_HELM_MERGE_EMULATION_CONDA_ENV"
EXIT_CODE=$?
if [[ "${EXIT_CODE}" != "0" ]]; then
    echo "${LLMB_LSF_JOB_NAME}: failed to activate the conda env: ${LLMB_LSF_HELM_MERGE_EMULATION_CONDA_ENV}" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
    exit 1
fi
python3 "$LLMB_LSF_HELM_MERGE_SCRIPT_PATH"
EXIT_CODE=$?
if [[ "${EXIT_CODE}" != "0" ]]; then
    echo "${LLMB_LSF_JOB_NAME}: the helm merge script failed: ${LLMB_LSF_HELM_MERGE_SCRIPT_PATH}" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
    exit 1
fi
conda deactivate

# Sanity check to see if the output file is actually there
echo "${LLMB_LSF_JOB_NAME}: ls -la ${LLMB_LSF_ASSET_MERGED_VALUES_PATH}" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
ls -la "$LLMB_LSF_ASSET_MERGED_VALUES_PATH" || exit 1

fi
fi
fi

{%- if virtual_env or conda_env %}
# --------------------------------------------------------------------------
# Deactivate python venvs

echo "${LLMB_LSF_JOB_NAME}: deactivate any venvs that are active" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"

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
    echo "${LLMB_LSF_JOB_NAME}: deactivating VIRTUAL_ENV: ${VIRTUAL_ENV}" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
    deactivate_venv
done

# --------------------------------------------------------------------------
# Deactivate python conda envs

echo "${LLMB_LSF_JOB_NAME}: deactivate any conda envs that are active" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"

while [[ -n "${CONDA_PREFIX}" ]];
do
    echo "${LLMB_LSF_JOB_NAME}: deactivating CONDA_PREFIX: ${CONDA_PREFIX}" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
    conda deactivate
done

# --------------------------------------------------------------------------
# Activate the appropriate python environments

if [[ -n "${LLMB_LSF_CONDA_ENV}" ]]; then
    echo "${LLMB_LSF_JOB_NAME}: looking for a conda env '${LLMB_LSF_CONDA_ENV}' in base dirs '${LLMB_WORKLOAD_PYTHON_ENV_ENV_DIRS}' using script '${LLMB_LSF_FIND_DIR_SCRIPT_PATH}'" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
    FOUND_DIR=$(INPUT_BASE_DIRS="$LLMB_WORKLOAD_PYTHON_ENV_ENV_DIRS" INPUT_TARGET_DIR="$LLMB_LSF_CONDA_ENV" python3 "$LLMB_LSF_FIND_DIR_SCRIPT_PATH")
    EXIT_CODE=$?
    if [[ "${EXIT_CODE}" != "0" ]]; then
        echo "${LLMB_LSF_JOB_NAME}: failed to find the conda env: ${LLMB_LSF_CONDA_ENV}" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
        exit 1
    fi
    echo "${LLMB_LSF_JOB_NAME}: activating conda env ${FOUND_DIR}" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
    conda activate "${FOUND_DIR}" || exit 1
fi

if [[ -n "${LLMB_LSF_VIRTUAL_ENV}" ]]; then
    echo "${LLMB_LSF_JOB_NAME}: looking for a venv '${LLMB_LSF_VIRTUAL_ENV}' in base dirs '${LLMB_WORKLOAD_PYTHON_ENV_ENV_DIRS}' using script '${LLMB_LSF_FIND_DIR_SCRIPT_PATH}'" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
    FOUND_DIR=$(INPUT_BASE_DIRS="$LLMB_WORKLOAD_PYTHON_ENV_ENV_DIRS" INPUT_TARGET_DIR="$LLMB_LSF_VIRTUAL_ENV" python3 "$LLMB_LSF_FIND_DIR_SCRIPT_PATH")
    EXIT_CODE=$?
    if [[ "${EXIT_CODE}" != "0" ]]; then
        echo "${LLMB_LSF_JOB_NAME}: failed to find the venv: ${LLMB_LSF_VIRTUAL_ENV}" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
        exit 1
    fi
    echo "${LLMB_LSF_JOB_NAME}: activating venv ${FOUND_DIR}" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
    source "${FOUND_DIR}/bin/activate" || exit 1
fi

{%- endif %}

# --------------------------------------------------------------------------
# Run the workload

echo "${LLMB_LSF_JOB_NAME}: running the workload script ${LLMB_LSF_SCRIPT_PATH}" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"

if [[ ! -x "${LLMB_LSF_SCRIPT_PATH}" ]]; then
  echo "${LLMB_LSF_JOB_NAME}: the script path ${LLMB_LSF_SCRIPT_PATH} is not executable, exiting" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
  exit 1
fi

{#- Create additional files #}
{%- set additional_files = config.gb.additional_files if config.gb is defined and config.gb.additional_files is defined else {} %}
{%- set additional_files = config.llmb.additional_files if config.llmb is defined and config.llmb.additional_files is defined else additional_files %}
{%- for filename, filecontents in additional_files.items() %}
echo "${LLMB_LSF_JOB_NAME}: creating the additional file named {{ filename }}"
echo '{{ filecontents | b64encode }}' | base64 -d > {{ filename }}
{%- endfor %}

"${LLMB_LSF_SCRIPT_PATH}" "$@" 2>&1 | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"

LLMB_LSF_JOB_EXIT_CODE="${PIPESTATUS[0]}"

{%- if clsf.skip_finding_output_artifacts is defined and clsf.skip_finding_output_artifacts %}

echo "${LLMB_LSF_JOB_NAME}: skip making artifacts out of ${LLMB_LSF_OUTPUT_DIR}"

{%- elif clsf.single_output_artifact is defined and clsf.single_output_artifact %}

echo "${LLMB_LSF_JOB_NAME}: making a single artifact out of ${LLMB_LSF_OUTPUT_DIR}"
echo "LLMB_ARTIFACT_ID:${LLMB_LSF_OUTPUT_DIR##*/} LLMB_ARTIFACT_PATH:${LLMB_LSF_OUTPUT_DIR}"

{%- else %}

echo "${LLMB_LSF_JOB_NAME}: making artifacts out of sub-directories of ${LLMB_LSF_OUTPUT_DIR}"
find "${LLMB_LSF_OUTPUT_DIR}" -depth -mindepth 1 -maxdepth 1 -type d -exec bash -c 'echo "LLMB_ARTIFACT_ID:${0##*/} LLMB_ARTIFACT_PATH:{}"' {} \; | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"

{%- endif %}

if [[ "${LLMB_LSF_JOB_EXIT_CODE}" != "0" ]]; then
    echo "${LLMB_LSF_JOB_NAME}: LLMB_EVENT_WORKLOAD_STATUS:failed" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
    echo "${LLMB_LSF_JOB_NAME}: workload script failed, exit code: ${LLMB_LSF_JOB_EXIT_CODE}" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
    exit 1
fi

echo "${LLMB_LSF_JOB_NAME}: LLMB_EVENT_WORKLOAD_STATUS:success" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
echo "${LLMB_LSF_JOB_NAME}: workload script finished successfully" | tee -a "${LLMB_LSF_LOG_FILE_COMBINED}"
# ===============================================
