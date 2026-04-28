#!/usr/bin/env bash

# ===============================================
# Script for launching jobs in LSF environments

# --------------------------------------------------------------------------
# Replaced Environment variables

export LLMB_LSF_LAUNCH_ID='LLMB_LSF_REPLACE_THIS_LAUNCH_ID'
export LLMB_LSF_ASSET_DIR='LLMB_LSF_REPLACE_THIS_ASSET_DIR'

# --------------------------------------------------------------------------
{#- Set useful template variables #}

{#- environment_config #}
{%- set env_config = environment_config if environment_config is defined else {} %}
{#- environment_config.workload #}
{%- set env_work = env_config.workload if env_config.workload is defined else {} %}
{%- set env_work_pyenv = env_work.python_env if env_work.python_env is defined else {} %}
{%- set py_env_dirs = env_work_pyenv.env_dirs if env_work_pyenv.env_dirs is defined else [] %}
{#- environment_config.lsf #}
{%- set env_lsf = env_config.lsf if env_config.lsf is defined else {} %}
{%- set env_lsf_bsub = env_lsf.bsub if env_lsf.bsub is defined else {} %}
{#- environment_config.lsf.bsub.user_mapping_file #}
{%- set user_mapping_file = env_lsf_bsub.user_mapping_file if env_lsf_bsub.user_mapping_file is defined else '' %}

{%- set extra_bsub_flags = env_lsf_bsub.additional_args if env_lsf_bsub.additional_args is defined else '' %}
{%- set queue = env_lsf_bsub.queue if env_lsf_bsub.queue is defined and env_lsf_bsub.queue else 'normal' %}
{%- set jobs_group = env_lsf_bsub.jobs_group if env_lsf_bsub.jobs_group is defined and env_lsf_bsub.jobs_group else 'grp_granite_dot_build' %}
{%- set use_functional_id = env_lsf_bsub.use_functional_id if env_lsf_bsub.use_functional_id is defined else False %}

{#- user to use with bsubmit for running workloads #}
{#- environment_config.authentication #}
{%- set env_auth = env_config.authentication if env_config.authentication is defined else {} %}
{%- set default_lsf_user = env_auth.login_node_username if env_auth.login_node_username is defined else '' %}
{%- set gh_user = run_metadata.username if run_metadata.username is defined else '' %}

{#- config.lsf #}
{%- set clsf = config.lsf if config.lsf is defined else {} %}
{#- config.lsf.bsub #}
{%- set clsf_bsub = clsf.bsub if clsf.bsub is defined else {} %}
{%- set job_id = clsf_bsub.jobid if clsf_bsub.jobid is defined else '' %}
{%- set job_name = clsf_bsub.job_name if clsf_bsub.job_name is defined and clsf_bsub.job_name else 'llmb-${LLMB_LSF_LAUNCH_ID}' %}

{%- if job_id != '' %}
echo "{{ job_name }}: job was already submitted with id: {{ job_id }}"
exit 0
{%- endif %}

{%- set unmanaged_bsub_args = clsf_bsub.args if clsf_bsub.args is defined else '' %}
{%- set is_managed = unmanaged_bsub_args == '' %}
{%- set not_is_managed = not is_managed %}

{#- config.lsf.bsub - managed #}
{%- set extra_bsub_flags = clsf_bsub.additional_args if clsf_bsub.additional_args is defined else extra_bsub_flags %}
{%- set queue = clsf_bsub.queue if clsf_bsub.queue is defined and clsf_bsub.queue else queue %}
{%- set jobs_group = clsf_bsub.jobs_group if clsf_bsub.jobs_group is defined and clsf_bsub.jobs_group else jobs_group %}
{%- set use_functional_id = clsf_bsub.use_functional_id if clsf_bsub.use_functional_id is defined else use_functional_id %}

{#- config.workload #}
{%- set cwork = config.workload if config.workload is defined else {} %}
{%- set cwork_env = cwork.env if cwork.env is defined else {} %}

{#- workload script #}
{%- set rel_script_path = launcher_config.script_path if launcher_config.script_path is defined and launcher_config.script_path else 'command.sh' %}
{%- set rel_script_path = cwork.path if cwork.path is defined and cwork.path else rel_script_path %}
{%- set script_path = rel_script_path if rel_script_path.startswith('/') else ('${LLMB_LSF_ASSET_LSF_SCRIPTS_DIR}/' ~ rel_script_path) %}
{%- set script_args = launcher_config.script_args if launcher_config.script_args is defined else '' %}
{%- set script_args = cwork.args if cwork.args is defined else script_args %}
{#- workload directories #}
{%- set workspace_dir = cwork.workspace_dir if cwork.workspace_dir is defined and cwork.workspace_dir else '${LLMB_LSF_ASSET_DIR}' %}
{%- set output_dir = cwork.output_dir if cwork.output_dir is defined and cwork.output_dir else '${LLMB_LSF_WORKSPACE_DIR}/outputs' %}
{%- set stdout_log_path = '${LLMB_LSF_OUTPUT_DIR}/job_log.out' %}
{%- set stderr_log_path = '${LLMB_LSF_OUTPUT_DIR}/job_log.err' %}
{%- set combined_log_path = clsf_bsub.log_path if clsf_bsub.log_path is defined and clsf_bsub.log_path else '${LLMB_LSF_OUTPUT_DIR}/job.log' %}
{#- config.workload.python_env #}
{%- set cwork_pyenv = cwork.python_env if cwork.python_env is defined else {} %}
{%- set py_env_dirs = cwork_pyenv.env_dirs if cwork_pyenv.env_dirs is defined else py_env_dirs %}
{%- set conda_env = cwork_pyenv.conda if cwork_pyenv.conda is defined else '' %}
{%- set virtual_env = cwork_pyenv.venv if cwork_pyenv.venv is defined else '' %}

{#- Resources #}
{%- set ccc = config.compute_config if config.compute_config is defined else {} %}
{%- set num_nodes = ccc.num_nodes if ccc.num_nodes is defined else 1 %}
{%- set num_gpus_per_node = ccc.num_gpus_per_node if ccc.num_gpus_per_node is defined else 1 %}
{%- set gpu_model = ccc.gpu_model if ccc.gpu_model is defined else '' %}
{%- set num_roce_gdr_per_node = ccc.num_roce_gdr_per_node if ccc.num_roce_gdr_per_node is defined else 0 %}
{%- set total_ephemeral_storage_per_node = ccc.total_ephemeral_storage_per_node if ccc.total_ephemeral_storage_per_node is defined else '' %}
{%- set num_cpus_per_node = ccc.num_cpus_per_node if ccc.num_cpus_per_node is defined else (num_gpus_per_node * 8) %}
{%- set total_memory_per_node = ccc.total_memory_per_node if ccc.total_memory_per_node is defined else '' %}
{%- if total_memory_per_node == '' %}
  {%- set total_memory_per_node = (num_gpus_per_node * 64) ~ 'G' %}
{%- elif total_memory_per_node.endswith('Gi') or total_memory_per_node.endswith('Mi') %}
  {%- set total_memory_per_node = total_memory_per_node[:-1] %}
{%- endif %}
{#- Resource flags to bsub #}
{%- set bsub_num_nodes_flag = '' %}
{%- set bsub_cpu_memory_flag = '' %}
{%- set bsub_gpus_flag = '' %}
{#- Check and set the flags #}
{%- if is_managed %}
{#- Nodes #}
{%- set bsub_num_nodes_flag = '-n "${LLMB_LSF_NUM_NODES}"' %}
{#- CPUs and memory #}
{%- set bsub_cpu_memory_flag = '-R "rusage[mem=${LLMB_LSF_MEMORY_SIZE},cpu=${LLMB_LSF_NUM_CPUS}]"' %}
{#- GPUs #}
{%- if num_gpus_per_node > 0 %}
  {%- if gpu_model %}
    {%- set bsub_gpus_flag = '-gpu "num=' ~ num_gpus_per_node ~ '/task:mode=exclusive_process:gmodel=' ~ gpu_model ~ '"' %}
  {%- else %}
    {%- set bsub_gpus_flag = '-gpu "num=' ~ num_gpus_per_node ~ '/task:mode=exclusive_process"' %}
  {%- endif %}
{%- endif %}
{%- endif %}
{#- Resources #}

# --------------------------------------------------------------------------
{#- Environment variables #}

export LLMB_LSF_STEP_FOLDER_NAME='{{ step.name | default(run_metadata.target_name) }}'
export LLMB_TARGET_STEP_RUN_ASSET_DIR="$LLMB_LSF_ASSET_DIR"
export LLMB_LSF_ASSET_LSF_SCRIPTS_DIR="${LLMB_LSF_ASSET_DIR}/lsf_scripts/${LLMB_LSF_STEP_FOLDER_NAME}"
export LLMB_LSF_WRAPPER_SCRIPT_PATH="${LLMB_LSF_ASSET_LSF_SCRIPTS_DIR}/llmb_lsf_wrapper.sh"
export LLMB_LSF_FIND_DIR_SCRIPT_PATH="${LLMB_LSF_ASSET_LSF_SCRIPTS_DIR}/llmb_resolve_dir_script.py"
export LLMB_LSF_HELM_MERGE_SCRIPT_PATH="${LLMB_LSF_ASSET_LSF_SCRIPTS_DIR}/llmb_lsf_helm_values_merge.py"
export LLMB_LSF_BSUBMIT_USER_SCRIPT_PATH="${LLMB_LSF_ASSET_LSF_SCRIPTS_DIR}/llmb_lsf_resolve_bsubmit_user.py"
export LLMB_LSF_ASSET_MERGED_VALUES_PATH="${LLMB_LSF_ASSET_LSF_SCRIPTS_DIR}/output-values.yaml"

export LLMB_LSF_USER_MAPPING_FILE="{{ user_mapping_file }}"
export LLMB_LSF_DEFAULT_LSF_USER="{{ default_lsf_user }}"
export LLMB_LSF_GH_USER="{{ gh_user }}"
export LLMB_LSF_JOBS_GROUP='{{ jobs_group }}'
export LLMB_LSF_QUEUE='{{ queue }}'
export LLMB_LSF_JOB_NAME="{{ job_name }}"
export LLMB_LSF_WORKSPACE_DIR="{{ workspace_dir }}"
export LLMB_LSF_OUTPUT_DIR="{{ output_dir }}"
export LLMB_WORKLOAD_PYTHON_ENV_ENV_DIRS='{{ py_env_dirs | json_dumps | b64encode }}'
export LLMB_LSF_LOG_FILE_STDOUT="{{ stdout_log_path }}"
export LLMB_LSF_LOG_FILE_STDERR="{{ stderr_log_path }}"
export LLMB_LSF_LOG_FILE_COMBINED="{{ combined_log_path }}"
export LLMB_LSF_SCRIPT_PATH="{{ script_path }}"
export LLMB_LSF_VIRTUAL_ENV="{{ virtual_env }}"
export LLMB_LSF_CONDA_ENV="{{ conda_env }}"
export LLMB_LSF_NUM_NODES='{{ num_nodes }}'
export LLMB_LSF_NUM_CPUS='{{ num_cpus_per_node }}'
export LLMB_LSF_NUM_GPUS='{{ num_gpus_per_node }}'
export LLMB_LSF_MEMORY_SIZE='{{ total_memory_per_node }}'
export LLMB_LSF_BUILD_ID='{{ run_metadata.build_id }}'
export LLMB_LSF_TARGET_RUN_ID='{{ run_metadata.targetrun_id }}'
export LLMB_LSF_TARGET_STEP_RUN_ID='{{ run_metadata.targetsteprun_id }}'
export LLMB_LSF_TARGET_NAME='{{ run_metadata.target_name }}'

{#- Input paths as env vars #}
{%- for b_input_name, b_input_details in bindings.items() %}
{%- set b_input_name_upper = b_input_name | upper %}
{%- set b_input_env = 'LLMB_LSF_INPUT_' ~ b_input_name_upper %}
export {{ b_input_env }}="{{ b_input_details.binding.path }}"
{%- endfor %}

{#- Extra env vars from `step.yaml` and `build.yaml` #}
{%- for env_var_name, env_var_value in cwork_env.items() %}
export {{ env_var_name }}={{ env_var_value | tojson }}
{%- endfor %}

# --------------------------------------------------------------------------
{#- Main code #}

{%- if is_managed %}

if [[ ! -x "${LLMB_LSF_SCRIPT_PATH}" ]]; then
  echo "${LLMB_LSF_JOB_NAME}: the script path ${LLMB_LSF_SCRIPT_PATH} is not executable, exiting"
  exit 1
fi

if [[ ! -d "${LLMB_LSF_WORKSPACE_DIR}" ]]; then
  echo "${LLMB_LSF_JOB_NAME}: the workspace directory ${LLMB_LSF_WORKSPACE_DIR} does not exist, creating"
  mkdir -p "${LLMB_LSF_WORKSPACE_DIR}" || exit 1
fi

echo "${LLMB_LSF_JOB_NAME}: using the workspace directory ${LLMB_LSF_WORKSPACE_DIR}"

cd "${LLMB_LSF_WORKSPACE_DIR}" || exit 1

echo "${LLMB_LSF_JOB_NAME}: recreate the output directory at ${LLMB_LSF_OUTPUT_DIR}"

# umask 0002
# rm -rf "${LLMB_LSF_OUTPUT_DIR}"

if [[ ! -d "${LLMB_LSF_OUTPUT_DIR}" ]]; then
  echo "${LLMB_LSF_JOB_NAME}: the output directory ${LLMB_LSF_OUTPUT_DIR} does not exist, creating"
  mkdir -p "${LLMB_LSF_OUTPUT_DIR}" || exit 1
fi

# --------------------------------------------------------------------------

{#- whether to use bsub or bsubmit with the proper lsf username #}
{%- set bsub_entrypoint = 'bsub' %}
{%- if not use_functional_id %}
{%- if user_mapping_file %}
{%- if default_lsf_user %}
echo "${LLMB_LSF_JOB_NAME}: checking the user mapping file ${LLMB_LSF_USER_MAPPING_FILE} for user ${LLMB_LSF_GH_USER}"
LLMB_LSF_BSUBMIT_USER="$(python3 "$LLMB_LSF_BSUBMIT_USER_SCRIPT_PATH" --user-mapping-file "$LLMB_LSF_USER_MAPPING_FILE" --input-username "$LLMB_LSF_GH_USER" --input-default-username "$LLMB_LSF_DEFAULT_LSF_USER")"
EXIT_CODE=$?
if [[ "${EXIT_CODE}" != "0" ]]; then exit 1; fi
echo "${LLMB_LSF_JOB_NAME}: username '${LLMB_LSF_GH_USER}' mapped to LSF user '${LLMB_LSF_BSUBMIT_USER}'"
{%- set bsub_entrypoint = 'bsubmit --user "$LLMB_LSF_BSUBMIT_USER"' %}
{%- endif %}
{%- endif %}
{%- endif %}

# --------------------------------------------------------------------------

echo "${LLMB_LSF_JOB_NAME}: submitting job using wrapper around script ${LLMB_LSF_SCRIPT_PATH}"

{{ bsub_entrypoint }} \
    -G "${LLMB_LSF_JOBS_GROUP}" \
    -J "${LLMB_LSF_JOB_NAME}" \
    -q "${LLMB_LSF_QUEUE}" \
    -o "${LLMB_LSF_LOG_FILE_STDOUT}" \
    -e "${LLMB_LSF_LOG_FILE_STDERR}" \
    {%- if bsub_num_nodes_flag != '' %}
    {{ bsub_num_nodes_flag }} \
    {%- endif %}
    {%- if bsub_cpu_memory_flag != '' %}
    {{ bsub_cpu_memory_flag }} \
    {%- endif %}
    {%- if bsub_gpus_flag != '' %}
    {{ bsub_gpus_flag }} \
    {%- endif %}
    {%- if extra_bsub_flags != '' %}
    {{ extra_bsub_flags }} \
    {%- endif %}
    blaunch "${LLMB_LSF_WRAPPER_SCRIPT_PATH}" {{ script_args }}

{%- else %}

echo "${LLMB_LSF_JOB_NAME}: submitting job with user specified bsub args"

echo bsub {{ unmanaged_bsub_args }}
bsub {{ unmanaged_bsub_args }}

{%- endif %}

LLMB_LSF_JOB_EXIT_CODE=$?
if [[ "${LLMB_LSF_JOB_EXIT_CODE}" != "0" ]]; then
    echo "${LLMB_LSF_JOB_NAME}: failed to submit, exit code: ${LLMB_LSF_JOB_EXIT_CODE}"
    exit 1
fi

echo "${LLMB_LSF_JOB_NAME}: submitted job successfully"
# ===============================================
