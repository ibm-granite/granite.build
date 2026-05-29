#!/usr/bin/env bash

set -euo pipefail
trap 'EC=$?; echo "${LLMB_LSF_JOB_NAME:-lhpush}: lhpush failed at line $LINENO, exit code: $EC" >&2; exit $EC' ERR

# ===============================================
echo 'lhpush start'

# --------------------------------------------------------------------------

{%- set lhp = config.lhpush_config %}
{%- set lh_path = lhp.path %}
{%- set lhpconf = lhp.lh %}
{%- set lh_env = lhpconf.env %}
{%- set lh_type = lhpconf.type %}
{%- set lh_namespace = lhpconf.namespace %}
{%- set lh_table_name = lhpconf.table_name %}
{%- set use_aspera_flag = '--use-aspera' if lhp.use_aspera is defined and lhp.use_aspera else '' %}

export LAKEHOUSE_ENVIRONMENT='{{ lh_env }}'
{%- if use_aspera_flag != '' %}
export LAKEHOUSE_REUSE_ASPERA_DAEMON=True
{%- endif %}

if [[ -z "${LAKEHOUSE_TOKEN:-}" ]]; then
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

echo 'Pushing URI: {{ lhp.uri }} from path {{ lh_path }}'

# -----------------------------------

{%- if lh_type == 'table' %}

# -----------------------------------
# CASE 1 TABLE
{%- set extension = lh_path | get_file_extension %}
{%- set batch_flag = '' %}
{%- if (extension == '.jsonl') or (extension == '.csv') %}
{%- set batch_flag = '--use-batches --batch-size 50000000' %}
{%- endif %}
{%- set public_flag = '--public False' %}
{%- if (lh_namespace == 'granite_dot_build.public') or (lh_namespace == 'granite_dot_build.public_dev') %}
{%- set public_flag = '--public True' %}
{%- endif %}
# -----------------------------------
if dmf table ls --namespace {{ lh_namespace }} | grep -q '{{ lh_table_name }}'; then
echo dmf table delete --namespace {{ lh_namespace }} --table {{ lh_table_name }}
dmf table delete --namespace {{ lh_namespace }} --table {{ lh_table_name }}
else
echo "Table {{ lh_namespace }}.{{ lh_table_name }} does not exist, skipping delete"
fi
echo dmf table push --filepath {{ lh_path }} --namespace {{ lh_namespace }} --table {{ lh_table_name }} {{ batch_flag }} {{ public_flag }}
dmf table push --filepath {{ lh_path }} --namespace {{ lh_namespace }} --table {{ lh_table_name }} {{ batch_flag }} {{ public_flag }}
# -----------------------------------

{%- elif lh_type == 'model' %}

# -----------------------------------
# CASE 2 MODEL
{%- set lh_model_revision = lhpconf.model_revision %}
{%- set lh_model_label = lhpconf.model_label %}
{%- set open_flag = '' %}
{%- if lh_table_name == 'model_shared' %}
{%- set open_flag = '--open True' %}
{%- endif %}
{%- if lh_model_revision == '' %}
model_revision=$(basename "{{ lhpconf.src }}")
{%- else %}
model_revision='{{ lh_model_revision }}'
{%- endif %}
# -----------------------------------
if [[ -e "{{ lh_path }}/config.json" ]]; then
# -----------------------------------
# Full checkpoint
echo 'Pushing a full checkpoint...'
# echo 'Pushing a full checkpoint, installing dependencies...'
# pip install safetensors torch;
path="{{ lh_path }}"; echo "Finding model_size from $path";
model_size=$(if ls $path/*.safetensors >/dev/null 2>&1; then python -c "import sys,glob,torch;from safetensors.torch import load_file;size=sum([sum([torch.numel(tensor) for _,tensor in list(load_file(model_file).items())]) for model_file in glob.glob(f'{sys.argv[1]}/*.safetensors')]);print(f'{size/1000**3:.1f}b')" $path; else python -c "import sys,glob,torch;size=sum([sum([torch.numel(tensor) for _,tensor in list(torch.load(model_file).items())]) for model_file in glob.glob(f'{sys.argv[1]}/*-of-*.bin')]);print(f'{size/1000**3:.1f}b')" $path; fi); echo "model_size=$model_size";
config_path="{{ lh_path }}/config.json";
model_type=$(cat $config_path | grep 'model_type' | sed "s/.*: \([^,]*\).*/\\1/"); echo "model_type=$model_type";
# -----------------------------------
echo dmf model push {{ use_aspera_flag }} --overwrite --namespace {{ lh_namespace }} --dir {{ lh_path }} --table {{ lh_table_name }} --type "$model_type" --size "$model_size" --variant fine-tuned --revision "$model_revision" {{ open_flag }} {{ lh_model_label }}
dmf model push {{ use_aspera_flag }} --overwrite --namespace {{ lh_namespace }} --dir {{ lh_path }} --table {{ lh_table_name }} --type "$model_type" --size "$model_size" --variant fine-tuned --revision "$model_revision" {{ open_flag }} {{ lh_model_label }}
# -----------------------------------
else
# -----------------------------------
# LoRA adapter checkpoint
echo 'Pushing a LoRA adapter checkpoint'
config_path="{{ lh_path }}/adapter_config.json";
base_model_name_or_path=$(cat $config_path | grep base_model_name_or_path | sed "s/.*: \([^,]*\).*/\\1/"); echo "base_model_name_or_path=$base_model_name_or_path";
base_model_name_or_path=$(echo $base_model_name_or_path | sed "s|/\"$|\"|"); echo "base_model_name_or_path=$base_model_name_or_path";
base_model=$(echo $base_model_name_or_path | sed -E "s|.*models/([^/]*).*/([^/]*)/(.*)\"$|\"\\1/\\2.\\3\"|"); echo "base_model=$base_model";
if [[ -z "$base_model" ]]; then base_model="unk"; fi;
base_config="$(echo $base_model_name_or_path | sed 's/^.\(.*\).$/\1/')/config.json";
model_type=$(cat $base_config | grep model_type | sed "s/.*: \([^,]*\).*/\\1/"); echo "model_type=$model_type";
if [[ -z "$model_type" ]]; then model_type="unk"; fi;
peft_type=$(cat $config_path | grep peft_type | sed "s/.*: \([^,]*\).*/\\1/"); echo "peft_type=$peft_type";
if [[ -z "$peft_type" ]]; then peft_type="fine-tuned"; fi;
rank=$(cat $config_path | grep '"r"' | sed "s/.*: \([^,]*\).*/\\1/"); echo "rank=$rank";
if [ -z "$rank" ]; then rank="unk"; fi;
# -----------------------------------
echo dmf model push {{ use_aspera_flag }} --overwrite --namespace {{ lh_namespace }} --dir {{ lh_path }} --table {{ lh_table_name }} --type "$model_type" --size "$rank" --variant "$peft_type" --base-model "$base_model" --revision "$model_revision" {{ open_flag }} {{ lh_model_label }}
dmf model push {{ use_aspera_flag }} --overwrite --namespace {{ lh_namespace }} --dir {{ lh_path }} --table {{ lh_table_name }} --type "$model_type" --size "$rank" --variant "$peft_type" --base-model "$base_model" --revision "$model_revision" {{ open_flag }} {{ lh_model_label }}
# -----------------------------------
fi
# -----------------------------------

{%- elif lh_type == 'fileset' %}

# -----------------------------------
# CASE 3 FILESET
{%- set lh_fileset_version = lhpconf.fileset_version %}
{%- set lh_fileset_label = lhpconf.fileset_label %}
{%- set lh_fileset_version_flag = '' %}
{%- if lh_fileset_version %}
{%- set lh_fileset_version_flag = '--version ' ~ lh_fileset_version %}
{%- endif %}
# -----------------------------------
if dmf table ls --namespace {{ lh_namespace }} | grep -q '{{ lh_table_name }}'; then
echo dmf fileset delete --namespace {{ lh_namespace }} --table {{ lh_table_name }} {{ lh_fileset_version_flag }} {{ lh_fileset_label }}
dmf fileset delete --namespace {{ lh_namespace }} --table {{ lh_table_name }} {{ lh_fileset_version_flag }} {{ lh_fileset_label }}
else
echo "Table {{ lh_namespace }}.{{ lh_table_name }} does not exist, skipping fileset delete"
fi
echo dmf fileset push {{ use_aspera_flag }} --namespace {{ lh_namespace }} --table {{ lh_table_name }} --dir {{ lh_path }} {{ lh_fileset_version_flag }} {{ lh_fileset_label }}
dmf fileset push {{ use_aspera_flag }} --namespace {{ lh_namespace }} --table {{ lh_table_name }} --dir {{ lh_path }} {{ lh_fileset_version_flag }} {{ lh_fileset_label }}
# -----------------------------------

{%- elif lh_type == 'dataset' %}

# -----------------------------------
# CASE 4 DATASET
{%- set dataset_name = lhp.lh.dataset_name %}
{%- set dataset_type = 'synthetic' %}
{%- set dataset_desc = 'Created by llm.build' %}
# -----------------------------------
if dmf table ls --namespace {{ lh_namespace }} | grep -q '{{ lh_table_name }}'; then
echo dmf dataset delete {{ dataset_name }} --namespace {{ lh_namespace }} --table {{ lh_table_name }}
dmf dataset delete {{ dataset_name }} --namespace {{ lh_namespace }} --table {{ lh_table_name }}
else
echo "Table {{ lh_namespace }}.{{ lh_table_name }} does not exist, skipping dataset delete"
fi
echo dmf dataset push {{ dataset_name }} --namespace {{ lh_namespace }} --table {{ lh_table_name }} --filepath {{ lh_path }} --type {{ dataset_type }} --description {{ dataset_desc }} {{ public_flag }}
dmf dataset push {{ dataset_name }} --namespace {{ lh_namespace }} --table {{ lh_table_name }} --filepath {{ lh_path }} --type {{ dataset_type }} --description {{ dataset_desc }} {{ public_flag }}
# --type is either real or synthetic
# --description should be 'This dataset is used for ....'
# -----------------------------------

{%- else %}

# -----------------------------------
# CASE 5 UNSUPPORTED
echo 'Unsupported data type for lhpush: "{{ lh_type }}"'; exit 1
# -----------------------------------

{%- endif %}

# --------------------------------------------------------------------------

echo 'Pushed URI: {{ lhp.uri }} from path {{ lh_path }}'

echo 'lhpush end'
# ===============================================
