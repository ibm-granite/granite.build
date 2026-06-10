{{- define "my_check_exit_code" }}
MY_RETURN_CODE=$?
if [[ "${MY_RETURN_CODE}" != "0" ]] ; then
    echo "command failed with exit code: ${MY_RETURN_CODE}"
    exit 1
fi
{{- end }}

{{- define "lhpush_command" }}
# -----------------------------------
# VARIABLES
{{- $config := .Values.lhpush_config | required ".Values.lhpush_config is required." }}
{{- $configlh := $config.lh | required ".Values.lhpush_config.lh is required." }}
{{- $lhtype := $config.lh.type | required ".Values.lhpush_config.lh.type is required." }}
{{- $namespace := $config.lh.namespace | required ".Values.lhpush_config.lh.namespace is required." }}
{{- $table_name := $config.lh.table_name | required ".Values.lhpush_config.lh.table_name is required." }}
# -----------------------------------
{{- $use_aspera_flag := "" }}
{{- if $config.use_aspera }}
{{- $use_aspera_flag = "--use-aspera" }}
{{- end }}
# -----------------------------------
{{- $public_flag := "--public False" }}
{{- if or (eq $namespace "granite_dot_build.public") (eq $namespace "granite_dot_build.public_dev") }}
{{- $public_flag = "--public True" }}
{{- end }}
# -----------------------------------

{{- if eq $lhtype "table" }}
# -----------------------------------
# CASE 1 TABLE
{{- $filename := $config.path | required ".Values.lhpush_config.path is required." }}
{{- $extension := $filename | ext }}
{{- $push_args := printf "--namespace %s --table %s --filepath %s %s" $namespace $table_name $filename $public_flag }}
{{- if or (eq $extension ".jsonl") (eq $extension ".csv") }}
{{- $push_args = printf "--namespace %s --table %s --filepath %s --use-batches --batch-size 50000000 %s" $namespace $table_name $filename $public_flag }}
{{- end }}
# -----------------------------------
echo "Start=$(date)";
echo dmf table push {{ $push_args }}
dmf table push {{ $push_args }}
if [ $? -ne 0 ]; then
  echo dmf table append {{ $push_args }}
  dmf table append {{ $push_args }}
  if [ $? -ne 0 ]; then
    if dmf table ls --namespace {{ $namespace }} | grep -q '{{ $table_name }}'; then
      echo dmf table delete --namespace {{ $namespace }} --table {{ $table_name }}
      dmf table delete --namespace {{ $namespace }} --table {{ $table_name }}
    else
      echo "Table {{ $namespace }}.{{ $table_name }} does not exist, skipping delete"
    fi
    echo dmf table push {{ $push_args }}
    dmf table push {{ $push_args }}
  fi
fi
# -----------------------------------
{{- include "my_check_exit_code" . }}
echo "End=$(date)";
# -----------------------------------

{{- else if eq $lhtype "model" }}
# -----------------------------------
# CASE 2 MODEL
{{- $open_flag := "" }}
{{- if eq $table_name "model_shared" }}
{{- $open_flag = "--open True" }}
{{- end }}
{{- $filename := $config.path | required ".Values.lhpush_config.path is required." }}
{{- $model_label := $config.lh.model_label | required ".Values.lhpush_config.lh.model_label is required." }}
# -----------------------------------
if [ -z "{{ $config.lh.model_revision }}" ]; then model_revision=$(basename "{{ $config.lh.src }}"); else model_revision="{{ $config.lh.model_revision }}"; fi;
# -----------------------------------
# Full checkpoint
if [ -e "{{ $filename }}/config.json" ]; then
echo 'Pushing a full checkpoint, installing dependencies...'
pip install safetensors torch;
path="{{ $filename }}"; echo "Finding model_size from $path";
model_size=$(if ls $path/*.safetensors >/dev/null 2>&1; then python -c "import sys,glob,torch;from safetensors.torch import load_file;size=sum([sum([torch.numel(tensor) for _,tensor in list(load_file(model_file).items())]) for model_file in glob.glob(f'{sys.argv[1]}/*.safetensors')]);print(f'{size/1000**3:.1f}b')" $path; else python -c "import sys,glob,torch;size=sum([sum([torch.numel(tensor) for _,tensor in list(torch.load(model_file).items())]) for model_file in glob.glob(f'{sys.argv[1]}/*-of-*.bin')]);print(f'{size/1000**3:.1f}b')" $path; fi); echo "model_size=$model_size";
config_path="{{ $filename }}/config.json";
model_type=$(cat $config_path | grep model_type | sed "s/.*: \([^,]*\).*/\\1/"); echo "model_type=$model_type";
# -----------------------------------
echo "Start=$(date)";
echo dmf model push {{ $use_aspera_flag }} {{ $model_label }} --namespace {{ $namespace }} --dir {{ $filename }} --table {{ $table_name }} --type "${model_type}" --size "${model_size}" --variant fine-tuned --overwrite --revision "${model_revision}" {{ $open_flag }};
dmf model push {{ $use_aspera_flag }} {{ $model_label }} --namespace {{ $namespace }} --dir {{ $filename }} --table {{ $table_name }} --type "${model_type}" --size "${model_size}" --variant fine-tuned --overwrite --revision "${model_revision}" {{ $open_flag }};
# -----------------------------------
{{- include "my_check_exit_code" . }}
echo "End=$(date)";
# -----------------------------------
else
# -----------------------------------
# LoRA adapter checkpoint
echo 'Pushing a LoRA adapter checkpoint'
config_path="{{ $filename }}/adapter_config.json";
base_model_name_or_path=$(cat $config_path | grep base_model_name_or_path | sed "s/.*: \([^,]*\).*/\\1/"); echo "base_model_name_or_path=$base_model_name_or_path";
base_model_name_or_path=$(echo $base_model_name_or_path | sed "s|/\"$|\"|"); echo "base_model_name_or_path=$base_model_name_or_path";
base_model=$(echo $base_model_name_or_path | sed -E "s|.*models/([^/]*).*/([^/]*)/(.*)\"$|\"\\1/\\2.\\3\"|"); echo "base_model=$base_model";
if [ -z "$base_model" ]; then base_model="unk"; fi;
base_config="$(echo $base_model_name_or_path | sed 's/^.\(.*\).$/\1/')/config.json"
model_type=$(cat $base_config | grep model_type | sed "s/.*: \([^,]*\).*/\\1/"); echo "model_type=$model_type";
if [ -z "$model_type" ]; then model_type="unk"; fi;
peft_type=$(cat $config_path | grep peft_type | sed "s/.*: \([^,]*\).*/\\1/"); echo "peft_type=$peft_type";
peft_type=$(cat $config_path | grep peft_type | sed "s/.*: \([^,]*\).*/\\1/"); echo "peft_type=$peft_type";
if [ -z "$peft_type" ]; then peft_type="fine-tuned"; fi;
rank=$(cat $config_path | grep '"r"' | sed "s/.*: \([^,]*\).*/\\1/"); echo "rank=$rank";
rank=$(cat $config_path | grep '"r"' | sed "s/.*: \([^,]*\).*/\\1/"); echo "rank=$rank";
if [ -z "$rank" ]; then rank="unk"; fi;
# -----------------------------------
echo "Start=$(date)";
echo dmf model push {{ $use_aspera_flag }} {{ $model_label }} --namespace {{ $namespace }} --dir {{ $filename }} --table {{ $table_name }} --type "$model_type" --size "$rank" --variant "$peft_type" --base-model "$base_model" --overwrite --revision "$model_revision" {{ $open_flag }};
dmf model push {{ $use_aspera_flag }} {{ $model_label }} --namespace {{ $namespace }} --dir {{ $filename }} --table {{ $table_name }} --type "$model_type" --size "$rank" --variant "$peft_type" --base-model "$base_model" --overwrite --revision "$model_revision" {{ $open_flag }};
# -----------------------------------
{{- include "my_check_exit_code" . }}
echo "End=$(date)";
fi;
# -----------------------------------

{{- else if eq $lhtype "fileset" }}
# -----------------------------------
# CASE 3 FILESET
{{- $filename := $config.path | required ".Values.lhpush_config.path is required." }}
echo "Start=$(date)";
# -----------------------------------
{{- if $config.lh.fileset_version }}
if dmf table ls --namespace {{ $namespace }} | grep -q '{{ $table_name }}'; then
echo dmf fileset delete {{ $config.lh.fileset_label }} --namespace {{ $namespace }} --table {{ $table_name }} --version {{ $config.lh.fileset_version }}
dmf fileset delete {{ $config.lh.fileset_label }} --namespace {{ $namespace }} --table {{ $table_name }} --version {{ $config.lh.fileset_version }}
else
echo "Table {{ $namespace }}.{{ $table_name }} does not exist, skipping fileset delete"
fi
# -----------------------------------
echo dmf fileset push {{ $use_aspera_flag }} {{ $config.lh.fileset_label }} --namespace {{ $namespace }} --table {{ $table_name }} --dir {{ $filename }} --version {{ $config.lh.fileset_version }}
dmf fileset push {{ $use_aspera_flag }} {{ $config.lh.fileset_label }} --namespace {{ $namespace }} --table {{ $table_name }} --dir {{ $filename }} --version {{ $config.lh.fileset_version }}
# -----------------------------------
{{- else }}
if dmf table ls --namespace {{ $namespace }} | grep -q '{{ $table_name }}'; then
echo dmf fileset delete {{ $config.lh.fileset_label }} --namespace {{ $namespace }} --table {{ $table_name }}
dmf fileset delete {{ $config.lh.fileset_label }} --namespace {{ $namespace }} --table {{ $table_name }}
else
echo "Table {{ $namespace }}.{{ $table_name }} does not exist, skipping fileset delete"
fi
# -----------------------------------
echo dmf fileset push {{ $use_aspera_flag }} {{ $config.lh.fileset_label }} --namespace {{ $namespace }} --table {{ $table_name }} --dir {{ $filename }}
dmf fileset push {{ $use_aspera_flag }} {{ $config.lh.fileset_label }} --namespace {{ $namespace }} --table {{ $table_name }} --dir {{ $filename }}
{{- end }}
# -----------------------------------
{{- include "my_check_exit_code" . }}
echo "End=$(date)";
# -----------------------------------

{{- else if eq $lhtype "dataset" }}
# -----------------------------------
# CASE 4 DATASET
{{- $filename := $config.path | required ".Values.lhpush_config.path is required." }}
{{- $dataset_name := $config.lh.dataset_name | required ".Values.lhpush_config.lh.dataset_name is required." }}
echo "Start=$(date)";
if dmf table ls --namespace {{ $namespace }} | grep -q '{{ $table_name }}'; then
echo dmf dataset delete {{ $dataset_name }} --namespace {{ $namespace }} --table {{ $table_name }}
dmf dataset delete {{ $dataset_name }} --namespace {{ $namespace }} --table {{ $table_name }}
else
echo "Table {{ $namespace }}.{{ $table_name }} does not exist, skipping dataset delete"
fi
# -----------------------------------
echo dmf dataset push {{ $dataset_name }} --namespace {{ $namespace }} --table {{ $table_name }} --filepath {{ $filename }} --type 'synthetic' --description 'Created by llm.build' {{ $public_flag }}
dmf dataset push {{ $dataset_name }} --namespace {{ $namespace }} --table {{ $table_name }} --filepath {{ $filename }} --type 'synthetic' --description 'Created by llm.build' {{ $public_flag }}
# The flags --filepath, --type and --description are also required.
# Example: --type synthetic --description "Created by llm.build"
# --type is either real or synthetic
# --description is 'This dataset is used for ....'
# -----------------------------------
{{- include "my_check_exit_code" . }}
echo "End=$(date)";
# -----------------------------------

{{- else }}
# -----------------------------------
# CASE 5 DEFAULT
echo 'Unsupported data type "{{ $lhtype }}"'; exit 1
{{- end }}
# -----------------------------------

# This is the line that we regex to trigger an event of type ARTIFACT_PUSHED_EVENT
echo 'Pushed URI: {{ $config.uri }} for binding {{ $config.binding_id }}'

{{- end }}
