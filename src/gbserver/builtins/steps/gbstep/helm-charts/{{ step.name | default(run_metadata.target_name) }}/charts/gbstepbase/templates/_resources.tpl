{{- define "gbstepbase.tplResourceRequests" }}
{{- $compute_config := .Values.compute_config | default dict }}
{{- $num_nodes := $compute_config.num_nodes | default 1 }}
{{- $num_gpus := $compute_config.num_gpus_per_node | default 0 }}
{{- $num_roce := $compute_config.num_roce_gdr_per_node | default 0 }}
{{- $ephemeral_storage := $compute_config.total_ephemeral_storage_per_node }}

{{- $gpusInt := $num_gpus | toString | int }}
{{- $podsInt := $num_nodes | toString | int }}
{{- $multiplePods := gt $podsInt 1 }}
{{- $recommendedCpus := mul $gpusInt 8 }}
{{- $recommendedCpus = default 1 $recommendedCpus }}
{{- $recommendedMemoInt := mul $gpusInt 64 }}
{{- $recommendedMemoInt = default 1 $recommendedMemoInt }}
{{- $recommendedMemo := printf "%dGi" $recommendedMemoInt }}
{{- $recommendedRoce := ternary 2 0 $multiplePods }}
cpu: {{ default $recommendedCpus $compute_config.num_cpus_per_node }}
memory: {{ default $recommendedMemo $compute_config.total_memory_per_node }}
{{- if $num_gpus }}
nvidia.com/gpu: {{ $num_gpus }}
{{- if or $num_roce $recommendedRoce }}
nvidia.com/roce_gdr: {{ default $recommendedRoce $num_roce }}
{{- end }}
{{- end }}
{{- if $ephemeral_storage }}
ephemeral-storage: {{ $ephemeral_storage }}
{{- end }}
{{- end }}