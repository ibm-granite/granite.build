{{- define "gbraystepbase.fullname" -}}
{{- if .Values.run_metadata.target_name -}}
{{ .Values.run_metadata.target_name }}-{{ .Release.Name }}
{{- else -}}
{{ .Release.Name }}
{{- end -}}
{{- end -}}


{{- define "gbraystepbase.imagePullSecretNames" -}}
{{- if and .Values.k8s .Values.k8s.userImagePullSecrets }}
  {{- $parentContext := . }}
  {{- $jobType := "job" }}
  {{- if hasKey .Values.run_metadata "job_type" }}
    {{- $jobType = .Values.run_metadata.job_type }}
  {{- end }}
  {{- $fullname := "" -}}
  {{- if eq $jobType "ray" }}
    {{- $fullname = (include "gbraystepbase.fullname" $parentContext | replace "_" "-") -}}
  {{- else }}
    {{- $fullname = (include "gbstepbase.fullname" $parentContext | replace "_" "-") -}}
  {{- end }}

  {{- $names := list -}}
  {{- range $idx, $secret := .Values.k8s.userImagePullSecrets }}
    {{- if $secret.name }}
      {{- $name := printf "%s-%s-%d" $fullname $secret.name $idx | trunc 63 | trimSuffix "-" -}}
      {{- $names = append $names $name -}}
    {{- end }}
  {{- end }}
  {{- join "\n" $names }}
{{- end }}
{{- end }}


{{- define "gbraystepbase.secretsToUseAsImagePullSecrets" -}}
{{- $parentContext := . }}
{{- $names := list -}}

{{- if and .Values.k8s .Values.k8s.userImagePullSecrets }}
  {{- $generated := include "gbraystepbase.imagePullSecretNames" $parentContext | splitList "\n" }}
  {{- range $idx, $secret := .Values.k8s.userImagePullSecrets }}
    {{- $rawName := index $generated $idx }}
    {{- $finalName := regexReplaceAll "[^a-z0-9-]" (lower $rawName) "-" | trimAll "-" }}
    {{- $names = append $names $finalName }}
  {{- end }}
{{- end }}

{{- if .Values.k8s.envImagePullSecrets }}
  {{- range .Values.k8s.envImagePullSecrets }}
    {{- if .name }}
      {{- $names = append $names .name }}
    {{- end }}
  {{- end }}
{{- end }}


{{- if and .Values.k8s .Values.k8s.imagePullSecrets }}
  {{- range .Values.k8s.imagePullSecrets }}
    {{- if .name }}
      {{- $names = append $names .name }}
    {{- end }}
  {{- end }}
{{- end }}

{{- $names = $names | uniq }}
{{- if $names }}
imagePullSecrets:
{{- range $idx, $secretName := $names }}
  - name: {{ $secretName }}
{{- end }}
{{- end }}
{{- end }}

{{- define "gbraystepbase.build-label" }}
granite-dot-build/build-id: {{ .Values.run_metadata.build_id | quote }}
granite-dot-build/build-step-id: {{ .Values.run_metadata.targetsteprun_id | quote }}
granite-dot-build/username: {{ .Values.run_metadata.username | default "none" | quote }}
{{- end }}

{{- define "gbraystepbase.build-anno" }}
granite-dot-build/source-uri: {{ .Values.run_metadata.targetstep_uri | quote }}
{{- end }}

{{- define "gbraystepbase.tplAdditionalFiles" }}
{{- if .Values.k8s.additional_files }}
echo 'create additional files'
{{- range $k, $v := .Values.k8s.additional_files }}
echo '{{ $v | b64enc }}' | base64 --decode > "{{ $k }}"
{{- end }}
{{- end }}
{{- end }}

{{- define "gbraystepbase.tplResourceRequests" }}
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

{{- define "gbraystepbase.addfilefromconfig" }}
{{- if .config }}
{{- $v := .config | toYaml | toString }}
{{- $filename := .filename | toString }}
echo '{{ $v | b64enc }}' | base64 --decode > {{ .filename }}
{{- end }}
{{- end }}

{{- define "gbraystepbase.replicas" }}
{{- $compute_config := .Values.compute_config | default dict }}
{{- $num_nodes := $compute_config.num_nodes | default 1 }}
{{- $podsInt := $num_nodes | toString | int }}
replicas: {{ sub $podsInt 1 }}
{{- end }}


{{/*
Node selector helper – picks up nodeSelector values from environment YAML
*/}}
{{- define "gbraystepbase.tplNodeSelector" -}}
{{- $ns := .Values.k8s.nodeSelector | default dict -}}
{{- /* Only render nodeSelector section if there are values present */ -}}
{{- if $ns }}
nodeSelector:
{{ toYaml $ns | indent 2 }}
{{- end }}
{{- end }}


{{- define "gbraystepbase.raycontainer" }}
spec:
  {{- if and .Values.monitor_config (or 
      (and .Values.monitor_config.sidecar_monitor (ge (len .Values.monitor_config.sidecar_monitor) 1)) 
      (and .Values.monitor_config.event_monitor (ge (len .Values.monitor_config.event_monitor) 1)) 
    ) 
  }}
  shareProcessNamespace: true
  {{- end }}
  {{- if hasKey .Values "automount_service_account_token" }}
  automountServiceAccountToken: {{ .Values.k8s.automount_service_account_token }}
  {{- end }}
  restartPolicy: Never

  {{- include "gbraystepbase.tplNodeSelector" . | nindent 2 }}

  {{- if .Values.k8s.scheduler_name }}
  schedulerName: {{ .Values.k8s.scheduler_name | default "scheduler-plugins-scheduler" }}
  {{- end }}
  containers:
    - name: ray-node
      ports:
        - containerPort: 6379
          name: gcs
        - containerPort: 8265
          name: dashboard
        - containerPort: 10001
          name: client
      lifecycle:
        preStop:
          exec:
            command: ["/bin/sh","-c","ray stop"]
      image: "{{ .Values.k8s.image }}"
      env:
        - name: EXPERIMENT
          value: {{ .Release.Name }}
        {{- $pods := .Values.compute_config.num_nodes | default 1 }}
        {{- if gt ( $pods | int ) 1 }}
        {{- if .Values.k8s.internode_networking }}
        {{- if .Values.k8s.internode_networking.topology_file_config_map }}
        - name: NCCL_TOPO_FILE
          value: /var/run/nvidia-topologyd/virtualTopology.xml
        {{- include "gbraystepbase.nccl-env-vars" . | indent 8 }}
        {{- end }}
        {{- end }}
        {{- end }}
        {{- range $key, $value := .Values.k8s.env }}
        {{- if and (kindIs "map" $value) (hasKey $value "value") (kindIs "invalid" (index $value "value")) }}
        {{- else }}
        - name: {{ $key | quote }}
          {{- $value | toYaml | trimAll " " | nindent 10 }}
        {{- end }}
        {{- end }}
      imagePullPolicy: {{ .Values.k8s.image_pull_policy | default "IfNotPresent" }}
      volumeMounts:
      - name: devshm
        mountPath: /dev/shm
      - name: logs
        mountPath: /logs
      - mountPath: /tmp/ray
        name: ray-logs
      {{- if .Values.k8s.internode_networking }}
      {{- if .Values.k8s.internode_networking.topology_file_config_map }}
      - name: topology-volume
        mountPath: /var/run/nvidia-topologyd
      {{- end }}
      {{- end }}
      {{- range $key, $value := .Values.k8s.volumes }}
      - name: {{ $key | quote }}
        mountPath: "/{{ $key }}"
      {{- end }}
      resources:
        limits:
          {{- include "gbraystepbase.tplResourceRequests" . | trimAll " " | indent 10 }}
        requests:
          {{- include "gbraystepbase.tplResourceRequests" . | trimAll " " | indent 10 }}
    {{- if and .Values.monitor_config (or 
      (and .Values.monitor_config.sidecar_monitor (ge (len .Values.monitor_config.sidecar_monitor) 1)) 
      (and .Values.monitor_config.event_monitor (ge (len .Values.monitor_config.event_monitor) 1)) 
    ) 
    }}
    - name: sidecar
      image: "{{ .Values.k8s.monitoring_sidecar_image }}"
      env:
      - name: MESSAGING_TYPE
        value: {{ .Values.k8s.messaging.type }}
      - name: MESSAGING_EXCHANGE
        value: {{ .Values.k8s.messaging.config.exchange }}
      {{- if .Values.k8s.space_secret }}
      - name: MESSAGING_AUTHENTICATION
        valueFrom:
          secretKeyRef:
            name: "{{ .Values.k8s.space_secret }}"
            key: {{ .Values.k8s.messaging.authentication_secret_name }}
      {{- end }}
      volumeMounts:
      - name: logs
        mountPath: /logs
      command:
      - bash
      - -c
      - |
        echo "===== Dumping monitor_config.yaml from ConfigMap ====="
        {{- $my_monitor_config := toYaml .Values.monitor_config }}
        echo '{{ $my_monitor_config | b64enc }}' | base64 --decode > monitor_config.yaml
        cat monitor_config.yaml || echo "No config found!"
        echo "====================================================="
        python /gbserver/src/gbserver/monitoring/sidecar.py --exchange $MESSAGING_EXCHANGE --queue {{ .Values.run_metadata.build_id }} --routing-key {{ .Values.run_metadata.targetrun_id }}.{{ .Values.run_metadata.targetsteprun_id }}.{{ .Values.run_metadata.launch_id }} --log /logs/output.log --cmd-sub "tee /logs/output.log"
      {{- end }}
  
  {{- include "gbraystepbase.secretsToUseAsImagePullSecrets" . | indent 2 }}

  volumes:
  - name: devshm
    emptyDir:
      medium: Memory
  - name: logs
    emptyDir: {}
  - name: ray-logs
    emptyDir: {}
  {{- if .Values.k8s.internode_networking }}
  {{- if .Values.k8s.internode_networking.topology_file_config_map }}
  - name: topology-volume
    configMap:
      name: {{ .Values.k8s.internode_networking.topology_file_config_map }}
  {{- end }}
  {{- end }}
  {{- range $key, $value := .Values.k8s.volumes }}
  - name: {{ $key }}
    {{- $value | toYaml | trimAll " " | nindent 4 }}
  {{- end }}
{{- end }}


{{- define "gbraystepbase.rayapp" }}
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  labels:
    kueue.x-k8s.io/queue-name: default-queue
    controller-tools.k8s.io: "1.0"
    {{- include "gbraystepbase.build-label" . | indent 4 }}
  annotations:
    {{- include "gbraystepbase.build-anno" . | indent 4 }}
  name: "{{ .Release.Name }}-ray-cluster"
spec:
  rayVersion: "{{ .Values.k8s.ray_version}}"
  headGroupSpec:
    serviceType: ClusterIP
    rayStartParams:
      dashboard-host: '0.0.0.0'
    template:
      {{- include "gbraystepbase.raycontainer" . | indent 6 }}
      {{- include "gbraystepbase.tplNodeSelector" . | indent 6 }}
  {{- $pods := .Values.compute_config.num_nodes | default 1 }}
  {{- if gt ( $pods | int ) 1 }}
  workerGroupSpecs:
    - groupName: small-group
      replicas: {{ sub ( $pods | int ) 1 }}
      rayStartParams: {}
      template:
        {{- include "gbraystepbase.raycontainer" . | indent 8 }}
        {{- include "gbraystepbase.tplNodeSelector" . | indent 8 }}
  {{- end }}
---
apiVersion: workload.codeflare.dev/v1beta2
kind: AppWrapper
metadata:
  name: {{ .Release.Name }}
  labels:
    kueue.x-k8s.io/queue-name: default-queue
spec:
  components:
  - podSetInfos:
    template:
      apiVersion: v1
      kind: Pod
      metadata:
        name: {{ .Release.Name }}
      spec:
        containers:
          - name: base
            image: "{{ .Values.k8s.image }}"
            imagePullPolicy: {{ .Values.k8s.image_pull_policy | default "IfNotPresent" }}
            command:
            - bash
            - -c
            - |
              set -o pipefail
              echo
              {{- include "gbraystepbase.tplAdditionalFiles" . | trimAll " " | indent 8 }}
              {{- range $filename, $value := .filesfromconfig }}
              {{- include "gbraystepbase.addfilefromconfig" (dict "config" $value "filename" $filename ) | trimAll " " | indent 8 }}
              {{- end }}
              {{- if .Values.k8s.show_pip_freeze }}
              echo 'pip freeze'
              pip freeze
              {{- end }}
              {{- range $setup_command := .Values.setup_commands }}
              {{ $setup_command }}
              {{- end }}
              echo "Starting experiment {{ .Release.Name }}"
              {{- if .Values.k8s.setupcommands }}
              {{- range $index, $item := .Values.k8s.setupcommands }}
              {{ $item }}
              {{- end }}
              {{- end }}
              {{- $rayAddr := printf "http://%s-ray-cluster-head-svc:8265" .Release.Name }}
              {{- $rayStatCmd := printf "RAY_ADDRESS=%s ray job list" $rayAddr }}
              echo "Waiting for Ray cluster to come up..."
              {{/* Wait for Ray Cluster to come up */}}
              job_s=$({{ $rayStatCmd }} | tail -n1)
              while [ "$(echo $job_s | grep -c '\[\]')" -ne 1 ]; do
                job_s=$({{ $rayStatCmd }} | tail -n1)
                echo "Cluster not ready yet. Will try again in 30 seconds..."
                sleep 30s
              done
              echo "Ray cluster is now ready. Proceeding..."
              {{- if .Values.k8s.interactive }}
              echo 'sleeping so that the user can exec into the container'
              tail -f /dev/null
              {{- end }}
              echo 'running command...'
              cd {{ .working_dir }}
              RAY_ADDRESS={{ $rayAddr }} ray job submit --working-dir . -- {{ .command }}
              COMMAND_SH_EXIT_CODE="$?"
              echo "COMMAND_SH_EXIT_CODE: ${COMMAND_SH_EXIT_CODE}"
              {{- if .Values.k8s.sleep_on_end }}
              echo
              echo 'sleeping at the end so that the user can exec inside the container'
              tail -f /dev/null
              {{- end }}
              if [[ "${COMMAND_SH_EXIT_CODE}" != "0" ]] ; then
                echo "The command.sh script failed with exit code: ${COMMAND_SH_EXIT_CODE}"
                exit 1
              fi
              # the driver pod does not do anything but submit the job
            resources:
              requests:
                cpu: 4
                nvidia.com/gpu: 0
                memory: 10Gi
              limits:
                cpu: 4
                nvidia.com/gpu: 0
                memory: 10Gi
            volumeMounts:
               - name: dshm
                 mountPath: "/dev/shm"
        restartPolicy: Never
        {{- include "gbraystepbase.secretsToUseAsImagePullSecrets" . | indent 2 }}
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
        {{- include "gbraystepbase.tplNodeSelector" . | nindent 8 }}
{{- end }}


{{- define "gbraystepbase.nccl-env-vars" }}
{{- if .Values.k8s.internode_networking }}
{{- range $k, $v := .Values.k8s.internode_networking.env }}
- name: {{ $k | quote }}
  value: {{ $v | quote }}
{{- end }}
{{- end }}
{{- end }}
