{{- define "gbstepbase.build-label" }}
granite-dot-build/build-id: {{ .Values.run_metadata.build_id | quote }}
granite-dot-build/build-step-id: {{ .Values.run_metadata.targetsteprun_id | quote }}
granite-dot-build/username: {{ .Values.run_metadata.username | default "none" | quote }}
{{- end }}

{{- define "gbstepbase.build-anno" }}
granite-dot-build/source-uri: {{ .Values.run_metadata.targetstep_uri | quote }}
{{- end }}


{{- define "gbstepbase.pyjobpod" }}
spec:
  {{- if and .Values.monitor_config (or 
      (and .Values.monitor_config.sidecar_monitor (ge (len .Values.monitor_config.sidecar_monitor) 1)) 
      (and .Values.monitor_config.event_monitor (ge (len .Values.monitor_config.event_monitor) 1)) 
    ) 
  }}
  shareProcessNamespace: true
  {{- end }}
  {{- if .Values.k8s.service_account_name }}
  serviceAccountName: {{ .Values.k8s.service_account_name }}
  {{- end }}
  {{- if hasKey .Values "automount_service_account_token" }}
  automountServiceAccountToken: {{ .Values.k8s.automount_service_account_token }}
  {{- end }}
  restartPolicy: Never
  {{- include "gbstepbase.tplNodeSelector" . }}
  {{- include "gbstepbase.tplNodeAffinity" . }}

  schedulerName: scheduler-plugins-scheduler
  containers:
  {{- $multi_containers := list }}

  {{- if and (hasKey .Values "k8s") (hasKey .Values.k8s "multi_containers") .Values.k8s.multi_containers }}
    {{- $multi_containers = .Values.k8s.multi_containers }}
  {{- end }}

  {{- $numContainers := ternary (len $multi_containers) 1 (gt (len $multi_containers) 0) }}

  {{- if gt (len $multi_containers) 1 }}
  {{- include "gbstepbase.render-multi-containers" (dict "containers" $multi_containers "context" .) }}
  {{- else }}
  {{- include "gbstepbase.render-single-container" . }}
  {{- end }}
  
  {{- if and .Values.monitor_config (or
      (and .Values.monitor_config.sidecar_monitor (ge (len .Values.monitor_config.sidecar_monitor) 1))
      (and .Values.monitor_config.event_monitor (ge (len .Values.monitor_config.event_monitor) 1))
    )
  }}
  {{- include "gbstepbase.render-sidecar" (dict "context" . "numContainers" $numContainers) }}
  {{- end }}

  {{- include "gbstepbase.secretsToUseAsImagePullSecrets" . | indent 2 }}
  
  volumes:
  - name: devshm
    emptyDir:
      medium: Memory
  - name: logs
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

{{- define "gbstepbase.app" }}
{{- if .Values.k8s.use_app_wrapper }}
{{- if .Values.k8s.old_app_wrapper_api_version }}
apiVersion: workload.codeflare.dev/v1beta1
{{- else }}
apiVersion: workload.codeflare.dev/v1beta2
{{- end }}
kind: AppWrapper
metadata:
  name: "{{ .Release.Name }}"
  labels:
    {{- if not .Values.k8s.old_app_wrapper_api_version }}
    kueue.x-k8s.io/queue-name: {{ .Values.k8s.kueue_queue_name }}
    {{- end }}
    {{- include "gbstepbase.build-label" . | indent 4 }}
  annotations:
    workload.codeflare.dev.appwrapper/failureGracePeriodDuration: {{- if .Values.k8s.app_wrapper_config.failureGracePeriodDuration }} {{ .Values.k8s.app_wrapper_config.failureGracePeriodDuration | quote }} {{ else }} "10s" {{ end }}
    workload.codeflare.dev.appwrapper/retryPausePeriodDuration: {{- if .Values.k8s.app_wrapper_config.retryPausePeriodDuration }} {{ .Values.k8s.app_wrapper_config.retryPausePeriodDuration | quote }} {{ else }} "10s" {{ end }}
    workload.codeflare.dev.appwrapper/retryLimit: {{- if .Values.k8s.app_wrapper_config.retryLimit }} {{ .Values.k8s.app_wrapper_config.retryLimit | quote }} {{ else }} "0" {{ end }}
    workload.codeflare.dev.appwrapper/warmupGracePeriodDuration: {{- if .Values.k8s.app_wrapper_config.warmupGracePeriodDuration }} {{ .Values.k8s.app_wrapper_config.warmupGracePeriodDuration | quote }} {{ else }} "15m" {{ end }}
spec:
  {{- if .Values.k8s.old_app_wrapper_api_version }}
  priority: 5
  priorityslope: 0.0
  schedulingSpec:
    minAvailable: {{ .Values.compute_config.num_nodes }}
    requeuing:
      {{- range $k, $v := .Values.app_wrapper_config.requeuing }}
      {{ $k | quote }}: {{ $v }}
      {{- end }}
  resources:
    GenericItems:
    - replicas: 1
      generictemplate:
        {{- if .Values.k8s.old_pod_group_api_version }}
        apiVersion: scheduling.x-k8s.io/v1alpha1
        {{- else }}
        apiVersion: scheduling.sigs.k8s.io/v1alpha1
        {{- end }}
        kind: PodGroup
        metadata:
          name: "{{ .Release.Name }}"
          namespace: "{{ .Release.Namespace }}"
          labels:
            appwrapper.mcad.ibm.com: "{{ .Release.Name }}"
            {{- include "gbstepbase.build-label" . | indent 12 }}
          annotations:
            {{- include "gbstepbase.build-anno" . | indent 12 }}
        spec:
          minMember: {{ .Values.compute_config.num_nodes | default 1 }}
  {{- else }}
  components:
  {{- end }}
    - {{- if .Values.k8s.old_app_wrapper_api_version }}
      replicas: 1
      completionstatus: "Succeeded"
      custompodresources:
      - replicas: {{ .Values.compute_config.num_nodes }}
        limits:
          {{- include "gbstepbase.tplResourceRequests" . | indent 12 | trimAll " " }}
        requests:
          {{- include "gbstepbase.tplResourceRequests" . | indent 12 | trimAll " " }}
      generictemplate:
      {{- else }}
      template:
      {{- end }}
      {{- if eq .job_type "job" }}
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: "{{ .Release.Name }}-job"
          namespace: "{{ .Release.Namespace }}"
          labels:
            {{- include "gbstepbase.build-label" . | indent 12 }}
          annotations:
            {{- include "gbstepbase.build-anno" . | indent 12 }}
        spec:
          backoffLimit: 0
          template:
            metadata:
              name: "{{ .Release.Name }}-job"
              namespace: "{{ .Release.Namespace }}"
              labels:
                {{- include "gbstepbase.build-label" . | indent 16 }}
              annotations:
                {{- include "gbstepbase.build-anno" . | indent 16 }}
            {{- include "gbstepbase.pyjobpod" . | indent 12 }}
      {{- else }}
        {{- include "gbstepbase.pyjob" . | indent 8 }}
      {{- end }}
{{- end }}
{{- end }}


{{- define "gbstepbase.nccl-env-vars" }}
{{- if .Values.k8s.internode_networking }}

{{- range $k, $v := .Values.k8s.internode_networking.env }}
- name: {{ $k | quote }}
  value: {{ $v | quote }}
{{- end }}
{{- end }}
{{- end }}

app: {{ .Values.jobName }}

{{- define "gbstepbase.pyjobpodmaster" }}
metadata:
  labels:
    {{- include "gbstepbase.build-label" . | indent 4 }}
    app: {{ .Release.Name }}
  annotations:
    {{- $pods := .Values.compute_config.num_nodes | default 1 }}
    {{- if gt ( $pods | int ) 1 }}
    {{- if .Values.k8s.internode_networking }}

    {{- if .Values.k8s.internode_networking.multi_nic_network_name }}
    k8s.v1.cni.cncf.io/networks: {{ .Values.k8s.internode_networking.multi_nic_network_name }}
    {{- end }}
    {{- end }}
    {{- end }}
    {{- include "gbstepbase.build-anno" . | indent 4 }}
{{- end }}
{{- define "gbstepbase.pyjobpodworker" }}
metadata:
  labels:
    {{- include "gbstepbase.build-label" . | indent 4 }}
  annotations:
    {{- $pods := .Values.compute_config.num_nodes | default 1 }}
    {{- if gt ( $pods | int ) 1 }}
    {{- if .Values.k8s.internode_networking }}
    {{- if .Values.k8s.internode_networking.multi_nic_network_name }}
    k8s.v1.cni.cncf.io/networks: {{ .Values.k8s.internode_networking.multi_nic_network_name }}
    {{- end }}
    {{- end }}
    {{- end }}
    {{- include "gbstepbase.build-anno" . | indent 4 }}
{{- end }}

{{- define "gbstepbase.pyjob" }}
{{- $pods := .Values.compute_config.num_nodes | default 1 }}
apiVersion: "kubeflow.org/v1"
kind: PyTorchJob
metadata:
  name: "{{ .Release.Name }}"
  labels:
    {{- include "gbstepbase.build-label" . | indent 4 }}
  annotations:
    {{- include "gbstepbase.build-anno" . | indent 4 }}
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      # Do not restart the pod on failure.
      # If you do set it to OnFailure, be sure to also set backoffLimit
      restartPolicy: Never
      template:
        {{- include "gbstepbase.pyjobpodmaster" . | indent 8 }}
        {{- include "gbstepbase.pyjobpod" . | indent 8 }}
    {{- if gt ( $pods | int ) 1 }} {{- /*Including a worker spec when only 1 pod (Master) is specified leads to strange behavior */}}
    Worker:
      replicas: {{ sub ( $pods | int ) 1 }}
      # Do not restart the pod on failure.
      # If you do set it to OnFailure, be sure to also set backoffLimit
      restartPolicy: Never
      template:
        {{- include "gbstepbase.pyjobpodworker" . | indent 8 }}
        {{- include "gbstepbase.pyjobpod" . | indent 8 }}
    {{- end }}
{{- end }}