{{- define "gbstepbase.fullname" -}}
{{- if .Values.run_metadata.target_name -}}
{{ .Values.run_metadata.target_name }}-{{ .Release.Name }}
{{- else -}}
{{ .Release.Name }}
{{- end -}}
{{- end -}}


{{- define "gbstepbase.imagePullSecretNames" -}}
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


{{- define "gbstepbase.secretsToUseAsImagePullSecrets" -}}
{{- $parentContext := . }}
{{- $names := list -}}

{{- if and .Values.k8s .Values.k8s.userImagePullSecrets }}
  {{- $generated := include "gbstepbase.imagePullSecretNames" $parentContext | splitList "\n" }}
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

{{- define "gbstepbase.tplAdditionalFiles" }}
{{- if .Values.k8s.additional_files }}
echo 'create additional files'
{{- range $k, $v := .Values.k8s.additional_files }}
echo '{{ $v | b64enc }}' | base64 --decode > "{{ $k }}"
{{- end }}
{{- end }}
{{- end }}

{{- define "gbstepbase.tplNodeSelector" }}
{{- if and (.Values.k8s.nodeSelector) (kindIs "map" .Values.k8s.nodeSelector) }}
  nodeSelector:
{{ toYaml .Values.k8s.nodeSelector | indent 4 }}
{{- end }}
{{- end }}


{{- define "gbstepbase.tplNodeAffinity" }}
{{- if and .Values.k8s .Values.k8s.affinity }}
  affinity:
{{ toYaml .Values.k8s.affinity | indent 4 }}
{{- end }}
{{- end }}

{{- define "gbstepbase.addfilefromconfig" }}
{{- if .config }}
{{- $v := .config | toYaml | toString }}
{{- $filename := .filename | toString }}
echo '{{ $v | b64enc }}' | base64 --decode > {{ .filename }}
{{- end }}
{{- end }}


{{- define "gbstepbase.create_files_from_config" }}
{{- $top := . }}  {{/* save top-level context */}}

{{- if $top.Values.gb.files_to_create }}
  echo 'create additional files from the internal "gb" config'
  {{- range $i, $entry := $top.Values.gb.files_to_create }}

    {{- /* Each entry is a map with 1 key-value pair: filename : configKey */}}
    {{- range $filename, $configKey := $entry }}

      {{- if hasKey $top.Values $configKey }}
        {{- $content := index $top.Values $configKey | toYaml }}
        echo '{{ $content | b64enc }}' | base64 --decode > {{ $filename }}
      {{- else }}
        {{- /* Config key not found — create an empty file */}}
        echo "" > {{ $filename }}
      {{- end }}

    {{- end }}

  {{- end }}
{{- end }}

{{- end }}

{{- define "gbstepbase.copyStepDirEnabled" }}
{{- if and (hasKey .Values "gb") (hasKey .Values.gb "step_contents_in_env") }}
{{- if .Values.gb.step_contents_in_env -}}true{{- else -}}false{{- end -}}
{{- else -}}
true
{{- end -}}
{{- end -}}
