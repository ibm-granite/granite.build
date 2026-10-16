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

{{- define "gbstepbase.normalizeOutputPermissions" }}
# Best-effort: make the produced artifact readable by the root group, whatever mode the
# workload chose. `umask` only masks bits off a caller's requested mode, so a writer that
# explicitly requests 0600 (safetensors' mkstemp does) still lands unreadable to a later
# pod on a different UID -- the pod UID is drawn from the namespace SCC range and is not
# stable across steps. A push step then fails with EACCES on a file that plainly exists.
#
# This is a GUARD, never a gate. The artifact may already be readable, and some mounts
# forbid chmod outright (read-only, root-squashed, or files owned by another UID), so a
# failure here says nothing about whether the step succeeded and must never fail it. It
# must not be silent either, since it predicts a later push failure. `chmod -R` continues
# past individual errors, so a partial pass still does its work.
#
# Call this after the workload's exit code is captured but BEFORE it is acted on, so a
# failed run's partial output is normalized too -- later pods still read and retry
# against it. `g+rwX` adds group-execute only to directories and already-executable
# files, never to data.
if [[ -n "${OUTPUT_PATH:-}" && -d "${OUTPUT_PATH}" ]]; then
  echo "Normalizing group permissions on ${OUTPUT_PATH}"
  # Capture rather than let chmod write to stderr: only command.sh is tee'd to
  # /logs/output.log, which is what the sidecar monitor tails, and this runs after that
  # pipeline, so a bare stderr write would not reach the log an operator reads. Cap the
  # report -- a wholly root-squashed tree emits one line per file, which would otherwise
  # flood both the log and the event stream.
  GB_CHMOD_ERR="$(chmod -R g+rwX "${OUTPUT_PATH}" 2>&1)" || true
  if [[ -n "${GB_CHMOD_ERR}" ]]; then
    GB_CHMOD_N="$(printf '%s\n' "${GB_CHMOD_ERR}" | wc -l | tr -d ' ')"
    echo "WARNING: could not fully normalize permissions on ${OUTPUT_PATH}" \
         "(${GB_CHMOD_N} path(s)). Not fatal: the artifact may already be readable," \
         "and some mounts forbid chmod. But if a later push fails with EACCES on" \
         "this artifact, this is why."
    printf '%s\n' "${GB_CHMOD_ERR}" | head -n 10 | sed 's/^/WARNING:   /'
    if (( GB_CHMOD_N > 10 )); then
      echo "WARNING:   ... and $(( GB_CHMOD_N - 10 )) more"
    fi
  fi
fi
{{- end }}
