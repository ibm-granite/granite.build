{{- define "hfpush_command" }}

echo "Pushing HF {{ .Values.hfpush_config.owner }}/{{ .Values.hfpush_config.repo }} from {{ .Values.hfpush_config.path }}"

python3 - <<'EOF'
from gbcommon.uri.hf import HfURI
exit(HfURI.hfpush_step(
    uri_str="{{ .Values.hfpush_config.uri }}",
    source_path="{{ .Values.hfpush_config.path }}",
    private={{ ternary "True" "False" (.Values.hfpush_config.hf.private | default true) }},
    resource_group_id="{{ .Values.hfpush_config.hf.resource_group_id | default "" }}",
))
EOF

MY_RETURN_CODE=$?
if [[ "${MY_RETURN_CODE}" != "0" ]] ; then
    echo "HF push failed, exit code: ${MY_RETURN_CODE}"
    exit 1
fi

echo 'Pushed HF URI: {{ .Values.hfpush_config.uri }} for binding {{ .Values.hfpush_config.binding_id }}'

{{- end }}
