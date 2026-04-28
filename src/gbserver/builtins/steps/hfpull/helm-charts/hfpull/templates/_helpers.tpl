{{- define "hfpull_command" }}

echo "Pulling HF {{ .Values.hfpull_config.owner }}/{{ .Values.hfpull_config.repo }} into {{ .Values.hfpull_config.path }}"

python3 -c "from gbcommon.uri.hf import HfURI; exit(HfURI.hfpull_step(uri_str='{{ .Values.hfpull_config.uri }}', dest='{{ .Values.hfpull_config.path }}'))"

MY_RETURN_CODE=$?
if [[ "${MY_RETURN_CODE}" != "0" ]] ; then
    echo "HF pull failed, exit code: ${MY_RETURN_CODE}"
    exit 1
fi

echo 'Pulled HF URI: {{ .Values.hfpull_config.uri }} into cache {{ .Values.hfpull_config.path }}'

{{- end }}
