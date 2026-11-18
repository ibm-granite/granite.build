{{- define "hfpull_command" }}
{{- /* The pull is driven entirely by uri (HfURI.hfpull_step re-parses owner/
       repo/revision from it), so this template depends only on uri + path.
       Guard against an absent/partial hfpull_config (e.g. the mocked-HF /
       simulate-failure paths) so the render degrades to empty strings instead
       of nil-pointering. */}}
{{- $c := .Values.hfpull_config | default dict }}
{{- $path := $c.path | default "" }}
{{- $uri := $c.uri | default "" }}

echo "Pulling HF {{ $uri }} into {{ $path }}"

python3 -c "from gbcommon.uri.hf import HfURI; exit(HfURI.hfpull_step(uri_str='{{ $uri }}', dest='{{ $path }}'))"

MY_RETURN_CODE=$?
if [[ "${MY_RETURN_CODE}" != "0" ]] ; then
    echo "HF pull failed, exit code: ${MY_RETURN_CODE}"
    exit 1
fi

echo 'Pulled HF URI: {{ $uri }} into cache {{ $path }}'

{{- end }}
