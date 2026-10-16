{{- define "hfpush_command" }}
{{- /* The push is driven by uri (HfURI.hfpush_step re-parses owner/repo from
       it) plus private + resource_group_id, so this template depends only on
       uri/path/binding_id/private and hf.resource_group_id. Guard against an
       absent/partial hfpush_config (e.g. the mocked-HF / simulate-failure
       paths) so the render degrades to safe defaults instead of nil-pointering. */}}
{{- $c := .Values.hfpush_config | default dict }}
{{- $hf := $c.hf | default dict }}
{{- $path := $c.path | default "" }}
{{- $uri := $c.uri | default "" }}
{{- $bindingId := $c.binding_id | default "" }}
{{- /* `default` treats an explicit `false` as empty, so resolve `private` via
       hasKey to preserve a deliberate `private: false`; default true when absent. */}}
{{- $private := true }}
{{- if hasKey $c "private" }}{{- $private = $c.private }}{{- end }}
{{- $resourceGroupId := $hf.resource_group_id | default "" }}

echo "Pushing HF {{ $uri }} from {{ $path }}"

python3 - <<'EOF'
from gbcommon.uri.hf import HfURI
exit(HfURI.hfpush_step(
    uri_str="{{ $uri }}",
    source_path="{{ $path }}",
    private={{ ternary "True" "False" $private }},
    resource_group_id="{{ $resourceGroupId }}",
))
EOF

MY_RETURN_CODE=$?
if [[ "${MY_RETURN_CODE}" != "0" ]] ; then
    echo "HF push failed, exit code: ${MY_RETURN_CODE}"
    exit 1
fi

echo 'Pushed HF URI: {{ $uri }} for binding {{ $bindingId }}'

{{- end }}
