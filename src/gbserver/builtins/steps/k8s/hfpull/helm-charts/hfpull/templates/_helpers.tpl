{{- define "hfpull_command" }}
{{- /* The pull is driven entirely by uri (HfURI.hfpull_step re-parses owner/
       repo/revision and any path in the repo from it), so this template needs
       only uri + path for it. path_in_repo is read only for the check below.
       Guard against an absent/partial hfpull_config (e.g. the mocked-HF /
       simulate-failure paths) so the render degrades to empty strings instead
       of nil-pointering. */}}
{{- $c := .Values.hfpull_config | default dict }}
{{- $path := $c.path | default "" }}
{{- $uri := $c.uri | default "" }}
{{- $path_in_repo := $c.path_in_repo | default "" }}

HF_DEST='{{ $path }}'
HF_PATH_IN_REPO='{{ $path_in_repo }}'

echo "Pulling HF {{ $uri }} into {{ $path }}"

python3 -c "from gbcommon.uri.hf import HfURI; exit(HfURI.hfpull_step(uri_str='{{ $uri }}', dest='{{ $path }}'))"

MY_RETURN_CODE=$?
if [[ "${MY_RETURN_CODE}" != "0" ]] ; then
    echo "HF pull failed, exit code: ${MY_RETURN_CODE}"
    exit 1
fi

{{- /* A path in the repo that matched nothing fails the pull, as in the
       skypilot and LSF hfpull steps. hfpull_step checks it too, but only if
       this pod image's gbcommon is new enough; this check needs no gbcommon.
       A bucket gets an empty path_in_repo (it is bound at the pull dir). */}}
if [[ -n "${HF_PATH_IN_REPO}" && ! -e "${HF_DEST}/${HF_PATH_IN_REPO}" ]]; then
    echo "hfpull: '${HF_PATH_IN_REPO}' matched nothing in {{ $uri }}"
    exit 1
fi

echo "Pulled HF URI: {{ $uri }} into cache ${HF_DEST}, bound at ${HF_DEST}${HF_PATH_IN_REPO:+/${HF_PATH_IN_REPO}}"

{{- end }}
