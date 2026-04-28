{{- define "my_check_exit_code" }}
MY_RETURN_CODE=$?
if [[ "${MY_RETURN_CODE}" != "0" ]] ; then
    echo "command failed with exit code: ${MY_RETURN_CODE}"
    exit 1
fi
{{- end }}

{{- define "cosrclone_command" }}
echo "Start=$(date)";
{{- $config := .Values.cos_config | required ".Values.cos_config is required." }}
{{- $bucket_name := $config.bucket_name | required ".Values.cos_config.bucket_name is required." }}
{{- $tmpconf := printf "/tmp/rclone-%s.conf" (randAlphaNum 8) -}}
{{- $useMount := default false $config.use_mount }}
{{- $push := default false $config.push }}

# Create temporary rclone.conf file
echo "Writing temporary rclone config to {{ $tmpconf }}"
cat <<EOF > {{ $tmpconf }}
[{{ $bucket_name }}]
type = s3
provider = IBMCOS
access_key_id = $COS_ACCESS_KEY_ID
secret_access_key = $COS_SECRET_ACCESS_KEY
region = {{ $config.cos.config.cos_region | default "us-east" }}
endpoint = {{ $config.cos.config.cos_endpoint | default "s3.us-east.cloud-object-storage.appdomain.cloud" }}
EOF

{{- if $useMount }}
echo "COS mounted; copying via filesystem"
mkdir -p "{{ $config.mount_dst }}"
echo cp -r "{{ $config.path }}/." "{{ $config.mount_dst }}"
cp -r "{{ $config.path }}/." "{{ $config.mount_dst }}"
{{- else if $push }}
echo "Pushing from cluster to COS: {{ $config.path }} -> {{ $bucket_name }}:{{ $config.uri }}"
echo rclone copy -P "{{ $config.path }}" "{{ $bucket_name }}:{{ $config.uri }}" --config "{{ $tmpconf }}"
rclone copy -P "{{ $config.path }}" "{{ $bucket_name }}:{{ $config.uri }}" --config "{{ $tmpconf }}"
{{- else }}
echo "Pulling from COS to cluster: {{ $bucket_name }}:{{ $config.uri }} -> {{ $config.path }}"
echo rclone copy -P "{{ $bucket_name }}:{{ $config.uri }}" "{{ $config.path }}" --config "{{ $tmpconf }}"
rclone copy -P "{{ $bucket_name }}:{{ $config.uri }}" "{{ $config.path }}" --config "{{ $tmpconf }}"
{{- end }}

# Clean up temporary rclone config
rm -f "{{ $tmpconf }}"
# -----------------------------------
{{- include "my_check_exit_code" . }}
echo "End=$(date)";
# -----------------------------------
# This is the line that we regex to trigger an event of type ARTIFACT_PUSHED_EVENT
echo 'Pushed URI: {{ $config.uri }} for binding {{ $config.binding_id }}'
{{- end }}
