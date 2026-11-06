
{{- define "lhpull_command" }}
# -----------------------------------
# VARIABLES
{{- $config := .Values.lhpull_config | required ".Values.lhpull_config is required." }}
# -----------------------------------

echo 'Pulling URI: {{ $config.uri }} for binding {{ $config.binding_id }}'
echo 'Using the lhpull_config: {{ $config.lh | toJson }}'

# -----------------------------------
{{- $use_aspera_flag := "" }}
{{- if $config.use_aspera }}
{{- $use_aspera_flag = "--use-aspera" }}
{{- end }}
# -----------------------------------
{{- $use_batch_reader_flag := "--use-batch-reader" }}
{{- if $config.use_aspera }}
{{- $use_batch_reader_flag = "" }}
{{- end }}
# -----------------------------------

# Directory where lhpull writes data (this is shared across pods)
LHPULL_DIR="{{ $config.path }}"

# Lock file used for synchronization - the lock is on the open FD
LOCK_FILE="${LHPULL_DIR}/.llmbpull.lock"

# State file for storing job state
STATE_FILE="${LHPULL_DIR}/.llmbpull.state"
echo "STATE_FILE: ${STATE_FILE}"

# Ensure the directory exists
echo "creating the LHPULL_DIR: ${LHPULL_DIR}"
mkdir -p "${LHPULL_DIR}" || exit 1

# ------------------------------------------------------------
# 9 = FD used for locking
#
# exec 9>file: opens LOCK_FILE for writing and assigns it to file descriptor 9
#
# flock 9:
#   - Acquires an exclusive lock on FD 9
#   - Blocks until lock becomes available
#   - Automatically released if process exits or crashes
# ------------------------------------------------------------

echo "creating/opening the LOCK_FILE: ${LOCK_FILE}"
# The cache is shared across pods whose UID can differ, so it must be group-writable.
# Name the owner and mode on failure: a bare "Permission denied" hides that.
if ! exec 9>"${LOCK_FILE}" || ! : >>"${STATE_FILE}"; then
  echo "ERROR: cannot write to ${LHPULL_DIR} (running as uid=$(id -u) gid=$(id -g))."
  stat -c '  %u:%g %A %n' "${LHPULL_DIR}" "${LOCK_FILE}" "${STATE_FILE}" 2>/dev/null
  echo "The shared cache directory must be group-writable (chmod -R g+rwX)."
  exit 1
fi

echo "[$(date)] Waiting to acquire lhpull lock..."
flock 9
echo "[$(date)] Acquired lhpull lock"

# BASHPID, not $$: Kubernetes collapses $$ to a literal $ in the container command.
# Read it here, not inside `| tee`, where it would name the pipeline subshell.
# HOSTNAME, not hostname: ubi10-minimal images do not ship the hostname binary.
MY_PID="${BASHPID}"
echo "RUNNING pid=${MY_PID} host=${HOSTNAME} at $(date)" | tee -a "${STATE_FILE}"

{{- if eq $config.lh.type "table" }}

echo dmf table pull -n {{ $config.lh.namespace }} -t {{ $config.lh.table_name }} -d {{ $config.path }} {{ $use_batch_reader_flag }} {{ $use_aspera_flag }} ;
dmf table pull -n {{ $config.lh.namespace }} -t {{ $config.lh.table_name }} -d {{ $config.path }} {{ $use_batch_reader_flag }} {{ $use_aspera_flag }} ;
MY_RETURN_CODE=$?

{{- else if eq $config.lh.type "model" }}

echo dmf model pull -n {{ $config.lh.namespace }} -t {{ $config.lh.table_name }} {{ $config.lh.model_label }} -d {{ $config.path }} --revision {{ $config.lh.model_revision | default "granite-dot-build" | quote }} {{ $use_aspera_flag }} ;
dmf model pull -n {{ $config.lh.namespace }} -t {{ $config.lh.table_name }} {{ $config.lh.model_label }} -d {{ $config.path }} --revision {{ $config.lh.model_revision | default "granite-dot-build" | quote }} {{ $use_aspera_flag }} ;
MY_RETURN_CODE=$?

{{- else if eq $config.lh.type "fileset" }}

echo dmf fileset pull -n {{ $config.lh.namespace }} -t  {{ $config.lh.table_name }} {{ $config.lh.fileset_label }} --version {{ $config.lh.fileset_version | default "granite-dot-build" }} -d {{ $config.path }} {{ $use_aspera_flag }} ;
dmf fileset pull -n {{ $config.lh.namespace }} -t  {{ $config.lh.table_name }} {{ $config.lh.fileset_label }} --version {{ $config.lh.fileset_version | default "granite-dot-build" }} -d {{ $config.path }} {{ $use_aspera_flag }} ;
MY_RETURN_CODE=$?

{{- else if eq $config.lh.type "dataset" }}

# --filepath, --public, --type and --description are also required.
echo dmf dataset pull -n {{ $config.lh.namespace }} -t {{ $config.lh.table_name }} -d {{ $config.path }} {{ $use_batch_reader_flag }} {{ $use_aspera_flag }} ;
dmf dataset pull -n {{ $config.lh.namespace }} -t {{ $config.lh.table_name }} -d {{ $config.path }} {{ $use_batch_reader_flag }} {{ $use_aspera_flag }} ;
MY_RETURN_CODE=$?

{{- else }}

echo 'Unsupported data type {{ $config.lh.type }}'
MY_RETURN_CODE=1

{{- end }}

if [[ "${MY_RETURN_CODE}" != '0' ]]; then
  echo "FAILED pid=${MY_PID} host=${HOSTNAME} at $(date)" | tee -a "${STATE_FILE}"
  echo "lhpull failed with exit code ${MY_RETURN_CODE} at $(date)"
  exit "${MY_RETURN_CODE}"
fi

echo "SUCCESS pid=${MY_PID} host=${HOSTNAME} at $(date)" | tee -a "${STATE_FILE}"
echo "lhpull completed successfully at $(date)"
echo 'Pulled URI: {{ $config.uri }} for binding {{ $config.binding_id }}'

# Lock is released automatically when this script exits
# because file descriptor 9 is closed

exit 0

{{- end }}
