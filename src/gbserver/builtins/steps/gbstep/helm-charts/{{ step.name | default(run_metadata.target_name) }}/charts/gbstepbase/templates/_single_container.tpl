{{- define "gbstepbase.render-single-container" }}
    - name: pytorch
      image: "{{ .Values.k8s.image }}"
      {{- if or .Values.k8s.service_account_name .Values.k8s.run_as_root_group }}
      securityContext:
        {{- if .Values.k8s.run_as_root_group }}
        runAsGroup: 0
        {{- end }}
        {{- if .Values.k8s.service_account_name }}
        capabilities:
          add:
          - IPC_LOCK
        {{- end }}
      {{- end }}
      env:
        - name: EXPERIMENT
          value: {{ .Release.Name }}
        - name: LLMB_TARGETSTEPRUN_ASSET_DIR
          value: {{ .Values.k8s.targetsteprun_assets_dir }}/llmb-targetsteprun-assets/{{ .Values.run_metadata.launch_id }}
        {{- $pods := .Values.compute_config.num_nodes | default 1 }}
        {{- if gt ( $pods | int ) 1 }}
        {{- if .Values.k8s.internode_networking }}

        {{- if .Values.k8s.internode_networking.topology_file_config_map }}
        - name: NCCL_TOPO_FILE
          value: /var/run/nvidia-topologyd/virtualTopology.xml
        {{- include "gbstepbase.nccl-env-vars" . | indent 8 }}
        {{- end }}
        {{- end }}
        {{- end }}
        {{- range $key, $value := .Values.k8s.env }}
        {{- if and (kindIs "map" $value) (hasKey $value "value") (kindIs "invalid" (index $value "value")) }}
        {{- else }}
        - name: {{ $key | quote }}
          {{- $value | toYaml | trimAll " " | nindent 10 }}
        {{- end }}
        {{- end }}
        {{- range $variable := .Values.environment_variables }}
        - name: {{ required "Missing 'name' in 'environmentVariables' list element" $variable.name }}
        {{- if $variable.value }}
          value: {{ $variable.value | quote }}
        {{- else if $variable.secret }}
          valueFrom:
            secretKeyRef:
              name: {{ required "Missing 'name' in 'environmentVariables.secret' list element" $variable.secret.name }}
              key: {{ required "Missing 'key' in 'environmentVariables.secret' list element" $variable.secret.key | quote }}
        {{- else if $variable.configmap }}
        valueFrom:
          configMapKeyRef:
            name: {{ required "Missing 'name' in 'environmentVariables.configmap' list element" $variable.configmap.name }}
            key: {{ required "Missing 'key' in 'environmentVariables.configmap' list element" $variable.configmap.key | quote }}
        {{- else if ( kindIs "float64" $variable.value ) }}
        value: "0"
        {{- else }}
        value: {{ required "Missing 'value' in 'environmentVariables' list element" "" }}
        {{- end }}
        {{- end }}
        
        {{- if and .Values.workload .Values.workload.k8s .Values.workload.k8s.env }}
        {{- range $item := .Values.workload.k8s.env }}
        - name: {{ required "Missing name in workload.k8s.env" $item.name }}
        {{- if $item.value }}
          value: {{ $item.value | quote }}

        {{- else if $item.valueFrom }}
          valueFrom:
            {{- if $item.valueFrom.secretKeyRef }}
            secretKeyRef:
              name: {{ $item.valueFrom.secretKeyRef.name }}
              key: {{ $item.valueFrom.secretKeyRef.key }}
            {{- else if $item.valueFrom.configMapKeyRef }}
            configMapKeyRef:
              name: {{ $item.valueFrom.configMapKeyRef.name }}
              key: {{ $item.valueFrom.configMapKeyRef.key }}
            {{- end }}

          {{- else }}
          {{- fail "workload.env entry must contain either value or valueFrom" }}
          {{- end }}

        {{- end }}
        {{- end }}
      command:
      - bash
      - -c
      - |
        set -o pipefail
        echo
        echo 'LLMB_EVENT_WORKLOAD_STATUS:running'
        {{- include "gbstepbase.tplAdditionalFiles" . | trimAll " " | indent 8 }}
        echo 'create additional files from sections in the config'
        {{- range $filename, $value := .filesfromconfig }}
        {{- include "gbstepbase.addfilefromconfig" (dict "config" $value "filename" $filename ) | trimAll " " | indent 8 }}
        {{- end }}

        {{- include "gbstepbase.create_files_from_config" . | indent 8 }}

        {{- if .Values.k8s.show_pip_freeze }}
        echo 'pip freeze'
        pip freeze
        {{- end }}

        {{- range $setup_command := .Values.setup_commands }}
        {{ $setup_command }}
        {{- end }}

        echo "Starting experiment {{ .Release.Name }}"
        
        {{- if eq (include "gbstepbase.copyStepDirEnabled" . | trim) "true" }}
          echo "Waiting for $LLMB_TARGETSTEPRUN_ASSET_DIR/.COMPLETE to appear..."

          timeout=600
          elapsed=0

          while [ ! -f $LLMB_TARGETSTEPRUN_ASSET_DIR/.COMPLETE ]; do
              if [ $elapsed -ge $timeout ]; then
                  echo "ERROR: Timeout reached (10 minutes). .COMPLETE marker not found."
                  exit 1
              fi
              sleep 1
              elapsed=$((elapsed + 1))
          done

          echo ".COMPLETE marker detected in $LLMB_TARGETSTEPRUN_ASSET_DIR. Proceeding with workload execution..."
        {{- end }}
        
        {{- if .Values.k8s.setupcommands }}
        {{- range $index, $item := .Values.k8s.setupcommands }}
        {{ $item }}
        {{- end }}
        {{- end }}

        {{- if eq (include "gbstepbase.copyStepDirEnabled" . | trim) "true" }}
          CMD_DIR="$LLMB_TARGETSTEPRUN_ASSET_DIR"
        {{- else }}
          CMD_DIR="."
        {{- end }}

        echo 'creating the command.sh'
        rm -f "$CMD_DIR/command.sh"
        cat <<'EOF' > "$CMD_DIR/command.sh"

          echo 'LLMB_EVENT_WORKLOAD_STATUS:running_command_sh'

        {{/* 1. If mock mode is enabled -> use gb.mock_command */}}
          {{- if .Values.gb.mock }}
            echo 'mock is enabled'
            {{- if .Values.gb.mock_commands }}
              {{- range $cmd := .Values.gb.mock_commands }}
                {{ "  " }}{{ $cmd }}
              {{- end }}
            {{- else }}
              {{- range $cmd := .commands }}
                {{ "  " }}{{ $cmd }}
              {{- end }}
            {{- end }}

        {{/* 2. Else if workload.commands exist -> use those */}}
        {{- else if and .Values.workload .Values.workload.commands }}
          {{- range $cmd := .Values.workload.commands }}
            {{ "  " }}{{ $cmd }}
          {{- end }}

        {{/* 3. Else fallback to appwrapper template-provided .commands */}}
        {{- else }}
          {{- range $cmd := .commands }}
            {{ "  " }}{{ $cmd }}
          {{- end }}
        {{- end }}

        {{ "EOF" }}

        sed -i 's/^  //' "$CMD_DIR/command.sh"
        chmod +x "$CMD_DIR/command.sh"

        {{- if .Values.k8s.interactive }}
        echo 'sleeping so that the user can exec into the container'
        tail -f /dev/null
        {{- end }}

        {{- $cwd := "." }}

        {{- if eq (include "gbstepbase.copyStepDirEnabled" . | trim) "true" }}
          {{- $cwd := "$LLMB_TARGETSTEPRUN_ASSET_DIR" }}
        {{- end }}

        {{- if and .Values.workload .Values.workload.cwd }}
          {{- $cwd = .Values.workload.cwd }}
        {{- end }}

        echo "Changing directory to: {{ $cwd }}"
        cd {{ $cwd }}

        echo "Current working directory: $(pwd)"
        echo "Listing contents in the current directory:"
        ls -a .

        echo "running command..."
        "$CMD_DIR/command.sh" 2>&1 | tee /logs/output.log
        
        COMMAND_SH_EXIT_CODE="$?"
        echo "COMMAND_SH_EXIT_CODE: ${COMMAND_SH_EXIT_CODE}"
        {{- if .Values.k8s.sleep_on_end }}
        echo
        echo 'sleeping at the end so that the user can exec inside the container'
        tail -f /dev/null
        {{- end }}
        if [[ "${COMMAND_SH_EXIT_CODE}" != "0" ]] ; then
          echo 'LLMB_EVENT_WORKLOAD_STATUS:failed'
          echo "The command.sh script failed with exit code: ${COMMAND_SH_EXIT_CODE}"
          exit 1
        fi
        echo 'LLMB_EVENT_WORKLOAD_STATUS:success'
      imagePullPolicy: {{ .Values.k8s.image_pull_policy | default "IfNotPresent" }}
      volumeMounts:
      - name: devshm
        mountPath: /dev/shm
      - name: logs
        mountPath: /logs
      {{- if .Values.k8s.internode_networking }}
      {{- if .Values.k8s.internode_networking.topology_file_config_map }}
      - name: topology-volume
        mountPath: /var/run/nvidia-topologyd
      {{- end }}
      {{- end }}
      {{- range $key, $value := .Values.k8s.volumes }}
      - name: {{ $key | quote }}
        mountPath: "/{{ $key }}"
      {{- end }}
      resources:
        limits:
          {{- include "gbstepbase.tplResourceRequests" . | trimAll " " | indent 10 }}
        requests:
          {{- include "gbstepbase.tplResourceRequests" . | trimAll " " | indent 10 }}
{{end}}