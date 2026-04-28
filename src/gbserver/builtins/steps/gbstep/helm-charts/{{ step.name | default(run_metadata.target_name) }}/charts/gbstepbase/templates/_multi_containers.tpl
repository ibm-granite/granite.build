{{- define "gbstepbase.render-multi-containers" }}
{{- $orig := .context }}
{{- $containers := .containers | default (list) }}
{{- range $index, $container := $containers }}
    - name: {{ printf "pytorch-%d" $index }}
      image: {{ ternary (quote $container.image) (quote $orig.Values.k8s.image) (hasKey $container "image") }}
      {{- if $orig.Values.k8s.service_account_name }}
      securityContext:
        capabilities:
          add:
          - IPC_LOCK
      {{- end }}
      env:
        - name: EXPERIMENT
          value: {{ $orig.Release.Name }}
        - name: LLMB_TARGETSTEPRUN_ASSET_DIR
          value: {{ $orig.Values.k8s.targetsteprun_assets_dir }}/llmb-targetsteprun-assets/{{ $orig.Values.run_metadata.launch_id }}
        {{- $pods := $orig.Values.compute_config.num_nodes | default 1 }}
        {{- if gt ( $pods | int ) 1 }}
        {{- if $orig.Values.k8s.internode_networking }}

        {{- if $orig.Values.k8s.internode_networking.topology_file_config_map }}
        - name: NCCL_TOPO_FILE
          value: /var/run/nvidia-topologyd/virtualTopology.xml
        {{- include "gbstepbase.nccl-env-vars" . | indent 8 }}
        {{- end }}
        {{- end }}
        {{- end }}
        {{- range $key, $value := $orig.Values.k8s.env }}
        {{- if and (kindIs "map" $value) (hasKey $value "value") (kindIs "invalid" (index $value "value")) }}
        {{- else }}
        - name: {{ $key | quote }}
          {{- $value | toYaml | trimAll " " | nindent 10 }}
        {{- end }}
        {{- end }}
        {{- range $variable := $orig.Values.environment_variables }}
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
        # ---- Merge from multi-container-specific env for this container
        {{- if hasKey $container "env" }}
        {{- range $env := $container.env }}
        - name: {{ $env.name }}
          {{- if hasKey $env "valueFrom" }}
          valueFrom:
            {{- if hasKey $env.valueFrom "secretKeyRef" }}
            secretKeyRef:
              name: {{ $env.valueFrom.secretKeyRef.name | quote }}
              key: {{ $env.valueFrom.secretKeyRef.key | quote }}
            {{- else if hasKey $env.valueFrom "configMapKeyRef" }}
            configMapKeyRef:
              name: {{ $env.valueFrom.configMapKeyRef.name | quote }}
              key: {{ $env.valueFrom.configMapKeyRef.key | quote }}
            {{- end }}
          {{- else if hasKey $env "value" }}
          value: {{ $env.value | quote }}
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
        {{- include "gbstepbase.tplAdditionalFiles" $orig | trimAll " " | indent 8 }}
        {{- range $filename, $value := $orig.filesfromconfig }}
        {{- include "gbstepbase.addfilefromconfig" (dict "config" $value "filename" $filename ) | trimAll " " | indent 8 }}
        {{- end }}
        {{- include "gbstepbase.create_files_from_config" $orig | indent 8 }}

        {{- if $orig.Values.k8s.show_pip_freeze }}
        echo 'pip freeze'
        pip freeze
        {{- end }}

        {{- range $setup_command := $orig.Values.setup_commands }}
        {{ $setup_command }}
        {{- end }}

        echo "Starting experiment {{ $orig.Release.Name }}"

        {{- if eq (include "gbstepbase.copyStepDirEnabled" $orig | trim) "true" }}
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
        

        {{- if $orig.Values.k8s.setupcommands }}
        {{- range $index, $item := $orig.Values.k8s.setupcommands }}
        {{ $item }}
        {{- end }}
        {{- end }}

        {{- if eq (include "gbstepbase.copyStepDirEnabled" $orig | trim) "true" }}
          CMD_DIR="$LLMB_TARGETSTEPRUN_ASSET_DIR"
        {{- else }}
          CMD_DIR="."
        {{- end }}
        
        
        rm -f "$CMD_DIR/command.sh"
        cat <<'EOF' > "$CMD_DIR/command.sh"

          echo 'LLMB_EVENT_WORKLOAD_STATUS:running_command_sh'

        {{/* 1. If mock mode is enabled -> use gb.mock_command */}}
        {{- if $orig.Values.gb.mock }}
            {{- if $orig.Values.gb.mock_commands }}
            {{- range $cmd := $orig.Values.gb.mock_commands }}
                {{ "  " }}{{ $cmd }}
            {{- end }}
            {{- else }}
            {{/* Fallback: default single mock command if list not provided */}}
            {{ "  " }}echo "Generated Data: /gb-read-write/outputs/digit/{{ $orig.Release.Name }}/tasks/watsonx/instructlab/knowledge/maximo/final_data.jsonl"
            {{- end }}

        {{/* 2. Else if workload.commands exist -> use those */}}
        {{- else if and $orig.Values.workload $orig.Values.workload.commands }}
        {{- range $cmd := $orig.Values.workload.commands }}
            {{ "  " }}{{ $cmd }}
        {{- end }}

        {{/* 3. Else fallback to appwrapper template-provided .commands */}}
        {{- else }}
        {{- range $cmd := $container.commands }}
            {{ "  " }}{{ $cmd }}
        {{- end }}
        {{- end }}

        {{ "EOF" }}

        sed -i 's/^  //' "$CMD_DIR/command.sh"
        chmod +x "$CMD_DIR/command.sh"

        {{- if $orig.Values.k8s.interactive }}
        echo 'sleeping so that the user can exec into the container'
        tail -f /dev/null
        {{- end }}

        {{- $cwd := "." }}

        {{- if eq (include "gbstepbase.copyStepDirEnabled" $orig | trim) "true" }}
          {{- $cwd := "$LLMB_TARGETSTEPRUN_ASSET_DIR" }}
        {{- end }}

        {{- if and $orig.Values.workload $orig.Values.workload.cwd }}
          {{- $cwd = $orig.Values.workload.cwd }}
        {{- end }}

        echo "Changing directory to: {{ $cwd }}"
        cd {{ $cwd }}

        echo "Current working directory: $(pwd)"
        echo "Listing contents in the current directory:"
        ls -a .
        

        echo "running command..."

        "$CMD_DIR/command.sh" 2>&1 | tee /logs/output-{{ $index }}.log

        COMMAND_SH_EXIT_CODE="$?"
        echo "COMMAND_SH_EXIT_CODE: ${COMMAND_SH_EXIT_CODE}"
        {{- if $orig.Values.k8s.sleep_on_end }}
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
      imagePullPolicy: {{ $orig.Values.k8s.image_pull_policy | default "IfNotPresent" }}
      volumeMounts:
      - name: devshm
        mountPath: /dev/shm
      - name: logs
        mountPath: /logs
      {{- if $orig.Values.k8s.internode_networking }}

      {{- if $orig.Values.k8s.internode_networking.topology_file_config_map }}
      - name: topology-volume
        mountPath: /var/run/nvidia-topologyd
      {{- end }}
      {{- end }}
      {{- range $key, $value := $orig.Values.k8s.volumes }}
      - name: {{ $key | quote }}
        mountPath: "/{{ $key }}"
      {{- end }}
      resources:
        limits:
          {{- include "gbstepbase.tplResourceRequests" $orig | trimAll " " | indent 10 }}
        requests:
          {{- include "gbstepbase.tplResourceRequests" $orig | trimAll " " | indent 10 }}
    {{ end }}
{{end}}