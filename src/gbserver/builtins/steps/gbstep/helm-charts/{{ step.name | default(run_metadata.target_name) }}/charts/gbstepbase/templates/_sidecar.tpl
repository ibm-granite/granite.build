{{- define "gbstepbase.render-sidecar" }}
{{- $ctx := .context }}
{{- $numContainers := .numContainers }}
    - name: sidecar
      image: "{{ $ctx.Values.k8s.monitoring_sidecar_image }}"
      env:
        - name: NUM_USER_PROCESSES
          value: "{{ $numContainers }}"
        - name: MESSAGING_TYPE
          value: {{ $ctx.Values.k8s.messaging.type }}
        - name: MESSAGING_EXCHANGE
          value: {{ $ctx.Values.k8s.messaging.config.exchange }}
        {{- if $ctx.Values.k8s.space_secret }}
        - name: MESSAGING_AUTHENTICATION
          valueFrom:
            secretKeyRef:
              name: "{{ $ctx.Values.k8s.space_secret }}"
              key: {{ $ctx.Values.k8s.messaging.authentication_secret_name }}
        {{- end }}
      volumeMounts:
      - name: logs
        mountPath: /logs
      command:
      - bash
      - -c
      - |
        echo "===== Dumping monitor_config.yaml from ConfigMap ====="
        {{- $my_monitor_config := toYaml $ctx.Values.monitor_config }}
        echo '{{ $my_monitor_config | b64enc }}' | base64 --decode > monitor_config.yaml
        cat monitor_config.yaml || echo "No config found!"
        echo "====================================================="
        python3 -m gbserver.monitoring.sidecar \
          --exchange $MESSAGING_EXCHANGE \
          --queue {{ $ctx.Values.run_metadata.build_id }} \
          --routing-key {{ $ctx.Values.run_metadata.targetrun_id }}.{{ $ctx.Values.run_metadata.targetsteprun_id }}.{{ $ctx.Values.run_metadata.launch_id }}
{{- end }}
