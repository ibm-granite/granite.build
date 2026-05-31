#!/usr/bin/env bash
# setup-skypilot.sh — Idempotent SkyPilot cluster setup for Kubernetes
#
# Reads docs/setup/skypilot-setup-config.yaml to resolve env vars with defaults,
# creates K8s resources idempotently, generates ~/.sky/config.yaml, and optionally
# deploys the SkyPilot API server for managed mode.
#
# Usage:
#   1. Copy env template:   cp docs/setup/skypilot-setup.env.template .env.skypilot
#   2. Edit your copy:      vi .env.skypilot
#   3. Source it:            source .env.skypilot
#   4. Run this script:     bash docs/setup/setup-skypilot.sh
#
# The script is idempotent — safe to re-run. Existing resources are skipped.

set -euo pipefail

# ── Resolve script directory (works even if called from elsewhere) ──────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_SCHEMA="${SCRIPT_DIR}/skypilot-setup-config.yaml"

# ── Logging helpers ─────────────────────────────────────────────────────────
log_skip()   { printf "  [SKIP]   %s\n" "$*"; }
log_create() { printf "  [CREATE] %s\n" "$*"; }
log_info()   { printf "  [INFO]   %s\n" "$*"; }
log_error()  { printf "  [ERROR]  %s\n" "$*" >&2; }

# Track what was created vs skipped for the summary
declare -a ACTIONS_CREATED=()
declare -a ACTIONS_SKIPPED=()

record_create() { ACTIONS_CREATED+=("$*"); }
record_skip()   { ACTIONS_SKIPPED+=("$*"); }

# ── Task 3: Config Resolution ──────────────────────────────────────────────

resolve_config() {
    # Use Python to parse the YAML schema and emit shell export statements.
    # Each leaf node with an 'env' key is resolved from the environment,
    # falling back to 'default' if present.
    local exports
    if ! exports="$(python3 - "$CONFIG_SCHEMA" <<'PYEOF'
import sys, os, yaml

def walk(node, prefix=""):
    """Walk the YAML tree and emit export lines for leaf config entries."""
    if not isinstance(node, dict):
        return
    # A leaf config entry has an 'env' key
    if "env" in node:
        env_var = node["env"]
        default = node.get("default", "")
        value = os.environ.get(env_var, str(default) if default is not None else "")
        # Shell-escape single quotes in value
        safe_value = value.replace("'", "'\\''")
        print(f"export {env_var}='{safe_value}'")
        return
    for key, child in node.items():
        if isinstance(child, dict):
            walk(child, prefix=f"{prefix}{key}.")

with open(sys.argv[1]) as f:
    schema = yaml.safe_load(f)

for key, section in schema.items():
    walk(section)
PYEOF
)"; then
        log_error "Failed to resolve config from ${CONFIG_SCHEMA}"
        exit 1
    fi
    # Safety: CONFIG_SCHEMA is a source-controlled file in this repo.
    # The Python resolver only reads env var names from it, not user input.
    eval "$exports"
}

# ── Prerequisites ───────────────────────────────────────────────────────────

check_prerequisites() {
    log_info "Checking prerequisites..."

    if ! command -v kubectl &>/dev/null; then
        log_error "kubectl is not installed or not in PATH"
        exit 1
    fi

    if ! command -v python3 &>/dev/null; then
        log_error "python3 is not installed or not in PATH"
        exit 1
    fi

    # Verify python3 has yaml module
    if ! python3 -c "import yaml" &>/dev/null; then
        log_error "Python 'yaml' module (PyYAML) is not installed. Run: pip install pyyaml"
        exit 1
    fi

    if ! kubectl cluster-info &>/dev/null; then
        log_error "Cannot reach Kubernetes cluster. Check your kubeconfig and cluster status."
        exit 1
    fi

    # Helm is only required for managed mode (API server deployment)
    if [[ "${SKYPILOT_MANAGED_ENABLED:-false}" == "true" ]]; then
        if ! command -v helm &>/dev/null; then
            log_error "helm is not installed or not in PATH (required for managed mode)"
            exit 1
        fi
    fi

    log_info "Prerequisites OK (kubectl, python3, PyYAML, cluster reachable)"
}

# ── Task 4: K8s Resource Creation ───────────────────────────────────────────

create_namespace() {
    local ns="${SKYPILOT_NAMESPACE}"
    if kubectl get namespace "${ns}" &>/dev/null; then
        log_skip "Namespace '${ns}' already exists"
        record_skip "namespace/${ns}"
    else
        kubectl create namespace "${ns}"
        log_create "Namespace '${ns}'"
        record_create "namespace/${ns}"
    fi
}

create_service_account_and_rbac() {
    local ns="${SKYPILOT_NAMESPACE}"
    local sa_name="${SKYPILOT_SERVICE_ACCOUNT}"

    if kubectl get serviceaccount "${sa_name}" -n "${ns}" &>/dev/null; then
        log_skip "Service account '${sa_name}' already exists in namespace '${ns}'"
        record_skip "serviceaccount/${sa_name}"
        return
    fi

    kubectl apply -f - <<EOF
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ${sa_name}
  namespace: ${ns}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: skypilot-${ns}
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["create", "get", "list", "watch", "delete", "patch", "update"]
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get"]
  - apiGroups: [""]
    resources: ["pods/exec"]
    verbs: ["create", "get"]
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["create", "get", "list", "watch", "delete", "patch", "update"]
  - apiGroups: ["apps"]
    resources: ["deployments/scale"]
    verbs: ["get", "patch", "update"]
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["create", "get", "list", "watch", "delete", "patch", "update"]
  - apiGroups: ["apps"]
    resources: ["statefulsets"]
    verbs: ["create", "get", "list", "watch", "delete", "patch", "update"]
  - apiGroups: [""]
    resources: ["services"]
    verbs: ["create", "get", "list", "watch", "delete", "deletecollection", "patch", "update"]
  - apiGroups: [""]
    resources: ["secrets"]
    verbs: ["get", "list"]
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list", "create", "patch", "update", "delete"]
  - apiGroups: [""]
    resources: ["persistentvolumeclaims"]
    verbs: ["create", "get", "list", "delete"]
  - apiGroups: [""]
    resources: ["events"]
    verbs: ["create", "patch"]
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["node.k8s.io"]
    resources: ["runtimeclasses"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: skypilot-${ns}
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: skypilot-${ns}
subjects:
  - kind: ServiceAccount
    name: ${sa_name}
    namespace: ${ns}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: skypilot-${ns}-namespace
  namespace: ${ns}
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: skypilot-${ns}-namespace
subjects:
  - kind: ServiceAccount
    name: ${sa_name}
    namespace: ${ns}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: skypilot-${ns}-namespace
  namespace: ${ns}
rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["create", "get", "list", "watch", "patch", "update", "delete"]
  - apiGroups: [""]
    resources: ["secrets"]
    verbs: ["create", "get", "list", "watch", "patch", "update", "delete"]
EOF

    log_create "Service account '${sa_name}' with RBAC in namespace '${ns}'"
    record_create "serviceaccount/${sa_name} + ClusterRole + bindings"
}

create_image_pull_secret() {
    local ns="${SKYPILOT_NAMESPACE}"
    local name="${IMAGE_PULL_SECRET_NAME}"
    local registry="${IMAGE_PULL_SECRET_REGISTRY}"
    local username="${IMAGE_PULL_SECRET_USERNAME}"
    local password="${IMAGE_PULL_SECRET_PASSWORD:-}"

    if kubectl get secret "${name}" -n "${ns}" &>/dev/null; then
        log_skip "Image pull secret '${name}' already exists in namespace '${ns}'"
        record_skip "secret/${name}"
        return
    fi

    if [[ -z "${password}" ]]; then
        log_error "IMAGE_PULL_SECRET_PASSWORD is required to create image pull secret '${name}'"
        log_error "Set it in your environment or .env.skypilot file and re-run."
        exit 1
    fi

    kubectl create secret docker-registry "${name}" \
        --namespace="${ns}" \
        --docker-server="${registry}" \
        --docker-username="${username}" \
        --docker-password="${password}"
    log_create "Image pull secret '${name}' in namespace '${ns}'"
    record_create "secret/${name} (docker-registry)"
}

create_s3_secret() {
    local ns="${SKYPILOT_NAMESPACE}"
    local name="${S3_SECRET_NAME}"
    local access_key="${S3_ACCESS_KEY_ID:-}"
    local secret_key="${S3_SECRET_ACCESS_KEY:-}"
    local endpoint="${S3_ENDPOINT_URL:-}"

    if kubectl get secret "${name}" -n "${ns}" &>/dev/null; then
        log_skip "S3 secret '${name}' already exists in namespace '${ns}'"
        record_skip "secret/${name}"
        return
    fi

    # All three credentials must be set to create the secret
    if [[ -z "${access_key}" || -z "${secret_key}" || -z "${endpoint}" ]]; then
        log_info "S3 credentials not fully set (S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_ENDPOINT_URL)"
        log_info "Skipping S3 secret creation. Set all three to create it."
        record_skip "secret/${name} (credentials not set)"
        return
    fi

    kubectl create secret generic "${name}" \
        --namespace="${ns}" \
        --from-literal=access-key-id="${access_key}" \
        --from-literal=secret-access-key="${secret_key}" \
        --from-literal=endpoint-url="${endpoint}"
    log_create "S3 secret '${name}' in namespace '${ns}'"
    record_create "secret/${name} (s3-credentials)"
}

create_rits_secret() {
    local ns="${SKYPILOT_NAMESPACE}"
    local name="${RITS_SECRET_NAME}"
    local key="${RITS_SECRET_KEY}"
    local api_key="${RITS_API_KEY:-}"

    if kubectl get secret "${name}" -n "${ns}" &>/dev/null; then
        log_skip "RITS secret '${name}' already exists in namespace '${ns}'"
        record_skip "secret/${name}"
        return
    fi

    if [[ -z "${api_key}" ]]; then
        log_info "RITS_API_KEY not set — skipping RITS secret creation"
        record_skip "secret/${name} (api key not set)"
        return
    fi

    kubectl create secret generic "${name}" \
        --namespace="${ns}" \
        --from-literal="${key}=${api_key}"
    log_create "RITS secret '${name}' in namespace '${ns}' (key: ${key})"
    record_create "secret/${name} (rits-credentials)"
}

create_hf_secret() {
    local ns="${SKYPILOT_NAMESPACE}"
    local name="${HF_SECRET_NAME:-hf-credentials}"
    local token="${HF_TOKEN:-}"

    if kubectl get secret "${name}" -n "${ns}" &>/dev/null; then
        log_skip "HF secret '${name}' already exists in namespace '${ns}'"
        record_skip "secret/${name}"
        return
    fi

    if [[ -z "${token}" ]]; then
        log_info "HF_TOKEN not set — skipping HuggingFace secret creation"
        record_skip "secret/${name} (token not set)"
        return
    fi

    kubectl create secret generic "${name}" \
        --namespace="${ns}" \
        --from-literal=token="${token}"
    log_create "HF secret '${name}' in namespace '${ns}' (key: token)"
    record_create "secret/${name} (hf-credentials)"
}

create_ghe_secret() {
    local ns="${SKYPILOT_NAMESPACE}"
    local name="${GHE_SECRET_NAME:-ghe-credentials}"
    local token="${GHE_TOKEN:-}"

    if kubectl get secret "${name}" -n "${ns}" &>/dev/null; then
        log_skip "GHE secret '${name}' already exists in namespace '${ns}'"
        record_skip "secret/${name}"
        return
    fi

    if [[ -z "${token}" ]]; then
        log_info "GHE_TOKEN not set — skipping GitHub Enterprise secret creation"
        record_skip "secret/${name} (token not set)"
        return
    fi

    kubectl create secret generic "${name}" \
        --namespace="${ns}" \
        --from-literal=token="${token}"
    log_create "GHE secret '${name}' in namespace '${ns}' (key: token)"
    record_create "secret/${name} (ghe-credentials)"
}

create_pvc() {
    local ns="${SKYPILOT_NAMESPACE}"
    local name="${PVC_NAME}"
    local storage="${PVC_STORAGE}"
    local storage_class="${PVC_STORAGE_CLASS:-}"

    if kubectl get pvc "${name}" -n "${ns}" &>/dev/null; then
        log_skip "PVC '${name}' already exists in namespace '${ns}'"
        record_skip "pvc/${name}"
        return
    fi

    # Build the PVC manifest — optionally include storageClassName
    local storage_class_line=""
    if [[ -n "${storage_class}" ]]; then
        storage_class_line="  storageClassName: \"${storage_class}\""
    fi

    kubectl apply -n "${ns}" -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${name}
spec:
${storage_class_line}
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: ${storage}
EOF
    log_create "PVC '${name}' (${storage}, ReadWriteMany) in namespace '${ns}'"
    record_create "pvc/${name} (${storage})"
}

# ── OpenShift SCC ─────────────────────────────────────────────────────────

create_openshift_scc() {
    local ns="${SKYPILOT_NAMESPACE}"
    local sa_name="${SKYPILOT_SERVICE_ACCOUNT}"

    # Check if oc CLI is available
    if ! command -v oc &>/dev/null; then
        log_error "oc CLI not found — required for OpenShift SCC setup"
        return 1
    fi

    # Grant 'anyuid' SCC to the SkyPilot service account so that worker pods
    # can run images as root (many ML/AI images don't set a non-root USER).
    # OpenShift's default restricted SCC forces a random UID, which breaks
    # images that expect HOME=/root to be writable.
    local subject="system:serviceaccount:${ns}:${sa_name}"
    if oc get rolebinding system:openshift:scc:anyuid -n "${ns}" -o jsonpath='{.subjects[*].name}' 2>/dev/null | grep -qw "${sa_name}"; then
        log_skip "anyuid SCC already granted to '${sa_name}' in namespace '${ns}'"
        record_skip "scc/anyuid -> ${sa_name}"
    else
        oc adm policy add-scc-to-user anyuid -z "${sa_name}" -n "${ns}"
        log_create "Granted 'anyuid' SCC to '${sa_name}' in namespace '${ns}'"
        record_create "scc/anyuid -> ${sa_name}"
    fi
}

# ── Task 5: Sky Config Generation ──────────────────────────────────────────

generate_sky_config() {
    local sky_dir="${HOME}/.sky"
    local sky_config="${sky_dir}/config.yaml"

    mkdir -p "${sky_dir}"

    # Back up existing config
    if [[ -f "${sky_config}" ]]; then
        local bak_suffix; bak_suffix="bak.$(date +%Y%m%d%H%M%S)"
        cp "${sky_config}" "${sky_config}.${bak_suffix}"
        log_info "Backed up existing config to ${sky_config}.${bak_suffix}"
    fi

    # Set namespace on the current kubectl context so SkyPilot targets the right namespace
    local ns="${SKYPILOT_NAMESPACE}"
    kubectl config set-context --current --namespace="${ns}"
    log_info "Set kubectl context namespace to '${ns}'"

    # Use Python + yaml.dump() to generate a well-formed YAML config
    if ! python3 - <<PYEOF
import yaml, os

ns = os.environ["SKYPILOT_NAMESPACE"]
sa_name = os.environ["SKYPILOT_SERVICE_ACCOUNT"]
ips_name = os.environ["IMAGE_PULL_SECRET_NAME"]
s3_secret = os.environ["S3_SECRET_NAME"]
rits_secret = os.environ["RITS_SECRET_NAME"]
rits_key = os.environ.get("RITS_SECRET_KEY", "api-key")
hf_secret = os.environ.get("HF_SECRET_NAME", "hf-credentials")
ghe_secret = os.environ.get("GHE_SECRET_NAME", "ghe-credentials")
pvc_enabled = os.environ.get("PVC_ENABLED", "false").lower() == "true"
pvc_name = os.environ.get("PVC_NAME", "gb-data-pvc")
pvc_mount = os.environ.get("PVC_MOUNT_PATH", "/gb-data")
managed_enabled = os.environ.get("SKYPILOT_MANAGED_ENABLED", "false").lower() == "true"

# Build env vars list — inject credentials from K8s secrets via secretKeyRef
env_list = [
    {
        "name": "S3_ACCESS_KEY_ID",
        "valueFrom": {"secretKeyRef": {"name": s3_secret, "key": "access-key-id", "optional": True}},
    },
    {
        "name": "S3_SECRET_ACCESS_KEY",
        "valueFrom": {"secretKeyRef": {"name": s3_secret, "key": "secret-access-key", "optional": True}},
    },
    {
        "name": "S3_ENDPOINT_URL",
        "valueFrom": {"secretKeyRef": {"name": s3_secret, "key": "endpoint-url", "optional": True}},
    },
    {
        "name": "RITS_API_KEY",
        "valueFrom": {"secretKeyRef": {"name": rits_secret, "key": rits_key, "optional": True}},
    },
    {
        "name": "HF_TOKEN",
        "valueFrom": {"secretKeyRef": {"name": hf_secret, "key": "token", "optional": True}},
    },
    {
        "name": "GHE_TOKEN",
        "valueFrom": {"secretKeyRef": {"name": ghe_secret, "key": "token", "optional": True}},
    },
]

# Build the container spec
# lifecycle hook creates .bashrc for UBI images where it's missing
# (SkyPilot's runtime setup sources ~/.bashrc and fails without it)
container = {
    "env": env_list,
    "lifecycle": {"postStart": {"exec": {"command": ["/bin/sh", "-c", "touch $HOME/.bashrc"]}}},
}

if pvc_enabled:
    container["volumeMounts"] = [{"name": pvc_name, "mountPath": pvc_mount}]

# Build pod_config
pod_config = {
    "spec": {
        "imagePullSecrets": [{"name": ips_name}],
        "containers": [container],
    }
}

if pvc_enabled:
    pod_config["spec"]["volumes"] = [
        {"name": pvc_name, "persistentVolumeClaim": {"claimName": pvc_name}}
    ]

# Build the top-level config
# Note: namespace is set via kubectl context (not a SkyPilot config field).
# Service account is set via remote_identity.
config = {
    "kubernetes": {
        "remote_identity": sa_name,
        "pod_config": pod_config,
    }
}

# Add managed mode api_server endpoint if enabled
# The Helm chart creates a service named skypilot-api-server-api-service on port 80
if managed_enabled:
    config["api_server"] = {
        "endpoint": f"http://skypilot-api-server-api-service.{ns}:80"
    }

sky_config_path = os.path.expanduser("~/.sky/config.yaml")
with open(sky_config_path, "w") as f:
    f.write("# Generated by setup-skypilot.sh — do not edit manually\n")
    f.write(f"# Namespace '{ns}' is set via kubectl context, not in this file.\n")
    f.write("# Re-run the setup script to regenerate.\n\n")
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

PYEOF
    then
        log_error "Failed to generate ${sky_config}"
        return 1
    fi

    log_create "Sky config at ${sky_config}"
    record_create "~/.sky/config.yaml"
}

# ── Task 6: Managed Mode — SkyPilot API Server ─────────────────────────────

deploy_api_server() {
    local ns="${SKYPILOT_NAMESPACE}"
    local sa_name="${SKYPILOT_SERVICE_ACCOUNT}"
    local ips_name="${IMAGE_PULL_SECRET_NAME}"
    local release_name="skypilot-api-server"

    # Check if the Helm release already exists
    if helm status "${release_name}" -n "${ns}" &>/dev/null; then
        log_skip "SkyPilot API server Helm release '${release_name}' already exists in namespace '${ns}'"
        record_skip "helm/${release_name}"
        return
    fi

    log_info "Deploying SkyPilot API server via Helm in namespace '${ns}'..."

    # Add the SkyPilot Helm repo (idempotent)
    helm repo add skypilot https://helm.skypilot.co 2>/dev/null || true
    helm repo update skypilot

    # Generate a Helm values file with the kubernetes config for the API server.
    # The API server needs its own sky config to know how to provision pods.
    local values_file; values_file="$(mktemp /tmp/skypilot-helm-values-XXXXXX.yaml)"
    trap "rm -f '${values_file}'" RETURN

    if ! python3 - "${values_file}" <<PYEOF
import yaml, os, sys

ns = os.environ["SKYPILOT_NAMESPACE"]
sa_name = os.environ["SKYPILOT_SERVICE_ACCOUNT"]
ips_name = os.environ["IMAGE_PULL_SECRET_NAME"]
s3_secret = os.environ["S3_SECRET_NAME"]
rits_secret = os.environ["RITS_SECRET_NAME"]
rits_key = os.environ.get("RITS_SECRET_KEY", "api-key")
hf_secret = os.environ.get("HF_SECRET_NAME", "hf-credentials")
ghe_secret = os.environ.get("GHE_SECRET_NAME", "ghe-credentials")
pvc_enabled = os.environ.get("PVC_ENABLED", "false").lower() == "true"
pvc_name = os.environ.get("PVC_NAME", "gb-data-pvc")
pvc_mount = os.environ.get("PVC_MOUNT_PATH", "/gb-data")

# Build the kubernetes config for the API server (same pod_config as local sky config)
env_list = [
    {"name": "S3_ACCESS_KEY_ID", "valueFrom": {"secretKeyRef": {"name": s3_secret, "key": "access-key-id", "optional": True}}},
    {"name": "S3_SECRET_ACCESS_KEY", "valueFrom": {"secretKeyRef": {"name": s3_secret, "key": "secret-access-key", "optional": True}}},
    {"name": "S3_ENDPOINT_URL", "valueFrom": {"secretKeyRef": {"name": s3_secret, "key": "endpoint-url", "optional": True}}},
    {"name": "RITS_API_KEY", "valueFrom": {"secretKeyRef": {"name": rits_secret, "key": rits_key, "optional": True}}},
    {"name": "HF_TOKEN", "valueFrom": {"secretKeyRef": {"name": hf_secret, "key": "token", "optional": True}}},
    {"name": "GHE_TOKEN", "valueFrom": {"secretKeyRef": {"name": ghe_secret, "key": "token", "optional": True}}},
]

container = {"env": env_list, "lifecycle": {"postStart": {"exec": {"command": ["/bin/sh", "-c", "touch $HOME/.bashrc"]}}}}
if pvc_enabled:
    container["volumeMounts"] = [{"name": pvc_name, "mountPath": pvc_mount}]

pod_config = {"spec": {"imagePullSecrets": [{"name": ips_name}], "containers": [container]}}
if pvc_enabled:
    pod_config["spec"]["volumes"] = [{"name": pvc_name, "persistentVolumeClaim": {"claimName": pvc_name}}]

sky_config = yaml.dump({
    "kubernetes": {"remote_identity": sa_name, "pod_config": pod_config},
    "allowed_clouds": ["kubernetes"],
}, default_flow_style=False, sort_keys=False)

# Build Helm values
values = {
    "fullnameOverride": "skypilot-api-server",
    "ingress": {"enabled": False},
    "ingress-nginx": {"enabled": False},
    "apiService": {
        "config": sky_config,
        "skipResourceCheck": True,
        "resources": {
            "requests": {"cpu": "2", "memory": "4Gi"},
            "limits": {"cpu": "2", "memory": "4Gi"},
        },
    },
}

with open(sys.argv[1], "w") as f:
    yaml.dump(values, f, default_flow_style=False, sort_keys=False)
PYEOF
    then
        log_error "Failed to generate Helm values file"
        return 1
    fi

    # Install the SkyPilot API server via Helm (without --wait so we can fix SCC first)
    if ! helm upgrade --install "${release_name}" skypilot/skypilot-nightly \
        --devel \
        --namespace "${ns}" \
        --values "${values_file}" \
        --timeout 300s; then
        log_error "SkyPilot API server Helm install failed. Check: helm status ${release_name} -n ${ns}"
        record_create "helm/${release_name} (install failed)"
        return 1
    fi

    # On OpenShift, the API server container needs to run as root (creates /root, writes /etc).
    # Grant the anyuid SCC to the Helm chart's service account.
    local api_sa="skypilot-api-server-api-sa"
    if command -v oc &>/dev/null; then
        log_info "OpenShift detected — granting 'anyuid' SCC to API server service account '${api_sa}'"
        oc adm policy add-scc-to-user anyuid -z "${api_sa}" -n "${ns}" 2>/dev/null || true
        # Restart the deployment to pick up the new SCC
        kubectl rollout restart deployment/skypilot-api-server-api-server -n "${ns}"
    fi

    # Wait for rollout
    log_info "Waiting for API server rollout (timeout 300s)..."
    if kubectl rollout status deployment/skypilot-api-server-api-server \
        -n "${ns}" --timeout=300s; then
        log_create "SkyPilot API server deployed via Helm (release: ${release_name})"
        record_create "helm/${release_name} (API server + service)"
    else
        log_error "SkyPilot API server rollout timed out. Check: kubectl get pods -n ${ns}"
        record_create "helm/${release_name} (rollout pending)"
    fi
}

# ── Task 7: Verification ───────────────────────────────────────────────────

verify_sky_check() {
    log_info "Running 'sky check kubernetes'..."

    local output
    # Capture output but don't fail if sky check reports issues
    if output="$(sky check kubernetes 2>&1)"; then
        log_info "sky check kubernetes: OK"
    else
        log_info "sky check kubernetes returned non-zero (this may be normal)"
    fi

    # Print the output for the user
    echo "${output}" | sed 's/^/    /'

    # Check for API server connection errors (expected when running locally with ClusterIP)
    if echo "${output}" | grep -qi "ApiServerConnectionError\|could not connect to.*api server"; then
        echo ""
        log_info "sky check cannot reach the API server from this machine (ClusterIP is cluster-internal only)."
        log_info "This is expected. The API server is reachable from within the cluster."
        log_info "To verify from your machine: kubectl port-forward svc/skypilot-api-server-api-service 46580:80 -n ${SKYPILOT_NAMESPACE}"
    fi

    # Check for "no GPU" / GPU-related messages and print friendly info
    if echo "${output}" | grep -qi "no gpu\|gpu.*not.*found\|gpu.*not.*available\|no accelerator"; then
        echo ""
        log_info "Your cluster does not appear to have GPUs available."
        log_info "SkyPilot will work for CPU tasks. For GPU workloads:"
        log_info "  - Ensure the NVIDIA GPU Operator is installed"
        log_info "  - Label GPU nodes: kubectl label nodes <node> nvidia.com/gpu=true"
        log_info "  - See: docs/setup/skypilot-kubernetes-setup.md#label-gpu-nodes"
    fi
}

# ── Summary ─────────────────────────────────────────────────────────────────

print_summary() {
    echo ""
    echo "================================================================"
    echo "  SkyPilot Setup Summary"
    echo "================================================================"
    echo ""

    if [[ ${#ACTIONS_CREATED[@]} -gt 0 ]]; then
        echo "  Created:"
        for action in "${ACTIONS_CREATED[@]}"; do
            echo "    + ${action}"
        done
        echo ""
    fi

    if [[ ${#ACTIONS_SKIPPED[@]} -gt 0 ]]; then
        echo "  Skipped (already exist):"
        for action in "${ACTIONS_SKIPPED[@]}"; do
            echo "    - ${action}"
        done
        echo ""
    fi

    echo "  Namespace:   ${SKYPILOT_NAMESPACE}"
    echo "  Sky config:  ~/.sky/config.yaml"

    if [[ "${SKYPILOT_MANAGED_ENABLED:-false}" == "true" ]]; then
        echo "  API server:  http://skypilot-api-server-api-service.${SKYPILOT_NAMESPACE}:80"
    fi

    echo ""
    echo "  Next steps:"
    echo "    sky check kubernetes          # verify setup"
    echo "    sky gpus list --infra k8s     # list available GPUs"
    echo "    sky launch my-task.yaml       # launch a task"
    echo ""
    echo "================================================================"
}

# ── Main ────────────────────────────────────────────────────────────────────

main() {
    echo ""
    echo "================================================================"
    echo "  SkyPilot Cluster Setup"
    echo "================================================================"
    echo ""

    # 1. Resolve config from YAML schema + env vars
    log_info "Resolving configuration from ${CONFIG_SCHEMA}..."
    resolve_config

    # 2. Check prerequisites
    check_prerequisites

    # 3. Create namespace
    echo ""
    log_info "--- Kubernetes Resources ---"
    create_namespace

    # 4. Create service account and RBAC
    create_service_account_and_rbac

    # 4b. Create OpenShift SCC if enabled
    if [[ "${OPENSHIFT_SCC_ENABLED:-false}" == "true" ]]; then
        create_openshift_scc
    else
        log_info "OpenShift SCC creation disabled (OPENSHIFT_SCC_ENABLED=false)"
    fi

    # 5. Create image pull secret
    create_image_pull_secret

    # 6. Create S3 secret
    create_s3_secret

    # 7. Create RITS secret
    create_rits_secret

    # 7b. Create HuggingFace secret
    create_hf_secret

    # 7c. Create GitHub Enterprise secret
    create_ghe_secret

    # 8. Create PVC if enabled
    if [[ "${PVC_ENABLED:-false}" == "true" ]]; then
        create_pvc
    else
        log_info "PVC creation disabled (PVC_ENABLED=false)"
    fi

    # 9. Generate sky config
    echo ""
    log_info "--- Sky Config ---"
    generate_sky_config

    # 10. Deploy API server if managed mode enabled
    if [[ "${SKYPILOT_MANAGED_ENABLED:-false}" == "true" ]]; then
        echo ""
        log_info "--- Managed Mode ---"
        deploy_api_server
    else
        log_info "Managed mode disabled (SKYPILOT_MANAGED_ENABLED=false)"
    fi

    # 11. Verify with sky check
    echo ""
    log_info "--- Verification ---"
    if command -v sky &>/dev/null; then
        verify_sky_check
    else
        log_info "'sky' CLI not found — skipping verification."
        log_info "Install SkyPilot (pip install skypilot[kubernetes]) and run: sky check kubernetes"
    fi

    # 12. Print summary
    print_summary
}

main "$@"
