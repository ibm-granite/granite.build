# SkyPilot Kubernetes/OpenShift Setup Guide

This guide covers configuring SkyPilot to work with Kubernetes (K8s) or OpenShift clusters, enabling gbserver to provision and manage build jobs on cloud-native infrastructure.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Create SkyPilot Service Account](#create-skypilot-service-account)
- [Configure SkyPilot (Automated)](#configure-skypilot-automated)
- [Running DiGiT Builds](#running-digit-builds)
- [GPU Setup (Optional)](#gpu-setup-optional)
- [Troubleshooting](#troubleshooting)
- [Additional Resources](#additional-resources)

## Prerequisites

Before starting, ensure:

1. **Cluster Access**
   - Access to a Kubernetes or OpenShift cluster
   - `kubectl` or `oc` CLI installed and configured
   - Cluster admin or equivalent permissions
   - Valid kubeconfig file (typically at `~/.kube/config`)

2. **Python Environment**
   - Python 3.9 or higher installed
   - gbserver installed with SkyPilot extras:
     ```bash
     pip install -e ".[skypilot]"
     ```
   - This installs SkyPilot (~0.10+), kubernetes client, and dependencies

3. **Verify Cluster Access**
   ```bash
   # For Kubernetes
   kubectl cluster-info
   kubectl auth can-i create deployments --all-namespaces

   # For OpenShift
   oc cluster-info
   oc auth can-i create deployments --all-namespaces
   ```

## Create SkyPilot Service Account

SkyPilot needs a Kubernetes service account with appropriate RBAC permissions to create pods, jobs, and deployments.

> **Automated:** The setup script (`setup-skypilot.sh`) creates the service account and RBAC automatically. The manual steps below are for reference or if you need to customize permissions.

### 1. Create Namespace (Optional)

```bash
# If using a dedicated namespace:
kubectl create namespace skypilot
```

### 2. Create Service Account and RBAC

Apply the following manifest (supports both Kubernetes and OpenShift):

```yaml
---
# ServiceAccount for SkyPilot
apiVersion: v1
kind: ServiceAccount
metadata:
  name: skypilot
  namespace: skypilot

---
# ClusterRole with SkyPilot permissions
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: skypilot
rules:
  # Pod operations (create, get, list, watch, delete, logs, exec)
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["create", "get", "list", "watch", "delete", "patch", "update"]
  - apiGroups: [""]
    resources: ["pods/log"]
    verbs: ["get"]
  - apiGroups: [""]
    resources: ["pods/exec"]
    verbs: ["create", "get"]

  # Deployment operations
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["create", "get", "list", "watch", "delete", "patch", "update"]
  - apiGroups: ["apps"]
    resources: ["deployments/scale"]
    verbs: ["get", "patch", "update"]

  # Job operations
  - apiGroups: ["batch"]
    resources: ["jobs"]
    verbs: ["create", "get", "list", "watch", "delete", "patch", "update"]

  # StatefulSet operations
  - apiGroups: ["apps"]
    resources: ["statefulsets"]
    verbs: ["create", "get", "list", "watch", "delete", "patch", "update"]

  # Service operations
  - apiGroups: [""]
    resources: ["services"]
    verbs: ["create", "get", "list", "watch", "delete", "patch", "update"]

  # Secrets (for image pull secrets, credentials)
  - apiGroups: [""]
    resources: ["secrets"]
    verbs: ["get", "list"]

  # ConfigMaps
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list", "create", "patch", "update", "delete"]

  # PersistentVolumeClaims
  - apiGroups: [""]
    resources: ["persistentvolumeclaims"]
    verbs: ["create", "get", "list", "delete"]

  # Events (for monitoring)
  - apiGroups: [""]
    resources: ["events"]
    verbs: ["create", "patch"]

  # Node information (for scheduling)
  - apiGroups: [""]
    resources: ["nodes"]
    verbs: ["get", "list", "watch"]

---
# ClusterRoleBinding
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: skypilot
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: skypilot
subjects:
  - kind: ServiceAccount
    name: skypilot
    namespace: skypilot

---
# RoleBinding for namespace-specific operations (optional)
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: skypilot-namespace
  namespace: skypilot
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: skypilot-namespace
subjects:
  - kind: ServiceAccount
    name: skypilot
    namespace: skypilot

---
# Role for namespace operations
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: skypilot-namespace
  namespace: skypilot
rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["create", "get", "list", "watch", "patch", "update", "delete"]
  - apiGroups: [""]
    resources: ["secrets"]
    verbs: ["create", "get", "list", "watch", "patch", "update", "delete"]
```

**Apply the manifest:**

```bash
kubectl apply -f skypilot-rbac.yaml
```

### 3. For OpenShift: Create Security Context Constraint (SCC)

OpenShift requires additional SecurityContextConstraint (SCC) permissions:

```yaml
apiVersion: security.openshift.io/v1
kind: SecurityContextConstraint
metadata:
  name: skypilot
allowHostDirVolumePlugin: false
allowHostIPC: false
allowHostNetwork: false
allowHostPID: false
allowHostPorts: false
allowPrivilegedContainer: false
allowedCapabilities: null
allowedFlexVolumes: null
defaultAddCapabilities: null
fsGroup:
  type: MustRunAs
  ranges:
    - min: 1
      max: 65535
priority: null
readOnlyRootFilesystem: false
requiredDropCapabilities:
  - KILL
  - MKNOD
runAsUser:
  type: MustRunAsRange
  uidRangeMin: 1
  uidRangeMax: 65535
seLinuxContext:
  type: MustRunAs
supplementalGroups:
  type: MustRunAs
  ranges:
    - min: 1
      max: 65535
volumes:
  - configMap
  - downwardAPI
  - emptyDir
  - persistentVolumeClaim
  - projected
  - secret
users:
  - system:serviceaccount:skypilot:skypilot
groups: []
```

**Apply and verify:**

```bash
oc apply -f skypilot-scc.yaml
oc adm policy add-scc-to-user skypilot -z skypilot -n skypilot
oc get scc skypilot -o yaml
```

### 4. Verify Service Account

```bash
# Check service account exists
kubectl get serviceaccount skypilot -n skypilot

# Get service account token
kubectl get secret -n skypilot \
  $(kubectl get secret -n skypilot | grep skypilot-token | awk '{print $1}') \
  -o jsonpath='{.data.token}' | base64 --decode

# Verify permissions
kubectl auth can-i create pods --as=system:serviceaccount:skypilot:skypilot
kubectl auth can-i create deployments --as=system:serviceaccount:skypilot:skypilot
```

## Configure SkyPilot (Automated)

An automated setup script configures SkyPilot for your Kubernetes cluster. It creates the required K8s resources (namespace, secrets, optional PVC), generates `~/.sky/config.yaml`, and optionally deploys the SkyPilot API server for managed mode.

The script is idempotent — safe to re-run. Existing resources are skipped.

### 1. Copy and Edit the Environment File

```bash
cp docs/operators/setup/skypilot-setup.env.template .env.skypilot
```

Edit `.env.skypilot` and fill in your credentials. At minimum you need:
- `IMAGE_PULL_SECRET_PASSWORD` — your container registry password/API key
- `RITS_API_KEY` — your RITS inference API key
- `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`, `S3_ENDPOINT_URL` — if using S3/COS storage

For PVC-based builds, also set `PVC_ENABLED=true`.
For managed SkyPilot mode, set `SKYPILOT_MANAGED_ENABLED=true`.

### 2. Run the Setup Script

```bash
source .env.skypilot
bash docs/operators/setup/setup-skypilot.sh
```

The script will print `[CREATE]` for new resources and `[SKIP]` for existing ones.

### 3. Verify

The script runs `sky check kubernetes` automatically. You can also verify manually:

```bash
sky check kubernetes
kubectl get secrets -n skypilot
```

> **SkyPilot API Server:** To deploy the API server for managed mode, set `SKYPILOT_MANAGED_ENABLED=true` in your `.env.skypilot` and re-run the setup script.

## Running DiGiT Builds

DiGiT generates synthetic training data using the RITS remote inference service. It does not require GPUs — only CPU and memory.

Two build templates are available, each supporting two SkyPilot execution modes:

| Template | Data Strategy | Best For | Details |
|----------|--------------|----------|---------|
| [DiGiT_Skypilot](../../../spaces/standalone/public/templates/DiGiT_Skypilot/) | S3 file_mounts | Simple setup, single-step builds | Mounts S3/COS bucket directly into the pod |
| [DiGiT_Skypilot_PVC](../../../spaces/standalone/public/templates/DiGiT_Skypilot_PVC/) | PVC shared volume | Multi-step pipelines (download → generate → upload) | Data persists on a PVC between steps |

Both templates default to **unmanaged** SkyPilot mode (`space://environments/skypilot/kubernetes`). To switch to **managed** mode, edit the template's `build.yaml` and change:

```yaml
environment_uri: space://environments/skypilot-managed
```

### Quick Start

1. Complete [Configure SkyPilot (Automated)](#configure-skypilot-automated) above
2. Set S3 bucket URIs in `spaces/standalone/public/space.yaml`:
   ```yaml
   variables:
     S3_DIGIT_INPUT_URI: "s3://your-bucket/digit/input"
     S3_DIGIT_OUTPUT_URI: "s3://your-bucket/digit/output"
   ```
3. Launch a build:
   ```bash
   gbserver standalone --space-dir spaces/standalone/public &
   gbserver build run-and-monitor spaces/standalone/public/templates/DiGiT_Skypilot \
     --space-name public
   ```

See each template's README for full prerequisites and configuration.

### Using Managed Mode

Managed mode submits jobs to a persistent SkyPilot API server instead of launching pods directly. To use it:

1. Set `SKYPILOT_MANAGED_ENABLED=true` in your `.env.skypilot` and re-run the setup script
2. Edit the template's `build.yaml`: change `environment_uri` to `space://environments/skypilot-managed`
3. Launch the build as normal — gbserver routes it through the API server

## GPU Setup (Optional)

> **Note:** DiGiT builds do not require GPUs — they use the RITS remote inference service and only need CPU + memory. Skip this section if you are only running DiGiT.

### Install NVIDIA GPU Operator

If your cluster has GPU nodes and you want SkyPilot to provision GPU workloads, install the NVIDIA GPU Operator.

#### For Kubernetes

1. **Add NVIDIA Helm Repository**
   ```bash
   helm repo add nvidia https://nvidia.github.io/gpu-operator
   helm repo update
   ```

2. **Create GPU Operator Namespace**
   ```bash
   kubectl create namespace nvidia-gpu-operator
   ```

3. **Install GPU Operator**
   ```bash
   helm install gpu-operator nvidia/gpu-operator \
     --namespace nvidia-gpu-operator \
     --set driver.enabled=true \
     --set toolkit.enabled=true
   ```

4. **Verify Installation**
   ```bash
   kubectl get pods -n nvidia-gpu-operator
   kubectl describe nodes | grep -A 5 "gpu:"
   ```

#### For OpenShift

1. **Authenticate to Operator Hub**
   ```bash
   oc login -u kubeadmin -p <your-password> https://api.<cluster-name>:6443
   ```

2. **Install NVIDIA GPU Operator via OperatorHub**
   ```bash
   oc create namespace openshift-gpu-operator-system

   oc apply -f - <<'EOF'
   apiVersion: operators.coreos.com/v1
   kind: OperatorGroup
   metadata:
     name: gpu-operator-group
     namespace: openshift-gpu-operator-system
   ---
   apiVersion: operators.coreos.com/v1alpha1
   kind: Subscription
   metadata:
     name: gpu-operator
     namespace: openshift-gpu-operator-system
   spec:
     channel: stable
     installPlanApproval: Automatic
     name: gpu-operator
     source: community-operators
     sourceNamespace: openshift-marketplace
   EOF
   ```

3. **Wait for Operator Installation**
   ```bash
   oc wait --for condition=Succeeded csv -l operators.coreos.com/gpu-operator.openshift-gpu-operator-system='' \
     --namespace openshift-gpu-operator-system --timeout=300s
   ```

4. **Verify Installation**
   ```bash
   oc get pods -n openshift-gpu-operator-system
   oc describe nodes | grep -A 5 "gpu:"
   ```

### Label GPU Nodes

SkyPilot discovers GPU nodes via Kubernetes labels. Nodes must be labeled with `nvidia.com/gpu: "true"` and resource capacity labels.

#### Automatic Labeling (Recommended)

1. **Using SkyPilot's GPU Labeler**
   ```bash
   python -m sky.utils.kubernetes.gpu_labeler \
     --namespace-config ~/.kube/config \
     --label-gpu-nodes true
   ```

2. **Verify Labels**
   ```bash
   kubectl get nodes --show-labels | grep nvidia.com/gpu
   ```

#### Manual Labeling

If automatic labeling fails or doesn't apply all labels:

1. **List nodes with GPU resource capacity**
   ```bash
   kubectl get nodes -o custom-columns=NAME:.metadata.name,GPUS:.status.capacity.nvidia\.com/gpu
   ```

2. **Label GPU nodes**
   ```bash
   # For each node with GPU capacity:
   kubectl label nodes <node-name> nvidia.com/gpu=true
   kubectl label nodes <node-name> nvidia.com/gpu.memory=<memory-in-mb>

   # Example: Node with 80GB GPU
   kubectl label nodes node-1 nvidia.com/gpu=true
   kubectl label nodes node-1 nvidia.com/gpu.memory=81408
   ```

3. **Verify Labels**
   ```bash
   kubectl get nodes --show-labels
   kubectl describe node <node-name> | grep -A 10 "Labels:"
   ```

## Troubleshooting

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Service Account Not Found** | Error: "serviceaccount 'skypilot' not found" | Run `kubectl apply -f skypilot-rbac.yaml` to create RBAC resources. Ensure namespace `skypilot` exists: `kubectl create namespace skypilot` |
| **Insufficient Permissions** | Error: "pod creation forbidden" or "forbidden: User cannot create" | Verify RBAC: `kubectl auth can-i create pods --as=system:serviceaccount:skypilot:skypilot`. Check ClusterRoleBinding: `kubectl get clusterrolebinding skypilot` |
| **GPU Not Discovered** | `sky gpus list --infra k8s` returns empty | Verify GPU Operator is installed. Check node labels: `kubectl get nodes --show-labels \| grep gpu`. Run `python -m sky.utils.kubernetes.gpu_labeler` |
| **Kubeconfig Not Found** | Error: "kubeconfig not found" or "unable to load kubeconfig" | Ensure `~/.kube/config` exists. Verify `KUBECONFIG` env var if using custom path: `export KUBECONFIG=/path/to/kubeconfig` |
| **Pod Launch Timeout** | Pods stuck in "Pending" state | Check resource availability: `kubectl top nodes`. Check PVC availability if using persistent storage. Increase timeout in `sky.launch()` |
| **Image Pull Errors** | Error: "Failed to pull image" | Ensure container image is accessible from cluster. Create image pull secrets if using private registries: `kubectl create secret docker-registry regcred ...` |
| **OpenShift SCC Denied** | Error: "unable to validate against any security context constraint" | Set `OPENSHIFT_SCC_ENABLED=true` in your `.env.skypilot` and re-run the setup script. Or manually: `oc apply -f skypilot-scc.yaml && oc adm policy add-scc-to-user skypilot -z skypilot -n skypilot` |
| **Network Policy Blocking** | Pods cannot reach external services or API | Check NetworkPolicies in namespace: `kubectl get networkpolicies -n skypilot`. Ensure egress rules allow necessary traffic. Temporarily disable for testing: `kubectl delete networkpolicy -n skypilot --all` |
| **Memory/CPU Limits Exceeded** | Error: "Insufficient memory" or "Insufficient cpu" | Check node capacity: `kubectl describe nodes`. Adjust pod resource requests in task config. Add more nodes to cluster if needed. |
| **SkyPilot Config Not Loaded** | Config changes not taking effect | Verify config path: `cat ~/.sky/config.yaml`. Check for syntax errors: `python -c "import yaml; yaml.safe_load(open('~/.sky/config.yaml'))"` |
| **Cluster Context Not Set** | Error: "Unable to connect to server" | List contexts: `kubectl config get-contexts`. Switch context: `kubectl config use-context <context-name>` |
| **Setup Script Fails** | Error during `setup-skypilot.sh` | Ensure `kubectl` is configured and cluster is reachable. Check that required env vars are set: `source .env.skypilot && env \| grep -E 'SKYPILOT\|IMAGE_PULL\|S3_\|RITS\|PVC'` |
| **Sky Config Not Generated** | `~/.sky/config.yaml` missing after script | Check Python 3 is available: `python3 --version`. Check script output for `[ERROR]` lines. Ensure PyYAML is installed: `python3 -c "import yaml"` |
| **S3 Input Validation Fails** | Error: "input URI ... doesn't exist" with s3:// URIs | Known limitation: `CosURI.exists()` is not yet implemented and always returns False. Remove `inputs`/`outputs` blocks with s3:// URIs from build.yaml — data is accessed via `file_mounts` or step config instead |
| **Stale Space Registration** | Build uses wrong space directory or old config | Delete the old space from SQLite: `sqlite3 ~/.llmb/llmb-server.db "DELETE FROM gb_spaces WHERE name='public';"` then restart `gbserver standalone` |

## Additional Resources

- [SkyPilot Documentation](https://skypilot.readthedocs.io/)
- [SkyPilot Kubernetes Guide](https://skypilot.readthedocs.io/en/latest/docs/cloud-setup/cloud-container-setup.html)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [OpenShift Documentation](https://docs.openshift.com/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html)
- [gbserver Environment Configuration](../environment-yaml-config.md)
