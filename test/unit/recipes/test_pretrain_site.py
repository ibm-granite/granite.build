#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""The pretraining recipe's site boundary.

Everything a CLUSTER owns lives in a site file outside this repository; the
recipe and its environment template only name those keys. These tests hold that
boundary from both sides: the template renders from the shipped example and
produces a valid SkyPilot-Kubernetes environment, every templated key is either a
site key or a recipe parameter and never both, and no file under
recipes/pretrain/ carries a real cluster's name.
"""

import pathlib
import re
import subprocess
import sys

import jinja2
import pytest
import yaml
from jinja2 import meta

_REPO = pathlib.Path(__file__).resolve().parents[3]
_PRETRAIN = _REPO / "recipes" / "pretrain"
_TEMPLATE = _PRETRAIN / "environments" / "skypilot-k8s.template.yaml"
_SITE = _PRETRAIN / "site.example.yaml"
_RENDER = _REPO / "scripts" / "render_env_template.py"
_PARAMS = _PRETRAIN / "pretrain" / "parameters.yaml"
_BUILD = _PRETRAIN / "pretrain" / "build.yaml"


def _site(**over):
    values = yaml.safe_load(_SITE.read_text(encoding="utf-8"))
    values.update(over)
    return values


def _render(tmp_path, **over):
    """Render through the script the setup flow uses, not a re-implementation."""
    site = tmp_path / "site.yaml"
    site.write_text(yaml.safe_dump(_site(**over)), encoding="utf-8")
    out = tmp_path / "environment.yaml"
    done = subprocess.run(
        [sys.executable, str(_RENDER), str(_TEMPLATE), str(site), str(out)],
        capture_output=True,
        text=True,
        cwd=_REPO,
    )
    assert done.returncode == 0, done.stderr
    return yaml.safe_load(out.read_text(encoding="utf-8"))


def _pod_config(rendered):
    contexts = rendered["config"]["cloud_config"]["kubernetes"]["context_configs"]
    (context,) = contexts
    return contexts[context]["pod_config"]


def _ray_node(rendered):
    (container,) = [
        c
        for c in _pod_config(rendered)["spec"]["containers"]
        if c["name"] == "ray-node"
    ]
    return container


# ─── The template renders ─────────────────────────────────────────────────────


def test_the_example_site_renders_a_skypilot_kubernetes_environment(tmp_path):
    rendered = _render(tmp_path)
    assert rendered["type"] == "Skypilot"
    assert rendered["subtype"] == "kubernetes"
    assert rendered["config"]["default_cloud"] == "kubernetes"


def test_the_shared_volume_is_the_shared_workdir(tmp_path):
    """Each target is its own pod: a checkpoint `train` writes is visible to
    `export-N` only because both mount the same claim at the same path, and
    gbserver resolves env:// paths under shared_workdir."""
    rendered = _render(tmp_path)
    mount = _site()["RW_MOUNT"]
    assert rendered["config"]["shared_workdir"] == mount
    volume = next(
        v for v in _pod_config(rendered)["spec"]["volumes"] if v["name"] == "gb-shared"
    )
    assert volume["persistentVolumeClaim"]["claimName"] == _site()["RW_PVC"]
    mounts = {m["mountPath"] for m in _ray_node(rendered)["volumeMounts"]}
    assert mounts >= {mount, "/dev/shm"}


def test_skypilot_is_pinned_to_the_sites_context(tmp_path):
    """allowed_contexts makes SkyPilot ignore the current kube context, so a build
    cannot land on whatever cluster happens to be current."""
    rendered = _render(tmp_path)
    kube = rendered["config"]["cloud_config"]["kubernetes"]
    assert kube["allowed_contexts"] == [_site()["KUBE_CONTEXT"]]
    assert list(kube["context_configs"]) == [_site()["KUBE_CONTEXT"]]


def test_roce_is_requested_on_both_limits_and_requests(tmp_path):
    """An extended resource in only one of the two is rejected by the API server,
    and the Multus attachment count must match the VF count."""
    rendered = _render(tmp_path)
    resources = _ray_node(rendered)["resources"]
    count = _site()["ROCE_GDR"]
    assert resources["limits"]["nvidia.com/roce_gdr"] == count
    assert resources["requests"]["nvidia.com/roce_gdr"] == count
    annotations = _pod_config(rendered)["metadata"]["annotations"]
    assert annotations["k8s.v1.cni.cncf.io/networks"] == _site()["ROCE_NETWORK"]


def test_the_container_can_pin_rdma_memory_and_knows_its_node(tmp_path):
    """Without IPC_LOCK, NCCL's memory registration fails and it falls back to
    sockets -- a multi-node run that is slow rather than broken. NODE_NAME is what
    the train step's per-node logs are named by."""
    container = _ray_node(_render(tmp_path))
    assert "IPC_LOCK" in container["securityContext"]["capabilities"]["add"]
    node_name = next(e for e in container["env"] if e["name"] == "NODE_NAME")
    assert node_name["valueFrom"]["fieldRef"]["fieldPath"] == "spec.nodeName"


def test_the_container_patches_skypilots_own(tmp_path):
    """SkyPilot merges containers BY NAME (config_utils._PATCH_MERGE_KEYS), so the
    name must be its own `ray-node` (sky/templates/kubernetes-ray.yml.j2) or this
    block adds a second container that runs nothing."""
    names = [c["name"] for c in _pod_config(_render(tmp_path))["spec"]["containers"]]
    assert names == ["ray-node"]


# ─── Blocks that must render conditionally ────────────────────────────────────


def test_no_node_affinity_when_nothing_is_excluded(tmp_path):
    """An empty nodeSelectorTerms matches NO node: an unconditional affinity block
    would make every pod of every build unschedulable, which is why the default
    empty list must render nothing at all."""
    spec = _pod_config(_render(tmp_path, EXCLUDE_NODES=[]))["spec"]
    assert "affinity" not in spec


def test_excluded_nodes_render_a_notin_affinity(tmp_path):
    spec = _pod_config(_render(tmp_path, EXCLUDE_NODES=["n1", "n2"]))["spec"]
    (term,) = spec["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"]
    (expr,) = term["matchExpressions"]
    assert (expr["key"], expr["operator"]) == ("kubernetes.io/hostname", "NotIn")
    assert expr["values"] == ["n1", "n2"]


def test_the_topology_configmap_is_optional(tmp_path):
    """A cluster that publishes no topology file must not get a volume referring to
    a ConfigMap that does not exist -- the pod would stay Pending."""
    without = _pod_config(_render(tmp_path, TOPOLOGY_CONFIGMAP=""))
    assert "topology" not in {v["name"] for v in without["spec"]["volumes"]}
    with_cm = _render(tmp_path, TOPOLOGY_CONFIGMAP="topo")
    volume = next(
        v for v in _pod_config(with_cm)["spec"]["volumes"] if v["name"] == "topology"
    )
    assert volume["configMap"]["name"] == "topo"
    mounts = {m["mountPath"] for m in _ray_node(with_cm)["volumeMounts"]}
    assert _site()["TOPOLOGY_MOUNT"] in mounts


# ─── The site boundary ────────────────────────────────────────────────────────


def _template_names(path):
    source = path.read_text(encoding="utf-8")
    env = jinja2.Environment(
        variable_start_string="$${",
        variable_end_string="}",
        block_start_string="<%",
        block_end_string="%>",
    )
    return meta.find_undeclared_variables(env.parse(source))


def test_every_template_key_is_a_site_key():
    """render_env_template.py renders under StrictUndefined: a key the template
    uses and the site file lacks fails the setup, not a build an hour later."""
    missing = _template_names(_TEMPLATE) - set(_site())
    assert not missing, sorted(missing)


def test_the_example_site_has_no_unused_key():
    """A key neither the template nor the recipe reads is a value a user is asked
    to find out for nothing."""
    used = _template_names(_TEMPLATE) | _template_names(_BUILD)
    unused = set(_site()) - used
    assert not unused, sorted(unused)


def test_no_key_is_both_a_site_key_and_a_parameter():
    """The two layers are merged in order, so a key in both would be silently
    overridden by whichever came last -- and a cluster path would travel in the
    public parameters.yaml."""
    params = set(yaml.safe_load(_PARAMS.read_text(encoding="utf-8")))
    both = params & set(_site())
    assert not both, sorted(both)


# ─── Nothing site-specific is committed ───────────────────────────────────────

# Names and strings that belong to one organisation's clusters, images or fabric.
# The scan covers every file in the recipe directory, templates included.
_LEAKS = re.compile(
    r"vela|dipc|dmf-|multi-nic-network|roce_gdr_|cil15|us\.icr\.io|"
    r"gb-read-write|mlx5_|\.ibm\.com",
    re.IGNORECASE,
)


@pytest.mark.parametrize(
    "path",
    sorted(p for p in _PRETRAIN.rglob("*") if p.is_file()),
    ids=lambda p: str(p.relative_to(_PRETRAIN)),
)
def test_no_file_names_a_real_cluster(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    hits = sorted({m.group(0).lower() for m in _LEAKS.finditer(text)})
    assert not hits, f"{path.relative_to(_PRETRAIN)}: {hits}"
