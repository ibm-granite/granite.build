import json
import os
import ssl
import urllib.request

from gbserver.types.constants import GBSERVER_BUILTIN_STEP_IMAGE
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

_SA_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
_SA_NAMESPACE_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
_SA_CA_CERT_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
_DEFAULT_STEP_IMAGE = (
    "us.icr.io/cil15-shared-registry/gb-prod/gbserver:latest"
)


def _get_image_from_k8s_pod() -> str:
    """Read the current pod's first container image via the K8s API."""
    try:
        if not os.path.isfile(_SA_TOKEN_PATH):
            return ""

        with open(_SA_TOKEN_PATH, "r", encoding="utf-8") as f:
            token = f.read().strip()
        with open(_SA_NAMESPACE_PATH, "r", encoding="utf-8") as f:
            namespace = f.read().strip()

        pod_name = os.environ.get("HOSTNAME", "")
        if not pod_name:
            return ""

        api_host = os.environ.get(
            "KUBERNETES_SERVICE_HOST", "kubernetes.default.svc"
        )
        api_port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        url = (
            f"https://{api_host}:{api_port}"
            f"/api/v1/namespaces/{namespace}/pods/{pod_name}"
        )

        ctx = ssl.create_default_context(cafile=_SA_CA_CERT_PATH)
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(req, context=ctx, timeout=5) as resp:
            pod = json.loads(resp.read())

        containers = pod.get("spec", {}).get("containers", [])
        if containers:
            image = containers[0].get("image", "")
            if image:
                logger.info("Detected pod image from K8s API: %s", image)
                return image
    except Exception as e:
        logger.debug("Could not detect pod image from K8s API: %s", e)
    return ""


def get_step_image() -> str:
    """Resolve the image to use for built-in step pods.

    Priority:
    1. GBSERVER_BUILTIN_STEP_IMAGE env var (explicit override)
    2. Current pod's container image via K8s API
    3. Default fallback image
    """
    if GBSERVER_BUILTIN_STEP_IMAGE is not None:
        logger.info("Using step image from env var: %s", GBSERVER_BUILTIN_STEP_IMAGE)
        return GBSERVER_BUILTIN_STEP_IMAGE

    k8s_image = _get_image_from_k8s_pod()
    if k8s_image:
        return k8s_image

    logger.info("Using default step image: %s", _DEFAULT_STEP_IMAGE)
    return _DEFAULT_STEP_IMAGE
