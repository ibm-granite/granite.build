# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-environment-instance kubeconfigs for SkyPilot ``kube_configs`` (#571).

A Skypilot environment may bring credentials for its Kubernetes contexts from
space secrets (``kube_configs``). Each *environment instance* gets its own
kubeconfig file holding only its contexts, and each SkyPilot API *request* it
makes carries ``KUBECONFIG`` = the user's own kubeconfig files + that file:

* the SkyPilot client builds every request's environment in
  ``sky.server.requests.payloads.request_body_env_vars()``; :func:`install`
  patches it to take ``KUBECONFIG`` from :data:`_REQUEST_KUBECONFIG`, a
  contextvar the environment sets around its requests;
* a local, loopback API server applies that ``KUBECONFIG`` to that request only
  (``executor.override_request_env_and_config``).

The process ``KUBECONFIG`` is never changed, so nothing else in gbserver (the
native K8s environment, kubectl/helm subprocesses, other environments) sees an
instance's file, and a request can resolve only the user's contexts and its own
instance's. A server that would drop the forwarded value (remote, deployed, or
non-loopback) is refused up front (:func:`check_server_forwards_kubeconfig`).
"""

import base64
import binascii
import contextvars
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import psutil
import yaml

from gbserver.environment.skypilot_config import _home, _write_atomic
from gbserver.types.environmentconfig import KubeConfigEntry
from gbserver.types.errors import SkypilotConfigCollisionError
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

#: Directory (under ``~/.sky``) holding one kubeconfig per environment instance.
KUBE_DIR = "gbserver-kube"

#: Default kubeconfig when ``KUBECONFIG`` is unset (as kubectl and SkyPilot use).
_DEFAULT_KUBECONFIG = "~/.kube/config"

#: The ``KUBECONFIG`` the SkyPilot requests built in this context carry, or
#: ``None`` for the client's default (the process ``KUBECONFIG``).
_REQUEST_KUBECONFIG: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "gbserver_request_kubeconfig", default=None
)

#: Path of the gbserver ``sky_patches`` plugin endpoint reporting whether the API
#: server keeps a forwarded ``KUBECONFIG`` (see :func:`forwarding_endpoint`).
FORWARDING_ROUTE = "/gbserver/kubeconfig-forwarding"

# --------------------------------------------------------------------------- #
# Validation and extraction of one context from a secret
# --------------------------------------------------------------------------- #

#: A placeholder context each instance file carries as its ``current-context``
#: when the user's own kubeconfig sets no usable one. The kubernetes client
#: rejects a ``KUBECONFIG`` with no current-context (SkyPilot then sees no
#: contexts at all), but a real secret context there would become the default
#: for anything relying on it. The placeholder points at an unroutable address
#: with no credentials, so such a request fails loudly.
_KUBE_PLACEHOLDER = "gbserver-no-default-context"
_KUBE_PLACEHOLDER_ENTRIES = {
    "contexts": {
        "name": _KUBE_PLACEHOLDER,
        "context": {"cluster": _KUBE_PLACEHOLDER, "user": _KUBE_PLACEHOLDER},
    },
    # 192.0.2.0/24 is TEST-NET-1 (RFC 5737): never routed.
    "clusters": {
        "name": _KUBE_PLACEHOLDER,
        "cluster": {"server": "https://192.0.2.1:1"},
    },
    "users": {"name": _KUBE_PLACEHOLDER, "user": {}},
}

#: Prefix for the cluster and user names each context gets
#: (``gbserver/<context>``), so they never collide with a same-named cluster or
#: user in the user's kubeconfig: the kubernetes client keeps the first
#: definition of each name across ``KUBECONFIG``.
_KUBE_NAME_PREFIX = "gbserver/"

#: The only ``user`` fields a secret may carry: static credentials, embedded as
#: data. ``exec`` and ``auth-provider`` are refused because they make every use
#: of the context run a command on the gbserver host, so anyone able to write a
#: space secret could run code there. File paths (``client-certificate``,
#: ``client-key``, ``tokenFile``) are refused because they'd resolve on the host.
_KUBE_USER_FIELDS = frozenset(
    {"token", "client-certificate-data", "client-key-data", "username", "password"}
)
#: Same rule for the ``cluster`` entry: no ``certificate-authority`` file path.
_KUBE_CLUSTER_FIELDS = frozenset(
    {
        "server",
        "certificate-authority-data",
        "insecure-skip-tls-verify",
        "tls-server-name",
        "proxy-url",
        "disable-compression",
    }
)
#: The context entry may only name its cluster and user (rewritten here) and a
#: namespace.
_KUBE_CONTEXT_FIELDS = frozenset({"cluster", "user", "namespace"})
#: Fields whose value is a boolean; every other allowed field is a string.
_KUBE_BOOL_FIELDS = frozenset({"insecure-skip-tls-verify", "disable-compression"})

#: Context names may only use these characters: no whitespace, control or
#: invisible characters that could make two names look alike, or compare
#: differently in gbserver and in kubectl/SkyPilot.
_KUBE_CONTEXT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/+=-]*$")


def reserved_kube_context(name: str) -> bool:
    """Whether a ``kube_configs`` entry may not use context ``name``.

    gbserver's placeholder, ``ssh-*`` names (SkyPilot ignores them for
    Kubernetes), and SkyPilot's in-cluster context name (``in-cluster``, or
    ``$SKYPILOT_IN_CLUSTER_CONTEXT_NAME``).
    """
    in_cluster = os.environ.get("SKYPILOT_IN_CLUSTER_CONTEXT_NAME") or "in-cluster"
    return name in (_KUBE_PLACEHOLDER, in_cluster) or name.startswith("ssh-")


def _named(entries: Any, name: Any) -> Optional[Dict]:
    """The entry called ``name`` in a kubeconfig list, or None (non-lists are empty)."""
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, dict) and entry.get("name") == name:
            return entry
    return None


def _is_base64(value: str) -> bool:
    """Whether ``value`` is standard base64, as kubectl requires for ``*-data``.

    Whitespace is ignored: kubectl accepts line-wrapped values (a certificate
    pasted as a YAML block scalar, say).
    """
    try:
        base64.b64decode("".join(value.split()), validate=True)
    except (binascii.Error, ValueError):
        return False
    return True


def _check_fields(kind: str, context: str, body: Any, allowed: frozenset) -> None:
    """Refuse a cluster/user/context body with a field outside ``allowed``.

    Values must be strings (base64 for ``*-data``), or booleans for
    :data:`_KUBE_BOOL_FIELDS`: kubectl refuses to load a whole ``KUBECONFIG``
    over one bad value. Only field *names* appear in errors, never values.

    :raises ValueError: On a malformed body, a disallowed field or a bad value.
    """
    if not isinstance(body, dict) or not all(isinstance(k, str) for k in body):
        raise ValueError(f"kubeconfig context '{context}': its {kind} is malformed")
    refused = sorted(k for k in body if k not in allowed)
    if refused:
        raise ValueError(
            f"kubeconfig context '{context}': {kind} fields {refused} are not "
            f"allowed in kube_configs (allowed: {sorted(allowed)}). exec and "
            "auth-provider plugins and file paths are refused; embed static "
            "credentials as *-data fields or a token."
        )
    wrong = sorted(
        k
        for k, v in body.items()
        if not isinstance(v, bool if k in _KUBE_BOOL_FIELDS else str)
        or (k.endswith("-data") and isinstance(v, str) and not _is_base64(v))
    )
    if wrong:
        raise ValueError(
            f"kubeconfig context '{context}': {kind} fields {wrong} have invalid "
            "values (expected strings, base64 for *-data fields, true/false for "
            f"{sorted(_KUBE_BOOL_FIELDS)})"
        )


def extract_kube_context(kubeconfig_text: Any, context: str) -> Dict[str, Dict]:
    """Pull one context, with its cluster and user, out of a kubeconfig secret.

    The cluster and user are renamed ``gbserver/<context>``, and only their
    checked bodies are kept (a sibling ``exec`` next to ``user:`` is dropped).

    :param kubeconfig_text: The secret's value: a kubeconfig document (YAML).
    :param context: The context to extract.
    :returns: ``{"context": ..., "cluster": ..., "user": ...}`` list entries.
    :raises ValueError: If the text isn't a valid kubeconfig defining the
        context and its cluster and user. Never includes the text itself.
    """
    if not isinstance(kubeconfig_text, (str, bytes)):
        raise ValueError(f"kubeconfig for context '{context}' is not text")
    try:
        doc = yaml.safe_load(kubeconfig_text) or {}
    except (yaml.YAMLError, RecursionError):
        # `from None`: PyYAML's error echoes the offending line (a token, say).
        raise ValueError(
            f"kubeconfig for context '{context}' is not valid YAML"
        ) from None
    if not isinstance(doc, dict):
        raise ValueError(f"kubeconfig for context '{context}' is not a mapping")
    for section in ("contexts", "clusters", "users"):
        if not isinstance(doc.get(section) or [], list):
            raise ValueError(
                f"kubeconfig for context '{context}': '{section}' is not a list"
            )
    ctx = _named(doc.get("contexts"), context)
    if ctx is None:
        raise ValueError(f"kubeconfig does not define context '{context}'")
    refs = ctx.get("context") or {}
    _check_fields("context", context, refs, _KUBE_CONTEXT_FIELDS)
    if not refs.get("cluster") or not refs.get("user"):
        raise ValueError(
            f"kubeconfig context '{context}' must name both a cluster and a user"
        )
    cluster = _named(doc.get("clusters"), refs["cluster"])
    user = _named(doc.get("users"), refs["user"])
    if cluster is None or user is None:
        raise ValueError(
            f"kubeconfig context '{context}' names a cluster or user it does not define"
        )
    _check_fields("cluster", context, cluster.get("cluster"), _KUBE_CLUSTER_FIELDS)
    _check_fields("user", context, user.get("user"), _KUBE_USER_FIELDS)
    name = _KUBE_NAME_PREFIX + context
    context_body = {"cluster": name, "user": name}
    if "namespace" in refs:
        context_body = {"namespace": refs["namespace"], **context_body}
    return {
        "context": {"name": context, "context": context_body},
        "cluster": {"name": name, "cluster": cluster["cluster"]},
        "user": {"name": name, "user": user["user"]},
    }


# --------------------------------------------------------------------------- #
# The user's own kubeconfig
# --------------------------------------------------------------------------- #


def kubeconfig_paths(raw: Optional[str] = None) -> List[str]:
    """``KUBECONFIG`` (or ``~/.kube/config``) as absolute, ``~``-expanded paths.

    kubectl doesn't expand ``~`` in ``KUBECONFIG`` (the Python client does), so
    a literal ``~/.kube/config`` there would hide the file from kubectl.
    """
    raw = raw if raw is not None else os.environ.get("KUBECONFIG")
    raw = raw or _DEFAULT_KUBECONFIG
    return [
        os.path.abspath(os.path.expanduser(part))
        for part in raw.split(os.pathsep)
        if part
    ]


class UserKubeconfig:
    """What the user's own kubeconfig files define."""

    def __init__(self: "UserKubeconfig") -> None:
        self.contexts: Set[str] = set()
        self.clusters: Set[str] = set()
        self.users: Set[str] = set()
        # The first and last current-context set: kubectl uses the first, the
        # kubernetes Python client the last.
        self.first_current: Optional[str] = None
        self.last_current: Optional[str] = None

    @property
    def has_default(self: "UserKubeconfig") -> bool:
        """Whether both tools' default is one of the user's own contexts."""
        return all(
            c is not None and c in self.contexts
            for c in (self.first_current, self.last_current)
        )


def read_user_kubeconfig() -> UserKubeconfig:
    """Read the user's kubeconfig files (unreadable ones are skipped, as kubectl does)."""
    found = UserKubeconfig()
    for part in kubeconfig_paths():
        path = Path(part)
        if not path.is_file():
            continue
        try:
            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError, RecursionError):
            continue
        if not isinstance(doc, dict):
            continue
        for section, names in (
            ("contexts", found.contexts),
            ("clusters", found.clusters),
            ("users", found.users),
        ):
            entries = doc.get(section)
            for entry in entries if isinstance(entries, list) else []:
                if isinstance(entry, dict) and entry.get("name"):
                    names.add(str(entry["name"]))
        if doc.get("current-context"):
            current = str(doc["current-context"])
            if found.first_current is None:
                found.first_current = current
            found.last_current = current
    return found


def _check_user_collisions(
    parts: Dict[str, Dict], user: UserKubeconfig, env_name: str
) -> None:
    """Refuse an entry that would shadow, or become, the user's own default.

    :raises SkypilotConfigCollisionError: On a same-named context, cluster or
        user in the user's kubeconfig, or a user current-context naming it.
    """
    for kind, item, theirs in (
        ("context", parts["context"], user.contexts),
        ("cluster", parts["cluster"], user.clusters),
        ("user", parts["user"], user.users),
    ):
        if item["name"] in theirs:
            raise SkypilotConfigCollisionError(
                f"environment '{env_name}': kube_configs {kind} '{item['name']}' "
                "is also defined in your own kubeconfig. Rename the context in "
                "the environment's kubeconfig secret, or drop it from kube_configs "
                "and use your own."
            )
    name = parts["context"]["name"]
    if name in (user.first_current, user.last_current):
        raise SkypilotConfigCollisionError(
            f"environment '{env_name}': kube context '{name}' is the "
            "current-context of your own kubeconfig, so it would become the "
            "default for every tool. Set your current-context to one of your own "
            "contexts, or rename the context in the environment's secret."
        )


# --------------------------------------------------------------------------- #
# The instance file
# --------------------------------------------------------------------------- #


def build_instance_kubeconfig(
    entries: List[KubeConfigEntry], secrets: Dict[str, str], env_name: str
) -> Dict[str, Any]:
    """The kubeconfig for one environment instance's ``kube_configs``.

    :param entries: The environment's ``kube_configs`` entries.
    :param secrets: Secret name -> value; each entry's ``kubeconfig`` names one.
    :param env_name: For error messages.
    :returns: A kubeconfig document with only these contexts, plus the
        placeholder default when the user's kubeconfig has no usable one.
    :raises ValueError: On a bad, repeated or reserved context name, a missing
        secret, or a secret that fails :func:`extract_kube_context`.
    :raises SkypilotConfigCollisionError: On a clash with the user's kubeconfig.
    """
    user = read_user_kubeconfig()
    doc: Dict[str, Any] = {
        "apiVersion": "v1",
        "kind": "Config",
        "clusters": [],
        "users": [],
        "contexts": [],
    }
    seen: Set[str] = set()
    for entry in entries:
        if not _KUBE_CONTEXT_NAME_RE.match(entry.context):
            raise ValueError(
                f"environment '{env_name}': kube_configs context name "
                f"{entry.context!r} may use only letters, digits and . _ : @ / + = - "
                "(starting with a letter or digit)"
            )
        if reserved_kube_context(entry.context):
            raise ValueError(
                f"environment '{env_name}': kube_configs context name "
                f"'{entry.context}' is reserved (SkyPilot's in-cluster context, an "
                "ssh-* name, or gbserver's placeholder)"
            )
        if entry.context in seen:
            raise ValueError(
                f"environment '{env_name}': kube_configs lists context "
                f"'{entry.context}' more than once"
            )
        seen.add(entry.context)
        if entry.kubeconfig not in secrets:
            # The configured value isn't echoed: it might be a pasted kubeconfig.
            raise ValueError(
                f"environment '{env_name}': kube_configs context '{entry.context}' "
                "names a secret (its 'kubeconfig' value) that is not in the "
                "space's secrets; set it to the secret's name"
            )
        parts = extract_kube_context(secrets[entry.kubeconfig], entry.context)
        _check_user_collisions(parts, user, env_name)
        doc["contexts"].append(parts["context"])
        doc["clusters"].append(parts["cluster"])
        doc["users"].append(parts["user"])
    if doc["contexts"] and not user.has_default:
        for section, placeholder in _KUBE_PLACEHOLDER_ENTRIES.items():
            doc[section].append(copy.deepcopy(placeholder))
        doc["current-context"] = _KUBE_PLACEHOLDER
    return doc


def instance_kubeconfig_path(instance_id: str, home: Optional[Path] = None) -> Path:
    """Where an environment instance's kubeconfig lives.

    The gbserver pid is part of the name, so :func:`sweep_stale_instance_files`
    can tell a running process's files from ones left by a stopped one.
    """
    return _home(home) / ".sky" / KUBE_DIR / f"{os.getpid()}-{instance_id}.yaml"


def write_instance_kubeconfig(doc: Dict[str, Any], path: Path) -> str:
    """Write ``doc`` to ``path`` (0600, atomically) if it differs; return its digest.

    Removes files left by stopped gbserver processes the first time it runs in
    a process (:func:`sweep_stale_instance_files`).
    """
    text = yaml.safe_dump(doc, default_flow_style=False, sort_keys=False)
    digest = hashlib.sha256(text.encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if _sweep_once(path.parent):
        sweep_stale_instance_files(path.parent)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        _write_atomic(path, text, mode=0o600)
    return digest


_SWEPT: Set[str] = set()


def _sweep_once(directory: Path) -> bool:
    key = f"{os.getpid()}:{directory}"
    if key in _SWEPT:
        return False
    _SWEPT.add(key)
    return True


def sweep_stale_instance_files(directory: Path) -> None:
    """Remove instance files whose gbserver process is gone.

    Files are named ``<pid>-<instance>.yaml``; one whose pid no longer runs
    (and isn't this process) is a leftover. Nothing reads another process's
    files, so a reused pid only delays the cleanup to a later start.
    """
    for path in directory.glob("*.yaml"):
        pid_text = path.name.split("-", 1)[0]
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        if pid != os.getpid() and not psutil.pid_exists(pid):
            try:
                path.unlink()
            except OSError as e:
                logger.debug("could not remove stale kubeconfig %s: %s", path, e)


def request_kubeconfig(instance_file: Path) -> str:
    """The ``KUBECONFIG`` an instance's requests carry: the user's files, then its own.

    The user's files come first, so the user's own entries win any name clash
    (none is possible: clashes are refused when the file is built).
    """
    return os.pathsep.join([*kubeconfig_paths(), str(instance_file)])


def fingerprint(entries: List[KubeConfigEntry], secrets: Dict[str, str]) -> str:
    """Digest of the inputs an instance file is built from (entries + secret values)."""
    material = json.dumps(
        [[e.context, e.kubeconfig, secrets.get(e.kubeconfig)] for e in entries]
        + [kubeconfig_paths()],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(material.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Per-request KUBECONFIG
# --------------------------------------------------------------------------- #


def current_request_kubeconfig() -> Optional[str]:
    """The ``KUBECONFIG`` requests built in the current context will carry."""
    return _REQUEST_KUBECONFIG.get()


def set_request_kubeconfig(value: Optional[str]) -> contextvars.Token:
    """Make requests built in the current context carry ``KUBECONFIG=value``."""
    return _REQUEST_KUBECONFIG.set(value)


def reset_request_kubeconfig(token: contextvars.Token) -> None:
    _REQUEST_KUBECONFIG.reset(token)


_INSTALLED = False


def install() -> None:
    """Patch the SkyPilot client to forward :data:`_REQUEST_KUBECONFIG`.

    Wraps ``payloads.request_body_env_vars()``, the one place every request
    body's environment is built (``RequestBody.__init__``). Outside a
    ``kube_configs`` request the contextvar is unset and the wrapper returns
    SkyPilot's own result unchanged. Idempotent.
    """
    global _INSTALLED  # pylint: disable=global-statement
    if _INSTALLED:
        return
    from sky.server import common as server_common
    from sky.server.requests import payloads

    original = payloads.request_body_env_vars

    def request_body_env_vars() -> dict:
        env_vars = original()
        value = _REQUEST_KUBECONFIG.get()
        # Like SkyPilot itself, only a local server is sent a KUBECONFIG.
        if value is not None and server_common.is_api_server_local():
            env_vars["KUBECONFIG"] = value
        return env_vars

    request_body_env_vars.__wrapped__ = original  # type: ignore[attr-defined]
    payloads.request_body_env_vars = request_body_env_vars
    _INSTALLED = True


def forwarding_endpoint() -> Dict[str, bool]:
    """Server side: whether this API server keeps a forwarded ``KUBECONFIG``.

    Served by the gbserver ``sky_patches`` plugin at :data:`FORWARDING_ROUTE`.
    ``executor._sanitize_request_env_vars`` keeps ``KUBECONFIG`` from a loopback
    request only when mount containment isn't enforced (a non-deployed server
    bound to loopback, unless an operator forces it).
    """
    from sky.server import common as server_common

    return {"forwarded": not server_common.should_enforce_mount_containment()}


def check_server_forwards_kubeconfig(env_name: str) -> None:
    """Refuse ``kube_configs`` unless the API server will apply each request's own.

    Fails closed: on a remote or deployed server, or one that drops the
    forwarded value, a request would silently fall back to the server's own
    kubeconfig, which may resolve the wrong context.

    :raises ValueError: When the guarantee can't be ensured.
    """
    from sky.server import common as server_common

    if not server_common.is_api_server_local():
        raise ValueError(
            f"environment '{env_name}': kube_configs needs a local SkyPilot API "
            "server; a remote one uses its own kubeconfig, never the "
            "environment's"
        )
    response = server_common.make_authenticated_request(
        "GET", FORWARDING_ROUTE, retry=False
    )
    if not response.ok or response.json().get("forwarded") is not True:
        raise ValueError(
            f"environment '{env_name}': the SkyPilot API server does not apply "
            "per-request KUBECONFIG (it is deployed, not bound to loopback, or "
            "SKYPILOT_ENFORCE_MOUNT_CONTAINMENT is set), so kube_configs "
            "credentials would be ignored. Restart it as a local server "
            "(`sky api stop`; gbserver starts it again)."
        )
