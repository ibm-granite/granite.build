# SPDX-License-Identifier: Apache-2.0

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

The API server's own Kubernetes client caches are keyed by context name
only; gbserver's ``sky_patches`` server plugin keys them by ``KUBECONFIG`` too
(``_patch_kube_caches_by_kubeconfig``), so two instances defining the same
context name never share a client, even on the server's shared threads.
"""

import atexit
import base64
import binascii
import contextvars
import copy
import fcntl
import hashlib
import os
import re
import socket
import tempfile
import threading
import weakref
from pathlib import Path
from secrets import token_hex
from typing import Any, Dict, List, NamedTuple, Optional, Set

import psutil
import yaml

from gbserver.environment.skypilot_config import _home
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
_KUBE_CONTEXT_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/+=-]*")
#: A Kubernetes namespace is a DNS-1123 label. Checked, not just typed: SkyPilot
#: puts it unquoted into the SSH ProxyCommand it writes on the gbserver host.
_KUBE_NAMESPACE_RE = re.compile(r"[a-z0-9]([-a-z0-9]{0,61}[a-z0-9])?")


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


def _load_kubeconfig_doc(kubeconfig_text: Any, context: str) -> Dict[str, Any]:
    """Parse a kubeconfig secret and check its sections are lists.

    :param kubeconfig_text: The secret's value (YAML).
    :param context: The context being extracted (for messages only).
    :returns: The parsed document.
    :raises ValueError: If it isn't a kubeconfig-shaped mapping. Never includes
        the text itself.
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
    return doc


def _context_refs(doc: Dict[str, Any], context: str) -> Dict[str, Any]:
    """The checked body of ``context`` in ``doc``: its cluster, user and namespace.

    :param doc: A parsed kubeconfig (:func:`_load_kubeconfig_doc`).
    :param context: The context to look up.
    :returns: The context's ``context:`` mapping.
    :raises ValueError: If the context is missing, has unexpected fields, lacks
        a cluster or user, or has an invalid namespace.
    """
    ctx = _named(doc.get("contexts"), context)
    if ctx is None:
        raise ValueError(f"kubeconfig does not define context '{context}'")
    refs = ctx.get("context") or {}
    _check_fields("context", context, refs, _KUBE_CONTEXT_FIELDS)
    if not refs.get("cluster") or not refs.get("user"):
        raise ValueError(
            f"kubeconfig context '{context}' must name both a cluster and a user"
        )
    if "namespace" in refs and not _KUBE_NAMESPACE_RE.fullmatch(refs["namespace"]):
        raise ValueError(
            f"kubeconfig context '{context}': namespace is not a valid Kubernetes "
            "namespace name (lowercase letters, digits and -, at most 63)"
        )
    return refs


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
    doc = _load_kubeconfig_doc(kubeconfig_text, context)
    refs = _context_refs(doc, context)
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


def _check_entry_names(
    entry: KubeConfigEntry, seen: Set[str], secrets: Dict[str, str], env_name: str
) -> None:
    """Check one ``kube_configs`` entry's context name and secret reference.

    :param entry: The entry.
    :param seen: Context names of the entries before it.
    :param secrets: Secret name -> value.
    :param env_name: For error messages.
    :raises ValueError: On a bad, reserved or repeated context name, or a secret
        that isn't in the space. The configured secret value is never echoed:
        it might be a pasted kubeconfig.
    """
    if not _KUBE_CONTEXT_NAME_RE.fullmatch(entry.context):
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
    if entry.kubeconfig not in secrets:
        raise ValueError(
            f"environment '{env_name}': kube_configs context '{entry.context}' "
            "names a secret (its 'kubeconfig' value) that is not in the "
            "space's secrets; set it to the secret's name"
        )


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
        _check_entry_names(entry, seen, secrets, env_name)
        seen.add(entry.context)
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


#: This process's owner key, ``<host>-<pid>-<start>-<nonce>`` (see
#: :func:`_owner_key`), and the lock it holds while it runs. Recomputed
#: after a fork.
_OWNER_KEY: Optional[str] = None


class _OwnerLock(NamedTuple):
    """The owner lock this process holds (:func:`hold_owner_lock`)."""

    pid: int
    fd: int
    path: Path


_OWNER_LOCK: Optional[_OwnerLock] = None
#: Serializes taking the owner lock: a second open of the lock file in this
#: process is a separate lock and would wait on the first forever.
_OWNER_LOCK_MUTEX = threading.Lock()

#: Suffix of an owner's lock file, ``<owner key>.lock``, beside its files.
_LOCK_SUFFIX = ".lock"


def _host_id() -> str:
    """A short id of this host's name.

    Not the liveness test (that's the owner lock, :func:`sweep_stale_instance_files`),
    but a limit on it: a sweep only judges files from a host of the same name,
    so a host whose filesystem doesn't share locks with another's (an NFS
    ``nolock`` mount, say) never removes that host's live files.
    """
    return hashlib.sha256(socket.gethostname().encode()).hexdigest()[:8]


def _process_tag(pid: Optional[int] = None) -> Optional[str]:
    """``<host>-<pid>-<start time>`` of a process (this one by default), or None if gone.

    The start time tells a running process from a later one that reused its pid
    (gbserver is often pid 1 in a container).
    """
    try:
        proc = psutil.Process(pid)
        return f"{_host_id()}-{proc.pid}-{int(proc.create_time())}"
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return None


def _owner_key() -> str:
    """This process's owner key: ``<host>-<pid>-<start>-<nonce>``.

    The random nonce makes it unique even for two processes that share a
    hostname, pid and start second (pid 1 in two containers), so each has its
    own lock file. Recomputed in a forked child.
    """
    global _OWNER_KEY  # pylint: disable=global-statement
    if _OWNER_KEY is None or _OWNER_KEY.split("-")[1] != str(os.getpid()):
        _OWNER_KEY = f"{_process_tag()}-{token_hex(4)}"
    return _OWNER_KEY


def hold_owner_lock(directory: Path) -> None:
    """Hold this process's owner lock in ``directory`` for as long as it runs.

    Taken before this process writes any instance file there. The kernel
    releases it when the process dies, however it dies, which is what
    :func:`sweep_stale_instance_files` tests. Idempotent per process.

    A sweep may remove a lock file between this process creating and locking
    it; then the lock is on a file no longer in the directory, so this checks
    after locking that the path is still the file it holds, and retries if not.
    If the filesystem can't lock, the file is left unlocked: a sweep then
    can't take it either, and keeps this process's files.
    """
    with _OWNER_LOCK_MUTEX:
        _hold_owner_lock(directory)


def _hold_owner_lock(directory: Path) -> None:
    """:func:`hold_owner_lock`, with :data:`_OWNER_LOCK_MUTEX` held."""
    global _OWNER_LOCK  # pylint: disable=global-statement
    if _OWNER_LOCK is not None and _OWNER_LOCK.pid == os.getpid():
        return
    path = directory / f"{_owner_key()}{_LOCK_SUFFIX}"
    while True:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if os.fstat(fd).st_ino == os.stat(path).st_ino:
                break
        except FileNotFoundError:
            pass  # removed by a sweep after we opened it: retry
        except OSError as e:
            logger.debug("can't lock %s (%s); sweeps will keep its files", path, e)
            break
        os.close(fd)
    _OWNER_LOCK = _OwnerLock(os.getpid(), fd, path)
    atexit.register(_release_owner_lock, os.getpid())


def _release_owner_lock(pid: int) -> None:
    """At exit: remove this process's files, then its lock file, then release the lock.

    The files go first, while the lock is still held: atexit runs this before
    the instances' own finalizers, and a file without its lock would be judged
    by the older pid check meanwhile.
    """
    global _OWNER_LOCK  # pylint: disable=global-statement
    owner = _OWNER_LOCK
    if owner is None or owner.pid != pid or os.getpid() != pid:
        return
    _OWNER_LOCK = None
    lock, fd = owner.path, owner.fd
    for owned in lock.parent.glob(f"{lock.name[: -len(_LOCK_SUFFIX)]}-*"):
        _remove_quietly(owned)
    lock.unlink(missing_ok=True)
    os.close(fd)


def _reset_owner_lock_in_child() -> None:
    """After a fork: a fresh mutex (another thread may have held the parent's).

    The child takes its own owner lock on its first write; the parent's
    stays shared with it, so the parent's files are kept while the child runs.
    """
    global _OWNER_LOCK_MUTEX  # pylint: disable=global-statement
    _OWNER_LOCK_MUTEX = threading.Lock()


os.register_at_fork(after_in_child=_reset_owner_lock_in_child)


def instance_kubeconfig_path(instance_id: str, home: Optional[Path] = None) -> Path:
    """Where an instance's kubeconfig lives: ``<owner key>-<id>.yaml``.

    The owner key (:func:`_owner_key`) names the owner lock that lets
    :func:`sweep_stale_instance_files` tell a running gbserver's files from
    ones left by a stopped one.
    """
    return _home(home) / ".sky" / KUBE_DIR / f"{_owner_key()}-{instance_id}.yaml"


def write_instance_kubeconfig(doc: Dict[str, Any], path: Path) -> None:
    """Write ``doc`` to ``path`` atomically, readable only by this user.

    The temporary file is created 0600 (``mkstemp``) with a unique name, so the
    credentials are never readable under a looser umask and concurrent writers
    never share a temporary file. The directory is kept at 0700.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    hold_owner_lock(path.parent)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, default_flow_style=False, sort_keys=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def remove_when_collected(owner: Any, path: Path) -> None:
    """Delete ``path`` once ``owner`` (the environment instance) is collected.

    ``weakref.finalize`` also runs at interpreter exit, so a cleanly stopped
    gbserver leaves no credentials behind; :func:`sweep_stale_instance_files`
    catches the rest.
    """
    weakref.finalize(owner, path.unlink, missing_ok=True)


def _take_lock_of_gone_owner(lock: Path) -> Optional[int]:
    """Take ``lock`` if its owner is gone; the fd, or None to keep its files.

    Kept (None) when the lock is held (the owner runs), when locking fails (we
    can't tell), and when the path no longer names the file we locked (it was
    replaced or removed meanwhile).
    """
    try:
        fd = os.open(lock, os.O_RDWR)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.fstat(fd).st_ino == os.stat(lock).st_ino:
            return fd
    except OSError:  # held (BlockingIOError), removed, or no locking here
        pass
    os.close(fd)
    return None


def sweep_stale_instance_files(directory: Optional[Path] = None) -> None:
    """Remove instance files (and temporary files) of gbserver processes that are gone.

    A file ``<owner key>-<id>.yaml`` belongs to the process holding the lock
    ``<owner key>.lock`` (:func:`hold_owner_lock`). The kernel releases that
    lock when the process dies, so the file is stale exactly when the lock can
    be taken: across pid namespaces, container restarts and reused pids alike.
    Its files are removed while the lock is held, then the lock file.

    Anything uncertain is kept: a held lock, a filesystem that can't lock, a
    file from a host of another name (:func:`_host_id`). Files of earlier
    versions, with no lock file, are judged as before: stale when their
    ``<host>-<pid>-<start>`` process no longer runs.
    """
    directory = directory or _home(None) / ".sky" / KUBE_DIR
    if not directory.is_dir():
        return
    host = _host_id()
    by_lock: Dict[Path, List[Path]] = {}
    for path in directory.iterdir():
        if path.name.endswith(_LOCK_SUFFIX):
            continue
        parts = path.name.split("-", 4)
        if len(parts) < 4 or parts[0] != host:
            continue
        if not parts[1].isdigit() or not parts[2].isdigit():
            continue
        lock = directory / ("-".join(parts[:4]) + _LOCK_SUFFIX)
        if len(parts) == 5 and lock.exists():
            by_lock.setdefault(lock, []).append(path)
        elif _process_tag(int(parts[1])) != "-".join(parts[:3]):
            _remove_quietly(path)
    for path in directory.glob(f"{host}-*{_LOCK_SUFFIX}"):
        by_lock.setdefault(path, [])
    for lock, files in by_lock.items():
        if _OWNER_LOCK is not None and _OWNER_LOCK.path == lock:
            continue  # this process's own
        fd = _take_lock_of_gone_owner(lock)
        if fd is None:
            continue
        try:
            for path in files:
                _remove_quietly(path)
            _remove_quietly(lock)
        finally:
            os.close(fd)


def _remove_quietly(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.debug("could not remove stale kubeconfig %s: %s", path, e)


def unresolvable_kubeconfig() -> str:
    """A ``KUBECONFIG`` that resolves no context at all (a path never written).

    Sent instead of the process default when an instance's own file can't be
    made, so its requests fail rather than resolve a same-named context from
    the user's or the server's kubeconfig.
    """
    return str(_home(None) / ".sky" / KUBE_DIR / "unavailable.yaml")


def request_kubeconfig(instance_file: Path) -> str:
    """The ``KUBECONFIG`` an instance's requests carry: the user's files, then its own.

    The user's files come first, so the user's own entries win any name clash
    (none is possible: clashes are refused when the file is built).
    """
    return os.pathsep.join([*kubeconfig_paths(), str(instance_file)])


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
    """Restore the ``KUBECONFIG`` in force before :func:`set_request_kubeconfig`.

    :param token: The token that call returned.
    """
    _REQUEST_KUBECONFIG.reset(token)


_INSTALLED = False


def install() -> None:
    """Patch the SkyPilot client to forward :data:`_REQUEST_KUBECONFIG`.

    Wraps ``payloads.request_body_env_vars()``, the one place every request
    body's environment is built (``RequestBody.__init__``). Outside a
    ``kube_configs`` request the contextvar is unset and the wrapper returns
    SkyPilot's own result unchanged. Also removes instance files left by
    stopped gbserver processes. Idempotent.
    """
    global _INSTALLED  # pylint: disable=global-statement
    if _INSTALLED:
        return
    try:
        sweep_stale_instance_files()
    except OSError as e:
        logger.debug("could not sweep stale kubeconfigs: %s", e)
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


def remote_identity_values(value: Any) -> List[str]:
    """Every value in a ``remote_identity`` setting (a string, or pattern maps)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for m in value if isinstance(m, dict) for v in m.values()]
    return []


def check_server_forwards_kubeconfig(env_name: str) -> None:
    """Refuse ``kube_configs`` unless the API server will apply each request's own.

    Fails closed: on a remote or deployed server, or one that drops the
    forwarded value, a request would silently fall back to the server's own
    kubeconfig, which may resolve the wrong context.

    :raises ValueError: When the guarantee can't be ensured.
    """
    from sky.server import common as server_common

    from gbserver.utils.sky_patches import KUBECONFIG_FORWARDING_ROUTE

    if not server_common.is_api_server_local():
        raise ValueError(
            f"environment '{env_name}': kube_configs needs a local SkyPilot API "
            "server; a remote one uses its own kubeconfig, never the "
            "environment's"
        )
    response = server_common.make_authenticated_request(
        "GET", KUBECONFIG_FORWARDING_ROUTE, retry=False
    )
    try:
        body = response.json() if response.ok else {}
    except ValueError:  # not JSON: an old server, or no gbserver plugin
        body = {}
    if not isinstance(body, dict) or body.get("forwarded") is not True:
        reason = body.get("reason") if isinstance(body, dict) else None
        raise ValueError(
            f"environment '{env_name}': the SkyPilot API server can't apply "
            f"per-request KUBECONFIG safely ({reason or 'no gbserver plugin answer'}), "
            "so kube_configs credentials would be ignored or exposed. Restart "
            "it as a local server (`sky api stop`; gbserver starts it again)."
        )
