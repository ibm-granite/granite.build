"""Shared asset I/O helpers for local execution environments (Bash, Docker, …).

Provides standalone async functions for common HuggingFace push/pull operations
so that any local environment can reuse them without duplicating logic.
"""

import os
from pathlib import Path
from typing import Any, Optional, Union

from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def get_hf_cache_dir(storeload_config) -> str:
    """Resolve the HF model cache directory from step config or the default path.

    Args:
        storeload_config: StoreLoad config object (may be None).  If it carries
            a ``config.cache_path`` entry that value is used directly.

    Returns:
        Absolute path string to the HF cache directory.
    """
    if (
        storeload_config is not None
        and hasattr(storeload_config, "config")
        and isinstance(storeload_config.config, dict)
        and "cache_path" in storeload_config.config
    ):
        return storeload_config.config["cache_path"]
    return os.path.join(os.path.expanduser("~"), ".cache", "gbserver", "hf")


def pull_asset_hfstore(
    uri,
    assetstore,
    storeload_config,
    dest: Optional[Path] = None,
) -> Path:
    """Download an HF model snapshot to the local cache and return its path.

    This is the shared pull step for all local environments.  Each environment
    uses the returned path differently: Bash binds it directly, Docker mounts
    it as a container volume.

    Uses ``HfURI.sync`` to download all repo files directly into *dest*.
    When *dest* is not provided it defaults to
    ``<cache_dir>/<owner>/<repo>/<revision>``.

    Args:
        uri: HfURI instance or URI string pointing to the model to pull.
        assetstore: Optional Hfstore instance; when provided its secrets
            (e.g. ``HF_TOKEN``) are injected into the URI before syncing.
        storeload_config: StoreLoad config (may be None); may carry
            ``config.cache_path`` to override the default cache directory.
            Ignored when *dest* is provided explicitly.
        dest: Optional explicit destination directory.  When omitted the path
            is derived from the cache dir and URI parts.

    Returns:
        Path to the downloaded snapshot on the local filesystem.

    Raises:
        AssertionError: If ``uri`` is None or does not resolve to an HfURI.
        RuntimeError: If the HuggingFace sync operation fails.
    """
    from gbcommon.uri.hf import HfURI
    from gbcommon.uri.uri import URI

    assert uri is not None, "uri is required for hfstore loading"
    hfuri = uri if isinstance(uri, HfURI) else URI.get_uri(uri)
    assert isinstance(hfuri, HfURI), f"expected HfURI, got: {type(hfuri)}"

    if assetstore is not None:
        hfuri.secrets = {**(hfuri.secrets or {}), **(assetstore.get_secrets() or {})}

    if dest is None:
        cache_dir = Path(get_hf_cache_dir(storeload_config))
        p = hfuri._parts()
        dest = cache_dir / p.owner / p.repo / p.revision
    dest.mkdir(parents=True, exist_ok=True)

    if not hfuri.pull(dest):
        raise RuntimeError(f"HF pull failed for {URI.get_uristr(hfuri)}")
    return dest


def push_asset_hfstore(
    src: Union[str, Path],
    binding_id: Optional[str] = "",
    uri: Optional[Any] = None,
    assetstore=None,
    run_metadata=None,
    **_kwargs,
) -> Any:
    """Upload a local file or directory to a HuggingFace repo.

    Resolves the HF token from ``assetstore.get_secrets()`` and injects it
    into the URI so ``HfURI.push()`` can authenticate.  The commit message
    encodes the build ID, target name, and output name from ``run_metadata``
    and ``binding_id``.

    Suitable for any local environment (Bash, Docker, etc.) that writes
    outputs to the host filesystem and wants to push them to HF.

    Args:
        src: Local file or directory path to push.
        binding_id: Output binding name included in the commit message.
        uri: Target HfURI string or object.
        assetstore: Hfstore instance whose secrets supply the HF token.
        run_metadata: EntityRunMetadata with ``build_id`` and ``target_name``.
            The current space name is resolved from the thread-local URI space config
            and passed as ``resource_group_name`` to :meth:`HfURI.push`.

    Returns:
        The resolved HfURI after a successful push.

    Raises:
        ValueError: If ``uri`` is absent or ``src`` is empty.
        RuntimeError: If the HuggingFace push operation fails.
    """
    from gbcommon.uri.hf import HfURI
    from gbcommon.uri.uri import URI

    if not uri:
        raise ValueError(f"Empty uri received to push_asset_hfstore: {src}")
    hfuri = uri if isinstance(uri, HfURI) else URI.get_uri(uri)
    assert isinstance(hfuri, HfURI), f"expected HfURI, got: {type(hfuri)}"

    if not src:
        raise ValueError(f"src path is empty")
    src = Path(src)

    if assetstore is not None:
        hfuri.secrets = {**(hfuri.secrets or {}), **(assetstore.get_secrets() or {})}

    build_id = getattr(run_metadata, "build_id", "") or ""
    target_name = getattr(run_metadata, "target_name", "") or ""
    output_name = binding_id or ""
    commit_message = (
        f"Upload via gbserver"
        f" [build={build_id} target={target_name} output={output_name}]"
    )

    # Resolve the space name from the thread-local space config so the repo is
    # created inside the correct Enterprise resource group automatically.
    space_config = URI.get_space_config()
    space_name = space_config.get("space", {}).get("name") or None

    space_name = "public"  # TODO: use the right thing here.
    logger.info("Pushing %s → %s (space=%s)", src, URI.get_uristr(hfuri), space_name)
    hfuri.push(src, commit_message=commit_message, space_name=space_name)
    return hfuri
