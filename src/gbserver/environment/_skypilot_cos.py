"""S3/COS asset helpers for the Skypilot environment's s3pull/s3push steps.

``Skypilot.pullasset_cosstore`` / ``Skypilot.pushasset_cosstore`` queue the
builtin ``s3pull`` / ``s3push`` steps (``builtins/steps/skypilot/s3p*/``). This
module turns the *dispatched* Cosstore's per-store settings into those steps'
config, so an environment may list several Cosstores (one ``store.yaml`` per
endpoint, routed by longest ``base_uri`` prefix) and each pull/push carries only
its own store's endpoint, region and credentials.

Credentials follow the hf pattern: the store.yaml names the secrets
(``cos_access_key_id_secret_name`` / ``cos_secret_access_key_secret_name``, as
``token_secretname`` does for hf), ``Cosstore.resolve_credentials`` resolves them
from the space secrets (then the same-named env var), and the values reach the
step as ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` via
``launcher_config.envs`` -- exactly how ``HF_TOKEN`` reaches hfpull/hfpush.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from gbcommon.types.testing import is_cos_mocked
from gbserver.asset.cosstore import normalize_s3_endpoint

# Env var names the s3pull/s3push step scripts (AWS CLI) read credentials from.
AWS_ACCESS_KEY_ID_ENV = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV = "AWS_SECRET_ACCESS_KEY"

DEFAULT_S3PULL_STEP_URI = "space://steps/s3pull"
DEFAULT_S3PUSH_STEP_URI = "space://steps/s3push"


@dataclass(frozen=True)
class CosSettings:
    """Per-store settings needed to run one s3pull/s3push step.

    :param s3_uri: the object or prefix as an ``s3://bucket/key`` URI.
    :param endpoint_url: the normalized cluster-side endpoint ('' = AWS default).
    :param region: the store's region ('' = AWS CLI default).
    """

    s3_uri: str
    endpoint_url: str
    region: str


def resolve_cos_store_settings(cosuri: Any, assetstore: Any) -> CosSettings:
    """Resolve a CosURI's settings from the store dispatch matched it to.

    Uses the dispatched ``assetstore`` (not ``Asset(cosuri)``) so the settings
    come from the exact store this environment routed the URI to.

    :param cosuri: the CosURI being pulled or pushed.
    :param assetstore: the dispatched Cosstore.
    :returns: the resolved :class:`CosSettings`.
    :raises ValueError: if the store's endpoint is not acceptable.
    """
    config = assetstore.get_metadata(cosuri)["config"]
    bucket, key = assetstore.get_bucket_and_key(cosuri)
    return CosSettings(
        s3_uri=f"s3://{bucket}/{key}" if key else f"s3://{bucket}",
        endpoint_url=normalize_s3_endpoint(config.get("cos_endpoint")),
        region=config.get("cos_region", "") or "",
    )


def cos_launcher_envs(cosuri: Any, assetstore: Any) -> Dict[str, str]:
    """Return the step's ``launcher_config.envs`` carrying the store's credentials.

    Fails fast (at pull/push planning time rather than inside the step) when a
    credential is missing. Under GBTEST_MOCK_COS none are required or passed.

    :param cosuri: the CosURI being pulled or pushed.
    :param assetstore: the dispatched Cosstore.
    :returns: ``{AWS_ACCESS_KEY_ID: ..., AWS_SECRET_ACCESS_KEY: ...}``, or {}
        when mocked.
    :raises ValueError: when a credential cannot be resolved (the message names
        no secret values; the store logs the missing secret names).
    """
    if is_cos_mocked():
        return {}
    credentials = assetstore.resolve_credentials(cosuri)
    if credentials is None:
        raise ValueError(
            f"no S3/COS credentials for {cosuri}: define the store's "
            "cos_access_key_id_secret_name / cos_secret_access_key_secret_name "
            "secrets in the space"
        )
    return {
        AWS_ACCESS_KEY_ID_ENV: credentials[0],
        AWS_SECRET_ACCESS_KEY_ENV: credentials[1],
    }


def resolve_step_uri(store_config: Any, default: str) -> str:
    """Return the store config's ``step_uri`` override, else ``default``.

    :param store_config: a StoreLoad/StorePush config object (may be None).
    :param default: the builtin step URI to use when no override is given.
    :returns: the step URI to queue.
    """
    config = getattr(store_config, "config", None)
    if isinstance(config, dict) and config.get("step_uri"):
        return config["step_uri"]
    return default


def store_nonneg_int(store_config: Any, key: str) -> Optional[int]:
    """Read integer setting ``key`` from a StoreLoad config, validated.

    Used for the s3pull cache settings (``cache_max_age_days``,
    ``lease_max_age_days``).

    :param store_config: the store's pull config object (may be None).
    :param key: the setting name.
    :returns: the value, or None when unset so the s3pull step's default applies.
    :raises ValueError: if the value is not a non-negative integer.
    """
    config = getattr(store_config, "config", None)
    if not isinstance(config, dict) or config.get(key) is None:
        return None
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer, got {value!r}")
    return value


def build_s3_step_config(
    kind: str,
    cosuri: Any,
    assetstore: Any,
    local_path: str,
    binding_id: str = "",
    cache_max_age_days: Optional[int] = None,
    lease_max_age_days: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the config dict for an s3pull or s3push step.

    :param kind: ``"s3pull"`` or ``"s3push"`` (selects the config key).
    :param cosuri: the CosURI being pulled or pushed.
    :param assetstore: the dispatched Cosstore.
    :param local_path: the cluster path pulled into / pushed from.
    :param binding_id: the output's binding id (push only); echoed back with
        the artifact ``uri`` in the step's ``Pushed S3 URI:`` line, which the
        monitor turns into the ARTIFACT_PUSHED_EVENT marking the artifact done.
    :param cache_max_age_days: s3pull only: evict cache entries downloaded more
        than this many days ago (0 disables); None keeps the step's default.
    :param lease_max_age_days: s3pull only: treat reader leases older than this
        many days as dead (0 = never); None keeps the step's default.
    :returns: ``{"<kind>_config": {...}, "launcher_config": {"envs": {...}}}``.
    :raises ValueError: for an unknown kind, a bad endpoint or missing credentials.
    """
    if kind not in ("s3pull", "s3push"):
        raise ValueError(f"unknown s3 step kind: {kind}")
    settings = resolve_cos_store_settings(cosuri, assetstore)
    step_config: Dict[str, Any] = {
        "uri": cosuri.get_uristr(cosuri),
        "s3_uri": settings.s3_uri,
        "local_path": local_path,
        "endpoint_url": settings.endpoint_url,
        "region": settings.region,
    }
    if kind == "s3push":
        step_config["binding_id"] = binding_id
    if kind == "s3pull":
        for key, value in (
            ("cache_max_age_days", cache_max_age_days),
            ("lease_max_age_days", lease_max_age_days),
        ):
            if value is not None:
                step_config[key] = value
    return {
        f"{kind}_config": step_config,
        "launcher_config": {"envs": cos_launcher_envs(cosuri, assetstore)},
    }
