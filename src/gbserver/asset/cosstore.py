#!/usr/bin/env python3

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

"""Access assets in COS buckets."""

import ipaddress
import os
from typing import Dict, Optional, Self, Tuple, Union
from urllib.parse import urlparse

from gbcommon.uri.cos import CosCheckTarget, CosURI, set_cos_check_resolver
from gbcommon.uri.uri import URI
from gbserver.asset.assetstore import Assetstore
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)


def _is_private_http_host(host: str) -> bool:
    """Return True if ``host`` may be reached over plain HTTP.

    Plain HTTP is only tolerated for loopback hosts and single-label hostnames
    (e.g. a ``gb-s3`` container on a private container network, as used by the
    local SeaweedFS fixture from ``make s3-setup``). Any routable host must use
    TLS.

    :param host: the endpoint's hostname (no port).
    :returns: True for loopback addresses, ``localhost`` and dot-free names.
    """
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return "." not in host


def normalize_s3_endpoint(endpoint: Optional[str]) -> str:
    """Normalize a store endpoint (``cos_endpoint``) into an S3 endpoint URL.

    A bare host (the store.yaml convention, e.g. ``s3.amazonaws.com``) gets an
    ``https://`` scheme. ``https://`` URLs pass through. ``http://`` is accepted
    only for loopback / single-label private hosts (see
    :func:`_is_private_http_host`).

    :param endpoint: the configured endpoint (may be empty or None).
    :returns: the endpoint URL, or '' when none is configured.
    :raises ValueError: for an unsupported scheme or a plain-HTTP routable host.
    """
    endpoint = (endpoint or "").strip()
    if not endpoint:
        return ""
    if "://" not in endpoint:
        return f"https://{endpoint}"
    parsed = urlparse(endpoint)
    if parsed.scheme == "https":
        return endpoint
    if parsed.scheme == "http" and _is_private_http_host(parsed.hostname or ""):
        return endpoint
    raise ValueError(
        f"unsupported S3 endpoint '{endpoint}': use https:// (plain http:// is "
        "only allowed for loopback or single-label private hosts)"
    )


class Cosstore(Assetstore):
    """A class for storing and accessing cos asset"""

    def __init__(self: Self, uri: Union[URI, str], **kwargs) -> None:
        super().__init__(uri, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def get_supported_uri_classes(cls):
        """Get the supported URI classes."""
        return [CosURI]

    def _get_config_key(self: Self, key: str, default: str) -> str:
        if (
            self.config
            and isinstance(self.config.config, dict)
            and key in self.config.config
        ):
            return self.config.config[key]
        return default

    def get_relpath(self: Self, uri: URI) -> str:
        if not isinstance(uri, CosURI):
            raise ValueError(f"Expected a CosURI, got {type(uri)}")
        cos_md = self.get_metadata(uri)["config"]
        bucket_name = cos_md.get("cos_bucket_name") or uri.get_uri_netloc()
        full_uri = uri.get_metadata()["bucket_path"]
        if not full_uri.startswith(f"{bucket_name}/"):
            parts = full_uri.split("/", 1)
            rel_path = parts[1] if len(parts) > 1 else ""
        else:
            rel_path = full_uri[len(f"{bucket_name}/") :]
        logger.debug(f"Computed COS rel_path: {rel_path}")
        return rel_path

    def get_metadata(self, uri: URI) -> Dict:
        if not isinstance(uri, CosURI):
            raise ValueError(f"Expected a CosURI, got {type(uri)}: {uri}")
        self.cos_secret_access_key_secret_name = self._get_config_key(
            "cos_secret_access_key_secret_name", "COS_SECRET_ACCESS_KEY"
        )
        self.cos_access_key_id_secret_name = self._get_config_key(
            "cos_access_key_id_secret_name", "COS_ACCESS_KEY_ID"
        )
        self.cos_bucket_name = self._get_config_key("cos_bucket_name", "")
        self.cos_endpoint = self._get_config_key(
            "cos_endpoint", "s3.us-east.cloud-object-storage.appdomain.cloud"
        )
        self.cos_region = self._get_config_key("cos_region", "us-east")
        # Endpoint gbserver itself uses (e.g. CosURI.exists); defaults to
        # cos_endpoint. Set it when the cluster-side endpoint is not reachable
        # from gbserver (local SeaweedFS: gb-s3:9000 in-network, localhost:9000
        # from the host).
        cos_gbserver_endpoint = self._get_config_key("cos_gbserver_endpoint", "")
        logger.info(
            "Using cos secret access key with config key %s from secrets",
            self.cos_secret_access_key_secret_name,
        )
        logger.info(
            "Using cos access key id with config key %s from secrets",
            self.cos_access_key_id_secret_name,
        )
        logger.info("Using cos bucket name %s", self.cos_bucket_name)
        logger.info("Using COS endpoint %s", self.cos_endpoint)
        logger.info("Using COS region %s", self.cos_region)
        return {
            "config": {
                "cos_secret_access_key_secret_name": self.cos_secret_access_key_secret_name,
                "cos_access_key_id_secret_name": self.cos_access_key_id_secret_name,
                "cos_bucket_name": self.cos_bucket_name,
                "cos_endpoint": self.cos_endpoint,
                "cos_region": self.cos_region,
                "cos_gbserver_endpoint": cos_gbserver_endpoint,
            },
        }

    def get_bucket_and_key(self: Self, uri: URI) -> Tuple[str, str]:
        """Return the real bucket and key/prefix a CosURI addresses.

        The store's ``cos_bucket_name`` (when set) replaces the URI's bucket, as
        on K8s/LSF.

        :param uri: the CosURI.
        :returns: ``(bucket, key)``; key is '' for a bucket-only URI.
        :raises ValueError: if ``uri`` is not a CosURI.
        """
        if not isinstance(uri, CosURI):
            raise ValueError(f"Expected a CosURI, got {type(uri)}: {uri}")
        bucket_name = self.get_metadata(uri)["config"].get("cos_bucket_name", "")
        return bucket_name or uri.get_uri_netloc(), self.get_relpath(uri)

    def _resolve_secret(self: Self, name: str) -> Optional[str]:
        """Resolve one secret exactly as ``Hfstore.resolve_token`` does.

        The store's secrets (the space secret bag) win; otherwise the env var of
        the same name. Blank values count as unset.

        :param name: the secret name from store.yaml.
        :returns: the value, or None when unset/blank.
        """
        if self.secrets and name in self.secrets:
            value = self.secrets[name] or None
        else:
            value = os.getenv(name, None)
        if value is not None and value.strip() == "":
            value = None
        return value

    def resolve_credentials(self: Self, uri: URI) -> Optional[Tuple[str, str]]:
        """Resolve the access key id and secret for ``uri``'s store.

        The secret *names* come from store.yaml (``cos_access_key_id_secret_name``
        / ``cos_secret_access_key_secret_name``, default ``COS_ACCESS_KEY_ID`` /
        ``COS_SECRET_ACCESS_KEY``), as ``token_secretname`` does for hf; values
        are resolved by :meth:`_resolve_secret`.

        :param uri: the CosURI.
        :returns: ``(access_key_id, secret_access_key)``, or None (logged,
            naming the missing secrets but never revealing values).
        """
        config = self.get_metadata(uri)["config"]
        names = (
            config["cos_access_key_id_secret_name"],
            config["cos_secret_access_key_secret_name"],
        )
        values = [self._resolve_secret(name) for name in names]
        missing = [name for name, value in zip(names, values) if value is None]
        if missing:
            logger.warning(
                "No S3/COS credentials for %s: missing secret(s) %s", uri, missing
            )
            return None
        return values[0], values[1]  # type: ignore[return-value]

    def get_check_target(self: Self, uri: URI) -> Optional[CosCheckTarget]:
        """Build the gbserver-side existence-check target for ``uri``.

        :param uri: the CosURI.
        :returns: the target, or None (logged) when a credential is missing.
        :raises ValueError: if the configured endpoint is not acceptable.
        """
        credentials = self.resolve_credentials(uri)
        if credentials is None:
            return None
        config = self.get_metadata(uri)["config"]
        bucket, key = self.get_bucket_and_key(uri)
        endpoint = config.get("cos_gbserver_endpoint") or config.get("cos_endpoint")
        return CosCheckTarget(
            bucket=bucket,
            key=key,
            endpoint_url=normalize_s3_endpoint(endpoint),
            region=config.get("cos_region", "") or "",
            access_key_id=credentials[0],
            secret_access_key=credentials[1],
        )


def _resolve_check_target(uri: CosURI) -> Optional[CosCheckTarget]:
    """CosURI.exists() resolver: route ``uri`` to its loaded Cosstore.

    Uses the same longest-prefix match as dispatch over the loaded assetstores.

    :param uri: the CosURI being checked.
    :returns: the check target, or None (logged) if no Cosstore matches.
    """
    from gbserver.asset.asset import Asset  # pylint: disable=import-outside-toplevel

    try:
        store = Asset.get_assetstore(uri=uri)
    except AttributeError:  # no assetstores loaded on this thread yet
        store = None
    if not isinstance(store, Cosstore):
        logger.warning("Cannot check %s exists: no Cosstore is configured for it", uri)
        return None
    return store.get_check_target(uri)


set_cos_check_resolver(_resolve_check_target)
