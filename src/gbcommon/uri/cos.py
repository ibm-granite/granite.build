#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""S3/COS bucket URI"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Self
from urllib.parse import ParseResult

from gbcommon.types.testing import is_cos_mocked
from gbcommon.uri.uri import URI
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

COS_SCHEME = "cos"
S3_SCHEME = "s3"

# Bounded timeouts so a gbserver-side existence check never hangs validation.
_CHECK_CONNECT_TIMEOUT_SECONDS = 10
_CHECK_READ_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class CosCheckTarget:
    """Where and how gbserver checks that a CosURI's object or prefix exists.

    :param bucket: the bucket to check.
    :param key: the object key or prefix ('' = the bucket itself).
    :param endpoint_url: endpoint reachable from gbserver ('' = AWS default).
    :param region: the region ('' = boto3 default).
    :param access_key_id: the access key id.
    :param secret_access_key: the secret access key (excluded from repr).
    """

    bucket: str
    key: str
    endpoint_url: str
    region: str
    access_key_id: str
    secret_access_key: str = field(repr=False)


def _status_code(error: Any) -> int:
    """Return the HTTP status of a botocore ClientError (0 if unknown)."""
    return int(error.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))


def _make_s3_client(target: CosCheckTarget) -> Any:
    """Build a boto3 S3 client for ``target`` (TLS verified, bounded timeouts).

    :param target: the check target.
    :returns: a boto3 S3 client.
    :raises ImportError: if boto3 is not installed.
    """
    import boto3  # pylint: disable=import-outside-toplevel
    from botocore.config import Config  # pylint: disable=import-outside-toplevel

    return boto3.client(
        "s3",
        endpoint_url=target.endpoint_url or None,
        region_name=target.region or None,
        aws_access_key_id=target.access_key_id,
        aws_secret_access_key=target.secret_access_key,
        config=Config(
            connect_timeout=_CHECK_CONNECT_TIMEOUT_SECONDS,
            read_timeout=_CHECK_READ_TIMEOUT_SECONDS,
            retries={"max_attempts": 2},
        ),
    )


def _s3_location_exists(client: Any, bucket: str, key: str) -> bool:
    """Return True if ``key`` is an object, or a non-empty prefix, in ``bucket``.

    An empty key checks the bucket itself. A key that is not an object (404, or
    403 without ListBucket semantics on HEAD) is retried as a prefix.

    :param client: a boto3 S3 client.
    :param bucket: the bucket name.
    :param key: the object key or prefix.
    :returns: whether the object/prefix exists.
    :raises botocore.exceptions.ClientError: for errors other than a missing object.
    """
    from botocore.exceptions import (
        ClientError,  # pylint: disable=import-outside-toplevel
    )

    if not key:
        client.head_bucket(Bucket=bucket)
        return True
    if not key.endswith("/"):
        try:
            client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as e:
            if _status_code(e) not in (403, 404):
                raise
    prefix = key.rstrip("/") + "/"
    resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
    return int(resp.get("KeyCount", 0)) > 0


def check_s3_location(target: CosCheckTarget, uristr: str) -> bool:
    """Check that ``target``'s object or prefix exists, with HfURI.exists' semantics.

    Returns False on any error: a genuine miss logs at DEBUG, while a missing
    boto3 or an auth/network error logs at WARNING/ERROR, since the object may
    exist and the False is then a false negative that must be visible.

    :param target: where and as whom to check (from the URI's own Cosstore).
    :param uristr: the URI, for log messages.
    :returns: whether the object/prefix exists.
    """
    try:
        client = _make_s3_client(target)
        found = _s3_location_exists(client, target.bucket, target.key)
    except ImportError:
        logger.error("Cannot check %s exists: boto3 is not installed", uristr)
        return False
    except Exception as e:  # pylint: disable=broad-exception-caught
        status = _status_code(e) if hasattr(e, "response") else 0
        log = logger.debug if status == 404 else logger.warning
        log("S3 exists check for %s failed (status %s): %s", uristr, status, e)
        return False
    logger.debug("S3 exists check for %s returned %s", uristr, found)
    return found


class CosURI(URI):
    """Defines support for a g.b url for entities stored in COS. The uri encodes the following
        bucket_name - the name of the COS bucket name
        format is as follows:
            <scheme_prefix>://<bucket_name>/<path>

            Example: s3://fm-openshift-training/granite-dot-build/gbxvsh3wc

    Args:
        URI (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(
        self: Self,
        uri: Optional[ParseResult] = None,
        context: Optional[str] = None,
        secrets: Optional[Dict] = None,
        config: Optional[Dict] = None,
        **kwargs: Dict,
    ) -> None:
        super().__init__(uri, context, secrets, **kwargs)
        try:
            bucket_name = self.get_uri_netloc()
            scheme = self.uri.scheme if self.uri else "<unknown>"
            if bucket_name == "":
                raise ValueError(
                    f"The bucket name cannot be empty in a '{scheme}://' URI"
                )
        except Exception as e:
            raise ValueError(f"failed to create from uri: {uri}") from e

    def get_uri_netloc(self) -> str:
        """Get the host/authority/netloc part of the URI."""
        assert self.uri is not None, "self.uri is None"
        return self.uri.netloc

    def get_metadata(self) -> Any:
        full_uri = self.get_uristr(self)
        bucket_name = self.get_uri_netloc()
        bucket_path = full_uri
        for scheme in self.get_supported_schemes():
            prefix = f"{scheme}://"
            if full_uri.startswith(prefix):
                bucket_path = full_uri[len(prefix) :]
                break
        md = {
            "uri": full_uri,
            "bucket_name": bucket_name,
            "bucket_path": bucket_path,
        }
        return md

    @staticmethod
    def get_supported_schemes() -> List[str]:
        """Return supported uri schemes as list"""
        return [COS_SCHEME, S3_SCHEME]

    def exists(self: Self, force: bool = False) -> bool:
        """Check existence without a store: only answerable when mocked.

        A real check needs the endpoint, bucket and credentials of the Cosstore
        the URI routes to, which a bare URI doesn't know -- and guessing it from
        a process-wide registry can pick the wrong store. Real checks therefore go
        through ``Cosstore.uri_exists`` via the environment that will pull the
        input (``Environment.input_exists``). Mirrors HfURI.exists when mocked.

        :param force: unused (kept for the URI interface).
        :returns: True when GBTEST_MOCK_COS is set, else False (logged).
        """
        if is_cos_mocked():
            return True
        logger.warning(
            "Cannot check %s exists without its assetstore; use the environment's "
            "input_exists()",
            self.get_uristr(self),
        )
        return False

    def is_accessible(self) -> bool:
        # TODO: Fix logic
        return False

    def pull(self: Self, dest: Path, force: bool = False) -> bool:
        # TODO: Fix logic
        raise NotImplementedError("CosURI pull is not implemented")

    def delete(self: Self) -> bool:
        # TODO: Fix logic
        raise NotImplementedError("CosURI delete is not implemented")
