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

"""Build tests whose ``s3://`` inputs/outputs use the local SeaweedFS store.

The store is the ``gb-s3`` container from ``make s3-setup``
(scripts/s3/setup-s3.sh), routed by the ``s3-local`` assetstore
(``s3://local/``, an alias for its ``cos_bucket_name`` bucket) that the local
SLURM environment lists. A fixture's build.yaml names its objects under a
per-run prefix::

  s3://local/gbtest/skypilot-s3/__GBTEST_S3_RUN__/...

:class:`S3BuildTestMixin` fills the placeholder, seeds the input object, runs
the build, asserts the pushed output round-trips its content, and deletes the
run's prefix, pass or fail.

These tests always run for real. The store is local infrastructure, like the
SLURM cluster the build runs on, so there is no external service to stand in for
(unlike hf://). The mixin pins ``GBTEST_MOCK_COS=false`` for the run, so a
session-wide mock setting can't turn the round trip into a no-op. Mocking stays
available for tests against non-local S3 endpoints.

The test talks to the store exactly as gbserver does: endpoint
(``cos_gbserver_endpoint``), bucket and credentials all come from the s3-local
store.yaml, and the credentials are the secrets it names
(``S3_LOCAL_ACCESS_KEY_ID`` / ``S3_LOCAL_SECRET_ACCESS_KEY``) resolved by
``Cosstore.resolve_credentials`` -- the local space's secret manager first, then
the same-named env var. ``make s3-setup`` registers them in the local space's
secrets, so both the test and the build find them; nothing is hard-coded here.
"""

import logging
import os
import tempfile
import urllib.request
import uuid
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Self

from gbcommon.types.testing import ENV_VAR_GBTEST_MOCK_COS
from gbcommon.uri.cos import CosCheckTarget
from gbcommon.uri.uri import URI
from libgbtest.buildrunner.buildtest import AbstractYamlBuildRunnerTest

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[2]
STORE_YAML = _REPO / "configurations/assets/assetstores/s3-local/store.yaml"
SPACE_YAML = _REPO / "configurations/spaces/local/space.yaml"

RUN_PLACEHOLDER = "__GBTEST_S3_RUN__"
STORE_URI = "s3://local/"
RUN_ROOT = "gbtest/skypilot-s3"
INPUT_KEY = "input/data.txt"
OUTPUT_KEY = "output/out.txt"
_HEALTH_TIMEOUT_SECONDS = 3


def _space_secrets() -> Dict[str, str]:
    """The local standalone space's secrets, via its configured secret manager.

    :returns: the secret bag ({} when the manager has none or can't load).
    """
    from gbserver.spacesecretmanager.spacesecretmanager import (  # pylint: disable=import-outside-toplevel
        SpaceSecretManager,
    )
    from gbserver.types.spaceconfig import SpaceConfig  # pylint: disable=import-outside-toplevel

    try:
        space = SpaceConfig.from_yaml(SPACE_YAML)
        SpaceSecretManager.load_spacesecretmanagers()
        manager = SpaceSecretManager.get_spacesecretmanager(
            secret_manager_type=space.secret_manager.type,
            uri=SPACE_YAML.parent.as_uri(),
            space_name=space.name,
            **(space.secret_manager.config or {}),
        )
        return manager.get_secrets() or {}
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("could not load the local space secrets: %s", e)
        return {}


@lru_cache(maxsize=1)
def local_s3_target() -> Optional[CosCheckTarget]:
    """Resolve the s3-local store's endpoint, bucket and credentials.

    Uses the same Cosstore code gbserver uses (``get_check_target``), so the
    test and the build agree on where and as whom they reach the store.

    :returns: the target (key = '' for the bucket root), or None when the
        store's credential secrets are not defined.
    """
    from gbserver.asset.cosstore import Cosstore  # pylint: disable=import-outside-toplevel
    from gbserver.types.assetstoreconfig import AssetStoreConfig  # pylint: disable=import-outside-toplevel

    store = Cosstore(AssetStoreConfig.from_yaml(STORE_YAML), secrets=_space_secrets())
    return store.get_check_target(URI.get_uri(STORE_URI))


def local_s3_reachable() -> bool:
    """True if the store's gbserver endpoint answers its health check."""
    target = local_s3_target()
    if target is None or not target.endpoint_url:
        return False
    try:
        with urllib.request.urlopen(  # noqa: S310 - endpoint from the store.yaml
            f"{target.endpoint_url}/healthz", timeout=_HEALTH_TIMEOUT_SECONDS
        ) as resp:
            return resp.status == 200
    except OSError:
        return False


def local_s3_unavailable_reason() -> Optional[str]:
    """Why an s3 build test against the local store can't run, or None if it can."""
    if local_s3_target() is None:
        return (
            "s3-local store credentials not found (run: make s3-setup, which "
            "registers S3_LOCAL_ACCESS_KEY_ID / S3_LOCAL_SECRET_ACCESS_KEY)"
        )
    if not local_s3_reachable():
        return "local S3 store not reachable (run: make s3-setup)"
    return None


@contextmanager
def cos_mocks_disabled() -> Iterator[None]:
    """Pin GBTEST_MOCK_COS=false for the block, then restore the prior value.

    The env var is read at call time in-process and forwarded to the step jobs,
    so this makes the whole build, existence check included, use real S3.
    """
    prior = os.environ.get(ENV_VAR_GBTEST_MOCK_COS)
    os.environ[ENV_VAR_GBTEST_MOCK_COS] = "false"
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(ENV_VAR_GBTEST_MOCK_COS, None)
        else:
            os.environ[ENV_VAR_GBTEST_MOCK_COS] = prior


def local_s3_client(target: CosCheckTarget) -> Any:
    """A boto3 client for the store described by ``target``."""
    import boto3  # pylint: disable=import-outside-toplevel

    return boto3.client(
        "s3",
        endpoint_url=target.endpoint_url or None,
        region_name=target.region or None,
        aws_access_key_id=target.access_key_id,
        aws_secret_access_key=target.secret_access_key,
    )


def delete_prefix(client: Any, bucket: str, prefix: str) -> None:
    """Delete every object under ``prefix`` in ``bucket``."""
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if keys:
            client.delete_objects(Bucket=bucket, Delete={"Objects": keys})


class S3BuildTestMixin(AbstractYamlBuildRunnerTest):
    """Run a YAML build test against the s3-local store, always for real.

    Put it before the test's base class. The fixture's build.yaml must name its
    objects under
    ``s3://local/gbtest/skypilot-s3/__GBTEST_S3_RUN__/``, reading
    :data:`INPUT_KEY` and writing :data:`OUTPUT_KEY` (relative to that prefix).
    """

    def _render(self: Self, dest_dir: Path) -> tuple[Path, str]:
        """Write the fixture's build.yaml with a fresh run id into ``dest_dir``.

        :returns: the rendered build.yaml and the run's key prefix in the bucket.
        """
        run = uuid.uuid4().hex[:8]
        src = self._get_yaml_spec_dir() / "build.yaml"
        dest = dest_dir / src.name
        dest.write_text(src.read_text(encoding="utf-8").replace(RUN_PLACEHOLDER, run))
        return dest, f"{RUN_ROOT}/{run}/"

    def _run_yaml_spec(self: Self, test_key: str, test_cancel: bool) -> None:
        with tempfile.TemporaryDirectory(prefix="gbtest-s3-") as tmp:
            build_yaml, prefix = self._render(Path(tmp))
            self._build_yaml_override = str(build_yaml)
            try:
                with cos_mocks_disabled():
                    self._run_real(test_key, test_cancel, prefix)
            finally:
                del self._build_yaml_override

    def _run_real(self: Self, test_key: str, test_cancel: bool, prefix: str) -> None:
        """Seed the input, run the build, verify the output, clean up the prefix."""
        target = local_s3_target()
        assert target is not None, "s3-local store credentials are not defined"
        client = local_s3_client(target)
        payload = f"gbtest payload {prefix}".encode()
        client.put_object(Bucket=target.bucket, Key=prefix + INPUT_KEY, Body=payload)
        try:
            super()._run_yaml_spec(test_key, test_cancel)
            if not test_cancel:
                obj = client.get_object(Bucket=target.bucket, Key=prefix + OUTPUT_KEY)
                assert obj["Body"].read() == payload, "s3:// output did not round-trip"
        finally:
            try:
                delete_prefix(client, target.bucket, prefix)
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning("could not delete test prefix %s: %s", prefix, e)
