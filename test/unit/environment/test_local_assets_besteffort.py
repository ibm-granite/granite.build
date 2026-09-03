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

"""Resource-group resolution is best-effort on the bash/docker inline push.

Two failures reach the same ``except`` on this path and must be treated in
opposite ways:

* A resolution **miss** — ``resolve_resource_group_id_for_org`` raises a plain
  ``ValueError`` when it cannot map a group name to an id, which is the ordinary
  outcome for a standalone user whose token is not org-admin (the HF
  ``/resource-groups`` endpoint 403s and the miss surfaces as "Could not resolve
  resource group id"). The push must proceed without a group.
* A **config error** — ``HfPushConfigError`` for a group pinned on a
  non-Enterprise org, or a pin contradicting ``use_resource_group: false``. The
  push must abort rather than silently ignore what was asked.

Because ``HfPushConfigError`` subclasses ``ValueError``, a bare
``except ValueError: raise`` cannot tell them apart and aborts the miss too —
regressing the documented best-effort behavior. These tests pin the distinction.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gbcommon.uri.hf import HfURI
from gbcommon.uri.uri import URI
from gbserver.asset.hfstore import Hfstore
from gbserver.environment.local_assets import push_asset_hfstore
from gbserver.spaces.resource_group import HfPushConfigError


@pytest.fixture
def src_dir(tmp_path):
    """A non-empty source dir (HfURI.push rejects empty ones)."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "weights.bin").write_text("x")
    return d


@pytest.fixture
def hfuri():
    """A real HfURI on an Enterprise org, with only ``push`` stubbed."""
    uri = URI.get_uri("hf://huggingface.co/ibm-research/some-model")
    assert isinstance(uri, HfURI)
    uri.push = MagicMock()
    return uri


@pytest.fixture
def enterprise_store():
    """An Hfstore that treats ``ibm-research`` as Enterprise."""
    store = MagicMock(spec=Hfstore)
    store.get_secrets.return_value = {}
    store.get_enterprise_organizations.return_value = ["ibm-research"]
    store.resolve_token.return_value = "tok"
    return store


def _run(hfuri, src, assetstore, output_config=None):
    with patch(
        "gbcommon.uri.uri.URI.get_space_config",
        return_value={"space": {"name": "public"}},
    ):
        return push_asset_hfstore(
            src=src,
            binding_id="out",
            uri=hfuri,
            assetstore=assetstore,
            run_metadata=SimpleNamespace(build_id="b1", target_name="t1"),
            output_config=output_config,
        )


def test_resolution_miss_still_pushes(src_dir, hfuri, enterprise_store):
    """A non-admin token cannot resolve the group; the push must still happen.

    This is the standalone default: pushing to an Enterprise org with a personal
    token. ``create_repo(exist_ok=True)`` succeeds for an existing repo, so
    aborting here would block a push that works.
    """
    with patch(
        "gbserver.environment.local_assets.resolve_hfpush_resource_group_id",
        side_effect=ValueError(
            "Could not resolve resource group id for 'gbspace-public' "
            "in organization 'ibm-research'"
        ),
    ):
        _run(src=src_dir, hfuri=hfuri, assetstore=enterprise_store)

    hfuri.push.assert_called_once()
    assert hfuri.push.call_args.kwargs.get("resource_group_id") is None
    # Still private: the best-effort path must not fall through to HF's default.
    assert hfuri.push.call_args.kwargs.get("private") is True


def test_config_error_aborts_the_push(src_dir, hfuri, enterprise_store):
    """A pinned group that cannot be honored must abort, not push silently."""
    with patch(
        "gbserver.environment.local_assets.resolve_hfpush_resource_group_id",
        side_effect=HfPushConfigError("resource group pinned for a non-Enterprise org"),
    ):
        with pytest.raises(HfPushConfigError):
            _run(src=src_dir, hfuri=hfuri, assetstore=enterprise_store)

    hfuri.push.assert_not_called()


def test_config_error_is_a_valueerror(src_dir, hfuri, enterprise_store):
    """``HfPushConfigError`` stays catchable as ``ValueError`` for old callers."""
    assert issubclass(HfPushConfigError, ValueError)
    with patch(
        "gbserver.environment.local_assets.resolve_hfpush_resource_group_id",
        side_effect=HfPushConfigError("nope"),
    ):
        with pytest.raises(ValueError):
            _run(src=src_dir, hfuri=hfuri, assetstore=enterprise_store)


def test_miss_from_the_real_resolver_chain_still_pushes(
    src_dir, hfuri, enterprise_store
):
    """End-to-end version: let the genuine resolver raise the miss.

    Rather than stubbing the helper, this drives the real chain — a 403 swallowed
    by ``_resolve_resource_group_id`` returning ``None``, which makes
    ``resolve_resource_group_id_for_org`` raise. Guards against the abort
    returning via a different route than the one stubbed above.
    """
    with (
        patch.object(HfURI, "_resolve_resource_group_id", return_value=None),
        patch("gbcommon.uri.hf.HfApi"),
        patch("gbcommon.uri.hf.is_hf_mocked", return_value=False),
        patch("gbserver.spaces.resource_group.get_admin_storage") as get_storage,
    ):
        get_storage.return_value.space_storage.get_by_name.return_value = None
        _run(src=src_dir, hfuri=hfuri, assetstore=enterprise_store)

    hfuri.push.assert_called_once()
    assert hfuri.push.call_args.kwargs.get("resource_group_id") is None
