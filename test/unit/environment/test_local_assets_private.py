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

"""``private`` must reach HuggingFace on the bash/docker inline push path.

These environments have no hfpush step directory, so they call
``push_asset_hfstore`` directly instead of rendering a worker template. The
surface flag is ``public`` (default false); it is flipped to the internal
``private`` at the resolver boundary. ``HfApi.create_repo``'s own default is
PUBLIC — so a dropped/mis-resolved value must fail closed to private rather than
publishing. Every test below feeds a ``public`` config and asserts on the
``private`` kwarg that actually reaches ``HfURI.push``.

The tests exercise the real ``push_asset_hfstore`` (only ``HfURI.push`` and the
resource-group lookup are mocked); mocking ``push_asset_hfstore`` itself, as the
bash/docker forwarding tests do, cannot observe this.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gbcommon.uri.hf import HfURI
from gbcommon.uri.uri import URI
from gbserver.asset.hfstore import Hfstore
from gbserver.environment.local_assets import push_asset_hfstore


@pytest.fixture
def src_dir(tmp_path):
    """A non-empty source dir (HfURI.push rejects empty ones)."""
    d = tmp_path / "model"
    d.mkdir()
    (d / "weights.bin").write_text("x")
    return d


def _hfuri():
    """A real HfURI with only ``push`` stubbed.

    It must be a genuine HfURI: ``push_asset_hfstore`` routes anything else
    through ``URI.get_uri()``, which templates the value.
    """
    hfuri = URI.get_uri("hf://huggingface.co/some-org/some-model")
    assert isinstance(hfuri, HfURI)
    hfuri.push = MagicMock()
    return hfuri


def _push_private_kwarg(hfuri):
    """Return the ``private`` kwarg passed to hfuri.push()."""
    hfuri.push.assert_called_once()
    return hfuri.push.call_args.kwargs.get("private")


def _run(hfuri, src, assetstore, output_config=None, storepush_config=None):
    # URI/HfURI are imported inside push_asset_hfstore, so patch them at source.
    with patch(
        "gbcommon.uri.uri.URI.get_space_config",
        return_value={"space": {"name": "public"}},
    ):
        push_asset_hfstore(
            src=src,
            binding_id="out",
            uri=hfuri,
            assetstore=assetstore,
            run_metadata=SimpleNamespace(build_id="b1", target_name="t1"),
            storepush_config=storepush_config,
            output_config=output_config,
        )
    return _push_private_kwarg(hfuri)


def _output_config(hf_cfg):
    """A BuildTargetOutputConfig double carrying an ``hf`` push config."""
    output_config = MagicMock()
    output_config.public = None  # no top-level public (real output defaults to None)
    output_config.store_push.config = {"hf": hf_cfg}
    return output_config


@pytest.fixture
def hfstore():
    """An Hfstore whose resource-group resolution is stubbed out."""
    store = MagicMock(spec=Hfstore)
    store.get_secrets.return_value = {}
    store.get_enterprise_organizations.return_value = []  # non-Enterprise
    store.resolve_token.return_value = "tok"
    return store


def test_no_config_defaults_to_private(src_dir, hfstore):
    """(1) With no push config at all, the repo must still be created private."""
    hfuri = _hfuri()
    assert _run(hfuri, src_dir, hfstore) is True


def test_explicit_public_false_is_not_discarded(src_dir, hfstore):
    """(3) A redundant explicit ``public: false`` must survive to HF as private."""
    hfuri = _hfuri()
    assert _run(hfuri, src_dir, hfstore, _output_config({"public": False})) is True


def test_explicit_public_true_is_honored(src_dir, hfstore):
    """(2) Public is opt-in: only an explicit ``public: true`` gets it."""
    hfuri = _hfuri()
    assert _run(hfuri, src_dir, hfstore, _output_config({"public": True})) is False


def test_yaml_null_public_does_not_publish(src_dir, hfstore):
    """A bare ``public:`` (yaml null) must not be read as "public".

    ``None`` is *present* as a key; the flip treats it (and any non-truthy value)
    as private, so a bare ``public:`` keeps the repo private.
    """
    hfuri = _hfuri()
    assert _run(hfuri, src_dir, hfstore, _output_config({"public": None})) is True


def test_non_hfstore_assetstore_still_pushes_private(src_dir):
    """The non-Hfstore branch skips the resolver, so it needs its own default.

    ``private`` is initialized before the branch precisely so this path cannot
    fall through to HuggingFace's public default.
    """
    hfuri = _hfuri()
    plain_store = MagicMock()  # not an Hfstore -> other resolution branch
    plain_store.get_secrets.return_value = {}
    plain_store.resolve_token.return_value = "tok"
    with patch(
        "gbserver.environment.local_assets.resolve_space_resource_group_id",
        return_value=None,
    ):
        assert _run(hfuri, src_dir, plain_store) is True


def test_resolver_failure_still_pushes_private(src_dir, hfstore):
    """A transient resolver failure is best-effort — but must not go public.

    The ``except Exception`` path logs and continues to the push, so ``private``
    has to already hold a safe value when it does.
    """
    hfuri = _hfuri()
    with patch(
        "gbserver.environment.local_assets.resolve_hfpush_resource_group_id",
        side_effect=RuntimeError("HF API down"),
    ):
        assert _run(hfuri, src_dir, hfstore) is True


def test_quoted_public_true_is_honored(src_dir, hfstore):
    """``public: "true"`` (quoted in yaml) must mean public, not a truthy string.

    A bare ``bool("true")`` would also be ``True``, but the fail-closed flip only
    honors *recognized* truthy tokens, folded case- and whitespace-insensitively.
    """
    hfuri = _hfuri()
    assert _run(hfuri, src_dir, hfstore, _output_config({"public": "true"})) is False


@pytest.mark.parametrize("value", ["yes", "on", "1", "True", " true "])
def test_quoted_truthy_forms_are_honored(src_dir, hfstore, value):
    """The whole truthy token set works, case- and whitespace-insensitively."""
    hfuri = _hfuri()
    assert _run(hfuri, src_dir, hfstore, _output_config({"public": value})) is False


def test_unparseable_public_falls_back_to_private(src_dir, hfstore):
    """A typo must fail *safe*: unrecognized ``public`` means private, not public."""
    hfuri = _hfuri()
    assert _run(hfuri, src_dir, hfstore, _output_config({"public": "treu"})) is True


def test_output_level_public_overrides_environment_level(src_dir, hfstore):
    """build.yaml outranks environment.yaml, so a per-output opt-in wins.

    The intended usage: the environment keeps everything private, and a single
    output selectively publishes.
    """
    hfuri = _hfuri()
    storepush_config = MagicMock()
    storepush_config.config = {"hf": {"public": False}}
    assert (
        _run(
            hfuri,
            src_dir,
            hfstore,
            output_config=_output_config({"public": True}),
            storepush_config=storepush_config,
        )
        is False
    )


def test_non_hfstore_honors_explicit_public_true(src_dir):
    """The non-Hfstore branch must honor an explicit ``public: true`` too.

    It cannot classify the org (no Enterprise list without an ``Hfstore``), but
    the public/private flag is store-independent: hardcoding ``private=True`` here
    would silently ignore a config the Hfstore branch obeys.
    """
    hfuri = _hfuri()
    plain_store = MagicMock()  # not an Hfstore
    plain_store.get_secrets.return_value = {}
    plain_store.resolve_token.return_value = "tok"
    with patch(
        "gbserver.environment.local_assets.resolve_space_resource_group_id",
        return_value=None,
    ):
        assert (
            _run(hfuri, src_dir, plain_store, _output_config({"public": True})) is False
        )
