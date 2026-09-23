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

"""Unit tests for Target.push_assets (inline-hfpush output resolution, #390)."""

from unittest.mock import MagicMock

import pytest

from gbserver.build.target import Target
from gbserver.environment.io.descriptors import HfOutputIO
from gbserver.types.buildconfig import BuildTargetConfig, BuildTargetOutputConfig
from gbserver.types.environmentconfig import AssetStoreEnvironmentConfig, StorePush

# Three-slash HF URI: HfURI.parse REJECTS the 2-slash form as malformed
# (Ruling R3). Real configs use e.g. hf:///ibm-granite/granite-4.0-h-350m.
HF_OUTPUT_URI = "hf:///ibm-granite/granite-4.0-h-350m"


def _make_target(outputs, storeenv, assetstore):
    """Build a real Target wired with the given outputs and a mocked env.

    push_assets only consumes self.config.outputs and self.environment, so we
    construct the Target directly (bypassing assimilate, which needs a live
    environment_uri) and attach a MagicMock environment whose
    _get_storeconfig / resolve_inline_hfpush mirror the real signatures.
    """
    target = object.__new__(Target)
    target.name = "t"
    target.config = BuildTargetConfig(
        environment_uri="env:///skypilot",
        outputs=outputs,
        steps=[],
    )

    environment = MagicMock()
    environment._get_storeconfig.return_value = (assetstore, storeenv)

    def _resolve(uri, storepush_config, assetstore, output_config, binding_id):
        return HfOutputIO(
            repo="ibm-granite/granite-4.0-h-350m",
            uri=str(uri),
            binding_id=binding_id,
        )

    environment.resolve_inline_hfpush.side_effect = _resolve
    target.environment = environment
    return target


@pytest.fixture
def inline_push_target():
    """A Target with one hf:// output whose push config is inline: true."""
    outputs = {"model_out": BuildTargetOutputConfig(uri=HF_OUTPUT_URI)}
    storeenv = AssetStoreEnvironmentConfig(
        store_uri="hf:///",
        push=[StorePush(config={"inline": True})],
    )
    assetstore = MagicMock()
    assetstore.type = "hfstore"
    return _make_target(outputs, storeenv, assetstore)


@pytest.fixture
def byo_push_target():
    """A Target with one hf:// output whose push config is NOT inline."""
    outputs = {"model_out": BuildTargetOutputConfig(uri=HF_OUTPUT_URI)}
    storeenv = AssetStoreEnvironmentConfig(
        store_uri="hf:///",
        push=[StorePush(config={"inline": False})],
    )
    assetstore = MagicMock()
    assetstore.type = "hfstore"
    return _make_target(outputs, storeenv, assetstore)


def test_push_assets_resolves_inline_hf_outputs(inline_push_target):
    resolved = inline_push_target.push_assets()
    assert "model_out" in resolved
    io = resolved["model_out"]["_hfpush"]
    assert isinstance(io, HfOutputIO)
    assert io.binding_id == "model_out"


@pytest.fixture
def glob_inline_push_target():
    """An inline hf output whose binding_id (key) is a shell glob.

    buildrun.py matches output configs to runtime artifact ids via
    fnmatch, so a key MAY be a glob. Inline hfpush cannot support that:
    the epilogue matches the GB_ARTIFACT_ID:<id> marker literally.
    """
    outputs = {"model-*": BuildTargetOutputConfig(uri=HF_OUTPUT_URI)}
    storeenv = AssetStoreEnvironmentConfig(
        store_uri="hf:///",
        push=[StorePush(config={"inline": True})],
    )
    assetstore = MagicMock()
    assetstore.type = "hfstore"
    return _make_target(outputs, storeenv, assetstore)


def test_push_assets_rejects_glob_binding_id(glob_inline_push_target):
    # An inline-push output key with a glob metachar must fail early at
    # resolve time, not silently produce an epilogue marker that can never
    # match the concrete runtime GB_ARTIFACT_ID:<id>.
    with pytest.raises(ValueError, match="inline hfpush requires a literal"):
        glob_inline_push_target.push_assets()


def test_push_assets_skips_non_inline_outputs(byo_push_target):
    # Outputs without inline: true resolve to nothing (separate-step path).
    assert byo_push_target.push_assets() == {}


def test_push_assets_skips_non_hf_store():
    """A non-hf store output resolves to nothing even if it declares push."""
    outputs = {"cos_out": BuildTargetOutputConfig(uri="cos://bucket/key")}
    storeenv = AssetStoreEnvironmentConfig(
        store_uri="cos://",
        push=[StorePush(config={"inline": True})],
    )
    assetstore = MagicMock()
    assetstore.type = "cosstore"
    target = _make_target(outputs, storeenv, assetstore)
    assert target.push_assets() == {}


def test_push_assets_skips_unrecognized_uri():
    """A URI with no matching store (_get_storeconfig -> (None, None))."""
    outputs = {"env_out": BuildTargetOutputConfig(uri="env:///out")}
    target = _make_target(outputs, storeenv=None, assetstore=None)
    assert target.push_assets() == {}


def test_push_assets_empty_when_no_outputs():
    target = object.__new__(Target)
    target.config = BuildTargetConfig(environment_uri="env:///skypilot", steps=[])
    target.environment = MagicMock()
    assert target.push_assets() == {}
