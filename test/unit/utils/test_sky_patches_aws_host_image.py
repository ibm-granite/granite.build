#!/usr/bin/env python3

# Copyright Granite.Build Authors
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

"""sky_patches._patch_aws_host_image: `docker.host_image_id` picks the host AMI.

The patched aws.AWS.make_deploy_resources_variables (applied by test/conftest.py)
runs on real sky.Resources. Two calls are replaced: AWS._get_image_id, which turns
the chosen image into an AMI, to record what it was given; and the instance
type's accelerator lookup, which reads SkyPilot's AWS catalog (a download when it
is not cached) and plays no part in choosing the image. So the tests run offline.
"""

import pytest

sky = pytest.importorskip("sky")
from sky import clouds  # noqa: E402  pylint: disable=wrong-import-position
from sky.clouds import aws  # noqa: E402  pylint: disable=wrong-import-position
from sky.utils import (  # noqa: E402  pylint: disable=wrong-import-position
    resources_utils,
)

_HOST = "ami-0123456789abcdef0"
_OWN = "ami-0aaaaaaaaaaaaaaaa"


class _Chosen(Exception):
    """Stops make_deploy_resources_variables once the image is chosen."""


def _chosen_image(monkeypatch, image_id=None, host_image_id=None):
    """The image_id make_deploy_resources_variables hands to _get_image_id."""
    seen = {}

    def record(_cls, chosen, *_args):
        seen["image_id"] = chosen
        raise _Chosen

    monkeypatch.setattr(aws.AWS, "_get_image_id", classmethod(record))
    monkeypatch.setattr(
        aws.AWS,
        "get_accelerators_from_instance_type",
        classmethod(lambda _cls, _instance_type: {"L4": 4}),
    )
    overrides = {"docker": {"host_image_id": host_image_id}} if host_image_id else None
    kwargs = {"image_id": image_id} if image_id else {}
    resources = sky.Resources(
        infra="aws/us-west-2",
        instance_type="g6.12xlarge",
        _cluster_config_overrides=overrides,
        **kwargs,
    )
    with pytest.raises(_Chosen):
        aws.AWS().make_deploy_resources_variables(
            resources,
            resources_utils.ClusterName("c", "c"),
            clouds.Region("us-west-2"),
            [clouds.Zone("us-west-2a")],
            1,
        )
    return seen["image_id"]


def test_a_container_runs_on_the_host_image(monkeypatch):
    chosen = _chosen_image(
        monkeypatch, image_id="docker:ubuntu:22.04", host_image_id=_HOST
    )
    assert chosen == {None: _HOST}


def test_a_bare_host_without_an_image_runs_on_the_host_image(monkeypatch):
    # byoc with no image, a command step on the bare node: the environment can
    # give them the AMI the same way it gives a container step its host.
    assert _chosen_image(monkeypatch, host_image_id=_HOST) == {None: _HOST}


def test_a_bare_host_keeps_its_own_image_id(monkeypatch):
    # An image the step names itself wins over the environment's host AMI.
    chosen = _chosen_image(monkeypatch, image_id=_OWN, host_image_id=_HOST)
    assert chosen == {"us-west-2": _OWN}


def test_without_a_host_image_skypilot_picks_its_default(monkeypatch):
    assert _chosen_image(monkeypatch) is None
    container = _chosen_image(monkeypatch, image_id="docker:ubuntu:22.04")
    assert container is None
