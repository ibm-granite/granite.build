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

"""libgbtest.xdist_groups puts every real-SkyPilot test in one xdist group."""

import pathlib

import pytest
from libgbtest import xdist_groups
from libgbtest.kube import KUBE_XDIST_GROUP

_REPO = pathlib.Path(__file__).resolve().parents[3]


class _Item:
    """The slice of pytest.Item the helper reads and writes."""

    def __init__(self, nodeid: str, *marks: pytest.MarkDecorator):
        self.nodeid = nodeid
        self.marks = [m.mark for m in marks]

    def get_closest_marker(self, name):
        return next((m for m in self.marks if m.name == name), None)

    def iter_markers(self, name):
        return (m for m in self.marks if m.name == name)

    def add_marker(self, marker):
        self.marks.append(marker.mark)

    def groups(self) -> list[str]:
        return [
            m.args[0] if m.args else m.kwargs["name"]
            for m in self.iter_markers("xdist_group")
        ]


def test_skypilot_integration_items_join_the_group():
    slurm = _Item("slurm", pytest.mark.skypilot_integration)
    other = _Item("unit")
    xdist_groups.group_skypilot_integration_tests([slurm, other])
    assert slurm.groups() == ["skypilot"]
    assert other.groups() == []


def test_kube_marker_is_the_same_group():
    """A kube test carries KUBE_XDIST_GROUP and gets the group again: one name."""
    kube = _Item("kube", pytest.mark.skypilot_integration, KUBE_XDIST_GROUP)
    xdist_groups.group_skypilot_integration_tests([kube])
    assert set(kube.groups()) == {"skypilot"}


def test_another_group_is_rejected():
    """xdist would join the names into a separate, still-concurrent group."""
    aws = _Item(
        "aws",
        pytest.mark.skypilot_integration,
        pytest.mark.xdist_group("buildtest_aws"),
    )
    with pytest.raises(pytest.UsageError, match="buildtest_aws"):
        xdist_groups.group_skypilot_integration_tests([aws], strict=True)


def test_another_group_is_tolerated_without_loadgroup():
    """Groups are inert outside loadgroup, so a plain local run never errors."""
    aws = _Item(
        "aws",
        pytest.mark.skypilot_integration,
        pytest.mark.xdist_group("buildtest_aws"),
    )
    xdist_groups.group_skypilot_integration_tests([aws])
    assert "skypilot" in aws.groups()


def test_slurm_and_kube_tests_are_marked():
    """The slurm and kube tests that collided carry the marker the hook keys on."""
    names = [
        "test/integration/environment/test_skypilot_slurm_e2e.py",
        "test/integration/standalone/buildrunner/skypilot_slurm/test_skypilot_1step.py",
        "test/integration/standalone/buildrunner/skypilot_kube/test_skypilot_kube_1step.py",
        "test/steps/data/dpk/skypilot/slurm-tok/test_skypilot_slurm_dpk_tok.py",
        "test/steps/data/dpk/skypilot/kube-tok/test_skypilot_kube_dpk_tok.py",
        "test/steps/byoc/skypilot/slurm/test_skypilot_byoc.py",
    ]
    for name in names:
        assert "pytest.mark.skypilot_integration" in (_REPO / name).read_text(), name
