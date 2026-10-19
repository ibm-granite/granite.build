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
import re
import types

import pytest
from libgbtest import xdist_groups
from libgbtest.kube import KUBE_XDIST_GROUP

_REPO = pathlib.Path(__file__).resolve().parents[3]
# An uncommented environment_uri naming a SkyPilot environment.
_SKYPILOT_ENV = re.compile(r"^\s*environment_uri:\s*\S*environments/skypilot/", re.M)


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


def _skypilot_fixture_test_modules() -> set[pathlib.Path]:
    """Test modules whose build.yaml fixture (test-data/<dir>, any depth) targets SkyPilot."""
    modules = set()
    for build in (_REPO / "test-data").rglob("build.yaml"):
        if not _SKYPILOT_ENV.search(build.read_text()):
            continue
        for parent in build.parents:
            if parent == _REPO / "test-data":
                break
            test_dir = _REPO / "test" / parent.relative_to(_REPO / "test-data")
            found = sorted(test_dir.glob("test_*.py"))
            if found:
                modules.update(found)
                break
    return modules


def test_every_extended_skypilot_build_test_is_grouped():
    """ibm-marked tests run only in ibm-extended-tests, never in CI, so are skipped."""
    modules = [
        m
        for m in _skypilot_fixture_test_modules()
        if "pytest.mark.ibm" not in m.read_text()
    ]
    assert len(modules) > 15, modules  # the fixture scan still finds them
    unmarked = [
        str(m.relative_to(_REPO))
        for m in modules
        if "pytest.mark.skypilot_integration" not in m.read_text()
    ]
    assert not unmarked, unmarked


def test_is_loadgroup_reads_the_worker_option():
    """On an xdist worker dist is "no"; loadgroup holds the mode."""
    worker = types.SimpleNamespace(
        option=types.SimpleNamespace(dist="no", loadgroup=True)
    )
    controller = types.SimpleNamespace(option=types.SimpleNamespace(dist="loadgroup"))
    plain = types.SimpleNamespace(option=types.SimpleNamespace(dist="no"))
    assert xdist_groups.is_loadgroup(worker)
    assert xdist_groups.is_loadgroup(controller)
    assert not xdist_groups.is_loadgroup(plain)
