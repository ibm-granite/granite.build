"""Unit tests for the real-AWS no-leak helpers used by the gated ephemeral-EFS
integration test (``test/integration/.../skypilot/aws/test_shared_fs.py``).

The helpers take an injected boto3-like ``session`` so the tag discovery, the
pagination, and the best-effort reap can be exercised here with a fake session,
without AWS or the ``sky`` extra. The integration test wraps a real build with
these helpers to assert nothing the build auto-provisioned survived teardown.
"""

import pytest
from libgbtest.aws_efs_noleak import list_gb_ephemeral, reap_leaked


def _fs(fsid, ephemeral=True, lifecycle="available"):
    tags = [{"Key": "app", "Value": "granite.build"}]
    if ephemeral:
        tags.append({"Key": "gb-ephemeral", "Value": "true"})
    return {"FileSystemId": fsid, "Tags": tags, "LifeCycleState": lifecycle}


class FakeEfs:
    def __init__(self, calls, fs_pages, mount_targets=None):
        self.calls = calls
        self._fs_pages = fs_pages  # list of pages, each a list of FS dicts
        self._mts = mount_targets or {}

    def describe_file_systems(self, **kw):
        self.calls.append(("efs", "describe_file_systems", kw))
        idx = 0 if not kw.get("Marker") else int(kw["Marker"])
        page = self._fs_pages[idx]
        resp = {"FileSystems": page}
        if idx + 1 < len(self._fs_pages):
            resp["NextMarker"] = str(idx + 1)
        return resp

    def describe_mount_targets(self, **kw):
        self.calls.append(("efs", "describe_mount_targets", kw))
        return {
            "MountTargets": [
                {"MountTargetId": m} for m in self._mts.get(kw["FileSystemId"], [])
            ]
        }

    def delete_mount_target(self, **kw):
        self.calls.append(("efs", "delete_mount_target", kw))
        return {}

    def delete_file_system(self, **kw):
        self.calls.append(("efs", "delete_file_system", kw))
        return {}


class FakeEc2:
    def __init__(self, calls, sg_pages):
        self.calls = calls
        self._sg_pages = sg_pages  # list of pages, each a list of SG dicts

    def describe_security_groups(self, **kw):
        self.calls.append(("ec2", "describe_security_groups", kw))
        idx = 0 if not kw.get("NextToken") else int(kw["NextToken"])
        page = self._sg_pages[idx]
        resp = {"SecurityGroups": page}
        if idx + 1 < len(self._sg_pages):
            resp["NextToken"] = str(idx + 1)
        return resp

    def delete_security_group(self, **kw):
        self.calls.append(("ec2", "delete_security_group", kw))
        return {}


class FakeSession:
    def __init__(self, efs, ec2):
        self.calls = []
        efs.calls = self.calls
        ec2.calls = self.calls
        self._efs = efs
        self._ec2 = ec2

    def client(self, kind, region_name=None):
        return self._efs if kind == "efs" else self._ec2


def _names(session, name):
    return [c for c in session.calls if c[2] is not None and c[1] == name]


def test_list_gb_ephemeral_filters_by_tag():
    """Only gb-ephemeral=true filesystems are returned, and the EC2 lookup pushes
    the tag filter server-side."""
    efs = FakeEfs(None, [[_fs("fs-eph"), _fs("fs-other", ephemeral=False)]])
    ec2 = FakeEc2(None, [[{"GroupId": "sg-eph"}]])
    s = FakeSession(efs, ec2)

    fsids, sgids = list_gb_ephemeral(s, "us-east-1")

    assert fsids == {"fs-eph"}
    assert sgids == {"sg-eph"}
    ing = _names(s, "describe_security_groups")[0][2]
    assert ing["Filters"] == [{"Name": "tag:gb-ephemeral", "Values": ["true"]}]


def test_list_gb_ephemeral_excludes_deleting_filesystems():
    """A just-deleted filesystem still returned in a terminal lifecycle state is
    NOT counted (EFS delete is eventually consistent -- else a correctly
    deprovisioned run reads as a leak)."""
    efs = FakeEfs(
        None,
        [
            [
                _fs("fs-live", lifecycle="available"),
                _fs("fs-going", lifecycle="deleting"),
                _fs("fs-gone", lifecycle="deleted"),
            ]
        ],
    )
    ec2 = FakeEc2(None, [[]])
    s = FakeSession(efs, ec2)

    fsids, _ = list_gb_ephemeral(s, "us-east-1")

    assert fsids == {"fs-live"}


def test_reap_leaked_tolerates_already_gone_filesystem(monkeypatch):
    """If the filesystem finished deleting between detection and reap, the
    resulting NotFound is swallowed (not a leak, not a crash)."""
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_mts_gone",
        lambda *a, **k: None,
    )

    class GoneEfs(FakeEfs):
        def describe_mount_targets(self, **kw):
            raise RuntimeError("FileSystem 'fs-leak' does not exist.")

    efs = GoneEfs(None, [[]])
    ec2 = FakeEc2(None, [[]])
    s = FakeSession(efs, ec2)

    failures = reap_leaked(s, "us-east-1", {"fs-leak"}, set())

    assert failures == []
    assert _names(s, "delete_file_system") == []


def test_list_gb_ephemeral_paginates():
    """FS Marker and SG NextToken pagination are both followed."""
    efs = FakeEfs(None, [[_fs("fs-1")], [_fs("fs-2")]])
    ec2 = FakeEc2(None, [[{"GroupId": "sg-1"}], [{"GroupId": "sg-2"}]])
    s = FakeSession(efs, ec2)

    fsids, sgids = list_gb_ephemeral(s, "us-east-1")

    assert fsids == {"fs-1", "fs-2"}
    assert sgids == {"sg-1", "sg-2"}


def test_reap_leaked_deletes_filesystem_then_security_group(monkeypatch):
    """A leaked FS is reaped via deprovision_efs (mount targets + filesystem) and
    a leaked SG is deleted directly; no failures are reported on success."""
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_mts_gone",
        lambda *a, **k: None,
    )
    efs = FakeEfs(None, [[]], mount_targets={"fs-leak": ["mt-a", "mt-b"]})
    ec2 = FakeEc2(None, [[]])
    s = FakeSession(efs, ec2)

    failures = reap_leaked(s, "us-east-1", {"fs-leak"}, {"sg-leak"})

    assert failures == []
    assert sorted(c[2]["MountTargetId"] for c in _names(s, "delete_mount_target")) == [
        "mt-a",
        "mt-b",
    ]
    assert _names(s, "delete_file_system")[0][2]["FileSystemId"] == "fs-leak"
    assert _names(s, "delete_security_group")[0][2]["GroupId"] == "sg-leak"


def test_reap_leaked_collects_failures(monkeypatch):
    """Reap is best-effort: a delete error is collected, not raised."""
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_mts_gone",
        lambda *a, **k: None,
    )

    class BoomEc2(FakeEc2):
        def delete_security_group(self, **kw):
            raise RuntimeError("dependency-violation")

    efs = FakeEfs(None, [[]], mount_targets={"fs-leak": []})
    ec2 = BoomEc2(None, [[]])
    s = FakeSession(efs, ec2)

    failures = reap_leaked(s, "us-east-1", {"fs-leak"}, {"sg-leak"})

    assert any("dependency-violation" in f for f in failures)
