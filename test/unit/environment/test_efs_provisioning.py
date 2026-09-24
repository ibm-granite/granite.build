import pytest

from gbserver.environment.shared_fs.base import ProvisionedResources
from gbserver.environment.shared_fs.efs_provisioning import (
    EfsDeprovisionError,
    deprovision_efs,
    provision_efs,
)


class FakeClient:
    def __init__(self, kind, calls):
        self.kind = kind
        self.calls = calls

    def _rec(self, name, **kw):
        self.calls.append((self.kind, name, kw))

    def describe_vpcs(self, **kw):
        self._rec("describe_vpcs", **kw)
        return {"Vpcs": [{"VpcId": "vpc-def", "CidrBlock": "172.31.0.0/16"}]}

    def describe_subnets(self, **kw):
        self._rec("describe_subnets", **kw)
        return {"Subnets": [{"SubnetId": "subnet-a"}, {"SubnetId": "subnet-b"}]}

    def create_security_group(self, **kw):
        self._rec("create_security_group", **kw)
        return {"GroupId": "sg-new"}

    def authorize_security_group_ingress(self, **kw):
        self._rec("authorize", **kw)
        return {}

    def delete_security_group(self, **kw):
        self._rec("delete_security_group", **kw)
        return {}

    def create_file_system(self, **kw):
        self._rec("create_file_system", **kw)
        return {"FileSystemId": "fs-new", "LifeCycleState": "available"}

    def describe_file_systems(self, **kw):
        return {"FileSystems": [{"LifeCycleState": "available"}]}

    def create_mount_target(self, **kw):
        self._rec("create_mount_target", **kw)
        return {"MountTargetId": "mt-" + kw["SubnetId"]}

    def describe_mount_targets(self, **kw):
        return {"MountTargets": [{"LifeCycleState": "available"}]}

    def delete_mount_target(self, **kw):
        self._rec("delete_mount_target", **kw)
        return {}

    def delete_file_system(self, **kw):
        self._rec("delete_file_system", **kw)
        return {}


class FakeSession:
    def __init__(self):
        self.calls = []

    def client(self, kind, region_name=None):
        return FakeClient(kind, self.calls)


TAGS = {
    "app": "granite.build",
    "gb-ephemeral": "true",
    "gb-build-id": "b1",
    "gb-targetrun-id": "r1",
    "gb-created-at": "2026-09-24T00:00:00Z",
}


def _names(session, name):
    return [c for c in session.calls if c[1] == name]


def test_provision_discovers_vpc_and_creates_all(monkeypatch):
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_fs_available",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_mts_available",
        lambda *a, **k: None,
    )
    s = FakeSession()
    pr = provision_efs(s, "us-east-1", TAGS)
    assert pr.file_system_id == "fs-new"
    assert pr.dns_name == "fs-new.efs.us-east-1.amazonaws.com"
    assert pr.created_sg is True and pr.security_group_id == "sg-new"
    assert sorted(pr.mount_target_ids) == ["mt-subnet-a", "mt-subnet-b"]
    # tags propagated to the FS
    fs_call = _names(s, "create_file_system")[0][2]
    assert {"Key": "gb-build-id", "Value": "b1"} in fs_call["Tags"]
    # SG ingress on NFS 2049 from VPC CIDR
    ing = _names(s, "authorize")[0][2]
    assert ing["IpPermissions"][0]["FromPort"] == 2049


def test_provision_reuses_byo_sg(monkeypatch):
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_fs_available",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_mts_available",
        lambda *a, **k: None,
    )
    s = FakeSession()
    pr = provision_efs(s, "us-east-1", TAGS, security_group_id="sg-byo")
    assert pr.created_sg is False and pr.security_group_id == "sg-byo"
    assert _names(s, "create_security_group") == []


def test_deprovision_deletes_in_order_and_skips_byo_sg(monkeypatch):
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_mts_gone",
        lambda *a, **k: None,
    )
    s = FakeSession()
    pr = ProvisionedResources(
        region="us-east-1",
        file_system_id="fs-1",
        dns_name="fs-1.efs.us-east-1.amazonaws.com",
        mount_target_ids=["mt-1"],
        subnet_ids=["subnet-a"],
        security_group_id="sg-byo",
        created_sg=False,
    )
    failures = deprovision_efs(s, pr)
    assert failures == []
    assert _names(s, "delete_mount_target") and _names(s, "delete_file_system")
    assert _names(s, "delete_security_group") == []  # BYO sg untouched


def test_deprovision_best_effort_collects_failures(monkeypatch):
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_mts_gone",
        lambda *a, **k: None,
    )

    class BoomClient(FakeClient):
        def delete_file_system(self, **kw):
            raise RuntimeError("boom-fs")

    class BoomSession(FakeSession):
        def client(self, kind, region_name=None):
            return BoomClient(kind, self.calls)

    pr = ProvisionedResources(
        region="us-east-1",
        file_system_id="fs-1",
        dns_name="d",
        mount_target_ids=["mt-1"],
        subnet_ids=["subnet-a"],
        security_group_id="sg-1",
        created_sg=True,
    )
    failures = deprovision_efs(BoomSession(), pr)
    assert any("boom-fs" in f for f in failures)


def test_deprovision_raises_helper_error_wraps_failures():
    """EfsDeprovisionError carries the ProvisionedResources + failure list."""
    pr = ProvisionedResources(
        region="us-east-1",
        file_system_id="fs-x",
        dns_name="d",
        mount_target_ids=[],
        subnet_ids=[],
        security_group_id=None,
        created_sg=False,
    )
    err = EfsDeprovisionError(pr, ["delete_file_system fs-x: boom"])
    assert err.provisioned is pr
    assert err.failures == ["delete_file_system fs-x: boom"]
    assert "fs-x" in str(err)
