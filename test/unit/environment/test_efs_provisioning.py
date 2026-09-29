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
        return {
            "Subnets": [
                {"SubnetId": "subnet-a", "AvailabilityZone": "us-east-1a"},
                {"SubnetId": "subnet-b", "AvailabilityZone": "us-east-1b"},
            ]
        }

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


def test_provision_dedupes_discovered_subnets_to_one_per_az(monkeypatch):
    """Auto-discovery must create at most one mount target per AZ: EFS allows a
    single mount target per AZ per filesystem, so two discovered subnets sharing
    an AZ would otherwise raise MountTargetConflict on the second (issue #391;
    custom VPCs commonly have >1 subnet per AZ)."""
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_fs_available",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_mts_available",
        lambda *a, **k: None,
    )

    class MultiAzClient(FakeClient):
        def describe_subnets(self, **kw):
            self._rec("describe_subnets", **kw)
            return {
                "Subnets": [
                    {"SubnetId": "subnet-a1", "AvailabilityZone": "us-east-1a"},
                    {"SubnetId": "subnet-a2", "AvailabilityZone": "us-east-1a"},
                    {"SubnetId": "subnet-b1", "AvailabilityZone": "us-east-1b"},
                ]
            }

    class MultiAzSession(FakeSession):
        def client(self, kind, region_name=None):
            return MultiAzClient(kind, self.calls)

    s = MultiAzSession()
    pr = provision_efs(s, "us-east-1", TAGS)
    # one subnet per AZ (first seen wins): a1 for us-east-1a, b1 for us-east-1b.
    mt_subnets = sorted(c[2]["SubnetId"] for c in _names(s, "create_mount_target"))
    assert mt_subnets == ["subnet-a1", "subnet-b1"]
    assert len(pr.mount_target_ids) == 2


def test_two_ephemeral_mounts_get_distinct_sg_names(monkeypatch):
    """setup_skypilot passes the same tags (same gb-targetrun-id) to every
    provider, so the SG name must also fold in the mount_point -- otherwise two
    ephemeral mounts in one VPC collide on InvalidGroup.Duplicate (issue #391)."""
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_fs_available",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_mts_available",
        lambda *a, **k: None,
    )
    s = FakeSession()
    provision_efs(s, "us-east-1", TAGS, mount_point="/mnt/a")
    provision_efs(s, "us-east-1", TAGS, mount_point="/mnt/b")
    names = [c[2]["GroupName"] for c in _names(s, "create_security_group")]
    assert len(names) == 2
    assert names[0] != names[1], f"SG names collide across mounts: {names}"
    # both still carry the target-run id so they're reclaimable by run
    assert all(n.startswith("gb-efs-r1") for n in names)


class FakeClientError(Exception):
    """Mimics botocore.exceptions.ClientError's error-code shape without pulling
    in botocore (this module is driven by an injected session)."""

    def __init__(self, code):
        self.response = {"Error": {"Code": code}}
        super().__init__(code)


def test_provision_adopts_leaked_sg_on_duplicate(monkeypatch):
    """If a prior retry leaked this mount's SG (stable targetrun-id + mount_point
    -> stable name), create_security_group raises InvalidGroup.Duplicate. Rather
    than wedge the retry, provision adopts the existing SG (which already carries
    the NFS ingress) and marks it ours so teardown reaps it (issue #391)."""
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_fs_available",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_mts_available",
        lambda *a, **k: None,
    )

    class DupSgClient(FakeClient):
        def create_security_group(self, **kw):
            self._rec("create_security_group", **kw)
            raise FakeClientError("InvalidGroup.Duplicate")

        def describe_security_groups(self, **kw):
            self._rec("describe_security_groups", **kw)
            return {"SecurityGroups": [{"GroupId": "sg-existing"}]}

    class DupSgSession(FakeSession):
        def client(self, kind, region_name=None):
            return DupSgClient(kind, self.calls)

    s = DupSgSession()
    pr = provision_efs(s, "us-east-1", TAGS, mount_point="/mnt/a")
    assert pr.security_group_id == "sg-existing"
    assert pr.created_sg is True  # adopted -> teardown deletes it
    # looked up by the exact name we tried to create, scoped to the VPC
    dsg = _names(s, "describe_security_groups")[0][2]
    assert {"Name": "vpc-id", "Values": ["vpc-def"]} in dsg["Filters"]
    # the leaked SG already has the NFS ingress; don't re-authorize
    assert _names(s, "authorize") == []


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


def test_provision_rolls_back_created_resources_when_mount_target_wait_fails(
    monkeypatch,
):
    """A failure after the SG/FS/mount targets are created must not leak: the
    partial resources are torn down (issue #391 no-leak-on-partial-provision)."""
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_fs_available",
        lambda *a, **k: None,
    )

    def _boom(*a, **k):
        raise TimeoutError("mount targets not available")

    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_mts_available",
        _boom,
    )
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_mts_gone",
        lambda *a, **k: None,
    )
    s = FakeSession()
    with pytest.raises(TimeoutError):
        provision_efs(s, "us-east-1", TAGS)
    # everything created is reaped, in the right order, so nothing leaks
    assert sorted(c[2]["MountTargetId"] for c in _names(s, "delete_mount_target")) == [
        "mt-subnet-a",
        "mt-subnet-b",
    ]
    assert _names(s, "delete_file_system")[0][2]["FileSystemId"] == "fs-new"
    assert _names(s, "delete_security_group")[0][2]["GroupId"] == "sg-new"


def test_provision_rollback_deletes_only_sg_when_fs_creation_fails(monkeypatch):
    """If the FS never gets created, rollback deletes the SG we created and does
    not attempt (and log spurious failures for) a filesystem/mount-target delete."""

    class NoFsClient(FakeClient):
        def create_file_system(self, **kw):
            raise RuntimeError("throttled")

    class NoFsSession(FakeSession):
        def client(self, kind, region_name=None):
            return NoFsClient(kind, self.calls)

    s = NoFsSession()
    with pytest.raises(RuntimeError, match="throttled"):
        provision_efs(s, "us-east-1", TAGS)
    assert _names(s, "delete_security_group")[0][2]["GroupId"] == "sg-new"
    assert _names(s, "delete_file_system") == []
    assert _names(s, "delete_mount_target") == []


def test_provision_rollback_keeps_byo_sg_on_failure(monkeypatch):
    """A BYO security group is never deleted during rollback (we did not create it)."""
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_fs_available",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_mts_available",
        lambda *a, **k: (_ for _ in ()).throw(TimeoutError("boom")),
    )
    monkeypatch.setattr(
        "gbserver.environment.shared_fs.efs_provisioning._wait_mts_gone",
        lambda *a, **k: None,
    )
    s = FakeSession()
    with pytest.raises(TimeoutError):
        provision_efs(s, "us-east-1", TAGS, security_group_id="sg-byo")
    assert _names(s, "delete_file_system")[0][2]["FileSystemId"] == "fs-new"
    assert _names(s, "delete_security_group") == []  # BYO sg untouched


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
