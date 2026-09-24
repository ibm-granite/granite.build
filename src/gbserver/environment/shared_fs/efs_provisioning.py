"""boto3 create/destroy for ephemeral EFS mounts (issue #391).

Pure, synchronous boto3 orchestration isolated from skypilot.py and the shell-
emitting provider so it can be unit-tested with a fake session. The EfsProvider
wraps these in asyncio.to_thread.

boto3 is intentionally NOT imported here: the caller injects a ``boto3.Session``
(``.client("efs"|"ec2", region_name=...)``), so this module imports cleanly in a
venv without boto3 and unit tests drive it with a fake session.
"""

import time
from typing import List, Optional

from gbserver.environment.shared_fs.base import ProvisionedResources
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

_WAIT_TIMEOUT_S = 600
_WAIT_INTERVAL_S = 5


class EfsDeprovisionError(RuntimeError):
    def __init__(self, provisioned: ProvisionedResources, failures: List[str]):
        self.provisioned = provisioned
        self.failures = failures
        super().__init__(
            f"ephemeral EFS deprovision failed for {provisioned.file_system_id}: "
            + "; ".join(failures)
        )


def _aws_tags(tags: dict) -> List[dict]:
    return [{"Key": k, "Value": v} for k, v in tags.items()]


def _wait_fs_available(efs, file_system_id: str) -> None:
    deadline = time.monotonic() + _WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        st = efs.describe_file_systems(FileSystemId=file_system_id)["FileSystems"][0]
        if st["LifeCycleState"] == "available":
            return
        time.sleep(_WAIT_INTERVAL_S)
    raise TimeoutError(f"EFS {file_system_id} not available within {_WAIT_TIMEOUT_S}s")


def _wait_mts_available(efs, file_system_id: str) -> None:
    deadline = time.monotonic() + _WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        mts = efs.describe_mount_targets(FileSystemId=file_system_id)["MountTargets"]
        if mts and all(m["LifeCycleState"] == "available" for m in mts):
            return
        time.sleep(_WAIT_INTERVAL_S)
    raise TimeoutError(f"EFS {file_system_id} mount targets not available")


def _wait_mts_gone(efs, file_system_id: str) -> None:
    deadline = time.monotonic() + _WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        mts = efs.describe_mount_targets(FileSystemId=file_system_id)["MountTargets"]
        if not mts:
            return
        time.sleep(_WAIT_INTERVAL_S)
    raise TimeoutError(f"EFS {file_system_id} mount targets not deleted")


def provision_efs(
    session, region, tags, vpc_id=None, subnets=None, security_group_id=None
) -> ProvisionedResources:
    efs = session.client("efs", region_name=region)
    ec2 = session.client("ec2", region_name=region)

    if not vpc_id:
        vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])[
            "Vpcs"
        ]
        if not vpcs:
            raise RuntimeError(f"no default VPC in {region}; set efs.vpc_id")
        vpc_id, vpc_cidr = vpcs[0]["VpcId"], vpcs[0]["CidrBlock"]
    else:
        vpc = ec2.describe_vpcs(VpcIds=[vpc_id])["Vpcs"][0]
        vpc_cidr = vpc["CidrBlock"]

    if not subnets:
        subnets = [
            s["SubnetId"]
            for s in ec2.describe_subnets(
                Filters=[{"Name": "vpc-id", "Values": [vpc_id]}]
            )["Subnets"]
        ]
    if not subnets:
        raise RuntimeError(f"no subnets found in VPC {vpc_id}")

    created_sg = False
    sg_id = security_group_id
    if not sg_id:
        sg_id = ec2.create_security_group(
            GroupName=f"gb-efs-{tags.get('gb-targetrun-id', 'x')[:12]}",
            Description="granite.build ephemeral EFS (NFS 2049)",
            VpcId=vpc_id,
            TagSpecifications=[
                {"ResourceType": "security-group", "Tags": _aws_tags(tags)}
            ],
        )["GroupId"]
        created_sg = True
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[
                {
                    "IpProtocol": "tcp",
                    "FromPort": 2049,
                    "ToPort": 2049,
                    "IpRanges": [{"CidrIp": vpc_cidr}],
                }
            ],
        )

    fsid = efs.create_file_system(
        PerformanceMode="generalPurpose",
        ThroughputMode="elastic",
        Encrypted=True,
        Tags=_aws_tags(tags),
    )["FileSystemId"]
    _wait_fs_available(efs, fsid)

    mt_ids = []
    for sn in subnets:
        mt_ids.append(
            efs.create_mount_target(
                FileSystemId=fsid, SubnetId=sn, SecurityGroups=[sg_id]
            )["MountTargetId"]
        )
    _wait_mts_available(efs, fsid)

    logger.info(
        "provisioned ephemeral EFS %s (%d mount targets) in %s",
        fsid,
        len(mt_ids),
        region,
    )
    return ProvisionedResources(
        region=region,
        file_system_id=fsid,
        dns_name=f"{fsid}.efs.{region}.amazonaws.com",
        mount_target_ids=mt_ids,
        subnet_ids=list(subnets),
        security_group_id=sg_id,
        created_sg=created_sg,
    )


def deprovision_efs(session, provisioned: ProvisionedResources) -> List[str]:
    efs = session.client("efs", region_name=provisioned.region)
    ec2 = session.client("ec2", region_name=provisioned.region)
    failures: List[str] = []
    for mt in provisioned.mount_target_ids:
        try:
            efs.delete_mount_target(MountTargetId=mt)
        except Exception as e:  # noqa: BLE001 - best-effort reap
            failures.append(f"delete_mount_target {mt}: {e}")
    try:
        _wait_mts_gone(efs, provisioned.file_system_id)
    except Exception as e:  # noqa: BLE001
        failures.append(f"wait_mts_gone {provisioned.file_system_id}: {e}")
    try:
        efs.delete_file_system(FileSystemId=provisioned.file_system_id)
    except Exception as e:  # noqa: BLE001
        failures.append(f"delete_file_system {provisioned.file_system_id}: {e}")
    if provisioned.created_sg and provisioned.security_group_id:
        try:
            ec2.delete_security_group(GroupId=provisioned.security_group_id)
        except Exception as e:  # noqa: BLE001
            failures.append(
                f"delete_security_group {provisioned.security_group_id}: {e}"
            )
    return failures
