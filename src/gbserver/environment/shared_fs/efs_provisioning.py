"""boto3 create/destroy for ephemeral EFS mounts (issue #391).

Pure, synchronous boto3 orchestration isolated from skypilot.py and the shell-
emitting provider so it can be unit-tested with a fake session. The EfsProvider
wraps these in asyncio.to_thread.

boto3 is intentionally NOT imported here: the caller injects a ``boto3.Session``
(``.client("efs"|"ec2", region_name=...)``), so this module imports cleanly in a
venv without boto3 and unit tests drive it with a fake session.
"""

import hashlib
import time
from typing import List, Optional

from gbserver.environment.shared_fs.base import ProvisionedResources
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

_WAIT_TIMEOUT_S = 600
_WAIT_INTERVAL_S = 5
# Bounded retries for delete_security_group when the mount-target ENIs are still
# detaching (DependencyViolation); ~_SG_DELETE_ATTEMPTS * _WAIT_INTERVAL_S window.
_SG_DELETE_ATTEMPTS = 6


class EfsDeprovisionError(RuntimeError):
    """Raised when :func:`deprovision_efs` could not delete every resource.

    Carries the :class:`ProvisionedResources` it was asked to reap and the
    non-empty ``failures`` list (one string per resource that could not be
    deleted) so the caller can log the orphan for tag-based reclamation.
    """

    def __init__(self, provisioned: ProvisionedResources, failures: List[str]):
        self.provisioned = provisioned
        self.failures = failures
        super().__init__(
            f"ephemeral EFS deprovision failed for {provisioned.file_system_id}: "
            + "; ".join(failures)
        )


def _aws_tags(tags: dict) -> List[dict]:
    return [{"Key": k, "Value": v} for k, v in tags.items()]


def _is_duplicate_sg(exc) -> bool:
    """True if ``exc`` is a boto3 ``InvalidGroup.Duplicate`` (SG name already
    exists). Read the ClientError code without importing botocore (the session is
    injected); fall back to the message so odd error shapes still match."""
    resp = getattr(exc, "response", None)
    code = resp.get("Error", {}).get("Code", "") if isinstance(resp, dict) else ""
    return code == "InvalidGroup.Duplicate" or "InvalidGroup.Duplicate" in str(exc)


def _is_duplicate_permission(exc) -> bool:
    """True if ``exc`` is ``InvalidPermission.Duplicate`` (the ingress rule
    already exists), so :func:`_ensure_nfs_ingress` is idempotent."""
    resp = getattr(exc, "response", None)
    code = resp.get("Error", {}).get("Code", "") if isinstance(resp, dict) else ""
    return (
        code == "InvalidPermission.Duplicate"
        or "InvalidPermission.Duplicate" in str(exc)
    )


def _is_dependency_violation(exc) -> bool:
    """True if ``exc`` is a ``DependencyViolation`` -- e.g. deleting a security
    group whose mount-target ENIs are still detaching."""
    resp = getattr(exc, "response", None)
    code = resp.get("Error", {}).get("Code", "") if isinstance(resp, dict) else ""
    return code == "DependencyViolation" or "DependencyViolation" in str(exc)


def _ensure_nfs_ingress(ec2, sg_id: str, vpc_cidr: str) -> None:
    """Open NFS 2049 ingress from ``vpc_cidr`` on ``sg_id`` (idempotent).

    Run on both the freshly-created and the adopted-duplicate SG path. A prior
    run that crashed between ``create_security_group`` and this authorize leaves
    an SG with no ingress; re-authorizing on adopt lets the next run self-heal
    instead of every mount timing out opaquely. ``InvalidPermission.Duplicate``
    (the rule already exists) is swallowed."""
    try:
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
    except Exception as e:  # noqa: BLE001 - duplicate rule is fine; re-raise others
        if not _is_duplicate_permission(e):
            raise


def _delete_sg_with_retry(ec2, sg_id: str) -> Optional[str]:
    """Delete ``sg_id``, retrying on ``DependencyViolation``.

    Mount-target ENIs can linger briefly after the filesystem/mount targets
    report deleted, so deleting the SG immediately raises ``DependencyViolation``
    on an otherwise-clean teardown. Retry a bounded number of times before giving
    up. Returns ``None`` on success, else a failure string; any non-dependency
    error returns immediately (it will not self-resolve)."""
    last = ""
    for attempt in range(_SG_DELETE_ATTEMPTS):
        try:
            ec2.delete_security_group(GroupId=sg_id)
            return None
        except Exception as e:  # noqa: BLE001 - best-effort reap
            if not _is_dependency_violation(e):
                return f"delete_security_group {sg_id}: {e}"
            last = str(e)
            if attempt < _SG_DELETE_ATTEMPTS - 1:
                time.sleep(_WAIT_INTERVAL_S)
    return f"delete_security_group {sg_id}: {last}"


def _find_sg_id(ec2, group_name: str, vpc_id: str) -> str:
    """Look up the id of the SG named ``group_name`` in ``vpc_id`` (the one a
    prior run left behind, since the name is stable per mount)."""
    sgs = ec2.describe_security_groups(
        Filters=[
            {"Name": "group-name", "Values": [group_name]},
            {"Name": "vpc-id", "Values": [vpc_id]},
        ]
    )["SecurityGroups"]
    if not sgs:
        raise RuntimeError(
            f"security group {group_name!r} reported duplicate but was not found "
            f"in VPC {vpc_id}"
        )
    return sgs[0]["GroupId"]


def _wait_fs_available(efs, file_system_id: str) -> None:
    deadline = time.monotonic() + _WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        st = efs.describe_file_systems(FileSystemId=file_system_id)["FileSystems"][0]
        if st["LifeCycleState"] == "available":
            return
        time.sleep(_WAIT_INTERVAL_S)
    raise TimeoutError(f"EFS {file_system_id} not available within {_WAIT_TIMEOUT_S}s")


def _wait_mts_available(efs, file_system_id: str, expected: int) -> None:
    """Wait until all ``expected`` mount targets exist and are ``available``.

    Requiring the full count (not merely a non-empty available subset) guards
    against ``describe_mount_targets`` eventual consistency reporting success
    before every AZ's mount target is listed, which would let a step VM in a
    lagging AZ fail to mount."""
    deadline = time.monotonic() + _WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        mts = efs.describe_mount_targets(FileSystemId=file_system_id)["MountTargets"]
        if len(mts) == expected and all(
            m["LifeCycleState"] == "available" for m in mts
        ):
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
    session,
    region,
    tags,
    vpc_id=None,
    subnets=None,
    security_group_id=None,
    mount_point="",
) -> ProvisionedResources:
    """Create an ephemeral EFS filesystem with a mount target per AZ and return
    its runtime identity.

    Creates (as needed) a security group opening NFS 2049 from the VPC CIDR, the
    encrypted elastic filesystem, and one mount target per AZ, all tagged from
    ``tags``. If any step fails part-way, whatever was already created is torn
    down (best-effort, via :func:`deprovision_efs`) before the original error is
    re-raised, so a partial provision does not leak (issue #391).

    :param session: an injected boto3-like ``Session`` (``.client("efs"|"ec2",
        region_name=...)``); boto3 is not imported here.
    :param region: AWS region to create the filesystem in.
    :param tags: tags applied to every created resource (includes the
        ``gb-targetrun-id``/``gb-build-id`` used for tag-based reclamation).
    :param vpc_id: VPC to place the filesystem in; the default VPC is discovered
        when unset.
    :param subnets: explicit subnet ids for the mount targets, left as-authored;
        when unset, subnets are auto-discovered and deduped to one per AZ.
    :param security_group_id: a BYO security group to reuse; when unset one is
        created (and adopted if a prior run left an identically-named one behind).
    :param mount_point: this mount's mount_point, folded into the created SG name
        so sibling ephemeral mounts (which share ``tags``) get distinct groups.
    :returns: a :class:`ProvisionedResources` describing the created filesystem,
        mount targets, and security group.
    """
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
        # EFS allows one mount target per AZ per filesystem, but a VPC commonly
        # has >1 subnet in an AZ; keep the first subnet seen per AZ so we don't
        # hit MountTargetConflict on a second same-AZ subnet. An explicit
        # efs.subnets list is left as-authored (the operator owns that choice).
        seen_azs: set = set()
        subnets = []
        for s in ec2.describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc_id]}])[
            "Subnets"
        ]:
            az = s.get("AvailabilityZone")
            if az in seen_azs:
                continue
            seen_azs.add(az)
            subnets.append(s["SubnetId"])
    if not subnets:
        raise RuntimeError(f"no subnets found in VPC {vpc_id}")

    created_sg = False
    sg_id = security_group_id
    fsid = ""
    mt_ids: List[str] = []
    # Everything below creates real AWS infra. Any failure part-way through must
    # not leak (issue #391): on error, best-effort tear down whatever we already
    # created (reusing deprovision_efs) before re-raising the original error.
    try:
        if not sg_id:
            # Every ephemeral mount in one setup gets the same tags (same
            # gb-targetrun-id), so fold a short stable hash of the mount_point in
            # to keep each mount's SG name distinct (else the 2nd create raises
            # InvalidGroup.Duplicate). Stable across retries of the same mount.
            mp_hash = hashlib.sha1(mount_point.encode()).hexdigest()[:8]
            group_name = f"gb-efs-{tags.get('gb-targetrun-id', 'x')[:12]}-{mp_hash}"
            try:
                sg_id = ec2.create_security_group(
                    GroupName=group_name,
                    Description="granite.build ephemeral EFS (NFS 2049)",
                    VpcId=vpc_id,
                    TagSpecifications=[
                        {"ResourceType": "security-group", "Tags": _aws_tags(tags)}
                    ],
                )["GroupId"]
            except Exception as e:
                if not _is_duplicate_sg(e):
                    raise
                # A prior retry of this exact mount leaked its SG (the name is
                # stable). Adopt it and mark it ours so teardown reaps it, instead
                # of wedging every retry on the duplicate.
                sg_id = _find_sg_id(ec2, group_name, vpc_id)
                created_sg = True
            else:
                created_sg = True
            # Open NFS ingress on both the created and the adopted SG (idempotent):
            # a leaked SG from a run that crashed before authorize has no ingress,
            # so re-authorizing on adopt lets the next run self-heal.
            _ensure_nfs_ingress(ec2, sg_id, vpc_cidr)

        fsid = efs.create_file_system(
            PerformanceMode="generalPurpose",
            ThroughputMode="elastic",
            Encrypted=True,
            Tags=_aws_tags(tags),
        )["FileSystemId"]
        _wait_fs_available(efs, fsid)

        for sn in subnets:
            mt_ids.append(
                efs.create_mount_target(
                    FileSystemId=fsid, SubnetId=sn, SecurityGroups=[sg_id]
                )["MountTargetId"]
            )
        _wait_mts_available(efs, fsid, len(mt_ids))
    except Exception:
        partial = ProvisionedResources(
            region=region,
            file_system_id=fsid,
            dns_name=f"{fsid}.efs.{region}.amazonaws.com" if fsid else "",
            mount_target_ids=mt_ids,
            subnet_ids=list(subnets),
            security_group_id=sg_id,
            created_sg=created_sg,
        )
        rb_failures = deprovision_efs(session, partial)
        if rb_failures:
            logger.warning(
                "provision_efs rollback incomplete; ORPHAN fsid=%s sg=%s "
                "region=%s tags(build=%s,targetrun=%s): %s",
                fsid or "<none>",
                sg_id if created_sg else "<byo>",
                region,
                tags.get("gb-build-id"),
                tags.get("gb-targetrun-id"),
                "; ".join(rb_failures),
            )
        raise

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
    """Best-effort delete the resources described by ``provisioned``.

    Deletes the mount targets, waits for them to clear, deletes the filesystem,
    and deletes the security group only if we created it (``created_sg``). Each
    delete is attempted independently; failures are collected rather than raised,
    so one wedged resource does not strand the others. Also used to roll back a
    partial provision (``file_system_id`` empty -> the FS wait/delete is skipped).

    :param session: an injected boto3-like ``Session``.
    :param provisioned: the resources to reap.
    :returns: a list of failure strings (one per resource that could not be
        deleted); empty when everything was deleted (or was already gone).
    """
    efs = session.client("efs", region_name=provisioned.region)
    ec2 = session.client("ec2", region_name=provisioned.region)
    failures: List[str] = []
    for mt in provisioned.mount_target_ids:
        try:
            efs.delete_mount_target(MountTargetId=mt)
        except Exception as e:  # noqa: BLE001 - best-effort reap
            failures.append(f"delete_mount_target {mt}: {e}")
    # file_system_id is empty only on a partial-provision rollback where the FS
    # was never created; skip the FS wait/delete so we don't log spurious errors.
    if provisioned.file_system_id:
        try:
            _wait_mts_gone(efs, provisioned.file_system_id)
        except Exception as e:  # noqa: BLE001
            failures.append(f"wait_mts_gone {provisioned.file_system_id}: {e}")
        try:
            efs.delete_file_system(FileSystemId=provisioned.file_system_id)
        except Exception as e:  # noqa: BLE001
            failures.append(f"delete_file_system {provisioned.file_system_id}: {e}")
    if provisioned.created_sg and provisioned.security_group_id:
        # Retry DependencyViolation: the mount-target ENIs can still be detaching
        # right after the FS delete, which would otherwise log a spurious orphan.
        sg_failure = _delete_sg_with_retry(ec2, provisioned.security_group_id)
        if sg_failure:
            failures.append(sg_failure)
    return failures
