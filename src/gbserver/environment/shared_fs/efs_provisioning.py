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
from typing import List, Optional, Tuple

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


def _ensure_nfs_ingress(ec2, sg_id: str, vpc_cidrs: List[str]) -> None:
    """Open NFS 2049 ingress from every CIDR in ``vpc_cidrs`` on ``sg_id``
    (idempotent).

    Run on both the freshly-created and the adopted-duplicate SG path. A prior
    run that crashed between ``create_security_group`` and this authorize leaves
    an SG with no ingress; re-authorizing on adopt lets the next run self-heal
    instead of every mount timing out opaquely. ALL of the VPC's CIDRs are opened
    (not just the primary) because an auto-discovered subnet can live in a
    secondary CIDR, whose mount target would otherwise have NFS 2049 denied.
    ``InvalidPermission.Duplicate`` (that rule already exists) is swallowed."""
    for cidr in vpc_cidrs:
        try:
            ec2.authorize_security_group_ingress(
                GroupId=sg_id,
                IpPermissions=[
                    {
                        "IpProtocol": "tcp",
                        "FromPort": 2049,
                        "ToPort": 2049,
                        "IpRanges": [{"CidrIp": cidr}],
                    }
                ],
            )
        except Exception as e:  # noqa: BLE001 - duplicate rule is fine; re-raise
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


def _vpc_cidrs(vpc: dict) -> List[str]:
    """Every associated IPv4 CIDR of ``vpc`` (primary + secondary).

    A mount target can be auto-discovered in a secondary-CIDR subnet, so the SG
    must open NFS 2049 from each associated CIDR, not just the primary
    ``CidrBlock`` (a mount target in an unlisted CIDR would have NFS denied).
    Falls back to the primary ``CidrBlock`` when the association set is absent."""
    cidrs: List[str] = []
    for assoc in vpc.get("CidrBlockAssociationSet") or []:
        state = (assoc.get("CidrBlockState") or {}).get("State")
        cidr = assoc.get("CidrBlock")
        if cidr and state in (None, "associated"):
            cidrs.append(cidr)
    if not cidrs and vpc.get("CidrBlock"):
        cidrs.append(vpc["CidrBlock"])
    return cidrs


def _resolve_network(ec2, region, vpc_id, subnets) -> Tuple[str, List[str], List[str]]:
    """Resolve ``(vpc_id, vpc_cidrs, subnets)`` for the mount targets.

    ``vpc_id`` is taken from config when set; else derived from an explicit
    ``subnets`` list (so ``efs.subnets`` without ``efs.vpc_id`` does not create
    the SG in the default VPC while the subnets live in another VPC, which fails
    ``create_mount_target``); else the region's default VPC. Subnets are left
    as-authored when given, else auto-discovered and deduped to one per AZ (EFS
    allows a single mount target per AZ per filesystem). Returns every associated
    VPC CIDR for the ingress rule."""
    if vpc_id:
        vpc = ec2.describe_vpcs(VpcIds=[vpc_id])["Vpcs"][0]
    elif subnets:
        sn = ec2.describe_subnets(SubnetIds=list(subnets))["Subnets"]
        if not sn:
            raise RuntimeError(f"none of the subnets {list(subnets)} were found")
        vpc_id = sn[0]["VpcId"]
        vpc = ec2.describe_vpcs(VpcIds=[vpc_id])["Vpcs"][0]
    else:
        vpcs = ec2.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])[
            "Vpcs"
        ]
        if not vpcs:
            raise RuntimeError(f"no default VPC in {region}; set efs.vpc_id")
        vpc = vpcs[0]
        vpc_id = vpc["VpcId"]

    cidrs = _vpc_cidrs(vpc)

    if not subnets:
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
    return vpc_id, cidrs, list(subnets)


def _create_or_adopt_sg(ec2, tags, vpc_id, mount_point) -> Tuple[str, bool]:
    """Create (or adopt a prior run's leaked) NFS security group for this mount.

    The name folds a short ``mount_point`` hash into the target-run id so sibling
    ephemeral mounts (which share ``tags``) get distinct groups; it is stable
    across retries of the same mount, so on ``InvalidGroup.Duplicate`` the
    existing group is adopted instead of wedging the retry. Returns
    ``(sg_id, created_sg)``; ``created_sg`` is True whenever teardown should reap
    it (created or adopted). The caller opens NFS ingress separately (idempotent
    on both paths)."""
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
    except Exception as e:  # noqa: BLE001 - adopt a leaked duplicate; re-raise else
        if not _is_duplicate_sg(e):
            raise
        return _find_sg_id(ec2, group_name, vpc_id), True
    return sg_id, True


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
    :param vpc_id: VPC to place the filesystem in; derived from ``subnets`` when
        unset, else the region's default VPC.
    :param subnets: explicit subnet ids for the mount targets, left as-authored
        (and used to derive ``vpc_id`` when it is unset); when unset, subnets are
        auto-discovered and deduped to one per AZ.
    :param security_group_id: a BYO security group to reuse; when unset one is
        created (and adopted if a prior run left an identically-named one behind).
    :param mount_point: this mount's mount_point, folded into the created SG name
        so sibling ephemeral mounts (which share ``tags``) get distinct groups.
    :returns: a :class:`ProvisionedResources` describing the created filesystem,
        mount targets, and security group.
    """
    efs = session.client("efs", region_name=region)
    ec2 = session.client("ec2", region_name=region)

    vpc_id, vpc_cidrs, subnets = _resolve_network(ec2, region, vpc_id, subnets)

    created_sg = False
    sg_id = security_group_id
    fsid = ""
    mt_ids: List[str] = []
    # Everything below creates real AWS infra. Any failure part-way through must
    # not leak (issue #391): on error, best-effort tear down whatever we already
    # created (reusing deprovision_efs) before re-raising the original error.
    try:
        if not sg_id:
            sg_id, created_sg = _create_or_adopt_sg(ec2, tags, vpc_id, mount_point)
            # Open NFS ingress on both the created and the adopted SG (idempotent;
            # all VPC CIDRs): a leaked SG from a run that crashed before authorize
            # has no ingress, so re-authorizing on adopt lets the next run
            # self-heal. A BYO security_group_id is the operator's to configure.
            _ensure_nfs_ingress(ec2, sg_id, vpc_cidrs)

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
