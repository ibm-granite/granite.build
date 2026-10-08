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

"""Real-AWS no-leak helpers for the gated ephemeral-EFS integration test (#391).

gbserver auto-provisions an ephemeral EFS (filesystem + a mount target per
subnet + an NFS security group) at target-run setup and destroys it at teardown,
tagging everything ``gb-ephemeral=true``. The integration test brackets a real
build with these helpers to prove teardown left nothing behind: snapshot the
tagged filesystems/security groups before the build, and assert no NEW ones
survive after (a before/after diff, robust because the AWS build tests are
serialized on one xdist group).

The helpers take an injected boto3-like ``session`` (``.client("efs"|"ec2",
region_name=...)``) so they are unit-tested with a fake session and this module
imports without boto3 installed. The reap path reuses the same, unit-tested
``deprovision_efs`` gbserver uses at teardown.
"""

from typing import Set, Tuple

from gbserver.environment.shared_fs.base import ProvisionedResources
from gbserver.environment.shared_fs.efs_provisioning import deprovision_efs

_EPHEMERAL_TAG_KEY = "gb-ephemeral"
_EPHEMERAL_TAG_VALUE = "true"
# EFS delete is eventually consistent: a just-deleted filesystem is still returned
# by describe_file_systems for a while in a terminal lifecycle state. Teardown
# having *issued* the delete is enough -- treat these as gone so a correctly
# deprovisioned run is not mis-read as a leak.
_DEAD_LIFECYCLE = {"deleting", "deleted"}


def _is_not_found(exc: Exception) -> bool:
    """True for a boto3/botocore 'resource does not exist' error (e.g. the
    filesystem finished deleting between detection and reap)."""
    return "NotFound" in type(exc).__name__ or "does not exist" in str(exc)


def list_gb_ephemeral(session, region: str) -> Tuple[Set[str], Set[str]]:
    """Return ``(filesystem_ids, security_group_ids)`` tagged gb-ephemeral=true in
    ``region``, excluding filesystems already in a terminal (deleting/deleted)
    lifecycle state.

    EFS ``describe_file_systems`` has no server-side tag filter, so its inline
    ``Tags`` are filtered client-side (paginated via ``Marker``/``NextMarker``);
    EC2 pushes the tag filter server-side (paginated via ``NextToken``).
    """
    efs = session.client("efs", region_name=region)
    ec2 = session.client("ec2", region_name=region)

    fsids: Set[str] = set()
    marker = None
    while True:
        resp = efs.describe_file_systems(**({"Marker": marker} if marker else {}))
        for fs in resp.get("FileSystems", []):
            if fs.get("LifeCycleState") in _DEAD_LIFECYCLE:
                continue
            tags = {t["Key"]: t["Value"] for t in fs.get("Tags", [])}
            if tags.get(_EPHEMERAL_TAG_KEY) == _EPHEMERAL_TAG_VALUE:
                fsids.add(fs["FileSystemId"])
        marker = resp.get("NextMarker")
        if not marker:
            break

    sgids: Set[str] = set()
    token = None
    while True:
        kwargs = {
            "Filters": [
                {"Name": f"tag:{_EPHEMERAL_TAG_KEY}", "Values": [_EPHEMERAL_TAG_VALUE]}
            ]
        }
        if token:
            kwargs["NextToken"] = token
        resp = ec2.describe_security_groups(**kwargs)
        for sg in resp.get("SecurityGroups", []):
            sgids.add(sg["GroupId"])
        token = resp.get("NextToken")
        if not token:
            break

    return fsids, sgids


def reap_leaked(session, region: str, fsids: Set[str], sgids: Set[str]):
    """Best-effort delete leaked ephemeral resources; return a list of failures.

    Filesystems are torn down first (via the same ``deprovision_efs`` gbserver
    uses -- mount targets, wait, then filesystem), which frees the NFS security
    group's ENIs, then any leaked security group is deleted directly. Reusing
    ``deprovision_efs`` keeps this on the tested teardown path. ``created_sg`` is
    False here because leaked SGs are handled separately (a leaked FS may share an
    SG that is itself in ``sgids``).
    """
    failures = []
    efs = session.client("efs", region_name=region)
    for fsid in sorted(fsids):
        try:
            mt_ids = [
                m["MountTargetId"]
                for m in efs.describe_mount_targets(FileSystemId=fsid).get(
                    "MountTargets", []
                )
            ]
            provisioned = ProvisionedResources(
                region=region,
                file_system_id=fsid,
                dns_name="",
                mount_target_ids=mt_ids,
                subnet_ids=[],
                security_group_id=None,
                created_sg=False,
            )
            failures.extend(deprovision_efs(session, provisioned))
        except Exception as e:  # noqa: BLE001 - best-effort reap
            # The filesystem may have finished deleting between detection and reap;
            # that is not a leak. Any other error is recorded, not raised.
            if not _is_not_found(e):
                failures.append(f"reap {fsid}: {e}")

    ec2 = session.client("ec2", region_name=region)
    for sgid in sorted(sgids):
        try:
            ec2.delete_security_group(GroupId=sgid)
        except Exception as e:  # noqa: BLE001 - best-effort reap
            if not _is_not_found(e):
                failures.append(f"delete_security_group {sgid}: {e}")

    return failures
