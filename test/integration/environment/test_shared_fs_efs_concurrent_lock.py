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

"""Gated real-infra test: the hfpull cross-node ``os.mkdir`` lock over EFS (#378).

This validates the property the hfpull download lock relies on — that
``os.mkdir`` is **atomic AND coherent across nodes** on the shared filesystem
(``gbcommon.utils.fs_lock.SharedFileSystemLock``) — on a real EFS mount, across
SEPARATE EC2 instances. It is the concurrent-hfpull scenario that a networked FS
must support and that a SkyPilot S3-``MOUNT`` cannot (no cross-mount coherence).

Rather than submit a gbserver build (the build-submission harness in
``test_shared_fs_efs_e2e.py`` is still a scaffold), this drives the mechanism
directly and reliably: one ``sky launch --num-nodes N`` where every node mounts
the BYO EFS fixture and runs the SAME mkdir-lock protocol (a faithful, compact
mirror of the hfpull step's shell — acquire via ``mkdir`` under
``<dest>/.gb-hfpull-locks/``, download UNDER the lock, release). It then asserts
the observed serialization:

* exactly ONE node acquires first and downloads;
* the OTHER node blocks, reading the holder's identity across the mount
  (cross-node coherence);
* the waiter acquires only AFTER the holder releases, and its own download is a
  cache hit off the shared EFS (cross-node read-after-write);
* both nodes verify the model and the job SUCCEEDS.

It COSTS MONEY (provisions EC2) and MUST NEVER run in CI. It self-skips unless:

* ``GB_RUN_SHARED_FS_E2E=1``   -- the explicit opt-in gate (spins up EC2; $);
* ``GB_TEST_EFS_FS_ID``        -- a validated BYO EFS filesystem id (fs-...);
* ``GB_TEST_EFS_REGION``       -- that filesystem's region (e.g. us-east-1);
* AWS credentials in the environment (``aws_credentials_present()``).

Everything past the gate lives inside the test body, so a normal collection run
imports cleanly and reports SKIPPED without touching AWS.

Run it (opt-in, real AWS):

    GB_RUN_SHARED_FS_E2E=1 \\
    GB_TEST_EFS_FS_ID=fs-03bbdc96a5fdcb873 \\
    GB_TEST_EFS_REGION=us-east-1 \\
    AWS_PROFILE=gb-skypilot \\
    PYTEST_ADDOPTS=-s \\
    .venv/bin/python -m pytest \\
      test/integration/environment/test_shared_fs_efs_concurrent_lock.py -q

Confirm the gate skips in a normal run (what CI does):

    .venv/bin/python -m pytest \\
      test/integration/environment/test_shared_fs_efs_concurrent_lock.py -q
    # -> 1 skipped
"""

import os
import re
import shutil
import subprocess
import uuid

import pytest

_GATE_ENV = "GB_RUN_SHARED_FS_E2E"

pytestmark = [
    pytest.mark.skypilot_integration,
    pytest.mark.skipif(
        os.environ.get(_GATE_ENV) != "1",
        reason=(
            f"real-infra lock test: set {_GATE_ENV}=1 (spins up EC2; costs $). "
            "Also requires GB_TEST_EFS_FS_ID, GB_TEST_EFS_REGION and AWS creds."
        ),
    ),
]

_MOUNT_POINT = "/mnt/gb-shared"
# Two nodes is the minimal meaningful cross-node contention; bump for a heavier
# soak. t3.medium (4 GiB) avoids the Ray OOM that t3.small (2 GiB) hits when
# `hf download` runs under Ray.
_NUM_NODES = 2
_INSTANCE_TYPE = "t3.medium"
# A tiny public model keeps the EFS footprint and download time negligible; the
# HOLD_SECS sleep UNDER the lock is what deterministically creates the contention
# window so the waiter is guaranteed to observe the lock held.
_MODEL = "hf-internal-testing/tiny-random-gpt2"
_HOLD_SECS = 8

# Faithful compact mirror of the hfpull mkdir lock (SharedFileSystemLock): the
# download runs UNDER the lock and the holder sleeps HOLD_SECS so a waiter must
# block; waiters only proceed after the holder releases, at which point their own
# `hf download` is an idempotent cache hit off the shared EFS. Parametrised via
# env so the one committed script serves any fixture/model.
_LOCK_TEST_SH = r"""#!/usr/bin/env bash
set -u
FS_DNS="${FS_DNS:?FS_DNS required}"
MP="${MP:-/mnt/gb-shared}"
MODEL="${MODEL:?MODEL required}"
RUNTAG="${RUNTAG:?RUNTAG required}"
HOLD_SECS="${HOLD_SECS:-8}"
DEST="$MP/hf_cache/locktest-${RUNTAG}"
NODE="$(hostname)"
ID="host:${NODE}|pid:$$"
TS() { date -u +%H:%M:%S; }
echo "[$(TS)] NODE=$NODE START runtag=$RUNTAG"

command -v mount.nfs4 >/dev/null 2>&1 || { sudo apt-get update -qq && sudo apt-get install -y -qq nfs-common; } || true
if ! mountpoint -q "$MP"; then
  sudo mkdir -p "$MP"
  sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport "${FS_DNS}:/" "$MP" \
    || { echo "[$(TS)] NODE=$NODE EFS_MOUNT_FAILED"; exit 1; }
fi
echo "[$(TS)] NODE=$NODE MOUNTED efs (root $(stat -c %A "$MP"))"
command -v hf >/dev/null 2>&1 || pip install --no-cache-dir -q 'huggingface_hub[cli]' >/dev/null 2>&1 || true

CONTAINER="$(dirname "$DEST")/.gb-hfpull-locks"
LOCKDIR="$CONTAINER/$(basename "$DEST").lock"
mkdir -p "$(dirname "$DEST")" "$CONTAINER" 2>/dev/null || true

HELD=0
for i in $(seq 1 300); do
  if mkdir "$LOCKDIR" 2>/dev/null; then
    printf '%s\n%s\n' "$ID" "$(date +%s)" > "$LOCKDIR/lock.info"
    HELD=1; echo "[$(TS)] NODE=$NODE ACQUIRED (attempt $i)"; break
  fi
  echo "[$(TS)] NODE=$NODE WAITING attempt=$i holder=$(head -n1 "$LOCKDIR/lock.info" 2>/dev/null || echo '?')"
  sleep 2
done
[ "$HELD" = 1 ] || { echo "[$(TS)] NODE=$NODE NEVER_ACQUIRED"; exit 1; }

t0=$(date +%s)
if hf download "$MODEL" --local-dir "$DEST" >/tmp/dl.log 2>&1; then
  echo "[$(TS)] NODE=$NODE DOWNLOAD_OK elapsed=$(( $(date +%s)-t0 ))s"
else
  echo "[$(TS)] NODE=$NODE DOWNLOAD_FAILED"; tail -8 /tmp/dl.log
fi
# Hold the lock so the peer is guaranteed to observe contention, then release.
sleep "$HOLD_SECS"
if [ "$(head -n1 "$LOCKDIR/lock.info" 2>/dev/null)" = "$ID" ]; then
  rm -rf "$LOCKDIR" 2>/dev/null || true
  echo "[$(TS)] NODE=$NODE RELEASED"
fi
if [ -f "$DEST/config.json" ]; then
  echo "[$(TS)] NODE=$NODE VERIFY_OK files=$(ls "$DEST" | wc -l)"
else
  echo "[$(TS)] NODE=$NODE VERIFY_FAIL"
fi
echo "[$(TS)] NODE=$NODE DONE"
"""


def _require_efs_fixture() -> tuple[str, str]:
    fs_id = os.environ.get("GB_TEST_EFS_FS_ID")
    region = os.environ.get("GB_TEST_EFS_REGION")
    if not fs_id or not region:
        pytest.skip(
            "lock test opted in but GB_TEST_EFS_FS_ID / GB_TEST_EFS_REGION are "
            "not both set (need a validated BYO EFS fixture)."
        )
    return fs_id, region


def _require_aws_credentials() -> None:
    # Lazy import so a normal (skipped) collection never imports the skypilot env.
    from gbserver.environment.skypilot import (  # pylint: disable=import-outside-toplevel
        aws_credentials_present,
    )

    if not aws_credentials_present():
        pytest.skip(
            "AWS credentials not in environment "
            "(set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE)."
        )


def _sky() -> str:
    return shutil.which("sky") or "sky"


def _run(argv: list[str], timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, check=False
    )


def test_concurrent_hfpull_mkdir_lock_over_efs():
    """N nodes contend on the hfpull mkdir lock over EFS; assert clean serialization."""
    _require_aws_credentials()
    fs_id, region = _require_efs_fixture()

    fs_dns = f"{fs_id}.efs.{region}.amazonaws.com"
    runtag = uuid.uuid4().hex[:8]
    cluster = f"gb-efs-locktest-{runtag}"
    workdir = os.path.join("/tmp", f"gb-efs-locktest-{runtag}")
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, "lock_test.sh"), "w", encoding="utf-8") as fh:
        fh.write(_LOCK_TEST_SH)

    run_cmd = (
        f"FS_DNS={fs_dns} MP={_MOUNT_POINT} MODEL={_MODEL} "
        f"RUNTAG={runtag} HOLD_SECS={_HOLD_SECS} bash lock_test.sh"
    )
    try:
        proc = _run(
            [
                _sky(),
                "launch",
                "-c",
                cluster,
                "--num-nodes",
                str(_NUM_NODES),
                "--cloud",
                "aws",
                "--region",
                region,
                "--instance-type",
                _INSTANCE_TYPE,
                "-y",
                "--down",
                "--workdir",
                workdir,
                run_cmd,
            ],
            timeout=1800,
        )
        out = proc.stdout + proc.stderr

        # 1) The job as a whole succeeded on real infra.
        assert "status: SUCCEEDED" in out, f"job did not succeed:\n{out[-3000:]}"

        # 2) Exactly one node is the initial holder (acquired on attempt 1).
        first_holders = re.findall(r"NODE=(\S+) ACQUIRED \(attempt 1\)", out)
        assert (
            len(first_holders) == 1
        ), f"expected exactly one first-acquirer, got {first_holders}"
        holder = first_holders[0]

        # 3) At least one OTHER node blocked, reading the holder's identity across
        #    the EFS mount (this is the cross-node coherence the lock needs).
        waits = re.findall(
            rf"NODE=(\S+) WAITING attempt=\d+ holder=host:{re.escape(holder)}\|",
            out,
        )
        waiters = {w for w in waits if w != holder}
        assert waiters, (
            "no peer observed the lock held by the first acquirer "
            f"(holder={holder}); lock not coherent across nodes?\n{out[-3000:]}"
        )

        # 4) A waiter acquired only AFTER the holder released (serialization).
        lines = out.splitlines()
        released_idx = next(
            (i for i, ln in enumerate(lines) if f"NODE={holder} RELEASED" in ln), None
        )
        assert released_idx is not None, "holder never released the lock"
        waiter = next(iter(waiters))
        waiter_acq_idx = next(
            (i for i, ln in enumerate(lines) if f"NODE={waiter} ACQUIRED" in ln), None
        )
        assert (
            waiter_acq_idx is not None and waiter_acq_idx > released_idx
        ), "waiter acquired before the holder released -> lock did not serialize"

        # 5) Every node verified the fully-downloaded model on the shared FS.
        verified = set(re.findall(r"NODE=(\S+) VERIFY_OK", out))
        assert (
            len(verified) == _NUM_NODES
        ), f"expected {_NUM_NODES} VERIFY_OK, got {sorted(verified)}\n{out[-3000:]}"
    finally:
        # Best-effort: remove this run's tiny dir from EFS while a node is still
        # up, then always tear the cluster down (never leak EC2).
        try:
            _run(
                [
                    _sky(),
                    "exec",
                    cluster,
                    f"sudo rm -rf {_MOUNT_POINT}/hf_cache/locktest-{runtag} "
                    f"{_MOUNT_POINT}/hf_cache/.gb-hfpull-locks/locktest-{runtag}.lock",
                ],
                timeout=300,
            )
        except Exception:  # pylint: disable=broad-except
            pass
        _run([_sky(), "down", cluster, "-y"], timeout=600)
        shutil.rmtree(workdir, ignore_errors=True)
