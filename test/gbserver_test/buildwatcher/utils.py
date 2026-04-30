import json
import os
import signal
import subprocess
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from filelock import FileLock
from pydantic import BaseModel, Field


class ProcessLock:
    """A file-based lock that handles stale locks from abnormal process termination.

    This class wraps FileLock and maintains a separate PID file to track which process
    holds the lock. When acquiring the lock, it checks if the process that previously
    held the lock is still alive. If the process has terminated (abnormal termination),
    the stale lock files are removed before attempting to acquire.

    Args:
        lock_file: Path to the lock file.
        timeout: Maximum time to wait for the lock (in seconds). Default is -1 (wait forever).
    """

    def __init__(self, lock_file: str, timeout: float = -1):
        self._lock_file = lock_file
        self._pid_file = f"{lock_file}.pid"
        self._timeout = timeout
        self._file_lock = FileLock(lock_file, timeout=timeout)

    def _read_pid(self) -> Optional[int]:
        """Read the PID from the PID file if it exists."""
        try:
            if os.path.exists(self._pid_file):
                with open(self._pid_file, "r") as f:
                    return int(f.read().strip())
        except (ValueError, IOError):
            pass
        return None

    def _write_pid(self) -> None:
        """Write the current process ID to the PID file."""
        with open(self._pid_file, "w") as f:
            f.write(str(os.getpid()))

    def _is_process_alive(self, pid: int) -> bool:
        """Check if a process with the given PID is still running."""
        import psutil

        r = psutil.pid_exists(pid)
        return r

    def _cleanup_stale_lock(self) -> None:
        """Remove stale lock and PID files."""
        try:
            if os.path.exists(self._lock_file):
                os.remove(self._lock_file)
        except OSError:
            pass
        try:
            if os.path.exists(self._pid_file):
                os.remove(self._pid_file)
        except OSError:
            pass

    def acquire(self) -> None:
        """Acquire the lock, handling stale locks from dead processes."""
        pid = self._read_pid()
        our_pid = os.getpid()
        if pid is not None and pid != our_pid and not self._is_process_alive(pid):
            # Process that held the lock has terminated abnormally
            self._cleanup_stale_lock()
            # Recreate the FileLock since we removed the lock file
            self._file_lock = FileLock(self._lock_file, timeout=self._timeout, is_singleton=True)

        self._file_lock.acquire()
        self._write_pid()

    def release(self) -> None:
        """Release the lock."""
        self._file_lock.release()

    def __enter__(self) -> "ProcessLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


from gbserver_test.constants import (
    ENV_VAR_GBTEST_COMPUTE_CLUSTER_API_KEY,
    ENV_VAR_GBTEST_COMPUTE_CLUSTER_TOKEN,
    ENV_VAR_GBTEST_GB_CLUSTER_API_KEY,
    ENV_VAR_GBTEST_GB_CLUSTER_TOKEN,
    GBTEST_COMPUTE_CLUSTER_API_KEY,
    GBTEST_COMPUTE_CLUSTER_PROJECT,
    GBTEST_COMPUTE_CLUSTER_SERVER_URI,
    GBTEST_COMPUTE_CLUSTER_TOKEN,
    GBTEST_GB_CLUSTER_API_KEY,
    GBTEST_GB_CLUSTER_PROJECT,
    GBTEST_GB_CLUSTER_SERVER_URI,
    GBTEST_GB_CLUSTER_TOKEN,
    GBTEST_USER_NAME,
)
from gbserver_test.test_utils import is_pytest_running_parallel

from gbcommon.uri.utils import get_artifact_type
from gbserver.storage.artifact_registration import ArtifactRegistration
from gbserver.storage.artifact_registry import IArtifactRegistry
from gbserver.utils.logger import get_logger

logger = get_logger(__name__)

# Global lock for oc login operations to prevent kubeconfig corruption during parallel test execution
_OC_LOGIN_LOCK_FILE = "/tmp/gbtest_oc_login.lock"
_oc_login_lock = ProcessLock(_OC_LOGIN_LOCK_FILE, timeout=300)  # 5 minute timeout


def _oc_login(server_uri: str, project_name: str, **kwargs):
    """Perform oc login with file locking to prevent concurrent kubeconfig corruption.

    This function acquires a file lock before executing oc login to ensure that only
    one test at a time can modify the shared kubeconfig file at /root/.kube/config.
    This prevents YAML corruption when tests run in parallel.
    """
    __tracebackhide__ = True  # Hide the token on stack traces.
    logger.info(f"Waiting on oc login lock for {server_uri}/{project_name}")
    with _oc_login_lock:
        logger.info(f"Acquired oc login lock for {server_uri}/{project_name}")
        token = kwargs.get("token", None)
        cmd = ["oc", "login", "--server", server_uri, "-n", project_name]
        if token is None:
            api_key = kwargs.get("api_key", None)
            if api_key is None:
                assert (
                    False
                ), f"Required Key 'token' not given. Please supply an token or api_key and then re-run"
            cmd.extend(["-u", "apikey", "-p", api_key])
        else:
            cmd.extend(["--token", token])
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            _, error_msg = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            assert False, f"oc login to {server_uri}/{project_name} timed out after 60s"
        # cmd = " ".join(cmd)
        if proc.returncode != 0:
            assert (
                False
            ), f"Could not oc login to {server_uri} and/or project {project_name}. 'oc login' produced '{error_msg}'"

    logger.info(f"Released oc login lock for {server_uri}/{project_name}")


class OcGetPodsPodStatusConditionsResponse(BaseModel):
    lastProbeTime: Optional[datetime] = None
    lastTransitionTime: Optional[datetime] = None
    status: str = ""  # usually boolean as a string e.g. True
    type: str = ""  # e.g. Initialized, Ready, ContainersReady, PodScheduled, etc.


class OcGetPodsPodStatusResponse(BaseModel):
    phase: str = ""  # e.g. Running, etc.
    startTime: Optional[datetime] = None
    conditions: List[OcGetPodsPodStatusConditionsResponse] = Field(default_factory=list)


class OcGetPodsPodResponse(BaseModel):
    kind: str = "Pod"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    spec: Dict[str, Any] = Field(default_factory=dict)
    status: Optional[OcGetPodsPodStatusResponse] = None


class OcGetPodsResponse(BaseModel):
    kind: str = "List"
    items: List[OcGetPodsPodResponse] = Field(default_factory=list)


def is_buildrunner_pod_finished(build_id: str) -> bool:
    pipe = f"oc get pods | grep {build_id}"
    pipe = f"oc get pods -l 'granite-dot-build/build-id={build_id}' -o json"
    cmd = ["bash", "-c", pipe]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error_msg = result.stderr
        assert False, f"Could not get the pods. {cmd} produced '{error_msg}'"
    pods_data = OcGetPodsResponse.model_validate_json(result.stdout)
    if len(pods_data.items) == 0:
        # logger.info("no pod found for build id %s , assuming finished", build_id)
        return True
    assert (
        len(pods_data.items) == 1
    ), f"more than one pod found for build id {build_id}: {pods_data.items}"
    pod_data = pods_data.items[0]
    if pod_data.status is None:
        # logger.warning("the pod has not status, build id %s: %s", build_id, pod_data)
        return False
    if pod_data.status.phase == "Terminating":
        # logger.info("pod is terminating: %s", pod_data)
        return True
    if pod_data.status.phase == "Running":
        # logger.info("pod is still running: %s", pod_data)
        return False
    logger.warning("pod in unsupported status: %s", pod_data)
    return False


def _get_buildrunner_job_name(build_id: str) -> Optional[str]:
    """Get the job name associated with the given build id

    Args:
        build_id (str): _description_

    Returns:
        Optional[str]: None on error and a message was issued, "" if no matching job found, else the name of the job.
    """
    pipe = f"oc get jobs | grep {build_id} | " + " awk '{print $1}'"
    cmd = ["bash", "-c", pipe]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error_msg = result.stderr
        logger.error(f"Could not get pod name for build {build_id}. {cmd} produced '{error_msg}'")
        return None
    else:
        return result.stdout.replace("\n", "")


def delete_buildrunner_pod(build_id: str) -> bool:
    job_name = _get_buildrunner_job_name(build_id)
    if not job_name:
        return False  # job_name was None or "", job not found
    cmd = (
        f"oc delete job {job_name}"  # buildrunners are run as job so kill that instead of the pod.
    )
    cmd = ["bash", "-c", cmd]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        error_msg = result.stderr
        logger.error(f"Could not delete buildrunner job {job_name}. {cmd} produced '{error_msg}'")
        return False
    else:
        return True


def _set_oc_project(server_uri: str, project_name: str) -> bool:
    """Assume the user is already oc logged into the host, and set to use the given project.
    Uses file locking to prevent concurrent kubeconfig modifications.

    Args:
        server_uri (str): _description_
        project_name (str): _description_
        do_assert (bool, optional): _description_. Defaults to False.

    Returns:
        bool: _description_
    """
    logger.info("Waiting for oc login lock for setting project")
    r = False
    with _oc_login_lock:
        logger.info("Acquired oc login lock for setting project")
        proc = subprocess.Popen(
            ["oc", "project", project_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, _ = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            logger.warning("oc project timed out after 30s")
            return False
        if proc.returncode == 0 and server_uri in stdout:
            r = True
    logger.info("Released oc login lock for setting project")
    return r


def cluster_logout():
    """Logout from the cluster with file locking to prevent concurrent kubeconfig modifications."""
    logger.info("Waiting for oc login lock for logout")
    with _oc_login_lock:
        logger.info("Acquired oc login lock for logout")
        proc = subprocess.Popen(
            ["oc", "logout"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        try:
            proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            logger.warning("oc logout timed out after 30s")
        logger.info("Released oc login lock for logout")


def gb_cluster_login() -> bool:
    """Try setting to the needed project (allowing user to manually login and need the env vars).
    If this fails, then Login and verify with gb cluster (RIS3) using api key or token from GBTEST_GB_CLUSTER_API_KEY or GBTEST_GB_CLUSTER_TOKEN env vars.
    Returns: True if already logged in.
    """
    server_uri = GBTEST_GB_CLUSTER_SERVER_URI
    project_name = GBTEST_GB_CLUSTER_PROJECT  # RIS3
    if not _set_oc_project(
        server_uri=server_uri, project_name=project_name
    ):  # Try this in case the user (running locally) is already logged in.
        assert (
            GBTEST_GB_CLUSTER_API_KEY or GBTEST_GB_CLUSTER_TOKEN
        ), f"One of {ENV_VAR_GBTEST_GB_CLUSTER_API_KEY} or {ENV_VAR_GBTEST_GB_CLUSTER_TOKEN} env vars must be set to oc login project {project_name} to {server_uri}"
        if GBTEST_GB_CLUSTER_API_KEY:
            _oc_login(
                server_uri=server_uri,
                project_name=project_name,
                api_key=GBTEST_GB_CLUSTER_API_KEY,
            )
        else:
            _oc_login(
                server_uri=server_uri,
                project_name=project_name,
                token=GBTEST_GB_CLUSTER_TOKEN,
            )
        already_logged_in = False
    else:
        already_logged_in = True
    return already_logged_in


def compute_cluster_login():
    """Try setting to the needed project (allowing user to manually login and need the env vars).
    If this fails, then Login and verify with compute cluster (Vela) using api key or token from GBTEST_COMPUTE_CLUSTER_API_KEY or GBTEST_COMPUTE_CLUSTER_TOKEN env vars.
    Returns: True if already logged in.
    """
    server_uri = GBTEST_COMPUTE_CLUSTER_SERVER_URI  # Vela, for example.
    project_name = GBTEST_COMPUTE_CLUSTER_PROJECT  # Vela
    if not _set_oc_project(
        server_uri=server_uri, project_name=project_name
    ):  # Try this in case the user (running locally) is already logged in.
        assert (
            GBTEST_COMPUTE_CLUSTER_API_KEY or GBTEST_COMPUTE_CLUSTER_TOKEN
        ), f"One of {ENV_VAR_GBTEST_COMPUTE_CLUSTER_API_KEY} or {ENV_VAR_GBTEST_COMPUTE_CLUSTER_TOKEN} env vars must be set to oc login to project {project_name} on {server_uri}"
        if GBTEST_COMPUTE_CLUSTER_API_KEY:
            _oc_login(
                server_uri=server_uri,
                project_name=project_name,
                api_key=GBTEST_COMPUTE_CLUSTER_API_KEY,
            )
        else:
            _oc_login(
                server_uri=server_uri,
                project_name=project_name,
                token=GBTEST_COMPUTE_CLUSTER_TOKEN,
            )
        already_logged_in = False
    else:
        already_logged_in = True
    return already_logged_in


def pre_register_input_artifacts(
    artifact_registry: IArtifactRegistry, space_name, artifact_uris: list[str]
):
    artifacts = []
    for input_artifact_uri in artifact_uris:
        # When running 1 build after another in the same test,
        # we need to NOT re-register inputs
        artifact = artifact_registry.get_by_uri(input_artifact_uri, space_name)
        if not artifact:
            type = get_artifact_type(input_artifact_uri)
            artifact = ArtifactRegistration(
                uri=input_artifact_uri,
                space_name=space_name,
                type=type,
                username=GBTEST_USER_NAME,
            )
            artifacts.append(artifact)
    if len(artifacts) > 0:
        artifact_registry.add(artifacts)


class ExceptionRaisingThread(threading.Thread):
    """
    This class allows us to use assert statements or to throw other exceptions in  a thread and have pytest report as a failure.
    Without this, pytest never sees the exception/error.
    In order to see such exceptions/errors, the join() method must be called on this instance/thread.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._exception = None

    def run(self):
        try:
            super(ExceptionRaisingThread, self).run()
        except Exception as e:
            self._exception = e

    def join(self, *args, **kwargs):
        super(ExceptionRaisingThread, self).join(*args, **kwargs)
        if self._exception is not None:
            raise self._exception
        # assert self._exception is None, f"Thread raised exception/assert failure {self._exception}"
