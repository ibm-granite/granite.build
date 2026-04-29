#!/usr/bin/env python3

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

# from .artifact import ArtifactStoreType, ArtifactType
# from .resources import ResourceSpec, ResourceTypeimport asyncio

"""
Run the build-runner as a K8s Job
"""

import asyncio
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Self, Tuple, Union

import yaml
from kubernetes_asyncio import client
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from gbcommon.types.testing import (
    ENV_VAR_GBTEST_MOCK_HF_CALLS,
    ENV_VAR_GBTEST_SIMULATE_FAILURE_SCENARIO,
    GBTEST_MOCK_HF,
    GBTEST_SIMULATE_FAILURE_SCENARIO,
)
from gbserver.buildwatcher.abstractbuildrunner import AbstractBuildRunner
from gbserver.buildwatcher.build_utils import finalize_build_status
from gbserver.environment.k8s import AtomicApiClient
from gbserver.storage.singleton_storage import get_admin_storage
from gbserver.storage.stored_build import StoredBuild
from gbserver.types.constants import (
    BUILDRUNNERJOB_BUILD_WORKSPACE_PVC_NAME,
    BUILDRUNNERJOB_CONFIGMAP_NAME,
    BUILDRUNNERJOB_IMAGE_OVERRIDE,
    BUILDRUNNERJOB_NAMESPACE,
    BUILDRUNNERJOB_SECRET_NAME,
    BUILDRUNNERJOB_SLEEP_ON_END,
    DEFAULT_GH_API_ENDPOINT,
    DEFAULT_ROOT_WORKSPACE_DIR,
    ENV_VAR_GBSERVER_K8S_USE_ASPERA,
    ENV_VAR_GBSERVER_LSF_USE_ASPERA,
    GB_ENVIRONMENT_CONFIG,
    GBSERVER_GBSERVER_IMAGE_TAG,
    GBSERVER_GITHUB_TOKEN,
    GBSERVER_METRICS_AUTH_TOKEN,
    GBSERVER_METRICS_ENDPOINT,
    K8S_USE_ASPERA,
    LSF_USE_ASPERA,
)
from gbserver.types.status import Status
from gbserver.utils.logger import get_logger
from gbserver.utils.template import fill_template

logger = get_logger(__name__)


def _is_retryable_k8s_exception(exc: BaseException) -> bool:
    """Return True if the exception is retryable (not a 404 Not Found)."""
    if isinstance(exc, client.ApiException):
        # Don't retry on 404 - the job no longer exists
        if exc.status == 404:
            return False
        # Retry on other API errors (5xx, timeouts, etc.)
        return True
    # Retry on other transient exceptions
    return isinstance(exc, (asyncio.TimeoutError, ConnectionError, OSError))


class BuildRunnerJob(AbstractBuildRunner):
    """
    This implementation of AbstractBuildRunner starts an BuildRunner in a K8s Job pod.
    """

    def __init__(
        self: Self,
        build: StoredBuild,
        gh_token: str = GBSERVER_GITHUB_TOKEN,
        workspace_dir: Union[str, Path] = DEFAULT_ROOT_WORKSPACE_DIR,
        monitoring_interval: int = 5,
        gh_api_endpoint: str = DEFAULT_GH_API_ENDPOINT,
    ) -> None:
        super().__init__(
            build=build,
            gh_token=gh_token,
            workspace_dir=workspace_dir,
            monitoring_interval=monitoring_interval,
            gh_api_endpoint=gh_api_endpoint,
        )
        self.namespace = BUILDRUNNERJOB_NAMESPACE
        self.is_stop_requested = False
        self.is_running = False
        curr_env = GB_ENVIRONMENT_CONFIG
        # The exact image tag with commit will be specified in env var in the container
        self.override_image = BUILDRUNNERJOB_IMAGE_OVERRIDE
        assert self.override_image != "", "invalid build-runner image"
        image_namespace = f"/gb-{curr_env.env.lower()}/"
        self.override_image = self.override_image.replace("/gb-prod/", image_namespace)
        if GBSERVER_GBSERVER_IMAGE_TAG is not None:
            image_without_tag, _ = self.override_image.split(":")
            self.override_image = image_without_tag + ":" + GBSERVER_GBSERVER_IMAGE_TAG
        logger.info("self.override_image: %s", self.override_image)
        # Load the deployment yaml template
        deployment_path = Path(curr_env.buildwatcher_deployment_yaml).resolve()
        assert deployment_path.is_file(), f"expected '{deployment_path}' to be a file"
        with open(deployment_path, "r", encoding="utf-8") as f:
            self.deployment_yaml_template = f.read()
        logger.info("self.deployment_yaml_template: %s", self.deployment_yaml_template)
        # Data common across builds, used to fill the template
        build_runner_extra_env_vars = {
            ENV_VAR_GBSERVER_K8S_USE_ASPERA: K8S_USE_ASPERA,
            ENV_VAR_GBSERVER_LSF_USE_ASPERA: LSF_USE_ASPERA,
        }

        simulate_scenario = GBTEST_SIMULATE_FAILURE_SCENARIO
        if simulate_scenario:
            build_runner_extra_env_vars[ENV_VAR_GBTEST_SIMULATE_FAILURE_SCENARIO] = (
                GBTEST_SIMULATE_FAILURE_SCENARIO
            )

        mock_hf = GBTEST_MOCK_HF
        if mock_hf:
            build_runner_extra_env_vars[ENV_VAR_GBTEST_MOCK_HF_CALLS] = mock_hf

        self.build_runner_data = {
            "build_runner_name": "",
            "build_runner_labels": {},
            "build_runner_annotations": {},
            "build_runner_command": [],
            "build_runner_namespace": self.namespace,
            "build_runner_image": self.override_image,
            "main_secret_name": BUILDRUNNERJOB_SECRET_NAME,
            "is_metrics_enabled": GBSERVER_METRICS_ENDPOINT != "",
            "is_metrics_auth_enabled": GBSERVER_METRICS_AUTH_TOKEN != "",
            "build_workspace_pvc_name": BUILDRUNNERJOB_BUILD_WORKSPACE_PVC_NAME,
            "configmap_name": BUILDRUNNERJOB_CONFIGMAP_NAME,
            "dev_staging_prod_env": curr_env.env,
            "build_runner_extra_env_vars": build_runner_extra_env_vars,
        }
        logger.info("self.build_runner_data: %s", self.build_runner_data)

    def stop(self: Self) -> None:
        """
        Stop the build that was started using start_and_wait().
        Upon returning, this must also cause the call to start_and_wait() to return.
        """
        if self.is_running:
            self.is_stop_requested = True
            # while self.is_running:
            #     time.sleep(1)
            # self.is_stop_requested = False

    def start_and_wait(self: Self) -> None:
        """
        Start job/pod running the BuildRunner using the gbserver build-runner CLI.
        The following should be passed to the CLI either as command line options or env vars :
            1) build id (stored in build storage.) (cli option)
            2) gh_token (cli option or GBSERVER_GITHUB_TOKEN env var)
            3) workspace_dir (cli option)
            4) monitoring interval (cli option)
            5) gh_api_endpoint (cli option)
        Returns after the build has completed/failed/cancelled or stop() has been called from another thread.
        """
        self.is_stop_requested = False
        self.is_running = True
        item = self.storage.build_storage.get_by_uuid(self.stored_build.uuid)
        if item is None:
            self.storage.build_storage.add(self.stored_build)
        try:
            asyncio.run(self.__start_job())
        except Exception as e:
            logger.error(
                "BuildRunnerJob.__start_job failed for build %s: %s",
                self.stored_build.uuid,
                e,
            )
            finalize_build_status(
                self.stored_build.uuid, Status.FAILED, failure_reason=str(e)
            )
        self.is_running = False

    def __get_command_to_run(
        self: Self, sleep_on_end: bool = BUILDRUNNERJOB_SLEEP_ON_END
    ) -> List[str]:
        # TODO: need the dns workaround? See buildwatcher yaml
        command = ["gbserver"]
        if self.storage.table_name_prefix:
            # Needed to support testing which uses prefixes
            command.extend(
                ["--gb-admin-table-prefix", f"{self.storage.table_name_prefix}"]
            )
        command.extend(
            [
                "build-runner",
                "--build-id",
                f"{self.stored_build.uuid}",
                "--monitoring-interval",
                f"{self.monitoring_interval}",
                "--create-pr",
            ]
        )
        if self.gh_api_endpoint:
            command.extend(["--gh-api-endpoint", self.gh_api_endpoint])
        if self.gh_token:
            logger.info("we will pass the GH token via GBSERVER_GITHUB_TOKEN env var")
            # command.extend(["--gh-token", self.gh_token])
        if sleep_on_end:
            logger.warning("sleep_on_end: %s", sleep_on_end)
            command_prefix = " ".join(command)
            command = ["bash", "-c", command_prefix + " ; tail -f /dev/null"]
        return command

    def __get_build_runner_yaml(
        self: Self,
        name: str,
        command: List[str],
        labels: Dict[str, str],
        annotations: Dict[str, str],
    ) -> Dict:
        """Get the build-runner yaml as a dict"""
        build_runner_data = self.build_runner_data.copy()
        build_runner_data["build_runner_name"] = name
        build_runner_data["build_runner_labels"] = labels
        build_runner_data["build_runner_annotations"] = annotations
        build_runner_data["build_runner_command"] = command
        logger.info("build_runner_data: %s", build_runner_data)
        filled_deployment_yaml = fill_template(
            templ=self.deployment_yaml_template, data=build_runner_data, strict=True
        )
        logger.info("filled_deployment_yaml: %s", filled_deployment_yaml)
        deployment_yaml = yaml.safe_load(filled_deployment_yaml)
        assert isinstance(
            deployment_yaml, dict
        ), f"invalid deployment_yaml: {deployment_yaml}"
        return deployment_yaml

    def __get_batchv1job_body(self: Self) -> Tuple[str, client.V1Job]:
        command = self.__get_command_to_run()
        logger.info("command: %s", command)
        # Create some dynamic metadata
        stored_build = self.stored_build
        build_id = stored_build.uuid or "no_build_id"
        source_uri = stored_build.source_uri or ""
        labels = {
            "granite-dot-build/build-id": build_id,
            "granite-dot-build/username": stored_build.username or "no_username",
            "granite-dot-build/spacename": stored_build.space_name or "no_spacename",
        }
        annotations = {"granite-dot-build/pr": source_uri}
        kube_job_name = f"gb-build-runner-{build_id}"
        deployment_yaml = self.__get_build_runner_yaml(
            name=kube_job_name,
            command=command,
            labels=labels,
            annotations=annotations,
        )
        return kube_job_name, deployment_yaml

    @retry(
        retry=retry_if_exception(_is_retryable_k8s_exception),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def __read_namespaced_job_with_retry(
        self: Self, batchv1: client.BatchV1Api, kube_job_name: str
    ) -> client.V1Job:
        """Read job status with retry for transient errors."""
        return await batchv1.read_namespaced_job(name=kube_job_name, namespace=self.namespace)  # type: ignore

    @retry(
        retry=retry_if_exception(_is_retryable_k8s_exception),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def __create_namespaced_job_with_retry(
        self: Self, batchv1: client.BatchV1Api, body: client.V1Job
    ) -> client.V1Job:
        return await batchv1.create_namespaced_job(namespace=self.namespace, body=body)  # type: ignore

    async def __start_job(self: Self) -> None:
        build_id = self.stored_build.uuid
        kube_job_name, body = self.__get_batchv1job_body()
        logger.info("kube_job_name: %s body: %s", kube_job_name, yaml.safe_dump(body))
        async with await AtomicApiClient.create_api_client(
            kube_config_string=None, kube_context=None
        ) as api:
            batchv1 = client.BatchV1Api(api)
            # exit()
            job_had_exception = False
            job_exception = None
            try:
                logger.info("Launching job: %s", kube_job_name)
                resp = await self.__create_namespaced_job_with_retry(batchv1, body=body)  # type: ignore
                logger.info("Job created: %s", kube_job_name)
                while not self.is_stop_requested:
                    try:
                        job = await self.__read_namespaced_job_with_retry(
                            batchv1, kube_job_name
                        )
                        # logger.info("Job  %s", str(job))
                    except Exception as inner_exc:
                        job_had_exception = True
                        job_exception = inner_exc
                        break  # Assuming this means the job is no longer present, exception or success
                    delay = self.monitoring_interval * (0.95 + random.random() / 10.0)
                    await asyncio.sleep(delay)
                logger.info(
                    "Done waiting for job completion on pod with name %s", kube_job_name
                )
            except Exception as e:
                logger.error(
                    "Got exception creating/monitoring job %s: %s",
                    kube_job_name,
                    e,
                )
                await self.__delete_job_and_pod_with_retry(batchv1, kube_job_name)
                raise e
            if self.is_stop_requested:  # Takes priority over an exception
                # If stopped then make sure the build is marked CANCELLED, unless already finished
                logger.info("Marking build %s as cancel requested.", build_id)
                update_if = lambda item: (
                    (not item.status.is_finished())
                    and (item.status != Status.CANCEL_REQUESTED)
                )
                get_admin_storage().build_storage.update_fields(
                    build_id,
                    fields={"status": Status.CANCEL_REQUESTED},
                    should_update=update_if,
                )
            elif job_had_exception:
                # Pod may have died, so see if the job is still RUNNING and if so, consider it FAILED.
                # If not finished, then the job/pod exited abnormally (with exception above)
                msg = f"Build {build_id} failed due to job exception: {job_exception}"
                logger.error("%s", msg)
                finalize_build_status(build_id, Status.FAILED, failure_reason=msg)
                # We have seen that when the pod creation fails, the job can be left running (a zombie).  In this case the
                # StoredBuild stays as PENDING and subsequent restarts of the watcher try to start the PENDING builds again,
                # but fail due to a job name collision with the zombie job.
                await self.__delete_job_and_pod_with_retry(batchv1, kube_job_name)
            # else the job finished normally.

    @retry(
        retry=retry_if_exception(_is_retryable_k8s_exception),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def __delete_namespaced_job_with_retry(
        self: Self, batchv1: client.BatchV1Api, job_name: str
    ) -> Any:
        resp = await batchv1.delete_namespaced_job(name=job_name, namespace=self.namespace)  # type: ignore
        return resp

    @retry(
        retry=retry_if_exception(_is_retryable_k8s_exception),
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def __delete_namespaced_pod_with_retry(self: Self, pod_name: str) -> None:
        v1 = client.CoreV1Api()
        v1.delete_namespaced_pod(
            name=pod_name,
            namespace=self.namespace,
            body=client.V1DeleteOptions(
                propagation_policy="Foreground", grace_period_seconds=5
            ),
            async_req=True,
        )

    async def __delete_job_and_pod_with_retry(
        self: Self, batchv1: client.BatchV1Api, kube_job_name: str
    ) -> None:
        try:
            logger.info("Deleting job %s", kube_job_name)
            resp = await self.__delete_namespaced_job_with_retry(batchv1, kube_job_name)
            logger.info("Job deleted: %s", resp)
        except Exception as e:  # Often a 404, not found
            logger.info(
                "Ignoring exception encountered while deleting job %s: %s",
                kube_job_name,
                e,
            )
        # It seems deleting the job can still leave the pod
        try:
            logger.info("Begin deleting pod %s", kube_job_name)
            self.__delete_namespaced_pod_with_retry(kube_job_name)
            logger.info("Done deleting pod %s", kube_job_name)
        except BaseException as e:  # Often a 404, not found
            logger.info(
                "Ignoring exception encountered while deleting pod %s: %s",
                kube_job_name,
                e,
            )
