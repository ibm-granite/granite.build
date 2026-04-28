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

"""
The environment for K8s/Openshift clusters.
"""

import asyncio
import base64
import json
import multiprocessing
import os
import shlex
import shutil
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Self,
    Set,
    Tuple,
    Type,
    Union,
)

from gbserver.resilience.strategies.aspera_failure import AsperaRetryStrategy
from gbserver.resilience.strategies.nccl_error import NCCLErrorRetryStrategy
from gbserver.resilience.strategies.pod_eviction import PodEvictionRetryStrategy
from gbserver.resilience.strategies.unhealthy_insufficient_pods import (
    UnhealthyInsufficientPodsRetryStrategy,
)

if TYPE_CHECKING:
    from gbserver.resilience.node_health_tracker import NodeHealthTracker
    from gbserver.resilience.retry_handler import RetryStrategy

import aiohttp
import kubernetes_asyncio
import yaml
from kubernetes_asyncio import client, config, watch

from gbcommon.uri.cos import CosURI
from gbcommon.uri.hf import HfURI
from gbcommon.uri.lh import LhURI
from gbcommon.uri.uri import URI
from gbserver.asset.asset import Asset
from gbserver.asset.assetstore import Assetstore
from gbserver.asset.cosstore import Cosstore
from gbserver.asset.hfstore import Hfstore
from gbserver.asset.lhstore import Lhstore
from gbserver.environment.environment import (
    BINDING_KEY,
    Environment,
    EventLogLineParserConfig,
)
from gbserver.types.buildconfig import BuildTargetOutputConfig, BuildTargetStepConfig
from gbserver.types.buildevent import (
    EntityRunMetadata,
)
from gbserver.types.constants import (
    K8S_USE_ASPERA,
)
from gbcommon.types.testing import ENV_VAR_GBTEST_MOCK_HF_CALLS, GBTEST_MOCK_HF
from gbserver.types.environment.k8s import StepK8sConfig
from gbserver.types.environmentconfig import (
    ENVIRONMENT_FILENAME,
    EnvironmentConfig,
    StoreLoad,
    StorePush,
)
from gbserver.types.errors import LogMonitoringFailedException
from gbserver.utils.launch import launch_command_and_raise_errors
from gbserver.utils.logger import get_logger
from gbserver.utils.utils import short_alphanumeric_lower_hash
from gbserver.utils.utils_k8s import is_helm_v4_or_higher

logger = get_logger(__name__)

CHART_KEY = "chart"
HELM_RELEASE_NAME_PREFIX = "gb"


class AtomicApiClient:
    """A client to interact with the K8s API server"""

    _lock = multiprocessing.Lock()  # use multiprocessing.Lock for process safety
    _thread_local = threading.local()  # use thread local storage for thread safety

    @classmethod
    async def create_api_client(
        cls: Type[Self],
        kube_config_string: Optional[str] = None,
        kube_context: Optional[str] = None,
        ssl_verification: Optional[bool] = True,
    ):
        """Factory method to create a client from kubeconfig"""
        with cls._lock:
            try:
                if kube_config_string:
                    kube_config_dict = yaml.safe_load(kube_config_string)
                    logger.info("parsed the kube_config_string as yaml")
                    if kube_context:
                        await config.load_kube_config_from_dict(
                            config_dict=kube_config_dict, context=kube_context
                        )
                        logger.info(
                            "loaded the context %s from kube_config_dict", kube_context
                        )
                    else:
                        await config.load_kube_config_from_dict(
                            config_dict=kube_config_dict
                        )
                        logger.info("loaded the default context from kube_config_dict")
                else:
                    logger.info("no kube_config_string was provided")
                    if kube_context:
                        await config.load_kube_config(context=kube_context)
                        logger.info(
                            "loaded the context %s from the default K8s config location",
                            kube_context,
                        )
                    else:
                        await config.load_kube_config()
                        logger.info(
                            "loaded the default context from the default K8s config location"
                        )
                # cls._thread_local.current_config = config.list_kube_config_contexts()[1]
                cfg = kubernetes_asyncio.client.Configuration.get_default_copy()
                logger.info("Default SSL verification: %s", cfg.verify_ssl)
                cfg.verify_ssl = ssl_verification
                logger.info(
                    "SSL verification from environment.yaml: %s", cfg.verify_ssl
                )
                logger.info("creating the kubernetes_asyncio.client.ApiClient()")
                api_client = kubernetes_asyncio.client.ApiClient(configuration=cfg)
                return api_client
            except kubernetes_asyncio.config.ConfigException as e:
                logger.error("Error loading Kubernetes configuration: %s", e)
                raise e


class K8s(Environment):
    """A K8s/Openshift cluster environment."""

    def __init__(
        self: Self,
        event_q: asyncio.Queue,
        environment_config: Optional[EnvironmentConfig] = None,
        secrets: Optional[Dict] = None,
        node_health_tracker: Optional["NodeHealthTracker"] = None,
        **kwargs,
    ) -> None:
        self.launched_releases: Dict[str, str] = {}
        self.created_setup_secrets: Dict[str, str] = {}
        self.launch_params: Dict[str, Dict] = {}  # Store launch params for retry
        self.monitors: Dict[str, "AppWrapperMonitor"] = (
            {}
        )  # Store monitor instances for retry
        # Lazily retrieve the process-wide singleton if not explicitly passed
        if node_health_tracker is None:
            from gbserver.resilience import get_node_health_tracker

            node_health_tracker = get_node_health_tracker()
        self.node_health_tracker = node_health_tracker
        assert environment_config is not None, "environment_config is None"
        assert (
            environment_config.config is not None
        ), "environment_config.config is None"
        self.namespace: str = environment_config.config["namespace"]
        super().__init__(
            event_q=event_q,
            environment_config=environment_config,
            secrets=secrets,
            **kwargs,
        )
        dmf = {} if self.config is None else self.config.config.get("dmf", {})
        self.dmf_use_aspera = dmf.get("use_aspera", K8S_USE_ASPERA)
        self.seen_pods: Set[str] = set()
        self.kube_config: Optional[str] = None
        self.kube_context: Optional[str] = None
        self.ssl_verification: Optional[str] = True
        authentication = environment_config.config.get("authentication", {})
        assert isinstance(
            authentication, dict
        ), f"expected authentication to be dict, actual: {authentication}"
        kube_config_secret_name = authentication.get("kube_config")
        kube_context_secret_name = authentication.get("kube_context")
        ssl_verification = authentication.get("ssl_verification")
        logger.info(
            "using kube_config_secret_name: %s kube_context_secret_name: %s ssl_verification %s",
            kube_config_secret_name,
            kube_context_secret_name,
            ssl_verification,
        )
        if kube_config_secret_name is not None:
            assert self.secrets is not None, "self.secrets is None"
            self.kube_config = self.secrets.get(kube_config_secret_name)
            if self.kube_config:
                logger.info("loaded self.kube_config from secrets")
        if kube_context_secret_name is not None:
            assert self.secrets is not None, "self.secrets is None"
            self.kube_context = self.secrets.get(kube_context_secret_name)
            if self.kube_context:
                logger.info(
                    "loaded self.kube_context %s from secrets", self.kube_context
                )
        if ssl_verification is not None:
            assert self.ssl_verification is not None, "self.ssl_verification is None"
            self.ssl_verification = ssl_verification
            if self.ssl_verification:
                logger.info(
                    "loaded self.ssl_verification %s from ssl_verification",
                    self.ssl_verification,
                )
                logger.info(
                    "loaded self.ssl_verification %s from ssl_verification",
                    self.ssl_verification,
                )
        # read messaging config from environment config
        self.messaging_config: Optional[Dict[str, Any]] = environment_config.config.get(
            "messaging", {}
        )
        # read messaging secret and convert to dictionary
        assert self.secrets is not None, "self.secrets is None"
        self.messaging_secret = self.secrets.get(
            self.messaging_config.get("authentication_secret_name", "")
        )
        if self.messaging_secret is not None:
            try:
                self.messaging_secret = json.loads(self.messaging_secret)
            except Exception as ex:
                logger.warning(
                    "failed to parse %s: %s: %s",
                    self.messaging_secret,
                    type(ex).__name__,
                    ex,
                )
                self.messaging_secret = None

    def _get_step_env_config(self: Self, config: dict) -> StepK8sConfig:
        """Extract k8s config if present inside config."""
        k8s_dict = (config.get("k8s") or config.get("K8s")) if config else None
        if not k8s_dict:
            return StepK8sConfig()  # empty default
        return StepK8sConfig(**k8s_dict)

    def _get_k8s_labels_and_annotations(self: Self, kwargs: Dict) -> Tuple[Dict, Dict]:
        labels = kwargs.get("labels", {})
        assert isinstance(labels, dict)
        annotations = kwargs.get("annotations", {})
        assert isinstance(annotations, dict)
        runmetadata = kwargs.get("runmetadata")
        if runmetadata is None:
            return labels, annotations
        assert isinstance(runmetadata, EntityRunMetadata)
        if not runmetadata.build_id:
            logger.warning("build_id is not specified")
        labels["granite-dot-build/build-id"] = runmetadata.build_id or "no_build_id"
        labels["granite-dot-build/target-name"] = (
            runmetadata.target_name or "no_target_name"
        )
        labels["granite-dot-build/build-target-id"] = (
            runmetadata.targetrun_id or "no_targetrun_id"
        )
        labels["granite-dot-build/build-step-id"] = (
            runmetadata.targetsteprun_id or "no_targetsteprun_id"
        )
        # labels["granite-dot-build/username"] = ""
        kwargs["labels"] = labels
        if runmetadata.targetstep_uri:
            annotations["granite-dot-build/source-uri"] = runmetadata.targetstep_uri
        kwargs["annotations"] = annotations
        return labels, annotations

    async def setup_helm(
        self: Self,
        setup_id: str,
        space_secrets: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> Dict:
        if space_secrets is None:
            return {}
        setup_id_hash = short_alphanumeric_lower_hash(setup_id)
        async with await AtomicApiClient.create_api_client(
            kube_config_string=self.kube_config,
            kube_context=self.kube_context,
            ssl_verification=self.ssl_verification,
        ) as api:
            try:
                async with aiohttp.ClientSession() as session:
                    encoded_data = {
                        key: base64.b64encode(str(value).encode()).decode()
                        for key, value in space_secrets.items()
                    }
                    v1 = client.CoreV1Api(api)
                    labels, annotations = self._get_k8s_labels_and_annotations(kwargs)
                    retry_count = 0
                    max_tries = 100
                    secret = client.V1Secret(
                        api_version="v1",
                        kind="Secret",
                        metadata={
                            "name": setup_id_hash,
                            "namespace": self.namespace,
                            "labels": labels,
                            "annotations": annotations,
                        },
                        type="Opaque",
                        data=encoded_data,
                    )
                    logger.info(
                        "creating a secret %s in the namespace: %s",
                        setup_id_hash,
                        self.namespace,
                    )
                    while retry_count < max_tries:
                        retry_count += 1
                        try:
                            await v1.create_namespaced_secret(
                                namespace=self.namespace, body=secret
                            )
                            break
                        except kubernetes_asyncio.client.exceptions.ApiException as ae:
                            if retry_count >= max_tries:
                                raise
                            logger.warning(
                                "retrying because secret creation failed, error: %s", ae
                            )
                            await asyncio.sleep(1)
                        except aiohttp.client_exceptions.ServerDisconnectedError as se:
                            if retry_count >= max_tries:
                                raise
                            logger.warning(
                                "retrying because secret creation failed, error: %s", se
                            )
                            await asyncio.sleep(1)
                    self.created_setup_secrets[setup_id] = setup_id_hash
                    logger.info(
                        "Secret '%s' created successfully in namespace '%s'",
                        setup_id_hash,
                        self.namespace,
                    )
            except Exception as e:
                raise RuntimeError(
                    f"failed to create a secret {setup_id_hash} in the namespace: {self.namespace}"
                ) from e
        return {"space": {"secret": setup_id_hash}}

    async def teardown_helm(self: Self, setup_id: str):
        if setup_id not in self.created_setup_secrets:
            return
        setup_id_hash = self.created_setup_secrets[setup_id]
        async with await AtomicApiClient.create_api_client(
            kube_config_string=self.kube_config,
            kube_context=self.kube_context,
            ssl_verification=self.ssl_verification,
        ) as api:
            try:
                v1 = client.CoreV1Api(api)
                await v1.delete_namespaced_secret(
                    namespace=self.namespace, name=setup_id_hash
                )
                logger.info(
                    f"Secret '{setup_id_hash}' deleted in namespace '{self.namespace}'."
                )
            except Exception as e:
                logger.error(
                    "failed to delete the space secret (%s) in namespace %s, error: %s",
                    setup_id_hash,
                    self.namespace,
                    e,
                )

    # ------------------------------------ copy step folder to pod env utilities ------------------------------

    async def get_appwrapper_pod_list(
        self, v1: client.CoreV1Api, namespace: str, appwrapper_name: str
    ) -> List:
        """Return list of pods owned by this AppWrapper."""
        pod_names = []
        label_selector = f"workload.codeflare.dev/appwrapper={appwrapper_name}"
        try:
            pod_list = await v1.list_namespaced_pod(
                namespace=namespace, label_selector=label_selector
            )
            if not pod_list.items:
                logger.warning(
                    "No pods found with the specified label %s in namespace %s",
                    label_selector,
                    namespace,
                )
                return None
            pod_names = [pod.metadata.name for pod in pod_list.items]
        except client.ApiException as e:
            logger.error("Error fetching pods: %s", str(e))
            return None
        logger.info(
            "Found the pods %s for appwrapper %s", str(pod_names), appwrapper_name
        )
        return pod_names

    async def wait_for_pod_running(
        self, v1: client.CoreV1Api, pod_name: str, namespace: str, timeout: int = 600
    ):
        """Wait until a pod is in Running state."""
        start = time.time()
        while time.time() - start < timeout:
            pod = await v1.read_namespaced_pod(name=pod_name, namespace=namespace)
            phase = pod.status.phase
            if phase == "Pending":
                logger.warning(
                    "Pod %s still Pending to go to Running before copy can be done",
                    pod_name,
                )
            elif phase == "Running":
                logger.info("Pod %s is now Running", pod_name)
                return True
            elif phase == "Succeeded":
                logger.info("Pod %s is now completed !!", pod_name)
                return False
            logger.debug("Pod %s still in phase %s, waiting...", pod_name, phase)
            await asyncio.sleep(1)
        return False

    async def get_user_containers(
        self, v1: client.CoreV1Api, pod_name: str, namespace: str
    ) -> list[str]:
        """
        Fetch all user container names for a given pod.
        """
        try:
            pod_object = await v1.read_namespaced_pod(
                name=pod_name, namespace=namespace
            )
            container_names = [
                c.name
                for c in pod_object.spec.containers
                if "sidecar" not in c.name.lower()
            ]
            logger.info(
                "======= Detected user containers for pod ======== '%s': %s",
                pod_name,
                container_names,
            )
            return container_names
        except Exception as e:
            logger.error("Failed to fetch containers for pod %s: %s", pod_name, str(e))
            return []

    async def copy_merged_dir_to_pvc(
        self: Self,
        kube_config: str,
        kube_context: str,
        pod_name: str,
        base_targetsteprun_assets_dir: Path,
        container_name: str,
        namespace: str,
        merged_dir: str,
        launch_id: str,
        raise_error: bool = True,
    ) -> None:
        """
        Copy contents of merged step directory to the PVC mounted at
        /base_targetsteprun_assets_dir/llmb-targetsteprun-assets/launch_id in the
        given pod.
        Creates the target directory, copies to the directory and
        finally creates a marker file to signal completion.
        """
        pvc_mount_path = base_targetsteprun_assets_dir
        assets_dir = f"{pvc_mount_path}/llmb-targetsteprun-assets"
        target_dir = f"{assets_dir}/{launch_id}"

        kube_config_path = None
        if kube_config:
            with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                temp_file.write(kube_config)
                kube_config_path = temp_file.name

        base_cmd = ["kubectl"]  # default -> fall back case

        if kube_config_path:
            base_cmd += ["--kubeconfig", kube_config_path]
        if kube_context:
            base_cmd += ["--context", kube_context]

        # try kubectl and oc both
        async def run_with_fallback(cmd_args):
            """
            Run the command args with kubectl and on failure try with
            oc. If both fails, raises an error
            Returns: stdout, stdeerr and command executed
            """
            stderr_decoded = ""
            for tool in ["kubectl", "oc"]:
                cmd = [tool] + base_cmd[1:] + cmd_args
                command_executed = " ".join(cmd)
                logger.info(
                    f"====== Executing the following command: {command_executed} =====\n"
                )
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                except FileNotFoundError as e:
                    # kubectl does not exist at all → retry with oc
                    logger.warning(
                        f"{tool} is not installed or not found in PATH. "
                        f"Error: {str(e)} → Retrying using oc..."
                    )
                    # continue with oc
                    continue

                stdout, stderr = await proc.communicate()
                stderr_decoded = stderr.decode(encoding="utf-8") if stderr else ""

                if "cannot exec into a container in a completed pod" in stderr_decoded:
                    logger.warning(
                        "Pod already completed. Skipping exec/cp. Error: %s",
                        stderr_decoded.strip(),
                    )
                    return None, stderr, command_executed

                if proc.returncode == 0:
                    # if kubectl succeeds, it will return and not try oc
                    return stdout, stderr, command_executed
                else:
                    # if kubectl is there but still failed, then also try with oc
                    logger.warning(
                        f"{tool} failed with exit code {proc.returncode}. "
                        f"stderr: {stderr_decoded} "
                        f"Retrying using oc..."
                    )
            # if both dont succeed and fail
            logger.error("Both kubectl and oc failed -> Error: %s", stderr_decoded)
            raise RuntimeError(f"Both kubectl and oc failed. Error: {stderr_decoded}")

        # 1. Create directory hierarchy inside pod (if not exists)
        mkdir_args = [
            "exec",
            "-i",
            pod_name,
            "-n",
            namespace,
            "-c",
            container_name,
            "--",
            "/bin/sh",
            "-c",
            f"mkdir -p {target_dir}",
        ]
        try:
            _, stderr, mkdir_cmd = await run_with_fallback(mkdir_args)
            stderr_decoded = stderr.decode(encoding="utf-8") if stderr else ""
        except RuntimeError as e:
            if raise_error:
                raise ValueError(
                    f"failed to create directory '{target_dir}' inside pod '{pod_name}', error: {e}"
                )
            logger.error(
                "failed to create directory '%s' inside pod '%s', error: %s",
                target_dir,
                pod_name,
                e,
            )
            return

        logger.info(
            "Created directory %s inside pod %s at container %s",
            target_dir,
            pod_name,
            container_name,
        )

        # 2. Copy merged directory contents into that path
        cp_args = [
            "cp",
            f"{merged_dir}/.",  # copy only contents
            f"{namespace}/{pod_name}:{target_dir}",
        ]

        try:
            _, stderr, cp_cmd = await run_with_fallback(cp_args)
            stderr_decoded = stderr.decode(encoding="utf-8") if stderr else ""
        except RuntimeError as e:
            if raise_error:
                raise ValueError(
                    f"failed to copy the asset dir contents to pod '{pod_name}', error: {e}"
                )
            logger.error(
                "failed to copy the asset dir contents to pod '%s', error: %s",
                pod_name,
                e,
            )
            return

        logger.info(
            "Copied contents successfully to container %s in pod %s",
            container_name,
            pod_name,
        )

        # 3. Create .COMPLETE marker file to signal copy completion
        complete_file_path = f"{target_dir}/.COMPLETE"
        complete_args = [
            "exec",
            "-i",
            pod_name,
            "-n",
            namespace,
            "-c",
            container_name,
            "--",
            "/bin/sh",
            "-c",
            f"touch {complete_file_path}",
        ]
        try:
            _, stderr, complete_cmd = await run_with_fallback(complete_args)
            stderr_decoded = stderr.decode(encoding="utf-8") if stderr else ""
        except RuntimeError as e:
            if raise_error:
                raise ValueError(
                    f"failed to create the marker file in pod '{pod_name}', error: {e}"
                )
            logger.error(
                "failed to create the marker file in pod '%s', error: %s",
                pod_name,
                e,
            )
            return

        logger.info("Marker file created successfully at %s", target_dir)

    # --------------------------------------------------------------------------------------------

    async def launch_helm(
        self: Self,
        launch_id: str,
        targetsteprun_asset_dir: Optional[Path] = None,
        launcher_config: Optional[Dict] = None,
        config: Optional[Dict] = None,
        setup_config: Optional[Dict] = None,
        environment_config: dict = {},
        merged_dir_path: str = "/",
        **kwargs,
    ) -> None:
        """
        Installs a helm chart and keep track of release names
        """
        if launcher_config is None:
            launcher_config = {}
        if config is None:
            config = {}
        if setup_config is None:
            setup_config = {}
        build_step_k8s_config = self._get_step_env_config(config)
        pull_secrets = (
            build_step_k8s_config.secrets.secret_names_to_use_as_pull_secret or []
        )
        environment_variables = (
            build_step_k8s_config.secrets.secret_names_to_use_as_env_variable or []
        )

        release_name = (
            HELM_RELEASE_NAME_PREFIX + short_alphanumeric_lower_hash(launch_id).lower()
        )
        chart_path_str = str(targetsteprun_asset_dir / launcher_config[CHART_KEY])
        values_flags = ""
        values_default_file_path_str = str(
            targetsteprun_asset_dir / launcher_config[CHART_KEY] / "values-default.yaml"
        )
        if os.path.exists(values_default_file_path_str):
            values_flags += " -f " + values_default_file_path_str
        values_file_path_str = str(
            targetsteprun_asset_dir / launcher_config[CHART_KEY] / "values.yaml"
        )
        if os.path.exists(values_file_path_str):
            values_flags += " -f " + values_file_path_str
        values_config_file_path_str = str(
            targetsteprun_asset_dir / launcher_config[CHART_KEY] / "values-config.yaml"
        )
        if os.path.exists(values_config_file_path_str):
            values_flags += " -f " + values_config_file_path_str
        kube_config_path = None
        command_list: list[str] = []
        # extract build_id, targetrun_id, targetsteprun_id from run_metadata,
        # and pass them to sidecar monitoring as value overrides
        extra_runmetadata_values = []
        extra_runmetadata_values.append(("run_metadata.launch_id", launch_id))

        # Secrets to be used from cluster: k8s.envImagePullSecrets (from enviroment.yaml), k8s.imagePullSecrets (from build.yaml)
        # Secrets to be created in the cluster from Secrets Manager: k8s.userImagePullSecrets[ (from secret_names_to_use_as_pull_secret)

        # Temp files to store dockerconfigjson
        dockerconfig_files = []

        for idx, pull_secret_name in enumerate(pull_secrets):
            pull_secret_value = self.secrets.get(pull_secret_name)
            if pull_secret_value:
                # Name can still be set with --set
                extra_runmetadata_values.append(
                    (f"k8s.userImagePullSecrets[{idx}].name", pull_secret_name)
                )
                # Write dockerconfigjson to a temp file
                tmp_file = tempfile.NamedTemporaryFile(
                    delete=False, mode="w", suffix=".json"
                )
                tmp_file.write(pull_secret_value)
                tmp_file.close()
                dockerconfig_files.append((idx, tmp_file.name))

        # --- Environment Variables ---
        space_secret = setup_config.get("space", {}).get("secret")
        for env_var in environment_variables:
            if not env_var.env_name:
                continue
            secret_key = env_var.secret_name or env_var.env_name.lower()
            if not space_secret:
                raise ValueError("setup_config['space']['secret'] is missing")

            # Helm nested --set for secretKeyRef
            extra_runmetadata_values.append(
                (
                    f"k8s.env.{env_var.env_name}.valueFrom.secretKeyRef.name",
                    space_secret,
                )
            )
            extra_runmetadata_values.append(
                (f"k8s.env.{env_var.env_name}.valueFrom.secretKeyRef.key", secret_key)
            )

        # Propagate test mock flag into every helm-step pod when set.
        if GBTEST_MOCK_HF: 
            extra_runmetadata_values.append(
                (f"k8s.env.{ENV_VAR_GBTEST_MOCK_HF_CALLS}.value", "true")
            )

        # --- Build --set overrides ---
        extra_runmetadata_overrides = " ".join(
            f"--set {k}={v}" for k, v in extra_runmetadata_values
        )

        # Build --set-file overrides for dockerconfigjson
        dockerconfig_overrides = " ".join(
            [
                f"--set-file k8s.userImagePullSecrets[{idx}].dockerconfigjson={file_path}"
                for idx, file_path in dockerconfig_files
            ]
        )
        command_str = f"helm install {release_name} {values_flags} {extra_runmetadata_overrides} {dockerconfig_overrides} {chart_path_str}"

        try:
            if self.kube_config:
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                    temp_file.write(self.kube_config)
                    kube_config_path = temp_file.name
                command_str = f"{command_str} --kubeconfig {kube_config_path}"
            if self.kube_context:
                command_str = f"{command_str} --kube-context {self.kube_context}"
            if is_helm_v4_or_higher():
                # required to avoid server-side apply
                # which requires extra 'patch' verb/perms
                command_str += " --server-side=false"
            command_list = shlex.split(command_str)
            # ---------------
            formatted_cmd_str = " \\\n    ".join(command_list)
            msg = f"⚡ Running the command:\n```\n{formatted_cmd_str}\n```\n"
            self._send_message(msg=msg, **kwargs)
            # ---------------
            try:
                logger.info("trying a helm install dry run first...")
                dry_run_command_list = [*command_list, "--dry-run", "--debug"]
                await launch_command_and_raise_errors(
                    command_list=dry_run_command_list,
                    launch_id=launch_id + "-dry-run",
                )
            except Exception as e:
                raise ValueError("failed to do a helm install dry run") from e
            # ---------------
            retry_for_quota = 0
            while True:
                try:
                    # ---------------
                    process, stdout, stderr = await launch_command_and_raise_errors(
                        command_list=command_list,
                        launch_id=launch_id,
                    )
                    logger.debug("process: %s", process)
                    # ---------------
                    msg = f"⚡ Output of the `helm install` command:\n{stdout}\n{stderr}\n"
                    self._send_message(msg=msg, **kwargs)
                    break
                    # ---------------
                except Exception as e:
                    logger.error("helm install failed: %s", str(e))
                    # Handle (409) Reason: Conflict stderr: Error: INSTALLATION FAILED: create: failed to create: Operation cannot be fulfilled on resourcequotas "granite-buildquota": the object has been modified; please apply your changes to the latest version and try again
                    if "Operation cannot be fulfilled on resourcequotas" in str(e):
                        if retry_for_quota < 3:
                            retry_for_quota += 1
                            await asyncio.sleep(1)
                            logger.warning(
                                "Retrying: Operation cannot be fulfilled on resourcequotas retrycount=%d",
                                retry_for_quota,
                            )
                            continue
                    if len(command_list) >= 2 and command_list[1] == "install":
                        command_list = ["helm", "template", "--debug"] + command_list[
                            2:
                        ]
                        logger.info(
                            "we will try to 'helm template' instead to get more information: %s",
                            command_list,
                        )
                        process, stdout, stderr = await launch_command_and_raise_errors(
                            command_list=command_list,
                            launch_id=launch_id,
                        )
                        logger.debug("helm template process: %s", process)
                        # ---------------
                        msg = f"⚡ Output of the `helm template` command:\n{stdout}\n{stderr}\n"
                        self._send_message(msg=msg, **kwargs)
                        # ---------------
                    raise ValueError("helm install failed:") from e
        finally:
            self.launched_releases[launch_id] = release_name
            # Store launch params for potential retry
            self.launch_params[launch_id] = {
                "targetsteprun_asset_dir": targetsteprun_asset_dir,
                "launcher_config": launcher_config,
                "config": config,
                "setup_config": setup_config,
                "environment_config": environment_config,
                "merged_dir_path": merged_dir_path,
                **kwargs,
            }
            # If the step config does not contain this flag, we explicitly set it
            # to True to enable copy
            copy_to_step_dir = config.get("gb", {}).get("step_contents_in_env", True)
            if copy_to_step_dir:
                logger.warning(
                    "Flag to copy step directory to pod is set to %s. Initiating copy...",
                    copy_to_step_dir,
                )

                logger.warning(
                    "To disable copy explicitly,"
                    " please set `step_contents_in_env: false` in"
                    " either step.yaml or build.yaml under config.gb"
                )

                # Override the default targetsteprun assets dir path with /gb-read-write if not provided in the environment config.
                targetsteprun_assets_dir = environment_config.get(
                    "targetsteprun_assets_dir", "/gb-read-write"
                )
                if "targetsteprun_assets_dir" in environment_config:
                    logger.warning(
                        f"===== Using configured targetsteprun_assets_dir from environment.yaml: {targetsteprun_assets_dir}======="
                    )
                else:
                    logger.warning(
                        f"===== 'targetsteprun_assets_dir' not found in environment.yaml. Using default: {targetsteprun_assets_dir} to copy ======"
                    )

                merged_step_folder_path = merged_dir_path
                appwrapper_name = self.launched_releases[launch_id]
                namespace = self.config.config["namespace"]

                async with await AtomicApiClient.create_api_client(
                    kube_config_string=self.kube_config,
                    kube_context=self.kube_context,
                    ssl_verification=self.ssl_verification,
                ) as api:
                    v1 = client.CoreV1Api(api)
                    poll_interval = 5
                    state = "Not Started"
                    while True:
                        state = await self.get_appwrapper_status(api, appwrapper_name)
                        if state == "Running":
                            break
                        await asyncio.sleep(poll_interval)
                    logger.info(f"======= AppWrapper is now: {state} !!! =========")
                    pod_names = await self.get_appwrapper_pod_list(
                        v1, namespace, appwrapper_name
                    )
                    logger.info(f"======= AppWrapper Pods: {pod_names} =========")

                    logger.info(
                        "Final Merged Path contents %s to be copied"
                        "at mount path %s for "
                        "appwrapper %s in namespace %s and pod %s",
                        merged_step_folder_path,
                        targetsteprun_assets_dir,
                        appwrapper_name,
                        namespace,
                        str(pod_names[0]),
                    )

                    # edge case: do the copy only after the pod has started running
                    await self.wait_for_pod_running(v1, str(pod_names[0]), namespace)

                    # Fetch container names dynamically
                    container_names = await self.get_user_containers(
                        v1, str(pod_names[0]), namespace
                    )
                    for container_name in container_names:
                        await self.copy_merged_dir_to_pvc(
                            kube_config=self.kube_config,
                            kube_context=self.kube_context,
                            pod_name=str(pod_names[0]),
                            base_targetsteprun_assets_dir=targetsteprun_assets_dir,
                            container_name=container_name,
                            namespace=namespace,
                            merged_dir=merged_step_folder_path,
                            launch_id=launch_id,
                        )
            else:
                logger.warning(
                    f"Flag to copy step directory to pod is set to {copy_to_step_dir}. Skipping the copy..."
                )

            # Wait for the pod to be running (w/o exception) before enabling the monitors
            self._release_monitors(launch_id)

            # Cleanup temp files after helm install
            for _, file_path in dockerconfig_files:
                os.remove(file_path)
            if kube_config_path is not None:
                try:
                    os.unlink(kube_config_path)  # remove the temporary kube_config file
                except:
                    pass  # if the file was not created, or already removed, do nothing

    async def cleanup_helm(self: Self, launch_id: str, **kwargs) -> None:
        """
        Uninstalls a helm chart
        """
        if launch_id not in self.launched_releases:
            return
        release_name = self.launched_releases[launch_id]
        if release_name is None or release_name == "":
            return

        kube_config_path = None
        try:
            command_str = f"helm uninstall {release_name}"
            if self.kube_config:
                with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
                    temp_file.write(self.kube_config)
                    kube_config_path = temp_file.name
                command_str = f"{command_str} --kubeconfig {kube_config_path}"
            if self.kube_context:
                command_str = f"{command_str} --kube-context {self.kube_context}"
            command_list = shlex.split(command_str)
            process, _, _ = await launch_command_and_raise_errors(
                command_list=command_list,
                launch_id=launch_id,
            )
            logger.debug("process: %s", process)
        except Exception as e:
            raise ValueError("helm uninstall failed:") from e
        finally:
            if kube_config_path is not None:
                try:
                    os.unlink(kube_config_path)  # remove the temporary kube_config file
                except:
                    pass  # if the file was not created, or already removed, do nothing

    async def _wait_for_appwrapper_deletion(
        self: Self,
        launch_id: str,
        release_name: str,
        max_wait: int = 300,
        poll_interval: int = 5,
    ) -> None:
        """
        Poll the K8s API until the named AppWrapper is fully deleted (404) or
        the timeout expires.  A fixed sleep is insufficient because AppWrappers
        use Kueue finalizers that can take arbitrarily long to drain.

        Args:
            launch_id:     Launch identifier (used in log messages only).
            release_name:  AppWrapper / helm release name to poll.
            max_wait:      Maximum seconds to wait before giving up.
            poll_interval: Seconds between polls.
        """
        namespace = self.config.config["namespace"]
        logger.info(
            "[K8s launch_id %s] Waiting for AppWrapper %s to be fully deleted (max %ds)...",
            launch_id,
            release_name,
            max_wait,
        )
        deadline = asyncio.get_event_loop().time() + max_wait
        async with await AtomicApiClient.create_api_client(
            kube_config_string=self.kube_config,
            kube_context=self.kube_context,
            ssl_verification=self.ssl_verification,
        ) as _api:
            _custom_api = client.CustomObjectsApi(api_client=_api)
            while asyncio.get_event_loop().time() < deadline:
                try:
                    await _custom_api.get_namespaced_custom_object(
                        group="workload.codeflare.dev",
                        version=os.getenv("K8S_APPWRAPPER_VERSION", "v1beta2"),
                        namespace=namespace,
                        plural="appwrappers",
                        name=release_name,
                    )
                    logger.info(
                        "[K8s launch_id %s] AppWrapper %s still exists, waiting %ds...",
                        launch_id,
                        release_name,
                        poll_interval,
                    )
                    await asyncio.sleep(poll_interval)
                except client.ApiException as e:
                    if e.status == 404:
                        logger.info(
                            "[K8s launch_id %s] AppWrapper %s fully deleted, proceeding",
                            launch_id,
                            release_name,
                        )
                        return
                    logger.warning(
                        "[K8s launch_id %s] Unexpected API error while polling for AppWrapper deletion: %s",
                        launch_id,
                        e,
                    )
                    await asyncio.sleep(poll_interval)
        logger.warning(
            "[K8s launch_id %s] AppWrapper %s not deleted after %ds, proceeding with reinstall anyway",
            launch_id,
            release_name,
            max_wait,
        )

    def _get_default_retry_strategies(self: Self) -> List["RetryStrategy"]:
        """Return all K8s retry strategies with AppWrapper as the default object type."""
        object_types = ["AppWrapper"]
        return [
            UnhealthyInsufficientPodsRetryStrategy(object_types),
            PodEvictionRetryStrategy(object_types, False),
            NCCLErrorRetryStrategy(),
            AsperaRetryStrategy(),
        ]

    def _get_retry_test_scenario(self: Self) -> Optional[str]:
        return "pod_eviction"

    async def retry_workload(
        self: Self,
        launch_id: str,
        nodes_to_avoid: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Retry a failed K8s workload with node anti-affinity.

        This method:
        1. Uninstalls the current helm release
        2. Waits for cleanup to complete
        3. Reinstalls with node anti-affinity rules to avoid specified nodes

        Args:
            launch_id: The launch identifier
            nodes_to_avoid: List of node names to avoid in the retry
            **kwargs: Additional parameters (currently unused)
        """
        logger.warning(
            "[K8s launch_id %s] Starting workload retry with node avoidance: %s",
            launch_id,
            nodes_to_avoid,
        )

        # Get the original launch parameters
        if launch_id not in self.launch_params:
            raise ValueError(
                f"[K8s launch_id {launch_id}] Cannot retry: launch parameters not found"
            )

        # Pause the monitor while we stop and restart the pod (see reset() below)
        if launch_id in self.monitors:
            self.monitors[launch_id].pause()
            logger.info(
                "[K8s launch_id %s] Paused AppWrapperMonitor before helm uninstall",
                launch_id,
            )

        original_params = self.launch_params[launch_id].copy()

        # Step 1: Uninstall the current release
        try:
            await self.cleanup_helm(launch_id=launch_id)
            logger.info(
                "[K8s launch_id %s] Successfully uninstalled helm release",
                launch_id,
            )
        except Exception as e:
            logger.error(
                "[K8s launch_id %s] Failed to uninstall helm release: %s",
                launch_id,
                e,
            )
            # Continue with reinstall anyway, as the release might already be gone

        # Step 2: Wait for the AppWrapper to be fully deleted before reinstalling.
        release_name = self.launched_releases.get(launch_id, "")
        await self._wait_for_appwrapper_deletion(launch_id, release_name)

        # Step 3: Add node anti-affinity configuration if nodes to avoid are specified
        if nodes_to_avoid:
            config = original_params.get("config", {})
            k8s_config = config.get("k8s", {})

            # Build node anti-affinity rules
            affinity = k8s_config.get("affinity", {})
            node_affinity = affinity.get("nodeAffinity", {})
            required = node_affinity.get(
                "requiredDuringSchedulingIgnoredDuringExecution", {}
            )
            node_selector_terms = required.get("nodeSelectorTerms", [])

            # Collect existing match expressions
            match_expressions = []
            for term in node_selector_terms:
                match_expressions.extend(term.get("matchExpressions", []))

            # Add our node avoidance expression
            match_expressions.append(
                {
                    "key": "kubernetes.io/hostname",
                    "operator": "NotIn",
                    "values": nodes_to_avoid,
                }
            )

            # Rebuild the affinity structure
            node_selector_terms = [{"matchExpressions": match_expressions}]
            required["nodeSelectorTerms"] = node_selector_terms
            node_affinity["requiredDuringSchedulingIgnoredDuringExecution"] = required
            affinity["nodeAffinity"] = node_affinity
            k8s_config["affinity"] = affinity
            config["k8s"] = k8s_config
            original_params["config"] = config

            logger.info(
                "[K8s launch_id %s] Added node anti-affinity rules to avoid nodes: %s",
                launch_id,
                nodes_to_avoid,
            )

        # Step 4: Reinstall with node avoidance
        try:
            # self._get_launch_ready_event(launch_id).clear()
            await self.launch_helm(launch_id=launch_id, **original_params)
            # await self._get_launch_ready_event(launch_id).wait() # Don't reset() the monitor until ready.

            logger.info(
                "[K8s launch_id %s] Successfully reinstalled helm release with node avoidance",
                launch_id,
            )

            # Reset the AppWrapperMonitor state for the new workload
            if launch_id in self.monitors:
                self.monitors[launch_id].unpause()
                logger.info(
                    "[K8s launch_id %s] Unpaused AppWrapperMonitor for new workload",
                    launch_id,
                )
            else:
                logger.warning(
                    "[K8s launch_id %s] No monitor found to reset (monitor may not have been created yet)",
                    launch_id,
                )
        except Exception as e:
            logger.error(
                "[K8s launch_id %s] Failed to reinstall helm release: %s",
                launch_id,
                e,
            )
            raise

    async def monitor_sidecar_monitor(
        self: Self,
        launch_id: str,
        event_q: Optional[asyncio.Queue] = None,
        entityrun_metadata: Optional[EntityRunMetadata] = None,
        event_configs: Optional[List] = None,
        retry_enabled: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """
        Monitor with both AppWrapper and RabbitMQ monitors with optional retry on failure.

        Retry behavior follows a two-level configuration model:
        1. Environment-level: Default setting from environment.yaml config.retry.enabled
        2. Step-level: Optional override via retry_enabled parameter

        If retry_enabled is None (not specified), uses environment-level default.
        If retry_enabled is explicitly True/False, overrides environment setting.

        Args:
            launch_id: The launch identifier
            event_q: Event queue for build events
            entityrun_metadata: Metadata for the entity run
            event_configs: Event configurations
            retry_enabled: Unused directly. Retry is resolved via _get_step_retry_config
                         from step-level config (step.yaml retry_enabled_default /
                         build.yaml retry_enabled).
            **kwargs: Additional arguments
        """
        from gbserver.monitoring.appwrapper_monitor import AppWrapperMonitor
        from gbserver.monitoring.rabbitmq_events_monitor import RabbitMQEventMonitor

        stop_event = self._get_launch_stopped_event(launch_id=launch_id)
        release_name = self.launched_releases[launch_id]

        retry_enabled, retry_transparently = self._get_step_retry_config(
            self.launch_params.get(launch_id, {}),
        )
        logger.info(
            "Starting sidecar monitoring for %s launch_id %s (retry: %s)",
            release_name,
            launch_id,
            retry_enabled,
        )

        assert self.messaging_config is not None, "self.messaging_config is None"
        assert (
            self.messaging_secret is not None
        ), "no messaging secret name provided in the space environment"

        build_id = entityrun_metadata.build_id if entityrun_metadata else launch_id
        assert self.config is not None, "K8s environment config is None"
        async with self._with_retry_handler(
            launch_id,
            event_q,
            build_id,
            self.node_health_tracker,
            enabled=retry_enabled,
            entityrun_metadata=entityrun_metadata,
            retry_transparently=retry_transparently,
        ) as monitor_event_queue:
            # Create RabbitMQ-based launch monitor with termination appwrapper monitor
            appwrapper_monitor = AppWrapperMonitor(
                name=release_name,
                namespace=self.config.config["namespace"],
                poll=5.0,
                kube_config=self.kube_config,
                kube_context=self.kube_context,
                ssl_verification=self.ssl_verification,
                launch_id=launch_id,
                entityrun_metadata=entityrun_metadata,
                event_queue=monitor_event_queue,
                stop_event=stop_event,
            )
            # Store monitor reference for retry access
            self.monitors[launch_id] = appwrapper_monitor
            # Create RabbitMQ-based launch monitor with stop_event for termination
            rabbitmq_event_monitor = RabbitMQEventMonitor(
                messaging_config=self.messaging_config,
                messaging_secret=self.messaging_secret,
                launch_id=launch_id,
                event_configs=json.dumps(event_configs),
                stop_event=stop_event,
                event_queue=monitor_event_queue,
                entityrun_metadata=entityrun_metadata,
            )

            logger.info("created rabbitmq_event_monitor: %s", rabbitmq_event_monitor)
            try:
                await asyncio.gather(
                    appwrapper_monitor.monitor(),
                    rabbitmq_event_monitor.monitor(),
                    # return_exceptions=True,
                )
                logger.info(
                    "Sidecar monitoring finished for %s, launch_id %s",
                    release_name,
                    launch_id,
                )
            except Exception as e:
                logger.error("sidecar monitoring failed: %s", e)
                raise e

    async def monitor_event_monitor(
        self: Self,
        launch_id: str,
        event_q: Optional[asyncio.Queue] = None,
        entityrun_metadata: Optional[EntityRunMetadata] = None,
        event_configs: Optional[List] = None,
        **kwargs,
    ) -> None:
        """Monitor events coming from the messaging service (RabbitMQ)"""

        # from gbserver.messaging.messaging_base import Address
        # from gbserver.messaging.rabbitmq_base import RabbitMQBase, RabbitSettings
        from gbserver.monitoring.rabbitmq_events_monitor import RabbitMQEventMonitor

        stop_event = self._get_launch_stopped_event(launch_id=launch_id)
        logger.info(
            "Starting event monitoring for %s launch_id %s",
            self.launched_releases[launch_id],
            launch_id,
        )
        assert self.messaging_config is not None, "self.messaging_config is None"
        assert (
            self.messaging_secret is not None
        ), "no messaging secret name provided in the space environment"

        # Create RabbitMQ-based launch monitor with stop_event for termination
        rabbitmq_event_monitor = RabbitMQEventMonitor(
            messaging_config=self.messaging_config,
            messaging_secret=self.messaging_secret,
            launch_id=launch_id,
            event_configs=json.dumps(event_configs),
            stop_event=stop_event,
            event_queue=event_q,
            entityrun_metadata=entityrun_metadata,
        )
        await rabbitmq_event_monitor.monitor()

        logger.info(
            "Event monitoring finished for %s, launch id = %s",
            self.launched_releases[launch_id],
            launch_id,
        )

    async def monitor_appwrapper_monitor(
        self: Self,
        launch_id: str,
        event_q: Optional[asyncio.Queue] = None,
        entityrun_metadata: Optional[EntityRunMetadata] = None,
        event_configs: Optional[List] = None,
        retry_enabled: Optional[bool] = None,
        **kwargs,
    ) -> None:
        """
        Monitor events on an AppWrapper with optional retry on failure.

        Retry behavior follows a two-level configuration model:
        1. Environment-level: Default setting from environment.yaml config.retry.enabled
        2. Step-level: Optional override via retry_enabled parameter

        If retry_enabled is None (not specified), uses environment-level default.
        If retry_enabled is explicitly True/False, overrides environment setting.

        Args:
            launch_id: The launch identifier
            event_q: Event queue for build events
            entityrun_metadata: Metadata for the entity run
            event_configs: Event configurations
            retry_enabled: Unused directly. Retry is resolved via _get_step_retry_config
                         from step-level config (step.yaml retry_enabled_default /
                         build.yaml retry_enabled).
            **kwargs: Additional arguments
        """
        from gbserver.monitoring.appwrapper_monitor import AppWrapperMonitor

        appwrapper_name = self.launched_releases[launch_id]
        stop_event = self._get_launch_stopped_event(launch_id=launch_id)

        retry_enabled, _ = self._get_step_retry_config(
            self.launch_params.get(launch_id, {}),
        )
        logger.info(
            "Starting appwrapper monitoring for %s, launch_id %s (retry: %s)",
            appwrapper_name,
            launch_id,
            retry_enabled,
        )

        build_id = entityrun_metadata.build_id if entityrun_metadata else launch_id
        assert self.config is not None, "K8s environment config is None"
        async with self._with_retry_handler(
            launch_id,
            event_q,
            build_id,
            self.node_health_tracker,
            enabled=retry_enabled,
            entityrun_metadata=entityrun_metadata,
        ) as monitor_event_queue:
            # Create appwrapper monitor
            appwrapper_monitor = AppWrapperMonitor(
                name=appwrapper_name,
                namespace=self.config.config["namespace"],
                poll=5.0,
                kube_config=self.kube_config,
                kube_context=self.kube_context,
                stop_event=stop_event,
                event_queue=monitor_event_queue,
                launch_id=launch_id,
                entityrun_metadata=entityrun_metadata,
            )
            await appwrapper_monitor.monitor()
            logger.info(
                "Appwrapper monitoring finished for %s, launch_id %s",
                appwrapper_name,
                launch_id,
            )

    async def monitor_log_monitor(
        self: Self,
        launch_id: str,
        event_q: Optional[asyncio.Queue] = None,
        entityrun_metadata: Optional[EntityRunMetadata] = None,
        event_configs: Optional[List] = None,
        build_id: str = "",
        **kwargs,
    ) -> None:
        """Monitor the logs directly using the K8s API."""

        event_log_parser_configs = []
        if event_configs is not None:
            event_log_parser_configs = [
                EventLogLineParserConfig.model_validate(event_config)
                for event_config in event_configs
            ]
        pods_queue: asyncio.Queue[str] = asyncio.Queue()

        async with await AtomicApiClient.create_api_client(
            kube_config_string=self.kube_config,
            kube_context=self.kube_context,
            ssl_verification=self.ssl_verification,
        ) as api:
            self._get_launch_stopped_event(launch_id).clear()
            try:
                tasks = [
                    asyncio.create_task(
                        self.watch_for_pods(
                            api,
                            build_id,
                            launch_id,
                            pods_queue,
                            self.launched_releases[launch_id],
                        )
                    ),
                    asyncio.create_task(
                        self._process_log_queue(
                            api,
                            launch_id,
                            pods_queue,
                            event_q,
                            entityrun_metadata,
                            event_log_parser_configs,
                        )
                    ),
                ]
                await asyncio.gather(*tasks)

            except LogMonitoringFailedException as lmfe:
                logger.error(
                    "Log monitoring has exited because it detected appwrapper failure: %s",
                    launch_id,
                )
                raise lmfe
            except asyncio.CancelledError:
                raise  # must propagate for asyncio task cancellation to work
            except Exception as e:
                logger.error("Log monitoring failed for launch_id %s: %s", launch_id, e)
                raise
            finally:
                # Ensure tasks are canceled after stopping
                for task in tasks:
                    if not task.done():
                        task.cancel()
        logger.info("Log monitoring has exited successfully, launch_id: %s", launch_id)

    async def _process_log_queue(
        self: Self,
        api: client.ApiClient,
        launch_id: str,
        pod_processes_queue: Optional[asyncio.Queue] = None,
        event_q: Optional[asyncio.Queue] = None,
        entity_run_metadata: Optional[EntityRunMetadata] = None,
        event_log_parser_configs: Optional[List] = None,
    ) -> None:
        """Process queued pod names and start log streaming."""
        log_tasks = []
        while True:
            try:
                assert pod_processes_queue is not None, "pod_processes_queue is None"
                assert event_q is not None, "event_q is None"
                assert entity_run_metadata is not None, "entity_run_metadata is None"
                assert (
                    event_log_parser_configs is not None
                ), "event_log_parser_configs is None"
                pod_name = await asyncio.wait_for(pod_processes_queue.get(), timeout=10)
                logger.info("Starting log monitoring for pod %s", pod_name)
                log_tasks.append(
                    asyncio.create_task(
                        self._stream_pod_logs(
                            api,
                            event_q,
                            pod_name,
                            entity_run_metadata,
                            event_log_parser_configs,
                        )
                    )
                )
            except asyncio.TimeoutError:
                if self._get_launch_stopped_event(launch_id).is_set():
                    break  # exit if stop event is set
        asyncio.gather(*log_tasks)

    async def _stream_pod_logs(
        self: Self,
        api: client.ApiClient,
        event_q: asyncio.Queue,
        pod_name: str,
        entity_run_metadata: EntityRunMetadata,
        event_log_parser_configs: List,
    ) -> None:
        """Stream logs from a specific pod asynchronously using kubernetes-asyncio."""
        line_counter = 0  # keep track of last parsed line when watch times out
        while True:
            current_line_counter = 0
            try:
                v1 = client.CoreV1Api(api)
                async with watch.Watch().stream(
                    v1.read_namespaced_pod_log,
                    name=pod_name,
                    namespace=self.namespace,
                    follow=True,
                    _preload_content=False,
                ) as stream:
                    async for log_line in stream:
                        logger.debug("log_line = %s", log_line)
                        current_line_counter += 1
                        if not log_line:
                            continue
                        if current_line_counter > line_counter:
                            # logger.warning(f"Checking {pod_name} {current_line_counter} > {line_counter} log_line = {log_line}")
                            await self.get_events_from_log_line(
                                log_line=log_line,
                                event_configs=event_log_parser_configs,
                                event_q=event_q,
                                entityrun_metadata=entity_run_metadata,
                            )
                logger.info(
                    "Pod %s has stopped logging, shutting down monitor",
                    pod_name,
                )
                try:
                    v1 = client.CoreV1Api(api)
                    # The monitored pod is usually still in the `Running` state when the stream ends
                    # Wait before exiting to find out what state was the pod in when it stopped running
                    # If the pod is in a `Failed` state, it is going to restart with the same name
                    # and needs to be removed from the `seen_pods` set. If all other cases, do not
                    # remove the pod from the `seen_pods` set; otherwise it will replay the entire log
                    phase = "Running"
                    running_count = 0
                    while phase == "Running":
                        assert self.config is not None, "self.config is None"
                        assert (
                            self.config.config is not None
                        ), "self.config.config is None"
                        pod = await v1.read_namespaced_pod(
                            name=pod_name, namespace=self.config.config["namespace"]
                        )
                        phase = pod.status.phase
                        time.sleep(0.1)
                        running_count += 1
                        if running_count > 600:
                            break
                    if running_count > 600:
                        logger.warning(
                            "Log monitor: Pod %s remains in %s state too long after log streaming stopped.",
                            pod_name,
                            phase,
                        )
                        line_counter = current_line_counter
                        continue

                    logger.info(
                        "Log monitor shutdown: the state of the pod %s is %s",
                        pod_name,
                        phase,
                    )
                    if phase in ["Failed", "Error"]:
                        logger.error("Pod %s failed, removing from seen_pods", pod_name)
                        try:
                            self.seen_pods.remove(pod_name)
                        except KeyError as ke:
                            logger.error(
                                "Failed to remove %s from seen_pods: %s",
                                pod_name,
                                ke,
                            )
                except client.ApiException as ex:
                    # pass
                    logger.error("- Error streaming logs for pod %s: %s", pod_name, ex)
                    line_counter = current_line_counter
                    if ex.reason == "Not Found" or ex.status == 404:
                        # remove pod from the seen_pods set, so that, if a pod starts with
                        # exact same name, we can bring it back in the set of monitored pods
                        try:
                            self.seen_pods.remove(pod_name)
                        except KeyError as ke:
                            logger.error(
                                "- Failed to remove %s from seen_pods: %s",
                                pod_name,
                                ke,
                            )
                        logger.info(
                            "- Stopped monitoring for shutdown pod %s", pod_name
                        )
                        break
                    # if "Cannot connect to host" in str(ex) or ex.status == 403 or ex.status == 429 or ex.status == 503:
                    logger.error(
                        "Retrying after client.ApiException streaming logs for pod %s: %s",
                        pod_name,
                        ex,
                    )
                    time.sleep(1.0)
                    continue
                break
            except Exception as e:
                logger.error("Error streaming logs for pod %s: %s", pod_name, e)
                line_counter = current_line_counter
                # break the loop if the pod does not exist anymore
                try:
                    v1 = client.CoreV1Api(api)
                    assert self.config is not None, "self.config is None"
                    assert self.config.config is not None, "self.config.config is None"
                    _ = await v1.read_namespaced_pod(
                        name=pod_name, namespace=self.config.config["namespace"]
                    )
                except client.ApiException as ex:
                    if ex.reason == "Not Found" or ex.status == 404:
                        # remove pod from the seen_pods set, so that, if a pod starts with
                        # exact same name, we can bring it back in the set of monitored pods
                        try:
                            self.seen_pods.remove(pod_name)
                        except KeyError as ke:
                            logger.error(
                                "failed to remove %s from seen_pods: %s", pod_name, ke
                            )
                        logger.info("Stopped monitoring for shutdown pod %s", pod_name)
                        break

    async def get_helm_pods(
        self: Self, api: client.ApiClient, appwrapper_name: str
    ) -> List[str]:
        """Retrieve the list of active pods for a given Helm release."""
        v1 = client.CoreV1Api(api)
        try:
            assert self.config is not None, "self.config is None"
            assert self.config.config is not None, "self.config.config is None"
            pod_list = await v1.list_namespaced_pod(self.config.config["namespace"])
            helm_pod_names = []
            for pod in pod_list.items:
                if (
                    pod.metadata.labels is not None
                    and pod.metadata.labels.get(
                        "workload.codeflare.dev/appwrapper", None
                    )
                    == appwrapper_name
                ):
                    # gather logs only for pods in these states: Running, Succeeded, and Completed.
                    # Last two are needed for pods with a short lifespan that can run and complete
                    # execution between two pod retrievals
                    # TODO: add more states so that we can also monitor e.g. pods that run out of
                    # physical memory, or even GPU memory, and they will enter a loop
                    pod_name = pod.metadata.name
                    pod_phase = pod.status.phase
                    logger.debug(f"Pod {pod_name} is in {pod_phase} state.")
                    if pod_phase in ["Running", "Succeeded", "Completed"]:
                        helm_pod_names.append(pod_name)
                    else:
                        # get the events for the pod.
                        # If the pod is in Failed or Unknown state,
                        #   - raise an exception containing all pod events
                        # If the pod is in Pending state:
                        #   - search for events with Failed reason
                        #   - if such events exist, raise an exception
                        failed_events = await self.get_pod_failed_events(api, pod_name)
                        if pod_phase in ["Failed", "Unknown"] or (
                            pod_phase == "Pending" and failed_events
                        ):
                            logger.error(f"Pod {pod_name} is in a {pod_phase} state.")
                            logger.error("\n".join(failed_events))
                            # raise LogMonitoringFailedException(f"{exception_header}\nFail events:\n{exception_str}")
            return helm_pod_names
        except LogMonitoringFailedException as lmfe:
            logger.error(lmfe)
            raise lmfe
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Error fetching pods: {e}")
            return []

    async def get_pod_failed_events(
        self, api: client.ApiClient, pod_name: str
    ) -> List[str]:
        """Fetches events related to a specific pod."""
        v1 = client.CoreV1Api(api)
        field_selector = f"involvedObject.name={pod_name}"
        assert self.config is not None, "self.config is None"
        assert self.config.config is not None, "self.config.config is None"
        events = await v1.list_namespaced_event(
            self.config.config["namespace"], field_selector=field_selector
        )
        failed_events = []
        for event in events.items:
            logger.debug(
                f"[EVENT] {event.last_timestamp} - {event.reason}: {event.message}"
            )
            if event.reason == "Failed":
                failed_events.append(event.message)
        return failed_events

    async def get_appwrapper_status(
        self, api: client.ApiClient, appwrapper_name: str
    ) -> str:
        """Fetch the status of the AppWrapper (displayed before terminating monitoring)."""
        custom_api = client.CustomObjectsApi(api)
        try:
            assert self.config is not None, "self.config is None"
            assert self.config.config is not None, "self.config.config is None"
            response = await custom_api.get_namespaced_custom_object(
                group="workload.codeflare.dev",
                version="v1beta2",
                namespace=self.config.config["namespace"],
                plural="appwrappers",
                name=appwrapper_name,
            )
            return response.get("status", {}).get("phase", "Unknown")
        except client.ApiException as e:
            logger.error(
                f"Failed to retrieve AppWrapper {appwrapper_name} status - {type(e).__name__}: {str(e)}"
            )
            if e.reason == "Not Found" or e.status == 404:
                return "Unknown"
            # 403 Forbidden: This indicates that the server understood the request but is refusing to fulfill it, typically because the client doesn't have the necessary permissions to access the resource. While similar to a 401 Unauthorized error, a 403 error implies the client is authenticated but still denied access, even if they re-authenticate
            # 408 Request Timeout: The server didn't receive a complete request within the time it was prepared to wait. This could be due to network issues
            # 429 Too Many Requests: indicates that the client has sent too many requests to a server in a given amount of time. This is a mechanism, also called rate limiting, used by servers to prevent overloading and manage resource consumption
            # 500 Internal Server Error: A generic server-side error. Could be a temporary glitch, a bug, or an overloaded component
            # 502 Bad Gateway: The API server (acting as a gateway) received an invalid response from an upstream server (e.g., etcd). Often transient
            # 503 Service Unavailable: indicates that the web server is temporarily unable to handle the request. This usually means the server is down for maintenance, overloaded, or experiencing other temporary issues
            # 504 Gateway Timeout: The API server, acting as a gateway, did not receive a timely response from an upstream server. This suggests a bottleneck or temporary unresponsiveness in the backend
            if "Cannot connect to host" in str(e) or e.status in [
                403,
                408,
                429,
                500,
                502,
                503,
                504,
            ]:
                return "Running"
            return "Unknown"
        except aiohttp.ClientError as aio_ce:
            logger.warning(
                f"Failed to retrieve AppWrapper {appwrapper_name} status aiohttp.ClientError - {type(aio_ce).__name__}: {str(aio_ce)}"
            )
            return "Running"
        except Exception as e:
            logger.error(
                f"Failed to retrieve AppWrapper {appwrapper_name} status - {type(e).__name__}: {str(e)}"
            )
            if "Cannot connect to host" in str(e):
                return "Running"
            return "Unknown"

    async def watch_for_pods(
        self: Self,
        api: client.ApiClient,
        build_id: str,
        launch_id: str,
        pods_queue: asyncio.Queue[str],
        appwrapper_name: str,
    ) -> None:
        """Continuously monitor for new or restarted pods."""
        while not self._get_launch_stopped_event(launch_id).is_set():
            # get the appwrapper status
            appwrapper_status = await self.get_appwrapper_status(api, appwrapper_name)
            pods = await self.get_helm_pods(api, appwrapper_name)

            new_pods = [pod for pod in pods if pod not in self.seen_pods]

            # If no active pods, and appwrapper status not 'Running', terminate monitoring
            if len(new_pods) == 0:
                # check the appwrapper status
                # https://github.com/project-codeflare/mlbatch/blob/main/CODEFLARE.md
                # The status of an AppWrapper is one of:
                #     Suspended: the AppWrapper is queued,
                #     Resuming: the AppWrapper is transitioning to Running,
                #     Running: the AppWrapper is running,
                #     Succeeded: the execution completed successfully,
                #     Failed: the execution failed and will not be retried,
                #     Resetting: a failure has been detected during the current execution and the AppWrapper is preparing to retry,
                #     Suspending: the AppWrapper has been evicted by Kueue and is transitioning back to Suspended.
                if appwrapper_status in ["Succeeded", "Completed"]:
                    logger.info(
                        "No active pods found. Terminating log monitoring. AppWrapper %s status: %s",
                        appwrapper_name,
                        appwrapper_status,
                    )
                    self._get_launch_stopped_event(launch_id).set()
                    return
                elif appwrapper_status in ["Failed", "Deleted", "Cancelled", "Unknown"]:
                    logger.error(
                        "Appwrapper %s status %s. Terminating log monitoring",
                        appwrapper_name,
                        appwrapper_status,
                    )
                    raise LogMonitoringFailedException(
                        f"Appwrapper {appwrapper_name} status {appwrapper_status}. Terminating log monitoring",
                        build_id=build_id,
                    )
                else:  # "Running", "Suspended", "Resuming"
                    logger.debug(
                        "No active pods found. Waiting as appwrapper %s is still %s ...",
                        appwrapper_name,
                        appwrapper_status,
                    )

            # Handle new and restarted pods
            for pod in new_pods:
                self.seen_pods.add(pod)
                await pods_queue.put(pod)

            await asyncio.sleep(5)  # Recheck every 5 seconds

    async def pullasset_cosstore(
        self: Self,
        uri: URI,
        binding: Optional[Any] = None,
        storeload_config: Optional[StoreLoad] = None,
        assetstore: Optional[Assetstore] = None,
        secrets: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[Dict, Optional[BuildTargetStepConfig]]:
        """
        Load data from COS bucket into the cluster storage or via mount.
        """

        assert isinstance(assetstore, Cosstore), f"invalid assetstore: {assetstore}"
        assert storeload_config is not None, "storeload_config is None"
        assert storeload_config.config is not None, "storeload_config.config is None"

        if storeload_config.mode in ["afm_mount", "cos_mount"]:
            volume = storeload_config.config.get("volume", None)
            if volume is None:
                raise ValueError("Missing 'volume' in storeload configuration")

            binding_config = {
                BINDING_KEY: {
                    "path": os.path.join("/", volume, assetstore.get_relpath(uri))
                }
            }
            return binding_config, None
        elif storeload_config.mode == "cos_pull":
            cosuri = uri if isinstance(uri, URI) else URI.get_uri(uri)
            assert isinstance(cosuri, CosURI)
            cache_path = storeload_config.config.get("cache_path", None)
            if cache_path is None:
                raise ValueError("Missing 'cache_path' in storeload configuration")
            binding_path = str(Path(cache_path) / cosuri.hash())
            bucket_path = cosuri.get_metadata()["bucket_path"]
            cos_metadata = Asset(uri=cosuri).get_metadata()
            cospull_config = {
                "path": binding_path,
                "uri": bucket_path,
                "push": False,
                "cos": cos_metadata,
            }
            cospull_stepuri = "file://" + str(
                Path(__file__).parent.parent / "builtins" / "steps" / "cosrclone"
            )
            if "step_uri" in storeload_config.config:
                cospull_stepuri = storeload_config.config["step_uri"]
            binding_config = {BINDING_KEY: {"path": str(Path(binding_path))}}
            return binding_config, BuildTargetStepConfig(
                step_uri=cospull_stepuri, config={"cos_config": cospull_config}
            )
        raise ValueError(f"No known mode of loading {uri}")

    async def pushasset_cosstore(
        self: Self,
        binding: Any,
        binding_id: Optional[str] = "",
        storepush_config: Optional[StorePush] = None,
        uri: Optional[Union[str, URI]] = None,
        assetstore: Optional[Assetstore] = None,
        **kwargs,
    ) -> Any:
        """
        Allow for copying folder/file from the cluster storage to a COS bucket.
        """
        if uri is None or uri == "":
            raise ValueError(f"Empty uri received to pushasset {binding}")
        cosuri = uri if isinstance(uri, URI) else URI.get_uri(uri)
        assert isinstance(cosuri, CosURI)
        logger.info("binding type %s value %s", type(binding), binding)
        assert isinstance(
            binding, dict
        ), f"expected binding to be a dict, actual: {type(binding)} {binding}"
        assert (
            "path" in binding
        ), f"expected 'path' to be in the binding, actual: {binding}"
        binding_path = binding["path"]
        logger.info("binding_path: %s", binding_path)
        binding_path_path = Path(binding_path)
        logger.info("binding_path_path: %s", binding_path_path)
        assert (
            len(binding_path_path.parts) >= 2
        ), f"expected at least 2 parts to the path: {binding_path_path}"
        volume = str(binding_path_path.parts[1])
        logger.info("volume: %s", volume)
        uri_bucket_path = cosuri.get_metadata()["bucket_path"]
        uri_bucket_name = cosuri.get_metadata()["bucket_name"]
        cos_md = Asset(cosuri).get_metadata()
        if cos_md and cos_md.get("cos_bucket_name"):
            bucket_name = cos_md["cos_bucket_name"]
        else:
            bucket_name = uri_bucket_name
        if not uri_bucket_path.startswith(f"{bucket_name}/"):
            uri_bucket_path = (
                f"{bucket_name}/{uri_bucket_path}" if uri_bucket_path else bucket_name
            )
        use_mount = (
            storepush_config is not None
            and storepush_config.config is not None
            and storepush_config.mode in ["afm_mount", "cos_mount"]
        )
        mount_dst = None
        if use_mount:
            assert isinstance(assetstore, Cosstore), f"invalid assetstore: {assetstore}"
            volume = storepush_config.config.get("volume")
            if not volume:
                raise ValueError("Missing 'volume' in storepush configuration")
            assert isinstance(volume, str), f"invalid volume: {volume}"
            mount_dst = os.path.join("/", volume, assetstore.get_relpath(uri))

        cospush_config = {
            "path": binding_path,
            "push": True,
            "uri": uri_bucket_path,
            "bucket_name": bucket_name,
            "binding_id": binding_id,
            "cos": cos_md,
            "use_mount": use_mount,
            "mount_dst": mount_dst,
        }
        cospush_stepuri = "file://" + str(
            Path(__file__).parent.parent / "builtins" / "steps" / "cosrclone"
        )
        if (
            storepush_config is not None
            and storepush_config.config is not None
            and "step_uri" in storepush_config.config
        ):
            cospush_stepuri = storepush_config.config["step_uri"]
        return BuildTargetStepConfig(
            step_uri=cospush_stepuri, config={"cos_config": cospush_config}
        )

    async def pullasset_hfstore(
        self: Self,
        uri: URI,
        binding: Optional[Any] = None,
        storeload_config: Optional[StoreLoad] = None,
        assetstore: Optional[Assetstore] = None,
        secrets: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[Dict, Optional[BuildTargetStepConfig]]:
        """
        Pull a HuggingFace Hub asset (model/dataset/space) in a K8s container.
        Supports mode: hf_pull
        """
        assert isinstance(assetstore, Hfstore), f"invalid assetstore: {assetstore}"

        if storeload_config is None or storeload_config.config is None:
            raise ValueError("storeload_config or storeload_config.config is None")

        if storeload_config.mode == "hf_pull":
            cache_path = storeload_config.config.get("cache_path", None)
            if cache_path is None:
                raise ValueError("Did not find 'cache_path' in storeload configuration")

            hf_uri = uri if isinstance(uri, HfURI) else HfURI.parse(uri)
            binding_path = (
                Path(cache_path)
                / hf_uri.get_owner()
                / hf_uri.get_repo()
                / hf_uri.hash()
            )
            # binding_path.mkdir(parents=True, exist_ok=True)
            hfpull_config = {
                "path": str(binding_path),
                "uri": str(hf_uri),
                "owner": hf_uri.get_owner(),
                "repo": hf_uri.get_repo(),
                "revision": hf_uri.get_revision(),
            }

            # Binding config for container
            binding_config = {BINDING_KEY: {"path": str(Path(binding_path))}}
            hfpull_stepuri = "file://" + str(
                Path(__file__).parent.parent / "builtins" / "steps" / "hfpull"
            )
            if (
                storeload_config is not None
                and storeload_config.config is not None
                and "step_uri" in storeload_config.config
            ):
                hfpull_stepuri = storeload_config.config["step_uri"]

            pull_step_config = BuildTargetStepConfig(
                step_uri=hfpull_stepuri,
                config={"hfpull_config": hfpull_config},
            )

            return binding_config, pull_step_config

        raise ValueError(f"No known mode of loading {uri}")

    async def pushasset_hfstore(
        self: Self,
        binding: Any,
        binding_id: Optional[str] = "",
        storepush_config: Optional[StorePush] = None,
        uri: Optional[Union[str, URI]] = None,
        assetstore: Optional[Assetstore] = None,
        output_config: Optional[BuildTargetOutputConfig] = None,
        **kwargs,
    ) -> Any:
        """Push an artifact from the cluster to a HuggingFace Hub repository.

        Args:
            binding: Dict with a ``"path"`` key pointing to the artifact on cluster.
            binding_id: Output binding name, used in logging and push step config.
            storepush_config: Environment-level push configuration (lower priority).
            uri: Target HF URI string or object.
            assetstore: Hfstore instance whose secrets supply HF credentials.
            output_config: Per-output config from build.yaml; ``store_push`` and
                ``space_name`` fields take precedence over environment-level config.

        Returns:
            A ``BuildTargetStepConfig`` for the hfpush step.

        Raises:
            ValueError: If ``uri`` is empty or ``binding`` has no ``"path"``.
        """
        if uri is None or uri == "":
            raise ValueError(f"Empty uri received to pushasset {binding}")
        hfuri = uri if isinstance(uri, HfURI) else HfURI.parse(uri)
        logger.info("binding type %s value %s", type(binding), binding)
        assert isinstance(
            binding, dict
        ), f"expected binding to be a dict, actual: {type(binding)} {binding}"
        assert (
            "path" in binding
        ), f"expected 'path' to be in the binding, actual: {binding}"
        binding_path = binding["path"]
        logger.info("binding_path: %s", binding_path)

        # Determine repo type from URI (defaults to "model" if not specified)
        hf_type = hfuri.get_hf_type() or "model"

        # space_name is used to derive the HF resource group when no explicit
        # resource_group_name is configured (ignored if resource_group_name is set).
        space_name = output_config.space_name if output_config else None

        # Resolve resource_group_name with explicit priority:
        #   build.yaml store_push.config.hf 
        hf_resource_group_name = None
        hf_private = True 
        if output_config is not None and output_config.store_push is not None:
            hf_resource_group_name = (
                output_config.store_push.config.get("hf", {}).get(
                    "resource_group_name", hf_resource_group_name
                )
            )
            hf_private = (
                output_config.store_push.config.get("hf", {}).get(
                    "private", hf_private 
                )
            )

        hfpush_config = {
            "path": binding_path,
            "uri": str(hfuri),
            "space_name": space_name,
            "hf": {
                "type": hf_type,
                "private": hf_private, 
                "resource_group_name": hf_resource_group_name,
            },
        }
        # Apply remaining hf fields from environment-level storepush_config
        if (
            storepush_config is not None
            and storepush_config.config is not None
            and "hf" in storepush_config.config
        ):
            hfpush_config["hf"].update(storepush_config.config["hf"])
        # Apply remaining hf fields from build.yaml store_push (highest priority)
        if (
            output_config is not None
            and output_config.store_push is not None
            and "hf" in output_config.store_push.config
        ):
            hfpush_config["hf"].update(output_config.store_push.config["hf"])
        hfpush_stepuri = "file://" + str(
            Path(__file__).parent.parent / "builtins" / "steps" / "hfpush"
        )
        if (
            storepush_config is not None
            and storepush_config.config is not None
            and "step_uri" in storepush_config.config
        ):
            hfpush_stepuri = storepush_config.config["step_uri"]
        return BuildTargetStepConfig(
            step_uri=hfpush_stepuri, config={"hfpush_config": hfpush_config}
        )

    async def pullasset_lhstore(
        self: Self,
        uri: URI,
        binding: Optional[Any] = None,
        storeload_config: Optional[StoreLoad] = None,
        assetstore: Optional[Assetstore] = None,
        secrets: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[Dict, Optional[BuildTargetStepConfig]]:
        """
        Allow for a asset from lh to be made available in k8s
        """
        assert isinstance(assetstore, Lhstore), f"invalid assetstore: {assetstore}"
        assert storeload_config is not None, "storeload_config is None"
        if storeload_config.mode == "afm_mount" or storeload_config.mode == "cos_mount":
            assert storeload_config is not None, "storeload_config is None"
            assert (
                storeload_config.config is not None
            ), "storeload_config.config is None"
            volume = storeload_config.config.get("volume", None)
            if volume is None:
                raise ValueError(
                    "Did not find either 'volume' keys in storeload configuration"
                )
            binding_config = {
                BINDING_KEY: {
                    "path": os.path.join("/", volume, assetstore.get_relpath(uri))
                }
            }
            return binding_config, None

        if storeload_config.mode == "dmf_pull":
            lhuri = uri if isinstance(uri, URI) else URI.get_uri(uri)
            assert isinstance(lhuri, LhURI)
            cache_path = storeload_config.config.get("cache_path", None)
            if cache_path is None:
                raise ValueError(
                    "Did not find either 'cache_path' keys in storeload configuration"
                )
            binding_path = str(Path(cache_path) / lhuri.hash())
            lhuristr = URI.get_uristr(lhuri)
            lh_metadata = Asset(uri=lhuri).get_metadata()
            lhpull_config = {
                "use_aspera": self.dmf_use_aspera,
                "path": binding_path,
                "uri": lhuristr,
                "lh": lh_metadata,
            }
            lhpull_stepuri = "file://" + str(
                Path(__file__).parent.parent / "builtins" / "steps" / "lhpull"
            )
            if (
                storeload_config is not None
                and storeload_config.config is not None
                and "step_uri" in storeload_config.config
            ):
                lhpull_stepuri = storeload_config.config["step_uri"]
            binding_config = {
                BINDING_KEY: {
                    "path": str(Path(binding_path) / assetstore.get_subdir(uri))
                }
            }
            return binding_config, BuildTargetStepConfig(
                step_uri=lhpull_stepuri, config={"lhpull_config": lhpull_config}
            )
        raise ValueError(f"No known mode of loading {uri}")

    async def pushasset_lhstore(
        self: Self,
        binding: Any,
        binding_id: Optional[str] = "",
        storepush_config: Optional[StorePush] = None,
        uri: Optional[Union[str, URI]] = None,
        assetstore: Optional[Assetstore] = None,
        **kwargs,
    ) -> Any:
        """
        Allow for a random folder/file to be copied from any mounted storage in the cluster to a lh bucket
        """
        if uri is None or uri == "":
            raise ValueError(f"Empty uri received to pushasset {binding}")
        lhuri = uri if isinstance(uri, URI) else URI.get_uri(uri)
        assert isinstance(lhuri, LhURI)
        logger.info("binding type %s value %s", type(binding), binding)
        assert isinstance(
            binding, dict
        ), f"expected binding to be a dict, actual: {type(binding)} {binding}"
        assert (
            "path" in binding
        ), f"expected 'path' to be in the binding, actual: {binding}"
        binding_path = binding["path"]
        logger.info("binding_path: %s", binding_path)
        binding_path_path = Path(binding_path)
        logger.info("binding_path_path: %s", binding_path_path)
        assert (
            len(binding_path_path.parts) >= 2
        ), f"expected at least 2 parts to the path: {binding_path_path}"
        binding_path_path_paths = binding_path_path.parts[1]
        volume = str(binding_path_path_paths)
        logger.info("volume: %s", volume)
        lhuristr = URI.get_uristr(lhuri)
        lhpush_config = {
            "use_aspera": K8S_USE_ASPERA,
            "path": binding_path,
            "volume": volume,
            "uri": lhuristr,
            "binding_id": binding_id,
            "lh": Asset(lhuri).get_metadata(),
        }
        lhpush_stepuri = "file://" + str(
            Path(__file__).parent.parent / "builtins" / "steps" / "lhpush"
        )
        if (
            storepush_config is not None
            and storepush_config.config is not None
            and "step_uri" in storepush_config.config
        ):
            lhpush_stepuri = storepush_config.config["step_uri"]
        return BuildTargetStepConfig(
            step_uri=lhpush_stepuri, config={"lhpush_config": lhpush_config}
        )

    async def test_list_pods(self: Self, **kwargs) -> None:
        """Try listing pods in the cluster."""
        async with await AtomicApiClient.create_api_client(
            kube_config_string=self.kube_config,
            kube_context=self.kube_context,
            ssl_verification=self.ssl_verification,
        ) as api:
            try:
                v1 = client.CoreV1Api(api)
                self_config = self.config
                assert self_config is not None, "self_config is None"
                pod_list = await v1.list_namespaced_pod(self_config.config["namespace"])
                for pod in pod_list.items:
                    logger.info(
                        "context = %s, namespace = %s, pod = %s",
                        self.kube_context,
                        self.namespace,
                        pod.metadata.name,
                    )
            except Exception as e:
                logger.error("%s", traceback.format_exc())
                logger.error("failed to fetch pods: %s", e)


async def log_main() -> None:
    """Main function to test K8s"""
    launch_id: str = (
        "gb-logging-1-10k"  # "feb19-digit-aw-digit-appwrapper" # input("Enter the Helm release name: ")
    )
    launch_id_1: str = (
        "gb-logging-1-6k"  # "feb19-digit-aw-digit-appwrapper" # input("Enter the Helm release name: ")
    )
    namespace: str = (
        "granite-build"  #  input("Enter the namespace (default: default): ") or "default"
    )
    log_filter: str = (
        ""  # input("Enter a regex pattern to filter logs (leave empty for all logs): ")
    )
    # appwrapper_name: str = "gb-logging-1-10k" # input("Enter the Helm release name: ")
    # namespace: str = "granite-build"  #  input("Enter the namespace (default: default): ") or "default"
    # log_filter: str = "" # input("Enter a regex pattern to filter logs (leave empty for all logs): ")

    queue = asyncio.Queue()
    env_cfg_file = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            f"../../../test/gbserver_test/environments/{ENVIRONMENT_FILENAME}",
        )
    )
    environment_config = EnvironmentConfig.from_yaml(Path(env_cfg_file), context=None)
    k8s_monitor = K8s(
        namespace=namespace, event_q=queue, environment_config=environment_config
    )
    k8s_monitor._get_launch_ready_event(launch_id).set()
    k8s_monitor.launched_releases[launch_id] = launch_id
    k8s_monitor._get_launch_ready_event(launch_id_1).set()
    k8s_monitor.launched_releases[launch_id_1] = launch_id_1
    event_configs: List[EventLogLineParserConfig] = []
    data = {"event_type": "ARTIFACT_EVENT", "line_regex": "{.*}$", "event_fields": []}
    event_config: EventLogLineParserConfig = EventLogLineParserConfig(**data)
    event_configs.append(event_config)

    tasks = [
        asyncio.create_task(
            k8s_monitor.monitor_log_monitor(
                launch_id=launch_id,
                event_q=queue,
                entityrun_metadata=EntityRunMetadata(build_id=launch_id),
                event_configs=event_configs,
            )
        ),
        asyncio.create_task(
            k8s_monitor.monitor_log_monitor(
                launch_id=launch_id_1,
                event_q=queue,
                entityrun_metadata=EntityRunMetadata(build_id=launch_id),
                event_configs=event_configs,
            )
        ),
    ]
    await asyncio.gather(*tasks)


async def log_main_multiple_instances() -> None:
    launch_id_1: str = (
        "gb-logging-1-10k"  # "feb19-digit-aw-digit-appwrapper" # input("Enter the Helm release name: ")
    )
    launch_id_2: str = (
        "gb-logging-1-6k"  # "feb19-digit-aw-digit-appwrapper" # input("Enter the Helm release name: ")
    )
    namespace_1: str = (
        "granite-build"  #  input("Enter the namespace (default: default): ") or "default"
    )
    namespace_2: str = (
        "granite-build-staging"  #  input("Enter the namespace (default: default): ") or "default"
    )

    queue_1 = asyncio.Queue()
    queue_2 = asyncio.Queue()

    env_cfg_file_1 = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            f"../../../test/gbserver_test/environments/{ENVIRONMENT_FILENAME}",
        )
    )
    env_cfg_file_2 = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../../test/gbserver_test/environments/environment_staging.yaml",
        )
    )

    kube_config_1 = os.path.join(os.getenv("HOME"), ".kube", "config")
    kube_config_2 = os.path.join(os.getenv("HOME"), ".kube", "config")
    kube_context_1 = "granite-build/api-dmf-dipc-res-ibm-com:6443/cmadam@us.ibm.com"
    kube_context_2 = "granite-build-staging/c100-e-us-south-containers-cloud-ibm-com:30049/IAM#cmadam@us.ibm.com"

    environment_config_1 = EnvironmentConfig.from_yaml(
        Path(env_cfg_file_1), context=None
    )
    k8s_monitor_1 = K8s(
        namespace=namespace_1,
        event_q=queue_1,
        environment_config=environment_config_1,
        kube_config=kube_config_1,
        kube_context=kube_context_1,
    )
    k8s_monitor_1._get_launch_ready_event(launch_id_1).set()
    k8s_monitor_1.launched_releases[launch_id_1] = launch_id_1

    environment_config_2 = EnvironmentConfig.from_yaml(
        Path(env_cfg_file_2), context=None
    )
    k8s_monitor_2 = K8s(
        namespace=namespace_2,
        event_q=queue_2,
        environment_config=environment_config_2,
        kube_config=kube_config_2,
        kube_context=kube_context_2,
    )
    k8s_monitor_2._get_launch_ready_event(launch_id_2).set()
    k8s_monitor_2.launched_releases[launch_id_2] = launch_id_2

    tasks = [
        asyncio.create_task(
            k8s_monitor_1.test_list_pods(
                kube_config=kube_config_1,
                kube_context=kube_context_1,
                namespace=namespace_1,
            )
        ),
        asyncio.create_task(
            k8s_monitor_2.test_list_pods(
                kube_config=kube_config_2,
                kube_context=kube_context_2,
                namespace=namespace_2,
            )
        ),
    ]
    await asyncio.gather(*tasks)


async def log_watch_multiple_clusters() -> None:
    launch_id_1: str = "gb-logging-1-10k"
    launch_id_2: str = "gb-logging-1-6k"
    namespace_1: str = "granite-build"
    namespace_2: str = "vela2-training"
    kube_config_1 = os.path.join(os.getenv("HOME"), ".kube", "config")
    kube_config_2 = os.path.join(os.getenv("HOME"), ".kube", "config")
    kube_context_1 = input("Enter kube context for the first cluster: ")
    kube_context_2 = input("Enter kube context for the second cluster: ")

    queue_1 = asyncio.Queue()
    queue_2 = asyncio.Queue()

    env_cfg_file_1 = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            f"../../../test/gbserver_test/environments/{ENVIRONMENT_FILENAME}",
        )
    )
    env_cfg_file_2 = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../../test/gbserver_test/environments/environment2.yaml",
        )
    )

    environment_config_1 = EnvironmentConfig.from_yaml(
        Path(env_cfg_file_1), context=None
    )
    k8s_monitor_1 = K8s(
        namespace=namespace_1,
        event_q=queue_1,
        environment_config=environment_config_1,
        kube_config=kube_config_1,
        kube_context=kube_context_1,
    )
    k8s_monitor_1._get_launch_ready_event(launch_id_1).set()
    k8s_monitor_1.launched_releases[launch_id_1] = launch_id_1

    environment_config_2 = EnvironmentConfig.from_yaml(
        Path(env_cfg_file_2), context=None
    )
    k8s_monitor_2 = K8s(
        namespace=namespace_2,
        event_q=queue_2,
        environment_config=environment_config_2,
        kube_config=kube_config_2,
        kube_context=kube_context_2,
    )
    k8s_monitor_2._get_launch_ready_event(launch_id_2).set()
    k8s_monitor_2.launched_releases[launch_id_2] = launch_id_2

    event_configs: List[EventLogLineParserConfig] = []
    data = {"event_type": "ARTIFACT_EVENT", "line_regex": "{.*}$", "event_fields": []}
    event_config: EventLogLineParserConfig = EventLogLineParserConfig(**data)
    event_configs.append(event_config)

    tasks = [
        asyncio.create_task(
            k8s_monitor_1.monitor_log_monitor(
                launch_id=launch_id_1,
                event_q=queue_1,
                entityrun_metadata=EntityRunMetadata(build_id=launch_id_1),
                event_configs=event_configs,
            )
        ),
        asyncio.create_task(
            k8s_monitor_2.monitor_log_monitor(
                launch_id=launch_id_2,
                event_q=queue_2,
                entityrun_metadata=EntityRunMetadata(build_id=launch_id_2),
                event_configs=event_configs,
            )
        ),
    ]
    await asyncio.gather(*tasks)


async def helm_multi_cluster() -> None:
    launch_id_1: str = (
        "gb-logging-1-10k"  # "feb19-digit-aw-digit-appwrapper" # input("Enter the Helm release name: ")
    )
    launch_id_2: str = (
        "gb-logging-1-6k"  # "feb19-digit-aw-digit-appwrapper" # input("Enter the Helm release name: ")
    )
    namespace_1: str = (
        "granite-build"  #  input("Enter the namespace (default: default): ") or "default"
    )
    namespace_2: str = (
        "granite-build-staging"  #  input("Enter the namespace (default: default): ") or "default"
    )

    queue_1 = asyncio.Queue()
    queue_2 = asyncio.Queue()

    test_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "../../../test/gbserver_test/environments"
        )
    )
    env_cfg_file_1 = os.path.join(test_path, ENVIRONMENT_FILENAME)
    env_cfg_file_2 = os.path.join(test_path, "environment_staging.yaml")
    kube_config_1 = os.path.join(os.getenv("HOME"), ".kube", "config")
    kube_config_2 = os.path.join(os.getenv("HOME"), ".kube", "config")
    kube_context_1 = "granite-build/api-dmf-dipc-res-ibm-com:6443/cmadam@us.ibm.com"
    kube_context_2 = "granite-build-staging/c100-e-us-south-containers-cloud-ibm-com:30049/IAM#cmadam@us.ibm.com"

    environment_config_1 = EnvironmentConfig.from_yaml(
        Path(env_cfg_file_1), context=None
    )
    k8s_monitor_1 = K8s(
        namespace=namespace_1,
        event_q=queue_1,
        environment_config=environment_config_1,
        kube_config=kube_config_1,
        kube_context=kube_context_1,
    )

    environment_config_2 = EnvironmentConfig.from_yaml(
        Path(env_cfg_file_2), context=None
    )
    k8s_monitor_2 = K8s(
        namespace=namespace_2,
        event_q=queue_2,
        environment_config=environment_config_2,
        kube_config=kube_config_2,
        kube_context=kube_context_2,
    )
    k8s_monitor_2._get_launch_ready_event(launch_id_1)
    k8s_monitor_2._get_launch_ready_event(launch_id_2)
    tasks = [
        asyncio.create_task(
            k8s_monitor_1.launch_helm(
                launch_id=launch_id_1,
                targetsteprun_asset_dir=Path(test_path),
                launcher_config={CHART_KEY: "test-vela"},
            )
        ),
        asyncio.create_task(
            k8s_monitor_2.launch_helm(
                launch_id=launch_id_2,
                targetsteprun_asset_dir=Path(test_path),
                launcher_config={CHART_KEY: "test-ris3"},
            )
        ),
    ]
    await asyncio.gather(*tasks)

    time.sleep(10)

    tasks = [
        asyncio.create_task(k8s_monitor_1.cleanup_helm(launch_id=launch_id_1)),
        asyncio.create_task(k8s_monitor_2.cleanup_helm(launch_id=launch_id_2)),
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    # asyncio.run(main())
    # asyncio.run(log_main())
    # asyncio.run(log_main_multiple_instances())
    # asyncio.run(log_watch_multiple_clusters())
    asyncio.run(helm_multi_cluster())
