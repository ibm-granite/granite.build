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

"""SkyPilot fixes not yet in the pinned release, applied inside its API server.

Provisioning runs in the API server and its executor workers, not gbserver, so
the fixes ship as a server plugin: ``ensure_plugin_config`` registers
``SkyPatchesPlugin``, which SkyPilot installs in every server process.

Most fixes change a few lines deep in a long function, so ``_rewrite``
recompiles it from source with exact-once text edits; an upgrade that moves the
code fails at server start instead of silently dropping a fix. Drop each fix
once the pin includes it.

Contains excerpts of, and changes to, SkyPilot
(https://github.com/skypilot-org/skypilot), Copyright the SkyPilot authors,
licensed under the Apache License, Version 2.0.
"""

import functools
import importlib.metadata
import inspect
import logging
import os
import shutil
import sys
import time
from typing import Any, Callable, Optional

import sky
import yaml
from sky import exceptions, skypilot_config
from sky.backends import cloud_vm_ray_backend, task_codegen, wheel_utils
from sky.clouds import aws
from sky.data import mounting_utils
from sky.provision.slurm import instance as slurm_instance
from sky.server import common as server_common
from sky.server import plugins as sky_plugins
from sky.server.requests import process
from sky.utils import schemas

logger = logging.getLogger(__name__)

PINNED_SKYPILOT_VERSION = "0.14.2"
_PLUGIN_CLASS = f"{__name__}.SkyPatchesPlugin"
_PLUGIN_NAME = "granite.build sky_patches"
_PLUGINS_CONFIG_ENV = "SKYPILOT_SERVER_PLUGINS_CONFIG"

# ssh exits 255 when the connection, not the remote command, fails. Over a
# ProxyCommand without ControlMaster this happens intermittently on fresh
# launches. Only operations that are safe to repeat are retried.
_SSH_255_RETRIES = 3
_SSH_255_BACKOFF_S = 5

# Requests a guaranteed API-server worker serves before it is replaced; a
# reused worker keeps its peak RSS, so long-lived workers grow without bound.
_WORKER_MAX_TASKS_ENV = "SKYPILOT_API_SERVER_WORKER_MAX_TASKS"
_DEFAULT_WORKER_MAX_TASKS = 100

# An image without rsync or apt-get can use a host rsync that an admin mounts
# here through slurm.container_mounts (`/opt/sky-host-tools/rsync: /usr/bin/rsync`).
_HOST_RSYNC = "/opt/sky-host-tools/rsync"


def _rewrite(owner: Any, name: str, *edits: tuple[str, str]) -> None:
    """Replace ``owner.name`` (a function or method) with an edited copy.

    The copy is compiled from the original's source, decorators included, and
    run in the original's module globals, so it behaves like upstream code
    with the edits applied. Each ``(old, new)`` edit must match exactly once.
    """
    current = vars(owner)[name]
    func = inspect.unwrap(current)
    lines, lineno = inspect.getsourcelines(func)
    src = "".join(lines)
    for old, new in edits:
        if src.count(old) != 1:
            raise RuntimeError(
                f"sky_patches: {old!r} matched {src.count(old)} times in "
                f"{func.__module__}.{func.__qualname__}, expected 1; SkyPilot "
                "changed, so update or drop this fix."
            )
        src = src.replace(old, new)
    # `if True:` lets indented method source compile as is; the padding keeps
    # upstream line numbers in tracebacks.
    header = "if True:\n" if src[0].isspace() else "\n"
    code = compile(
        "\n" * (lineno - 2) + header + src, func.__code__.co_filename, "exec"
    )
    namespace: dict[str, Any] = {}
    exec(code, func.__globals__, namespace)  # pylint: disable=exec-used
    patched = namespace[name]
    setattr(owner, name, patched)
    # `from x import f` copies the reference (sky.provision.slurm re-exports
    # instance's functions), so rebind every sky module that holds it.
    for mod in list(sys.modules.values()):
        if (
            getattr(mod, "__name__", "").startswith("sky.")
            and vars(mod).get(name) is current
        ):
            setattr(mod, name, patched)


def retry_on_ssh_255(
    run: Callable[[], Any],
    description: str,
    returncode_of: Optional[Callable[[Any], int]] = None,
    can_retry: Optional[Callable[[], bool]] = None,
) -> Any:
    """Run ``run``, retrying while it fails with ssh's exit code 255.

    Called from the rewritten backend methods. A failure is a CommandError
    with returncode 255, or a result whose ``returncode_of`` is 255.
    ``can_retry`` is asked before each retry and vetoes unsafe repeats.
    """
    for attempt in range(_SSH_255_RETRIES + 1):
        last = attempt == _SSH_255_RETRIES
        try:
            result = run()
        except exceptions.CommandError as e:
            if e.returncode != 255 or last or (can_retry and not can_retry()):
                raise
        else:
            rc = returncode_of(result) if returncode_of else None
            if rc != 255 or last or (can_retry and not can_retry()):
                return result
        logger.warning(
            "%s failed with ssh exit code 255; retrying (%d/%d).",
            description,
            attempt + 1,
            _SSH_255_RETRIES,
        )
        time.sleep(_SSH_255_BACKOFF_S)
    raise AssertionError("unreachable")


def _patch_slurm_provision() -> None:

    _rewrite(
        slurm_instance,
        "_create_virtual_instance",
        # Reuse a CONFIGURING allocation instead of submitting a second one.
        ("['pending', 'running'],", "['pending', 'configuring', 'running'],"),
        # Container init: drop dpkg statoverrides naming a missing group (they
        # abort every apt-get install), skip apt-get on images without it,
        # fall back to a host rsync, and create a $HOME the image lacks.
        (
            "apt-get update\napt-get install -y ca-certificates rsync curl git wget fuse\n"
            "echo 'alias sudo=\"\"' >> ~/.bashrc\n",
            "if command -v dpkg-statoverride >/dev/null 2>&1; then\n"
            "    dpkg-statoverride --list | while read -r _ group _ path; do\n"
            '        getent group "$group" >/dev/null ||\n'
            '            dpkg-statoverride --remove "$path" || true\n'
            "    done\n"
            "fi\n"
            "if command -v apt-get >/dev/null 2>&1; then\n"
            "    apt-get update\n"
            "    apt-get install -y ca-certificates rsync curl git wget fuse\n"
            "fi\n"
            f"if ! command -v rsync >/dev/null 2>&1 && [ -x {_HOST_RSYNC} ]; then\n"
            f"    mkdir -p /usr/local/bin && ln -sf {_HOST_RSYNC} /usr/local/bin/rsync\n"
            "fi\n"
            'mkdir -p "$HOME"\n'
            'echo \'alias sudo=""\' >> "$HOME/.bashrc"\n',
        ),
        # Remove a stale global-scope enroot container before every launch,
        # not only on snapshot restore, and fail the launch if that fails.
        (
            "            cluster_name_on_cloud)\n        if snapshot_manifest is None:\n",
            "            cluster_name_on_cloud)\n"
            "        remove_stale_script = (\n"
            "            'while IFS= read -r c; do if [ \"$c\" = '\n"
            "            + shlex.quote(global_enroot_name)\n"
            "            + ' ]; then enroot remove -f \"$c\" || exit 1; fi; '\n"
            "            'done < <(enroot list)')\n"
            "        remove_stale_step = (\n"
            "            f'srun --overlap --unbuffered --nodes={num_nodes} '\n"
            "            '--ntasks-per-node=1 bash -c '\n"
            "            + shlex.quote(remove_stale_script) + ' || exit 1')\n"
            "        if snapshot_manifest is None:\n",
        ),
        (
            "                f'CONTAINER_PIDS=()\\n'\n",
            "                f'{remove_stale_step}\\n'\n"
            "                f'CONTAINER_PIDS=()\\n'\n",
        ),
        (
            "                f'srun --overlap --unbuffered --nodes={num_nodes} '\n"
            "                f'--ntasks-per-node=1 bash -c '\n"
            "                f'{shlex.quote(remove_stale_container_script)}',\n",
            "                remove_stale_step,\n",
        ),
        # enroot injects the NVIDIA driver only when NVIDIA_VISIBLE_DEVICES is
        # set; images not built on a CUDA base leave it unset.
        (
            "                      f'--container-writable')",
            "                      f'--container-writable')\n"
            "        if accelerator_count > 0:\n"
            "            pyxis_args += (' --container-env='\n"
            "                           'NVIDIA_VISIBLE_DEVICES,NVIDIA_DRIVER_CAPABILITIES')",
        ),
        (
            "            f'mkdir -p {container_init_done_dir}\\n'\n",
            "            f'mkdir -p {container_init_done_dir}\\n'\n"
            "            + ('export NVIDIA_VISIBLE_DEVICES=\"${NVIDIA_VISIBLE_DEVICES:-all}\"\\n'\n"
            "               'export NVIDIA_DRIVER_CAPABILITIES='\n"
            "               '\"${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}\"\\n'\n"
            "               if accelerator_count > 0 else '') +\n",
        ),
    )
    _rewrite(
        slurm_instance,
        "query_instances",
        (
            "        'pending': status_lib.ClusterStatus.INIT,\n",
            "        'pending': status_lib.ClusterStatus.INIT,\n"
            "        'configuring': status_lib.ClusterStatus.INIT,\n",
        ),
    )
    # Node cleanup is skipped for duplicates; the stale-container removal
    # above clears their enroot containers on the next launch.
    # A leaked same-named allocation (e.g. a launch that failed before its job
    # id was recorded) made teardown assert and cancel nothing. Cancel them all
    # with a plain scancel; their EXIT traps still run.
    _rewrite(
        slurm_instance,
        "_cancel_slurm_job",
        (
            "    assert len(jobs_state) == 1, (\n"
            "        f'Multiple jobs found for cluster {cluster_name_on_cloud}: {jobs_state}'\n"
            "    )\n",
            "    if len(jobs_state) > 1:\n"
            "        if all(s.strip() in _EXITING_JOB_STATES for s in jobs_state):\n"
            "            return\n"
            "        logger.warning(f'Cancelling {len(jobs_state)} Slurm jobs named '\n"
            "                       f'{cluster_name_on_cloud}: {jobs_state}')\n"
            "        client.cancel_jobs_by_name(cluster_name_on_cloud)\n"
            "        if inside_slurm_cluster or _wait_for_job_states(\n"
            "                client, cluster_name_on_cloud, _TERMINAL_JOB_STATES,\n"
            "                _JOB_TERMINATION_TIMEOUT_SECONDS):\n"
            "            return\n"
            "        raise RuntimeError(f'Slurm jobs for {cluster_name_on_cloud} still '\n"
            "                           'running after scancel; check squeue.')\n",
        ),
    )
    _rewrite(
        slurm_instance,
        "terminate_instances",
        (
            "        if len(running_jobs) > 1:\n"
            "            raise RuntimeError(f'Multiple running jobs found for cluster '\n"
            "                               f'{cluster_name_on_cloud}: {running_jobs}')\n",
            "",
        ),
    )
    # File mounts under the configured Slurm workdir must not be
    # symlink-wrapped: the wrap's `sudo mkdir` runs on the compute node as the
    # (sudo-less) login user. The workdir is the cluster's sky_base_dir, which
    # the user can write and which is identity-mounted into the container.
    _rewrite(
        slurm_instance,
        "get_command_runners",
        (
            "    return runners\n",
            "    workdir = skypilot_config.get_effective_region_config(\n"
            "        cloud='slurm',\n"
            "        region=slurm_utils.get_slurm_cluster_from_config(provider_config),\n"
            "        keys=('workdir',),\n"
            "        default_value=None)\n"
            "    for runner in runners:\n"
            "        runner.gb_unwrapped_mount_prefixes = (\n"
            "            [sky_base_dir] if workdir is not None else [])\n"
            "    return runners\n",
        ),
    )


def _patch_backend() -> None:
    cls = cloud_vm_ray_backend.CloudVmRayBackend
    setattr(cloud_vm_ray_backend, "retry_on_ssh_255", retry_on_ssh_255)
    _rewrite(
        cls,
        "_execute_file_mounts",
        (
            "        log_path = os.path.join(self.log_dir, 'file_mounts.log')\n",
            "        log_path = os.path.join(self.log_dir, 'file_mounts.log')\n"
            "        unwrapped = (getattr(runners[0], 'gb_unwrapped_mount_prefixes', [])\n"
            "                     if runners else [])\n",
        ),
        (
            "            if not dst.startswith('~/') and not dst.startswith('/tmp/'):\n",
            "            if (not dst.startswith('~/') and not dst.startswith('/tmp/') and\n"
            "                    not any(dst == p or dst.startswith(p.rstrip('/') + '/')\n"
            "                            for p in unwrapped)):\n",
        ),
        # Remote-store sync is idempotent: retry it on ssh 255.
        (
            "            # dst is only used for message printing.\n"
            "            backend_utils.parallel_data_transfer_to_nodes(\n",
            "            retry_on_ssh_255(lambda: backend_utils.parallel_data_transfer_to_nodes(\n",
        ),
        (
            "                num_threads=num_threads,\n            )\n        # (2)",
            "                num_threads=num_threads,\n            ), f'Syncing {src} to {dst}')\n"
            "        # (2)",
        ),
        # The symlink command replaces an existing link: safe to repeat.
        (
            "returncode = runner.run(symlink_command, log_path=log_path)",
            "returncode = retry_on_ssh_255(\n"
            "                    lambda: runner.run(symlink_command, log_path=log_path),\n"
            "                    'Creating file_mounts symlinks', returncode_of=lambda rc: rc)",
        ),
    )
    # The mount script unmounts an already-mounted path first: safe to repeat.
    _rewrite(
        cls,
        "_execute_storage_mounts",
        (
            "                backend_utils.parallel_data_transfer_to_nodes(\n",
            "                retry_on_ssh_255(lambda: backend_utils.parallel_data_transfer_to_nodes(\n",
        ),
        (
            "                    num_threads=num_threads,\n                )\n"
            "            except exceptions.CommandError as e:",
            "                    num_threads=num_threads,\n"
            "                ), f'Mounting {src_print} at {dst}')\n"
            "            except exceptions.CommandError as e:",
        ),
    )
    # A retried add_job can only leave an unused job id (marked FAILED_DRIVER).
    _rewrite(
        cls,
        "_add_job",
        (
            "            returncode, result_str, stderr = self.run_on_head(\n"
            "                handle,\n"
            "                code,\n"
            "                stream_logs=False,\n"
            "                require_outputs=True,\n"
            "                separate_stderr=True)\n",
            "            returncode, result_str, stderr = retry_on_ssh_255(\n"
            "                lambda: self.run_on_head(handle, code, stream_logs=False,\n"
            "                                         require_outputs=True,\n"
            "                                         separate_stderr=True),\n"
            "                'Adding a job', returncode_of=lambda r: r[0])\n",
        ),
    )
    # Queueing is not idempotent: retry only while the job is still INIT (the
    # failed attempt never reached the cluster), and never on Slurm, where the
    # submit runs in the background.
    _rewrite(
        cls,
        "_exec_code_on_head",
        (
            "            returncode, stdout, stderr = self.run_on_head(\n"
            "                handle,\n"
            "                job_submit_cmd,\n"
            "                stream_logs=False,\n"
            "                require_outputs=True,\n"
            "                run_in_background=is_slurm,\n"
            "                head_runner=head_runner)\n"
            "            # Happens when",
            "            def _job_not_queued() -> bool:\n"
            "                if is_slurm:\n"
            "                    return False\n"
            "                try:\n"
            "                    statuses = self.get_job_status(handle, [job_id],\n"
            "                                                   stream_logs=False)\n"
            "                except Exception:  # pylint: disable=broad-except\n"
            "                    return False\n"
            "                return statuses.get(job_id) == job_lib.JobStatus.INIT\n"
            "            returncode, stdout, stderr = retry_on_ssh_255(\n"
            "                lambda: self.run_on_head(handle, job_submit_cmd,\n"
            "                                         stream_logs=False,\n"
            "                                         require_outputs=True,\n"
            "                                         run_in_background=is_slurm,\n"
            "                                         head_runner=head_runner),\n"
            "                f'Submitting job {job_id}', returncode_of=lambda r: r[0],\n"
            "                can_retry=_job_not_queued)\n"
            "            # Happens when",
        ),
    )
    # Slurm binds the run step to --cpus-per-task cores; the task's CPU demand
    # (1 by default) would pin every rank of a multi-GPU task to one core. Use
    # the CPUs the allocation holds per node, falling back to the demand.
    _rewrite(
        task_codegen.SlurmCodeGen,
        "add_task",
        (
            "--cpus-per-task={task_cpu_demand} --mem=0",
            "--cpus-per-task={{_job_cpus_per_node()}} --mem=0",
        ),
        (
            "                def run_thread_func():\n",
            "                def _job_cpus_per_node():\n"
            "                    try:\n"
            "                        result = subprocess.run(\n"
            "                            ['squeue', '-h', '-j', {self._slurm_job_id!r}, '-o', '%c'],\n"
            "                            capture_output=True, text=True, timeout=30, check=False)\n"
            "                        return max({task_cpu_demand}, int(result.stdout.split()[0]))\n"
            "                    except Exception:\n"
            "                        return {task_cpu_demand}\n\n"
            "                def run_thread_func():\n",
        ),
    )


def _patch_aws_host_image() -> None:
    # `docker.host_image_id` sets the host AMI of an AWS cluster (e.g. when the
    # default GPU AMI ships a broken driver, or lacks the Fabric Manager an
    # 8-GPU NVSwitch instance needs). With a docker image it is the container's
    # host; on a bare host it is the AMI itself, unless the task names its own
    # image_id. An environment can then set it per step type whether or not
    # the step runs a container. Declared in the schema and registered through
    # register_task_overrideable_config_key, so the server's strict override
    # validation accepts it.
    _rewrite(
        schemas,
        "get_config_schema",
        (
            "        'properties': {\n            'run_options': {",
            "        'properties': {\n"
            "            'host_image_id': {'type': 'string'},\n"
            "            'run_options': {",
        ),
    )
    skypilot_config.register_task_overrideable_config_key(("docker", "host_image_id"))
    _rewrite(
        aws.AWS,
        "make_deploy_resources_variables",
        (
            "        docker_run_options = []\n"
            "        if resources.extract_docker_image() is not None:\n",
            "        docker_run_options = []\n"
            "        host_image_id = skypilot_config.get_nested(\n"
            "            ('docker', 'host_image_id'), None,\n"
            "            override_configs=resources.cluster_config_overrides)\n"
            "        if resources.extract_docker_image() is not None:\n",
        ),
        (
            "            image_id_to_use = resources.get_cloud_image_id()\n",
            "            image_id_to_use = ({None: host_image_id} if host_image_id\n"
            "                               else resources.get_cloud_image_id())\n",
        ),
        (
            "            image_id_to_use = resources.image_id\n",
            "            image_id_to_use = (resources.image_id\n"
            "                               or ({None: host_image_id}\n"
            "                                   if host_image_id else None))\n",
        ),
    )


def _patch_misc() -> None:
    # Concurrent launches read the per-hash wheel outside the wheel lock;
    # shutil.copy rewrites it in place, so one could ship a 0-byte wheel.
    setattr(wheel_utils, "replace_file_atomically", replace_file_atomically)
    _rewrite(
        wheel_utils,
        "build_sky_wheel",
        (
            "shutil.copy(latest_wheel, temp_wheel_dir)",
            "replace_file_atomically(latest_wheel, temp_wheel_dir)",
        ),
    )

    # `wget -nc -O` keeps a 0-byte goofys from an earlier failed attempt.
    orig_s3_install = mounting_utils.get_s3_mount_install_cmd

    @functools.wraps(orig_s3_install)
    def get_s3_mount_install_cmd() -> str:
        return orig_s3_install().replace("sudo wget -nc ", "sudo wget ")

    mounting_utils.get_s3_mount_install_cmd = get_s3_mount_install_cmd

    # Recycle guaranteed API-server workers after N requests (0 disables).
    raw = os.environ.get(_WORKER_MAX_TASKS_ENV, str(_DEFAULT_WORKER_MAX_TASKS))
    try:
        max_tasks = int(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid %s=%r; using %d.",
            _WORKER_MAX_TASKS_ENV,
            raw,
            _DEFAULT_WORKER_MAX_TASKS,
        )
        max_tasks = _DEFAULT_WORKER_MAX_TASKS
    orig_pool_init = process.PoolExecutor.__init__

    @functools.wraps(orig_pool_init)
    def pool_init(self: Any, max_workers: int, **kwargs: Any) -> None:
        if max_tasks > 0:
            kwargs.setdefault("max_tasks_per_child", max_tasks)
        orig_pool_init(self, max_workers=max_workers, **kwargs)

    setattr(process.PoolExecutor, "__init__", pool_init)


def replace_file_atomically(src: Any, dst_dir: Any) -> None:
    """Copy ``src`` into ``dst_dir`` so readers see the old or new file whole."""
    tmp = os.path.join(dst_dir, f".{os.path.basename(src)}.{os.getpid()}.tmp")
    try:
        shutil.copy(src, tmp)
        os.replace(tmp, os.path.join(dst_dir, os.path.basename(src)))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@functools.cache
def apply() -> None:
    """Apply every fix once per process. Raises if SkyPilot is not the pin."""
    # The fork also reports the pinned version, so pip leaves it installed.
    direct_url = importlib.metadata.distribution("skypilot").read_text(
        "direct_url.json"
    )
    if direct_url and '"vcs_info"' in direct_url:
        raise RuntimeError(
            "SkyPilot is installed from a git checkout, not the PyPI release. "
            "Run `pip install --force-reinstall --no-deps "
            f"skypilot=={PINNED_SKYPILOT_VERSION}`, then `sky api stop`."
        )
    if sky.__version__ != PINNED_SKYPILOT_VERSION:
        raise RuntimeError(
            f"sky_patches targets SkyPilot {PINNED_SKYPILOT_VERSION}, found "
            f"{sky.__version__}; re-check each fix against the new release."
        )
    _patch_slurm_provision()
    _patch_backend()
    _patch_aws_host_image()
    _patch_misc()


def ensure_plugin_config(gb_home: str) -> None:
    """Point the API server gbserver starts at a plugins config under ``gb_home``.

    Sets ``SKYPILOT_SERVER_PLUGINS_CONFIG`` (inherited by the spawned server), so
    servers started outside gbserver never load ``SkyPatchesPlugin``. Keeps any
    other plugins in that file. Call before starting the server, which loads
    plugins only at startup.
    """
    path = os.path.join(gb_home, "sky_plugins.yaml")
    os.environ[_PLUGINS_CONFIG_ENV] = path
    config: dict[str, Any] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    plugins = config.setdefault("plugins", [])
    if any(isinstance(p, dict) and p.get("class") == _PLUGIN_CLASS for p in plugins):
        return
    plugins.append({"class": _PLUGIN_CLASS})
    os.makedirs(gb_home, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)


def server_has_patches() -> bool:
    """Whether the running API server loaded ``SkyPatchesPlugin``."""
    response = server_common.make_authenticated_request(
        "GET", "/api/plugins", retry=False
    )
    if not response.ok:  # an old server without /api/plugins
        return False
    return any(
        p.get("name") == _PLUGIN_NAME for p in response.json().get("plugins", [])
    )


class SkyPatchesPlugin(sky_plugins.BasePlugin):
    """Applies the fixes in every SkyPilot server process context."""

    name = _PLUGIN_NAME  # type: ignore[assignment]

    def install(self, extension_context: Any) -> None:
        apply()
        app = getattr(extension_context, "app", None)
        if app is not None:
            # Lets gbserver refuse kube_configs on a server that would drop each
            # request's KUBECONFIG (gbserver.environment.skypilot_kube).
            from gbserver.environment import skypilot_kube

            app.add_api_route(
                skypilot_kube.FORWARDING_ROUTE,
                skypilot_kube.forwarding_endpoint,
                methods=["GET"],
            )
