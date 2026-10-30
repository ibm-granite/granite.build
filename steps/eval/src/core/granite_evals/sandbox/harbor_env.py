"""Harbor ``BaseEnvironment`` backed by a granite ``Sandbox``.

Harbor loads environments by import path (``EnvironmentConfig.import_path``),
so this plugs in without patching it. The agent loop, prompts, timeouts and
verifier stay upstream; only where commands run changes (enroot on SLURM,
podman/docker elsewhere).

enroot gives a container no PID namespace, and each ``enroot start`` is a new
user+mount namespace over the same rootfs. Harbor tasks need more than that:
Terminus-2 keeps a tmux server alive across commands, and an agent that runs
``pkill python`` must not reach the harness or vLLM (same host uid). So the
sandbox starts one *keeper* per environment,
``enroot start ... unshare --pid --fork --mount-proc sleep infinity``, and
every command joins its namespaces with ``nsenter``: one private PID
namespace per task container, torn down (with everything in it) when the
keeper dies. If the keeper can't start (no ``unshare`` in the image, kernel
policy), commands fall back to one ``enroot start`` each, like EnrootSandbox,
and leftover processes are killed by an environment marker on stop.

The host network is shared (enroot has no network namespace): tasks that
serve on fixed ports hold node-wide locks on them (sandbox/nodelock.py),
and ports below 1024 can't be bound.

Files move by ``tar`` over the command's stdin/stdout, which works the same on
every backend. Command output goes to files, not pipes, so a command that
leaves a background process holding stdout still returns when it exits (the
behaviour of ``docker exec``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import signal
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.trial.paths import EnvironmentPaths

from granite_evals.sandbox import make_sandbox
from granite_evals.sandbox.enroot import _kill_rooted, enroot_rootfs

log = logging.getLogger(__name__)

MARKER = "GRANITE_EVALS_SANDBOX_ID"
TIMEOUT_RC = 124  # what harbor's own environments return (coreutils `timeout`)
KEEPER_CMD = "exec unshare --pid --fork --mount-proc --kill-child sleep infinity"
# enroot maps only the caller's uid to root; apt's privilege drop to _apt fails there.
APT_CONF = 'APT::Sandbox::User "root";\n'


def _descendants(pid: int) -> list[int]:
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        try:
            for tid in os.listdir(f"/proc/{p}/task"):
                kids = Path(f"/proc/{p}/task/{tid}/children").read_text().split()
                todo += [int(k) for k in kids]
                out += [int(k) for k in kids]
        except OSError:
            continue
    return out


def _ns_init(pid: int) -> int | None:
    """The descendant of ``pid`` that is PID 1 of a nested PID namespace."""
    for p in _descendants(pid):
        try:
            for line in Path(f"/proc/{p}/status").read_text().splitlines():
                if line.startswith("NSpid:"):
                    ids = line.split()[1:]
                    if len(ids) > 1 and ids[-1] == "1":
                        return p
        except OSError:
            continue
    return None


def _read_environ(pid: int) -> dict[str, str]:
    raw = Path(f"/proc/{pid}/environ").read_bytes()
    env = {}
    for item in raw.split(b"\0"):
        key, sep, value = item.decode("utf-8", "replace").partition("=")
        if sep and key:
            env[key] = value
    return env


def _kill_marked(marker: str) -> int:
    """SIGKILL every process of ours whose environment carries ``marker``."""
    needle = f"{MARKER}={marker}".encode()
    killed = 0
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            if needle in Path(f"/proc/{entry}/environ").read_bytes().split(b"\0"):
                os.kill(int(entry), signal.SIGKILL)
                killed += 1
        except (OSError, ProcessLookupError):
            continue
    return killed


def dockerfile_workdir(environment_dir: Path) -> str | None:
    """Last ``WORKDIR`` of the task's Dockerfile (the prebuilt image's)."""
    try:
        lines = (environment_dir / "Dockerfile").read_text().splitlines()
    except OSError:
        return None
    workdir = None
    for line in lines:
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].upper() == "WORKDIR":
            workdir = parts[1].strip().strip("\"'")
    return workdir


class SandboxEnvironment(BaseEnvironment):
    """A harbor task container on a granite sandbox backend (``backend`` kwarg or
    ``GRANITE_EVALS_SANDBOX``; enroot by default)."""

    def __init__(self, *args: Any, backend: str = "", pid_namespace: bool = True, **kwargs: Any):
        self._backend = backend or os.environ.get("GRANITE_EVALS_SANDBOX", "enroot")
        self._pid_namespace = pid_namespace
        self._sandbox = None
        self._keeper: asyncio.subprocess.Process | None = None
        self._ns_pid: int | None = None
        self._ns_env: dict[str, str] = {}
        self._scratch: Path | None = None
        self._nsenter = "nsenter"
        super().__init__(*args, **kwargs)
        self._workdir = (
            self.task_env_config.workdir or dockerfile_workdir(self.environment_dir) or "/"
        )

    @staticmethod
    def type() -> str:
        return "granite-sandbox"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # Logs are copied out (not bind-mounted); internet is always on.
        return EnvironmentCapabilities()

    @property
    def exec_mode(self) -> str:
        return "nsenter" if self._ns_pid else self._backend

    def _validate_definition(self):
        if not self.task_env_config.docker_image:
            raise ValueError("the granite sandbox environment needs [environment].docker_image")

    # -- lifecycle -----------------------------------------------------------

    async def start(self, force_build: bool) -> None:
        self._sandbox = make_sandbox(self.task_env_config.docker_image, backend=self._backend)
        self._scratch = Path(tempfile.mkdtemp(prefix=f"{self._sandbox.name}-io-"))
        await asyncio.to_thread(self._sandbox.start)
        if self._backend == "enroot" and self._pid_namespace:
            await self._start_keeper()
        log.info("%s: %s in %s (exec: %s)", self.environment_name, self.task_env_config.docker_image, self._sandbox.name, self.exec_mode)
        dirs = [str(EnvironmentPaths.agent_dir), str(EnvironmentPaths.verifier_dir), str(EnvironmentPaths.artifacts_dir)]
        dirs += [str(m["target"]) for m in self._mounts if m.get("target")]
        setup = f"mkdir -p {' '.join(map(shlex.quote, dirs))} && chmod 777 {' '.join(map(shlex.quote, dirs))}"
        if self._backend == "enroot":
            setup += f" && if [ -d /etc/apt/apt.conf.d ]; then printf %s {shlex.quote(APT_CONF)} > /etc/apt/apt.conf.d/99granite-sandbox; fi"
        r = await self.exec(setup, cwd="/", user="root")
        if r.return_code != 0:
            raise RuntimeError(f"sandbox setup failed: {r.stdout}{r.stderr}")
        await self._upload_environment_dir_after_start()

    async def _start_keeper(self) -> None:
        sb = self._sandbox
        self._nsenter = shutil.which("nsenter") or "nsenter"
        argv = sb._exec_argv(KEEPER_CMD, "/", {MARKER: sb.name})
        err_path = self._scratch / "keeper.err"
        with open(err_path, "wb") as err_f:
            self._keeper = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=err_f,
                stderr=err_f,
                env=self._enroot_env(),
                start_new_session=True,
            )
        why = "no PID 1 after 30s"
        for _ in range(300):
            if self._keeper.returncode is not None:
                why = f"keeper exited {self._keeper.returncode}"
                break
            if (pid := _ns_init(self._keeper.pid)) is not None:
                try:
                    self._ns_env = _read_environ(pid)
                    self._ns_pid = pid
                except OSError as e:
                    why = f"reading its environment: {e}"
                    break
                rc, out, err = await self._run("true", cwd="/", timeout_sec=60)
                if rc == 0:
                    return
                why = f"nsenter exit {rc}: {(out + err).strip()[-500:]}"
                self._ns_pid = None
                break
            await asyncio.sleep(0.1)
        keeper_err = err_path.read_text("utf-8", "replace").strip()[-500:]
        log.warning("no PID namespace for %s (%s; %s); one enroot start per command", sb.name, why, keeper_err)
        await self._stop_keeper()

    async def _stop_keeper(self) -> None:
        if self._ns_pid:
            try:
                os.kill(self._ns_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._ns_pid = None
        if self._keeper and self._keeper.returncode is None:
            try:
                os.killpg(self._keeper.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await self._keeper.wait()
        self._keeper = None

    async def stop(self, delete: bool) -> None:
        await self._stop_keeper()
        if self._sandbox is not None:
            name = self._sandbox.name
            killed = await asyncio.to_thread(_kill_marked, name)
            if self._backend == "enroot":
                killed += await asyncio.to_thread(_kill_rooted, enroot_rootfs(name))
            if killed:
                log.info("%s: killed %d leftover processes of %s", self.environment_name, killed, name)
            await asyncio.to_thread(self._sandbox.close)
            self._sandbox = None
        if self._scratch:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None

    # -- commands ------------------------------------------------------------

    def _argv(self, command: str, cwd: str, env: dict[str, str]) -> tuple[list[str], dict[str, str] | None]:
        """argv + local process env running ``command`` in the container."""
        sb = self._sandbox
        if self._ns_pid:
            argv = [self._nsenter, "-t", str(self._ns_pid), "-U", "-m", "-p", "-r", "-w", "--preserve-credentials",
                    "--", "bash", "-c", f"cd {shlex.quote(cwd)} && {command}"]  # fmt: skip
            return argv, {**self._ns_env, **env}
        if self._backend == "enroot":
            return sb._exec_argv(command, cwd, env), self._enroot_env()
        return sb._exec_argv(command, cwd, env), sb._run_env()

    def _enroot_env(self) -> dict[str, str]:
        """Local env of ``enroot start``, which passes it into the container: only
        what enroot needs, so no harness variable (API keys, tokens) reaches the
        agent's container."""
        keep = ("PATH", "HOME", "USER", "LANG", "TERM", "TMPDIR")
        env = {k: v for k, v in os.environ.items() if k.startswith("ENROOT_") or k in keep}
        env["NVIDIA_VISIBLE_DEVICES"] = "void"
        return env

    async def _run(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: float | None = None,
        stdin: Path | None = None,
        stdout: Path | None = None,
    ) -> tuple[int, str, str]:
        """Run ``command``; stdout to ``stdout`` if given. Returns (rc, out, err)."""
        if self._sandbox is None:
            raise RuntimeError("sandbox environment not started")
        env = {**(env or {}), MARKER: self._sandbox.name}
        argv, proc_env = self._argv(command, cwd or self._workdir, env)
        fd, err_path = tempfile.mkstemp(dir=self._scratch, suffix=".err")
        os.close(fd)
        out_path = stdout or Path(tempfile.mkstemp(dir=self._scratch, suffix=".out")[1])
        timed_out = False
        with open(out_path, "wb") as out_f, open(err_path, "wb") as err_f:
            stdin_f = open(stdin, "rb") if stdin else None
            try:
                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    stdin=stdin_f or asyncio.subprocess.DEVNULL,
                    stdout=out_f,
                    stderr=err_f,
                    env=proc_env,
                    start_new_session=True,
                )
                try:
                    await asyncio.wait_for(proc.wait(), timeout_sec)
                except (asyncio.TimeoutError, asyncio.CancelledError) as e:
                    timed_out = isinstance(e, asyncio.TimeoutError)
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await proc.wait()
                    if not timed_out:
                        raise
            finally:
                if stdin_f:
                    stdin_f.close()
        err = Path(err_path).read_text("utf-8", "replace")
        os.unlink(err_path)
        out = ""
        if stdout is None:
            out = Path(out_path).read_text("utf-8", "replace")
            os.unlink(out_path)
        if timed_out:
            return TIMEOUT_RC, out, err + f"\nCommand timed out after {timeout_sec} seconds"
        return proc.returncode, out, err

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        user = self._resolve_user(user)
        if user is not None and str(user) not in ("root", "0"):
            name = f"$(getent passwd {user} | cut -d: -f1)" if isinstance(user, int) else shlex.quote(str(user))
            command = f"su {name} -s /bin/bash -c {shlex.quote(command)}"
        rc, out, err = await self._run(command, cwd=cwd, env=self._merge_env(env) or {}, timeout_sec=timeout_sec)
        return ExecResult(stdout=out, stderr=err, return_code=rc)

    # -- files ---------------------------------------------------------------

    async def _tar_in(self, archive: Path, target_dir: str) -> None:
        q = shlex.quote(target_dir)
        rc, out, err = await self._run(f"mkdir -p {q} && tar -xf - -C {q} --no-same-owner", cwd="/", stdin=archive)
        if rc != 0:
            raise RuntimeError(f"upload to {target_dir} failed: {out}{err}")

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        q = shlex.quote(target_path)
        rc, out, err = await self._run(f'mkdir -p "$(dirname {q})" && cat > {q}', cwd="/", stdin=source)
        if rc != 0:
            raise RuntimeError(f"upload to {target_path} failed: {out}{err}")

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        source = Path(source_dir)
        if not source.is_dir():
            raise FileNotFoundError(source)
        archive = Path(tempfile.mkstemp(dir=self._scratch, suffix=".tar")[1])
        try:
            # Dereference: task files in an HF snapshot are symlinks into its blob store.
            with tarfile.open(archive, "w", dereference=True) as tar:
                for item in sorted(source.iterdir()):
                    tar.add(item, arcname=item.name)
            await self._tar_in(archive, target_dir)
        finally:
            archive.unlink(missing_ok=True)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + ".part")
        rc, _, err = await self._run(f"cat {shlex.quote(source_path)}", cwd="/", stdout=part)
        if rc != 0:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"download of {source_path} failed: {err}")
        part.replace(target)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        archive = Path(tempfile.mkstemp(dir=self._scratch, suffix=".tar")[1])
        try:
            rc, _, err = await self._run(f"tar -cf - -C {shlex.quote(source_dir)} .", cwd="/", stdout=archive)
            if rc != 0:
                raise RuntimeError(f"download of {source_dir} failed: {err}")
            with tarfile.open(archive) as tar:
                tar.extractall(target, filter="tar")
        finally:
            archive.unlink(missing_ok=True)
