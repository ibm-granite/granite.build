"""Terminal-Bench 2.1 (Granite: agentic coding).

Metric: pass@1 resolve rate, i.e. the fraction of tasks whose tests pass
(harbor reward 1) in one trial per task (``repeats`` 1). With ``--repeats k``
the value is pass@1[avg-of-k], that fraction per independent repeat averaged
over the k, and ``details.pass_at_k`` adds pass@k: the fraction of tasks
resolved in at least one of the k trials.

The harness is Harbor, Terminal-Bench 2.x's official one, with its reference
agent Terminus 2. Each (task, repeat) is one harbor ``Trial``: harbor's agent
loop, prompts, per-task agent/verifier timeouts and test verification run
unchanged. The one substitution is the container backend: harbor's Docker
environment becomes :class:`granite_evals.sandbox.harbor_env.SandboxEnvironment`
(enroot), started from each task's prebuilt ``docker_image``.

Tasks are read from the pinned HF mirror of the dataset (``registry.json`` +
``tasks/``), downloaded as plain files into ``<output_dir>/dataset``. At the pinned revision, the harbor content digests of all 89
tasks are checked against the digests of the published dataset manifest, so a
score always names the exact task bytes.

``--option agent=oracle`` runs each task's reference ``solution/solve.sh``
instead of a model (no GPU / server needed): every runnable task should pass.

Trials are written to ``<output_dir>/repeat-<k>/<task>/`` (harbor's own trial
layout plus ``granite.json``); a finished trial is skipped on restart.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import Any

from granite_evals import data
from granite_evals.registry import Benchmark, pass_at_k, register
from granite_evals.sandbox.nodelock import async_node_locks

log = logging.getLogger(__name__)

N_TASKS = 89
# sha256 of the comma-joined sorted harbor content digests of the 89 tasks,
# equal to the digests in harbor-framework/terminal-bench-2-1@7131e43
# tasks/dataset.toml (which the HF revision below mirrors).
TASKS_DIGEST = "0fd3dbc55e6d7a6207dc0d93c5e933684b3ec87ac12941de0420b73c08627ee5"
# Per-task harbor content digests from that dataset.toml, so a mismatch names
# the tasks that differ.
TASK_DIGESTS = {
    "adaptive-rejection-sampler": "bcaa2399985cd57666018025846289ab25e193ae0dd8fb7f0ffab2410c24d4de",
    "bn-fit-modify": "b5f9644970c17ad9ddb46b7266f7bcd87c761d77d7e6f55d7cfe7284d5ff66e9",
    "break-filter-js-from-html": "678008d1a4fd1e6e1b9b3cc9a327219fe4b410a31eafc52e9099bbf947eea600",
    "build-cython-ext": "35dc561790e73f13e59837828ac0863c0893edc6b71d42cf7d58656c843adf12",
    "build-pmars": "63f0449b276f081986740c941ea9372068b310a02b062c1e5e3400f575241893",
    "build-pov-ray": "4f08dab602fe2e9e2f42051cc6aef684a88cf2822e8b499b3ca19e670759fddb",
    "caffe-cifar-10": "7b0045106d7d5af724efe96b610ba64f7893f5c88528401c573c4d47e384e2bf",
    "cancel-async-tasks": "a3d048d351136e48070696cda8bb79660dfd74db1fea3b6da88559f0332699c1",
    "chess-best-move": "9ab8e4b3674282e751edafbd9b5bd551fef995fd6601585a2cdb04fd70c520da",
    "circuit-fibsqrt": "9bcffe1054bb33249aa578a9a2a74f3c8cca66b0cb7aa1328233f1d31822aae3",
    "cobol-modernization": "e03aa03965eee48c9f772d566ff9bcd741fceeac9693123f82a13fdd46be0d88",
    "code-from-image": "ef2907bb300d9b3352410c2a75dbf831278c187c073298371870ea9c83526f78",
    "compile-compcert": "22bde2a73d62fa46f5ab96f710561a72e8d1aa4266f008605a17f174622ad785",
    "configure-git-webserver": "e3dc712acfecdb1338a889f86c97b37422c3778351253f9aca4236e895fdb36e",
    "constraints-scheduling": "91dd6f87e1ee508328d5aafbd99a6bd1ccc47bc22671b4aaa4060bda492deb53",
    "count-dataset-tokens": "cd0574ec8281f53854256acc93702ed1441407c0428849e1dd4f27e863c6b08a",
    "crack-7z-hash": "99cbb2269f6bd112d3387fd01cb6900118fe4aded3f75a8d656580a8296a1ae5",
    "custom-memory-heap-crash": "61e021e0304c9818a9be2c9173c8a0938ce5d136e95c5422e099f1fdb2dc080e",
    "db-wal-recovery": "07e9193513bfce038b9f9d84962ecc85683eee0a642d14ee6fb7104b086ddab4",
    "distribution-search": "8787d3004ac4dd9d78f0c14f5b5ef94a4ad3e69d8828d7d35b9fbd7828a69ed1",
    "dna-assembly": "e41a8e94d86019949b08d3b5f88a85f6d943ba0fd85d5e1d5ebb95cb8f66223f",
    "dna-insert": "37fc023970e3464c1acc31f43e04a0b4b94758130d340788f261f00764d9e530",
    "extract-elf": "1ef31d566be4fe3459d5368621ae7ef7a31b23ef675737e473bbc43c8c7b3fce",
    "extract-moves-from-video": "168e2b9168511b8f8d32b1697f4703b08626203e7949ad396658f4bef7dee66e",
    "feal-differential-cryptanalysis": "8ea56995fcc43fb94f0e4e15adb12dd28836bf3e9c766b2cc7ae78a7ce90341f",
    "feal-linear-cryptanalysis": "0f3234c5fde85f9dc610d94cfefd4c892afb75c9e571f21f18fee66972b67332",
    "filter-js-from-html": "2d1496b6fc62adeccdba7a56f4bc24e5ef265840434d2011234ed20b6c240759",
    "financial-document-processor": "63dc43a15d9ec401613016e9114d1a93ec860eea63b2db02abfe8060deea30fb",
    "fix-code-vulnerability": "d31348aa16b533a15f013420e4d9726dc529e6086adc7d6d48067d22cc18fe71",
    "fix-git": "16948b980df9d96de616a205f5acca1c5d395de83ff4f8ffabcafacb93226f2e",
    "fix-ocaml-gc": "2031f3d5224b891fe443a31526b569eb756be2f88b663cb025448af9c4f97e2c",
    "gcode-to-text": "7dd2af04820e71ecaba96bc493cbffa16317e96816c6c8003385c034adf6c23b",
    "git-leak-recovery": "22a9ec10dbd4cd8b99477b70e1944103775ca41de9b9e0025ec4898cd17bd334",
    "git-multibranch": "7759a92a373c967dd31535975bad8659410644025673fe62385444fd2605c5fa",
    "gpt2-codegolf": "fe42af8e9c5aa927c3b680acd08e43032c84088fa9d770b03ce63dbc66fea4e7",
    "headless-terminal": "203953871ebdae4efbf163af9499849368dab5e219b70d447e5ee9701ad382d9",
    "hf-model-inference": "01370f37e61286f920dfca1e471496640ebcc8d06e1cc48d874c790715dbd4ad",
    "install-windows-3.11": "1f1361b0012e1ad24054e7cfc039829459e8ab0631b3573d9b93499bf6b3563e",
    "kv-store-grpc": "973c5d4c111fb61a344457936f1c36400acd2d9e44389e7b319586fe23a7a307",
    "large-scale-text-editing": "1f1cddc3df15e452fe2d3c6928f6b1e5b5330a7ae67cab373a0d089ea7d334a2",
    "largest-eigenval": "1b6f17c344d23e97435ba3fa93400d141f6bb7763ed2014fab418830983791b3",
    "llm-inference-batching-scheduler": "a3bf47589118daec5124fe3689b4439802c805b2f4ed9d402a48ae73ca2fea67",
    "log-summary-date-ranges": "27b074a2f10fff7606e096f3abd8dced418ad8fda0f53d88acbe477f2d9ceaf6",
    "mailman": "831b3ed00807153963c05f91599443e4839099521d1d56db811c5547ab280dbe",
    "make-doom-for-mips": "2d83dd3dee8e0f055e09973934cf0d7e3169a9cd90704cba5c8940b170be9498",
    "make-mips-interpreter": "41a55da0abec5d7b32a0c2321f8b18e84000ca8074ae62c6874d6ed4a3a1cd3c",
    "mcmc-sampling-stan": "443cd2b94ed797944793e97a56527a5dc0f8ad40f78ed418fe1b2e367cf22546",
    "merge-diff-arc-agi-task": "6aab6511a5344ce87698293bb1ce4cc51d9a45f1ad9f0c075d2a83197b36727d",
    "model-extraction-relu-logits": "1ae5045ad68b5d34c3398b612066a07c4a08b6dc330d28868ec4021e17c94b17",
    "modernize-scientific-stack": "67a9952aca67df9c70510acf62faea336644a0e9d379f3aaf9df79f14c8fcb12",
    "mteb-leaderboard": "484f6d7008a05b5b8640fc6618a384b8c9447cd76f85416c8a595028d29bff9c",
    "mteb-retrieve": "fa7c777df6ed8987a2bb144b4b73a1200935bbea69da03a2aaa1dc8aa9bd9dc9",
    "multi-source-data-merger": "70367c38732e1beda7b229968a48d60242277c2fa4db91339c3f064c4c230d49",
    "nginx-request-logging": "9d1b8bebd989ea0bc8080c3b159480068caf183c5fd61385868b2574b206e097",
    "openssl-selfsigned-cert": "d4afa2bd2a9ba1420db8d6cfde42ffdb4873ae2d955c35014e8da94444c83302",
    "overfull-hbox": "3c3b16e4b2de5fea90ca98db242b155a6843f9cc4dfd00be3479985a34bab4f7",
    "password-recovery": "9b5c6bdce0cf03f075b21f6a7af5030b825ce07d7b16b37610f3964217ab84a4",
    "path-tracing": "cf56094c881a488b27e9f204a638a7e78ed7d55e12dc3064108c93357190314c",
    "path-tracing-reverse": "035880d7ef5554b217e53a75ce5e87e43193b7a44c30ad17226a6ab10fc11a6c",
    "polyglot-c-py": "25570ccacf63f573a252eaa2a9c62ee8b55f5aa228cc803b5beb39dd6d8afeb7",
    "polyglot-rust-c": "a33dc72e2278d225513c0724abe2e4539653d5ee7e4c2a90ba2d676586ba9f3d",
    "portfolio-optimization": "d112f8945600f18ec47bbbe76935dc05953173cc3d0a375a7aa2712b091584e4",
    "protein-assembly": "c491d45acce234e1aa1d44dcc6778c55fcf875e9eb254b8dd3ae5ef58119621b",
    "prove-plus-comm": "d5ae25720df1f7cd619ed8408cda4b87b79a3d8b0ff4f1919e2c96e9547339c2",
    "pypi-server": "1a1e0542f58e2d3362fec17a9bbb98667717d9a4a3e9a4c8413d3150a4fa0ff1",
    "pytorch-model-cli": "6f11544dcc81a380dc4611902961f5520259c462d9fceb413419d66ab9a96c79",
    "pytorch-model-recovery": "2e628841cff93290919172398e573e794f34c95d9382b7425adde4364022decc",
    "qemu-alpine-ssh": "60b7050b0e0aa51641208cf65766743d340e59575db2c4d2f8628240846c2a28",
    "qemu-startup": "8e58263747da7dc688ad470688fb72825c4ba2c1c40443fa0f52963645bbd999",
    "query-optimize": "169496ea6843cb403b0860132675baf7aa0df0ac8b86221068e27d86395260f4",
    "raman-fitting": "97ca4025332b20739da553cab0658f2ead925d1b5b3b9b7664dc9eb5e7fbeab7",
    "regex-chess": "e763e0ac1c9759081af0a4a82ba51b8cf9ae5485a93de3bbe42d7d344597bd78",
    "regex-log": "802c16cfd132e6c457529cb864be5a757c1b23b6cadc57f2d01983cb0110292a",
    "reshard-c4-data": "a402bbfede73dd04168d697de9b9146026f69e14c8cd333b6a91fc87e44ecd4b",
    "rstan-to-pystan": "592f15f75751f3c4999c3d05087f2f8b2a4dbaaedbfad5a30ae54058b80b925f",
    "sam-cell-seg": "37c182b91df18b9c2ea2dceda47430bbd585e7880c56fca2f1975907ce0df8ce",
    "sanitize-git-repo": "73c94a21ebe370bae843adbeeaaa9e991374867b18483aaf56c7cd470dcddea7",
    "schemelike-metacircular-eval": "58130c2166c3115276dc8592f358e326ff2d81ea852e3d88636c82fd1dff57e6",
    "sparql-university": "02aeca67b6c5b0d2d72c91ab471beb51ded6f42f0dc3dec580e20d489f09a867",
    "sqlite-db-truncate": "956f038b479cc3b9b493553b57a60a8ff4154526386c3914c0b99e93e1ab6e87",
    "sqlite-with-gcov": "9f9bd57fbf9f4831e9031755e83aea6b9d60d2b2d54e8a12d48cff4dca3c231d",
    "torch-pipeline-parallelism": "db605337c749a872cea7b5b413429b3915bb4c3efe0f7875f0c46ce81bd8c4fb",
    "torch-tensor-parallelism": "f32ce74a5aeb6638480247ab799fe46127bbee631acdd0921b0f394ec49b3684",
    "train-fasttext": "460fc0818971ec83545a76805267b65459128fad52e68c26a199a0d74022badb",
    "tune-mjcf": "7da0fd3b906624df9eff8829ccd4f19cbb1ba967411b6d64d92b85b2f3fbdfb0",
    "video-processing": "d3f02e177b49e5768b6ce6709fc4ae3ef2ce0cdecb63b09fc9b07f9d3ddb7203",
    "vulnerable-secret": "d76dfa9e256487c5542905b892156f694137aeef784e1abf3f41e15a8c946eac",
    "winning-avg-corewars": "a9f2c630fb7d656e96f3a42ade600a8abcb500631d153d62d5ceb2df073bc256",
    "write-compressor": "d9ddd9a8e925e2c566b37b2492cbf995afecefe58874e4043ef78d7f3c892c7e",
}

# Tasks that can't run on the sandbox, with the reason; excluded from n and
# recorded in results.json. Established by oracle runs.
_SSHD = (
    "runs sshd and its tests git-clone over ssh to localhost:22; the enroot sandbox shares the "
    "host network, so that reaches the host's sshd (oracle fails there)"
)
EXCLUDED: dict[str, str] = {
    "configure-git-webserver": _SSHD,
    "git-multibranch": _SSHD,
}

# Tasks whose services listen on fixed ports (from their instructions, tests
# and solutions). The enroot sandbox shares the host network, so each trial
# holds a node-wide lock per port (sandbox.nodelock): tasks on the same port
# never overlap, in this process or in another job on the node.
HOST_PORTS: dict[str, tuple[int, ...]] = {
    "headless-terminal": (8000,),  # python -m http.server 8000
    "hf-model-inference": (5000,),  # flask API
    "install-windows-3.11": (80, 5901, 8080),  # nginx, QEMU VNC :1, noVNC
    "kv-store-grpc": (5328,),
    "nginx-request-logging": (8080,),
    "pypi-server": (8080,),
    "qemu-alpine-ssh": (2222, 6665),  # hostfwd ssh, QEMU telnet console
    "qemu-startup": (2222, 6665),
}
HOST_PORT_TASKS = frozenset(HOST_PORTS)
# The lock of a task known to use fixed ports whose ports aren't listed.
HOST_PORT_KEY = "tbench-host-port"


def port_locks(name: str) -> list[str]:
    """Node lock keys of a task: ``port-<n>`` (shared with other benchmarks'
    locks on the same port), or none for a task on no fixed port."""
    if name not in HOST_PORT_TASKS:
        return []
    return [f"port-{p}" for p in HOST_PORTS.get(name, ())] or [HOST_PORT_KEY]

# Exception types harbor itself doesn't retry (RetryConfig default): outcomes
# of the agent or the tests, not of the infrastructure.
FINAL_EXCEPTIONS = frozenset(
    {
        "AgentTimeoutError",
        "VerifierTimeoutError",
        "RewardFileNotFoundError",
        "RewardFileEmptyError",
        "VerifierOutputParseError",
        "ApiUsageLimitError",
        "AgentSafetyRefusalError",
        "AgentAuthenticationError",
        "ModelNotFoundError",
        "ContextLengthExceededError",
        "OutputLengthExceededError",
    }
)
AGENTS = ("terminus-2", "oracle")
IMAGE_IMPORT_WORKERS = 4


def task_digest_map(task_dirs: dict[str, Path]) -> dict[str, str]:
    """``{name: harbor content digest}`` of ``{name: task dir}``."""
    from harbor.publisher.packager import Packager

    return {n: Packager.compute_content_hash(d)[0] for n, d in task_dirs.items()}


def task_digests(task_dirs: dict[str, Path]) -> str:
    """Aggregate harbor content digest: sha256 of the sorted, comma-joined task digests."""
    return hashlib.sha256(",".join(sorted(task_digest_map(task_dirs).values())).encode()).hexdigest()


def _has_symlinks(root: Path) -> bool:
    return any(p.is_symlink() for p in root.rglob("*"))


def load_tasks(source: str, revision: str | None, local_dir: Path) -> tuple[Path, list[dict]]:
    """(dataset root, registry rows ``{name, path}``) of ``source``: an HF
    dataset repo in harbor registry layout, or a local directory of one.

    Harbor refuses task files that are symlinks out of the task (its input
    path check), which is how the HF cache stores snapshots; so the tasks
    are materialized as plain files in ``local_dir``."""
    if Path(source).exists():
        root = Path(source)
        if _has_symlinks(root):
            log.info("copying %s to %s (symlinks resolved)", root, local_dir)
            shutil.copytree(root, local_dir, symlinks=False, dirs_exist_ok=True)
            root = local_dir
    else:
        from huggingface_hub import snapshot_download

        log.info("downloading %s@%s to %s", source, revision or "main", local_dir)
        root = Path(snapshot_download(source, repo_type="dataset", revision=revision, local_dir=local_dir))
    registry = json.loads((root / "registry.json").read_text())
    rows = [dict(t) for entry in registry for t in entry["tasks"]]
    return root, rows


def _task_files(task_dir: Path) -> list[Path]:
    from harbor.publisher.packager import Packager

    return Packager.collect_files(task_dir)


def _task_config(task_dir: Path) -> dict:
    import tomllib

    return tomllib.loads((task_dir / "task.toml").read_text())


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return counts


@register
class TerminalBench21(Benchmark):
    id = "terminal-bench-2.1"
    metric = "pass@1 resolve rate"
    extra = "tbench"
    harness_packages = ("harbor",)
    dataset = "harborframework/terminal-bench-2.1"
    dataset_revision = "3e235dff6880252a587fa479c09bcb1e16edf2eb"

    # -- options -----------------------------------------------------------

    def opt(self, key: str, default: Any) -> Any:
        """A ``--option key=value`` (always a string on the CLI), cast like ``default``."""
        value = self.config.options.get(key)
        return default if value is None else type(default)(value)

    @property
    def agent(self) -> str:
        agent = self.opt("agent", "terminus-2")
        if agent not in AGENTS:
            raise SystemExit(f"{self.id}: agent must be one of {', '.join(AGENTS)}, not {agent!r}")
        return agent

    def needs_server(self) -> bool:
        return self.agent != "oracle"

    def sampling(self) -> dict[str, Any]:
        """Sampling options given on the CLI; unset ones fall back to the
        checkpoint's generation_config, which vLLM applies by default."""
        out = {}
        for key, cast in (("temperature", float), ("top_p", float), ("max_tokens", int)):
            if key in self.config.options:
                out[key] = cast(self.config.options[key])
        return out

    def excluded(self) -> dict[str, str]:
        """``--option exclude=a,b`` adds to (``exclude=none`` clears) EXCLUDED."""
        spec = self.opt("exclude", "")
        if spec == "none":
            return {}
        extra = {n: "excluded by --option exclude" for n in spec.split(",") if n}
        return {**EXCLUDED, **extra}

    # -- entry point -------------------------------------------------------

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        source, revision = self.dataset_source()
        root, rows = load_tasks(source, revision, self.config.output_dir / "dataset")
        names = [r["name"] for r in rows]
        digest = None
        if source == self.dataset and revision == self.dataset_revision:
            dirs = {r["name"]: root / r["path"] for r in rows}
            got = task_digest_map(dirs)
            bad = sorted(set(got) ^ set(TASK_DIGESTS)) + sorted(n for n in got if TASK_DIGESTS.get(n, got[n]) != got[n])
            digest = hashlib.sha256(",".join(sorted(got.values())).encode()).hexdigest()
            if bad or digest != TASKS_DIGEST:
                for n in bad[:10]:
                    files = [str(f.relative_to(dirs[n].resolve())) for f in _task_files(dirs[n])] if n in dirs else []
                    log.error("%s: %s digest %s != %s; files %s", self.id, n, got.get(n), TASK_DIGESTS.get(n), files)
                raise SystemExit(f"{self.id}: task digests of {source}@{revision} don't match the pinned dataset: {bad}")

        excluded = {n: why for n, why in self.excluded().items() if n in names}
        rows = [r for r in rows if r["name"] not in excluded]
        if pattern := self.opt("tasks", ""):
            rows = [r for r in rows if re.search(pattern, r["name"])]
        tasks = data.take(rows, self.config.limit, key="name")
        for t in tasks:
            t["dir"] = root / t["path"]
            t["image"] = _task_config(t["dir"]).get("environment", {}).get("docker_image", "")
        log.info("%s: %d tasks (%d excluded) x %d repeats, agent %s", self.id, len(tasks), len(excluded), self.repeats, self.agent)

        agent = self._agent_config(base_url, served_model_name)
        image_errors = self._prefetch_images(tasks)
        reports = asyncio.run(self._run_all(tasks, agent, image_errors))

        per_repeat = []
        for k in range(self.repeats):
            rs = [r for r in reports if r["repeat"] == k]
            resolved = sum(r["resolved"] for r in rs)
            per_repeat.append(
                {
                    "repeat": k,
                    "resolved": resolved,
                    "n": len(rs),
                    "resolve_rate": resolved / len(rs) if rs else 0.0,
                    "statuses": _count(r["status"] for r in rs),
                }
            )
        per_task = {t["name"]: sum(r["resolved"] for r in reports if r["task"] == t["name"]) for t in tasks}
        per_task_status = {t["name"]: _count(r["status"] for r in reports if r["task"] == t["name"]) for t in tasks}
        return {
            "value": sum(r["resolve_rate"] for r in per_repeat) / len(per_repeat) if per_repeat else 0.0,
            "n": len(tasks),
            "n_total": len(names),
            "excluded": excluded,
            "dataset": source,
            "dataset_revision": revision,
            "tasks_digest": digest,
            "agent": self.agent,
            "sampling": self.sampling(),
            "per_repeat": per_repeat,
            "pass_at_k": pass_at_k(
                ([r["resolved"] for r in reports if r["task"] == t["name"]] for t in tasks),
                self.repeats,
                "resolved in any of the k trials",
            ),
            "per_task_resolved": per_task,
            "per_task_status": per_task_status,
            "tasks": [t["name"] for t in tasks],
        }

    # -- setup -------------------------------------------------------------

    def _agent_config(self, base_url: str, served: str) -> dict:
        """Kwargs for harbor's ``AgentConfig``."""
        if self.agent == "oracle":
            return {"name": "oracle"}
        sampling = self.sampling()
        max_len = self._max_model_len(base_url, served)
        call_kwargs: dict[str, Any] = {"api_key": "EMPTY"}
        call_kwargs.update({k: v for k, v in sampling.items() if k != "temperature"})
        kwargs: dict[str, Any] = {
            "api_base": base_url,
            "temperature": sampling.get("temperature"),  # None: not sent
            "llm_call_kwargs": call_kwargs,
            # LiteLLM knows nothing about a locally served model.
            "model_info": {
                "max_input_tokens": max_len,
                "max_output_tokens": sampling.get("max_tokens", max_len),
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
        }
        if max_turns := self.opt("max_turns", 0):
            kwargs["max_turns"] = max_turns
        return {"name": "terminus-2", "model_name": f"hosted_vllm/{served}", "kwargs": kwargs}

    def _max_model_len(self, base_url: str, served: str) -> int:
        import httpx

        try:
            models = httpx.get(f"{base_url.rstrip('/')}/models", timeout=30).json()["data"]
            for m in models:
                if m.get("id") == served and m.get("max_model_len"):
                    return int(m["max_model_len"])
        except Exception:
            log.warning("%s: could not read max_model_len from %s", self.id, base_url, exc_info=True)
        return self.opt("max_model_len", 32768)

    def _backend(self) -> str:
        import os

        return self.opt("sandbox", "") or os.environ.get("GRANITE_EVALS_SANDBOX", "enroot")

    def _prefetch_images(self, tasks: list[dict]) -> dict[str, str]:
        """Import every task image into the enroot cache before any trial, so a
        slow import doesn't eat harbor's environment start timeout. Returns
        ``{image: error}`` for images that failed."""
        if self._backend() != "enroot":
            return {}
        from granite_evals.sandbox.enroot import ensure_squashfs

        images = sorted({t["image"] for t in tasks if t["image"]})
        errors: dict[str, str] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=IMAGE_IMPORT_WORKERS) as pool:
            futures = {pool.submit(ensure_squashfs, image): image for image in images}
            for f in concurrent.futures.as_completed(futures):
                if e := f.exception():
                    log.error("%s: importing %s failed: %s", self.id, futures[f], e)
                    errors[futures[f]] = type(e).__name__
        return errors

    # -- trials ------------------------------------------------------------

    async def _run_all(self, tasks: list[dict], agent: dict, image_errors: dict[str, str]) -> list[dict]:
        sem = asyncio.Semaphore(self.config.workers)

        async def one(task: dict, k: int) -> dict:
            async with sem, async_node_locks(port_locks(task["name"]), what=f"{task['name']} repeat {k}"):
                return await self._trial(task, k, agent, image_errors)

        # Repeat-major, so a partial run has whole repeats done first.
        jobs = [one(t, k) for k in range(self.repeats) for t in tasks]
        return list(await asyncio.gather(*jobs))

    async def _trial(self, task: dict, k: int, agent: dict, image_errors: dict[str, str]) -> dict:
        name = task["name"]
        repeat_dir = self.config.output_dir / f"repeat-{k}"
        tdir = repeat_dir / name
        summary = tdir / "granite.json"
        if summary.exists():
            return json.loads(summary.read_text())
        base = {"task": name, "repeat": k, "image": task["image"], "resolved": False, "reward": 0.0}
        if task["image"] in image_errors:
            return {**base, "status": f"error:image:{image_errors[task['image']]}"}

        attempts = self.opt("max_retries", 3) + 1
        report: dict = {}
        for attempt in range(attempts):
            if tdir.exists():
                old = repeat_dir / ".attempts" / f"{name}-{int(time.time())}-{attempt}"
                old.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(tdir, old)
            try:
                result = await self._run_trial(task, k, agent, repeat_dir)
            except Exception as e:  # one broken task must not sink the run
                log.exception("%s: %s repeat %d attempt %d failed", self.id, name, k, attempt)
                report = {**base, "status": f"error:{type(e).__name__}"}
                continue
            report = self._report(result, base)
            exc = result.exception_info
            if exc is None or exc.exception_type in FINAL_EXCEPTIONS:
                break
            log.warning("%s: %s repeat %d attempt %d: %s", self.id, name, k, attempt, exc.exception_type)
        report["attempts"] = attempt + 1
        log.info("%s repeat %d: %s %s reward=%s", self.id, k, name, report["status"], report["reward"])
        if not report["status"].startswith("error:"):
            # Infrastructure errors aren't persisted: a resumed run retries them.
            tdir.mkdir(parents=True, exist_ok=True)
            summary.write_text(json.dumps(report, indent=2))
        return report

    async def _run_trial(self, task: dict, k: int, agent: dict, repeat_dir: Path):
        from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, TrialConfig
        from harbor.trial.trial import Trial

        config = TrialConfig(
            task=TaskConfig(path=task["dir"]),
            trial_name=task["name"],
            trials_dir=repeat_dir,
            timeout_multiplier=self.opt("timeout_multiplier", 1.0),
            agent=AgentConfig(**agent),
            environment=EnvironmentConfig(
                import_path="granite_evals.sandbox.harbor_env:SandboxEnvironment",
                kwargs={"backend": self.opt("sandbox", "")},
                delete=True,
            ),
        )
        if agent["name"] == "terminus-2":
            config.agent.kwargs = {
                **config.agent.kwargs,
                "llm_call_kwargs": {**agent["kwargs"]["llm_call_kwargs"], "seed": self.config.seed + k},
            }
        trial = await Trial.create(config)
        return await trial.run()

    @staticmethod
    def _report(result, base: dict) -> dict:
        rewards = (result.verifier_result.rewards or {}) if result.verifier_result else {}
        reward = float(rewards.get("reward", 0.0) or 0.0)
        exc = result.exception_info
        status = "graded" if exc is None else exc.exception_type
        if exc is not None and exc.exception_type not in FINAL_EXCEPTIONS:
            status = f"error:{exc.exception_type}"
        ar = result.agent_result
        started, finished = getattr(result, "started_at", None), getattr(result, "finished_at", None)
        return {
            **base,
            "duration_sec": round((finished - started).total_seconds(), 1) if started and finished else None,
            "resolved": reward >= 1.0,
            "reward": reward,
            "status": status,
            "n_input_tokens": ar.n_input_tokens if ar else None,
            "n_output_tokens": ar.n_output_tokens if ar else None,
            "exception": exc.exception_message[:2000] if exc else None,
        }
