"""Start and stop a local vLLM OpenAI-compatible server for one step.

Same shape as granite.build's run-bfcl.sh: the step owns its server, waits for
``/v1/models`` to answer, and tears the whole process group down on exit, so
a crashed eval never leaves GPUs held on the node.

Parsers set to ``auto`` are resolved from the checkpoint's own files, since
vLLM has no "auto": the tool-call parser from the chat template's tool-call
format, the reasoning parser from a ``*_parser.py`` vLLM plugin shipped with
the model (Granite 4.2 ships ``granite_thinking_parser.py``). That keeps
distilled checkpoints, which carry the teacher's template, on the same parsers.
"""

from __future__ import annotations

import logging
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    model: str
    served_model_name: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    # "auto" resolves from the checkpoint (module docstring); "" disables. Granite
    # 4.2 resolves to qwen3_coder + its granite_thinking_parser plugin, as in the
    # model card's "Serving with vLLM" section.
    tool_call_parser: str = "auto"
    reasoning_parser: str = "auto"
    extra_args: list[str] = field(default_factory=list)
    port: int = 0
    """0 picks a free port."""
    startup_timeout_s: int = 1800


def model_dir(model: str) -> Path | None:
    """The checkpoint's directory: ``model`` itself, or the hub snapshot of its
    small files (vLLM downloads the weights into the same cache)."""
    if Path(model).is_dir():
        return Path(model)
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(model, allow_patterns=["*.json", "*.jinja", "*.py"]))
    except Exception as e:  # noqa: BLE001 - no metadata just means no parsers
        log.warning("cannot fetch %s metadata for parser detection: %s", model, e)
        return None


def _chat_template(d: Path) -> str:
    if (d / "chat_template.jinja").is_file():
        return (d / "chat_template.jinja").read_text()
    try:
        t = json.loads((d / "tokenizer_config.json").read_text()).get("chat_template") or ""
    except (OSError, ValueError):
        return ""
    return t if isinstance(t, str) else " ".join(x.get("template", "") for x in t)


def detect_tool_call_parser(d: Path) -> str:
    template = _chat_template(d)
    if "<function=" in template:  # <tool_call><function=f><parameter=p>...: Qwen3-Coder XML
        return "qwen3_coder"
    if "<tool_call>" in template:  # <tool_call>{"name": ..., "arguments": ...}
        return "hermes"
    return ""


def detect_reasoning_plugin(d: Path) -> tuple[str, Path] | None:
    """(parser name, plugin file) of a reasoning parser the checkpoint ships."""
    for f in sorted(d.glob("*_parser.py")):
        m = re.search(r"ReasoningParserManager\.register_module\(\s*[\"']([^\"']+)", f.read_text())
        if m:
            return m.group(1), f
    return None


def server_env() -> dict[str, str]:
    """vLLM's environment. The image has no CUDA toolkit, so nothing may JIT
    CUDA code: FlashInfer's top-k/top-p sampler does on first use (it needs
    nvcc), vLLM's PyTorch sampler doesn't. Set in code, not as image ENV,
    which granite.build's SLURM provider can drop."""
    return {"VLLM_USE_FLASHINFER_SAMPLER": "0", **os.environ}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class VLLMServer:
    def __init__(self, config: ServerConfig, log_path: Path):
        self.config = config
        self.log_path = log_path
        self.port = config.port or _free_port()
        self.base_url = f"http://127.0.0.1:{self.port}/v1"
        self._proc: subprocess.Popen | None = None

    def command(self) -> list[str]:
        c = self.config
        cmd = [
            "vllm", "serve", c.model,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--served-model-name", c.served_model_name,
            "--tensor-parallel-size", str(c.tensor_parallel_size),
            "--gpu-memory-utilization", str(c.gpu_memory_utilization),
            "--trust-remote-code",
        ]  # fmt: skip
        if c.max_model_len:
            cmd += ["--max-model-len", str(c.max_model_len)]
        tool, reasoning, plugin = c.tool_call_parser, c.reasoning_parser, None
        d = model_dir(c.model) if "auto" in (tool, reasoning) else None
        if tool == "auto":
            tool = detect_tool_call_parser(d) if d else ""
        if reasoning == "auto":
            reasoning, plugin = (detect_reasoning_plugin(d) if d else None) or ("", None)
        if tool:
            cmd += ["--enable-auto-tool-choice", "--tool-call-parser", tool]
        if reasoning:
            cmd += ["--reasoning-parser", reasoning]
        if plugin:
            cmd += ["--reasoning-parser-plugin", str(plugin)]
        return cmd + c.extra_args

    def __enter__(self) -> VLLMServer:
        cmd = self.command()
        log.info("starting vLLM: %s (log: %s)", shlex.join(cmd), self.log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        logfile = self.log_path.open("ab")
        self._proc = subprocess.Popen(
            cmd, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True, env=server_env()
        )
        self._wait_ready()
        return self

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM exited with code {self._proc.returncode} before becoming ready; "
                    f"see {self.log_path}"
                )
            try:
                if httpx.get(f"{self.base_url}/models", timeout=5).status_code == 200:
                    log.info("vLLM ready at %s", self.base_url)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(5)
        self.__exit__(None, None, None)
        raise TimeoutError(f"vLLM not ready after {self.config.startup_timeout_s}s; see {self.log_path}")

    def __exit__(self, *exc) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        pgid = os.getpgid(self._proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            self._proc.wait()
