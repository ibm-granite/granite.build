"""Tools for NeMo-Skills tool-calling generation (``benchmarks/tools_agentic.py``).

NeMo-Skills' generate (``++tool_modules=[module::Class, ...]``) runs a chat loop
against the served model with vLLM tool calling (``ToolCallingWrapper``): every
tool call the model makes is executed and its result appended as a ``tool``
message, until the model answers without a tool call. These classes implement
ns's module ``Tool`` interface (``nemo_skills.mcp.tool_manager.Tool``: duck typed,
so this module imports without NeMo-Skills):

- ``SandboxPythonTool`` -> ``stateful_python_code_exec(code)``, the name and
  description of ns's own PythonTool, but run in a granite container sandbox
  (enroot) instead of ns's sandbox server: one container per run,
  one persistent Python process per rollout (Jupyter-like: state is kept between
  calls, the value of a trailing expression is printed). The process cuts its
  own network (``unshare(CLONE_NEWNET)``) before running any code unless
  ``network`` is "on"; with "off" (default) a sandbox that cannot isolate
  fails the run instead of silently allowing network.
- ``WebSearchTool`` -> ``web_search(query, num_results)``: Google results
  through a Google PSE MCP server (the ``google_pse_search`` tool, as
  BFCL web search via ``mcp_search.py``).
- ``FetchUrlTool`` -> ``fetch_url(url, start)``: one web page as text, in
  pages of ``max_chars`` characters.

Every tool reads its settings from one JSON file (``++tool_overrides.<Class>.
config=<path>``, written by the benchmark): keys ``python`` / ``search`` /
``fetch`` (see ``DEFAULTS``). URLs matching ``blocked_url_patterns`` (the
benchmark's own data: answer leakage) are neither returned by search nor
fetched. Tool errors are returned to the model as text; an unusable sandbox
raises ns's ``FatalToolError``, which stops generation.
"""

from __future__ import annotations

import asyncio
import atexit
import html.parser
import ipaddress
import json
import logging
import os
import queue
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit

log = logging.getLogger(__name__)

DEFAULTS: dict[str, dict[str, Any]] = {
    "python": {
        "image": "python:3.13-bookworm",
        "backend": "",  # "" = GRANITE_EVALS_SANDBOX (enroot); "unsafe-local" = no container (tests)
        "packages": [],  # pip requirement strings, installed once per run (with network)
        "timeout_s": 60.0,
        "network": "off",  # off: isolate or fail; best-effort: try; on: no isolation
        "max_output_chars": 20000,
        "probe_file": "",  # where the sandbox's network isolation is recorded
    },
    "search": {
        "mcp_url": "",
        "num_results": 10,
        "max_results": 10,
    },
    "fetch": {
        "max_chars": 20000,
        "timeout_s": 30.0,
        "max_bytes": 5_000_000,
        "allow_private_hosts": False,  # tests only: the fake page server is on localhost
    },
    "blocked_url_patterns": [],
}


def load_config(overrides: dict | None) -> dict[str, Any]:
    """The merged tool settings: ``DEFAULTS`` updated from the JSON file named by
    the ``config`` tool override (tool_overrides.<Class>.config)."""
    cfg = json.loads(json.dumps(DEFAULTS))
    path = (overrides or {}).get("config")
    if path:
        for key, value in json.loads(Path(path).read_text()).items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg


def _blocked(url: str, patterns: list[str]) -> bool:
    return any(re.search(p, url) for p in patterns)


def _fatal(msg: str) -> Exception:
    try:
        from nemo_skills.mcp.tool_manager import FatalToolError
    except ImportError:  # without ns (unit tests of this module)
        return RuntimeError(msg)
    return FatalToolError(msg)


def truncate_middle(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head, tail = text[: limit // 2], text[-(limit // 2) :]
    return f"{head}\n[... {len(text) - len(head) - len(tail)} characters truncated ...]\n{tail}"


class _Metrics:
    """Per-request counters, reported by ns as ``tool_metrics`` in each output row."""

    def __init__(self):
        self.by_request: dict[str, Counter] = defaultdict(Counter)
        self.lock = threading.Lock()

    def add(self, request_id: str, **counts: int) -> None:
        with self.lock:
            self.by_request[request_id].update(counts)

    def pop(self, request_id: str) -> dict[str, int]:
        with self.lock:
            return dict(self.by_request.pop(request_id, {}))


# -- python ---------------------------------------------------------------------

_REPL_SOURCE = r'''
"""granite persistent Python session: JSON lines in (code, timeout), JSON lines out."""
import ast, json, linecache, os, signal, sys, traceback


def isolate(mode):
    if mode == "on":
        return "not requested"
    try:
        if hasattr(os, "unshare"):
            os.unshare(os.CLONE_NEWNET)
        else:
            import ctypes
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.unshare(0x40000000) != 0:
                err = ctypes.get_errno()
                raise OSError(err, os.strerror(err))
    except Exception as e:
        return "failed: %s: %s" % (type(e).__name__, e)
    return "isolated"


class Timeout(BaseException):
    pass


def on_alarm(signum, frame):
    raise Timeout()


def run_cell(code, g, name):
    linecache.cache[name] = (len(code), None, code.splitlines(True), name)
    tree = ast.parse(code, name, "exec")
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, name, "exec"), g)
    if last is not None:
        value = eval(compile(last, name, "eval"), g)
        if value is not None:
            g["_"] = value
            print(repr(value))


def main():
    mode, capture, max_bytes = sys.argv[1], sys.argv[2], int(sys.argv[3])
    proto_in = os.fdopen(os.dup(0), "r", encoding="utf-8")
    proto_out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    network = isolate(mode)
    hello = {"ready": mode != "off" or network == "isolated", "network": network,
             "python": sys.version.split()[0]}
    if not hello["ready"]:
        proto_out.write(json.dumps(hello) + "\n")
        proto_out.flush()
        return
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    cap = os.open(capture, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
    os.dup2(cap, 1)
    os.dup2(cap, 2)
    sys.stdin = open(os.devnull)
    sys.stdout = os.fdopen(1, "w", encoding="utf-8", errors="replace", closefd=False)
    sys.stderr = os.fdopen(2, "w", encoding="utf-8", errors="replace", closefd=False)
    signal.signal(signal.SIGALRM, on_alarm)
    proto_out.write(json.dumps(hello) + "\n")
    proto_out.flush()
    g = {"__name__": "__main__", "__builtins__": __builtins__}
    for n, line in enumerate(proto_in):
        req = json.loads(line)
        os.ftruncate(cap, 0)
        os.lseek(cap, 0, 0)
        status = "ok"
        try:
            signal.setitimer(signal.ITIMER_REAL, float(req["timeout"]))
            run_cell(req["code"], g, "<cell-%d>" % (n + 1))
        except Timeout:
            status = "timeout"
            print("TimeoutError: execution timed out after %s seconds" % req["timeout"], file=sys.stderr)
        except BaseException as e:  # SystemExit and KeyboardInterrupt included: the session lives on
            status = "error"
            tb = e.__traceback__
            while tb is not None and tb.tb_frame.f_code.co_filename == __file__:
                tb = tb.tb_next
            sys.stderr.write("".join(traceback.format_exception(type(e), e, tb)))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                pass
        size = os.lseek(cap, 0, 2)
        os.lseek(cap, 0, 0)
        data = b""
        while len(data) < min(size, max_bytes):
            chunk = os.read(cap, min(size, max_bytes) - len(data))
            if not chunk:
                break
            data += chunk
        out = data.decode("utf-8", "replace")
        if size > max_bytes:
            out += "\n[... output truncated: %d bytes in all ...]" % size
        proto_out.write(json.dumps({"status": status, "output": out}) + "\n")
        proto_out.flush()


main()
'''


def _local_sandbox(image: str):
    """No container at all: commands run on the host, in a temp dir. For tests
    only (the ``unsafe-local`` backend); never isolates anything."""
    from granite_evals.sandbox import Sandbox
    from granite_evals.sandbox.enroot import SECRET_NAME

    class LocalSandbox(Sandbox):
        workdir = ""

        def start(self) -> None:
            self.workdir = tempfile.mkdtemp(prefix="granite-local-sandbox-")

        def _exec_argv(self, command: str, cwd: str, env: dict[str, str]) -> list[str]:
            exports = "".join(f"export {k}={shlex.quote(v)}; " for k, v in env.items())
            return ["bash", "-c", f"{exports}cd {shlex.quote(cwd)} && {command}"]

        def _run_env(self) -> dict[str, str]:
            return {k: v for k, v in os.environ.items() if not SECRET_NAME.search(k)}

        def close(self) -> None:
            shutil.rmtree(self.workdir, ignore_errors=True)

    return LocalSandbox(image)


class _Session:
    """One persistent Python process in the sandbox (one rollout's state)."""

    def __init__(self, sandbox, root: str, sid: int, cfg: dict[str, Any]):
        self.cfg = cfg
        workdir = f"{root}/session-{sid}"
        cmd = (
            f"mkdir -p {workdir} && cd {workdir} && exec python3 -u {root}/granite_repl.py "
            f"{shlex.quote(cfg['network'])} {workdir}/.output {4 * int(cfg['max_output_chars']) + 4096}"
        )
        env = {"MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1", "HOME": workdir}
        argv = sandbox._exec_argv(cmd, "/", {**sandbox.env, **env})
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=sandbox._run_env(),
            start_new_session=True,
        )
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()
        self.hello = self._reply(timeout=300)

    def _read(self) -> None:
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def _reply(self, timeout: float) -> dict | None:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            return None
        return None if line is None else json.loads(line)

    def run(self, code: str, timeout: float) -> tuple[str, str]:
        """(status, output); status ok / error / timeout / dead."""
        try:
            self.proc.stdin.write(json.dumps({"code": code, "timeout": timeout}) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return "dead", "The Python process has exited; its state is lost."
        reply = self._reply(timeout=timeout + 30)
        if reply is None:
            exited = self.proc.poll() is not None
            self.close()
            if exited:
                return "dead", "The Python process exited; its state is lost."
            return "dead", (
                f"TimeoutError: execution did not stop after {timeout} seconds; "
                "the Python process was killed and its state is lost."
            )
        return reply["status"], reply["output"]

    @property
    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                self.proc.kill()
            self.proc.wait()


_PROBE = """import socket
try:
    socket.create_connection(("1.1.1.1", 53), timeout=3).close()
    print("reachable")
except OSError:
    print("unreachable")"""


class SandboxPythonTool:
    """``stateful_python_code_exec`` in a granite sandbox (module docstring)."""

    def __init__(self) -> None:
        self.cfg: dict[str, Any] = dict(DEFAULTS["python"])
        self.sandbox = None
        self.root = "/tmp"
        self.sessions: dict[str, _Session] = {}
        self.lock = threading.Lock()
        self.start_error: str | None = None
        self.metrics = _Metrics()
        self._sid = 0

    def default_config(self) -> dict[str, Any]:
        return {"config": ""}

    def configure(self, overrides: dict | None = None, context: dict | None = None) -> None:
        self.cfg = load_config(overrides)["python"]
        if self.cfg["network"] not in ("off", "best-effort", "on"):
            raise ValueError(f"python network must be off, best-effort or on, not {self.cfg['network']!r}")

    def post_configure(self) -> None:
        return None

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "stateful_python_code_exec",
                "description": (
                    "Call this function to execute Python code in a stateful Jupyter notebook environment. "
                    "Python will respond with the output of the execution or time out after "
                    f"{float(self.cfg['timeout_s']):.1f} seconds."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "Code to execute"}},
                    "required": ["code"],
                },
            }
        ]

    async def execute(self, tool_name: str, arguments: dict, extra_args: dict | None = None) -> str:
        request_id = str((extra_args or {}).get("request_id") or "default")
        code = arguments.get("code")
        if not isinstance(code, str):
            self.metrics.add(request_id, calls=1, errors=1)
            return "Error: missing required argument 'code'"
        return await asyncio.to_thread(self._execute, request_id, code)

    def _execute(self, request_id: str, code: str) -> str:
        self._ensure_sandbox()
        with self.lock:
            session = self.sessions.get(request_id)
            if session is None or not session.alive:
                self._sid += 1
                session = self.sessions[request_id] = _Session(self.sandbox, self.root, self._sid, self.cfg)
                if not (session.hello or {}).get("ready"):
                    raise _fatal(f"python sandbox session did not start: {session.hello}")
        status, output = session.run(code, float(self.cfg["timeout_s"]))
        self.metrics.add(request_id, calls=1, **({status: 1} if status != "ok" else {}))
        output = truncate_middle(output, int(self.cfg["max_output_chars"]))
        return output.removesuffix("\n")

    def _ensure_sandbox(self) -> None:
        with self.lock:
            if self.start_error:
                raise _fatal(self.start_error)
            if self.sandbox is not None:
                return
            try:
                self._start_sandbox()
            except Exception as e:
                self.start_error = f"python sandbox ({self.cfg['image']}) failed to start: {type(e).__name__}: {e}"
                log.exception("python sandbox failed")
                raise _fatal(self.start_error) from e

    def _start_sandbox(self) -> None:
        backend = self.cfg["backend"]
        started = time.time()
        if backend == "unsafe-local":
            sb = _local_sandbox(self.cfg["image"])
        else:
            from granite_evals.sandbox import make_sandbox

            sb = make_sandbox(self.cfg["image"], backend=backend)
        sb.start()
        self.sandbox = sb
        atexit.register(self.close)
        self.root = getattr(sb, "workdir", "") or "/tmp"
        if self.cfg["packages"]:
            pkgs = " ".join(shlex.quote(p) for p in self.cfg["packages"])
            r = sb.execute(
                "python3 -m pip install --quiet --no-cache-dir --disable-pip-version-check "
                f"--root-user-action=ignore {pkgs}",
                timeout=1800,
            )
            if r.returncode:
                raise RuntimeError(f"pip install failed: {r.output[-2000:]}")
        sb.write_file(f"{self.root}/granite_repl.py", _REPL_SOURCE)
        probe = _Session(sb, self.root, 0, self.cfg)
        try:
            hello = probe.hello or {}
            ok, net = probe.run(_PROBE, 30) if hello.get("ready") else ("not started", "")
        finally:
            probe.close()
        record = {
            "image": self.cfg["image"],
            "backend": backend or os.environ.get("GRANITE_EVALS_SANDBOX", "enroot"),
            "packages": self.cfg["packages"],
            "network_mode": self.cfg["network"],
            "network_isolation": hello.get("network"),
            "network_probe": net.strip() if ok == "ok" else f"{ok}: {net[-200:]}",
            "python": hello.get("python"),
            "ready": hello.get("ready", False),
            "startup_seconds": round(time.time() - started, 1),
        }
        log.info("python sandbox: %s", json.dumps(record))
        if self.cfg["probe_file"]:
            Path(self.cfg["probe_file"]).write_text(json.dumps(record, indent=2) + "\n")
        if not record["ready"]:
            raise RuntimeError(
                f"cannot isolate the sandbox's network ({hello.get('network')}); "
                "use python_network=best-effort or on to allow it"
            )

    async def cleanup_request(self, request_id: str) -> None:
        with self.lock:
            session = self.sessions.pop(request_id, None)
        if session:
            session.close()

    async def get_request_metrics(self, request_id: str) -> dict[str, Any]:
        return self.metrics.pop(request_id)

    async def shutdown(self) -> None:
        self.close()

    def close(self) -> None:
        with self.lock:
            sessions, self.sessions = list(self.sessions.values()), {}
            sb, self.sandbox = self.sandbox, None
        for s in sessions:
            s.close()
        if sb is not None:
            sb.close()


# -- web search -----------------------------------------------------------------


class WebSearchTool:
    """``web_search`` through the Google PSE MCP server (module docstring)."""

    def __init__(self) -> None:
        self.cfg: dict[str, Any] = dict(DEFAULTS["search"])
        self.blocked: list[str] = []
        self.client = None
        self.metrics = _Metrics()

    def default_config(self) -> dict[str, Any]:
        return {"config": ""}

    def configure(self, overrides: dict | None = None, context: dict | None = None) -> None:
        cfg = load_config(overrides)
        self.cfg, self.blocked = cfg["search"], list(cfg["blocked_url_patterns"])

    def post_configure(self) -> None:
        return None

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "web_search",
                "description": "Search the web with Google. Returns the top results: title, URL and a snippet each.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "num_results": {
                            "type": "integer",
                            "description": f"Number of results (1-{self.cfg['max_results']}, "
                            f"default {self.cfg['num_results']})",
                        },
                    },
                    "required": ["query"],
                },
            }
        ]

    async def execute(self, tool_name: str, arguments: dict, extra_args: dict | None = None) -> str:
        request_id = str((extra_args or {}).get("request_id") or "default")
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            self.metrics.add(request_id, calls=1, errors=1)
            return "Error: missing required argument 'query'"
        try:
            n = int(arguments.get("num_results") or self.cfg["num_results"])
        except (TypeError, ValueError):
            n = int(self.cfg["num_results"])
        n = max(1, min(n, int(self.cfg["max_results"])))
        try:
            hits = await asyncio.to_thread(self._search, query, n)
        except Exception as e:
            self.metrics.add(request_id, calls=1, errors=1)
            return f"Error: web search failed: {e}"
        kept = [h for h in hits if not _blocked(h["href"], self.blocked)]
        self.metrics.add(request_id, calls=1, blocked_results=len(hits) - len(kept), **({} if kept else {"empty": 1}))
        if not kept:
            return f"No results for: {query}"
        lines = [f"Results for: {query}", ""]
        for i, h in enumerate(kept, 1):
            lines += [f"{i}. {h['title']}", f"URL: {h['href']}", h["body"], ""]
        return "\n".join(lines).rstrip()

    def _search(self, query: str, n: int) -> list[dict[str, str]]:
        from granite_evals.mcp_search import MCPSearch

        if self.client is None:
            if not self.cfg["mcp_url"]:
                raise RuntimeError("web_search needs an MCP search endpoint; pass --option search_mcp_url=<URL>")
            self.client = MCPSearch(self.cfg["mcp_url"])
        return self.client.search(query, n)

    async def cleanup_request(self, request_id: str) -> None:
        return None

    async def get_request_metrics(self, request_id: str) -> dict[str, Any]:
        return self.metrics.pop(request_id)

    async def shutdown(self) -> None:
        return None


# -- page fetch -----------------------------------------------------------------


class _TextExtractor(html.parser.HTMLParser):
    """Visible text of an HTML page, one block element per line."""

    SKIP = frozenset({"script", "style", "noscript", "template", "svg", "head"})
    BLOCK = frozenset({"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article",
             "header", "footer", "table", "ul", "ol", "pre", "blockquote", "dt", "dd", "hr"})  # fmt: skip

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" | ")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if not self.skip:
            self.parts.append(data)


def html_to_text(page: str) -> tuple[str, str]:
    """(title, text) of an HTML page."""
    p = _TextExtractor()
    p.feed(page)
    p.close()
    lines = (re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in "".join(p.parts).split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return p.title.strip(), text


def _private_host(host: str) -> bool:
    """Whether ``host`` is, or resolves to, a non-public address (no requests to
    the job's internal network: the judge gateway, cluster services)."""
    try:
        addrs = {ipaddress.ip_address(host)}
    except ValueError:
        try:
            addrs = {ipaddress.ip_address(a[4][0].split("%")[0]) for a in socket.getaddrinfo(host, None)}
        except socket.gaierror:
            return False  # unresolvable here: a proxy may resolve it; never a local address
    return any(not a.is_global for a in addrs)


class FetchUrlTool:
    """``fetch_url``: a web page as text (module docstring)."""

    def __init__(self) -> None:
        self.cfg: dict[str, Any] = dict(DEFAULTS["fetch"])
        self.blocked: list[str] = []
        self.metrics = _Metrics()
        self.cache: dict[str, tuple[str, str, str]] = {}
        self.lock = threading.Lock()

    def default_config(self) -> dict[str, Any]:
        return {"config": ""}

    def configure(self, overrides: dict | None = None, context: dict | None = None) -> None:
        cfg = load_config(overrides)
        self.cfg, self.blocked = cfg["fetch"], list(cfg["blocked_url_patterns"])

    def post_configure(self) -> None:
        return None

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "fetch_url",
                "description": (
                    "Fetch a web page (http or https URL) and return its text content, "
                    f"at most {self.cfg['max_chars']} characters per call. "
                    "For a longer page, call again with `start` set to the offset where the previous part ended."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to fetch"},
                        "start": {"type": "integer", "description": "Character offset to start from (default 0)"},
                    },
                    "required": ["url"],
                },
            }
        ]

    async def execute(self, tool_name: str, arguments: dict, extra_args: dict | None = None) -> str:
        request_id = str((extra_args or {}).get("request_id") or "default")
        url = arguments.get("url")
        if not isinstance(url, str) or not url.strip():
            self.metrics.add(request_id, calls=1, errors=1)
            return "Error: missing required argument 'url'"
        try:
            start = max(0, int(arguments.get("start") or 0))
        except (TypeError, ValueError):
            start = 0
        try:
            final, title, text = await asyncio.to_thread(self._get, url.strip())
        except _Refused as e:
            self.metrics.add(request_id, calls=1, refused=1)
            return f"Error: {e}"
        except Exception as e:
            self.metrics.add(request_id, calls=1, errors=1)
            return f"Error: fetching {url} failed: {type(e).__name__}: {e}"
        self.metrics.add(request_id, calls=1)
        n = int(self.cfg["max_chars"])
        part = text[start : start + n]
        head = [f"URL: {final}"] + ([f"Title: {title}"] if title else [])
        end = start + len(part)
        if start or end < len(text):
            more = f"; call again with start={end} for more" if end < len(text) else ""
            head.append(f"[characters {start}-{end} of {len(text)}{more}]")
        return "\n".join(head) + "\n\n" + (part or "(no text at this offset)")

    def _get(self, url: str) -> tuple[str, str, str]:
        with self.lock:
            if url in self.cache:
                return self.cache[url]
        import httpx

        current = url
        with httpx.Client(timeout=float(self.cfg["timeout_s"]), follow_redirects=False,
                          headers={"User-Agent": "Mozilla/5.0 (compatible; granite-evals)"}) as client:  # fmt: skip
            for _ in range(6):
                self._check(current)
                with client.stream("GET", current) as r:
                    if r.is_redirect and r.headers.get("location"):
                        current = urljoin(current, r.headers["location"])
                        continue
                    r.raise_for_status()
                    body = b""
                    for chunk in r.iter_bytes():
                        body += chunk
                        if len(body) > int(self.cfg["max_bytes"]):
                            break
                    ctype = r.headers.get("content-type", "").lower()
                    charset = r.charset_encoding or "utf-8"
                break
            else:
                raise _Refused(f"too many redirects from {url}")
        if "pdf" in ctype or body[:5] == b"%PDF-":
            title, text = "", _pdf_text(body)
        else:
            page = body.decode(charset, "replace")
            title, text = html_to_text(page) if ("html" in ctype or "<html" in page[:2000].lower()) else ("", page)
        result = (current, title, text)
        with self.lock:
            self.cache[url] = result
        return result

    def _check(self, url: str) -> None:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise _Refused(f"only http(s) URLs can be fetched: {url}")
        if _blocked(url, self.blocked):
            raise _Refused(f"{url} is blocked in this evaluation")
        if not self.cfg["allow_private_hosts"] and _private_host(parts.hostname):
            raise _Refused(f"{parts.hostname} is not a public address")

    async def cleanup_request(self, request_id: str) -> None:
        return None

    async def get_request_metrics(self, request_id: str) -> dict[str, Any]:
        return self.metrics.pop(request_id)

    async def shutdown(self) -> None:
        return None


class _Refused(Exception):
    pass


def _pdf_text(body: bytes) -> str:
    try:
        import io

        import pypdf
    except ImportError:
        return "(a PDF document; this tool cannot extract its text)"
    reader = pypdf.PdfReader(io.BytesIO(body))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)
