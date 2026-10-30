"""HumanEval and MBPP (with and without tools): pass@1 by running the model's
code on HumanEval's own execution harness, inside a granite sandbox.

Both ride the shared ``NemoSkillsBenchmark`` (``nemo_skills.py``): ns's generic
code-generation prompt (``generic/codegen``), generation loop, repeats, phases,
``answers=gold`` and metrics. What is NOT ns's is these two benchmarks' data
preparation and evaluation: ns's ``human-eval`` / ``mbpp`` dataset modules read
EvalPlus's GitHub-release data, re-download it at evaluation time and grade
inside the job process (``evalplus.evaluate``) — a pinned-read and a
sandboxed-execution violation. Instead:

- prepare: the official HF datasets (``openai/openai_humaneval``, MIT, 164
  problems; ``google-research-datasets/mbpp`` config ``sanitized``, CC-BY-4.0,
  257 problems), pinned to a commit and count-checked, mapped to ns's row
  shape: the same fields ns's own modules write (``question`` carries ns's tab
  quirk; MBPP's ``prompt`` is evalplus's docstring form: the problem and its
  first test inside one module docstring). MBPP's ``sanitized`` ``test``
  split is the 257 manually-verified problems the MBPP paper evaluates on (the
  same sanitized set evalplus's 378-problem MBPP+ builds on); the ``full``
  500-problem split is reachable with ``--dataset`` and a locally prepared
  jsonl. No short-circuit: every phase re-reads (the HF cache holds the bytes).
- generate: ns's ``generic/codegen`` prompt with ``++eval_type=null`` baked in
  — no ns evaluator runs inside generation, in any phase.
- score: each completion (ns's ``preprocess_code`` code-block extraction, the
  completion's own indentation kept) becomes a program exactly as the
  benchmark's own harness assembles it — HumanEval's
  ``prompt + completion + "\\n" + test + "\\n" + check(entry_point)``, or the
  completion alone when it defines ``entry_point`` at top level (evalplus's
  ``solution`` mode: chat models emit the whole function, and a raw concat
  would IndentationError); MBPP: always the docstring prompt + completion +
  tests — and runs in ONE granite container per scoring phase (enroot on SLURM,
  podman / docker locally, ``grade_image``, default python:3.13-bookworm:
  stdlib only) through a port of HumanEval's own ``execution.py`` (MIT): a
  fresh process per problem with its own tempdir, ``reliability_guard``,
  swallowed io and a SIGALRM time limit; the network namespace cut
  (``unshare(CLONE_NEWNET)``) unless ``grade_network`` allows it. Model code
  never runs outside that sandbox.
- metrics: ns's ``ComputeMetrics`` over ``PassingTestsMetrics`` (this module):
  ``passing_tests`` from each row's ``is_correct``, pass@1 / pass@k exactly as
  ns's own code benchmarks score them.

Port notes (upstream ``execution.py`` at HUMANEVAL_EXEC_COMMIT, fetched
verbatim and adapted): ``__builtins__["help"] = None`` is ``builtins.help =
None`` here (the harness runs as a script, where ``__builtins__`` is the
builtins module, not its ``__dict__``, and the subscript would TypeError); a
killed process is reaped; the check program arrives pre-assembled (the
standalone-vs-concat decision is the host's, and one problem means one
completion, so upstream's ``completion_id`` matching is a task_id map); no
memory rlimit is passed (neither does upstream's own CLI); the per-problem
waits are overlapped on a thread pool, as upstream's evaluate.py does.

Options (besides the ns ones): ``grade_image`` (default
python:3.13-bookworm), ``sandbox`` (the grading container's backend: enroot /
podman / docker, or ``unsafe-local`` for tests — the same option the tools
variants' python tool reads), ``grade_timeout`` (seconds per problem, 10),
``grade_workers`` (grading processes in the container, 8), ``grade_network``
(off / best-effort / on, like the python tool's network modes).
"""

from __future__ import annotations

import ast
import contextlib
import json
import logging
import os
import shlex
import time
from pathlib import Path
from typing import Any, ClassVar, Iterator

from granite_evals import data
from granite_evals.benchmarks.nemo_skills import (
    NemoSkillsBenchmark,
    _read_jsonl,
    _sha256,
    _unscored,
    _write_jsonl,
    log,
)
from granite_evals.registry import register

# -- pins -------------------------------------------------------------------------

HUMANEVAL_DATASET = "openai/openai_humaneval"
HUMANEVAL_REVISION = "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544"
HUMANEVAL_N = 164
"""HumanEval proper (164 problems, MIT): the five fields the harness needs.
NOT gated on HF (the task brief said gated; the repo is open, MIT-licensed)."""

MBPP_DATASET = "google-research-datasets/mbpp"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
MBPP_CONFIG = "sanitized"
MBPP_N = 257
"""MBPP's sanitized test split (257 manually-verified problems, CC-BY-4.0)."""

HUMANEVAL_EXEC_COMMIT = "37c4dd63798c3c9ba32fa69a2fb49c5e2c43a181"
"""openai/human-eval commit of the ported human_eval/execution.py (MIT)."""

GRADE_IMAGE = "python:3.13-bookworm"
"""The grading container: stdlib only, nothing to pip-install (both harnesses
and both benchmarks' tests are pure stdlib Python)."""

# -- the grading runner -----------------------------------------------------------

GRADE_RUNNER = r'''"""granite codegen grader: HumanEval's own execution harness (MIT), ported.

JSON lines in ({"task_id", "program"} per row), JSON lines out (a hello line
with {"ready", "network", "python"}, then one {"task_id", "passed", "result"}
per row). argv: <network-mode> <rows> <out> <workers> <timeout-seconds>.
"""
import contextlib
import faulthandler
import io
import json
import multiprocessing
import os
import platform
import signal
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor


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


# -- openai/human-eval human_eval/execution.py (MIT), ported -------------------
# Adapted: the program arrives pre-assembled (the standalone-vs-prompt decision
# is the host's); `__builtins__["help"] = None` is `builtins.help = None` (this
# runs as a script, where __builtins__ is the module, not its __dict__); the
# killed process is reaped.


def unsafe_execute(problem, timeout, result):
    with create_tempdir():

        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()

        try:
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    # WARNING
                    # This program exists to execute untrusted model-generated code. Although
                    # it is highly unlikely that model-generated code will do something overtly
                    # malicious in response to this test suite, model-generated code may act
                    # destructively due to a lack of model capability or alignment.
                    # Users are strongly encouraged to sandbox this evaluation suite so that it
                    # does not perform destructive actions on their host or network. For more
                    # information on how OpenAI sandboxes its code, see the accompanying paper.
                    # Once you have read this disclaimer and taken appropriate precautions,
                    # uncomment the following line and proceed at your own risk:
                    exec(problem["program"], exec_globals)
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append("failed: %s" % e)

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


def check_correctness(problem, timeout):
    """Evaluates the functional correctness of a completion by running the test
    suite provided in the problem (upstream doc)."""
    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute, args=(problem, timeout, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    p.join()

    if not result:
        result.append("timed out")

    return {"task_id": problem["task_id"], "passed": result[0] == "passed", "result": result[0]}


@contextlib.contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read from."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes=None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    builtins.help = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def main():
    mode, rows_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    workers, timeout = int(sys.argv[4]), float(sys.argv[5])
    network = isolate(mode)
    with open(out_path, "w", encoding="utf-8") as out:
        hello = {"ready": mode != "off" or network == "isolated", "network": network,
                 "python": sys.version.split()[0]}
        out.write(json.dumps(hello) + "\n")
        out.flush()
        if not hello["ready"]:
            return
        with open(rows_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        if not rows:
            return
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            results = list(pool.map(lambda row: check_correctness(row, timeout), rows))
        for r in results:
            out.write(json.dumps(r) + "\n")


if __name__ == "__main__":  # the spawn start method re-imports this file in children
    main()
'''

# -- metrics ------------------------------------------------------------------------

try:  # ns's BaseMetrics with this family's passing_tests score
    from nemo_skills.evaluation.metrics.base import BaseMetrics as _BaseMetrics
except ImportError:  # without the nemoskills extra: the class is only used inside ns
    _BaseMetrics = object


class PassingTestsMetrics(_BaseMetrics):
    """ns metrics over this run's grade (ns's EvalPlusMetrics shape): pass@1 /
    pass@k of the fraction of problems whose program passed its tests."""

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {"passing_tests": prediction["is_correct"]}

    def get_incorrect_sample(self, prediction: dict) -> dict:
        return {"is_correct": False}

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)


# -- rows and programs ----------------------------------------------------------------


def humaneval_row(r: dict) -> dict:
    """One ns row from a raw ``openai_humaneval`` row: the five harness fields
    plus ``question`` (ns's prepare quirk: the prompt with 4 spaces -> tabs,
    which is what the model sees; grading uses the raw ``prompt``)."""
    row = {
        k: r[k]
        for k in ("task_id", "prompt", "canonical_solution", "test", "entry_point")
    }
    row["question"] = row["prompt"].replace("    ", "\t")
    return row


def mbpp_row(r: dict) -> dict:
    """One ns row from a raw sanitized ``mbpp`` row: evalplus's shape — the
    prompt is the docstring it builds (the problem and its first test inside
    one module docstring, MbppPlus's own prompt field), the tests are its
    imports plus all three asserts, the reference is its full ``code``
    function."""
    prompt = f'"""\n{r["prompt"]}\n{r["test_list"][0]}\n"""\n'
    tests = [*list(r.get("test_imports") or []), *list(r["test_list"])]
    return {
        "task_id": r["task_id"],
        "prompt": prompt,
        "canonical_solution": r["code"],
        "test": "\n".join(tests),
        "question": prompt.replace("    ", "\t"),
    }


def _standalone(completion: str, entry_point: str) -> bool:
    """Whether the completion defines ``entry_point`` at top level (evalplus's
    ``solution`` mode) instead of continuing the prompt's signature."""
    try:
        tree = ast.parse(completion)
    except SyntaxError:
        return False
    names = {
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    return entry_point in names


def humaneval_program(completion: str, row: dict) -> str:
    """The check program exactly as HumanEval's harness assembles it: the
    completion appended to the prompt (upstream), or alone when it is a full
    definition of ``entry_point`` (evalplus's "solution" mode) — then the tests
    and the ``check(entry_point)`` call, no trailing newline (upstream exact)."""
    solution = (
        completion
        if _standalone(completion, row["entry_point"])
        else row["prompt"] + completion
    )
    return solution + "\n" + row["test"] + "\n" + f"check({row['entry_point']})"


def mbpp_program(completion: str, row: dict) -> str:
    """The check program as evalplus assembles MBPP: the docstring prompt, the
    completion (a full function — the docstring gives no signature to continue),
    then the tests."""
    return row["prompt"] + completion + "\n" + row["test"]


def extract_completion(generation: str) -> str:
    """The completion ns's code evaluators take from a generation: the last
    ```python block (``preprocess_code``), the completion's own indentation
    kept — no strip, the fence's one leading newline removed (ns's infilling
    evaluator does the same): upstream HumanEval concatenates the completion as
    the model wrote it, body indentation included."""
    from nemo_skills.evaluation.evaluator.code import preprocess_code

    completion = preprocess_code(
        {"generation": generation or ""}, "python", strip_whitespace=False
    )["completion"]
    return completion[1:] if completion.startswith("\n") else completion


# -- the benchmark ---------------------------------------------------------------------


class CodegenBenchmark(NemoSkillsBenchmark):
    """HumanEval / MBPP's shared run (module docstring): ns's codegen prompt and
    generation, grading on HumanEval's own harness in one granite container per
    scoring phase, ns's pass@1 metrics over the grade."""

    requires_sandbox = (
        False  # grading runs in its own container, not ns's sandbox server
    )
    metric = "pass@1 accuracy"
    default_repeats = 1
    ns_metric = "passing_tests"
    ns_metrics_type = (
        "granite_evals.benchmarks.nemo_skills_codegen::PassingTestsMetrics"
    )
    ns_generation_args: ClassVar[tuple[str, ...]] = ("++eval_type=null",)
    """Never let ns's own evaluator run: with it off in every phase, grading is
    always ours (``_evaluate``), so no file is ever graded twice."""

    expected_rows: ClassVar[int]
    dataset_config: ClassVar[str | None] = None
    ns_row: ClassVar[Any]
    grade_program: ClassVar[Any]

    def _module_attr(self, name: str, default: Any = None) -> Any:
        """ns's dataset module is never imported (its EvalPlus data path is what
        this module replaces); its one setting kept is the generic codegen prompt."""
        return {
            "GENERATION_ARGS": "++prompt_config=generic/codegen",
            "EVAL_SPLIT": "test",
        }.get(name, default)

    def prepare_data(self) -> tuple[Path, dict[str, Any]]:
        """The pinned HF dataset as ns rows (``ns-data/<benchmark>/test.jsonl``)."""
        src = self.config.dataset
        if src and Path(src).exists():
            return NemoSkillsBenchmark.prepare_data(
                self
            )  # an already prepared jsonl/dir
        dataset, revision = self.dataset_source()
        pinned = (dataset, revision) == (self.dataset, self.dataset_revision)
        rows = data.load_split(
            dataset, revision=revision, split="test", name=self.dataset_config
        )
        out_rows = [self.ns_row(r) for r in rows]
        if pinned and len(out_rows) != self.expected_rows:
            raise RuntimeError(
                f"{dataset}: {len(out_rows)} rows, expected {self.expected_rows}"
            )
        ddir = self.config.output_dir / "ns-data" / self.ns_benchmark
        ddir.mkdir(parents=True, exist_ok=True)
        out = ddir / "test.jsonl"
        _write_jsonl(out, out_rows)
        return out, {
            "source": "raw HF dataset, ns row shape",
            "dataset": dataset,
            "revision": revision,
            "config": self.dataset_config,
            "rows": len(out_rows),
            "sha256": _sha256(out),
        }

    def gold_generation(self, row: dict) -> str:
        """The reference solution as the codegen prompt's answer: fenced, so it
        goes through extraction and grading exactly like a model completion."""
        return f"```python\n{row['canonical_solution']}\n```"

    # -- grading ------------------------------------------------------------------

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        with self.grading_sandbox():
            result = super().run(base_url, served_model_name)
        if self.scoring:
            result["grading_sandbox"] = self._grade_report
        return result

    @contextlib.contextmanager
    def grading_sandbox(self) -> Iterator[None]:
        """One grading container per scoring run (every repeat and resume shares
        it): ``grade_image`` on the ``sandbox`` option's backend. The generate
        phase never starts it."""
        if not self.scoring:
            yield
            return
        mode = self.opt("grade_network", "off")
        if mode not in ("off", "best-effort", "on"):
            raise SystemExit(f"{self.id}: grade_network must be off, best-effort or on")
        image = self.opt("grade_image", GRADE_IMAGE)
        backend = self.opt("sandbox", "")
        if backend == "unsafe-local":
            from granite_evals.agent_tools import _local_sandbox

            sb = _local_sandbox(image)
        else:
            from granite_evals.sandbox import make_sandbox

            sb = make_sandbox(image, backend=backend)
        started = time.time()
        sb.start()
        root = getattr(sb, "workdir", "") or "/tmp"
        sb.write_file(f"{root}/granite_grade.py", GRADE_RUNNER)
        self._grading_sb = sb
        self._grade_report = {
            "image": image,
            "backend": backend or os.environ.get("GRANITE_EVALS_SANDBOX", "enroot"),
            "root": root,
            "network_mode": mode,
            "workers": int(self.opt("grade_workers", 8)),
            "timeout_s": float(self.opt("grade_timeout", 10.0)),
            "harness": f"openai/human-eval execution.py @ {HUMANEVAL_EXEC_COMMIT[:12]} (MIT), ported",
            "startup_seconds": round(time.time() - started, 1),
        }
        log.info("%s: grading sandbox: %s", self.id, json.dumps(self._grade_report))
        try:
            yield
        finally:
            self._grading_sb = None
            sb.close()

    def _evaluate(self, output: Path, args: list[str], what: str) -> None:
        """Grade one repeat's file in the grading sandbox — unconditionally:
        ``++eval_type=null`` is baked into the generation args, so no file is
        ever evaluated inside generation and there is no marker to check
        (unlike the base's deferred evaluation). Rows already graded keep their
        grade; a resumed scoring phase grades only what is missing."""
        if not output.exists():
            raise self.not_generated(what)
        rows = _read_jsonl(output)
        todo = [r for r in rows if "is_correct" not in r]
        if todo:
            self._grade_rows(todo)
            _write_jsonl(output, rows)
        _unscored(output).unlink(missing_ok=True)

    def _grade_rows(self, rows: list[dict]) -> None:
        """Assemble and run one repeat's programs in the phase's sandbox."""
        sb = getattr(self, "_grading_sb", None)
        if sb is None:
            raise RuntimeError(
                f"{self.id}: no grading sandbox (the score/all phases start one)"
            )
        report, root = self._grade_report, self._grade_report["root"]
        to_grade = [
            {
                "task_id": str(r["task_id"]),
                "program": self.grade_program(
                    extract_completion(r.get("generation") or ""), r
                ),
            }
            for r in rows
        ]
        rows_path, out_path, runner = (
            f"{root}/grade-rows.jsonl",
            f"{root}/grade-results.jsonl",
            f"{root}/granite_grade.py",
        )
        sb.write_file(rows_path, "".join(json.dumps(r) + "\n" for r in to_grade))
        # the runner's own argv: <network mode> <rows> <out> <workers> <timeout>
        cmd = "python3 {} {} {} {} {} {}".format(
            shlex.quote(runner),
            shlex.quote(report["network_mode"]),
            shlex.quote(rows_path),
            shlex.quote(out_path),
            report["workers"],
            report["timeout_s"],
        )
        run = sb.execute(
            cmd,
            timeout=int(
                600
                + len(to_grade) * (report["timeout_s"] + 2) / max(1, report["workers"])
            ),
        )
        if run.returncode:
            raise RuntimeError(
                f"{self.id}: grading exited {run.returncode}: {run.output[-2000:]}"
            )
        cat = sb.execute(f"cat {shlex.quote(out_path)}", timeout=120)
        if cat.returncode:
            raise RuntimeError(
                f"{self.id}: reading {out_path} failed: {cat.output[-2000:]}"
            )
        lines = [json.loads(line) for line in cat.output.splitlines() if line.strip()]
        hello, results = lines[0], lines[1:]
        report.update(
            network_isolation=hello.get("network"),
            python=hello.get("python"),
            graded=len(results),
        )
        if not hello.get("ready"):
            raise SystemExit(
                f"{self.id}: cannot isolate the grading sandbox's network ({hello.get('network')}); "
                "use grade_network=best-effort or on to allow it"
            )
        by_id = {r["task_id"]: r for r in results}
        missing = [r["task_id"] for r in to_grade if r["task_id"] not in by_id]
        if missing:
            raise RuntimeError(
                f"{self.id}: the grader returned no result for {missing[:5]}"
            )
        for row in rows:
            graded = by_id[str(row["task_id"])]
            row["is_correct"] = bool(graded["passed"])
            row["grade_status"] = graded["result"]
        log.info(
            "%s: graded %d rows, %d passed",
            self.id,
            len(rows),
            sum(r["is_correct"] for r in rows),
        )


# -- benchmarks -----------------------------------------------------------------------


@register
class HumanEval(CodegenBenchmark):
    """OpenAI HumanEval (164 problems, MIT), ns's ``generic/codegen`` prompt and
    generation, pass@1 on HumanEval's own execution harness in a granite
    container (``CodegenBenchmark`` doc). Not gated on HF."""

    id = "human-eval"
    ns_benchmark = "human-eval"
    dataset = HUMANEVAL_DATASET
    dataset_revision = HUMANEVAL_REVISION
    expected_rows = HUMANEVAL_N
    ns_row = staticmethod(humaneval_row)
    grade_program = staticmethod(humaneval_program)


@register
class MBPP(CodegenBenchmark):
    """Google MBPP, the sanitized test split (257 manually-verified problems,
    CC-BY-4.0 — the paper's eval split; evalplus's MBPP+ builds on the same
    sanitized set), ns's ``generic/codegen`` prompt and generation, pass@1 on
    the ported HumanEval harness in a granite container. The ``full`` split's
    500 (known-flawed) problems are reachable with ``--dataset`` and a locally
    prepared jsonl."""

    id = "mbpp"
    ns_benchmark = "mbpp"
    dataset = MBPP_DATASET
    dataset_revision = MBPP_REVISION
    dataset_config = MBPP_CONFIG
    expected_rows = MBPP_N
    ns_row = staticmethod(mbpp_row)
    grade_program = staticmethod(mbpp_program)
