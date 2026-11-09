"""NeMo-Skills code benchmarks (Granite: LiveCodeBench v6, SciCode).

Both run through the shared ``NemoSkillsBenchmark`` (``nemo_skills.py``): ns's own
prepare, prompts, generation loop, evaluators and metrics, against the step's vLLM.
What is specific here:

LiveCodeBench v6 (``livecodebench-v6``)
    ns's ``test_v6_2408_2505`` split: release v6, contests 2024-08..2025-05, 454
    problems, from ``livecodebench/code_generation_lite`` pinned to a commit. ns's
    evaluator runs the ``livecodebench`` package (wasiahmad's fork, pinned in the
    ``nemoskills`` extra) in the job container. Upstream it reloads the test cases
    from HF at an unpinned ref when no ``test_file`` is given, so prepare keeps all
    columns and the evaluator gets a ``test_file`` with the selected problems' full
    rows; the generation input drops the (large) test-case columns. The test file
    is rewritten from the prepared data in every phase (``--phase score`` grades
    with it).

SciCode (``scicode``)
    ns's ``test`` split (65 problems, 288 evaluated subtasks), the "with background"
    prompt, ns's multi-step generation module and its sandboxed evaluator. The
    sandbox is ns's local sandbox server, run with a separate Python 3.10 env baked
    into the image (``/opt/ns-sandbox``; numpy/scipy/sympy/h5py/matplotlib at the
    versions ns's sandbox image ends up with) and the SciCode test data at
    ``/data/test_data.h5`` (sha256-verified at image build). ``pip`` inside the
    sandbox is a no-op shim, so the evaluator's own ``pip install`` calls cannot
    change the pinned environment at run time. Only grading uses the sandbox:
    ``--phase generate`` neither starts nor checks it.

    Departure from ns (``prefill_fixes``, on by default): the harness-supplied code
    of steps 13.6 and 62.1 is SciCode's original ``eval/data`` file (``class
    Maxwell``; ``class Block`` and ``class EnlargedBlock``), not ns's lone
    ``__init__``, which leaves 13.8-13.15 and 62's later steps unable to find the
    class. SciCode's own harness has the same defect. See ``scicode_prefill.py``;
    results.json lists each fix under ``prefill_fixes``.

Gold (``--option answers=gold``): LiveCodeBench has no reference solutions, so
five hand-written, hand-checked solutions (``LCB_GOLD``) run through the real
prompt -> generate -> evaluate path; SciCode serves each subtask's
``ground_truth_code`` on the ``dev`` split (the validation problems, the only
ones with public reference code), one step at a time.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterator

from granite_evals.benchmarks import scicode_prefill
from granite_evals.benchmarks.nemo_skills import (
    NemoSkillsBenchmark,
    _free_port,
    _GoldServer,
    _kill,
    _wait_http,
    _write_jsonl,
    log,
)
from granite_evals.registry import register

LCB_DATASET = "livecodebench/code_generation_lite"
LCB_REVISION = "c52cd175916e995019dcd848d1054b419d2e70b5"
"""The commit behind ``refs/pr/7``, the ref ns's prepare.py reads."""

LCB_TEST_COLUMNS = ("public_test_cases", "private_test_cases", "metadata")
"""Columns only the evaluator needs; kept out of the generation input."""

LCB_GOLD: dict[str, str] = {
    "abc365_a": """y = int(input())
leap = (y % 4 == 0 and y % 100 != 0) or y % 400 == 0
print(366 if leap else 365)
""",
    "abc365_b": """n = int(input())
a = list(map(int, input().split()))
order = sorted(range(n), key=lambda i: a[i], reverse=True)
print(order[1] + 1)
""",
    "abc366_b": """n = int(input())
s = [input() for _ in range(n)]
m = max(len(x) for x in s)
for j in range(m):
    t = []
    for i in range(n - 1, -1, -1):
        t.append(s[i][j] if j < len(s[i]) else "*")
    print("".join(t).rstrip("*"))
""",
    "3519": """from typing import List
from collections import Counter

class Solution:
    def winningPlayerCount(self, n: int, pick: List[List[int]]) -> int:
        cnt = Counter((x, y) for x, y in pick)
        return sum(1 for p in range(n) if any(cnt[(p, c)] > p for c in range(11)))
""",
    "3533": """from typing import List

class Solution:
    def finalPositionOfSnake(self, n: int, commands: List[str]) -> int:
        i = j = 0
        for c in commands:
            if c == "UP":
                i -= 1
            elif c == "DOWN":
                i += 1
            elif c == "LEFT":
                j -= 1
            else:
                j += 1
        return i * n + j
""",
}
"""Reference solutions for ``answers=gold`` (LiveCodeBench ships none):
AtCoder stdin programs and LeetCode ``Solution`` classes, one of each kind
from both platforms."""


@register
class LiveCodeBenchV6(NemoSkillsBenchmark):
    id = "livecodebench-v6"
    metric = "pass@1 accuracy"
    ns_benchmark = "livecodebench"
    ns_split = "test_v6_2408_2505"
    ns_prepare_args = ("--release_version", "v6", "--start_date", "2024-08", "--end_date", "2025-05", "--keep_all_columns")
    ns_metric = "accuracy"
    dataset = LCB_DATASET
    dataset_revision = LCB_REVISION
    harness_packages = (*NemoSkillsBenchmark.harness_packages, "livecodebench")

    def tests_file(self) -> Path:
        return self.config.output_dir / "lcb-tests.jsonl"

    def prepare_rows(self, rows: list[dict]) -> list[dict]:
        if self.gold:
            rows = [r for r in rows if r["task_id"] in LCB_GOLD]
        return rows

    def load_rows(self, path: Path) -> list[dict]:
        """The selected problems' full rows go to the evaluator's test file; the
        generation input keeps everything but the test cases."""
        rows = super().load_rows(path)
        _write_jsonl(self.tests_file(), rows)
        slim = [{k: v for k, v in r.items() if k not in LCB_TEST_COLUMNS} for r in rows]
        if self.gold:  # the gold server finds a row by its text in the prompt
            for r in slim:
                r["question"] = r["question_content"]
        return slim

    def generation_args(self, sandbox_args: list[str]) -> list[str]:
        args = super().generation_args(sandbox_args)
        return [*args, f"++eval_config.test_file={self.tests_file()}"]

    def gold_generation(self, row: dict) -> str:
        return f"```python\n{LCB_GOLD[row['task_id']]}```"


SCICODE_DATASET = "SciCode1/SciCode"
SCICODE_REVISION = "4510f6a6aa27c43fad7b43da2c59602a86e88480"
SCICODE_H5 = "/data/test_data.h5"
"""Where ns's SciCode evaluator reads the test targets (hardcoded upstream)."""
SCICODE_H5_SHA256 = "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890"
SANDBOX_ENV = "/opt/ns-sandbox"
"""The image's Python 3.10 env for ns's sandbox server (docker/extras/nemoskills.sh)."""
NEXT_STEP_MARKER = "NEXT STEP - PROBLEM STEP AND FUNCTION HEADER:"


def sandbox_env(python: Path) -> dict[str, str]:
    """The environment for the sandbox env's interpreter: none of the job venv's
    Python settings, so only the sandbox env's own packages import."""
    env = dict(os.environ, PATH=f"{python.parent}:{os.environ.get('PATH', '')}", PYTHONNOUSERSITE="1")
    for k in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(k, None)
    return env


@register
class SciCode(NemoSkillsBenchmark):
    id = "scicode"
    metric = "pass@1 subtask accuracy"
    ns_benchmark = "scicode"
    ns_metric = "subtask_accuracy"
    dataset = SCICODE_DATASET
    dataset_revision = SCICODE_REVISION

    def split(self) -> str:
        # Only the validation problems ("dev") ship reference code.
        return "dev" if self.gold else super().split()

    def fixes_prefill(self) -> bool:
        return str(self.opt("prefill_fixes", True)).lower() not in ("false", "0", "no", "off")

    def generation_module(self) -> str:
        """ns's SciCode generation, behind scicode_prefill (restores the classes of
        prefilled steps 13.6 and 62.1) unless ``prefill_fixes=false``."""
        module = super().generation_module()
        if self.fixes_prefill():
            if module != scicode_prefill.NS_SCICODE_MODULE:
                raise RuntimeError(f"{self.id}: prefill fixes wrap {scicode_prefill.NS_SCICODE_MODULE}, not {module}")
            return "granite_evals.benchmarks.scicode_prefill"
        return module

    def prefill_record(self) -> dict[str, Any]:
        from nemo_skills.inference.eval.scicode_utils import prefilled_steps_code

        fixes = scicode_prefill.plan(prefilled_steps_code)[1] if self.fixes_prefill() else []
        return {"prefill_fixes": fixes, "prefill_fixes_enabled": self.fixes_prefill()}

    def sandbox_python(self) -> Path:
        python = Path(self.opt("sandbox_env", SANDBOX_ENV)) / "bin" / "python"
        if not python.exists():
            raise SystemExit(f"{self.id}: no sandbox env at {python.parent.parent} (built by docker/extras/nemoskills.sh)")
        return python

    def sandbox_app(self) -> Path:
        """ns's sandbox server module, copied out of the job's venv.

        `flask --app <file>` puts the top of the file's package on sys.path; for the
        installed module that is the job's Python 3.12 site-packages, which would
        shadow the sandbox env's own IPython, numpy, scipy... in the server and in
        the code it runs. The server imports nothing from nemo_skills, so a lone copy
        in a directory with no __init__.py runs on the sandbox env alone."""
        import importlib.util

        origin = importlib.util.find_spec("nemo_skills.code_execution.local_sandbox.local_sandbox_server").origin
        app_dir = self.config.output_dir / "ns-sandbox-app"
        app_dir.mkdir(parents=True, exist_ok=True)
        app = app_dir / "ns_local_sandbox_server.py"
        shutil.copyfile(origin, app)
        return app

    @contextlib.contextmanager
    def ns_sandbox(self) -> Iterator[list[str]]:
        """ns's local sandbox server, run by the image's pinned sandbox env
        (the job's own venv has none of SciCode's scientific stack)."""
        python = self.sandbox_python()
        server = self.sandbox_app()
        port = _free_port()
        env = sandbox_env(python)
        cmd = [str(python), "-m", "flask", "--app", str(server), "run", "--host", "127.0.0.1", "--port", str(port)]
        log.info("%s: starting ns sandbox: %s", self.id, shlex.join(cmd))
        logf = (self.config.output_dir / "ns-sandbox.log").open("ab")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        before = {k: os.environ.get(k) for k in ("NEMO_SKILLS_SANDBOX_HOST", "NEMO_SKILLS_SANDBOX_PORT")}
        os.environ.update(NEMO_SKILLS_SANDBOX_HOST="127.0.0.1", NEMO_SKILLS_SANDBOX_PORT=str(port))
        try:
            _wait_http(f"http://127.0.0.1:{port}/health", proc, timeout_s=300)
            yield ["++sandbox.sandbox_type=local", "++sandbox.host=127.0.0.1", f"++sandbox.port={port}"]
        finally:
            for k, v in before.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            _kill(proc)
            logf.close()

    def sandbox_report(self, rows: list[dict]) -> dict[str, Any]:
        """The sandbox env as the run saw it: package versions, the test data's
        sha256, and every problem's dependency block imported once (a failing
        import there would score that problem's subtasks as wrong)."""
        python = self.sandbox_python()
        deps = sorted({r.get("required_dependencies", "") for r in rows} - {""})
        script = (
            "import hashlib, importlib.metadata as md, json, os, sys\n"
            f"deps = {deps!r}\n"
            "failed = {}\n"
            "for d in deps:\n"
            "    try:\n"
            "        exec(d, {})\n"
            "    except Exception as e:\n"
            "        failed[d] = repr(e)\n"
            f"h5 = {SCICODE_H5!r}\n"
            "sha = None\n"
            "if os.path.exists(h5):\n"
            "    h = hashlib.sha256()\n"
            "    with open(h5, 'rb') as f:\n"
            "        for b in iter(lambda: f.read(1 << 24), b''):\n"
            "            h.update(b)\n"
            "    sha = h.hexdigest()\n"
            "vers = {}\n"
            "for p in ('numpy', 'scipy', 'sympy', 'mpmath', 'h5py', 'matplotlib', 'ipython', 'flask'):\n"
            "    try:\n"
            "        vers[p] = md.version(p)\n"
            "    except md.PackageNotFoundError:\n"
            "        vers[p] = None\n"
            "print(json.dumps({'python': sys.version.split()[0], 'packages': vers, 'h5_sha256': sha, 'import_failures': failed}))\n"
        )
        r = subprocess.run([str(python), "-c", script], capture_output=True, text=True, timeout=1800, env=sandbox_env(python))
        try:
            report = json.loads(r.stdout.strip().splitlines()[-1])
        except (IndexError, ValueError):
            raise RuntimeError(f"{self.id}: sandbox env check failed ({python}): {r.stderr[-2000:]}") from None
        report["h5_sha256_ok"] = report.get("h5_sha256") == SCICODE_H5_SHA256
        report["dependency_blocks"] = len(deps)
        if not report["h5_sha256_ok"]:
            raise SystemExit(f"{self.id}: {SCICODE_H5} sha256 {report.get('h5_sha256')} != pinned {SCICODE_H5_SHA256}")
        if report["import_failures"]:
            log.warning("%s: dependency imports failing in the sandbox: %s", self.id, report["import_failures"])
        return report

    def load_rows(self, path: Path) -> list[dict]:
        rows = super().load_rows(path)
        if self.scoring:  # the sandbox grades; generation does not use it
            self._sandbox_report = self.sandbox_report(rows)
        return rows

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        result = super().run(base_url, served_model_name)
        if self.scoring:
            result["sandbox"] = getattr(self, "_sandbox_report", {})
        result.update(self.prefill_record())
        return result

    def _generate(self, input_file: Path, output: Path, base_url: str, served: str, k: int, sandbox_args) -> None:
        if not self.gold:
            return super()._generate(input_file, output, base_url, served, k, sandbox_args)
        # The base gold server answers per problem; SciCode asks once per step.
        rows = [json.loads(line) for line in input_file.read_text().splitlines() if line.strip()]
        with SciCodeGoldServer(rows) as gold:
            return super()._generate(input_file, output, gold.base_url, "gold", k, sandbox_args)


class SciCodeGoldServer(_GoldServer):
    """Answers each step's prompt with that step's ``ground_truth_code``: the step
    is the one whose description and function header follow the prompt's
    "NEXT STEP" marker (earlier steps appear before it)."""

    def __init__(self, rows: list[dict]):
        self.steps = [s for r in rows for s in r["sub_steps"] if s.get("ground_truth_code")]
        super().__init__(rows, answer=lambda r: "")

    def lookup(self, prompt: str) -> str:
        nxt = prompt.split(NEXT_STEP_MARKER, 1)[-1]
        hits = [
            s for s in self.steps if s["step_description_prompt"] in nxt and s["function_header"].strip() in nxt
        ]
        if not hits:
            return ""
        best = max(hits, key=lambda s: len(s["step_description_prompt"]) + len(s["function_header"]))
        return f"```python\n{best['ground_truth_code']}\n```"
