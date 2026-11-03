"""Benchmarks with tools: HLE (python + web search), CritPt (python), BrowseComp
(web search + page fetch).

All three are NeMo-Skills runs (``benchmarks/nemo_skills.py``: ns's prepare,
generate, judge, metrics, phases, ``answers=gold``) with ns's tool-calling
generation (``++tool_modules``): the served model calls tools through vLLM's
tool calling (its own tool parser, ``--tool-call-parser auto``); every call is
executed and returned as a ``tool`` message until the model answers without
one, or after ``max_tool_calls`` calls (``finish_reason: tool_call_limit_reached``,
in ``details.statuses``). The tools are ``granite_evals.agent_tools``:

- ``python`` -> ``stateful_python_code_exec``: ns's Python tool (name,
  description, Jupyter-like state per rollout) executed in a granite container
  (enroot on SLURM; ``python_image``, default python:3.13-bookworm with
  ``python_packages`` pip-installed once per run), its network cut by default
  (``python_network``: off / best-effort / on). What the sandbox could isolate is
  recorded in ``details.tools.python_sandbox``.
- ``search`` -> ``web_search``: Google via a Google PSE MCP server (the
  search behind ``bfcl-v4``'s web search).
- ``fetch`` -> ``fetch_url``: one page as text, paged.

Results matching the benchmark's own data (e.g. the BrowseComp CSV, HLE on HF)
are filtered from search and refused by fetch.

Options (besides the ns ones): ``tools`` (comma list of python, search, fetch;
default per benchmark), ``max_tool_calls`` (default 100), ``python_timeout``
(seconds per call, 60), ``python_image``, ``python_packages`` (space or comma
separated pip requirements; ``none`` for an image with its own),
``python_network``, ``sandbox`` (backend: enroot / podman / docker; default
``GRANITE_EVALS_SANDBOX``), ``search_mcp_url``, ``search_results`` (10),
``fetch_max_chars`` (20000), ``tool_output_max_chars`` (python, 20000).
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, ClassVar

import httpx

from granite_evals.benchmarks.nemo_skills import (
    NemoSkillsBenchmark,
    _passthrough,
    _read_jsonl,
    _sha256,
    _unscored,
    _write_jsonl,
)
from granite_evals.registry import register

log = logging.getLogger(__name__)

TOOL_CLASSES = {
    "python": "granite_evals.agent_tools::SandboxPythonTool",
    "search": "granite_evals.agent_tools::WebSearchTool",
    "fetch": "granite_evals.agent_tools::FetchUrlTool",
}

PYTHON_IMAGE = "python:3.13-bookworm"
PYTHON_PACKAGES = (
    "numpy==2.1.3",
    "scipy==1.17.1",
    "sympy==1.14.0",
    "mpmath==1.3.0",
    "pandas==3.0.6",
    "networkx==3.6.1",
)
"""Pinned scientific stack for the python tool (the versions in granite's uv.lock)."""


class ToolBenchmark(NemoSkillsBenchmark):
    """A NeMo-Skills benchmark generated with granite's tools (module docstring)."""

    requires_sandbox = False  # ns's own sandbox server: not used (the python tool has its own)
    default_tools: ClassVar[tuple[str, ...]] = ()
    blocked_url_patterns: ClassVar[tuple[str, ...]] = ()
    default_max_tool_calls: ClassVar[int] = 100

    # -- tools -----------------------------------------------------------------

    def tool_names(self) -> list[str]:
        raw = self.opt("tools", ",".join(self.default_tools))
        names = [t.strip() for t in raw.split(",") if t.strip() and t.strip() != "none"]
        unknown = [t for t in names if t not in TOOL_CLASSES]
        if unknown:
            raise SystemExit(f"{self.id}: unknown tools {unknown} (known: {', '.join(TOOL_CLASSES)})")
        return names

    @property
    def tools_dir(self) -> Path:
        return self.config.output_dir / "tools"

    def tool_config(self) -> dict[str, Any]:
        """The settings every tool reads (``agent_tools.load_config``)."""
        pkgs = self.opt("python_packages", " ".join(PYTHON_PACKAGES))
        return {
            "python": {
                "image": self.opt("python_image", PYTHON_IMAGE),
                "backend": self.opt("sandbox", ""),
                "packages": [] if pkgs.strip() == "none" else [p for p in re.split(r"[\s,]+", pkgs) if p],
                "timeout_s": self.opt("python_timeout", 60.0),
                "network": self.opt("python_network", "off"),
                "max_output_chars": self.opt("tool_output_max_chars", 20000),
                "probe_file": str(self.tools_dir / "python-sandbox.json"),
            },
            "search": {
                "mcp_url": self.opt("search_mcp_url", ""),
                "num_results": self.opt("search_results", 10),
                "max_results": max(10, self.opt("search_results", 10)),
            },
            "fetch": {
                "max_chars": self.opt("fetch_max_chars", 20000),
                "allow_private_hosts": self.opt("allow_private_hosts", False),  # tests only
            },
            "blocked_url_patterns": list(self.blocked_url_patterns),
        }

    def tool_args(self) -> list[str]:
        names = self.tool_names()
        if not names:
            return []
        classes = [TOOL_CLASSES[n] for n in names]
        cfg = self.tools_dir / "tools.json"
        return [
            f"++tool_modules=[{','.join(classes)}]",
            *(f"++tool_overrides.{c.split('::')[1]}.config={cfg}" for c in classes),
            f"++max_tool_calls={self.opt('max_tool_calls', self.default_max_tool_calls)}",
        ]

    def generation_args(self, sandbox_args: list[str]) -> list[str]:
        args = super().generation_args(sandbox_args)
        tail = _passthrough(self.config.options, "ns.")  # stays last: the user's overrides win
        head = args[: len(args) - len(tail)]
        return [*head, *self.tool_args(), *tail]

    def _write_tool_config(self) -> None:
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.tool_config()
        if cfg["python"]["network"] not in ("off", "best-effort", "on"):
            raise SystemExit(f"{self.id}: python_network must be off, best-effort or on")
        (self.tools_dir / "tools.json").write_text(json.dumps(cfg, indent=2) + "\n")

    # -- run ---------------------------------------------------------------------

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        self._write_tool_config()
        self.prepare_prompts()
        result = super().run(base_url, served_model_name)
        files = [self.config.output_dir / "generation" / f"output-rs{k}.jsonl" for k in range(self.repeats)]
        probe = self.tools_dir / "python-sandbox.json"
        result["tools"] = {
            "enabled": self.tool_names(),
            "max_tool_calls": self.opt("max_tool_calls", self.default_max_tool_calls),
            "config": json.loads((self.tools_dir / "tools.json").read_text()),
            "calls": tool_stats(files),
            "python_sandbox": json.loads(probe.read_text()) if probe.exists() else None,
        }
        return result

    def prepare_prompts(self) -> None:
        """Hook: write prompt configs the generation args point at."""

    def compute_metrics(self, files: list[Path]) -> dict[str, Any]:
        """ns metrics without the problems whose judgement is invalid (in any of
        ``files``, so repeats stay aligned): ns's MathMetrics would score them
        wrong, but an invalid judgement is not a model loss (failure_policy;
        counted in ``judge_invalid``, re-judged on resume)."""
        if not self.uses_judge():
            return super().compute_metrics(files)
        rows = [_read_jsonl(f) for f in files]
        bad = {i for rs in rows for i, r in enumerate(rs) if self.judge_failures(r)[0]}
        if not bad:
            return super().compute_metrics(files)
        out = self.config.output_dir / "metrics-input"
        out.mkdir(parents=True, exist_ok=True)
        kept = []
        for f, rs in zip(files, rows, strict=True):
            _write_jsonl(out / f.name, [r for i, r in enumerate(rs) if i not in bad])
            kept.append(out / f.name)
        log.info("%s: %d problems with an invalid judgement left out of the metrics", self.id, len(bad))
        return super().compute_metrics(kept)


def tool_stats(files: list[Path]) -> dict[str, Any]:
    """Tool calls per generation (ns ``num_tool_calls``) and the tools' own
    counters (``tool_metrics``: calls, errors, timeouts, refused fetches, ...)."""
    calls: list[int] = []
    per_tool: dict[str, Counter] = {}
    for f in files:
        for row in _read_jsonl(f):
            calls.append(int(row.get("num_tool_calls") or 0))
            for tool, m in (row.get("tool_metrics") or {}).items():
                c = per_tool.setdefault(tool, Counter())
                c.update({k: v for k, v in (m or {}).items() if isinstance(v, (int, float))})
    return {
        "generations": len(calls),
        "total": sum(calls),
        "mean": sum(calls) / len(calls) if calls else 0.0,
        "max": max(calls, default=0),
        "without_tools": sum(1 for c in calls if not c),
        "per_tool": {k: dict(v) for k, v in sorted(per_tool.items())},
    }


# -- HLE ---------------------------------------------------------------------------


@register
class HLETools(ToolBenchmark):
    """Humanity's Last Exam, text-only questions (ns ``--split text``: the 2500
    minus those with an image), ns's HLE prompt and judge, with a python tool and
    web search. Gated on HF (``HF_TOKEN`` with access to cais/hle).

    The judge is ns's ``judge/hle`` (the official HLE judge prompt) with
    Azure/gpt-4o-ncf, the model card's official judge ,
    not the HLE leaderboard/ns default (o3-mini) nor NVIDIA's "HLE with tools"
    judge (gpt-4o). A judgement without a yes/no verdict is re-judged
    (``max_judge_invalid_frac``), not counted wrong as ns would."""

    id = "hle-tools"
    metric = "pass@1 judge_correct"
    default_repeats = 1
    ns_benchmark = "hle"
    ns_split = "text"
    ns_prepare_args = ("--split", "text")
    ns_metric = "judge_correct"
    dataset = "cais/hle"  # gated
    dataset_revision = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
    official_judge = (
        "o3-mini-2025-01-31 (HLE leaderboard, ns default); gpt-4o (Artificial Analysis / NVIDIA HLE with tools); "
        "Azure/gpt-4o-ncf (the model card's official judge, )"
    )
    # judge_model/judge_base_url overrides pending team sign-off on switching
    # the actual runtime judge to Azure/gpt-4o-ncf; shared Sonnet-5 default
    # stays in effect for now.
    gold_judged = True
    default_tools = ("python", "search")
    blocked_url_patterns = (
        r"(?i)huggingface\.co/datasets/cais/hle",
        r"(?i)lastexam\.ai",
        r"(?i)agi\.safe\.ai",
        r"(?i)humanity'?s?[-_ ]?last[-_ ]?exam",
        r"(?i)github\.com/centerforaisafety/hle",
    )

    def prepare_rows(self, rows: list[dict]) -> list[dict]:
        return sorted(rows, key=lambda r: str(r["id"]))  # --limit N: the first N by id

    def judge_failures(self, row: dict) -> tuple[int, int]:
        from nemo_skills.evaluation.metrics.utils import is_correct_judgement

        return int(is_correct_judgement(row.get("judgement"), return_none=True) is None), 1

    def gold_generation(self, row: dict) -> str:
        return f"Explanation: the reference answer.\nAnswer: {row['expected_answer']}\nConfidence: 100%"


# -- CritPt --------------------------------------------------------------------------

CRITPT_API_URL = "https://artificialanalysis.ai/api/v2/critpt/evaluate"
CRITPT_N = 70


@register
class CritPtTools(ToolBenchmark):
    """CritPt (70 physics research challenges), ns's two-turn CritPt generation
    (solve, then fill the code template) with a python tool in both turns (a
    fresh session per turn: ns runs them as separate requests).

    Answers are hidden: grading is Artificial Analysis's CritPt API (as ns's
    critpt evaluator: the last ```python block of the final answer per problem,
    all 70 in one request), which returns only the aggregate accuracy. The API
    needs an approved key (``ARTIFICIAL_ANALYSIS_API_KEY``, option
    ``critpt_api_key_env``) and allows 10 requests a day, so each repeat's
    response is cached under its submissions' sha256 and never re-sent. With
    ``--limit N`` the other 70-N problems are sent as their bare code template
    (scored wrong) and the accuracy is rescaled to the N submitted
    (``details.critpt``). No ``answers=gold``."""

    id = "critpt-tools"
    metric = "Challenge Accuracy"
    default_repeats = 1
    ns_benchmark = "critpt"
    ns_metric = "accuracy"
    dataset = "CritPt-Benchmark/CritPt"
    dataset_revision = "9b9fc8498596ec08ab5437a72f4aa18beef2b876"
    ns_generation_args = ("++eval_type=null",)  # graded here (_evaluate), not inside generation
    # ns's critpt generation, but with a nullable temperature (see ns_critpt)
    ns_generation_module = "granite_evals.ns_critpt"
    default_tools = ("python",)
    blocked_url_patterns = (r"(?i)critpt",)

    def needs_server(self) -> bool:
        return True

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        if self.gold:
            raise SystemExit(f"{self.id}: answers are hidden (graded by the CritPt API), no answers=gold")
        result = super().run(base_url, served_model_name)
        if self.scoring:
            gen = self.config.output_dir / "generation"
            result["critpt"] = [
                json.loads(p.read_text())
                for k in range(self.repeats)
                if (p := gen / f"output-rs{k}-critpt.json").exists()
            ]
        return result

    def prepare_rows(self, rows: list[dict]) -> list[dict]:
        rows = sorted(rows, key=lambda r: _natural(str(r["problem_id"])))
        self._all_rows = rows
        return rows

    def _evaluate(self, output: Path, args: list[str], what: str) -> None:
        """Grade one repeat through the CritPt API (class docstring)."""
        if not output.exists():
            raise self.not_generated(what)
        rows = _read_jsonl(output)
        by_id = {r["problem_id"]: r for r in rows}
        submissions, real = [], 0
        for r in self._all_rows:
            if r["problem_id"] in by_id:
                code, real = _critpt_code(by_id[r["problem_id"]].get("generation") or ""), real + 1
            else:
                code = _critpt_code(f"```python\n{r.get('code_template') or ''}\n```")
            submissions.append({"problem_id": r["problem_id"], "generated_code": code, "model": "unknown",
                                "generation_config": {}})  # fmt: skip
        digest = hashlib.sha256(json.dumps(submissions, sort_keys=True).encode()).hexdigest()
        cache = self.config.output_dir / "critpt-api" / f"{digest}.json"
        _write_jsonl(output.with_name(output.stem + "-submissions.jsonl"), submissions)
        if cache.exists():
            response = json.loads(cache.read_text())
            log.info("%s: CritPt API response for %s from cache %s", self.id, output.name, cache)
        else:
            response = self._submit(submissions)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(response, indent=2) + "\n")
        api_accuracy = float(response["accuracy"])
        accuracy = min(1.0, api_accuracy * len(submissions) / real) if real else 0.0
        record = {
            "repeat_file": output.name,
            "api_accuracy": api_accuracy,
            "submitted": real,
            "padded_with_template": len(submissions) - real,
            "accuracy": accuracy,
            "response": response,
            "cache": str(cache),
        }
        output.with_name(output.stem + "-critpt.json").write_text(json.dumps(record, indent=2) + "\n")
        meta = {k: response.get(k) for k in ("accuracy", "timeout_rate", "server_timeout_count", "judge_error_count")}
        for r in rows:
            r["full_dataset_accuracy"] = accuracy  # what ns's CritPtMetrics reads
            r["evaluation_metadata"] = {**meta, "submitted": real, "rescaled_accuracy": accuracy}
        _write_jsonl(output, rows)
        _unscored(output).unlink(missing_ok=True)

    def _submit(self, submissions: list[dict]) -> dict[str, Any]:
        env = self.opt("critpt_api_key_env", "ARTIFICIAL_ANALYSIS_API_KEY")
        key = os.environ.get(env, "")
        if not key:
            raise SystemExit(f"{self.id}: {env} is empty: the CritPt API needs an Artificial Analysis key")
        url = self.opt("critpt_api_url", CRITPT_API_URL)
        log.info("%s: submitting %d solutions to %s", self.id, len(submissions), url)
        r = httpx.post(url, json={"submissions": submissions, "batch_metadata": {}},
                       headers={"x-api-key": key, "Content-Type": "application/json"}, timeout=900)  # fmt: skip
        if r.status_code in (401, 403):
            raise SystemExit(f"{self.id}: CritPt API refused the key (HTTP {r.status_code}): {r.text[:500]}")
        if r.status_code == 429:
            raise SystemExit(f"{self.id}: CritPt API rate limit (10 requests / 24h): {r.text[:500]}; rerun later")
        r.raise_for_status()
        return r.json()


def _critpt_code(generation: str) -> str:
    """The submitted code, extracted as ns's CritPt evaluator does."""
    matches = re.findall(r"```(?:python)?\s*\n(.*?)\n```", generation, re.DOTALL)
    code = matches[-1].strip() if matches else generation.strip()
    return code if code.startswith("```") else f"```python\n{code}\n```"


def _natural(s: str) -> list:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


# -- BrowseComp ------------------------------------------------------------------------

SIMPLE_EVALS_COMMIT = "652c89d0ca9df547706735883097e9537d40dc47"
BROWSECOMP_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"
BROWSECOMP_SHA256 = "7b24471cd5b3eb2a46830a14802b5c029ea62f488ff75a0f88af7923d1454abf"
BROWSECOMP_N = 1266

# simple-evals browsecomp_eval.py (SIMPLE_EVALS_COMMIT), verbatim except the
# placeholders, renamed to ns's fields ({Question} -> {problem}, ...).
BROWSECOMP_QUERY = """{problem}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}"""

BROWSECOMP_GRADER = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {problem}

[response]: {generation}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {expected_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()
BROWSECOMP_GRADER_SYSTEM = "You are a helpful assistant."  # simple-evals OPENAI_SYSTEM_MESSAGE_API


def browsecomp_derive_key(password: str, length: int) -> bytes:
    """simple-evals browsecomp_eval.derive_key."""
    key = hashlib.sha256(password.encode()).digest()
    return key * (length // len(key)) + key[: length % len(key)]


def browsecomp_decrypt(ciphertext_b64: str, password: str) -> str:
    """simple-evals browsecomp_eval.decrypt: base64, XOR with the canary's sha256."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = browsecomp_derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


def browsecomp_encrypt(plaintext: str, password: str) -> str:
    """The inverse of ``browsecomp_decrypt`` (tests' fake CSVs)."""
    data = plaintext.encode()
    key = browsecomp_derive_key(password, len(data))
    return base64.b64encode(bytes(a ^ b for a, b in zip(data, key))).decode()


def browsecomp_verdict(judgement: str | None) -> str | None:
    """'yes' / 'no' from a grader reply, None if it has none.

    simple-evals: ``re.search(r"correct: (yes|no)", response)``, then
    ``match.group(0) if match else "no"`` compared with "yes", which can never
    match (group(0) is "correct: yes"): taken as the evident intent, group(1).
    A reply in another layout (markdown bold, capitalised) is read leniently."""
    if not judgement:
        return None
    m = re.search(r"correct: (yes|no)", judgement)
    if m:
        return m.group(1)
    m = re.search(r"(?im)^\W*correct\W*:\W*(yes|no)\b", judgement)
    return m.group(1).lower() if m else None


try:  # ns's MathMetrics (judge_correct from ``judgement``) with BrowseComp's verdict
    from nemo_skills.evaluation.metrics.math_metrics import MathMetrics as _MathMetrics
except ImportError:  # without the nemoskills extra: the class is only used inside ns
    _MathMetrics = object


class BrowseCompMetrics(_MathMetrics):
    def is_correct_judgement(self, judgement: str) -> bool:
        return browsecomp_verdict(judgement) == "yes"


@register
class BrowseComp(ToolBenchmark):
    """OpenAI BrowseComp (1266 questions; simple-evals' encrypted CSV, decrypted
    with each row's canary), simple-evals' query template and grader prompt,
    with web search and page fetch. ``value``: the fraction graded correct
    (simple-evals' mean score, the suite's "mean reward").

    Deviations from simple-evals: grader claude-sonnet-5 instead of
    gpt-4.1-2025-04-14; the verdict parsed as ``browsecomp_verdict`` says (the
    upstream parse never counts a "yes"); a reply without a verdict is re-judged
    (``max_judge_invalid_frac``) instead of counted "no"; ``--limit N`` takes the
    first N rows of the CSV, not a seeded random sample."""

    id = "browsecomp"
    metric = "mean reward"
    default_repeats = 1
    ns_benchmark = "browsecomp"
    ns_metric = "judge_correct"
    ns_metrics_kwargs: ClassVar[dict[str, Any]] = {"compute_no_answer": False, "answer_key": "generation"}
    dataset = BROWSECOMP_URL
    dataset_revision = f"sha256:{BROWSECOMP_SHA256}"
    judge = True
    official_judge = f"gpt-4.1-2025-04-14 (openai/simple-evals {SIMPLE_EVALS_COMMIT[:12]} browsecomp_eval.py)"
    gold_judged = True
    default_tools = ("search", "fetch")
    blocked_url_patterns = (r"(?i)browse[_-]?comp", r"(?i)openaipublic\.blob\.core\.windows\.net/simple-evals")

    @property
    def prompts_dir(self) -> Path:
        return self.config.output_dir / "prompts"

    def _module_attr(self, name: str, default: Any = None) -> Any:
        """No ns dataset module: its settings, here."""
        p = self.prompts_dir
        return {
            "METRICS_TYPE": "granite_evals.benchmarks.tools_agentic::BrowseCompMetrics",
            "GENERATION_ARGS": f"++prompt_config={p / 'browsecomp.yaml'}",
            "EVAL_SPLIT": "test",
            "JUDGE_ARGS": (
                f"++prompt_config={p / 'browsecomp-judge.yaml'} ++generation_key=judgement ++add_generation_stats=False"
            ),
        }.get(name, default)

    def prepare_prompts(self) -> None:
        import yaml

        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        (self.prompts_dir / "browsecomp.yaml").write_text(yaml.safe_dump({"user": BROWSECOMP_QUERY}, sort_keys=False))
        (self.prompts_dir / "browsecomp-judge.yaml").write_text(
            yaml.safe_dump({"system": BROWSECOMP_GRADER_SYSTEM, "user": BROWSECOMP_GRADER}, sort_keys=False)
        )

    def prepare_data(self) -> tuple[Path, dict[str, Any]]:
        """The decrypted CSV as ns jsonl (``ns-data/browsecomp/test.jsonl``)."""
        src = self.config.dataset
        if src and Path(src).exists() and not src.endswith(".csv"):
            return super().prepare_data()  # an already prepared jsonl
        ddir = self.config.output_dir / "ns-data" / "browsecomp"
        ddir.mkdir(parents=True, exist_ok=True)
        if src and Path(src).exists():
            csv_path, pinned = Path(src), False
        else:
            csv_path, pinned = ddir / "browse_comp_test_set.csv", True
            if not csv_path.exists() or _sha256(csv_path) != BROWSECOMP_SHA256:
                url = src or BROWSECOMP_URL
                log.info("%s: downloading %s", self.id, url)
                r = httpx.get(url, follow_redirects=True, timeout=300)
                r.raise_for_status()
                digest = hashlib.sha256(r.content).hexdigest()
                if digest != BROWSECOMP_SHA256:
                    raise RuntimeError(f"{url}: sha256 {digest}, expected {BROWSECOMP_SHA256}")
                csv_path.write_bytes(r.content)
        rows = browsecomp_rows(csv_path)
        if pinned and len(rows) != BROWSECOMP_N:
            raise RuntimeError(f"{csv_path}: {len(rows)} rows, expected {BROWSECOMP_N}")
        out = ddir / "test.jsonl"
        _write_jsonl(out, rows)
        return out, {
            "source": "simple-evals browsecomp CSV" if pinned else "local CSV",
            "url": BROWSECOMP_URL if pinned else str(csv_path),
            "csv_sha256": _sha256(csv_path),
            "simple_evals_commit": SIMPLE_EVALS_COMMIT,
            "sha256": _sha256(out),
        }

    def judge_failures(self, row: dict) -> tuple[int, int]:
        return int(browsecomp_verdict(row.get("judgement")) is None), 1

    def gold_generation(self, row: dict) -> str:
        return f"Explanation: the reference answer.\nExact Answer: {row['expected_answer']}\nConfidence: 100%"


def browsecomp_rows(csv_path: Path) -> list[dict]:
    """ns rows from the BrowseComp CSV (problem, answer, problem_topic, canary)."""
    import csv

    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for i, r in enumerate(rows):
        out.append(
            {
                "id": f"browsecomp-{i}",
                "problem": browsecomp_decrypt(r["problem"], r["canary"]),
                "expected_answer": browsecomp_decrypt(r["answer"], r["canary"]),
                "problem_topic": r.get("problem_topic", ""),
                "subset_for_metrics": r.get("problem_topic") or "other",
            }
        )
    return out
