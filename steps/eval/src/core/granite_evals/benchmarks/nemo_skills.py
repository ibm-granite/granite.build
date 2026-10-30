"""NVIDIA NeMo-Skills family (Granite: AIME25, HMMT Feb25, GPQA, MMLU-Pro, Arena-Hard-V2, ...).

The suite names these metrics the way NeMo-Skills reports them ("pass@1
symbolic correct", "judge_correct", ...), so NeMo-Skills (``ns``) is the harness:
its own dataset preparation, prompts, generation loop, evaluators and metrics run
unchanged. This module is only the glue that runs them *locally*, against the vLLM
server the step already started (no Slurm, no NeMo-Run cluster config):

1. prepare: ns's ``nemo_skills/dataset/<name>/prepare.py`` runs inside the job on
   a copy of the dataset dir under ``<output_dir>/ns-data``, with every upstream
   read pinned: HF datasets to a commit (``hf_revisions``), URLs to a pinned URL
   plus sha256 (``pinned_urls``). An unpinned read fails. The prepared jsonl's
   sha256 is recorded, and checked against ``prepared_sha256`` when set.
2. generate: ``python -m nemo_skills.inference.generate`` (or the benchmark's
   GENERATION_MODULE), one output file per repeat (``output-rs<k>.jsonl``, seed
   ``seed + k``), with the benchmark's ns GENERATION_ARGS (prompt config,
   eval_type). ns evaluates each generation itself (``eval_type``).
3. judge (judge benchmarks only): ns's judge module (JUDGE_PIPELINE_ARGS, e.g.
   ``nemo_skills.inference.eval.arena_judge``) over the generations, against the
   judge endpoint, through a local proxy that records the judge's token usage.
   A judgement ns cannot score (a failed call, no verdict) is left out of the
   score, as ns does, counted in ``details.judge_invalid`` / ``judge_total``, and
   re-judged on resume; above ``max_judge_invalid_frac`` the run fails, below it
   the result is flagged ``incomplete``.
4. metrics: ns's ``ComputeMetrics`` over all repeats. ``value`` is
   ``metrics["_all_"][<aggregation>][<ns_metric>]`` scaled to a fraction; the
   full ns metrics dict (all aggregations, subsets) goes to ``details``.
   ``details.pass_at_k`` (``registry.pass_at_k_record``) is ns's own
   ``pass@1[avg-of-k]`` and ``pass@k`` of ``ns_metric`` over the k repeats
   (default 1: pass@1 on one sample), ``details.metrics`` the
   ``ns_report_metrics`` next to the headline, from the same aggregation.

Everything is under ``<output_dir>`` and ns resumes a partial output file
(``++skip_filled``), so granite.build retries resume. A failed request is a
soft failure (``++server.enable_soft_fail``): an empty, wrong answer with
``finish_reason: error``, counted in ``details.statuses``.

Phases (``registry``). ``--phase generate`` is step 2 with ns's evaluation off
(``++eval_type=null``); each such file has an ``output-rs<k>.jsonl.unscored``
marker. ``--phase score`` runs ns's batch evaluation on marked files (the
``eval_type`` / ``eval_config`` generation would have used, as ns generate runs it
after generating), then steps 3-4, with the sandbox but no model; a repeat with
no output file fails the run (``not_generated``). ``--phase all`` evaluates inside
generation as before, and grades any marked file it finds.

Adding a NeMo-Skills benchmark (siblings: start here)
-----------------------------------------------------
Subclass ``NemoSkillsBenchmark`` and ``@register`` it; class attributes do the rest::

    @register
    class HMMTFeb25(NemoSkillsBenchmark):
        id = "hmmt-feb25"                       # the suite id
        metric = "pass@1 symbolic correct"      # the suite metric string
        ns_benchmark = "hmmt_feb25"             # dir in nemo_skills/dataset
        ns_metric = "symbolic_correct"          # key inside the ns aggregation
        dataset = "MathArena/hmmt_feb_2025"     # what prepare.py reads ...
        dataset_revision = "6fdc4277..."        # ... pinned (40-char HF commit)

Knobs (all optional; ``None`` / empty means "what the ns dataset module says"):

- ``ns_split``: the split to prepare/evaluate (default: module EVAL_SPLIT or test).
- ``ns_prepare_args``: argv for prepare.py, e.g. ``("--split", "diamond")``.
- ``hf_revisions``: every *other* HF repo prepare.py reads -> commit
  (``dataset``/``dataset_revision`` is added automatically).
- ``pinned_urls``: {url prepare.py fetches: (pinned url, sha256)}.
- ``prepared_sha256``: expected sha256 of the prepared ``<split>.jsonl``.
- ``ns_aggregation``: default ``pass@1[avg-of-<repeats>]`` (``pass@1`` if 1).
- ``ns_report_metrics``: other keys of that aggregation reported in
  ``details.metrics`` (scaled like the value), e.g. IFBench's strict accuracy.
- ``ns_metrics_type`` / ``ns_metrics_kwargs``: ns METRICS_TYPE override + kwargs.
- ``value_scale``: ns reports percentages; 0.01 turns them into fractions.
- ``ns_generation_args``: extra ``++key=value`` generation overrides (after the
  module's GENERATION_ARGS); ``ns_generation_module`` replaces GENERATION_MODULE.
- ``prompt_config``: replaces the module's ``++prompt_config``.
- ``requires_sandbox``: start ns's code-execution sandbox server for the run
  (default: module REQUIRES_SANDBOX); override ``ns_sandbox()`` to run it elsewhere.
- ``judge`` / ``judge_generation_module`` / ``judge_args``: an LLM judge step
  (default: from the module's JUDGE_PIPELINE_ARGS / JUDGE_ARGS). ``official_judge``
  is recorded next to the judge actually used.
- ``gold_answer_key`` / ``gold_generation(row)``: the oracle answer for
  ``--option answers=gold``, which serves reference answers from a local fake
  endpoint through the *same* ns generate + evaluate + metrics path (~100%).
  ``gold_judged``: a judge benchmark whose gold answers go through the real
  judge (default: no gold mode for judge benchmarks).
- ``prepare_rows(rows)``: filter/reorder prepared rows before ``--limit``.

Options (``--option k=v``): ``temperature``, ``top_p``, ``top_k``,
``max_tokens`` (unset = the checkpoint's generation_config, which vLLM applies),
``answers=gold``, ``ns.<key>=<value>`` (any ns generation override, e.g.
``ns.chat_template_kwargs.enable_thinking=false``), and for judges
``judge_base_url``, ``judge_model`` (``self`` = the served model),
``judge_api_key_env``, ``judge_max_tokens``, ``judge_workers``,
``max_judge_invalid_frac`` (default 0.05; 0 = any invalid judgement fails),
``judge.<key>=<value>`` (any ns override for the judge step).
"""

from __future__ import annotations

import contextlib
import hashlib
import http.server
import json
import logging
import os
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, ClassVar, Iterator

import httpx

from granite_evals import data
from granite_evals.registry import Benchmark, failure_policy, pass_at_k_record, register

log = logging.getLogger(__name__)

NS_REPO = "https://github.com/NVIDIA-NeMo/Skills"
NS_COMMIT = "bcf059af55c20a89f797724598f9908d126153e6"
"""The NeMo-Skills commit pinned in pyproject's ``nemoskills`` extra."""

JUDGE_BASE_URL = ""  # set --option judge_base_url=<OpenAI-compatible judge endpoint>
JUDGE_MODEL = "aws/claude-sonnet-5"
JUDGE_API_KEY_ENV = "GRANITE_EVALS_JUDGE_API_KEY"

AZURE_JUDGE_BASE_URL = JUDGE_BASE_URL
AZURE_JUDGE_MODEL = "azure/gpt-4o-ncf"
"""HLE's official judge, reached through
the same gateway as the shared judge default."""

GENERATE_MODULE = "nemo_skills.inference.generate"
GENERATION_ATTEMPTS = 3


class NemoSkillsBenchmark(Benchmark):
    """A benchmark run by NeMo-Skills, configured by class attributes (module doc)."""

    extra = "nemoskills"
    harness_packages = ("nemo-skills", "litellm", "math-verify")
    splittable = True

    ns_benchmark: ClassVar[str]
    ns_split: ClassVar[str] = ""
    ns_prepare_args: ClassVar[tuple[str, ...]] = ()
    hf_revisions: ClassVar[dict[str, str]] = {}
    pinned_urls: ClassVar[dict[str, tuple[str, str]]] = {}
    prepared_sha256: ClassVar[str | None] = None

    ns_metric: ClassVar[str] = "symbolic_correct"
    ns_report_metrics: ClassVar[tuple[str, ...]] = ()
    ns_aggregation: ClassVar[str | None] = None
    ns_metrics_type: ClassVar[str | None] = None
    ns_metrics_kwargs: ClassVar[dict[str, Any]] = {}
    value_scale: ClassVar[float] = 0.01

    prompt_config: ClassVar[str | None] = None
    ns_generation_args: ClassVar[tuple[str, ...]] = ()
    ns_generation_module: ClassVar[str | None] = None

    requires_sandbox: ClassVar[bool | None] = None

    judge: ClassVar[bool | None] = None
    judge_generation_module: ClassVar[str | None] = None
    judge_args: ClassVar[tuple[str, ...]] = ()
    official_judge: ClassVar[str] = ""
    judge_model_default: ClassVar[str | None] = None
    judge_base_url_default: ClassVar[str | None] = None
    judge_api_key_env_default: ClassVar[str | None] = None
    """A per-class judge override (e.g. a benchmark whose official judge needs a
    different gateway/model than the shared JUDGE_MODEL/JUDGE_BASE_URL/
    JUDGE_API_KEY_ENV). None means fall through to the shared module default;
    still overridable per run by the usual judge_model/judge_base_url/
    judge_api_key_env options."""

    gold_answer_key: ClassVar[str] = "expected_answer"
    gold_judged: ClassVar[bool] = False
    """Whether ``answers=gold`` runs a judge benchmark too: the reference answers
    through the real judge, a check of the judge (expect ~100%)."""

    # -- options -----------------------------------------------------------

    def opt(self, key: str, default: Any) -> Any:
        """A ``--option key=value`` (always a string on the CLI), cast like ``default``."""
        value = self.config.options.get(key)
        if value is None:
            return default
        if isinstance(default, bool):
            return str(value).lower() in ("1", "true", "yes")
        return value if default is None else type(default)(value)

    @property
    def gold(self) -> bool:
        return self.opt("answers", "model") == "gold"

    def needs_server(self) -> bool:
        return not self.gold

    def score_needs_server(self) -> bool:
        return self.uses_judge() and self.opt("judge_model", self.judge_model_default or JUDGE_MODEL) == "self"

    # -- the ns dataset module ------------------------------------------------

    def ns_module(self):
        import importlib

        return importlib.import_module(f"nemo_skills.dataset.{self.ns_benchmark}")

    def _module_attr(self, name: str, default: Any = None) -> Any:
        return getattr(self.ns_module(), name, default)

    def split(self) -> str:
        return self.ns_split or self._module_attr("EVAL_SPLIT", "test")

    def metrics_type(self) -> str:
        return self.ns_metrics_type or self._module_attr("METRICS_TYPE")

    def aggregation(self) -> str:
        if self.ns_aggregation:
            return self.ns_aggregation
        return f"pass@1[avg-of-{self.repeats}]" if self.repeats > 1 else "pass@1"

    def uses_sandbox(self) -> bool:
        if self.requires_sandbox is not None:
            return self.requires_sandbox
        return bool(self._module_attr("REQUIRES_SANDBOX", False))

    def uses_judge(self) -> bool:
        if self.judge is not None:
            return self.judge
        return bool(self._module_attr("JUDGE_PIPELINE_ARGS") or self._module_attr("JUDGE_ARGS"))

    # -- 1. data -----------------------------------------------------------

    def pins(self) -> dict[str, str]:
        """HF repo -> commit for everything prepare.py reads."""
        pins = dict(self.hf_revisions)
        if self.dataset and self.dataset_revision and "/" in self.dataset and ":" not in self.dataset:
            pins.setdefault(self.dataset, self.dataset_revision)
        return pins

    def prepare_data(self) -> tuple[Path, dict[str, Any]]:
        """The prepared ns ``<split>.jsonl`` and its provenance."""
        split = self.split()
        if self.config.dataset and Path(self.config.dataset).exists():
            p = Path(self.config.dataset)
            path = p / f"{split}.jsonl" if p.is_dir() else p
            return path, {"source": "local", "path": str(path), "sha256": _sha256(path)}

        pins, redirects = self.pins(), {}
        if self.config.dataset:  # a hub id replacing the pinned primary dataset
            redirects[self.dataset] = self.config.dataset
            pins[self.config.dataset] = self.config.dataset_revision
        spec = {
            "benchmark": self.ns_benchmark,
            "argv": list(self.ns_prepare_args),
            "hf_revisions": pins,
            "redirects": redirects,
            "pinned_urls": {k: list(v) for k, v in self.pinned_urls.items()},
        }
        spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        ddir = self.config.output_dir / "ns-data" / self.ns_benchmark.replace(".", "/")
        path = ddir / f"{split}.jsonl"
        prov_path = ddir / "granite-provenance.json"
        if path.exists() and prov_path.exists() and json.loads(prov_path.read_text()).get("spec_hash") == spec_hash:
            prov = json.loads(prov_path.read_text())
        else:
            src = Path(self.ns_module().__file__).parent
            if ddir.exists():
                shutil.rmtree(ddir)
            shutil.copytree(src, ddir, ignore=shutil.ignore_patterns("__pycache__"))
            spec["dir"] = str(ddir)
            (ddir / "granite-prepare.json").write_text(json.dumps(spec, indent=2))
            log.info("%s: ns prepare_data %s %s", self.id, self.ns_benchmark, shlex.join(spec["argv"]))
            cmd = [sys.executable, "-c", "from granite_evals.benchmarks.nemo_skills import _prepare_main; _prepare_main()"]
            subprocess.run([*cmd, str(ddir / "granite-prepare.json")], check=True)
            if not path.exists():
                raise RuntimeError(f"ns prepare for {self.ns_benchmark} wrote no {path}")
            reads = json.loads((ddir / "granite-reads.json").read_text())
            prov = {"spec_hash": spec_hash, "reads": reads}
            prov_path.write_text(json.dumps(prov, indent=2))
        prov = {
            "source": "ns prepare_data",
            "ns_benchmark": self.ns_benchmark,
            "split": split,
            "ns_commit": NS_COMMIT,
            "reads": prov["reads"],
            "sha256": _sha256(path),
        }
        if self.prepared_sha256 and not self.config.dataset and prov["sha256"] != self.prepared_sha256:
            raise RuntimeError(
                f"{self.id}: prepared {path.name} has sha256 {prov['sha256']}, expected {self.prepared_sha256}"
            )
        return path, prov

    def prepare_rows(self, rows: list[dict]) -> list[dict]:
        """Hook: filter or reorder the prepared rows before ``--limit``."""
        return rows

    def load_rows(self, path: Path) -> list[dict]:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows = self.prepare_rows(rows)
        # The prepared file's order is fixed by the pins: --limit N takes its first N.
        indexed = data.take([{"i": i, "row": r} for i, r in enumerate(rows)], self.config.limit, key="i")
        return [x["row"] for x in indexed]

    # -- entry point -------------------------------------------------------

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        if self.gold and self.uses_judge() and not self.gold_judged:
            raise SystemExit(f"{self.id}: judge-scored, no reference answers for answers=gold")
        prepared, provenance = self.prepare_data()
        rows = self.load_rows(prepared)
        out = self.config.output_dir
        input_file = out / "input.jsonl"
        self._write_input(input_file, rows)
        log.info("%s: %d examples x %d repeats", self.id, len(rows), self.repeats)

        gen_dir, judged_dir = out / "generation", out / "judged"
        gen_files = [gen_dir / f"output-rs{k}.jsonl" for k in range(self.repeats)]
        with contextlib.ExitStack() as stack:
            # ns's code evaluators (SciCode) run in the sandbox: scoring only.
            sandbox_args = stack.enter_context(self.ns_sandbox()) if self.uses_sandbox() and self.scoring else []
            if self.generating:
                if self.gold:
                    base_url = stack.enter_context(_GoldServer(rows, self.gold_generation)).base_url
                    served_model_name = "gold"
                for k, f in enumerate(gen_files):
                    self._generate(input_file, f, base_url, served_model_name, k, sandbox_args)
                    log.info("%s repeat %d: generated %s", self.id, k, _status_line(f))
            if not self.scoring:
                return {
                    "n": len(rows),
                    "ns_benchmark": self.ns_benchmark,
                    "data_provenance": provenance,
                    "answers": "gold" if self.gold else "model",
                    "sampling": self.sampling(),
                    "generation_args": self.generation_args(sandbox_args=[]),
                    "statuses": _merge_counts(_statuses(f) for f in gen_files),
                    "generation_tokens": _token_stats(gen_dir, self.repeats),
                }
            for k, f in enumerate(gen_files):
                self._evaluate(f, self.generation_args(sandbox_args=[]), what=f"repeat {k} ({f})")
            judge_details = {}
            if self.uses_judge():
                judge_details = self._judge_all(gen_dir, judged_dir, base_url, served_model_name)

        scored_dir = judged_dir if self.uses_judge() else gen_dir
        files = [scored_dir / f"output-rs{k}.jsonl" for k in range(self.repeats)]
        metrics = self.compute_metrics(files)
        agg, key = self.aggregation(), self.ns_metric
        try:
            raw = metrics["_all_"][agg][key]
        except KeyError:
            raise RuntimeError(f"{self.id}: ns metrics have no [_all_][{agg}][{key}]: {_keys(metrics)}") from None
        reported = {m: _scaled(metrics["_all_"][agg].get(m), self.value_scale) for m in (key, *self.ns_report_metrics)}
        per_repeat = []
        for k, f in enumerate(files):
            m = self.compute_metrics([f])["_all_"].get("pass@1", {})
            per_repeat.append({"repeat": k, key: _scaled(m.get(key), self.value_scale), "statuses": _statuses(f)})
            log.info("%s repeat %d: %s=%s", self.id, k, key, per_repeat[-1][key])

        return {
            "value": raw * self.value_scale,
            "n": len(rows),
            "dataset": self.dataset_source()[0],
            "dataset_revision": self.dataset_source()[1],
            "harness": {"name": "nemo-skills", "repo": NS_REPO, "commit": NS_COMMIT},
            "ns_benchmark": self.ns_benchmark,
            "ns_metric": {"aggregation": agg, "key": key, "raw": raw, "scale": self.value_scale},
            "data_provenance": provenance,
            "answers": "gold" if self.gold else "model",
            "sampling": self.sampling(),
            "generation_args": self.generation_args(sandbox_args=[]),
            "metrics": reported,
            "pass_at_k": self.pass_at_k(metrics, len(rows)),
            "per_repeat": per_repeat,
            "statuses": _merge_counts(r["statuses"] for r in per_repeat),
            "generation_tokens": _token_stats(gen_dir, self.repeats),
            **judge_details,
            "ns_metrics": _jsonable(metrics),
        }

    def pass_at_k(self, metrics: dict[str, Any], n: int) -> dict[str, Any]:
        """ns's ``pass@1[avg-of-k]`` and ``pass@k`` of ``ns_metric`` (k = repeats;
        for k = 1 both are ns's pass@1)."""
        k, agg = self.repeats, metrics.get("_all_", {})
        p1 = agg.get(f"pass@1[avg-of-{k}]" if k > 1 else "pass@1", {}).get(self.ns_metric)
        pk = agg.get(f"pass@{k}", {}).get(self.ns_metric)
        return pass_at_k_record(k, _scaled(p1, self.value_scale), _scaled(pk, self.value_scale), n, "ns metrics")

    def _write_input(self, input_file: Path, rows: list[dict]) -> None:
        """The generation input; the score phase only checks that it is what was generated."""
        if self.generating:
            _write_jsonl(input_file, rows)
            self._reset_if_input_changed(input_file)
        else:
            _check_input(self, self.config.output_dir / "input.sha256", rows)

    def _reset_if_input_changed(self, input_file: Path) -> None:
        """A rerun with another --limit or dataset must not reuse old outputs."""
        stamp = self.config.output_dir / "input.sha256"
        digest = _sha256(input_file)
        if stamp.exists() and stamp.read_text().strip() != digest:
            log.warning("%s: input changed since the last run; discarding old generations", self.id)
            for d in ("generation", "judged", "judge"):
                shutil.rmtree(self.config.output_dir / d, ignore_errors=True)
        stamp.write_text(digest + "\n")

    # -- 2. generation -------------------------------------------------------

    def sampling(self) -> dict[str, Any]:
        """Sampling as sent: unset keys fall back to the checkpoint's
        generation_config (vLLM's defaults), not ns's greedy defaults."""
        s: dict[str, Any] = {"temperature": None, "top_p": None, "top_k": -1, "max_tokens": None}
        for key, cast in (("temperature", float), ("top_p", float), ("top_k", int), ("max_tokens", int)):
            if key in self.config.options:
                s[key] = cast(self.config.options[key])
        return s

    def generation_module(self) -> str:
        return self.ns_generation_module or self._module_attr("GENERATION_MODULE", GENERATE_MODULE)

    def generation_args(self, sandbox_args: list[str]) -> list[str]:
        """ns ``++`` overrides for one generation run (without files/server/seed)."""
        module_args = shlex.split(self._module_attr("GENERATION_ARGS", "") or "")
        prompt = self._module_attr("PROMPT_CONFIG", "")
        if prompt:
            module_args.insert(0, f"++prompt_config={prompt}")
        if self.prompt_config:
            module_args = [a for a in module_args if not a.startswith("++prompt_config=")]
            module_args.insert(0, f"++prompt_config={self.prompt_config}")
        s = self.sampling()
        args = [
            *module_args,
            *self.ns_generation_args,
            f"++inference.temperature={_hydra(s['temperature'])}",
            f"++inference.top_p={_hydra(s['top_p'])}",
            f"++inference.top_k={s['top_k']}",
            f"++inference.tokens_to_generate={_hydra(s['max_tokens'])}",
            *sandbox_args,
        ]
        return args + _passthrough(self.config.options, "ns.")

    def _generate(self, input_file: Path, output: Path, base_url: str, served: str, k: int, sandbox_args) -> None:
        if output.exists():
            return
        cmd = [
            sys.executable, "-m", self.generation_module(),
            f"++input_file={input_file}",
            f"++output_file={output}",
            "++server.server_type=vllm",
            f"++server.base_url={base_url}",
            f"++server.model={served}",
            "++server.enable_soft_fail=True",
            "++skip_filled=True",
            f"++max_concurrent_requests={self.config.workers}",
            f"++inference.random_seed={self.config.seed + k}",
            *self.generation_args(sandbox_args),
        ]  # fmt: skip
        cmd = self._defer_eval(cmd, output)
        _run_ns(cmd, output, log_path=output.parent / f"output-rs{k}.log", what=f"{self.id} generate rs{k}")

    def _defer_eval(self, cmd: list[str], output: Path) -> list[str]:
        """The generate phase generates without ns's evaluation, and marks the file
        unscored before it starts (a resumed half-generated file keeps its mark:
        its early rows were never evaluated)."""
        marker = _unscored(output)
        if self.phase != "generate" and not marker.exists():
            return cmd
        output.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        return [*cmd, "++eval_type=null"]  # after every other override: the last one wins

    def _evaluate(self, output: Path, args: list[str], what: str) -> None:
        """Grade a file generated without ns's evaluation: ns's batch evaluation with
        the ``eval_type`` / ``eval_config`` of the generation ``args``, as ns generate
        runs it after generating (``run_batch_evaluation``; a per-answer evaluator
        like math's gives the same fields in batch). An unmarked file was graded
        inside generation and is left as is."""
        if not output.exists():
            raise self.not_generated(what)
        marker = _unscored(output)
        if not marker.exists():
            return
        cmd = [sys.executable, "-c", "from granite_evals.benchmarks.nemo_skills import _evaluate_main; _evaluate_main()"]
        cmd += [str(output), *_eval_overrides(args)]
        log_path = output.with_name(output.stem + "-eval.log")
        log.info("%s evaluate %s: %s", self.id, output.name, shlex.join(cmd))
        with log_path.open("ab") as logf:
            r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=_ns_env())
        if r.returncode:
            raise RuntimeError(f"{self.id}: ns evaluation of {output} exited {r.returncode}; see {log_path}")
        marker.unlink()

    def gold_generation(self, row: dict) -> str:
        """The oracle response for ``answers=gold``: the reference answer in the
        form the benchmark's prompt asks for (``\\boxed{}`` for math)."""
        return f"\\boxed{{{row[self.gold_answer_key]}}}"

    # -- sandbox hook --------------------------------------------------------

    @contextlib.contextmanager
    def ns_sandbox(self) -> Iterator[list[str]]:
        """ns's code-execution sandbox for benchmarks that need it: its own flask
        server (``local_sandbox_server``) in the job container on a free port.
        Yields the ``++sandbox.*`` generation overrides. Override to run it
        elsewhere (e.g. in an enroot sandbox)."""
        port = _free_port()
        logf = (self.config.output_dir / "ns-sandbox.log").open("ab")
        cmd = [
            sys.executable, "-m", "flask",
            "--app", "nemo_skills.code_execution.local_sandbox.local_sandbox_server",
            "run", "--host", "127.0.0.1", "--port", str(port),
        ]  # fmt: skip
        log.info("%s: starting ns sandbox: %s", self.id, shlex.join(cmd))
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        env_before = {k: os.environ.get(k) for k in ("NEMO_SKILLS_SANDBOX_HOST", "NEMO_SKILLS_SANDBOX_PORT")}
        os.environ.update(NEMO_SKILLS_SANDBOX_HOST="127.0.0.1", NEMO_SKILLS_SANDBOX_PORT=str(port))
        try:
            _wait_http(f"http://127.0.0.1:{port}/health", proc, timeout_s=300)
            yield ["++sandbox.sandbox_type=local", "++sandbox.host=127.0.0.1", f"++sandbox.port={port}"]
        finally:
            for key, v in env_before.items():
                if v is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = v
            _kill(proc)
            logf.close()

    # -- 3. judge ------------------------------------------------------------

    def judge_endpoint(self, base_url: str, served: str) -> dict[str, Any]:
        model = self.opt("judge_model", self.judge_model_default or JUDGE_MODEL)
        if model == "self":
            return {"base_url": base_url, "model": served, "api_key_env": "", "is_self": True}
        return {
            "base_url": self.opt("judge_base_url", self.judge_base_url_default or JUDGE_BASE_URL),
            "model": model,
            "api_key_env": self.opt("judge_api_key_env", self.judge_api_key_env_default or JUDGE_API_KEY_ENV),
            "is_self": False,
        }

    def judge_upstream(self, ep: dict[str, Any]) -> str:
        """Where judge calls go: a paid judge always through the spend meter
        (``granite_evals.meter``, which caps the shared budget and answers 402 at
        the cap). A ``judge_base_url`` option is metered by the CLI already;
        the class default is wrapped here. ``judge_model=self`` is not paid."""
        from granite_evals import meter

        if ep["is_self"] or "judge_base_url" in self.config.options:
            return ep["base_url"]
        return meter.metered(ep["base_url"], role="judge")

    def judge_module(self) -> str:
        pipeline = self._module_attr("JUDGE_PIPELINE_ARGS", {}) or {}
        return self.judge_generation_module or pipeline.get("generation_module") or GENERATE_MODULE

    def judge_overrides(self) -> list[str]:
        module_args = shlex.split(self._module_attr("JUDGE_ARGS", "") or "")
        return [
            *module_args,
            *self.judge_args,
            # Judges run greedy; top_p unset because Anthropic-served judges reject
            # temperature and top_p together.
            "++inference.temperature=0.0",
            "++inference.top_p=null",
            f"++inference.tokens_to_generate={self.opt('judge_max_tokens', 16000)}",
            *_passthrough(self.config.options, "judge."),
        ]

    def _judge_all(self, gen_dir: Path, judged_dir: Path, base_url: str, served: str) -> dict[str, Any]:
        ep = self.judge_endpoint(base_url, served)
        official = self.official_judge or (self._module_attr("JUDGE_PIPELINE_ARGS", {}) or {}).get("model", "")
        key = os.environ.get(ep["api_key_env"], "") if ep["api_key_env"] else ""
        if ep["api_key_env"] and not key:
            raise SystemExit(f"{self.id}: judge key env {ep['api_key_env']} is empty (or use judge_model=self)")
        usage_log = self.config.output_dir / "judge" / "usage.jsonl"
        files = [judged_dir / f"output-rs{k}.jsonl" for k in range(self.repeats)]
        with _MeteringProxy(self.judge_upstream(ep), key, usage_log) as proxy:
            for k, out in enumerate(files):
                if out.exists() and not self._retry_invalid(out):
                    continue
                cmd = [
                    sys.executable, "-m", self.judge_module(),
                    f"++input_file={gen_dir / f'output-rs{k}.jsonl'}",
                    f"++output_file={out}",
                    "++server.server_type=openai",
                    f"++server.base_url={proxy.base_url}",
                    f"++server.model={ep['model']}",
                    "++server.api_key=granite-proxy",
                    "++server.enable_soft_fail=True",
                    "++skip_filled=True",
                    f"++max_concurrent_requests={self.opt('judge_workers', self.config.workers)}",
                    *self.judge_overrides(),
                ]  # fmt: skip
                _run_ns(cmd, out, log_path=judged_dir / f"output-rs{k}.log", what=f"{self.id} judge rs{k}")
                log.info("%s repeat %d: judged %s", self.id, k, _status_line(out))
                if proxy.budget_exhausted and k + 1 < self.repeats:
                    # Every later call would be refused too. The judged file stays: its
                    # refused judgements are invalid, and re-judged on resume.
                    raise SystemExit(
                        f"{self.id}: judge refused with HTTP 402, spend budget exhausted after repeat {k}; "
                        "rerun to resume"
                    )
        failed = total = 0
        for out in files:
            for row in _read_jsonl(out):
                f, t = self.judge_failures(row)
                failed, total = failed + f, total + t
        policy = failure_policy(
            self.id, "judge", failed, total, self.opt("max_judge_invalid_frac", 0.05), failed_key="judge_invalid"
        )
        if failed:
            log.warning("%s: %d/%d judgements invalid, left out of the score; rerun to re-judge them",
                        self.id, failed, total)  # fmt: skip
        usage = judge_usage(usage_log, n_examples=sum(_count_lines(f) for f in files))
        log.info("%s: judge usage %s", self.id, json.dumps(usage))
        return {
            "judge_model": ep["model"],
            "judge_base_url": ep["base_url"],
            "judge_is_self": ep["is_self"],
            "official_judge": official,
            "judge_module": self.judge_module(),
            "judge_args": self.judge_overrides(),
            "judge_usage": usage,
            **policy,
            "judge_budget_exhausted": proxy.budget_exhausted,
        }

    def judge_failures(self, row: dict) -> tuple[int, int]:
        """(invalid, total) judgements in one judged row. An invalid judgement is
        one ns leaves out of the score (a failed call, no parseable verdict)."""
        return 0, 0

    def _retry_invalid(self, out: Path) -> bool:
        """Set up a judged file's invalid rows for re-judging; False if it has none.

        The valid rows go back to ns's resume file (``<out>-async``, each at its
        ``_async_position``, generate.py skip_completed_samples) and the final
        file is removed, so ns with ``++skip_filled`` judges only the rest."""
        rows = _read_jsonl(out)
        keep = [{**r, "_async_position": i} for i, r in enumerate(rows) if not self.judge_failures(r)[0]]
        if len(keep) == len(rows):
            return False
        log.info("%s: re-judging %d rows of %s with invalid judgements", self.id, len(rows) - len(keep), out.name)
        _write_jsonl(out.with_name(out.name + "-async"), keep)
        out.unlink()
        return True

    # -- 4. metrics ----------------------------------------------------------

    def compute_metrics(self, files: list[Path]) -> dict[str, Any]:
        from nemo_skills.evaluation.metrics import ComputeMetrics

        cm = ComputeMetrics(
            self.ns_benchmark, metric_type=self.metrics_type(), metrics_kwargs=dict(self.ns_metrics_kwargs) or None
        )
        return cm.compute_metrics([str(f) for f in files])


# -- prepare (runs in a subprocess) ----------------------------------------


def _prepare_main() -> None:
    """Run one ns prepare.py with every upstream read pinned (spec: argv[1])."""
    import runpy
    import urllib.request

    spec = json.loads(Path(sys.argv[1]).read_text())
    ddir = Path(spec["dir"])
    pins, redirects = spec["hf_revisions"], spec["redirects"]
    urls = {k: tuple(v) for k, v in spec["pinned_urls"].items()}
    reads: list[dict] = []

    import datasets
    import huggingface_hub

    orig_load = datasets.load_dataset
    local_builders = {"json", "csv", "parquet", "text", "arrow", "webdataset"}

    def load_dataset(path, *args, **kwargs):
        if path in local_builders or Path(str(path)).exists():
            return orig_load(path, *args, **kwargs)
        repo = redirects.get(path, path)
        if repo not in pins:
            raise SystemExit(f"prepare.py reads HF dataset {path!r}, which the benchmark does not pin (hf_revisions)")
        kwargs["revision"] = pins[repo]
        reads.append({"hf_dataset": repo, "revision": pins[repo], "name": args[0] if args else kwargs.get("name")})
        if Path(repo).exists():
            kwargs.pop("revision")
        return orig_load(repo, *args, **kwargs)

    def _pinned_hub(fn):
        def wrapped(repo_id, *args, **kwargs):
            repo = redirects.get(repo_id, repo_id)
            if repo not in pins:
                raise SystemExit(f"prepare.py downloads {repo_id!r} from the HF hub, which is not pinned")
            kwargs["revision"] = pins[repo]
            reads.append({"hf_repo": repo, "revision": pins[repo], "via": fn.__name__})
            return fn(repo, *args, **kwargs)

        return wrapped

    def _pinned_url(url: str) -> tuple[str, str]:
        if url not in urls:
            raise SystemExit(f"prepare.py fetches {url}, which the benchmark does not pin (pinned_urls)")
        return urls[url]

    def urlretrieve(url, filename=None, *args, **kwargs):
        pinned, sha = _pinned_url(url)
        with httpx.Client(follow_redirects=True, timeout=300) as c:
            r = c.get(pinned)
            r.raise_for_status()
        digest = hashlib.sha256(r.content).hexdigest()
        if digest != sha:
            raise SystemExit(f"{pinned}: sha256 {digest} != pinned {sha}")
        Path(filename).write_bytes(r.content)
        reads.append({"url": url, "pinned_url": pinned, "sha256": sha})
        return filename, None

    datasets.load_dataset = load_dataset
    for name in ("hf_hub_download", "snapshot_download"):
        setattr(huggingface_hub, name, _pinned_hub(getattr(huggingface_hub, name)))
    urllib.request.urlretrieve = urlretrieve

    sys.argv = [str(ddir / "prepare.py"), *spec["argv"]]
    sys.path.insert(0, str(ddir))
    runpy.run_path(str(ddir / "prepare.py"), run_name="__main__")
    (ddir / "granite-reads.json").write_text(json.dumps(reads, indent=2))


# -- running ns ----------------------------------------------------------------


def _run_ns(cmd: list[str], output: Path, *, log_path: Path, what: str) -> None:
    """Run one ns generate/judge command; ns resumes a partial run, so an
    interrupted or crashed attempt is retried in place."""
    output.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, GENERATION_ATTEMPTS + 1):
        log.info("%s (attempt %d): %s", what, attempt, shlex.join(cmd))
        with log_path.open("ab") as logf:
            r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=_ns_env())
        if r.returncode == 0 and output.exists():
            return
        log.warning("%s: exit %d, see %s", what, r.returncode, log_path)
    raise RuntimeError(f"{what} failed {GENERATION_ATTEMPTS} times; see {log_path}")


def _evaluate_main() -> None:
    """ns's batch evaluation of one generation file (argv[1]), as ns generate runs it
    after generating: ``eval_type`` / ``eval_config`` from the ``++`` overrides
    (argv[2:]), composed by Hydra onto ns's generation config as ns generate does."""
    from hydra import compose, initialize
    from nemo_skills.evaluation.evaluator import evaluate
    from nemo_skills.inference import generate  # noqa: F401  registers base_generation_config
    from omegaconf import OmegaConf

    output, overrides = sys.argv[1], sys.argv[2:]
    with initialize(version_base=None):
        cfg = compose(config_name="base_generation_config", overrides=overrides)
    if cfg.eval_type is None:
        return
    # A DictConfig, as ns generate passes it (class evaluators read attributes).
    eval_config = OmegaConf.create({**OmegaConf.to_container(cfg.eval_config, resolve=True), "input_file": output})
    evaluate(cfg.eval_type, eval_config)


def _eval_overrides(args: list[str]) -> list[str]:
    """The ``eval_type`` / ``eval_config`` overrides among ns generation args."""
    keys = [a.lstrip("+").split("=", 1)[0] for a in args]
    return [a for a, k in zip(args, keys) if k == "eval_type" or k.split(".")[0] == "eval_config"]


def _unscored(output: Path) -> Path:
    """The marker of a generation file ns has not evaluated yet (``_defer_eval``)."""
    return output.with_name(output.name + ".unscored")


def _check_input(benchmark: Benchmark, stamp: Path, rows: list[dict]) -> None:
    """The score phase grades the input that was generated, or nothing."""
    digest = hashlib.sha256("".join(json.dumps(r) + "\n" for r in rows).encode()).hexdigest()
    if not stamp.exists():
        raise benchmark.not_generated(f"input {stamp.parent}")
    if stamp.read_text().strip() != digest:
        raise SystemExit(
            f"{benchmark.id}: the input in {stamp.parent} differs from what was generated "
            "(other options or data?); score with the generate phase's options"
        )


def _ns_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("OPENAI_API_KEY", "EMPTY")
    # litellm otherwise fetches its model price map from GitHub on import.
    env.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    return env


def _hydra(value: Any) -> str:
    return "null" if value is None else str(value)


def _passthrough(options: dict[str, Any], prefix: str) -> list[str]:
    return [f"++{k[len(prefix):]}={v}" for k, v in sorted(options.items()) if k.startswith(prefix)]


# -- local OpenAI-compatible helpers ---------------------------------------------


class _Server:
    """A threaded local HTTP server; subclasses implement ``handle``."""

    def __enter__(self):
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _serve(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(n) if n else b""
                status, headers, content = outer.handle(self.command, self.path, dict(self.headers), body)
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            do_GET = do_POST = _serve

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    def handle(self, method: str, path: str, headers: dict, body: bytes) -> tuple[int, dict, bytes]:
        raise NotImplementedError


def _json_response(obj: Any, status: int = 200) -> tuple[int, dict, bytes]:
    return status, {"Content-Type": "application/json"}, json.dumps(obj).encode()


class _GoldServer(_Server):
    """A fake endpoint answering each prompt with its row's oracle response, so
    ``answers=gold`` exercises ns's real prompt -> generate -> evaluate path."""

    def __init__(self, rows: list[dict], answer):
        self.rows, self.answer = rows, answer
        self.keys = [_row_text(r) for r in rows]

    def lookup(self, prompt: str) -> str:
        best = max(
            (i for i, key in enumerate(self.keys) if key and key in prompt),
            key=lambda i: len(self.keys[i]),
            default=None,
        )
        return "" if best is None else self.answer(self.rows[best])

    def handle(self, method, path, headers, body):
        if method == "GET" and path.rstrip("/").endswith("/models"):
            return _json_response({"object": "list", "data": [{"id": "gold", "object": "model"}]})
        if not path.rstrip("/").endswith("/completions"):
            return _json_response({"error": "not found"}, 404)
        req = json.loads(body or b"{}")
        if "messages" in req:
            prompt = "\n".join(_content_text(m.get("content")) for m in req["messages"])
        else:
            prompt = str(req.get("prompt", ""))
        text = self.lookup(prompt)
        usage = {"prompt_tokens": len(prompt.split()), "completion_tokens": len(text.split()) or 1, "total_tokens": 0}
        if "messages" in req:
            choice = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
            obj = "chat.completion"
        else:
            choice = {"index": 0, "text": text, "finish_reason": "stop", "logprobs": None}
            obj = "text_completion"
        return _json_response(
            {"id": "gold", "object": obj, "created": int(time.time()), "model": "gold", "choices": [choice], "usage": usage}
        )


def _row_text(row: dict) -> str:
    for key in ("problem", "question", "prompt"):
        if isinstance(row.get(key), str) and row[key].strip():
            return row[key].strip()
    return ""


def _content_text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content or "")


class _MeteringProxy(_Server):
    """Forwards OpenAI-compatible requests to ``target`` (adding the API key, so
    it never reaches an ns command line) and appends each response's token
    usage to ``usage_log``: the judge cost record behind ``details.judge_usage``."""

    def __init__(self, target: str, api_key: str, usage_log: Path, timeout_s: float = 1800):
        self.target = target.rstrip("/")
        self.api_key = api_key
        self.usage_log = usage_log
        self.lock = threading.Lock()
        self.client = httpx.Client(timeout=timeout_s)
        self.budget_exhausted = False
        usage_log.parent.mkdir(parents=True, exist_ok=True)

    def __exit__(self, *exc):
        super().__exit__(*exc)
        self.client.close()

    def handle(self, method, path, headers, body):
        sub = path[len("/v1") :] if path.startswith("/v1") else path
        fwd = {k: v for k, v in headers.items() if k.lower() in ("content-type", "accept")}
        if self.api_key:
            fwd["Authorization"] = f"Bearer {self.api_key}"
        started = time.time()
        try:
            r = self.client.request(method, self.target + sub, content=body or None, headers=fwd)
        except httpx.HTTPError as e:
            self._record({"status": f"error:{type(e).__name__}", "path": sub})
            return _json_response({"error": {"message": f"granite proxy: {e}"}}, 502)
        if r.status_code == 402:  # granite_evals.meter: the shared spend budget is used up
            self.budget_exhausted = True
        if method == "POST":
            record = {"status": r.status_code, "path": sub, "seconds": round(time.time() - started, 2)}
            with contextlib.suppress(ValueError):
                record.update(usage_fields(r.json().get("usage") or {}))
                record["model"] = r.json().get("model")
            self._record(record)
        ctype = r.headers.get("content-type", "application/json")
        return r.status_code, {"Content-Type": ctype}, r.content

    def _record(self, record: dict) -> None:
        with self.lock, self.usage_log.open("a") as f:
            f.write(json.dumps(record) + "\n")


def usage_fields(usage: dict) -> dict[str, int]:
    """prompt / completion / cached tokens from an OpenAI (or LiteLLM /
    Anthropic-style) usage object."""
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or usage.get("cache_read_input_tokens") or 0
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        "cached_tokens": int(cached),
        "cache_creation_tokens": int(usage.get("cache_creation_input_tokens") or 0),
    }


def judge_usage(usage_log: Path, n_examples: int) -> dict[str, Any]:
    """Totals and per-example averages of the judge's token usage."""
    records = [json.loads(line) for line in usage_log.read_text().splitlines()] if usage_log.exists() else []
    keys = ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_creation_tokens")
    total = {k: sum(r.get(k, 0) for r in records) for k in keys}
    return {
        "requests": len(records),
        "failed_requests": sum(1 for r in records if r.get("status") != 200),
        "examples_judged": n_examples,
        **total,
        "per_example": {k: (total[k] / n_examples if n_examples else 0.0) for k in keys},
        "log": str(usage_log),
    }


# -- small helpers ---------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_http(url: str, proc: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{url}: server exited with {proc.returncode}")
        with contextlib.suppress(httpx.HTTPError):
            if httpx.get(url, timeout=5).status_code == 200:
                return
        time.sleep(1)
    raise TimeoutError(f"{url} not ready after {timeout_s}s")


def _kill(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        import signal

        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _count_lines(path: Path) -> int:
    return len(_read_jsonl(path))


def _statuses(path: Path) -> dict[str, int]:
    """finish_reason counts; ns marks a soft-failed request ``error``."""
    counts: dict[str, int] = {}
    for r in _read_jsonl(path):
        s = str(r.get("finish_reason") or ("error" if r.get("error") else "unknown"))
        counts[s] = counts.get(s, 0) + 1
    return counts


def _status_line(path: Path) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(_statuses(path).items())) or "nothing"


def _merge_counts(dicts) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def _token_stats(gen_dir: Path, repeats: int) -> dict[str, Any]:
    toks = [
        int(r.get("num_generated_tokens") or 0)
        for k in range(repeats)
        for r in _read_jsonl(gen_dir / f"output-rs{k}.jsonl")
    ]
    return {"generations": len(toks), "mean": sum(toks) / len(toks) if toks else 0.0, "max": max(toks, default=0)}


def _scaled(v: Any, scale: float) -> float | None:
    return None if v is None else float(v) * scale


def _keys(metrics: dict) -> dict[str, list[str]]:
    return {agg: sorted(v) for agg, v in metrics.get("_all_", {}).items() if isinstance(v, dict)}


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


# -- benchmarks ------------------------------------------------------------------


@register
class AIME25(NemoSkillsBenchmark):
    """AIME 2025 I+II (30 problems), shipped in the ns repo (dataset/aime25/test.txt)."""

    id = "aime25"
    metric = "pass@1 symbolic correct"
    ns_benchmark = "aime25"
    dataset = f"{NS_REPO}/tree/{NS_COMMIT}/nemo_skills/dataset/aime25"
    dataset_revision = NS_COMMIT
    prepared_sha256 = "b672f9c6660b68dd42f5fc705b17405f24eaa9991c98991d63463c6b683468a8"


@register
class HMMTFeb25(NemoSkillsBenchmark):
    id = "hmmt-feb25"
    metric = "pass@1 symbolic correct"
    ns_benchmark = "hmmt_feb25"
    dataset = "MathArena/hmmt_feb_2025"
    dataset_revision = "6fdc4277120810ff75aa22d2d5489b91f7a262a1"


@register
class GPQA(NemoSkillsBenchmark):
    """GPQA Diamond (198), ns's 4-choice prompt; choices shuffled with ns's seed 42."""

    id = "gpqa"
    metric = "pass@1 symbolic correct"
    default_repeats = 2
    """avg-of-2: the granite-4.2 reference run scored pass@1[avg-of-2]."""
    ns_benchmark = "gpqa"
    ns_split = "diamond"
    ns_prepare_args = ("--split", "diamond")
    dataset = "Idavidrein/gpqa"  # gated: needs HF_TOKEN
    dataset_revision = "83022cefff930aea54f654c0b282e74b9eeda5c6"

    def gold_generation(self, row: dict) -> str:
        return f"Answer: {row[self.gold_answer_key]}"


MMLU_PRO_SHOTS = 5
"""ns ships 5 CoT examples per MMLU-Pro category, so ``shots`` is 0 or 5."""

MMLU_PRO_FEW_SHOT_ARGS = ("++prompt_config=generic/general-boxed", "++examples_type='{examples_type}'")
"""ns's few-shot mechanism for MMLU-Pro: prepare.py writes each row's
``examples_type`` (``mmlu_pro_few_shot_<category>``), ns formats it per row into
the 5 examples of ``prompt/few_shot_examples/mmlu_pro.py`` (MMLU-Pro's validation
split CoT, ending ``The answer is \\boxed{X}``), and ``generic/general-boxed``
puts them before the question. The quotes keep Hydra from reading ``{...}`` as
a dict (no shell here)."""


@register
class MMLUPro(NemoSkillsBenchmark):
    """MMLU-Pro (12032), 5-shot CoT by default: ns's per-category validation-split
    examples (``MMLU_PRO_FEW_SHOT_ARGS``), graded by ns's multichoice match on the
    ``\\boxed{}`` letter. ``--option shots=0``: ns's default 0-shot prompt
    (``eval/aai/mcq-10choices``, ``Answer: X``)."""

    id = "mmlu-pro"
    metric = "5-shot CoT symbolic correct"
    default_repeats = 1
    ns_benchmark = "mmlu-pro"
    ns_prepare_args = ("--split", "test")
    dataset = "TIGER-Lab/MMLU-Pro"
    dataset_revision = "b189ec765aa7ed75c8acfea42df31fdae71f97be"

    def shots(self) -> int:
        n = self.opt("shots", MMLU_PRO_SHOTS)
        if n not in (0, MMLU_PRO_SHOTS):
            raise SystemExit(f"{self.id}: shots={n}: ns has {MMLU_PRO_SHOTS} examples per category; use 0 or {MMLU_PRO_SHOTS}")
        return n

    def generation_args(self, sandbox_args: list[str]) -> list[str]:
        args = super().generation_args(sandbox_args)
        if not self.shots():
            return args
        return [*MMLU_PRO_FEW_SHOT_ARGS, *(a for a in args if not a.startswith("++prompt_config="))]

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        return {**super().run(base_url, served_model_name), "shots": self.shots()}

    def gold_generation(self, row: dict) -> str:
        if self.shots():
            return super().gold_generation(row)  # \boxed{X}
        return f"Answer: {row[self.gold_answer_key]}"


_ARENA_HARD = "https://raw.githubusercontent.com/lmarena/arena-hard-auto/{}/data/arena-hard-v2.0/"
_ARENA_HARD_COMMIT = "196f6b826783b3da7310e361a805fa36f0be83f3"


_ARENA_VERDICTS = ("A=B", "A>B", "A>>B", "B>A", "B>>A")
"""The verdicts ns's get_battles_from_judgment scores (evaluator/arena.py:103-158)."""


def _arena_url(ref: str, path: str) -> str:
    return _ARENA_HARD.format(ref) + path


@register
class ArenaHardV2(NemoSkillsBenchmark):
    """Arena-Hard-Auto v2.0 (750: 500 hard prompts vs o3-mini-2025-01-31, 250 creative
    writing vs gemini-2.0-flash-001), judged pairwise both orders by ns's arena judge;
    win rate from ns's bootstrapped Bradley-Terry fit (no style control). The judge
    is the family's ``judge_model`` (aws/claude-sonnet-5), not the official GPT-4.1
    (``official_judge``, recorded in results.json), so scores are not directly
    comparable with published Arena-Hard-V2 numbers."""

    id = "arena-hard-v2"
    metric = "win rate"
    default_repeats = 1
    ns_benchmark = "arena-hard-v2"
    ns_metric = "score"
    dataset = f"https://github.com/lmarena/arena-hard-auto/tree/{_ARENA_HARD_COMMIT}/data/arena-hard-v2.0"
    dataset_revision = _ARENA_HARD_COMMIT
    pinned_urls = {
        _arena_url("main", p): (_arena_url(_ARENA_HARD_COMMIT, p), sha)
        for p, sha in (
            ("question.jsonl", "a75cef6623db6b27aec810497b059f0659705d7ff48dcbccc5bd5b130728ac73"),
            ("model_answer/o3-mini-2025-01-31.jsonl", "c23a7c255eead99f7f86ad48aa1d8ccdb5b3334f2097e0e1e09184df87265e03"),
            ("model_answer/gemini-2.0-flash-001.jsonl", "70f614e1345f787a1da98c239d27f6df3c1d26b411cf894949f934a3039d6f0b"),
        )
    }
    official_judge = "gpt-4.1 (arena-hard-auto v2.0; NeMo-Skills default)"

    def judge_failures(self, row: dict) -> tuple[int, int]:
        """Both judgements of a row, parsed as ns does (arena_metrics.py:34-44):
        anything but the five verdicts is an invalid score, which ns counts and
        leaves out of the battles (evaluator/arena.py:127-129, :152-154)."""
        from nemo_skills.evaluation.metrics.arena_metrics import ArenaMetrics

        parse = ArenaMetrics()._get_judge_score
        scores = [parse(row.get(key) or "") for key in ("judgement-gen-base", "judgement-base-gen")]
        return sum(s not in _ARENA_VERDICTS for s in scores), len(scores)

    def compute_metrics(self, files: list[Path]) -> dict[str, Any]:
        """ns's Bradley-Terry fit needs both outcomes in the sample; a small or
        one-sided sample (smoke runs, one category) makes it raise. There the fit's
        answer is the plain weighted win rate, used instead and flagged."""
        from nemo_skills.evaluation.evaluator import arena

        fit, fallbacks = arena.get_aggregate_score, []
        arena.get_aggregate_score = lambda scores, weight=3: _arena_score(fit, scores, weight, fallbacks)
        try:
            metrics = super().compute_metrics(files)
        finally:
            arena.get_aggregate_score = fit
        metrics["win_rate_method"] = (
            f"weighted win fraction for {len(fallbacks)} one-sided (sub)sample(s), Bradley-Terry fit undefined"
            if fallbacks
            else "bootstrapped Bradley-Terry (ns)"
        )
        return metrics


def _arena_score(fit, scores, weight: int = 3, fallbacks: list | None = None) -> dict[str, Any]:
    try:
        return fit(scores, weight)
    except ValueError:
        if fallbacks is not None:
            fallbacks.append(len(scores))
        from nemo_skills.evaluation.evaluator.arena import get_battles_from_judgment

        battles, invalid = get_battles_from_judgment(scores, weight)
        winners = list(battles["winner"]) if len(battles) else []
        points = sum(1.0 if w == "model_a" else 0.5 if w == "tie" else 0.0 for w in winners)
        return {
            "score": round(100 * points / len(winners), 2) if winners else 0.0,
            "95_CI": (0.0, 0.0),
            "invalid_scores": invalid,
        }
