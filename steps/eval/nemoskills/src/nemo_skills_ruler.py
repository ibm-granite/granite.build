"""RULER through NeMo-Skills (Granite: RULER 64k / 128k / 256k / 512k / 1M completions).

ns's RULER setup, run locally: ``nemo_skills/dataset/ruler/prepare.py`` (pinned ns)
generates the 13 synthetic tasks with NVIDIA/RULER's generators (pinned commit,
baked into the image at ``/opt/ruler`` with its four source json files verified by
sha256) for the evaluated model's tokenizer, 100 samples per task at
``max_seq_length - 50`` tokens (ns's template allowance), seed 42. Each task then
runs as its own ns generation with the task's ns GENERATION_ARGS: the text
completions endpoint (ns applies the model's chat template with its tokenizer and
prefills RULER's answer prefix), per-task ``tokens_to_generate`` and ns's RULER
match (``all``, or ``part`` for QA). The headline is ns's ``ruler_score``: the
mean accuracy over the 13 tasks.

The data depend on the tokenizer, so they are generated in the job (a few minutes
at 128k) and cached under ``<output_dir>/ns-data`` keyed by the RULER commit,
length and tokenizer; ``--limit N`` takes the first N samples of *each* task.

Thinking (a Granite decision, and a departure from ns/RULER). ns's RULER setup is
for non-reasoning generation: RULER's answer prefix is prefilled after the chat
template and each task gets a few dozen tokens (``tokens_to_generate``: niah 128,
vt 30, cwe 120, fwe 50, qa 32). A thinking model then writes its answer inside a
cut-off thinking block, or not at all. So with thinking on (the chat template's
default for Granite: ``enable_thinking`` unset or true):

* the data are ns's own ``--data_format chat``: no answer-prefix prefill, the chat
  endpoint, and the chat template's thinking;
* each sample generates up to the context cap: ``max_tokens = cap - prompt
  tokens``, counted per sample with the model's tokenizer and chat template. The
  cap is the served ``max_model_len`` (``--option context_cap=N`` lowers it);
  ``--option max_tokens=N`` bounds the per-sample budget further;
* only the post-thinking content is scored. ns scores ``generation``, which is the
  message content after vLLM's reasoning parser; the thinking goes to
  ``reasoning_content``. A generation cut off inside its thinking has no content
  and scores 0, so a needle quoted in the thinking cannot match. A ``<think>`` or
  ``</think>`` in any ``generation`` (no reasoning parser) fails the run.

RULER sizes a sample to ``max_seq_length`` minus the task's answer budget, so at
``max_model_len == max_seq_length`` (granite-4.2 at 128k) thinking gets only the
slack RULER leaves; a longer served context leaves more. A sample scores 0, and
does not stop the run, when its prompt plus the answer budget exceeds the cap
(``prompt_exceeds_cap``: no request is sent), when the server answers with a
context-length error (``context_length_error``), or when generation reaches the
cap before any answer (``length_before_answer``). Each is appended, flushed, as
one json line to ``<output_dir>/failures.jsonl`` as it happens (task, repeat,
sample index, reason, prompt and generated tokens); results.json ``failures``
counts them by reason and task from the output rows (the log is append-only: a
sample retried after a crash can appear twice there, not in the counts).

``--option sample_length=N`` builds shorter samples: a departure (the headline is
RULER at N tokens, not at ``max_seq_length``). ``enable_thinking=false`` runs ns's
RULER exactly (text endpoint, answer prefix, task budgets; a sample whose prompt
plus budget exceeds the cap is still recorded as ``prompt_exceeds_cap``).
results.json records ``context_cap``, ``thinking`` (mode, budget rule, per-task
answer budgets, prompt and max-token percentiles, data format, scoring source,
cut-off counts), ``failures`` and ``departures``.

Lengths: ruler-64k and ruler-128k (Granite 4.2); ruler-256k, ruler-512k and
ruler-1m are in the extended suite and need a server with that much context.
granite-4.2 is only verified to 128k.

Phases: ``--phase generate`` checks the served context, builds the data (the
tokenizer) and generates each task with ns's evaluation off; ``--phase score``
runs ns's RULER match, the think-tag check and the metrics on those files, on
the data generate built (no tokenizer, no model).

Options: ``tokenizer`` (default: ``--model``), ``enable_thinking``,
``context_cap``, ``sample_length`` (default ``max_seq_length``), ``tasks``
(comma-separated subset, for debugging; the headline then averages only those),
``ruler_dir``, the sampling options and ``ns.<key>=<value>`` of the NeMo-Skills
family (``max_tokens`` replaces every task's ``tokens_to_generate`` with thinking
off, and bounds ``cap - prompt tokens`` with thinking on), and ``answers=gold``
(serves each sample's expected outputs; needs a tokenizer but no GPU; the cap is
then ``context_cap`` or ``sample_length``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import runpy
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar

import httpx

from granite_evals import data
from granite_evals.benchmarks.nemo_skills import (
    NS_COMMIT,
    NS_REPO,
    NemoSkillsBenchmark,
    _GoldServer,
    _hydra,
    _jsonable,
    _passthrough,
    _read_jsonl,
    _run_ns,
    _scaled,
    _sha256,
    _status_line,
    _statuses,
    _merge_counts,
    _check_input,
    _write_jsonl,
    log,
)
from granite_evals.registry import pass_at_k_record, register
from granite_evals.results import GENERATION_FILE

RULER_REPO = "https://github.com/NVIDIA/RULER"
RULER_COMMIT = "c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a"
RULER_DIR = "/opt/ruler"
"""The pinned RULER checkout with its source data (docker/extras/nemoskills.sh)."""

RULER_JSON_SHA256 = {
    "english_words.json": "affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca",
    "hotpotqa.json": "e3da074df24e8369009918aa5cdbdd254dadcde4c63f7569d36afd6f2268caa8",
    "PaulGrahamEssays.json": "8d31e1b660e0f2180bcca6d238e18f77921df9d158611582b860da1762b6d3dd",
    "squad.json": "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
}
"""RULER's generator inputs (scripts/data/synthetic/json): LFS word list, HotpotQA
dev distractor, the essays RULER's script scrapes, SQuAD v2.0 dev."""

TASKS = (
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
)
"""ns's RULER tasks (the 13 ``ruler_score`` averages)."""

NUM_SAMPLES = 100
TEMPLATE_TOKENS = 50

TOKENS_TO_GENERATE = {"niah": 128, "vt": 30, "cwe": 120, "fwe": 50, "qa": 32}
"""ns/RULER's per-task answer budget (ns prepare.py). RULER sizes each sample to
leave room for it within ``max_seq_length``."""

FAILURES_FILE = "failures.jsonl"
"""Per-sample failures (scored 0), appended as they happen: ``<output_dir>/failures.jsonl``."""

FAILURE_REASONS = ("prompt_exceeds_cap", "context_length_error", "length_before_answer")

SPEC_ARG = "--granite-spec="
"""``_generate_main``'s argument: the json spec of one task/repeat generation."""

SCORING_SOURCES = {
    # thinking on: ns's chat format on the chat endpoint
    "chat": "generation: message content after the vLLM reasoning parser (reasoning_content not scored)",
    # thinking off: ns's default format on the text endpoint (no reasoning parser runs)
    "default": "generation: raw text completion after the prefilled answer prefix (text endpoint; no reasoning parser)",
}


class RulerBenchmark(NemoSkillsBenchmark):
    """RULER at one context length; subclasses set ``id`` and ``max_seq_length``."""

    metric = "accuracy"
    default_repeats = 1
    ns_benchmark = "ruler"
    ns_metric = "accuracy"
    ns_metrics_type = "ruler"
    dataset = f"{RULER_REPO}/tree/{RULER_COMMIT}"
    dataset_revision = RULER_COMMIT
    harness_packages = (*NemoSkillsBenchmark.harness_packages, "wonderwords", "nltk", "transformers")
    max_seq_length: ClassVar[int]
    splittable = True

    # -- config ------------------------------------------------------------

    def tokenizer(self) -> str:
        return self.opt("tokenizer", self.config.model)

    def tasks(self) -> list[str]:
        tasks = [t for t in str(self.opt("tasks", ",".join(TASKS))).split(",") if t]
        unknown = sorted(set(tasks) - set(TASKS))
        if unknown:
            raise SystemExit(f"{self.id}: unknown RULER tasks {unknown}; known: {', '.join(TASKS)}")
        return tasks

    def thinking(self) -> bool:
        """Thinking on unless ``enable_thinking=false`` (the chat template's default is on)."""
        return self.opt("enable_thinking", True)

    def data_format(self) -> str:
        """ns prepare's format: ``chat`` for thinking, ns's ``default`` (answer prefix) otherwise."""
        return "chat" if self.thinking() else "default"

    def sample_length(self) -> int:
        n = self.opt("sample_length", self.max_seq_length)
        if not 0 < n <= self.max_seq_length:
            raise SystemExit(f"{self.id}: sample_length={n} must be in 1..{self.max_seq_length}")
        return n

    def setup_name(self) -> str:
        return f"granite_{self.sample_length()}" + ("_chat" if self.data_format() == "chat" else "")

    def answer_budget(self, task: str) -> int:
        """ns/RULER's answer budget for the task (RULER leaves room for it in the sample)."""
        return TOKENS_TO_GENERATE[task.split("_")[0]]

    def tokens_to_generate(self, task: str) -> int | None:
        """The ns ``tokens_to_generate``: thinking off, the task's answer budget or
        ``max_tokens``; thinking on, ``max_tokens`` or None (the per-sample budget is
        ``cap - prompt tokens``, bounded by it)."""
        s = self.sampling()
        if s["max_tokens"] is not None:
            return s["max_tokens"]
        return None if self.thinking() else self.answer_budget(task)

    def required_context(self) -> int:
        """Served context for a full sample (generation fills what is left of the cap)."""
        return self.sample_length()

    def context_cap(self, served: int | None) -> int:
        """Prompt plus generation per sample: ``context_cap``, else the served
        max_model_len; ``sample_length`` for gold answers or thinking off when the
        server does not say."""
        if "thinking_budget" in self.config.options:
            raise SystemExit(
                f"{self.id}: thinking_budget was replaced by max_tokens = context cap - prompt tokens per sample; "
                "use --option context_cap=N or --option max_tokens=N to bound it"
            )
        cap = self.opt("context_cap", 0) or served
        if served is not None and cap > served:
            raise SystemExit(f"{self.id}: context_cap={cap} > the served max_model_len={served}")
        if cap is None:
            if self.thinking() and not self.gold:
                raise SystemExit(f"{self.id}: could not read the served max_model_len; pass --option context_cap=N")
            cap = self.sample_length()
        if cap < self.sample_length():
            raise SystemExit(f"{self.id}: context_cap={cap} < sample_length={self.sample_length()}")
        return cap

    def departures(self, cap: int | None) -> list[str]:
        out = []
        if self.thinking():
            out.append(
                f"thinking on: ns --data_format chat (no answer-prefix prefill); max_tokens per sample = context "
                f"cap {cap} - prompt tokens (ns/RULER: the task's answer budget, for non-reasoning generation); "
                "prompt + answer budget over the cap, context-length errors and generations that reach the cap "
                f"before an answer score 0 ({FAILURES_FILE}); scored on the post-thinking content only"
            )
        if self.sample_length() != self.max_seq_length:
            out.append(f"samples built at {self.sample_length()} tokens, not {self.max_seq_length}")
        return out

    def ruler_dir(self) -> Path:
        return Path(self.opt("ruler_dir", RULER_DIR))

    # -- 1. data -----------------------------------------------------------

    def check_ruler_source(self) -> dict[str, str]:
        """The RULER checkout is the pinned commit and its json inputs match."""
        root = self.ruler_dir()
        stamp = root / "GRANITE_EVALS_COMMIT"
        if not stamp.exists() or stamp.read_text().strip() != RULER_COMMIT:
            raise SystemExit(f"{self.id}: {root} is not RULER {RULER_COMMIT} (see docker/extras/nemoskills.sh)")
        shas = {}
        for name, want in RULER_JSON_SHA256.items():
            got = _sha256(root / "scripts/data/synthetic/json" / name)
            if got != want:
                raise SystemExit(f"{self.id}: RULER {name} sha256 {got} != pinned {want}")
            shas[name] = got
        return shas

    def tokenizer_fingerprint(self) -> dict[str, str]:
        tok = Path(self.tokenizer())
        if not tok.is_dir():
            return {"tokenizer": self.tokenizer()}
        files = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model")
        return {"tokenizer": str(tok), **{f: _sha256(tok / f) for f in files if (tok / f).exists()}}

    def prepare_data(self) -> tuple[Path, dict[str, Any]]:
        """ns's RULER prepare for this length and tokenizer: ``<setup>/<task>/test.jsonl``."""
        tasks = self.tasks()
        if self.config.dataset and Path(self.config.dataset).is_dir():
            setup_dir = Path(self.config.dataset)
            return setup_dir, {"source": "local", "path": str(setup_dir), **self._task_files(setup_dir, tasks)}
        if self.tokenizer() in ("", "none") and self.generating:
            raise SystemExit(f"{self.id}: RULER data are built for a tokenizer: pass --model or --option tokenizer=")
        spec = {
            "ns_commit": NS_COMMIT,
            "ruler_commit": RULER_COMMIT,
            "max_seq_length": self.sample_length(),
            "template_tokens": TEMPLATE_TOKENS,
            "data_format": self.data_format(),
            "num_samples": NUM_SAMPLES,
            "tasks": tasks,
            **(self.tokenizer_fingerprint() if self.generating else {}),
        }
        spec_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        ddir = self.config.output_dir / "ns-data" / "ruler"
        setup_dir = ddir / self.setup_name()
        prov_path = setup_dir / "granite-provenance.json"
        if not self.generating:  # score grades the data generate built (no tokenizer needed)
            if not prov_path.exists():
                raise self.not_generated(f"RULER data {setup_dir}")
            prov = json.loads(prov_path.read_text())
        elif prov_path.exists() and json.loads(prov_path.read_text()).get("spec_hash") == spec_hash:
            prov = json.loads(prov_path.read_text())
        else:
            json_shas = self.check_ruler_source()
            self._run_prepare(ddir, tasks)
            prov = {"spec_hash": spec_hash, "spec": spec, "ruler_json_sha256": json_shas}
            prov.update(self._task_files(setup_dir, tasks))
            prov_path.write_text(json.dumps(prov, indent=2))
        return setup_dir, {
            "source": "ns prepare_data ruler",
            "ns_commit": NS_COMMIT,
            "ruler": {"repo": RULER_REPO, "commit": RULER_COMMIT},
            "setup": self.setup_name(),
            **{k: v for k, v in prov.items() if k != "spec_hash"},
        }

    def _task_files(self, setup_dir: Path, tasks: list[str]) -> dict[str, Any]:
        files = {}
        for t in tasks:
            path = setup_dir / t / "test.jsonl"
            rows = len(_read_jsonl(path))
            # RULER's generator scripts can fail without failing prepare.py.
            if rows != NUM_SAMPLES:
                raise RuntimeError(f"{self.id}: {path} has {rows} samples, expected {NUM_SAMPLES} (see ns-data/ruler/prepare.log)")
            files[t] = {"rows": rows, "sha256": _sha256(path)}
        return {"task_files": files}

    def _run_prepare(self, ddir: Path, tasks: list[str]) -> None:
        import nemo_skills.dataset.ruler as ns_ruler

        if ddir.exists():
            shutil.rmtree(ddir)
        shutil.copytree(Path(ns_ruler.__file__).parent, ddir, ignore=shutil.ignore_patterns("__pycache__"))
        # RULER's prepare writes next to its scripts: work on a copy of the checkout.
        tmp = ddir / "tmp"
        shutil.copytree(self.ruler_dir(), tmp / "RULER", ignore=shutil.ignore_patterns(".git", "__pycache__"))
        # ns's prepare pip-installs RULER's deps; they are in the nemoskills extra
        # already, and the image env must not change at run time.
        shim = ddir / "bin"
        shim.mkdir()
        (shim / "pip").write_text("#!/bin/sh\necho \"granite: pip disabled during RULER prepare: $*\" >&2\n")
        (shim / "pip").chmod(0o755)
        argv = [
            "--setup", self.setup_name(),
            "--max_seq_length", str(self.sample_length()),
            "--template_tokens", str(TEMPLATE_TOKENS),
            "--data_format", self.data_format(),
            "--tmp_data_dir", str(tmp),
            "--tasks", *tasks,
            # passed through to RULER's prepare.py
            "--tokenizer_path", self.tokenizer(),
            "--num_samples", str(NUM_SAMPLES),
        ]  # fmt: skip
        spec = {"dir": str(ddir), "argv": argv, "hf_revisions": {}, "redirects": {}, "pinned_urls": {}}
        (ddir / "granite-prepare.json").write_text(json.dumps(spec, indent=2))
        env = dict(os.environ, PATH=f"{shim}:{Path(sys.executable).parent}:{os.environ.get('PATH', '')}")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        cmd = [sys.executable, "-c", "from granite_evals.benchmarks.nemo_skills import _prepare_main; _prepare_main()"]
        log.info("%s: ns prepare_data ruler %s", self.id, shlex.join(argv))
        with (ddir / "prepare.log").open("ab") as logf:
            subprocess.run([*cmd, str(ddir / "granite-prepare.json")], check=True, env=env, cwd=ddir, stdout=logf, stderr=subprocess.STDOUT)

    # -- 2. generation -------------------------------------------------------

    def task_generation_args(self, setup_dir: Path, task: str) -> list[str]:
        """The task's ns GENERATION_ARGS (written by prepare), plus sampling and
        the generation budget (Hydra: the last override wins)."""
        module = runpy.run_path(str(setup_dir / task / "__init__.py"))
        args = shlex.split(module.get("GENERATION_ARGS", ""))
        prefilled = any(a.startswith("++start_assistant_response_key=") for a in args)
        if prefilled == (self.data_format() == "chat"):
            raise SystemExit(
                f"{self.id}: {setup_dir / task} is not ns's {self.data_format()!r} RULER format "
                f"(thinking {'on' if self.thinking() else 'off'}); rebuild the data or set enable_thinking"
            )
        s = self.sampling()
        args += [
            f"++tokenizer={self.tokenizer()}",
            f"++inference.temperature={_hydra(s['temperature'])}",
            f"++inference.top_p={_hydra(s['top_p'])}",
            f"++inference.top_k={s['top_k']}",
        ]
        if self.data_format() == "chat":
            args.append("++inference.endpoint_type=chat")
        if s["max_tokens"] is not None or self.data_format() == "chat":  # ns's task value otherwise
            # thinking on: a bound on (null: no bound on) cap - prompt tokens (capped_generation_task)
            args.append(f"++inference.tokens_to_generate={_hydra(self.tokens_to_generate(task))}")
        if "enable_thinking" in self.config.options:
            args.append(f"++chat_template_kwargs.enable_thinking={str(self.opt('enable_thinking', True)).lower()}")
        return args + _passthrough(self.config.options, "ns.")

    def check_context(self, base_url: str, served: str) -> int | None:
        """The served max_model_len (the context cap) must hold a full sample."""
        with contextlib.suppress(httpx.HTTPError, ValueError, KeyError):
            models = httpx.get(base_url.rstrip("/") + "/models", timeout=30).json()["data"]
            lens = [m.get("max_model_len") for m in models if m.get("id") == served] or [m.get("max_model_len") for m in models]
            n = next((x for x in lens if x), None)
            need = self.required_context()
            if n is not None and n < need:
                raise SystemExit(
                    f"{self.id}: {served} is served with max_model_len={n} < {need}-token samples; serve with "
                    f"--max-model-len {need} if the model supports it, otherwise --option sample_length=N "
                    "(shorter samples: a departure)"
                )
            return n
        return None

    def check_generations(self, output: Path) -> None:
        """Only post-thinking content is scored: a think tag in ``generation`` means
        the reasoning was not split off, and ns's substring match would see it."""
        tags = ("<think>", "</think>")
        bad = [r.get("index") for r in _read_jsonl(output) if any(t in (r.get("generation") or "") for t in tags)]
        if bad:
            raise RuntimeError(
                f"{self.id}: {output}: think tags in 'generation' for samples {bad[:10]}: the reasoning was not "
                "split off (serve with the model's --reasoning-parser); not scoring thinking"
            )

    def _task_rows(self, out: Path, task: str) -> list[dict]:
        return [r for k in range(self.repeats) for r in _read_jsonl(out / task / f"output-rs{k}.jsonl")]

    def thinking_record(self, out: Path, tasks: list[str]) -> dict[str, Any]:
        """What was generated and scored: mode, budgets, cut-offs, token percentiles."""
        per_task = {}
        for t in tasks:
            rows = self._task_rows(out, t)
            toks = sorted(int(r.get("num_generated_tokens") or 0) for r in rows)
            cut = [r for r in rows if r.get("finish_reason") == "length"]
            per_task[t] = {
                "answer_budget": self.answer_budget(t),
                "tokens_to_generate": self.tokens_to_generate(t),
                "prompt_tokens": _percentiles(_ints(rows, "granite_prompt_tokens")),
                "max_tokens": _percentiles(_ints(rows, "granite_max_tokens")),
                "generations": len(rows),
                # cut off before any content: nothing to score (0)
                "length_no_content": sum(not (r.get("generation") or "").strip() for r in cut),
                "length_with_content": sum(bool((r.get("generation") or "").strip()) for r in cut),
                "with_reasoning": sum(bool(r.get("reasoning_content")) for r in rows),
                "generated_tokens": _percentiles(toks),
            }
        return {
            "enabled": self.thinking(),
            "enable_thinking": self.config.options.get("enable_thinking", "unset (chat template default)"),
            "budget": self.budget_rule(),
            "data_format": self.data_format(),
            "endpoint": "chat" if self.data_format() == "chat" else "text (answer prefix prefilled)",
            "scoring_source": self.scoring_source(),
            "per_task": per_task,
        }

    def budget_rule(self) -> str:
        bound = self.sampling()["max_tokens"]
        if self.thinking():
            return "context cap - prompt tokens" + (f", at most {bound}" if bound else "")
        return f"max_tokens {bound}" if bound else "ns/RULER task budget"

    def failures_record(self, out: Path, tasks: list[str]) -> dict[str, Any]:
        """Samples scored 0 for a context reason, from the output rows (resume-safe)."""
        by_reason = dict.fromkeys(FAILURE_REASONS, 0)
        per_task = {}
        for t in tasks:
            counts: dict[str, int] = {}
            for r in self._task_rows(out, t):
                if r.get("granite_failure"):
                    counts[r["granite_failure"]] = counts.get(r["granite_failure"], 0) + 1
                    by_reason[r["granite_failure"]] = by_reason.get(r["granite_failure"], 0) + 1
            if counts:
                per_task[t] = counts
        return {"file": FAILURES_FILE, "total": sum(by_reason.values()), "by_reason": by_reason, "per_task": per_task}

    def _generated_detail(self, key: str) -> Any:
        """A value the generate phase recorded (generation.json details)."""
        path = self.config.output_dir / GENERATION_FILE
        return json.loads(path.read_text()).get("details", {}).get(key) if path.exists() else None

    def scoring_source(self) -> str:
        """What ns's RULER match reads as ``generation`` in this mode."""
        return SCORING_SOURCES[self.data_format()]

    # -- entry point ---------------------------------------------------------

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        # Before building (128k) data for a server that cannot hold it.
        if self.generating:
            max_model_len = self.check_context(base_url, served_model_name) if not self.gold else None
            cap = self.context_cap(max_model_len)
        else:
            max_model_len, cap = self._generated_detail("served_max_model_len"), self._generated_detail("context_cap")
        setup_dir, provenance = self.prepare_data()
        tasks = self.tasks()
        out = self.config.output_dir / "ruler"
        inputs = {}
        for t in tasks:
            rows = data.take(_read_jsonl(setup_dir / t / "test.jsonl"), self.config.limit, key="index")
            inputs[t] = out / t / "input.jsonl"
            if self.generating:
                _write_jsonl(inputs[t], rows)
                _reset_task_if_input_changed(out / t, self.config.output_dir / FAILURES_FILE)
            else:
                _check_input(self, out / t / "input.sha256", rows)
        n = sum(len(_read_jsonl(p)) for p in inputs.values())
        log.info("%s: %d tasks, %d samples x %d repeats", self.id, len(tasks), n, self.repeats)

        with contextlib.ExitStack() as stack:
            if self.gold and self.generating:
                all_rows = [r for p in inputs.values() for r in _read_jsonl(p)]
                base_url = stack.enter_context(RulerGoldServer(all_rows)).base_url
                served_model_name = "gold"
            for t in tasks:
                for k in range(self.repeats):
                    output = inputs[t].parent / f"output-rs{k}.jsonl"
                    if self.generating:
                        self._generate_task(setup_dir, t, inputs[t], base_url, served_model_name, k, cap)
                    if self.scoring:
                        self.check_generations(output)
                        self._evaluate(output, self.task_generation_args(setup_dir, t), what=f"{t} repeat {k} ({output})")
        if not self.scoring:
            files = [out / t / f"output-rs{k}.jsonl" for t in tasks for k in range(self.repeats)]
            return {
                "n": n,
                "ns_benchmark": f"ruler.{self.setup_name()}",
                "sample_length": self.sample_length(),
                "served_max_model_len": max_model_len,
                "context_cap": cap,
                "required_context": self.required_context(),
                "thinking": self.thinking_record(out, tasks),
                "failures": self.failures_record(out, tasks),
                "departures": self.departures(cap),
                "tasks": tasks,
                "statuses": _merge_counts(_statuses(f) for f in files),
                "data_provenance": provenance,
                "answers": "gold" if self.gold else "model",
                "sampling": self.sampling(),
                "generation_args": {t: self.task_generation_args(setup_dir, t) for t in tasks},
            }

        agg = self.aggregation()
        per_task, per_task_metrics = {}, {}
        for t in tasks:
            files = [out / t / f"output-rs{k}.jsonl" for k in range(self.repeats)]
            m = self.compute_metrics(files)
            per_task_metrics[f"{self.setup_name()}.{t}"] = m["_all_"]
            per_task[t] = {
                "accuracy": m["_all_"][agg]["accuracy"] * self.value_scale,
                "n": len(_read_jsonl(inputs[t])),
                "statuses": _merge_counts(_statuses(f) for f in files),
            }
            log.info("%s %s: accuracy=%.4f", self.id, t, per_task[t]["accuracy"])

        if set(tasks) == set(TASKS):  # ns's own group score
            from nemo_skills.dataset.ruler.ruler_score import compute_score

            group = compute_score(dict(per_task_metrics))[self.setup_name()]
        else:  # the same mean over the tasks run
            aggs = [a for a, v in next(iter(per_task_metrics.values())).items() if isinstance(v, dict) and "accuracy" in v]
            group = {a: {"accuracy": sum(m[a]["accuracy"] for m in per_task_metrics.values()) / len(tasks)} for a in aggs}
        raw = group[agg]["accuracy"]
        pk = group.get(f"pass@{self.repeats}", {}).get("accuracy")
        return {
            "value": raw * self.value_scale,
            "n": n,
            "dataset": self.dataset_source()[0],
            "dataset_revision": self.dataset_source()[1],
            "harness": {"name": "nemo-skills", "repo": NS_REPO, "commit": NS_COMMIT},
            "ns_benchmark": f"ruler.{self.setup_name()}",
            "ns_metric": {"aggregation": agg, "key": "accuracy", "raw": raw, "scale": self.value_scale, "score": "ruler_score"},
            # pass@k: per sample the best of its k (RULER's match is a fraction), then ruler_score's mean
            "pass_at_k": pass_at_k_record(self.repeats, raw * self.value_scale, _scaled(pk, self.value_scale), n, "ns metrics, ruler_score"),
            "max_seq_length": self.max_seq_length,
            "sample_length": self.sample_length(),
            "served_max_model_len": max_model_len,
            "context_cap": cap,
            "required_context": self.required_context(),
            "thinking": self.thinking_record(out, tasks),
            "failures": self.failures_record(out, tasks),
            "departures": self.departures(cap),
            "tasks": tasks,
            "partial_task_set": set(tasks) != set(TASKS),
            "per_task": per_task,
            "statuses": _merge_counts(v["statuses"] for v in per_task.values()),
            "data_provenance": provenance,
            "answers": "gold" if self.gold else "model",
            "sampling": self.sampling(),
            "generation_args": {t: self.task_generation_args(setup_dir, t) for t in tasks},
            "ns_metrics": _jsonable(per_task_metrics),
        }

    def _generate_task(self, setup_dir: Path, task: str, input_file: Path, base_url: str, served: str, k: int, cap: int) -> None:
        output = input_file.parent / f"output-rs{k}.jsonl"
        if output.exists():
            return
        spec = {
            "task": task,
            "repeat": k,
            "cap": cap,
            "answer_budget": self.answer_budget(task),
            "thinking": self.thinking(),
            "failures_file": str(self.config.output_dir / FAILURES_FILE),
        }
        spec_path = input_file.parent / f"output-rs{k}.granite.json"
        spec_path.write_text(json.dumps(spec, indent=2))
        cmd = [
            sys.executable, "-c", f"from {__name__} import _generate_main; _generate_main()", f"{SPEC_ARG}{spec_path}",
            f"++input_file={input_file}",
            f"++output_file={output}",
            "++server.server_type=vllm",
            f"++server.base_url={base_url}",
            f"++server.model={served}",
            "++server.enable_soft_fail=True",
            "++skip_filled=True",
            f"++max_concurrent_requests={self.config.workers}",
            f"++inference.random_seed={self.config.seed + k}",
            *self.task_generation_args(setup_dir, task),
        ]  # fmt: skip
        cmd = self._defer_eval(cmd, output)
        _run_ns(cmd, output, log_path=output.parent / f"output-rs{k}.log", what=f"{self.id} {task} rs{k}")
        log.info("%s %s repeat %d: generated %s", self.id, task, k, _status_line(output))


def _percentiles(sorted_toks: list[int]) -> dict[str, int]:
    if not sorted_toks:
        return {"p50": 0, "p95": 0, "max": 0}
    last = len(sorted_toks) - 1
    return {"p50": sorted_toks[round(0.5 * last)], "p95": sorted_toks[round(0.95 * last)], "max": sorted_toks[-1]}


def _ints(rows: list[dict], key: str) -> list[int]:
    return sorted(int(r[key]) for r in rows if r.get(key) is not None)


def _reset_task_if_input_changed(task_dir: Path, failures: Path | None = None) -> None:
    stamp = task_dir / "input.sha256"
    digest = _sha256(task_dir / "input.jsonl")
    if stamp.exists() and stamp.read_text().strip() != digest:
        log.warning("%s: input changed since the last run; discarding old generations", task_dir.name)
        for f in task_dir.glob("output-rs*"):
            f.unlink()
        if failures is not None and failures.exists():  # and the task's logged failures
            _write_jsonl(failures, [r for r in _read_jsonl(failures) if r.get("task") != task_dir.name])
    stamp.write_text(digest + "\n")


# -- generation with a per-sample budget ------------------------------------------


def sample_budget(spec: dict, prompt_tokens: int, bound: int | None) -> tuple[int | None, str | None]:
    """``(max_tokens, failure)`` for one sample. Thinking on: ``cap - prompt``
    (at most ``bound``), failing when that leaves less than the answer budget.
    Thinking off: ns's task budget ``bound``, failing when it does not fit."""
    room = spec["cap"] - prompt_tokens
    if spec["thinking"]:
        n = room if bound is None else min(room, bound)
        return n, ("prompt_exceeds_cap" if room < spec["answer_budget"] else None)
    return bound, ("prompt_exceeds_cap" if bound is None or room < bound else None)


_CONTEXT_ERRORS = ("context_window_exceeded", "maximum context length", "max_model_len", "context length")
"""In ns's soft-fail ``error`` / ``detailed_error`` (ns's reason, vLLM's 400 messages)."""


def classify_result(result: dict) -> str | None:
    """The context failure behind an ns result (soft-failed request or cut-off), if any."""
    if result.get("error") or result.get("finish_reason") == "error":
        text = f"{result.get('error', '')} {result.get('detailed_error', '')}".lower()
        return "context_length_error" if any(e in text for e in _CONTEXT_ERRORS) else None
    if result.get("finish_reason") == "length" and not (result.get("generation") or "").strip():
        return "length_before_answer"
    return None


def count_prompt_tokens(tokenizer, prompt: Any, chat_template_kwargs: dict | None) -> int:
    """Tokens of the prompt as vLLM sees it: the chat template with the generation
    prompt (and the request's template kwargs) for messages; the completions
    endpoint's encoding (special tokens added) for a text prompt."""
    if isinstance(prompt, str):
        return len(tokenizer.encode(prompt, add_special_tokens=True))
    ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, **(chat_template_kwargs or {}))
    return len(ids if isinstance(ids, list) else ids["input_ids"])


def capped_generation_task(base):
    """ns's GenerationTask (``base``) with RULER's per-sample budget: counts each
    prompt, generates ``sample_budget``'s max_tokens, scores a context failure 0
    (an empty generation) and appends it to the failures file at once."""
    from dataclasses import asdict, is_dataclass

    class CappedGeneration(base):
        def __init__(self, cfg, spec: dict):
            self.granite_spec = spec
            super().__init__(cfg)
            from transformers import AutoTokenizer

            self.granite_tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer or cfg.server["model"], trust_remote_code=True)

        def granite_template_kwargs(self) -> dict:
            # ns moves chat_template_kwargs into extra_body for the chat endpoint
            return dict(self.cfg.inference.extra_body.get("chat_template_kwargs") or self.cfg.chat_template_kwargs or {})

        async def process_single_datapoint(self, data_point, all_data, prompt_format=None):
            # ns's process_single_datapoint, with the sample's tokens_to_generate
            prompt = self.fill_prompt(data_point=data_point, data=all_data, prompt_format=prompt_format)
            n_prompt = count_prompt_tokens(self.granite_tokenizer, prompt, self.granite_template_kwargs())
            inference = asdict(self.cfg.inference) if is_dataclass(self.cfg.inference) else dict(self.cfg.inference)
            max_tokens, failure = sample_budget(self.granite_spec, n_prompt, inference["tokens_to_generate"])
            if failure:  # the answer cannot fit: no request
                result = {"generation": "", "num_generated_tokens": 0, "error": failure, "finish_reason": "error"}
            else:
                params = {
                    **inference,
                    **self.extra_generate_params,
                    "prompt": prompt,
                    "stop_phrases": [self.cfg.stop_phrase] if self.cfg.stop_phrase else None,
                    "tokens_to_generate": max_tokens,
                }
                result = await self.generate_with_semaphore(**params)
                failure = classify_result(result)
            result.update(granite_prompt_tokens=n_prompt, granite_max_tokens=max_tokens, granite_failure=failure)
            if failure:
                self.granite_log_failure(data_point, result)
            return result

        def granite_log_failure(self, data_point: dict, result: dict) -> None:
            spec = self.granite_spec
            rec = {
                "task": spec["task"],
                "repeat": spec["repeat"],
                "index": data_point.get("index"),
                "reason": result["granite_failure"],
                "prompt_tokens": result["granite_prompt_tokens"],
                "generated_tokens": int(result.get("num_generated_tokens") or 0),
                "cap": spec["cap"],
                "max_tokens": result["granite_max_tokens"],
                "finish_reason": result.get("finish_reason"),
                "detail": str(result.get("detailed_error") or "")[:500],
            }
            with open(spec["failures_file"], "a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()

    return CappedGeneration


def _generate_main() -> None:
    """ns generate (``nemo_skills.inference.generate``'s Hydra config and ``++``
    overrides) with ``capped_generation_task``; argv carries ``--granite-spec=<json>``."""
    import hydra
    from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig
    from nemo_skills.utils import setup_logging

    arg = next(a for a in sys.argv[1:] if a.startswith(SPEC_ARG))
    sys.argv.remove(arg)
    spec_path = Path(arg[len(SPEC_ARG) :])
    spec = json.loads(spec_path.read_text())
    # Hydra's run dir next to the outputs, not ./outputs in the job's cwd
    sys.argv += [f"hydra.run.dir={spec_path.parent}", "hydra.output_subdir=null"]
    task_cls = capped_generation_task(GenerationTask)
    setup_logging()

    @hydra.main(version_base=None, config_name="base_generation_config")
    def generate(cfg) -> None:
        cfg = GenerationTaskConfig(_init_nested=True, **cfg)
        task_cls(cfg, spec).generate()

    generate()


class RulerGoldServer(_GoldServer):
    """Answers each sample with its expected outputs. A sample is found by the
    tail of its (up to 128k-token) context, which ends the prompt text."""

    TAIL = 2000

    def __init__(self, rows: list[dict]):
        super().__init__(rows, answer=lambda r: " ".join(map(str, r["expected_answer"])))
        self.tails = [r["question"][-self.TAIL :] for r in rows]

    def lookup(self, prompt: str) -> str:
        window = prompt[-(self.TAIL + 20000) :]
        for row, tail in zip(self.rows, self.tails):
            if tail and tail in window:
                return self.answer(row)
        return ""


@register
class Ruler128k(RulerBenchmark):
    id = "ruler-128k"
    max_seq_length = 131072


@register
class Ruler64k(RulerBenchmark):
    id = "ruler-64k"
    max_seq_length = 65536


@register
class Ruler256k(RulerBenchmark):
    """Extended suite: needs a 256k-context server (granite-4.2 is verified to 128k only)."""

    id = "ruler-256k"
    max_seq_length = 262144


@register
class Ruler512k(RulerBenchmark):
    """Extended suite: needs a 512k-context server."""

    id = "ruler-512k"
    max_seq_length = 524288


@register
class Ruler1m(RulerBenchmark):
    """Extended suite: needs a 1M-context server."""

    id = "ruler-1m"
    max_seq_length = 1048576
