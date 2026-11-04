"""Extended NeMo-Skills benchmarks: HLE, AA-Omniscience, CritPt, AA-LCR, WMT24++.

All run through the shared ``NemoSkillsBenchmark`` (``nemo_skills.py``): ns's own
prepare, prompts, generation loop, judges/evaluators and metrics at the pinned ns
commit, against the step's vLLM. Sampling is the checkpoint's (no temperature,
top_p or max_tokens sent; thinking on unless the chat template says otherwise).
Default 1 repeat (``pass@1``); ``--repeats k`` reports ``pass@1[avg-of-k]`` as the
value and ns's ``pass@k`` under ``details.pass_at_k``.

What is common here:

* Only post-thinking text is scored. vLLM's reasoning parser puts the thinking in
  ``reasoning_content``; a ``<think>`` or ``</think>`` left in a scored
  ``generation`` (no reasoning parser) fails the run, as for RULER.
* LLM judges (hle, omniscience, aa-lcr) are ns's judge prompts, temperature 0,
  through the spend meter, all currently using the shared judge default
  (``aws/claude-sonnet-5`` on the configured LiteLLM gateway), as for arena-hard-v2.
  hle and omniscience each have a documented confirmed-official judge that
  differs from this shared default (``official_judge``); a per-class runtime
  override mechanism exists (``judge_model_default`` / ``judge_base_url_default``
  on the class, see ``nemo_skills.judge_endpoint``) but switching either
  benchmark's actual runtime judge to its official one is pending team
  sign-off and is not yet active. A judgement ns cannot parse is left out of
  the score (``registry.failure_policy``: counted in ``judge_invalid``,
  re-judged on resume, fatal above ``max_judge_invalid_frac``), where ns would
  score it wrong.
* ``--option answers=gold`` (where reference answers exist) serves them through
  the same ns generate path and then the *real* judge (or COMET), so it checks
  the whole pipeline end to end (~100% expected).

HLE (``hle``)
    ``cais/hle`` (gated: HF_TOKEN), ns's ``text`` split (the questions without an
    image), ns's HLE answer-format prompt, ns's HLE judge prompt; ns metric
    ``judge_correct``. Judge: the shared default (``aws/claude-sonnet-5``) for
    now; the model card's official judge is Azure/gpt-4o-ncf, not
    activated by default (the HLE leaderboard's own
    default is o3-mini-2025-01-31; see ``official_judge``).

AA-Omniscience (``omniscience``, ``omniscience-hallucination``)
    ``ArtificialAnalysis/AA-Omniscience-Public`` (600 questions), ns's ``text``
    split and AA's prompt; the judge grades A correct / B incorrect / C partial /
    D not attempted. ``omniscience`` reports ``judge_correct``;
    ``omniscience-hallucination`` reports ``judge_omni_hallucination`` = incorrect /
    (all not correct), lower is better. ns's ``++parse_reasoning=True`` is turned
    off: the server already strips the thinking, and ns would blank every answer
    without a ``</think>``. Judge: the shared default for now; Nemo-Gym's
    official judge is gpt-oss-20b via NVIDIA's API (needs ``NVIDIA_API_KEY``),
    pending team sign-off to activate. 8 repeats by default (Nemo-Gym). ``omniscience-hallucination``
    with ``--option generations_from=<omniscience output dir>`` scores that run's
    generations and judgements (copied in, nothing regenerated or re-judged unless
    invalid); without it, it is a full run of its own.

CritPt (``critpt``)
    ``CritPt-Benchmark/CritPt`` (70 problems), ns's two-turn generation (solve, then
    fill the code template). Grading is external: ns posts all 70 answers to
    Artificial Analysis's CritPt API (``x-api-key`` from ``ARTIFICIAL_ANALYSIS_API_KEY``,
    or the env named by ``critpt_api_key_env``), which returns only the aggregate
    accuracy (no per-problem results, no public answers: no gold mode). The API
    grades exactly 70 submissions, so ``--limit`` works for ``--phase generate``
    only. Generation never calls the API; scoring does (once per repeat, the
    response cached next to the file). The API is not an LLM bill and is not
    routed through the spend meter. Its ``judge_error_count`` /
    ``server_timeout_count`` are reported, and a judge error flags ``incomplete``.

AA-LCR (``aa-lcr``)
    ``ArtificialAnalysis/AA-LCR`` (100 questions over ~100k-token document sets; the
    extracted texts zip at the same pin), ns's ``generic/default`` prompt with the
    documents inline, ns's AA-LCR judge (CORRECT / INCORRECT); ``judge_correct``.
    Prompts reach ~130k tokens (cl100k), so the server needs a max model length of
    prompt plus thinking (e.g. 160k+). Official judge: Qwen3-235B-A22B-Instruct-2507
    (AA); ns uses gpt-4.1.

WMT24++ (``wmt24pp``)
    ``google/wmt24pp`` en->xx, ns's default languages de_DE es_MX fr_FR it_IT ja_JP
    (998 segments each; ``--option languages=de_DE,ja_JP`` to choose), ns's
    segment-translation prompt. The value is ns's ``en->xx`` ``comet``: XCOMET-XXL
    (``Unbabel/XCOMET-XXL``, gated, CC-BY-NC-SA-4.0, ~10.7B parameters: score on
    a GPU, ~22 GB in bf16) as ns runs it (``evaluator/comet.py``, bf16, batch 16, all
    visible GPUs), in the image's separate COMET env (``/opt/comet``: unbabel-comet
    needs transformers<5, the job venv has vLLM's transformers 5). The model and
    its encoder's config/tokenizer (``facebook/xlm-roberta-xxl``) are downloaded at
    pinned revisions (``comet_model`` / ``comet_encoder`` = ``repo@rev`` or a local
    dir). ns's ``bleu`` (sacrebleu, ja-mecab for Japanese) is always reported too;
    ``--option score_metric=bleu`` makes it the value and skips COMET (no GPU).
    Values are fractions (ns's COMET x 100 and BLEU, both x 0.01).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from granite_evals.benchmarks.nemo_skills import (
    NemoSkillsBenchmark,
    _GoldServer,
    _read_jsonl,
    _write_jsonl,
    log,
)
from granite_evals.registry import register

HLE_DATASET = "cais/hle"
HLE_REVISION = "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29"
OMNI_DATASET = "ArtificialAnalysis/AA-Omniscience-Public"
OMNI_REVISION = "e4883edbb9f5ccf2b2a8fdc6fb65e01a58e99849"
CRITPT_DATASET = "CritPt-Benchmark/CritPt"
CRITPT_REVISION = "9b9fc8498596ec08ab5437a72f4aa18beef2b876"
CRITPT_PROBLEMS = 70
CRITPT_API_KEY_ENV = "ARTIFICIAL_ANALYSIS_API_KEY"
LCR_DATASET = "ArtificialAnalysis/AA-LCR"
LCR_REVISION = "9a77ef56b717057ade24ceab4d273712a0b4f19e"
"""v1.1 (16 corrected answer keys)."""
WMT_DATASET = "google/wmt24pp"
WMT_REVISION = "fd7405c06494bc66a57b25f55d217a72f96e60dc"
WMT_LANGUAGES = ("de_DE", "es_MX", "fr_FR", "it_IT", "ja_JP")
"""ns's prepare default."""

COMET_PYTHON = "/opt/comet/bin/python"
"""The image's COMET env (docker/extras/nemoskills.sh)."""
COMET_MODEL = "Unbabel/XCOMET-XXL@873bac1b1c461e410c4a6e379f6790d3d1c7c214"
COMET_ENCODER = "facebook/xlm-roberta-xxl@03e0fb540c3c9afd4bdda0072e7cb82d2eafd060"
COMET_MODEL_FILES = ("hparams.yaml", "checkpoints/model.ckpt")
COMET_ENCODER_FILES = (
    "config.json",
    "sentencepiece.bpe.model",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
"""The encoder's config and tokenizer only: the weights are the COMET checkpoint's."""

THINK_TAGS = ("<think>", "</think>")

NVIDIA_JUDGE_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_JUDGE_MODEL = "openai/gpt-oss-20b"
NVIDIA_JUDGE_API_KEY_ENV = "NVIDIA_API_KEY"
"""Nemo-Gym's official Omniscience judge: gpt-oss-20b via NVIDIA's API, a
different gateway than the shared JUDGE_BASE_URL."""


class _Extended(NemoSkillsBenchmark):
    """The shared extended-benchmark behaviour (module doc)."""

    think_fields: tuple[str, ...] = ("generation",)

    def _evaluate(self, output: Path, args: list[str], what: str) -> None:
        # Every scored file passes here first, in every phase that scores.
        if output.exists():
            _check_no_think(self.id, output, self.think_fields)
        super()._evaluate(output, args, what)


def _check_no_think(benchmark_id: str, path: Path, fields: tuple[str, ...]) -> None:
    rows = _read_jsonl(path)
    bad = [i for i, r in enumerate(rows) if any(t in str(r.get(f) or "") for f in fields for t in THINK_TAGS)]
    if bad:
        raise SystemExit(
            f"{benchmark_id}: {len(bad)}/{len(rows)} generations in {path} contain a <think>/</think> tag "
            f"(first: row {bad[0]}); only post-thinking text is scored, so serve the model with a "
            "reasoning parser (vLLM --reasoning-parser)"
        )


class _Judged(_Extended):
    """An ns LLM-judge benchmark with the shared Granite-evals judge (module doc)."""

    gold_judged = True

    def judge_valid(self, judgement: str) -> bool:
        """Whether ns can read a verdict from this judgement."""
        raise NotImplementedError

    def judge_failures(self, row: dict) -> tuple[int, int]:
        return (0 if self.judge_valid(str(row.get("judgement") or "")) else 1), 1

    def compute_metrics(self, files: list[Path]) -> dict[str, Any]:
        """ns's metrics without the rows whose judgement is invalid (in any repeat:
        every repeat must score the same examples), per ``failure_policy``."""
        rows = [_read_jsonl(f) for f in files]
        bad = {i for rs in rows for i, r in enumerate(rs) if self.judge_failures(r)[0]}
        if not bad:
            return super().compute_metrics(files)
        with tempfile.TemporaryDirectory() as d:
            kept = []
            for f, rs in zip(files, rows):
                kept.append(Path(d) / f.name)
                _write_jsonl(kept[-1], [r for i, r in enumerate(rs) if i not in bad])
            metrics = super().compute_metrics(kept)
        metrics["judge_invalid_left_out"] = len(bad)
        return metrics


# -- HLE -------------------------------------------------------------------------


@register
class HLE(_Judged):
    id = "hle"
    metric = "pass@1 judge_correct"
    ns_benchmark = "hle"
    ns_split = "text"
    ns_prepare_args = ("--split", "text")
    ns_metric = "judge_correct"
    dataset = HLE_DATASET  # gated: needs HF_TOKEN
    dataset_revision = HLE_REVISION
    official_judge = "o3-mini-2025-01-31 (HLE leaderboard, NeMo-Skills default); Azure/gpt-4o-ncf (the model card's official judge, )"
    # judge_model/judge_base_url overrides pending team sign-off on switching
    # the actual runtime judge to Azure/gpt-4o-ncf; shared Sonnet-5 default
    # stays in effect for now.

    def judge_valid(self, judgement: str) -> bool:
        from nemo_skills.evaluation.metrics.utils import is_correct_judgement

        return is_correct_judgement(judgement, return_none=True) is not None

    def gold_generation(self, row: dict) -> str:
        return f"Explanation: this is the reference answer.\nAnswer: {row[self.gold_answer_key]}\nConfidence: 100%"


# -- AA-Omniscience ------------------------------------------------------------------


@register
class Omniscience(_Judged):
    id = "omniscience"
    metric = "pass@1 judge_correct"
    ns_benchmark = "omniscience"
    ns_split = "text"
    ns_prepare_args = ("--splits", "text")
    ns_metric = "judge_correct"
    default_repeats = 8
    """Nemo-Gym's official repeat count."""
    # The server strips the thinking; ns's parse_reasoning would blank every
    # generation without a "</think>".
    ns_generation_args = ("++parse_reasoning=False",)
    dataset = OMNI_DATASET
    dataset_revision = OMNI_REVISION
    official_judge = "gpt-oss-20b via NVIDIA API (Nemo-Gym's official judge)"
    # judge_model/judge_base_url/judge_api_key_env overrides pending team
    # sign-off on provisioning NVIDIA_API_KEY and switching the runtime
    # judge; shared Sonnet-5 default stays in effect for now.

    def judge_valid(self, judgement: str) -> bool:
        # ns's OmniMetrics: the stripped judgement is exactly one of a/b/c/d.
        return judgement.strip().lower() in ("a", "b", "c", "d")

    def gold_generation(self, row: dict) -> str:
        return str(row[self.gold_answer_key])


@register
class OmniscienceHallucination(Omniscience):
    """incorrect / (incorrect + partial + not attempted): lower is better."""

    id = "omniscience-hallucination"
    metric = "pass@1 judge_omni_hallucination"
    ns_metric = "judge_omni_hallucination"

    def reuse_dir(self) -> Path | None:
        src = self.opt("generations_from", "")
        return Path(src) if src else None

    @property
    def generating(self) -> bool:
        return super().generating and self.reuse_dir() is None

    def needs_server(self) -> bool:
        return super().needs_server() and self.reuse_dir() is None

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        src = self.reuse_dir()
        if src is None:
            return super().run(base_url, served_model_name)
        if self.phase == "generate":
            raise SystemExit(f"{self.id}: generations_from={src} has nothing to generate; use --phase score or all")
        self._adopt(src)
        result = super().run(base_url, served_model_name)
        result["generations_from"] = str(src)
        return result

    def _adopt(self, src: Path) -> None:
        """Copy an omniscience run's input stamp, generations and judgements in (the
        input check then guarantees the same examples and options). Files already
        here are kept, unless they belong to another input."""
        stamp = src / "input.sha256"
        if not stamp.exists():
            raise SystemExit(f"{self.id}: {src} holds no omniscience run (no input.sha256)")
        out = self.config.output_dir
        out.mkdir(parents=True, exist_ok=True)
        mine = out / "input.sha256"
        if mine.exists() and mine.read_text() != stamp.read_text():
            log.warning("%s: %s has another input than this dir; discarding this dir's outputs", self.id, src)
            for d in ("generation", "judged", "judge"):
                shutil.rmtree(out / d, ignore_errors=True)
        shutil.copyfile(stamp, mine)
        copied = 0
        for d in ("generation", "judged"):
            if not (src / d).is_dir():
                continue
            (out / d).mkdir(parents=True, exist_ok=True)
            for f in sorted((src / d).iterdir()):
                if f.is_file() and not (out / d / f.name).exists():
                    shutil.copy2(f, out / d / f.name)
                    copied += 1
        log.info("%s: scoring the omniscience run in %s (%d files copied)", self.id, src, copied)


# -- CritPt --------------------------------------------------------------------------


@register
class CritPt(_Extended):
    id = "critpt"
    metric = "Challenge Accuracy"
    ns_benchmark = "critpt"
    ns_metric = "accuracy"
    dataset = CRITPT_DATASET
    dataset_revision = CRITPT_REVISION
    think_fields = ("generation", "intermediate")  # turn 2's answer, and turn 1's (sent back in turn 2)
    # ns's critpt generation, but with a nullable temperature (see ns_critpt)
    ns_generation_module = "granite_evals.ns_critpt"

    def api_key_env(self) -> str:
        return self.opt("critpt_api_key_env", CRITPT_API_KEY_ENV)

    def generation_args(self, sandbox_args: list[str]) -> list[str]:
        args = [*super().generation_args(sandbox_args), f"++eval_config.api_key_env={self.api_key_env()}"]
        if self.opt("critpt_api_url", ""):
            args.append(f"++eval_config.api_url={self.opt('critpt_api_url', '')}")
        return args

    def _defer_eval(self, cmd: list[str], output: Path) -> list[str]:
        """Generation never calls the grading API: every file is graded by the
        scoring step (``_evaluate``), in --phase all too."""
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_name(output.name + ".unscored").touch()
        return [*cmd, "++eval_type=null"]

    def load_rows(self, path: Path) -> list[dict]:
        rows = super().load_rows(path)
        if self.scoring:  # checked before generating, so --phase all fails early
            if len(rows) != CRITPT_PROBLEMS:
                raise SystemExit(
                    f"{self.id}: the CritPt API grades exactly {CRITPT_PROBLEMS} problems, this run has "
                    f"{len(rows)} (--limit): run --phase generate only, or no --limit"
                )
            if not os.environ.get(self.api_key_env()):
                raise SystemExit(f"{self.id}: grading needs an Artificial Analysis API key in {self.api_key_env()}")
        return rows

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        if self.gold:
            raise SystemExit(f"{self.id}: no public reference answers (graded by AA's API), no answers=gold")
        result = super().run(base_url, served_model_name)
        if result.get("value") is None:
            return result
        api = []
        for k in range(self.repeats):
            resp = self.config.output_dir / "generation" / f"output-rs{k}_critpt_response.json"
            r = json.loads(resp.read_text()) if resp.exists() else {}
            api.append({"repeat": k, **{key: r.get(key) for key in (
                "accuracy", "timeout_rate", "server_timeout_count", "judge_error_count", "status_code")}})  # fmt: skip
        errors = sum(int(a.get("judge_error_count") or 0) for a in api)
        if errors:
            log.warning("%s: the CritPt API reports %d judge errors (scored as it returned them)", self.id, errors)
        result.update(
            critpt_api=api,
            critpt_api_url=self.opt("critpt_api_url", "")
            or "ns default (artificialanalysis.ai/api/v2/critpt/evaluate)",
            grader="Artificial Analysis CritPt API (aggregate accuracy only)",
            incomplete=bool(errors) or result.get("incomplete", False),
        )
        return result


# -- AA-LCR -------------------------------------------------------------------------


@register
class AALCR(_Judged):
    id = "aa-lcr"
    metric = "pass@1 judge correct"
    default_repeats = 16
    """Nemo-Gym's official repeat count."""
    ns_benchmark = "aalcr"
    ns_metric = "judge_correct"
    dataset = LCR_DATASET
    dataset_revision = LCR_REVISION
    official_judge = "Qwen3-235B-A22B-Instruct-2507, non-reasoning (AA-LCR); gpt-4.1 (NeMo-Skills default)"

    def judge_valid(self, judgement: str) -> bool:
        # ns's AALCRMetrics reads CORRECT* as correct and anything else as wrong.
        return judgement.strip().upper().startswith(("CORRECT", "INCORRECT"))

    def gold_generation(self, row: dict) -> str:
        return str(row[self.gold_answer_key])


# -- WMT24++ ------------------------------------------------------------------------


@register
class WMT24pp(_Extended):
    id = "wmt24pp"
    metric = "en→xx comet"
    ns_benchmark = "wmt24pp"
    ns_aggregation = "en->xx"
    dataset = WMT_DATASET
    dataset_revision = WMT_REVISION
    harness_packages = (*NemoSkillsBenchmark.harness_packages, "sacrebleu")
    gold_judged = True  # gold translations go through COMET too

    def score_metric(self) -> str:
        m = self.opt("score_metric", "comet")
        if m not in ("comet", "bleu"):
            raise SystemExit(f"{self.id}: score_metric must be comet or bleu, not {m!r}")
        return m

    @property
    def ns_metric(self) -> str:  # type: ignore[override]
        return self.score_metric()

    def languages(self) -> tuple[str, ...]:
        langs = tuple(x for x in str(self.opt("languages", ",".join(WMT_LANGUAGES))).replace(" ", ",").split(",") if x)
        bad = [x for x in langs if not re.fullmatch(r"[a-z]{2,3}_[A-Z][A-Za-z]{1,3}", x)]
        if bad or not langs:
            raise SystemExit(f"{self.id}: languages are wmt24pp configs like de_DE,ja_JP; got {bad or langs}")
        return langs

    @property
    def ns_prepare_args(self) -> tuple[str, ...]:  # type: ignore[override]
        return ("--target_languages", *self.languages())

    def uses_judge(self) -> bool:
        """COMET is the "judge" step (scores into ``judged/``)."""
        return self.score_metric() == "comet"

    def compute_metrics(self, files: list[Path]) -> dict[str, Any]:
        """ns's TranslationMetrics; its run-time ``pip install sacrebleu[ja]`` is a
        no-op (installed by the extra). It has no ``pass@1`` entry: for a single
        file that is its ``en->xx``."""
        from nemo_skills.evaluation.metrics import translation_metrics

        install = translation_metrics.install_packages
        translation_metrics.install_packages = lambda lang: None
        try:
            metrics = super().compute_metrics(files)
        finally:
            translation_metrics.install_packages = install
        if len(files) == 1 and "en->xx" in metrics.get("_all_", {}):
            metrics["_all_"].setdefault("pass@1", metrics["_all_"]["en->xx"])
        return metrics

    def _generate(self, input_file: Path, output: Path, base_url: str, served: str, k: int, sandbox_args) -> None:
        if not self.gold:
            return super()._generate(input_file, output, base_url, served, k, sandbox_args)
        with WMTGoldServer(_read_jsonl(input_file)) as gold:
            return super()._generate(input_file, output, gold.base_url, "gold", k, sandbox_args)

    # -- COMET ---------------------------------------------------------------------

    def comet_python(self) -> Path:
        python = Path(self.opt("comet_python", COMET_PYTHON))
        if not python.exists():
            raise SystemExit(f"{self.id}: no COMET env at {python} (docker/extras/nemoskills.sh builds it)")
        return python

    def comet_model_dir(self) -> tuple[Path, dict[str, Any]]:
        """The COMET model as ``load_from_checkpoint`` reads it: ``<dir>/hparams.yaml``
        (``pretrained_model`` = the local encoder snapshot) and
        ``<dir>/checkpoints/model.ckpt`` (a link to the pinned checkpoint)."""
        model_dir, model_ref = _hub_dir(self.opt("comet_model", COMET_MODEL), COMET_MODEL_FILES)
        enc_dir, enc_ref = _hub_dir(self.opt("comet_encoder", COMET_ENCODER), COMET_ENCODER_FILES)
        hparams, n = re.subn(
            r"(?m)^pretrained_model:.*$", f"pretrained_model: {enc_dir}", (model_dir / "hparams.yaml").read_text()
        )
        if n != 1:
            raise RuntimeError(f"{self.id}: {model_dir}/hparams.yaml has {n} pretrained_model lines, expected 1")
        local = self.config.output_dir / "comet" / "model"
        (local / "checkpoints").mkdir(parents=True, exist_ok=True)
        (local / "hparams.yaml").write_text(hparams)
        link = local / "checkpoints" / "model.ckpt"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to((model_dir / "checkpoints" / "model.ckpt").resolve())
        return local, {"model": model_ref, "encoder": enc_ref}

    def _judge_all(self, gen_dir: Path, judged_dir: Path, base_url: str, served: str) -> dict[str, Any]:
        """ns's COMET scoring of every repeat (``judged/output-rs<k>.jsonl``, done
        when its ``.done`` marker is), in the COMET env."""
        import nemo_skills

        from granite_evals import comet_score

        precision = self.opt("comet_precision", "bf16")
        batch = self.opt("comet_batch_size", 16)
        names = [f"output-rs{k}.jsonl" for k in range(self.repeats)]
        todo = [n for n in names if not (judged_dir / (n + ".done")).exists()]
        model_dir, refs = (None, {}) if not todo else self.comet_model_dir()
        if todo:
            work = self.config.output_dir / "comet"
            work.mkdir(parents=True, exist_ok=True)
            spec = {
                "ns_comet": str(Path(nemo_skills.__file__).parent / "evaluation" / "evaluator" / "comet.py"),
                "checkpoint": str(model_dir / "checkpoints" / "model.ckpt"),
                "precision": precision,
                "batch_size": batch,
                "files": [[str(gen_dir / n), str(judged_dir / n)] for n in todo],
            }
            (work / "spec.json").write_text(json.dumps(spec, indent=2))
            (work / "refs.json").write_text(json.dumps(refs, indent=2))
            python = self.comet_python()
            cmd = [str(python), comet_score.__file__, str(work / "spec.json")]
            log.info("%s: COMET (%s, %s) on %d file(s)", self.id, refs.get("model"), precision, len(todo))
            self._run_comet(cmd, work, _comet_env(python))
            missing = [n for n in todo if not (judged_dir / (n + ".done")).exists()]
            if missing:
                raise RuntimeError(f"{self.id}: COMET did not score {missing}; see {work / 'comet.log'}")
        refs_file = self.config.output_dir / "comet" / "refs.json"
        return {
            "comet": {
                **(json.loads(refs_file.read_text()) if refs_file.exists() else refs),
                "precision": precision,
                "batch_size": batch,
                "scorer": "nemo_skills/evaluation/evaluator/comet.py (process_file), via granite_evals/comet_score.py",
            },
            "languages": list(self.languages()),
        }

    def _run_comet(self, cmd: list[str], work: Path, env: dict[str, str]) -> None:
        with (work / "comet.log").open("ab") as logf:
            r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=work)
        if r.returncode:
            raise RuntimeError(f"{self.id}: COMET exited {r.returncode}; see {work / 'comet.log'}")


def _hub_dir(ref: str, files: tuple[str, ...]) -> tuple[Path, str]:
    """A local dir (as is) or ``repo@revision`` downloaded from the HF hub at that
    revision (only ``files``). The revision is required: no unpinned model."""
    if Path(ref).is_dir():
        return Path(ref), ref
    repo, _, rev = ref.partition("@")
    if not rev:
        raise SystemExit(f"COMET model {ref!r}: give repo@revision (or a local dir)")
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=rev, allow_patterns=list(files))), ref


def _comet_env(python: Path) -> dict[str, str]:
    """The COMET env's own packages only, offline (everything it reads is local)."""
    env = dict(os.environ, PATH=f"{python.parent}:{os.environ.get('PATH', '')}", PYTHONNOUSERSITE="1")
    for k in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
        env.pop(k, None)
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


class WMTGoldServer(_GoldServer):
    """Answers each translation prompt with its row's reference: the row whose
    source is in the prompt and whose target language the prompt names (one
    source appears once per language)."""

    def __init__(self, rows: list[dict]):
        super().__init__(rows, answer=lambda r: r["reference"])

    def lookup(self, prompt: str) -> str:
        hits = [
            r
            for r in self.rows
            if r["source"].strip() and r["source"].strip() in prompt and f"into {r['target_lang_name']}," in prompt
        ]
        best = max(hits, key=lambda r: len(r["source"]), default=None)
        return "" if best is None else best["reference"]
