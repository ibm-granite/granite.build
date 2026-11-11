"""Benchmark interface and registry.

Every benchmark is one class registered under a stable id (e.g.
``swebench-verified``). The id is what granite.build step templates pass to
``granite-evals run``, and what the suite files in ``granite_evals/suites`` list.

Phases. A ``splittable`` benchmark runs as two jobs on one output dir, so no
job holds a GPU it does not use: ``--phase generate`` serves the model and
saves its outputs (the benchmark's own per-example files, which resume already
reads), ``--phase score`` serves nothing and grades them (tests, verifiers,
paid judges) on CPU. ``--phase all`` (the default) does both in one job. The
score phase never generates: a missing generation is a failed example, not a
call to a model that is not there.
"""

from __future__ import annotations

import abc
import importlib
import pkgutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

PHASES = ("all", "generate", "score")

# Every module in granite_evals.benchmarks is imported to register its classes.
# Modules import their harness inside methods, never at the top, so
# `granite-evals list` works without any benchmark extra installed.

_REGISTRY: dict[str, type[Benchmark]] = {}


@dataclass
class RunConfig:
    """Options shared by every benchmark, set from the CLI / step config."""

    model: str
    """Model path or HF id served by vLLM; also the default served name."""
    output_dir: Path
    base_url: str = ""
    """OpenAI-compatible endpoint. Empty means granite-evals starts vLLM itself."""
    served_model_name: str = ""
    limit: int | None = None
    """Evaluate only the first N examples (smoke mode). None = full dataset."""
    repeats: int | None = None
    """Independent samples per example (k). None = the benchmark default, 1 for
    every benchmark (pass@1 on one sample). With k > 1 the headline is
    pass@1[avg-of-k] and ``details.pass_at_k`` adds pass@k (``pass_at_k``)."""
    workers: int = 8
    seed: int = 0
    dataset: str = ""
    """Override the dataset hub id or local path (default: the benchmark's pin)."""
    dataset_revision: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    """Benchmark-specific key=value options (``--option k=v``)."""
    phase: str = "all"
    """``all``, ``generate`` or ``score`` (see the module docstring)."""


class Benchmark(abc.ABC):
    """One Granite benchmark: fetch data, run the model, score, report."""

    id: ClassVar[str]
    metric: ClassVar[str]
    """Human-readable headline metric, as named in the suite sheet."""
    default_repeats: ClassVar[int] = 1
    extra: ClassVar[str] = ""
    """The pyproject extra this benchmark needs (for error messages)."""
    harness_packages: ClassVar[tuple[str, ...]] = ()
    """Distributions whose versions results.json records (the pinned harness)."""
    dataset: ClassVar[str] = ""
    """Upstream HF dataset id, read directly (public data is not mirrored)."""
    dataset_revision: ClassVar[str | None] = None
    """Commit of ``dataset`` the score is defined on."""
    splittable: ClassVar[bool] = False
    """Whether ``--phase generate`` / ``--phase score`` are supported."""

    def __init__(self, config: RunConfig):
        self.config = config
        self.repeats = config.repeats or self.default_repeats

    def dataset_source(self) -> tuple[str, str | None]:
        """(dataset, revision): the CLI override, else the benchmark's pin. An
        overridden dataset drops the pin unless a revision is given too."""
        if self.config.dataset:
            return self.config.dataset, self.config.dataset_revision
        return self.dataset, self.config.dataset_revision or self.dataset_revision

    @abc.abstractmethod
    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        """Run the benchmark against an OpenAI-compatible endpoint.

        Returns a dict with at least ``value`` (headline metric) and ``n``
        (examples scored). Anything else is copied into results.json. In the
        generate phase ``n`` counts the examples generated and ``value`` is
        None (or left out).
        """

    def needs_server(self) -> bool:
        """Whether generating talks to the served model at all (False for the
        gold/oracle modes)."""
        return True

    def score_needs_server(self) -> bool:
        """Whether scoring talks to the served model (e.g. ``judge_model=self``).
        The score phase serves nothing, so such a run must use ``--phase all``."""
        return False

    @property
    def phase(self) -> str:
        return self.config.phase

    @property
    def generating(self) -> bool:
        return self.config.phase in ("all", "generate")

    @property
    def scoring(self) -> bool:
        return self.config.phase in ("all", "score")

    def serves(self) -> bool:
        """Whether this run's phase needs the served model."""
        if self.phase == "generate":
            return self.needs_server()
        if self.phase == "score":
            return False
        return self.needs_server() or self.score_needs_server()

    def not_generated(self, what: str) -> RuntimeError:
        """The error for an example the score phase finds without its generation."""
        return RuntimeError(f"{self.id}: no generation for {what} (run --phase generate on this output dir)")


def failure_policy(
    benchmark_id: str, what: str, failed: int, total: int, max_frac: float, *, failed_key: str = ""
) -> dict[str, Any]:
    """The one rule for judged and infra-dependent benchmarks.

    A failed item (judge error, unparseable judgement, infrastructure error,
    402 budget exhaustion) is never scored as a model loss: the caller leaves
    it out of the score, as upstream does, and resume retries it. This
    reports ``<what>_failed`` (or ``failed_key``) and ``<what>_total``, flags
    ``incomplete`` when any failed, and exits with an error when the failed
    fraction exceeds ``max_frac`` (0 = any failure is fatal).
    """
    key = failed_key or f"{what}_failed"
    if failed and (not total or failed / total > max_frac):
        raise SystemExit(
            f"{benchmark_id}: {key} {failed}/{total} is above the allowed fraction {max_frac}; "
            "rerun to retry the failed ones"
        )
    return {key: failed, f"{what}_total": total, "incomplete": failed > 0}


def pass_at_k_record(k: int, pass_at_1: float | None, pass_at_k: float | None, n: int, how: str = "") -> dict[str, Any]:
    """results.json ``details.pass_at_k``, one shape for every benchmark:
    ``pass_at_1`` is the mean over examples of the mean over its k samples
    (pass@1[avg-of-k], the headline unless the benchmark says otherwise),
    ``pass_at_k`` the fraction of examples with a correct sample among the k
    (any-correct; for a graded score, the best of the k). ``pass_at_k`` is None
    where it has no meaning, and ``how`` says why or how it was computed."""
    rec: dict[str, Any] = {"k": k, "pass_at_1": pass_at_1, "pass_at_k": pass_at_k, "n": n}
    if how:
        rec["how"] = how
    return rec


def pass_at_k(scores: Iterable[Sequence[float]], k: int, how: str = "") -> dict[str, Any]:
    """``pass_at_k_record`` from each example's scores over its samples (1 =
    correct; bool or a fraction). An example with no sample counts 0 in both."""
    rows = [[float(x) for x in s] for s in scores]
    n = len(rows)
    if not n:
        return pass_at_k_record(k, None, None, 0, how)
    p1 = sum(sum(s) / len(s) for s in rows if s) / n
    pk = sum(max(s) for s in rows if s) / n
    return pass_at_k_record(k, p1, pk, n, how)


def register(cls: type[Benchmark]) -> type[Benchmark]:
    if cls.id in _REGISTRY:
        raise ValueError(f"duplicate benchmark id {cls.id!r}")
    _REGISTRY[cls.id] = cls
    return cls


def _load_all() -> None:
    from granite_evals import benchmarks

    for m in pkgutil.iter_modules(benchmarks.__path__):
        importlib.import_module(f"{benchmarks.__name__}.{m.name}")


def get(benchmark_id: str) -> type[Benchmark]:
    _load_all()
    try:
        return _REGISTRY[benchmark_id]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise SystemExit(f"unknown benchmark {benchmark_id!r}; implemented: {known}") from None


def all_benchmarks() -> dict[str, type[Benchmark]]:
    _load_all()
    return dict(sorted(_REGISTRY.items()))
