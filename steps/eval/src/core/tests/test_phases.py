"""The generate / score split (registry, cli, results): no GPU, no harness."""

import json

import pytest

from granite_evals import cli, registry
from granite_evals.registry import Benchmark, RunConfig


class _Split(Benchmark):
    """Generates one file per example; scores those files; never generates in score."""

    id = "split-dummy"
    metric = "acc"
    splittable = True
    calls: list = []

    def run(self, base_url, served):
        type(self).calls.append((self.phase, base_url, served))
        root = self.config.output_dir / "gen"
        root.mkdir(exist_ok=True)
        n = self.config.limit or 3
        if self.generating:
            for i in range(n):
                (root / f"{i}.txt").write_text("right" if i else "wrong")
        if not self.scoring:
            return {"n": n}
        scores = []
        for i in range(n):
            f = root / f"{i}.txt"
            if not f.exists():
                raise self.not_generated(f"example {i}")
            scores.append(f.read_text() == "right")
        return {"value": sum(scores) / n, "n": n}


class _Inline(_Split):
    id = "inline-dummy"
    splittable = False


@pytest.fixture(autouse=True)
def _registered(monkeypatch):
    monkeypatch.setitem(registry._REGISTRY, _Split.id, _Split)
    monkeypatch.setitem(registry._REGISTRY, _Inline.id, _Inline)
    monkeypatch.setattr(registry, "_load_all", lambda: None)
    _Split.calls = []


def _run(out, *extra):
    base = ["run", "split-dummy", "--model", "/m/granite", "--output-dir", str(out), "--base-url", "http://x", "--limit", "2"]
    return cli.main([*base, *extra])


def test_generate_then_score(tmp_path, capsys):
    assert _run(tmp_path, "--phase", "generate") == 0
    gen = json.loads((tmp_path / "generation.json").read_text())
    assert gen["phase"] == "generate" and gen["value"] is None and gen["n"] == 2
    assert not (tmp_path / "results.json").exists()
    # The score job serves nothing (no --base-url) and names the model as generated.
    argv = ["run", "split-dummy", "--model", "/m/granite", "--output-dir", str(tmp_path), "--limit", "2", "--phase", "score"]
    assert cli.main(argv) == 0
    res = json.loads((tmp_path / "results.json").read_text())
    assert res["phase"] == "score" and res["value"] == 0.5
    assert res["generation"]["served_model_name"] == "granite"
    assert _Split.calls[-1] == ("score", "", "granite")


def test_score_needs_a_matching_generation(tmp_path):
    with pytest.raises(SystemExit, match="run --phase generate"):
        _run(tmp_path, "--phase", "score")
    _run(tmp_path, "--phase", "generate")
    with pytest.raises(SystemExit, match="limit: generated 2, scoring 3"):
        cli.main(["run", "split-dummy", "--model", "/m/granite", "--output-dir", str(tmp_path),
                  "--limit", "3", "--phase", "score"])  # fmt: skip


def test_score_warns_on_other_options(tmp_path, caplog):
    _run(tmp_path, "--phase", "generate", "--option", "agent=gold")
    _run(tmp_path, "--phase", "score", "--option", "timeout=9")
    assert "agent 'gold' -> None, timeout None -> '9'" in caplog.text


def test_score_never_generates(tmp_path):
    _run(tmp_path, "--phase", "generate")
    (tmp_path / "gen" / "1.txt").unlink()
    with pytest.raises(RuntimeError, match="no generation for example 1"):
        _run(tmp_path, "--phase", "score")


def test_unsplittable_benchmark_refuses_a_phase(tmp_path):
    with pytest.raises(SystemExit, match="--phase all"):
        cli.main(["run", "inline-dummy", "--model", "m", "--output-dir", str(tmp_path), "--phase", "generate"])


def test_serves_by_phase(tmp_path):
    class SelfJudged(_Split):
        def needs_server(self):
            return False  # a gold mode

        def score_needs_server(self):
            return True  # judge_model=self

    def b(phase):
        return SelfJudged(RunConfig(model="m", output_dir=tmp_path, phase=phase))

    assert [b(p).serves() for p in ("all", "generate", "score")] == [True, False, False]
    with pytest.raises(SystemExit, match="score with the served model"):
        registry._REGISTRY["split-dummy"] = SelfJudged
        cli.main(["run", "split-dummy", "--model", "m", "--output-dir", str(tmp_path), "--phase", "score"])
