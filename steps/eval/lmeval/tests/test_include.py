"""include family tests: fakes only (no GPU, no network, no server)."""

import json
from types import SimpleNamespace

import pytest

from granite_evals.benchmarks import include
from granite_evals.registry import RunConfig, get

# albanian has five domains, georgian one (arts_humanities, shared with
# albanian's first): the no-limit path builds a task per domain, and lm-eval
# cannot build a task with zero docs, so the fixture must cover them all.
LANGS = ("albanian", "georgian")
CODES = {"albanian": "al", "georgian": "ka"}


def _row(slug, i, domain):
    return {
        "language": include.LANGUAGES[slug][0],
        "country": "somewhere",
        "domain": include.DOMAINS[domain],
        "subject": f"subject {i}",
        "regional_feature": "region implicit",
        "level": "Academic",
        "question": f"[{CODES[slug]}] question {i} ",
        **{f"option_{l}": f"option {l}" for l in "abcd"},
        "answer": i % 4,
    }


@pytest.fixture
def dataset_dir(tmp_path, monkeypatch):
    """A local INCLUDE-base-44 look-alike: one config per language (the real
    config names), one row per domain of each language plus a second row for
    albanian's first two (so some tasks have doc index 1), answers cycling
    0-3. A private datasets cache: datasets keys a local folder's cache by its
    name, so tests would read each other's rows."""
    datasets = pytest.importorskip("datasets")
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "hf-cache")
    root = tmp_path / "include-44"
    configs = []
    for slug in LANGS:
        config = include.LANGUAGES[slug][0]
        doms = include.LANGUAGES[slug][1]
        rows = [_row(slug, i, dom) for i, dom in enumerate(doms)] + [_row(slug, 100 + i, dom) for i, dom in enumerate(doms[:2])]
        (root / slug).mkdir(parents=True)
        (root / slug / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        configs.append(f"- config_name: {config}\n  data_files:\n  - split: test\n    path: {slug}/test.jsonl\n")
    (root / "README.md").write_text("---\nconfigs:\n" + "".join(configs) + "---\n")
    return root


def test_registered_with_suite_metadata():
    cls = get("include-44")
    assert cls is include.Include44
    assert cls.metric == "accuracy (0-shot loglikelihood)"
    assert cls.default_repeats == 1 and cls.extra == "lmeval"
    assert cls.dataset == "CohereLabs/include-base-44"
    assert len(cls.dataset_revision) == 40


def test_task_names_round_trip():
    names = [include.task_name(lang, dom) for lang, (_, doms) in include.LANGUAGES.items() for dom in doms]
    assert len(names) == 178 == len(set(names))
    for name in names:
        assert include.task_name(*include.parse_task(name)) == name
    assert include.parse_task("include_base_44_north_macedonian_social_science") == ("north_macedonian", "social_science")


def test_languages_table_matches_the_upstream_tasks():
    assert len(include.LANGUAGES) == 44
    assert len({c for c, _ in include.LANGUAGES.values()}) == 44  # one config per language
    assert {d for _, doms in include.LANGUAGES.values() for d in doms} == set(include.DOMAINS)
    assert set(include.DOMAIN_OF_CATEGORY) == set(include.DOMAINS.values())


def test_languages_option(tmp_path):
    def langs(**options):
        return include.Include44(RunConfig(model="m", output_dir=tmp_path, options=options)).languages()

    assert langs() == list(include.LANGUAGES) and len(langs()) == 44
    assert langs(languages="albanian, chinese") == ["albanian", "chinese"]
    with pytest.raises(SystemExit):
        langs(languages="en")  # INCLUDE slugs are full language names


def test_select_examples_interleaves_languages_and_maps_to_task_indices():
    arts, stem = include.DOMAINS["arts_humanities"], include.DOMAINS["stem"]
    rows = {"albanian": [{"domain": arts if p % 2 == 0 else stem} for p in range(4)],
            "georgian": [{"domain": arts} for p in range(4)]}  # georgian's only domain
    # Per task (language x domain) the doc indices are positions in the
    # domain-filtered split: albanian arts 0,1 are rows 0,2 and stem rows 1,3;
    # georgian arts is rows 0-3. Order is (in-task index, language), so the
    # first five are every task's first question, then albanian's seconds.
    assert include.select_examples(rows, 5) == {
        "include_base_44_albanian_arts_humanities": [0, 1],
        "include_base_44_albanian_stem": [0, 1],
        "include_base_44_georgian_arts_humanities": [0],
    }
    full = include.select_examples(rows, None)
    assert sum(len(v) for v in full.values()) == 8


def test_sample_status():
    assert include.sample_status({"filtered_resps": [(-1e9, False), (-1.2, False), (0.0, True), (-0.3, False)]}) == "error"
    assert include.sample_status({"filtered_resps": [(-1.2, False)] * 4}) == "answered"
    assert include.sample_status({"filtered_resps": [(-100.0, False)]}) == "answered"  # gold's non-answer letter
    assert include.sample_status({}) == "answered"


def test_summarize_macro_over_languages_and_ignores_other_filters():
    def s(score, resps=((-1.2, False),) * 4, flt=include.FILTER):
        return {"filter": flt, "acc": score, "filtered_resps": list(resps)}

    out = include.summarize(
        {
            "include_base_44_albanian_arts_humanities": [s(1.0), s(0.0, ((-1e9, False),) * 4), s(1.0, flt="other")],
            "include_base_44_albanian_stem": [s(1.0), s(1.0)],
            "include_base_44_chinese_stem": [s(0.0)],
        }
    )
    assert out["per_language"]["albanian"] == {"acc": 0.75, "n": 4}
    assert out["per_language"]["chinese"] == {"acc": 0.0, "n": 1}
    assert out["value"] == pytest.approx(0.375) and out["n"] == 5
    assert out["micro"] == {"acc": 0.6, "n": 5}
    assert out["per_domain"]["stem"] == {"acc": 2 / 3, "n": 3}
    assert out["statuses"] == {"answered": 4, "error": 1}


def test_include_pass_at_k_macro_over_languages():
    def s(doc_id, score, flt=include.FILTER):
        return {"filter": flt, "acc": score, "doc_id": doc_id}

    r0 = include.sample_scores({"include_base_44_albanian_stem": [s(0, 1.0), s(1, 0.0), s(1, 1.0, flt="other")],
                                "include_base_44_chinese_stem": [s(0, 0.0)]})  # fmt: skip
    r1 = include.sample_scores({"include_base_44_albanian_stem": [s(0, 0.0), s(1, 1.0)], "include_base_44_chinese_stem": [s(0, 0.0)]})
    assert r0 == {"albanian": {"include_base_44_albanian_stem/0": 1.0, "include_base_44_albanian_stem/1": 0.0},
                  "chinese": {"include_base_44_chinese_stem/0": 0.0}}  # fmt: skip
    # albanian: pass@1 0.5, pass@2 1.0; chinese: 0 and 0
    out = include.include_pass_at_k([r0, r1], 2)
    how = "mean over languages; an example counts if it was correct in any repeat"
    assert out == {"k": 2, "pass_at_1": 0.25, "pass_at_k": 0.5, "n": 3, "how": how}


def test_task_spec_repoints_the_renamed_dataset(tmp_path):
    b = include.Include44(RunConfig(model="m", output_dir=tmp_path))
    spec = b.task_spec(["include_base_44_albanian_stem"], b.dataset, b.dataset_revision)
    assert spec["group"] == "granite_include_44"
    assert spec["task"] == [{"task": "include_base_44_albanian_stem", "dataset_path": "CohereLabs/include-base-44",
                             "dataset_kwargs": {"revision": b.dataset_revision}}]  # fmt: skip
    local = b.task_spec(["include_base_44_albanian_stem"], "/data/x", None)
    assert local["task"] == [{"task": "include_base_44_albanian_stem", "dataset_path": "/data/x", "dataset_kwargs": {}}]


def test_completions_payload_is_greedy_loglikelihood():
    """What lm-eval's local-completions puts on the wire to score one option:
    the whole prompt echoed back, one token, logprobs, temperature 0."""
    pytest.importorskip("lm_eval")
    from lm_eval.models.openai_completions import LocalCompletionsAPI

    p = LocalCompletionsAPI._create_payload(SimpleNamespace(model="m"), "Q. Answer:", generate=False, seed=7)
    assert p == {"model": "m", "prompt": "Q. Answer:", "temperature": 0, "max_tokens": 1, "logprobs": 1, "seed": 7, "echo": True}


# --- end to end through lm-eval (needs the lmeval extra) ---------------------


def _bench(tmp_path, dataset_dir, limit=None, **options):
    options = {"languages": ",".join(LANGS), **options}
    cfg = RunConfig(model="m", output_dir=tmp_path / "out", limit=limit, dataset=str(dataset_dir), options=options)
    return include.Include44(cfg)


def test_gold_scores_one_end_to_end(tmp_path, dataset_dir):
    pytest.importorskip("lm_eval")
    b = _bench(tmp_path, dataset_dir, answers="gold")
    assert not b.needs_server()
    out = b.run("", "")
    assert out["value"] == 1.0 and out["n"] == 9  # albanian 5 domains + 2 extra rows, georgian 1 domain + 1
    assert out["mode"] == "gold" and out["languages"] == list(LANGS)
    assert {k: out["pass_at_k"][k] for k in ("k", "pass_at_1", "pass_at_k", "n")} == {"k": 1, "pass_at_1": 1.0,
                                                                                    "pass_at_k": 1.0, "n": 9}  # fmt: skip
    rep = out["per_repeat"][0]
    assert "sample_scores" not in rep
    assert rep["statuses"] == {"answered": 9} and rep["micro"] == {"acc": 1.0, "n": 9}
    assert rep["per_language"]["albanian"]["n"] == 7 and rep["per_language"]["georgian"]["n"] == 2
    assert set(rep["per_language"]) == set(LANGS) and set(rep["per_domain"]) == set(include.LANGUAGES["albanian"][1])
    rows = [json.loads(x) for x in (tmp_path / "out/repeat-0/samples/include_base_44_albanian_stem.jsonl").read_text().splitlines()]
    assert rows and all(r["filter"] == "none" and r["acc"] == 1.0 for r in rows)
    # The gold answer's letter is the argmax at 0.0; the others score -100,
    # not the error tuple, so gold rows never count as errors.
    assert all([0.0, True] in r["filtered_resps"] and r["filtered_resps"].count([-100.0, False]) == 3 for r in rows)


class FakeCompletionsLM:
    """Stands in for local-completions: answers the reference letter, fails
    on one prompt."""

    calls: list = []

    @staticmethod
    def make(fail_on=None):
        from lm_eval.api.model import LM

        class _LM(LM):
            def loglikelihood(self, requests, disable_tqdm=False):
                out = []
                for req in requests:
                    FakeCompletionsLM.calls.append(req.args[0])
                    if fail_on and fail_on in req.args[0]:
                        out.append(include.ERROR_LL)
                    else:
                        letter = include.LETTERS[int(req.doc["answer"])]
                        out.append((0.0, True) if req.args[1].strip() == letter else (-1.0, False))
                        self.cache_hook.add_partial("loglikelihood", req.args, out[-1])
                return out

            def generate_until(self, requests, disable_tqdm=False):
                raise NotImplementedError

            def loglikelihood_rolling(self, requests, disable_tqdm=False):
                raise NotImplementedError

            @property
            def tokenizer_name(self):
                return "fake"

        return _LM()


def test_limit_errors_and_resume(tmp_path, dataset_dir, monkeypatch):
    pytest.importorskip("lm_eval")
    FakeCompletionsLM.calls = []
    # LIMIT=6 interleaves by (in-task index, language): Albanian's five first
    # questions, then Georgian's. Question 3 (answer 3) is the only failing
    # one; the all-options-failed argmax (A) scores it wrong, as documented.
    # (The template strips the question, so match without the trailing space.)
    monkeypatch.setattr(include.Include44, "make_lm", lambda self, *a, **k: FakeCompletionsLM.make("[al] question 3"))
    out = _bench(tmp_path, dataset_dir, limit=6).run("http://x/v1", "m")
    rep = out["per_repeat"][0]
    assert out["n"] == 6 and rep["statuses"] == {"answered": 5, "error": 1}
    assert len(FakeCompletionsLM.calls) == 24  # 6 questions x 4 options
    assert rep["per_language"]["albanian"] == {"acc": 0.8, "n": 5}
    assert rep["per_language"]["georgian"] == {"acc": 1.0, "n": 1}
    assert out["value"] == 0.9 and rep["micro"]["acc"] == pytest.approx(5 / 6)

    # Resume: everything answered comes from the cache; the error is retried.
    FakeCompletionsLM.calls = []
    monkeypatch.setattr(include.Include44, "make_lm", lambda self, *a, **k: FakeCompletionsLM.make())
    out2 = _bench(tmp_path, dataset_dir, limit=6).run("http://x/v1", "m")
    assert len(FakeCompletionsLM.calls) == 4 and all("[al] question 3" in c for c in FakeCompletionsLM.calls)
    assert out2["per_repeat"][0]["statuses"] == {"answered": 6}
    assert out2["value"] == 1.0
    assert json.dumps(out2)  # results.json-serialisable
