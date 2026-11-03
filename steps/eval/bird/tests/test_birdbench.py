"""BIRD (birdbench) tests: execution-match scoring, data extraction and the run
loop (gold and model mode) on a synthetic dev set, with fakes (no network)."""

import hashlib
import io
import json
import sqlite3
import zipfile
from types import SimpleNamespace

import pytest

from granite_evals.benchmarks import birdbench
from granite_evals.registry import RunConfig, get


def make_db(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, score REAL);
        INSERT INTO t VALUES (1, 'a', 1.0), (2, 'b', 2.0), (3, 'c', 3.0);
    """)
    conn.commit()
    conn.close()
    return path


QUESTIONS = [
    {"question_id": 1, "db_id": "toy", "question": "How many rows?", "evidence": "rows are in t",
     "SQL": "SELECT COUNT(*) FROM t", "difficulty": "simple"},
    {"question_id": 0, "db_id": "toy", "question": "Names?", "evidence": "",
     "SQL": "SELECT name FROM t ORDER BY id", "difficulty": "moderate"},
    {"question_id": 2, "db_id": "toy", "question": "Max score?", "evidence": "",
     "SQL": "SELECT MAX(score) FROM t", "difficulty": "challenging"},
]


@pytest.fixture
def dev(tmp_path):
    d = tmp_path / "data" / birdbench.DEV_DIR
    make_db(d / "dev_databases" / "toy" / "toy.sqlite")
    (d / "dev.json").write_text(json.dumps(QUESTIONS))
    return d


def test_registered_with_suite_metric():
    cls = get("birdbench")
    assert cls is birdbench.BirdBench
    assert cls.metric == "pass@1 execution match" and cls.extra == "bird"
    assert cls.dataset_revision == f"sha256:{birdbench.DEV_ZIP_SHA256}"


def test_execution_match_statuses(tmp_path):
    db = make_db(tmp_path / "toy.sqlite")
    em = birdbench.execution_match
    # Set semantics: row order and duplicates don't matter (BIRD's comparison).
    assert em("SELECT name FROM t ORDER BY id DESC", "SELECT name FROM t", db, 5)["status"] == "correct"
    assert em("SELECT name FROM t UNION ALL SELECT 'a'", "SELECT name FROM t", db, 5)["correct"]
    assert em("SELECT name FROM t WHERE id < 3", "SELECT name FROM t", db, 5)["status"] == "wrong"
    assert em("SELEC nonsense", "SELECT 1", db, 5)["status"] == "pred_error"
    assert em("SELECT 1", "SELECT nope FROM t", db, 5)["status"] == "gold_error"


def test_execution_is_read_only(tmp_path):
    db = make_db(tmp_path / "toy.sqlite")
    out = birdbench.execution_match("DROP TABLE t", "SELECT 1", db, 5)
    assert out["status"] == "pred_error" and "readonly" in out["error"]
    assert birdbench.execution_match("SELECT COUNT(*) FROM t", "SELECT 3", db, 5)["correct"]


def test_execution_timeout_aborts_query(tmp_path):
    db = make_db(tmp_path / "toy.sqlite")
    slow = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
    out = birdbench.execution_match(slow, "SELECT 1", db, 0.3)
    assert out["status"] == "timeout" and out["exec_s"] < 5
    # The budget is shared: a slow gold after the prediction also times out.
    assert birdbench.execution_match("SELECT 1", slow, db, 0.3)["status"] == "timeout"


def test_question_text_and_summarize():
    q = QUESTIONS[0]
    assert birdbench.question_text(q, False) == "How many rows?"
    assert birdbench.question_text(q, True) == "rows are in t\nHow many rows?"
    assert birdbench.question_text(QUESTIONS[1], True) == "Names?"
    s = birdbench.summarize([
        {"difficulty": "simple", "correct": True, "status": "correct"},
        {"difficulty": "simple", "correct": False, "status": "timeout"},
        {"difficulty": "moderate", "correct": False, "status": "wrong"},
    ])
    assert s["accuracy"] == pytest.approx(1 / 3)
    assert s["by_difficulty"]["simple"] == {"n": 2, "correct": 1, "accuracy": 0.5}
    assert "challenging" not in s["by_difficulty"]
    assert s["statuses"] == {"correct": 1, "timeout": 1, "wrong": 1}


def test_extract_dev_nested_zip_skips_macos_junk(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("dev_databases/toy/toy.sqlite", b"db")
        z.writestr("dev_databases/.DS_Store", b"x")
        z.writestr("__MACOSX/dev_databases/._toy", b"x")
    outer = tmp_path / "dev.zip"
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr(f"{birdbench.DEV_DIR}/dev.json", "[]")
        z.writestr(f"{birdbench.DEV_DIR}/dev_databases.zip", inner.getvalue())
        z.writestr(f"__MACOSX/{birdbench.DEV_DIR}/._dev.json", b"x")
    cache = tmp_path / "cache"
    birdbench.extract_dev(outer, cache)
    dev = cache / birdbench.DEV_DIR
    assert (dev / "dev_databases" / "toy" / "toy.sqlite").read_bytes() == b"db"
    assert not (dev / "dev_databases.zip").exists()
    assert not (dev / "dev_databases" / ".DS_Store").exists()
    assert not (cache / "__MACOSX").exists()


def test_download_rejects_sha_mismatch(tmp_path, monkeypatch):
    import contextlib

    import httpx

    payload = b"not the pinned zip"

    @contextlib.contextmanager
    def fake_stream(method, url, **kw):
        yield httpx.Response(200, content=payload, request=httpx.Request(method, url))

    monkeypatch.setattr(httpx, "stream", fake_stream)
    dest = tmp_path / "dev.zip"
    with pytest.raises(RuntimeError, match="sha256"):
        birdbench.download("https://example.invalid/dev.zip", dest, "0" * 64)
    assert not dest.exists() and not dest.with_suffix(".partial").exists()
    birdbench.download("https://example.invalid/dev.zip", dest, hashlib.sha256(payload).hexdigest())
    assert dest.read_bytes() == payload


def test_gold_mode_scores_everything_and_resumes(tmp_path, dev):
    cfg = RunConfig(model="none", output_dir=tmp_path / "out", dataset=str(dev), limit=2,
                    workers=2, options={"sql": "gold"})
    b = birdbench.BirdBench(cfg)
    assert not b.needs_server()
    out = b.run("", "")
    assert out["value"] == 1.0 and out["n"] == 2 and out["mode"] == "gold"
    # --limit takes the first questions by question_id.
    assert sorted(p.name for p in (tmp_path / "out" / "repeat-0").iterdir()) == ["0.json", "1.json"]
    rec = json.loads((tmp_path / "out" / "repeat-0" / "1.json").read_text())
    assert rec["pred_sql"] == rec["gold_sql"] and rec["status"] == "correct"
    # A persisted result is reused on restart, not re-executed.
    rec["correct"], rec["status"] = False, "wrong"
    (tmp_path / "out" / "repeat-0" / "1.json").write_text(json.dumps(rec))
    assert birdbench.BirdBench(cfg).run("", "")["value"] == 0.5


def test_model_mode_with_fakes(tmp_path, dev, monkeypatch):
    sent = []

    class Completions:
        def create(self, **kw):
            sent.append(kw)
            user = kw["messages"][-1]["content"]
            sql = {"How many rows?": "SELECT COUNT(*) FROM t", "Names?": "SELECT id FROM t"}.get(user, "SELEC")
            msg = SimpleNamespace(content=f"thinking\n```sql\n{sql}\n```", reasoning="r")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None)

    class Prompt:
        def fill(self, values):
            assert values["sql_context"] == "CREATE TABLE t (...)"
            return [{"role": "system", "content": "sys"}, {"role": "user", "content": values["question"]}]

    def extract(text):
        return text.split("```sql")[-1].split("```")[0].strip()

    monkeypatch.setattr(birdbench, "_client", lambda url: SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
    monkeypatch.setattr(birdbench, "_prompt", lambda: Prompt())
    monkeypatch.setattr(birdbench, "_extractor", lambda dev: extract)
    monkeypatch.setattr(birdbench.BirdBench, "schemas", lambda self, dev: {"toy": "CREATE TABLE t (...)"})

    cfg = RunConfig(model="m", output_dir=tmp_path / "out", dataset=str(dev), workers=1, repeats=2, seed=7,
                    options={"temperature": "0.5", "max_tokens": "100"})
    out = birdbench.BirdBench(cfg).run("http://srv/v1", "served")
    assert out["value"] == pytest.approx(1 / 3) and out["n"] == 3
    assert out["per_repeat"][0]["statuses"] == {"correct": 1, "pred_error": 1, "wrong": 1}
    assert out["sampling"] == {"max_tokens": 100, "temperature": 0.5, "top_p": 0.95, "top_k": 20}
    assert {kw["seed"] for kw in sent} == {7, 8} and all(kw["model"] == "served" for kw in sent)
    # ns's GENERATION_ARGS unless overridden; top_k as vLLM's extra_body
    assert all((kw["max_tokens"], kw["temperature"], kw["top_p"], kw["extra_body"]) == (100, 0.5, 0.95, {"top_k": 20})
               for kw in sent)  # fmt: skip
    assert (out["questions_failed"], out["questions_total"], out["incomplete"]) == (0, 6, False)
    rec = json.loads((tmp_path / "out" / "repeat-1" / "1.json").read_text())
    assert rec["pred_sql"] == "SELECT COUNT(*) FROM t" and rec["reasoning"] == "r"


def test_sampling_is_nemo_skills_generation_args():
    ns = pytest.importorskip("nemo_skills.dataset.birdbench")
    args = dict(a[2:].split("=", 1) for a in ns.GENERATION_ARGS.split())
    assert birdbench.NS_SAMPLING == {
        "max_tokens": int(args["inference.tokens_to_generate"]),
        "temperature": float(args["inference.temperature"]),
        "top_p": float(args["inference.top_p"]),
        "top_k": int(args["inference.top_k"]),
    }
    assert args["prompt_config"] == birdbench.PROMPT_CONFIG
    b = birdbench.BirdBench(RunConfig(model="m", output_dir=None, options={"top_k": "-1"}))
    assert "extra_body" not in b.request_args()  # top_k <= 0: not sent, as ns


def _failing_client(fail):
    class Completions:
        def create(self, **kw):
            question = kw["messages"][-1]["content"]
            if question in fail:
                raise ConnectionError("server gone")
            sql = "SELECT COUNT(*) FROM t" if question == "How many rows?" else "SELECT 0"
            msg = SimpleNamespace(content=f"```sql\n{sql}\n```", reasoning=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None)

    return lambda url: SimpleNamespace(chat=SimpleNamespace(completions=Completions()))


def test_failed_requests_are_left_out_counted_and_retried(tmp_path, dev, monkeypatch):
    prompt = SimpleNamespace(fill=lambda v: [{"role": "user", "content": v["question"]}])
    monkeypatch.setattr(birdbench, "_prompt", lambda: prompt)
    monkeypatch.setattr(birdbench, "_extractor", lambda dev: lambda t: t.split("```sql")[-1].split("```")[0].strip())
    monkeypatch.setattr(birdbench.BirdBench, "schemas", lambda self, dev: {"toy": ""})
    cfg = RunConfig(model="m", output_dir=tmp_path / "out", dataset=str(dev), workers=1,
                    options={"max_failed_frac": "0.5"})
    monkeypatch.setattr(birdbench, "_client", _failing_client({"Names?"}))
    out = birdbench.BirdBench(cfg).run("http://srv/v1", "served")
    # 1 of 2 scored questions right, not 1 of 3: the failed one is no wrong answer
    assert (out["value"], out["n"], out["questions_failed"], out["questions_total"]) == (0.5, 2, 1, 3)
    assert out["incomplete"] and not (tmp_path / "out" / "repeat-0" / "0.json").exists()
    monkeypatch.setattr(birdbench, "_client", _failing_client(set()))
    out = birdbench.BirdBench(cfg).run("http://srv/v1", "served")  # retried on resume
    assert (out["n"], out["questions_failed"], out["incomplete"]) == (3, 0, False)
    # default threshold 0.05: one failed question of three fails the run
    for p in (tmp_path / "out" / "repeat-0").iterdir():
        p.unlink()
    monkeypatch.setattr(birdbench, "_client", _failing_client({"Names?"}))
    with pytest.raises(SystemExit, match="questions_failed 1/3"):
        birdbench.BirdBench(RunConfig(model="m", output_dir=tmp_path / "out", dataset=str(dev))).run("u", "s")


def test_nemo_skills_prompt_and_extraction(tmp_path):
    pytest.importorskip("nemo_skills")
    prompt = birdbench._prompt()
    messages = prompt.fill({"question": "How many rows?", "sql_context": "CREATE TABLE t (id INT);"})
    assert messages[0]["role"] == "system" and "```sql" in messages[0]["content"]
    assert "How many rows?" in messages[-1]["content"] and "CREATE TABLE t" in messages[-1]["content"]
    extract = birdbench._extractor(tmp_path)
    assert extract("x\n```sql\nSELECT 1\n```\n```sql\nSELECT 2\n```").strip() == "SELECT 2"


def test_nemo_skills_schema_context(tmp_path, dev):
    pytest.importorskip("nemo_skills")
    cfg = RunConfig(model="m", output_dir=tmp_path / "out", dataset=str(dev))
    schemas = birdbench.BirdBench(cfg).schemas(dev)
    assert "CREATE TABLE t" in schemas["toy"]
    assert (dev.parent / "sql_context.json").exists()


# -- generate / score phases ---------------------------------------------------


def _fake_model(monkeypatch, sent):
    """The fakes of test_model_mode_with_fakes: question -> SQL, one wrong, one broken."""

    class Completions:
        def create(self, **kw):
            sent.append(kw)
            sql = {"How many rows?": "SELECT COUNT(*) FROM t", "Names?": "SELECT id FROM t"}.get(
                kw["messages"][-1]["content"], "SELEC")
            msg = SimpleNamespace(content=f"```sql\n{sql}\n```", reasoning=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], usage=None)

    monkeypatch.setattr(birdbench, "_client", lambda url: SimpleNamespace(chat=SimpleNamespace(completions=Completions())))
    monkeypatch.setattr(birdbench, "_prompt", lambda: SimpleNamespace(fill=lambda v: [{"role": "user", "content": v["question"]}]))
    monkeypatch.setattr(birdbench, "_extractor", lambda dev: lambda t: t.split("```sql")[-1].split("```")[0].strip())
    monkeypatch.setattr(birdbench.BirdBench, "schemas", lambda self, dev: {"toy": ""})


def _no_model(monkeypatch):
    def refuse(*a, **kw):
        raise AssertionError("the score phase must not touch the model")

    for name in ("_client", "_prompt", "_extractor"):
        monkeypatch.setattr(birdbench, name, refuse)
    monkeypatch.setattr(birdbench.BirdBench, "schemas", refuse)


def _cfg(tmp_path, dev, phase, **kw):
    return RunConfig(model="m", output_dir=tmp_path / "out", dataset=str(dev), workers=1, phase=phase, **kw)


def test_generate_then_score_equals_all(tmp_path, dev, monkeypatch):
    sent = []
    _fake_model(monkeypatch, sent)
    gen = birdbench.BirdBench(_cfg(tmp_path, dev, "generate", repeats=2)).run("http://srv/v1", "served")
    assert "value" not in gen and gen["n"] == 3 and len(sent) == 6
    assert gen["per_repeat"] == [{"repeat": k, "generated": 3, "failed": 0} for k in range(2)]
    rec = json.loads((tmp_path / "out" / "repeat-0" / "1.json").read_text())
    assert rec["status"] == birdbench.GENERATED and "correct" not in rec and rec["pred_sql"] == "SELECT COUNT(*) FROM t"
    # A second generate run resumes: nothing is regenerated, nothing is executed.
    birdbench.BirdBench(_cfg(tmp_path, dev, "generate", repeats=2)).run("http://srv/v1", "served")
    assert len(sent) == 6 and json.loads((tmp_path / "out" / "repeat-0" / "1.json").read_text()) == rec

    _no_model(monkeypatch)
    out = birdbench.BirdBench(_cfg(tmp_path, dev, "score", repeats=2)).run("", "served")
    assert out["value"] == pytest.approx(1 / 3) and out["n"] == 3 and not out["incomplete"]
    assert out["per_repeat"][0]["statuses"] == {"correct": 1, "pred_error": 1, "wrong": 1}
    rec = json.loads((tmp_path / "out" / "repeat-0" / "1.json").read_text())
    assert rec["status"] == "correct" and rec["correct"] and rec["pred_rows"] == 1
    # Re-running score reuses the scored files (a hand edit shows they are not re-executed).
    rec["correct"], rec["status"] = False, "wrong"
    (tmp_path / "out" / "repeat-0" / "1.json").write_text(json.dumps(rec))
    out = birdbench.BirdBench(_cfg(tmp_path, dev, "score", repeats=2)).run("", "served")
    assert out["value"] == pytest.approx(1 / 6)
    # question 1 now matches in repeat 1 only: pass@1 halves, pass@2 still counts it
    pak = {"k": 2, "pass_at_1": pytest.approx(1 / 6), "pass_at_k": pytest.approx(1 / 3), "n": 3}
    assert out["pass_at_k"] == {**pak, "how": "matched in any scored repeat"}

    # --phase all on fresh generations gives the same records and score.
    sent.clear()
    _fake_model(monkeypatch, sent)
    cfg = _cfg(tmp_path, dev, "all", repeats=2)
    cfg.output_dir = tmp_path / "all"
    assert birdbench.BirdBench(cfg).run("http://srv/v1", "served")["value"] == pytest.approx(1 / 3)
    for k in range(2):
        for qid in range(3):
            split = json.loads((tmp_path / "out" / f"repeat-{k}" / f"{qid}.json").read_text())
            whole = json.loads((tmp_path / "all" / f"repeat-{k}" / f"{qid}.json").read_text())
            if (k, qid) != (0, 1):  # hand-edited above
                assert {x: v for x, v in split.items() if x != "exec_s"} == {x: v for x, v in whole.items() if x != "exec_s"}


def test_score_counts_a_missing_generation_as_failed(tmp_path, dev, monkeypatch):
    _fake_model(monkeypatch, [])
    birdbench.BirdBench(_cfg(tmp_path, dev, "generate")).run("http://srv/v1", "served")
    (tmp_path / "out" / "repeat-0" / "0.json").unlink()
    _no_model(monkeypatch)
    out = birdbench.BirdBench(_cfg(tmp_path, dev, "score", options={"max_failed_frac": "0.5"})).run("", "served")
    # Left out and counted, like a failed request; its file is not created.
    assert (out["value"], out["n"], out["questions_failed"], out["incomplete"]) == (0.5, 2, 1, True)
    assert not (tmp_path / "out" / "repeat-0" / "0.json").exists()
    with pytest.raises(SystemExit, match="questions_failed 1/3"):
        birdbench.BirdBench(_cfg(tmp_path, dev, "score")).run("", "served")


def test_gold_mode_in_every_phase(tmp_path, dev, monkeypatch):
    _no_model(monkeypatch)
    opts = {"sql": "gold"}
    gen = birdbench.BirdBench(_cfg(tmp_path, dev, "generate", options=opts)).run("", "")
    assert gen["n"] == 3 and gen["mode"] == "gold"
    rec = json.loads((tmp_path / "out" / "repeat-0" / "2.json").read_text())
    assert rec["status"] == birdbench.GENERATED and rec["pred_sql"] == rec["gold_sql"]
    out = birdbench.BirdBench(_cfg(tmp_path, dev, "score", options=opts)).run("", "")
    assert out["value"] == 1.0 and out["n"] == 3
