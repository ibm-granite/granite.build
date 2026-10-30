"""OfficeQA adapter tests with a fake corpus and a fake model (no network, no
GPU, no pandas needed except the table-rendering tests).

Set ``OFFICEQA_UPSTREAM=<checkout of databricks/OfficeQA at
officeqa.UPSTREAM_COMMIT>`` to also run the parity test, which holds the
reward port against the pinned upstream file.
"""

import csv
import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import httpx
import pytest
from granite_evals.benchmarks import officeqa as oq
from granite_evals.registry import RunConfig, get

_REAL_CORPUS = oq.Corpus

# -- fixtures and fakes ---------------------------------------------------------


def make_csv(root: Path, name: str, rows: list[dict]) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=oq.QUESTION_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def make_json(root: Path, name: str, elements: list[tuple[str, int | None]]) -> Path:
    """A parsed-corpus document: ``elements`` of ``(content, page_id)``; a
    ``None`` page_id drops the element's bbox entirely."""
    doc = {
        "document": {
            "elements": [
                {
                    "content": content,
                    "bbox": ([{"page_id": page}] if page is not None else []),
                }
                for content, page in elements
            ]
        }
    }
    path = root / oq.JSONS_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return path


def row(
    uid,
    answer="2,602",
    docs="https://example.org/tb/1939_01/index.html?page=1",
    files="treasury_bulletin_1939_01.txt",
    difficulty="easy",
    question=None,
):
    return {
        "uid": uid,
        "question": question or f"What was the value of {uid}?",
        "answer": answer,
        "source_docs": docs,
        "source_files": files,
        "difficulty": difficulty,
    }


PRO_ROWS = [
    row("p1", answer="2,602", difficulty="easy"),
    row(
        "p2",
        answer="1608.80%",
        docs="https://example.org/tb/1939_01/index.html?page=1\nhttps://example.org/tb/1941_02/index.html?page=2",
        files="treasury_bulletin_1939_01.txt\ntreasury_bulletin_1941_02.txt",
        difficulty="hard",
    ),
]
FULL_ROWS = [row("f1", answer="0.096")]


def make_tree(root: Path) -> Path:
    """A small dataset checkout: both CSVs and two parsed documents."""
    make_csv(root, oq.PRO_CSV, PRO_ROWS)
    make_csv(root, oq.FULL_CSV, FULL_ROWS)
    make_json(
        root,
        "treasury_bulletin_1939_01.json",
        [("page one text", 1), ("page two text", 2)],
    )
    make_json(
        root,
        "treasury_bulletin_1941_02.json",
        [("other page one", 1), ("other page two", 2)],
    )
    return root


@pytest.fixture(autouse=True)
def _fresh_caches():
    """Neither the module's page cache nor FakeModel's class state may leak
    across tests (FakeModel is a stand-in for a module-level class)."""
    oq._pages_cached.cache_clear()
    FakeModel.instances = []
    FakeModel.answers = {}
    FakeModel.fail_substrings = ()
    yield
    oq._pages_cached.cache_clear()


@pytest.fixture
def pinned_tree(tmp_path, monkeypatch) -> Path:
    """A small tree with the module's pins re-pointed at it (counts, CSV blob
    sha1s and the corpus digest), so fetch/verify paths can be exercised
    offline."""
    tree = make_tree(tmp_path / "data")
    monkeypatch.setattr(oq, "PRO_N", len(PRO_ROWS))
    monkeypatch.setattr(oq, "FULL_N", len(FULL_ROWS))
    monkeypatch.setattr(oq, "N_JSONS", 2)
    monkeypatch.setattr(
        oq, "PRO_CSV_SHA1", oq.git_blob_sha1((tree / oq.PRO_CSV).read_bytes())
    )
    monkeypatch.setattr(
        oq, "FULL_CSV_SHA1", oq.git_blob_sha1((tree / oq.FULL_CSV).read_bytes())
    )
    lines = sorted(
        f"{oq.JSONS_DIR}/{p.name} {oq.git_blob_sha1(p.read_bytes())}"
        for p in sorted((tree / oq.JSONS_DIR).glob("*.json"))
    )
    monkeypatch.setattr(
        oq,
        "JSONS_SHA256",
        hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(),
    )
    return tree


class FakeModel:
    """ModelClient stand-in: answers by matching the user message, so thread
    scheduling cannot scramble the script; ``fail_substrings`` raise."""

    instances: list["FakeModel"] = []
    answers: dict[str, str] = {}
    fail_substrings: tuple[str, ...] = ()

    def __init__(self, base_url, model, *, timeout=1.0, transport=None):
        self.base_url, self.model, self.timeout = base_url, model, timeout
        self.calls = []  # (messages, max_tokens, temperature)
        FakeModel.instances.append(self)

    def close(self) -> None:
        pass

    def complete(self, messages, *, max_tokens, temperature):
        self.calls.append((messages, max_tokens, temperature))
        user = messages[-1]["content"]
        if any(s in user for s in FakeModel.fail_substrings):
            raise RuntimeError("model endpoint unreachable: down")
        for needle, answer in FakeModel.answers.items():
            if needle in user:
                return (
                    f"<FINAL_ANSWER>{answer}</FINAL_ANSWER>",
                    "stop",
                    {"prompt_tokens": 10},
                )
        return "<FINAL_ANSWER>?</FINAL_ANSWER>", "stop", {"prompt_tokens": 10}


@pytest.fixture
def fake_model(monkeypatch):
    monkeypatch.setattr(oq, "ModelClient", FakeModel)
    return FakeModel


def make_bench(tmp_path, *, phase="all", options=None, dataset=None, **kw):
    return oq.OfficeQAPro(
        RunConfig(
            model="m",
            output_dir=tmp_path / "out",
            phase=phase,
            options=options or {},
            dataset=dataset,
            **kw,
        )
    )


# -- registration, options and pins ---------------------------------------------


def test_registered_with_its_pins():
    assert get("officeqa-pro") is oq.OfficeQAPro
    assert get("officeqa-full") is oq.OfficeQAFull
    assert oq.OfficeQAPro.id == "officeqa-pro"
    assert oq.OfficeQAFull.id == "officeqa-full"
    for cls in (oq.OfficeQAPro, oq.OfficeQAFull):
        assert cls.splittable is True
        assert cls.metric == "accuracy@0%"
        assert cls.dataset == oq.DATASET
        assert cls.dataset_revision == oq.REVISION
        assert cls.extra == "officeqa"
        assert cls.harness_packages == ("pandas", "lxml")
    assert oq.DATASET == "databricks/OfficeQA"
    assert oq.REVISION == "763a8366abf2a3605c381d53586d844dc60fa756"
    assert oq.OfficeQAPro.csv_name == oq.PRO_CSV and oq.OfficeQAPro.n_questions == 133
    assert (
        oq.OfficeQAFull.csv_name == oq.FULL_CSV and oq.OfficeQAFull.n_questions == 246
    )
    assert oq.PRO_CSV_SHA1 == "8dc109a1611b29f79685b51e98f7de9cec463466"
    assert oq.FULL_CSV_SHA1 == "cc59c600c6bbdf04e5bcf31f43dd58f0c7b3978c"
    assert oq.N_JSONS == 697
    assert (
        oq.JSONS_SHA256
        == "49d7b00c2392efa355850b1d58c2e5987e4e0e21abbeddea036bab5766c8c842"
    )
    assert oq.UPSTREAM_COMMIT == "7b9a3c154ef9fb40215bb67934afc43e6799de16"
    assert oq.UPSTREAM_FILES["reward.py"] == (
        "0d91698c87df6d889339aac36f63ae0966607f169890b0bf8b472b26bfe8138f"
    )


def test_gold_mode_needs_no_server(tmp_path):
    plain = oq.OfficeQAPro(RunConfig(model="m", output_dir=tmp_path))
    gold = oq.OfficeQAPro(
        RunConfig(model="m", output_dir=tmp_path, options={"answers": "gold"})
    )
    assert plain.needs_server() and not plain.gold
    assert gold.gold and not gold.needs_server()
    assert plain.score_needs_server() is False and gold.score_needs_server() is False


def test_option_parsing(tmp_path):
    bench = oq.OfficeQAPro(RunConfig(model="m", output_dir=tmp_path))
    assert bench.opt("mode", "oracle_parsed") == "oracle_parsed"
    assert bench.opt("max_tokens", 4096) == 4096
    bench.config.options["max_context_chars"] = "4096"
    assert bench.opt("max_context_chars", 0) == 4096
    bench.config.options["temperature"] = "0.6"
    assert bench.opt("temperature", 0.0) == pytest.approx(0.6)


def test_mode_option_rejects_anything_but_oracle_parsed(tmp_path):
    bench = oq.OfficeQAPro(
        RunConfig(model="m", output_dir=tmp_path, options={"mode": "bm25"})
    )
    with pytest.raises(SystemExit, match="mode='bm25' is not implemented"):
        bench.run("http://llm:8000/v1", "m")


def test_mode_defaults_to_oracle_parsed(tmp_path):
    bench = oq.OfficeQAPro(RunConfig(model="m", output_dir=tmp_path))
    assert bench.mode == "oracle_parsed"


# -- the reward port -------------------------------------------------------------


def test_git_blob_sha1_is_gits_blob_oid():
    # git's canonical blob hash of b"hello" (git hash-object).
    assert oq.git_blob_sha1(b"hello") == "b6fc4c620b67d95f953a5c1c1230aaab5db5a1b0"


def test_normalize_text_collapses_whitespace_and_the_unicode_minus():
    assert oq.normalize_text("−1  234\t5678") == "-1 234 5678"
    with pytest.raises(ValueError):
        oq.normalize_text("")


def test_extract_final_answer_takes_the_last_tag_case_insensitively():
    text = "reason\n<final_answer>first</final_answer>\nmore <FINAL_ANSWER>last</FINAL_ANSWER>"
    assert oq.extract_final_answer_from_xml(text) == (
        "last",
        "reason\n<final_answer>first</final_answer>\nmore",
    )
    assert oq.extract_final_answer_from_xml("<FINAL_ANSWER>2,602</FINAL_ANSWER>") == (
        "2,602",
        None,
    )
    assert oq.extract_final_answer_from_xml("no tags at all") == (
        "no tags at all",
        None,
    )
    assert oq.extract_final_answer("") == ""  # empty text extracts to empty


@pytest.mark.parametrize(
    "gt,pred,expected",
    [
        ("2,602", "2,602", (1.0, 1.0, 1.0, 1.0)),
        ("2,602", "2602", (1.0, 1.0, 1.0, 1.0)),  # comma formatting is normalized
        ("1000", "1001", (0.0, 1.0, 1.0, 1.0)),  # 0.1% off: passes from 0.1% up
        ("1000", "1010", (0.0, 0.0, 1.0, 1.0)),  # 1% off: passes from 1% up
        ("1000", "1100", (0.0, 0.0, 0.0, 0.0)),  # 11% off: no tolerance saves it
        ("1608.80%", "1608.79%", (0.0, 1.0, 1.0, 1.0)),
        ("[0.096, -184.143]", "[0.096, -184.14]", (0.0, 1.0, 1.0, 1.0)),  # lists
        ("[0.096, -184.143]", "[0.096, -184.143]", (1.0, 1.0, 1.0, 1.0)),
        ("2,602", "3,000", (0.0, 0.0, 0.0, 0.0)),
        ("January 1941", "January 1941", (1.0, 1.0, 1.0, 1.0)),  # text answers
        ("January 1941", "February 1941", (0.0, 0.0, 0.0, 0.0)),
        ("$1,234.56", "(1,234.56)", (0.0, 0.0, 0.0, 0.0)),  # accounting negative
        ("36080 million", "36,080", (1.0, 1.0, 1.0, 1.0)),
        ("0.096", "0.096%", (1.0, 1.0, 1.0, 1.0)),
        ("2,602", "", (0.0, 0.0, 0.0, 0.0)),  # empty prediction
    ],
)
def test_score_answer_battery(gt, pred, expected):
    scores = oq.score_all_tolerances(gt, pred)
    assert tuple(scores[label] for label in oq.tolerance_labels()) == expected


def test_score_answer_extracts_the_xml_itself():
    assert oq.score_answer("2,602", "sure: <FINAL_ANSWER>2,602</FINAL_ANSWER>") == 1.0
    assert oq.score_answer("2,602", "sure: <FINAL_ANSWER>9,999</FINAL_ANSWER>") == 0.0


def test_tolerance_labels():
    assert oq.tolerance_labels() == ("0%", "0.1%", "1%", "5%")
    assert oq.TOLERANCES == (0.0, 0.001, 0.01, 0.05)


# -- Corpus: the launch-time fetch and its pins ----------------------------------


def test_ensure_verifies_and_returns_the_root(pinned_tree):
    calls = []

    def download(allow_patterns):
        calls.append(tuple(allow_patterns))
        return pinned_tree

    corpus = oq.Corpus(download=download).ensure(need_pages=True)
    assert corpus.root == pinned_tree
    assert calls == [tuple(oq.ALLOW_PATTERNS)]  # the corpus patterns
    oq.Corpus(download=download).ensure(need_pages=False)
    assert calls[1] == tuple(oq.CSV_ONLY_PATTERNS)  # the score phase: CSVs only


def test_verify_refuses_a_tampered_csv(pinned_tree, monkeypatch):
    monkeypatch.setattr(oq, "PRO_CSV_SHA1", oq.git_blob_sha1(b"swapped benchmark"))
    with pytest.raises(RuntimeError, match="blob sha1 .* refusing to run"):
        oq.Corpus(download=lambda allow: pinned_tree).ensure(need_pages=False)


def test_verify_refuses_a_wrong_question_count(pinned_tree, monkeypatch):
    monkeypatch.setattr(oq, "PRO_N", 3)
    with pytest.raises(RuntimeError, match="holds 2 questions, expected 3"):
        oq.Corpus(download=lambda allow: pinned_tree).ensure(need_pages=False)


def test_verify_refuses_a_missing_csv(pinned_tree):
    (pinned_tree / oq.FULL_CSV).unlink()
    with pytest.raises(RuntimeError, match="is missing from the fetch"):
        oq.Corpus(download=lambda allow: pinned_tree).ensure(need_pages=False)


def test_verify_refuses_a_tampered_corpus_file(pinned_tree):
    victim = sorted((pinned_tree / oq.JSONS_DIR).glob("*.json"))[0]
    victim.write_text(victim.read_text() + " tampered")
    with pytest.raises(RuntimeError, match="digest .* refusing to run"):
        oq.Corpus(download=lambda allow: pinned_tree).ensure(need_pages=True)


def test_verify_refuses_a_wrong_document_count(pinned_tree, monkeypatch):
    (sorted((pinned_tree / oq.JSONS_DIR).glob("*.json"))[0]).unlink()
    with pytest.raises(RuntimeError, match="holds 1 documents, expected 2"):
        oq.Corpus(download=lambda allow: pinned_tree).ensure(need_pages=True)


def test_the_score_phase_fetch_skips_the_corpus_check(pinned_tree):
    shutil.rmtree(pinned_tree / oq.JSONS_DIR)  # no corpus at all
    oq.Corpus(download=lambda allow: pinned_tree).ensure(need_pages=False)  # passes


def test_a_gated_fetch_explains_the_gate(monkeypatch):
    import huggingface_hub

    def gated(**kw):
        raise Exception(
            "403 Client Error: Forbidden for url: Access to dataset "
            "databricks/OfficeQA is restricted. You must be authenticated."
        )

    monkeypatch.setattr(huggingface_hub, "snapshot_download", gated)
    with pytest.raises(SystemExit, match="gated.*HF_TOKEN"):
        oq.Corpus().ensure(need_pages=True)


def test_an_ungated_fetch_error_is_raised_as_is(monkeypatch):
    import huggingface_hub

    def broken(**kw):
        raise ValueError("dns broke")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", broken)
    with pytest.raises(ValueError, match="dns broke"):
        oq.Corpus().ensure(need_pages=True)


# -- page extraction ---------------------------------------------------------------


def test_extract_pages_groups_elements_by_page_id(tmp_path):
    path = make_json(
        tmp_path,
        "d.json",
        [("alpha", 1), ("beta", 2), ("gamma", 1), ("no page", None), ("  ", 3)],
    )
    pages = oq.extract_pages_from_json(path)
    assert set(pages) == {1, 2}  # pageless and empty content drop out
    assert pages[1] == "alpha\n\ngamma\n"
    assert pages[2] == "beta\n"


def test_extract_pages_normalizes_crlf(tmp_path):
    path = make_json(tmp_path, "d.json", [("a\r\nb", 1)])
    assert oq.extract_pages_from_json(path) == {1: "a\nb\n"}


def test_dataframe_to_markdown_flattens_sanitizes_and_dedups():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        [["a|b", "c\nd", None], ["1", "2", "3"]],
        columns=pd.MultiIndex.from_tuples([("Year", 1941), ("Rate", "avg"), ("", "")]),
    )
    md = oq._dataframe_to_markdown(df)
    lines = md.split("\n")
    assert lines[0] == "| Year > 1941 | Rate > avg | col_3 |"
    assert lines[1] == "| --- | --- | --- |"
    assert (
        lines[2] == "| a\\|b | c d | nan |"
    )  # pandas turns None into NaN; upstream renders it
    assert lines[3] == "| 1 | 2 | 3 |"
    assert oq._sanitize_md_cell(None) == ""  # a true None renders empty
    # duplicate headers get a _2 suffix (upstream's dedup)
    dup = oq._dataframe_to_markdown(pd.DataFrame([[1, 2]], columns=["Year", "Year"]))
    assert dup.split("\n")[0] == "| Year | Year_2 |"


def test_html_tables_render_to_markdown(tmp_path):
    pytest.importorskip("lxml")
    path = make_json(
        tmp_path,
        "d.json",
        [
            (
                "<table><tr><th>Year</th><th>Rate</th></tr><tr><td>1941</td><td>0.096</td></tr></table>",
                1,
            )
        ],
    )
    pages = oq.extract_pages_from_json(path)
    assert "| Year | Rate |" in pages[1]
    assert "| 1941.0 | 0.096 |" in pages[1]  # read_html parses the year as a float


def test_html_tables_that_do_not_parse_fall_back_to_a_code_block(monkeypatch):
    class _BrokenPandas:
        @staticmethod
        def read_html(buf):
            raise ValueError("no table parsed")

    monkeypatch.setattr(oq, "_pandas", lambda: _BrokenPandas)
    html = "<table><tr><td>x</td></tr></table>"
    assert oq._html_tables_to_markdown(html) == ["```html\n" + html + "\n```"]


def test_missing_pandas_names_the_family_extra(monkeypatch):
    # None in sys.modules makes the real `import pandas` raise ImportError.
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(RuntimeError, match="granite-evals\\[officeqa\\]"):
        oq._flatten_columns([])


def test_corpus_page_reads_by_txt_name_and_refuses_unknown_pages(pinned_tree):
    corpus = oq.Corpus(pinned_tree)
    assert corpus.page("treasury_bulletin_1939_01.txt", 2) == "page two text\n"
    with pytest.raises(FileNotFoundError, match="Page 3 not found in"):
        corpus.page("treasury_bulletin_1939_01.txt", 3)
    with pytest.raises(FileNotFoundError, match="no parsed document"):
        corpus.page("treasury_bulletin_1999_99.txt", 1)


# -- questions ----------------------------------------------------------------


def test_load_questions_validates_the_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("uid,question\np1,what?\n")
    with pytest.raises(RuntimeError, match="missing column\\(s\\) answer"):
        oq.load_questions(path)


def test_question_rows_must_have_every_column():
    # A CSV round-trip stringifies everything, so feed the row dict directly.
    with pytest.raises(RuntimeError, match="'difficulty' column"):
        oq.Question({**row("p1"), "difficulty": None})


def test_parse_gold_pages_zips_urls_with_files():
    q = oq.Question(
        row(
            "p9",
            docs="https://x/a.html?page=1\r\nhttps://x/b.html?page=7",
            files="a.txt\nb.txt",
        )
    )
    assert oq.parse_gold_pages(q) == [("a.txt", 1), ("b.txt", 7)]


def test_parse_gold_pages_reuses_the_last_file_and_skips_pageless_urls():
    q = oq.Question(
        row("p9", docs="https://x/a.html?page=2\nhttps://x/no-page.html", files="a.txt")
    )
    assert oq.parse_gold_pages(q) == [("a.txt", 2)]


def test_parse_gold_pages_without_files_is_empty():
    q = oq.Question(row("p9", docs="https://x/a.html?page=1", files=""))
    assert oq.parse_gold_pages(q) == []


def test_oracle_context_joins_the_gold_pages(pinned_tree):
    q = oq.Question(PRO_ROWS[1])
    context, pages = oq.oracle_context(q, oq.Corpus(pinned_tree))
    assert pages == [
        ("treasury_bulletin_1939_01.txt", 1),
        ("treasury_bulletin_1941_02.txt", 2),
    ]
    assert (
        context == "page one text\n\n\n---\n\nother page two\n"
    )  # pages end in \n; the join adds \n\n---\n\n


def test_oracle_context_without_gold_pages_raises(pinned_tree):
    q = oq.Question(row("p9", docs="https://x/a.html", files="a.txt"))
    with pytest.raises(ValueError, match="No gold pages found for p9"):
        oq.oracle_context(q, oq.Corpus(pinned_tree))


def test_build_messages_is_upstreams_shape():
    q = oq.Question(PRO_ROWS[0])
    messages = oq.build_messages(q, "THE CONTEXT")
    assert messages[0] == {"role": "system", "content": oq.SYSTEM_PROMPT}
    assert messages[1] == {
        "role": "user",
        "content": "## Document Context\n\nTHE CONTEXT\n\n## Question\n\nWhat was the value of p1?",
    }
    assert "<FINAL_ANSWER>" in oq.SYSTEM_PROMPT


def test_safe_uid_keeps_the_record_path_safe():
    assert oq._safe_uid("pro/0001: x") == "pro_0001__x"
    assert oq._safe_uid("p1") == "p1"


# -- ModelClient ----------------------------------------------------------------


def _handler(records, responses):
    def handle(request):
        records.append(request)
        item = responses.pop(0)
        if isinstance(item, int):
            return httpx.Response(item, text="boom")
        if callable(item):
            return item(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": item},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 3},
            },
        )

    return handle


def test_complete_posts_upstreams_payload_and_reads_the_choice():
    requests, responses = [], ["<FINAL_ANSWER>42</FINAL_ANSWER>"]
    client = oq.ModelClient(
        "http://llm:8000/v1",
        "granite-4.2",
        timeout=5,
        transport=httpx.MockTransport(_handler(requests, responses)),
    )
    text, finish, usage = client.complete(
        [{"role": "user", "content": "q"}], max_tokens=128, temperature=0.2
    )
    assert (text, finish, usage) == (
        "<FINAL_ANSWER>42</FINAL_ANSWER>",
        "stop",
        {"prompt_tokens": 7, "completion_tokens": 3},
    )
    r = requests[0]
    assert str(r.url) == "http://llm:8000/v1/chat/completions"
    assert r.headers["Authorization"] == "Bearer EMPTY"
    assert json.loads(r.read()) == {
        "model": "granite-4.2",
        "messages": [{"role": "user", "content": "q"}],
        "max_tokens": 128,
        "temperature": 0.2,
    }


def test_complete_appends_the_chat_path_only_when_missing():
    client = oq.ModelClient(
        "http://llm:8000/v1/chat/completions",
        "m",
        timeout=5,
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]
                },
            )
        ),
    )
    assert client.url == "http://llm:8000/v1/chat/completions"
    client.complete([], max_tokens=1, temperature=0.0)
    client.close()


def test_complete_retries_5xx_then_succeeds(monkeypatch):
    monkeypatch.setattr(oq.time, "sleep", lambda s: None)
    requests, responses = [], [500, "<FINAL_ANSWER>ok</FINAL_ANSWER>"]
    client = oq.ModelClient(
        "http://llm/v1",
        "m",
        timeout=5,
        transport=httpx.MockTransport(_handler(requests, responses)),
    )
    assert (
        client.complete([], max_tokens=1, temperature=0.0)[0]
        == "<FINAL_ANSWER>ok</FINAL_ANSWER>"
    )
    assert len(requests) == 2


def test_complete_retries_429_then_succeeds(monkeypatch):
    monkeypatch.setattr(oq.time, "sleep", lambda s: None)
    requests, responses = [], [429, "<FINAL_ANSWER>ok</FINAL_ANSWER>"]
    client = oq.ModelClient(
        "http://llm/v1",
        "m",
        timeout=5,
        transport=httpx.MockTransport(_handler(requests, responses)),
    )
    assert (
        client.complete([], max_tokens=1, temperature=0.0)[0]
        == "<FINAL_ANSWER>ok</FINAL_ANSWER>"
    )
    assert len(requests) == 2


def test_complete_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(oq.time, "sleep", lambda s: None)
    requests, responses = [], [503] * oq.MAX_ATTEMPTS
    client = oq.ModelClient(
        "http://llm/v1",
        "m",
        timeout=5,
        transport=httpx.MockTransport(_handler(requests, responses)),
    )
    with pytest.raises(RuntimeError, match="HTTP 503"):
        client.complete([], max_tokens=1, temperature=0.0)
    assert len(requests) == oq.MAX_ATTEMPTS


def test_complete_times_out_then_raises(monkeypatch):
    monkeypatch.setattr(oq.time, "sleep", lambda s: None)
    n = [0]

    def timeout(request):
        n[0] += 1
        raise httpx.TimeoutException("slow")

    client = oq.ModelClient(
        "http://llm/v1", "m", timeout=5, transport=httpx.MockTransport(timeout)
    )
    with pytest.raises(RuntimeError, match="model call timed out"):
        client.complete([], max_tokens=1, temperature=0.0)
    assert n[0] == oq.MAX_ATTEMPTS


def test_a_hard_4xx_fails_immediately():
    requests, responses = [], [404]
    client = oq.ModelClient(
        "http://llm/v1",
        "m",
        timeout=5,
        transport=httpx.MockTransport(_handler(requests, responses)),
    )
    with pytest.raises(RuntimeError, match="HTTP 404"):
        client.complete([], max_tokens=1, temperature=0.0)
    assert len(requests) == 1


# -- run(): selection, aggregation, phases, resume ---------------------------------


def test_run_scores_a_model_run_end_to_end(tmp_path, fake_model):
    tree = make_tree(tmp_path / "data")
    FakeModel.answers = {"value of p1?": "2,602", "value of p2?": "9999"}
    out = make_bench(tmp_path, dataset=str(tree)).run(
        "http://llm:8000/v1", "granite-4.2"
    )

    assert out["value"] == pytest.approx(0.5)  # p1 exact, p2 wrong
    assert out["n"] == 2
    assert out["questions_failed"] == 0 and out["incomplete"] is False
    assert out["mode"] == "oracle_parsed" and out["answers"] == "model"
    assert out["accuracy"]["0%"] == pytest.approx(0.5)
    assert out["accuracy"]["1%"] == pytest.approx(0.5)
    assert out["accuracy_by_difficulty"] == {
        "easy": {"n": 1, "accuracy@0%": 1.0},
        "hard": {"n": 1, "accuracy@0%": 0.0},
    }
    assert out["pass_at_k"]["k"] == 1 and out["pass_at_k"][
        "pass_at_1"
    ] == pytest.approx(0.5)
    assert out["per_repeat"][0]["statuses"] == {"graded": 2}
    assert out["sampling"] == {"temperature": 0.0, "max_tokens": 4096}
    assert out["dataset"] == str(tree) and out["dataset_revision"] is None
    assert out["corpus"]["jsons_sha256"] is None  # a local dataset is trusted
    assert any("BM25-index" in d for d in out["deviations"])

    rec = json.loads((tmp_path / "out" / "repeat-0" / "p1.json").read_text())
    assert rec["status"] == "graded"
    assert rec["generation"] == "<FINAL_ANSWER>2,602</FINAL_ANSWER>"
    assert rec["pred"] == "2,602"
    assert rec["scores"]["0%"] == 1.0
    assert rec["pages"] == [
        ["treasury_bulletin_1939_01.txt", 1]
    ]  # json: tuples become lists
    assert rec["context_chars"] == len("page one text\n")
    assert rec["context_truncated"] is False
    assert rec["usage"] == {"prompt_tokens": 10}
    assert rec["difficulty"] == "easy" and rec["answer"] == "2,602"

    messages = FakeModel.instances[0].calls[0][0]
    assert messages[0]["content"] == oq.SYSTEM_PROMPT
    assert "page one text" in messages[1]["content"]
    assert "## Question" in messages[1]["content"]


def test_run_passes_the_sampling_options_through(tmp_path, fake_model):
    tree = make_tree(tmp_path / "data")
    FakeModel.answers = {"value of p1?": "2,602", "value of p2?": "1608.80%"}
    bench = make_bench(
        tmp_path,
        dataset=str(tree),
        options={"temperature": "0.6", "max_tokens": "512"},
    )
    out = bench.run("http://llm:8000/v1", "m")
    assert out["value"] == 1.0
    calls = FakeModel.instances[0].calls
    assert all(c[1] == 512 and c[2] == pytest.approx(0.6) for c in calls)


def test_run_truncates_the_context_when_asked(tmp_path, fake_model):
    tree = make_tree(tmp_path / "data")
    FakeModel.answers = {"value of p1?": "2,602", "value of p2?": "1608.80%"}
    bench = make_bench(tmp_path, dataset=str(tree), options={"max_context_chars": "5"})
    bench.run("http://llm:8000/v1", "m")
    rec = json.loads((tmp_path / "out" / "repeat-0" / "p1.json").read_text())
    assert rec["context_chars"] == len("page one text\n")
    assert rec["context_truncated"] is True
    user = FakeModel.instances[0].calls[0][0][1]["content"]
    assert user.startswith("## Document Context\n\npage ")  # cut at 5 chars
    assert "page one text" not in user


def test_run_scores_gold_answers_without_a_model(tmp_path):
    tree = make_tree(tmp_path / "data")
    bench = make_bench(tmp_path, dataset=str(tree), options={"answers": "gold"})
    out = bench.run("", "")  # no endpoint: gold never talks to a model
    assert FakeModel.instances == []
    assert out["value"] == 1.0  # every gold answer matches itself
    assert out["answers"] == "gold" and out["sampling"] == {}
    rec = json.loads((tmp_path / "out" / "repeat-0" / "p1.json").read_text())
    assert rec["answers"] == "gold"
    assert rec["generation"] == "2,602"  # the gold answer is the prediction
    assert rec["scores"]["0%"] == 1.0
    assert rec["pages"]  # the context was still built and recorded


def test_run_selects_by_regex_and_limit(tmp_path, fake_model):
    tree = make_tree(tmp_path / "data")
    FakeModel.answers = {"value of p1?": "2,602", "value of p2?": "1608.80%"}
    bench = make_bench(tmp_path, dataset=str(tree), options={"questions": "^p1$"})
    assert bench.run("http://llm:8000/v1", "m")["n"] == 1
    bench = make_bench(tmp_path, dataset=str(tree), limit=1)
    out = bench.run("http://llm:8000/v1", "m")
    assert out["n"] == 1
    assert out["questions"] == 1
    assert (
        json.loads((tmp_path / "out" / "repeat-0" / "p1.json").read_text())["uid"]
        == "p1"
    )


def test_run_sorts_questions_by_uid_for_comparable_smoke_runs(tmp_path):
    tree = make_tree(tmp_path / "data")
    make_csv(tree, oq.PRO_CSV, [row("z9"), row("a0"), *PRO_ROWS])
    bench = make_bench(tmp_path, dataset=str(tree), limit=2)
    questions = bench._select(tree)
    assert [q.uid for q in questions] == ["a0", "p1"]


def test_rerun_skips_finished_questions(tmp_path, fake_model):
    tree = make_tree(tmp_path / "data")
    FakeModel.answers = {"value of p1?": "2,602", "value of p2?": "1608.80%"}
    bench = make_bench(tmp_path, dataset=str(tree))
    first = bench.run("http://llm:8000/v1", "m")
    assert len(FakeModel.instances[0].calls) == 2
    FakeModel.instances = []
    FakeModel.answers = {}  # any new call would answer "?" and change the score
    second = make_bench(tmp_path, dataset=str(tree)).run("http://llm:8000/v1", "m")
    assert FakeModel.instances[0].calls == []  # every record short-circuits
    assert second["value"] == first["value"] == 1.0


def test_repeats_feed_pass_at_k(tmp_path, fake_model):
    tree = make_tree(tmp_path / "data")
    FakeModel.answers = {"value of p1?": "2,602", "value of p2?": "1608.80%"}
    bench = make_bench(tmp_path, dataset=str(tree), repeats=2)
    out = bench.run("http://llm:8000/v1", "m")
    assert out["pass_at_k"]["k"] == 2
    assert out["pass_at_k"]["pass_at_1"] == pytest.approx(1.0)
    assert out["pass_at_k"]["pass_at_k"] == pytest.approx(1.0)
    assert len(out["per_repeat"]) == 2
    assert (tmp_path / "out" / "repeat-1" / "p2.json").is_file()


def test_generate_phase_writes_records_and_scores_nothing(tmp_path, fake_model):
    tree = make_tree(tmp_path / "data")
    FakeModel.answers = {"value of p1?": "2,602", "value of p2?": "1608.80%"}
    out = make_bench(tmp_path, dataset=str(tree), phase="generate").run(
        "http://llm:8000/v1", "m"
    )
    assert out["n"] == 2 and "value" not in out
    assert out["per_repeat"] == [{"repeat": 0, "generated": 2, "failed": 0}]
    rec = json.loads((tmp_path / "out" / "repeat-0" / "p1.json").read_text())
    assert rec["status"] == oq.GENERATED and "scores" not in rec


def test_score_phase_scores_saved_generations_without_the_model(tmp_path, fake_model):
    tree = make_tree(tmp_path / "data")
    FakeModel.answers = {"value of p1?": "2,602", "value of p2?": "1608.80%"}
    make_bench(tmp_path, dataset=str(tree), phase="generate").run(
        "http://llm:8000/v1", "m"
    )
    FakeModel.instances = []
    out = make_bench(tmp_path, dataset=str(tree), phase="score").run("", "")
    assert FakeModel.instances == []  # the score phase never generates
    assert out["value"] == 1.0
    rec = json.loads((tmp_path / "out" / "repeat-0" / "p1.json").read_text())
    assert rec["status"] == "graded" and rec["scores"]["0%"] == 1.0


def test_score_phase_without_a_generation_records_not_generated(tmp_path, caplog):
    tree = make_tree(tmp_path / "data")
    bench = make_bench(tmp_path, dataset=str(tree), phase="score")
    with caplog.at_level(logging.WARNING):
        # Every question is unresolved: the failure policy fails the run (the
        # score phase never generates), leaving the records for generate.
        with pytest.raises(SystemExit, match="questions_failed 2/2"):
            bench.run("", "")
    assert "no generation for p1 repeat 0" in caplog.text
    assert not (tmp_path / "out" / "repeat-0" / "p1.json").exists()  # left for generate


def test_a_repeat_with_nothing_scored_fails_the_run(tmp_path):
    tree = make_tree(tmp_path / "data")
    bench = make_bench(
        tmp_path,
        dataset=str(tree),
        phase="score",
        options={"max_failed_frac": "1"},  # let the failures through: 2/2 == 1
    )
    with pytest.raises(SystemExit, match="no scored question"):
        bench.run("", "")


def test_infrastructure_failures_fail_the_run_over_the_allowed_fraction(
    tmp_path, fake_model
):
    tree = make_tree(tmp_path / "data")
    FakeModel.fail_substrings = ("value of",)
    bench = make_bench(tmp_path, dataset=str(tree))
    with pytest.raises(SystemExit, match="questions_failed 2/2"):
        bench.run("http://llm:8000/v1", "m")


def test_infrastructure_failures_within_the_fraction_are_reported(tmp_path, fake_model):
    tree = make_tree(tmp_path / "data")
    FakeModel.fail_substrings = ("value of p2?",)
    FakeModel.answers = {"value of p1?": "2,602"}
    bench = make_bench(tmp_path, dataset=str(tree), options={"max_failed_frac": "1"})
    out = bench.run("http://llm:8000/v1", "m")
    assert out["questions_failed"] == 1 and out["incomplete"] is True
    assert out["per_repeat"][0]["statuses"] == {"error:RuntimeError": 1, "graded": 1}
    assert out["value"] == pytest.approx(1.0)  # the failure is left out, not a loss
    assert not (tmp_path / "out" / "repeat-0" / "p2.json").exists()  # retried on rerun


def test_a_local_dataset_and_the_env_var_are_trusted(tmp_path, monkeypatch):
    tree = make_tree(tmp_path / "data")
    # Tamper after building: a trusted root is never digest-checked.
    (tree / oq.PRO_CSV).write_text("garbage,that,matches,no,columns,x\n")
    monkeypatch.delenv(oq.CACHE_ENV, raising=False)
    bench = make_bench(tmp_path, dataset=str(tree), options={"answers": "gold"})
    corpus, pinned = bench._corpus_root()
    assert pinned is False and corpus.root == tree
    monkeypatch.setenv(oq.CACHE_ENV, str(tree))
    bench = make_bench(tmp_path, options={"answers": "gold"})
    corpus, pinned = bench._corpus_root()
    assert pinned is False


def test_a_missing_local_dataset_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv(oq.CACHE_ENV, raising=False)
    bench = make_bench(tmp_path, dataset=str(tmp_path / "nope"))
    with pytest.raises(SystemExit, match="not a directory"):
        bench._corpus_root()
    monkeypatch.setenv(oq.CACHE_ENV, str(tmp_path / "nope"))
    with pytest.raises(SystemExit, match="not a directory"):
        make_bench(tmp_path)._corpus_root()


def test_run_fetches_only_the_csvs_in_the_score_phase(
    tmp_path, monkeypatch, pinned_tree, fake_model
):
    FakeModel.answers = {"value of p1?": "2,602", "value of p2?": "1608.80%"}
    fetched = []

    class PatchedCorpus(_REAL_CORPUS):
        def __init__(self, root=None, *, download=None):
            super().__init__(
                root,
                download=download
                or (lambda allow: fetched.append(tuple(allow)) or pinned_tree),
            )

    monkeypatch.setattr(oq, "Corpus", PatchedCorpus)
    make_bench(tmp_path, phase="generate").run("http://llm:8000/v1", "m")
    out = make_bench(tmp_path, phase="score").run("", "")
    # generate fetches the corpus; score re-scores saved generations, CSVs only
    assert fetched == [tuple(oq.ALLOW_PATTERNS), tuple(oq.CSV_ONLY_PATTERNS)]
    assert out["corpus"]["jsons_sha256"] is None  # the corpus was not verified
    assert out["corpus"]["root"] == str(pinned_tree)
    assert out["dataset"] == oq.DATASET and out["dataset_revision"] == oq.REVISION


def test_the_fetched_pin_is_recorded_as_verified(
    tmp_path, monkeypatch, pinned_tree, fake_model
):
    FakeModel.answers = {"value of p1?": "2,602", "value of p2?": "1608.80%"}

    class PatchedCorpus(_REAL_CORPUS):
        def __init__(self, root=None, *, download=None):
            super().__init__(root, download=download or (lambda allow: pinned_tree))

    monkeypatch.setattr(oq, "Corpus", PatchedCorpus)
    out = make_bench(tmp_path).run("http://llm:8000/v1", "m")
    assert out["corpus"]["jsons_sha256"] == oq.JSONS_SHA256
    assert out["value"] == 1.0


def test_a_fetch_that_does_not_match_the_pin_refuses_to_run(
    tmp_path, monkeypatch, pinned_tree
):
    (sorted((pinned_tree / oq.JSONS_DIR).glob("*.json"))[0]).write_text("{}")

    class PatchedCorpus(_REAL_CORPUS):
        def __init__(self, root=None, *, download=None):
            super().__init__(root, download=download or (lambda allow: pinned_tree))

    monkeypatch.setattr(oq, "Corpus", PatchedCorpus)
    with pytest.raises(RuntimeError, match="refusing to run"):
        make_bench(tmp_path).run("http://llm:8000/v1", "m")


# -- parity with the pinned upstream file -----------------------------------------


_UPSTREAM_PAIRS = [
    ("2,602", "2,602"),
    ("2,602", "2602"),
    ("1000", "1001"),
    ("1000", "1010"),
    ("1000", "1100"),
    ("1608.80%", "1608.79%"),
    ("[0.096, -184.143]", "[0.096, -184.14]"),
    ("$1,234.56", "(1,234.56)"),
    ("36080 million", "36,080"),
    ("January 1941", "February 1941"),
    ("0.096", "0.096%"),
    ("The reasoning is long.\n<FINAL_ANSWER>2,602</FINAL_ANSWER>", "2,602"),
]


@pytest.mark.skipif(
    not os.environ.get("OFFICEQA_UPSTREAM"), reason="OFFICEQA_UPSTREAM not set"
)
def test_port_matches_upstream():
    root = Path(os.environ["OFFICEQA_UPSTREAM"])
    for rel, digest in oq.UPSTREAM_FILES.items():
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == digest, rel
    ns = {}
    exec(compile((root / "reward.py").read_text(), str(root / "reward.py"), "exec"), ns)
    for gt, pred in _UPSTREAM_PAIRS:
        for t in oq.TOLERANCES:
            assert ns["score_answer"](gt, pred, t) == oq.score_answer(gt, pred, t), (
                gt,
                pred,
                t,
            )
        assert (
            ns["fuzzy_match_answer"](gt, pred)[0] == oq.fuzzy_match_answer(gt, pred)[0]
        ), (gt, pred)
    for text in ("2,602 with  1,234.56 and −7", "$1,234.56 (456.78) 0.096% 1941"):
        assert ns["normalize_text"](text) == oq.normalize_text(text), text
        assert ns["extract_numbers_with_context"](
            text
        ) == oq.extract_numbers_with_context(text), text
    for text in (
        "a <FINAL_ANSWER>x</FINAL_ANSWER> b",
        "<final_answer>one</final_answer> <FINAL_ANSWER>two</FINAL_ANSWER>",
        "no tags",
    ):
        assert ns["extract_final_answer_from_xml"](
            text
        ) == oq.extract_final_answer_from_xml(text), text
