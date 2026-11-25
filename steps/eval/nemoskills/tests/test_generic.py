# SPDX-License-Identifier: Apache-2.0

"""nemoskills generic-bridge tests: fakes only (no GPU, no network, no ns;
the wget shim is exercised against a localhost server)."""

import contextlib
import hashlib
import http.server
import json
import subprocess
import threading
from pathlib import Path

import pytest
from granite_evals.benchmarks import generic
from granite_evals.benchmarks.nemo_skills import NS_COMMIT, NS_REPO, NemoSkillsBenchmark
from granite_evals.registry import RunConfig, get

COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _config(tmp_path, dataset="", dataset_revision="", **options):
    return RunConfig(
        model="m",
        output_dir=tmp_path,
        dataset=dataset,
        dataset_revision=dataset_revision,
        options=options,
    )


def _bench(tmp_path, dataset="org/data", dataset_revision=COMMIT, **options):
    options.setdefault("benchmark", "math-500")
    return generic.Nemoskills(_config(tmp_path, dataset, dataset_revision, **options))


def test_benchmark_is_required(tmp_path):
    with pytest.raises(SystemExit, match="benchmark= is required"):
        generic.Nemoskills(_config(tmp_path))


def test_ns_benchmark_and_metric_option(tmp_path):
    bench = _bench(tmp_path)
    assert bench.ns_benchmark == "math-500"
    assert bench.ns_metric == "symbolic_correct"
    bench = _bench(tmp_path, benchmark="ifbench", metric="if")
    assert bench.ns_metric == "if"


def test_dataset_is_required(tmp_path):
    """No unpinned mode: without dataset= the bridge refuses to start."""
    with pytest.raises(SystemExit, match="dataset= is required"):
        generic.Nemoskills(_config(tmp_path, benchmark="math-500"))


def test_dataset_forms(tmp_path):
    data = tmp_path / "prepared.jsonl"
    data.write_text("{}\n")
    sha = "sha256:" + "a" * 64
    assert generic.dataset_form(str(data), "") == "local"
    assert generic.dataset_form("org/data", COMMIT) == "hub"
    assert generic.dataset_form("https://example.org/test.jsonl", sha) == "url"
    with pytest.raises(SystemExit, match="40-hex commit"):
        generic.dataset_form("org/data", "")
    with pytest.raises(SystemExit, match="40-hex commit"):
        generic.dataset_form("org/data", "main")  # a branch floats
    with pytest.raises(SystemExit, match="sha256:<64 hex>"):
        generic.dataset_form("https://example.org/test.jsonl", COMMIT)
    with pytest.raises(SystemExit, match="no existing path"):
        generic.dataset_form(str(tmp_path / "missing.jsonl"), "")
    with pytest.raises(SystemExit, match="no existing path"):
        generic.dataset_form("not a hub id", COMMIT)


def test_url_form_pins_urlretrieve(tmp_path):
    """The URL form feeds the base class's pinned_urls (prepare.py's
    urlretrieve) with the content pin."""
    url = "https://example.org/test.jsonl"
    bench = _bench(tmp_path, dataset=url, dataset_revision="sha256:" + "b" * 64)
    assert bench.pinned_urls == {url: (url, "b" * 64)}
    assert _bench(tmp_path).pinned_urls == {}  # the hub form pins no URL


def test_local_dataset(tmp_path, monkeypatch):
    data = tmp_path / "prepared.jsonl"
    data.write_text(json.dumps({"problem": "1+1", "expected_answer": "2"}) + "\n")
    # split() reads the ns module (not installed here); the local-path branch
    # only needs the name it composes.
    monkeypatch.setattr(NemoSkillsBenchmark, "split", lambda self: "test")
    bench = _bench(tmp_path, dataset=str(data), dataset_revision="")
    prepared, prov = bench.prepare_data()
    assert prepared == data and prov["source"] == "local"


def test_split_option(tmp_path, monkeypatch):
    """split= picks the prepared <split>.jsonl; unset, the base class's."""
    monkeypatch.setattr(NemoSkillsBenchmark, "split", lambda self: "test")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "validation.jsonl").write_text("{}\n")
    bench = _bench(tmp_path, dataset=str(tmp_path / "data"), dataset_revision="")
    assert bench.split() == "test"
    bench = _bench(
        tmp_path,
        dataset=str(tmp_path / "data"),
        dataset_revision="",
        split="validation",
    )
    assert bench.split() == "validation"
    prepared, _ = bench.prepare_data()
    assert prepared == tmp_path / "data" / "validation.jsonl"


@contextlib.contextmanager
def _serve(root: Path):
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(  # noqa: E731
        *a, directory=str(root), **k
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


def test_wget_shim_serves_the_pinned_url(tmp_path, monkeypatch):
    """ns's math-500 prepare shells out to wget (absent from the images).
    The shim on the prepare's PATH fetches the pinned URL, checks its
    sha256, logs the read; an unpinned URL or a changed file fails."""
    served = tmp_path / "served"
    served.mkdir()
    body = b'{"problem": "1+1", "answer": "2"}\n'
    (served / "test.jsonl").write_bytes(body)
    sha = hashlib.sha256(body).hexdigest()
    with _serve(served) as base:
        url = f"{base}/test.jsonl"

        def fake_prepare(self):
            """The base prepare, reduced to what matters here: prepare.py
            runs in a subprocess, inside ddir, calling wget."""
            ddir = self.ns_data_dir()
            ddir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["wget", "-q", "--no-check-certificate", "-O", str(ddir / "raw"), url],
                check=True,
            )
            return ddir / "raw", {"reads": []}

        monkeypatch.setattr(NemoSkillsBenchmark, "prepare_data", fake_prepare)
        bench = _bench(tmp_path / "run", dataset=url, dataset_revision=f"sha256:{sha}")
        path, prov = bench.prepare_data()
        assert path.read_bytes() == body
        assert prov["reads"] == [
            {"url": url, "pinned_url": url, "sha256": sha, "via": "wget"}
        ]
        assert prov["pinned_by"] == {
            "dataset": url,
            "dataset_revision": f"sha256:{sha}",
        }

        # The content pin: a changed file fails the prepare.
        bench = _bench(
            tmp_path / "run2", dataset=url, dataset_revision="sha256:" + "0" * 64
        )
        with pytest.raises(subprocess.CalledProcessError):
            bench.prepare_data()

        # The hub form pins no URL: any wget fails, naming the URL.
        bench = _bench(tmp_path / "run3")
        with pytest.raises(subprocess.CalledProcessError):
            bench.prepare_data()


def test_pins():
    assert generic.Nemoskills.dataset == NS_REPO
    assert generic.Nemoskills.dataset_revision == NS_COMMIT


def test_registry():
    assert get("nemoskills") is generic.Nemoskills
