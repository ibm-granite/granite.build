# SPDX-License-Identifier: Apache-2.0

"""nemoskills generic-bridge tests: fakes only (no GPU, no network, no ns;
the wget shim is exercised against a localhost server)."""

import contextlib
import hashlib
import http.server
import json
import subprocess
import sys
import threading
import types
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


SECRET = "hf_SECRETTOKEN0123456789"


class _Handler(http.server.SimpleHTTPRequestHandler):
    seen_auth: list = []

    def do_GET(self):  # noqa: N802 - http.server's name
        _Handler.seen_auth.append(self.headers.get("Authorization"))
        super().do_GET()

    def log_message(self, *args):
        pass


@contextlib.contextmanager
def _serve(root: Path):
    handler = lambda *a, **k: _Handler(*a, directory=str(root), **k)  # noqa: E731
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()


@pytest.fixture
def served(tmp_path):
    root = tmp_path / "served"
    root.mkdir()
    body = b'{"problem": "1+1", "answer": "2"}\n'
    (root / "test.jsonl").write_bytes(body)
    with _serve(root) as base:
        yield f"{base}/test.jsonl", body, hashlib.sha256(body).hexdigest()


def _with_prepare(monkeypatch, run):
    """The base prepare, reduced to what matters here: prepare.py runs in a
    subprocess (inheriting the guards' env), inside ddir; ``run(ddir)`` is
    what it does. Returns its recorded reads (none: the guards log theirs)."""

    def fake_prepare(self):
        ddir = self.ns_data_dir()
        ddir.mkdir(parents=True, exist_ok=True)
        extra = run(ddir) or []
        return ddir / "raw", {"reads": list(extra)}

    monkeypatch.setattr(NemoSkillsBenchmark, "prepare_data", fake_prepare)


def test_wget_shim_serves_the_pinned_url(tmp_path, monkeypatch, served):
    """ns's math-500 prepare shells out to wget (absent from the images).
    The shim on the prepare's PATH fetches the pinned URL, checks its
    sha256, logs the read; an unpinned URL or a changed file fails."""
    url, body, sha = served
    cmd = [
        "wget",
        "-q",
        "--no-check-certificate",
        "--content-disposition",
        "-O",
        "raw",
        url,
    ]
    _with_prepare(
        monkeypatch, lambda d: subprocess.run(cmd, cwd=d, check=True) and None
    )
    bench = _bench(tmp_path / "run", dataset=url, dataset_revision=f"sha256:{sha}")
    path, prov = bench.prepare_data()
    assert path.read_bytes() == body
    assert prov["reads"] == [
        {"url": url, "pinned_url": url, "sha256": sha, "via": "wget"}
    ]
    assert prov["pinned_by"] == {"dataset": url, "dataset_revision": f"sha256:{sha}"}

    # The content pin: a changed file fails the prepare.
    bench = _bench(
        tmp_path / "run2", dataset=url, dataset_revision="sha256:" + "0" * 64
    )
    with pytest.raises(subprocess.CalledProcessError):
        bench.prepare_data()

    # The hub form pins no URL: any wget fails.
    bench = _bench(tmp_path / "run3")
    with pytest.raises(subprocess.CalledProcessError):
        bench.prepare_data()


@pytest.mark.parametrize(
    ("argv", "parsed"),
    [
        (["-q", "-O", "f", "https://h/x"], ("https://h/x", "f", None, [])),
        (["-Of", "https://h/x"], ("https://h/x", "f", None, [])),
        (["--output-document=f", "-nv", "https://h/x"], ("https://h/x", "f", None, [])),
        (["-P", "d", "https://h/x"], ("https://h/x", None, "d", [])),
        (
            ["--header", "Authorization: Bearer t", "https://h/x", "-O", "f"],
            ("https://h/x", "f", None, ["Authorization: Bearer t"]),
        ),
        (
            ["--header=A: b", "-U", "ua", "-t", "3", "https://h/x"],
            ("https://h/x", None, None, ["A: b"]),
        ),
    ],
)
def test_parse_wget_args(argv, parsed):
    assert generic.guard["parse_wget_args"](argv) == parsed


@pytest.mark.parametrize(
    "argv",
    [
        ["--bogus", "https://h/x"],
        ["-x", "https://h/x"],
        ["-O"],
        ["notaurl"],
        ["https://h/x", "https://h/y"],
        [],
    ],
)
def test_parse_wget_args_refuses(argv):
    with pytest.raises(generic.guard["GuardError"]):
        generic.guard["parse_wget_args"](argv)


@pytest.mark.parametrize(
    "argv",
    [
        # mmau-pro's shape, on a URL the recipe does not pin
        [
            "--header",
            f"Authorization: Bearer {SECRET}",
            "https://h/data.zip",
            "-O",
            "d.zip",
        ],
        [f"--header=Authorization: Bearer {SECRET}", "--bogus", "https://h/x"],
        [f"--bogus={SECRET}", "https://h/x"],
        [SECRET, "https://h/x"],
        [f"https://user:{SECRET}@h/x?token={SECRET}"],
    ],
)
def test_wget_shim_never_prints_a_secret(tmp_path, monkeypatch, served, argv):
    """A refusal names option names and http(s) URLs (no userinfo, no
    query), never argv values or header values."""
    url, _, sha = served
    out = {}

    def run(ddir):
        proc = subprocess.run(["wget", *argv], cwd=ddir, capture_output=True, text=True)
        out["proc"] = proc
        return [
            {"url": url}
        ]  # satisfy the pin check; the shim's verdict is what counts

    _with_prepare(monkeypatch, run)
    _bench(
        tmp_path / "run", dataset=url, dataset_revision=f"sha256:{sha}"
    ).prepare_data()
    proc = out["proc"]
    assert proc.returncode != 0
    assert "granite:" in proc.stderr
    assert SECRET not in proc.stderr + proc.stdout


def test_wget_shim_forwards_headers(tmp_path, monkeypatch, served):
    url, _, sha = served
    _Handler.seen_auth.clear()
    cmd = ["wget", "--header", f"Authorization: Bearer {SECRET}", url, "-O", "raw"]
    _with_prepare(
        monkeypatch, lambda d: subprocess.run(cmd, cwd=d, check=True) and None
    )
    _, prov = _bench(
        tmp_path / "run", dataset=url, dataset_revision=f"sha256:{sha}"
    ).prepare_data()
    assert _Handler.seen_auth == [f"Bearer {SECRET}"]
    assert SECRET not in json.dumps(prov)


URLOPEN = (
    "import sys, urllib.request\n"
    "sys.stdout.write(urllib.request.urlopen(sys.argv[1]).read().decode())\n"
)


def test_urlopen_is_guarded(tmp_path, monkeypatch, served):
    """In the prepare subprocess urlopen serves the pinned URL only (the base
    class patches urlretrieve, not urlopen)."""
    url, body, sha = served
    out = {}

    def run(ddir):
        for target in (
            url,
            url + "?token=" + SECRET + "&x=1",
            url.replace("test", "other"),
        ):
            out[target] = subprocess.run(
                [sys.executable, "-c", URLOPEN, target], capture_output=True, text=True
            )

    _with_prepare(monkeypatch, run)
    _, prov = _bench(
        tmp_path / "run", dataset=url, dataset_revision=f"sha256:{sha}"
    ).prepare_data()
    pinned, with_query, other = out.values()
    assert pinned.returncode == 0 and pinned.stdout == body.decode()
    assert {"url": url, "pinned_url": url, "sha256": sha, "via": "urlopen"} in prov[
        "reads"
    ]
    for proc in (with_query, other):
        assert proc.returncode != 0 and "does not pin" in proc.stderr
        assert SECRET not in proc.stderr


def test_local_builders_refuse_remote_data_files():
    check = generic.guard["check_local_builder"]
    check("json", (), {"data_files": "/data/x.jsonl"})  # local: fine
    check(
        "org/repo", (), {"data_files": "hf://datasets/a/b.zip"}
    )  # a hub id: the base pins it
    for args, kwargs in [
        (
            (),
            {"data_files": "hf://datasets/Steefano/LCB/LongCodeQA.zip"},
        ),  # longcodebench
        ((), {"data_files": {"test": ["a.json", "https://h/b.json?t=1"]}}),
        ((None, None, "s3://bucket/x.csv"), {}),
        ((), {"data_dir": "hf://datasets/a/b"}),
    ]:
        with pytest.raises(SystemExit, match="not local") as err:
            check("json", args, kwargs)
        assert "t=1" not in str(err.value)


def test_install_guards_wraps_load_dataset(monkeypatch):
    """The sitecustomize half: with the guard env set, datasets.load_dataset
    refuses remote data_files for a local builder before calling through."""
    calls = []
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda path, *a, **k: calls.append(path) or "ok"
    monkeypatch.setitem(sys.modules, "datasets", fake)
    monkeypatch.setenv("GRANITE_NS_READS", "/dev/null")
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", urllib.request.urlopen)
    generic.guard["install_guards"]()
    assert fake.load_dataset("json", data_files="/x.jsonl") == "ok"
    with pytest.raises(SystemExit, match="not local"):
        fake.load_dataset("json", data_files="hf://datasets/a/b.zip")
    assert calls == ["json"]
    assert urllib.request.urlopen is generic.guard["guarded_urlopen"]


def test_the_pin_must_be_read(tmp_path, monkeypatch, served):
    """A prepare whose checked reads never include the recipe's pin (its
    data came another way) fails, however it succeeded."""
    url, _, sha = served
    _with_prepare(monkeypatch, lambda d: None)
    with pytest.raises(SystemExit, match="never read the pinned"):
        _bench(
            tmp_path / "a", dataset=url, dataset_revision=f"sha256:{sha}"
        ).prepare_data()
    # The hub form: the base class's recorded HF reads.
    read = [{"hf_dataset": "org/data", "revision": COMMIT, "name": None}]
    _with_prepare(monkeypatch, lambda d: read)
    _, prov = _bench(tmp_path / "b").prepare_data()
    assert prov["reads"] == read
    other = [{"hf_dataset": "org/other", "revision": COMMIT}]
    _with_prepare(monkeypatch, lambda d: other)
    with pytest.raises(SystemExit, match="never read the pinned"):
        _bench(tmp_path / "c").prepare_data()


def test_pins():
    assert generic.Nemoskills.dataset == NS_REPO
    assert generic.Nemoskills.dataset_revision == NS_COMMIT


def test_registry():
    assert get("nemoskills") is generic.Nemoskills
