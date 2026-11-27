# SPDX-License-Identifier: Apache-2.0

"""nemoskills generic-bridge tests: fakes only (no GPU, no network, no ns;
the wget shim is exercised against a localhost server)."""

import contextlib
import hashlib
import http.server
import json
import os
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


def test_a_local_dataset_revision_is_checked_or_refused(tmp_path):
    import hashlib

    data = tmp_path / "prepared.jsonl"
    data.write_text("{}\n")
    sha = "sha256:" + hashlib.sha256(data.read_bytes()).hexdigest()
    assert generic.dataset_form(str(data), sha) == "local"  # the file's own sha256
    with pytest.raises(SystemExit, match="has sha256 .* not the dataset_revision"):
        generic.dataset_form(str(data), "sha256:" + "0" * 64)
    with pytest.raises(SystemExit, match="local file, and dataset_revision=.*cannot be checked"):
        generic.dataset_form(str(data), COMMIT)  # a commit means nothing for a file
    with pytest.raises(SystemExit, match="local directory, and dataset_revision=.*cannot be checked"):
        generic.dataset_form(str(tmp_path), sha)


@pytest.mark.parametrize(
    ("statuses", "options", "fails"),
    [
        ({"stop": 97, "error": 3}, {}, False),
        ({"stop": 95, "error": 5}, {}, False),  # exactly 0.05 is allowed
        ({"stop": 94, "error": 6}, {}, True),
        ({"stop": 90, "error": 10}, {"max_failed_frac": "0.2"}, False),
        ({"stop": 99, "error": 1}, {"max_failed_frac": "0"}, True),
        ({"stop": 10}, {"max_failed_frac": "0"}, False),
        ({"error": 2}, {}, True),
    ],
)
def test_the_generation_failure_threshold(tmp_path, monkeypatch, statuses, options, fails):
    """ns scores a soft-failed request as an empty, wrong answer: over
    max_failed_frac of all generations the run fails instead."""
    monkeypatch.setattr(NemoSkillsBenchmark, "run", lambda self, b, s: {"value": 0.5, "statuses": dict(statuses)})
    bench = _bench(tmp_path, **options)
    if fails:
        with pytest.raises(SystemExit, match="generations_failed .* above the allowed fraction"):
            bench.run("http://s/v1", "m")
        return
    outcome = bench.run("http://s/v1", "m")
    assert outcome["generations_failed"] == statuses.get("error", 0)
    assert outcome["generations_total"] == sum(statuses.values())


@pytest.mark.parametrize(("value", "match"), [("lots", "a number in"), ("1.5", r"in \[0, 1\]"), ("-1", r"in \[0, 1\]")])
def test_a_bad_max_failed_frac_is_refused_at_startup(tmp_path, value, match):
    with pytest.raises(SystemExit, match=match):
        _bench(tmp_path, max_failed_frac=value)
    assert generic.MAX_FAILED_FRAC == 0.05
    assert _bench(tmp_path).max_failed_frac == 0.05


@pytest.mark.parametrize("dataset", ["~/data", "~user/data", "./org/data", "/org/data"])
def test_a_missing_path_is_never_a_hub_id(tmp_path, dataset):
    # A path the user meant (it starts with ~, . or /) must exist; it never
    # falls through to the hub form (HUB_ID starts with a letter or digit).
    with pytest.raises(SystemExit, match="no existing path, hub id or URL"):
        generic.dataset_form(dataset, COMMIT)


def test_an_http_error_is_named_not_a_digest_mismatch(tmp_path, monkeypatch, served):
    url, body, sha = served
    missing = url.rsplit("/", 1)[0] + "/missing.jsonl"
    monkeypatch.setenv("GRANITE_NS_PINNED_URLS", json.dumps({missing: [missing, sha]}))
    monkeypatch.setenv("GRANITE_NS_READS", str(tmp_path / "reads.jsonl"))
    with pytest.raises(generic.guard["GuardError"], match="HTTP 404"):
        generic.guard["fetch_pinned"](missing)


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


def _with_prepare(monkeypatch, run, python_flags=()):
    """The base prepare, reduced to what matters here: prepare.py runs in a
    Python subprocess (inheriting the guards' env, so the sitecustomize loads
    and writes its marker), inside ddir; ``run(ddir)`` is what it does.
    Returns its recorded reads (none: the guards log theirs)."""

    def fake_prepare(self):
        ddir = self.ns_data_dir()
        ddir.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, *python_flags, "-c", "pass"], check=True)
        extra = run(ddir) or []
        return ddir / "raw", {"reads": list(extra)}

    monkeypatch.setattr(NemoSkillsBenchmark, "prepare_data", fake_prepare)


def test_a_prepare_without_the_guards_is_refused(tmp_path, monkeypatch, served):
    """Fail closed: a prepare that ran without the sitecustomize (python -I
    drops PYTHONPATH) left no marker, so its data is refused even though the
    pin was read; with the guards, the marker is not reported as a read."""
    url, _, sha = served
    read = [{"url": url}]
    _with_prepare(monkeypatch, lambda d: read, python_flags=("-I",))
    with pytest.raises(SystemExit, match="ran without the bridge's guards"):
        _bench(
            tmp_path / "a", dataset=url, dataset_revision=f"sha256:{sha}"
        ).prepare_data()
    _with_prepare(monkeypatch, lambda d: read)
    _, prov = _bench(
        tmp_path / "b", dataset=url, dataset_revision=f"sha256:{sha}"
    ).prepare_data()
    assert prov["reads"] == read


@pytest.mark.parametrize(
    "header",
    [
        f"Authorization Bearer {SECRET}",  # no colon: httpx/h11 reject it, quoting it
        f"Authorization: Bearer {SECRET}\r\nX-Injected: 1",  # a CR/LF in the value
        f"Authorization: Bearer {SECRET}\rmore",  # a bare CR inside the value
    ],
)
def test_wget_shim_bad_header_never_prints_the_token(
    tmp_path, monkeypatch, served, header
):
    """The shim runs under python -I (no sitecustomize): an unexpected error
    from the HTTP client prints only its type, never its message."""
    url, _, sha = served
    out = {}

    def run(ddir):
        out["proc"] = subprocess.run(
            ["wget", "--header", header, url, "-O", "raw"],
            cwd=ddir,
            capture_output=True,
            text=True,
        )
        return [{"url": url}]

    _with_prepare(monkeypatch, run)
    _bench(
        tmp_path / "run", dataset=url, dataset_revision=f"sha256:{sha}"
    ).prepare_data()
    proc = out["proc"]
    assert proc.returncode != 0
    assert "granite: wget failed (" in proc.stderr
    assert SECRET not in proc.stderr + proc.stdout


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


MMAU_PREPARE = (
    "import os, subprocess\n"
    "token = os.environ['HF_TOKEN']\n"
    "subprocess.run(['wget', '--header', f'Authorization: Bearer {token}',\n"
    "    'https://h.example/data.zip', '-O', 'data.zip'], check=True)\n"
)


def test_a_failed_prepare_never_prints_the_token(tmp_path, monkeypatch, served):
    """mmau-pro's shape end to end: the shim refuses the unpinned URL, and the
    prepare's own CalledProcessError (whose message is the wget argv, token
    included) is printed through the guards' redacting excepthook."""
    url, _, sha = served
    out = {}

    def run(ddir):
        out["proc"] = subprocess.run(
            [sys.executable, "-c", MMAU_PREPARE],
            cwd=ddir,
            capture_output=True,
            text=True,
            env={**os.environ, "HF_TOKEN": SECRET},
        )
        return [{"url": url}]

    _with_prepare(monkeypatch, run)
    _bench(
        tmp_path / "run", dataset=url, dataset_revision=f"sha256:{sha}"
    ).prepare_data()
    proc = out["proc"]
    assert proc.returncode != 0
    assert "CalledProcessError" in proc.stderr and "<redacted>" in proc.stderr
    assert SECRET not in proc.stderr + proc.stdout


def test_redact(monkeypatch):
    redact = generic.guard["redact"]
    monkeypatch.setenv("SOME_API_KEY", "k-0123456789")
    text = "['wget', '--header', 'Authorization: Bearer abc.def', 'x'] k-0123456789 token=bearer zzz"
    assert redact(text) == (
        "['wget', '--header', 'Authorization: Bearer <redacted>', 'x'] <redacted> "
        "token=bearer <redacted>"
    )


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
        ((None, "hf://datasets/a/b"), {}),  # data_dir, positional (load_dataset's 3rd)
    ]:
        with pytest.raises(SystemExit, match="not local") as err:
            check("json", args, kwargs)
        assert "t=1" not in str(err.value)


@pytest.mark.parametrize("builder", sorted(generic.guard["LOCAL_BUILDERS"]))
def test_every_local_builder_refuses_remote_data(builder):
    assert generic.guard["LOCAL_BUILDERS"] == {"json", "csv", "parquet", "text", "arrow", "webdataset"}
    with pytest.raises(SystemExit, match="not local"):
        generic.guard["check_local_builder"](builder, (), {"data_files": "https://h/x"})


def test_urlopen_is_guarded_before_datasets_is_imported(tmp_path, monkeypatch):
    """datasets may fetch at import time: the urlopen guard is in place
    before install_guards imports it."""
    pkg = tmp_path / "site" / "datasets"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(
        "import urllib.request\n"
        "SEEN = urllib.request.urlopen.__name__\n"
        "def load_dataset(path, *a, **k):\n"
        "    return 'ok'\n"
    )
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    monkeypatch.syspath_prepend(str(tmp_path / "site"))
    monkeypatch.setenv("GRANITE_NS_READS", "/dev/null")
    import runpy
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", urllib.request.urlopen)
    monkeypatch.setattr(runpy, "run_path", runpy.run_path)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    generic.guard["install_guards"]()
    assert sys.modules["datasets"].SEEN == "guarded_urlopen"


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
    import runpy

    monkeypatch.setattr(runpy, "run_path", runpy.run_path)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    generic.guard["install_guards"]()
    assert fake.load_dataset("json", data_files="/x.jsonl") == "ok"
    with pytest.raises(SystemExit, match="not local"):
        fake.load_dataset("json", data_files="hf://datasets/a/b.zip")
    assert calls == ["json"]
    assert urllib.request.urlopen is generic.guard["guarded_urlopen"]
    assert sys.excepthook is generic.guard["redacting_excepthook"]
    # _prepare_main runs prepare.py through runpy.run_path: now the guarded one.
    assert runpy.run_path is not ORIGINAL_RUN_PATH
    assert "guarded_run_path" in runpy.run_path.__qualname__


ORIGINAL_RUN_PATH = __import__("runpy").run_path


def test_guarded_run_path_redacts_a_system_exit(monkeypatch):
    """The base class's refusals are SystemExits whose message Python prints
    itself (no excepthook): the wrapper re-raises them with URLs cut and
    secrets redacted; other exits pass through unchanged."""
    monkeypatch.setenv("HF_TOKEN", SECRET)

    def refuses(*a, **k):
        raise SystemExit(
            f"prepare.py fetches https://h.example/x.csv?token={SECRET}, which the "
            f"benchmark does not pin; token {SECRET}"
        )

    with pytest.raises(SystemExit) as err:
        generic.guard["guarded_run_path"](refuses)("prepare.py")
    assert str(err.value.code) == (
        "prepare.py fetches https://h.example/x.csv, which the benchmark does not pin; "
        "token <redacted>"
    )

    def exits(*a, **k):
        raise SystemExit(3)

    with pytest.raises(SystemExit) as err:
        generic.guard["guarded_run_path"](exits)("prepare.py")
    assert err.value.code == 3


def test_redacting_excepthook_cuts_urls(capsys):
    try:
        raise RuntimeError(f"GET https://h.example/data?token={SECRET} failed")
    except RuntimeError as e:
        generic.guard["redacting_excepthook"](type(e), e, e.__traceback__)
    err = capsys.readouterr().err
    assert "GET https://h.example/data failed" in err and SECRET not in err


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


def test_redact_authorization_without_bearer(monkeypatch):
    redact = generic.guard["redact"]
    assert redact("-H 'Authorization: Basic dXNlcjpwYXNzd29yZA=='") == (
        "-H 'Authorization: Basic <redacted>'"
    )
    assert redact("authorization: tok123456") == "authorization: <redacted>"


@pytest.mark.parametrize("value", ["", "short"])
def test_an_empty_or_short_secret_redacts_nothing(monkeypatch, value):
    # HF_TOKEN= (set but empty): replacing "" would put <redacted> between
    # every character.
    redact = generic.guard["redact"]
    monkeypatch.setenv("HF_TOKEN", value)
    assert redact("prepare.py failed: exit 1") == "prepare.py failed: exit 1"


def test_an_empty_token_header_leaves_the_message_alone(monkeypatch):
    # mmau-pro's argv with HF_TOKEN=: the header has no value, so nothing is
    # redacted -- not the scheme word, not the next line's first word.
    redact = generic.guard["redact"]
    monkeypatch.setenv("HF_TOKEN", "")
    text = "Authorization: Bearer \nresult 42 written\n"
    assert redact(text) == text
    assert redact("['-H', 'Authorization: Bearer ', 'https://x']") == (
        "['-H', 'Authorization: Bearer ', 'https://x']"
    )
    # a real value on the header line is still redacted, the next line kept
    assert redact("Authorization: Bearer abc123\nnext") == "Authorization: Bearer <redacted>\nnext"


def test_redact_urls():
    redact_urls = generic.guard["redact_urls"]
    text = f"fetches https://u:{SECRET}@h.example/x.csv?token={SECRET}#f, or http://h/y"
    assert redact_urls(text) == "fetches https://h.example/x.csv, or http://h/y"


NS_PREPARE_URLRETRIEVE = (
    "import urllib.request\n"
    f"urllib.request.urlretrieve('https://h.example/x.csv?token={SECRET}', 'raw')\n"
)


def test_a_base_refusal_prints_no_url_query(tmp_path, monkeypatch, capfd):
    """The real base prepare (_prepare_main) on an ns prepare.py whose
    urlretrieve URL the recipe does not pin: the base class refuses it with
    the full URL, query and token included; through the bridge's guards the
    message shows scheme://host/path only."""
    pytest.importorskip("datasets")
    pytest.importorskip("huggingface_hub")
    mod = tmp_path / "nsmod"
    mod.mkdir()
    (mod / "__init__.py").write_text("EVAL_SPLIT = 'test'\n")
    (mod / "prepare.py").write_text(NS_PREPARE_URLRETRIEVE)
    monkeypatch.setattr(
        generic.Nemoskills,
        "ns_module",
        lambda self: types.SimpleNamespace(__file__=str(mod / "__init__.py")),
    )
    bench = _bench(
        tmp_path / "run",
        dataset="https://h.example/pinned.csv",
        dataset_revision="sha256:" + "0" * 64,
    )
    with pytest.raises(subprocess.CalledProcessError):
        bench.prepare_data()
    err = capfd.readouterr().err
    assert (
        "prepare.py fetches https://h.example/x.csv, which the benchmark does not pin"
        in err
    )
    assert SECRET not in err


def test_the_hub_pin_needs_its_revision(tmp_path, monkeypatch, served):
    """The hub form's pin check compares the revision too: a read of the
    pinned repo at another revision is not the pin."""
    other = [{"hf_dataset": "org/data", "revision": "f" * 40}]
    _with_prepare(monkeypatch, lambda d: other)
    with pytest.raises(SystemExit, match="never read the pinned"):
        _bench(tmp_path / "a").prepare_data()


URLOPEN_REQUEST = (
    "import sys, urllib.request\n"
    "req = urllib.request.Request(sys.argv[1], headers={'Authorization': sys.argv[2]})\n"
    "sys.stdout.write(urllib.request.urlopen(req).read().decode())\n"
)


def test_urlopen_forwards_request_headers(tmp_path, monkeypatch, served):
    url, body, sha = served
    _Handler.seen_auth.clear()
    out = {}

    def run(ddir):
        out["proc"] = subprocess.run(
            [sys.executable, "-c", URLOPEN_REQUEST, url, f"Bearer {SECRET}"],
            capture_output=True,
            text=True,
        )

    _with_prepare(monkeypatch, run)
    _bench(
        tmp_path / "run", dataset=url, dataset_revision=f"sha256:{sha}"
    ).prepare_data()
    assert out["proc"].returncode == 0 and out["proc"].stdout == body.decode()
    assert _Handler.seen_auth == [f"Bearer {SECRET}"]


def test_wget_to_stdout(tmp_path, monkeypatch, served):
    url, body, sha = served
    out = {}

    def run(ddir):
        out["proc"] = subprocess.run(
            ["wget", "-q", "-O", "-", url], capture_output=True
        )

    _with_prepare(monkeypatch, run)
    _bench(
        tmp_path / "run", dataset=url, dataset_revision=f"sha256:{sha}"
    ).prepare_data()
    assert out["proc"].returncode == 0 and out["proc"].stdout == body


def test_the_guards_leave_the_environment_as_it_was(tmp_path, monkeypatch, served):
    url, _, sha = served
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", "") or "/keep")
    monkeypatch.delenv("GRANITE_NS_READS", raising=False)
    before = {
        k: os.environ.get(k)
        for k in ("PATH", "PYTHONPATH", "GRANITE_NS_READS", "GRANITE_NS_PINNED_URLS")
    }
    inside = {}

    def run(ddir):
        inside.update({k: os.environ.get(k) for k in before})
        return [{"url": url}]

    _with_prepare(monkeypatch, run)
    _bench(
        tmp_path / "run", dataset=url, dataset_revision=f"sha256:{sha}"
    ).prepare_data()
    assert inside["GRANITE_NS_READS"] and inside["PATH"] != before["PATH"]
    assert {k: os.environ.get(k) for k in before} == before
