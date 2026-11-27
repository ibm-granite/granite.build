# SPDX-License-Identifier: Apache-2.0

"""nemoskills family: the generic bridge to any benchmark in the pinned NeMo-Skills.

One registered id (``nemoskills``) for every benchmark the pinned ns checkout
ships: ``benchmark=<ns name>`` (required, e.g. ``benchmark=math-500`` -- the
directory names under ns's ``nemo_skills/dataset``) runs it with **ns's own
defaults** -- its prepare, its prompts, its metrics -- where the curated
classes freeze Granite-specific protocol choices on top of the same machinery.
The curated ids stay as they are: they are the validated protocols.

Data: ``dataset=`` is required, and pins the data in one of three forms:

* a local path: ns's prepared rows (a dir holding ``<split>.jsonl``, or one
  jsonl file), read directly -- ns's prepare does not run. A
  ``dataset_revision`` with it must be ``sha256:<hex>`` of the file, and is
  checked; any other (a commit, or any revision on a directory) is refused,
  since nothing could check it;
* a hub id with ``dataset_revision=<40-hex commit>``: the HF dataset ns's
  ``prepare.py`` reads, pinned to that commit (the base class's
  ``hf_revisions`` mechanism);
* an ``http(s)`` URL with ``dataset_revision=sha256:<hex>``: the file ns's
  ``prepare.py`` downloads, pinned by content.
  ``recipes/evals/nemoskills/ns-math-500`` pins ns's math-500 download so.

What ns's prepare is checked on, exactly. In the prepare subprocess the base
class pins ``datasets.load_dataset`` on hub ids, ``huggingface_hub``'s
``hf_hub_download`` and ``snapshot_download``, and
``urllib.request.urlretrieve``; this bridge adds ``urllib.request.urlopen``
(served only for the pinned URL), a ``wget`` command on the prepare's PATH
(the images ship none; served only for the pinned URL, its options parsed and
its header values forwarded, never printed), and a refusal of remote
``data_files``/``data_dir`` (``hf://``, ``https://``, ...) given to the local
``load_dataset`` builders (``json``, ``csv``, ``parquet``, ...). A checked read
the recipe does not pin fails the prepare, naming only its http(s) URL; the
base class's own refusals, which print the full URL, are re-raised with every
URL cut to scheme://host/path. An uncaught error in prepare.py (a failed
``subprocess.run(..., check=True)`` prints its argv) is printed with the
values of secret-looking environment variables, of Authorization/Bearer
headers and of URL queries redacted. After the prepare the
run fails unless the recorded reads include the recipe's pin, and unless the
guards' sitecustomize recorded that it loaded in the prepare (fail closed). The
wget shim prints no traceback: an unexpected error is reported by type only.

Not intercepted: ``requests``, ``httpx``, ``aiohttp``, ``fsspec`` reads
(``hf://``, ``https://`` paths opened directly), ``huggingface_hub``'s
``HfApi().hf_hub_download`` (a method, not the module function the base pins)
and ``HfFileSystem``, an opener from ``urllib.request.build_opener`` (its
``open`` bypasses ``urlopen``), the ``wget`` Python package, ``git``,
``curl`` and any other subprocess, remote ``data_files`` given to a pinned
hub id (``load_dataset(<pinned repo>, data_files="https://...")``: the
revision pin covers the repo, not the URL), and remote data given to a local
builder other than through ``datasets.load_dataset`` (``load_dataset_builder``,
``Dataset.from_json`` and the like with a URL). A prepare that fetches that
way still runs, so it can read data beyond the pin (the pin check above
still demands the pinned data was read too): give such a benchmark a local
``dataset=``. The
run's provenance records each checked read (``reads``) and the pin
(``pinned_by``).

Generation failures: a request that kept failing is an ns soft failure
(``finish_reason: error``), which ns scores as an empty, wrong answer (ns
computes the metrics, so the bridge cannot leave it out). Over
``max_failed_frac`` (default 0.05, in [0, 1]) of all generations the run
fails instead of reporting that score; ``generations_failed`` and
``generations_total`` record the counts.

``metric=`` overrides the headline metric key (the class default, ns's
``symbolic_correct``). ``split=`` picks which prepared ``<split>.jsonl`` the
run reads (default: the ns module's ``EVAL_SPLIT``, else ``test``) -- in a
local dir, or among what ns's prepare writes with its default arguments.
Everything else -- the sampling options, the judge options, the two phases,
resume -- is the base class's own surface. See ``nemo_skills.py`` for the
protocol.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterator

from granite_evals.benchmarks.nemo_skills import NS_COMMIT, NS_REPO, NemoSkillsBenchmark
from granite_evals.registry import register

HUB_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^sha256:([0-9a-f]{64})$")
GUARD_READS = "granite-guard-reads.jsonl"

# The prepare guards, one source for both places they run: written as the
# prepare's ``wget`` (run with ``-I``, so this sitecustomize is not loaded
# there) and as a ``sitecustomize.py`` on the prepare subprocess's PYTHONPATH
# (loaded before ``_prepare_main`` wraps ``datasets.load_dataset``, so the
# base wrapper's local-builder path calls through this one). Also exec'd
# below, so the tests call the same functions. Messages name option names and
# http(s) URLs without their query, never argv values or header values.
GUARD_LIB = r'''
import hashlib
import io
import json
import os
import sys
import urllib.parse
from pathlib import Path

WGET_FLAGS = {
    "-q", "--quiet", "-nv", "--no-verbose", "-v", "--verbose",
    "--no-check-certificate", "--content-disposition", "-c", "--continue",
    "-nc", "--no-clobber", "-N", "--timestamping", "--show-progress",
}
# value-taking wget options -> what the shim does with the value (None: ignored)
WGET_VALUE = {
    "-O": "out", "--output-document": "out",
    "-P": "prefix", "--directory-prefix": "prefix",
    "--header": "header",
    "-U": None, "--user-agent": None,
    "-o": None, "--output-file": None, "-a": None, "--append-output": None,
    "-t": None, "--tries": None, "-T": None, "--timeout": None,
    "--progress": None, "--limit-rate": None,
}
LOCAL_BUILDERS = {"json", "csv", "parquet", "text", "arrow", "webdataset"}
GUARD_MARKER = "granite_guards"


class GuardError(Exception):
    """A refused read."""


def safe_url(url):
    """An http(s) URL as scheme://host/path (no userinfo, query or fragment);
    anything else only by its kind."""
    try:
        parts = urllib.parse.urlsplit(str(url))
    except ValueError:
        return "a malformed URL"
    if parts.scheme not in ("http", "https"):
        return "a non-http(s) URL"
    return f"{parts.scheme}://{parts.hostname or ''}{parts.path}"


def safe_remote(value):
    """A remote data_files entry as scheme://netloc/path (no query)."""
    parts = urllib.parse.urlsplit(str(value))
    return f"{parts.scheme}://{parts.hostname or ''}{parts.path}"


def parse_wget_args(argv):
    """(url, -O file or None, -P prefix or None, --header values)."""
    urls, out, prefix, headers = [], None, None, []
    i = 0
    while i < len(argv):
        arg = argv[i]
        i += 1
        if arg == "--":
            urls += argv[i:]
            break
        if not arg.startswith("-") or arg == "-":
            urls.append(arg)
            continue
        if arg in WGET_FLAGS:
            continue
        if arg.startswith("--"):
            name, eq, value = arg.partition("=")
        elif arg[:2] in WGET_VALUE and len(arg) > 2:  # -Ofile
            name, eq, value = arg[:2], "=", arg[2:]
        else:
            name, eq, value = arg, "", ""
        if name not in WGET_VALUE:
            raise GuardError(f"wget option {name.split('=', 1)[0]} is not supported here")
        if not eq:
            if i >= len(argv):
                raise GuardError(f"wget option {name} needs a value")
            value = argv[i]
            i += 1
        role = WGET_VALUE[name]
        if role == "out":
            out = value
        elif role == "prefix":
            prefix = value
        elif role == "header":
            headers.append(value)
    if any(not str(u).startswith(("http://", "https://")) for u in urls):
        raise GuardError("wget got an argument that is neither an option nor an http(s) URL")
    if len(urls) != 1:
        raise GuardError(
            f"wget expects one URL here, got {len(urls)}: " + ", ".join(safe_url(u) for u in urls)
        )
    return urls[0], out, prefix, headers


def fetch_pinned(url, headers=(), via="wget"):
    """The pinned URL's bytes, sha256-checked and logged; GuardError for any
    other URL. ``headers`` ("Name: value") are sent, never logged."""
    pins = json.loads(os.environ.get("GRANITE_NS_PINNED_URLS", "{}"))
    if url not in pins:
        raise GuardError(
            f"ns prepare.py fetches {safe_url(url)} ({via}), which the recipe does not pin "
            "(dataset=<that url> dataset_revision=sha256:<hex>)"
        )
    fetch, sha = pins[url]
    import httpx

    sent = {}
    for header in headers:
        key, _, value = header.partition(":")
        sent[key.strip()] = value.strip()
    response = httpx.get(fetch, headers=sent, follow_redirects=True, timeout=300)
    if response.status_code >= 400:
        raise GuardError(f"{safe_url(fetch)}: HTTP {response.status_code}")
    digest = hashlib.sha256(response.content).hexdigest()
    if digest != sha:
        raise GuardError(f"{safe_url(fetch)}: sha256 {digest} != pinned {sha}")
    with open(os.environ["GRANITE_NS_READS"], "a") as f:
        f.write(json.dumps({"url": url, "pinned_url": fetch, "sha256": sha, "via": via}) + "\n")
    return response.content


def wget_main(argv):
    # The shim runs under python -I, without the sitecustomize: redact here
    # too, and print no traceback at all for an unexpected error (an httpx
    # error on a malformed --header quotes the header, token included).
    sys.excepthook = redacting_excepthook
    try:
        url, out, prefix, headers = parse_wget_args(argv)
        body = fetch_pinned(url, headers, "wget")
        if out == "-":
            sys.stdout.buffer.write(body)
            return
        name = urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1] or "index.html"
        Path(out or Path(prefix or ".") / name).write_bytes(body)
    except GuardError as e:
        print(f"granite: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:  # noqa: BLE001 - its message may carry a header value
        print(f"granite: wget failed ({type(e).__name__})", file=sys.stderr)
        sys.exit(1)


class _Response(io.BytesIO):
    """What urlopen callers use of a response: read, context manager,
    status/getcode, geturl."""

    def __init__(self, body, url):
        super().__init__(body)
        self.status, self.url, self.headers = 200, url, {}

    def getcode(self):
        return self.status

    def geturl(self):
        return self.url

    def info(self):
        return self.headers


def _flat(value):
    if value is None:
        return []
    if isinstance(value, (str, os.PathLike)):
        return [value]
    if isinstance(value, dict):
        return [x for v in value.values() for x in _flat(v)]
    return [x for v in value for x in _flat(v)]


def guarded_urlopen(url, *args, **kwargs):
    target = str(getattr(url, "full_url", url))
    headers = [f"{k}: {v}" for k, v in getattr(url, "header_items", lambda: [])()]
    try:
        return _Response(fetch_pinned(target, headers, "urlopen"), target)
    except GuardError as e:
        raise SystemExit(f"granite: {e}") from None


def check_local_builder(path, args, kwargs):
    """SystemExit if a local builder gets remote data_files or data_dir."""
    if path not in LOCAL_BUILDERS:
        return
    data_dir = kwargs.get("data_dir", args[1] if len(args) > 1 else None)
    data_files = kwargs.get("data_files", args[2] if len(args) > 2 else None)
    for entry in _flat(data_files) + _flat(data_dir):
        if "://" in str(entry):
            raise SystemExit(
                f"granite: ns prepare.py loads {path} data from {safe_remote(entry)}, which "
                "is not local; the bridge reads remote data only through the recipe's pin"
            )


SECRET_ENV_WORDS = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")


def redact(text):
    """``text`` without the values of secret-looking environment variables
    (HF_TOKEN, *_API_KEY, ...) and without Authorization/Bearer values: a
    failed ``subprocess.run(..., check=True)`` in prepare.py prints its argv,
    and mmau-pro's carries ``Authorization: Bearer $HF_TOKEN``."""
    import re

    for key, value in os.environ.items():
        if len(value) >= 8 and any(w in key.upper() for w in SECRET_ENV_WORDS):
            text = text.replace(value, "<redacted>")
    text = re.sub(r"(?i)(authorization:\s*)(\w+\s+)?[^\s'\",\]]+", r"\1\2<redacted>", text)
    return re.sub(r"(?i)(bearer\s+)[^\s'\",\]]+", r"\1<redacted>", text)


def redact_urls(text):
    """Every http(s) URL in ``text`` as scheme://host/path: a query string or
    userinfo may carry a token. A URL ends at whitespace, a quote, <, > or a
    comma (the base class's messages put one right after the URL)."""
    import re

    return re.sub(r"https?://[^\s'\",<>]+", lambda m: safe_url(m.group(0)), text)


def redacting_excepthook(kind, value, tb):
    import traceback

    text = "".join(traceback.format_exception(kind, value, tb))
    sys.stderr.write(redact(redact_urls(text)))


def guarded_run_path(original):
    """runpy.run_path, which _prepare_main runs prepare.py with: a SystemExit
    message (the base class's refusals print the full unpinned URL, query
    included) is re-raised with its URLs and secrets redacted. Python prints
    a SystemExit's message itself, bypassing sys.excepthook."""

    def run_path(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except SystemExit as e:
            if isinstance(e.code, str):
                raise SystemExit(redact(redact_urls(e.code))) from None
            raise

    return run_path


def install_guards():
    if "GRANITE_NS_READS" not in os.environ:
        return
    import urllib.request

    import runpy

    sys.excepthook = redacting_excepthook
    runpy.run_path = guarded_run_path(runpy.run_path)
    urllib.request.urlopen = guarded_urlopen
    try:
        import datasets
    except ImportError:
        datasets = None
    if datasets is not None:
        original = datasets.load_dataset

        def load_dataset(path, *args, **kwargs):
            check_local_builder(path, args, kwargs)
            return original(path, *args, **kwargs)

        datasets.load_dataset = load_dataset
    # The marker the bridge requires after the prepare (fail closed: a
    # prepare run without this sitecustomize, e.g. under python -I, would
    # have none of these guards).
    with open(os.environ["GRANITE_NS_READS"], "a") as f:
        f.write(json.dumps({GUARD_MARKER: "loaded", "pid": os.getpid()}) + "\n")
'''
guard: dict[str, Any] = {"__name__": "granite_ns_guard"}
exec(
    compile(GUARD_LIB, "granite_ns_guard", "exec"), guard
)  # noqa: S102 - our own source


def check_local_pin(path: Path, revision: str) -> None:
    """A local dataset= with a dataset_revision: checked where it can be (a
    file's sha256), refused where it cannot -- a pin that is never checked
    reads as one that was."""
    match = SHA256.match(revision)
    if match and path.is_file():
        import hashlib

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != match.group(1):
            raise SystemExit(
                f"nemoskills: dataset={path} has sha256 {digest}, not the "
                f"dataset_revision={revision} the recipe pins"
            )
        return
    raise SystemExit(
        f"nemoskills: dataset={path} is a local "
        f"{'file' if path.is_file() else 'directory'}, and dataset_revision={revision} cannot be "
        "checked against it (only sha256:<64 hex> of a local file can); drop dataset_revision, "
        "or pin the file's sha256"
    )


MAX_FAILED_FRAC = 0.05


def max_failed_frac(value: Any) -> float:
    """``max_failed_frac=`` (default 0.05), a fraction in [0, 1], or a
    SystemExit naming it."""
    if value is None:
        return MAX_FAILED_FRAC
    try:
        frac = float(str(value))
    except ValueError:
        raise SystemExit(f"nemoskills: max_failed_frac must be a number in [0, 1], got {value!r}") from None
    if not 0 <= frac <= 1:
        raise SystemExit(f"nemoskills: max_failed_frac must be in [0, 1], got {frac}")
    return frac


def dataset_form(dataset: str, revision: str) -> str:
    """Which pinned form ``dataset=`` is (``local``, ``hub``, ``url``), or
    SystemExit naming what is missing."""
    if not dataset:
        raise SystemExit(
            "nemoskills: dataset= is required, pinning the data: a local path to ns's "
            "prepared rows, a hub id with dataset_revision=<40-hex commit> (the dataset "
            "ns's prepare.py reads), or an http(s) URL with dataset_revision=sha256:<hex> "
            "(the file ns's prepare.py downloads)"
        )
    if dataset.startswith(("http://", "https://")):
        if not SHA256.match(revision):
            raise SystemExit(
                f"nemoskills: dataset={dataset} is a URL: pin its content with "
                "dataset_revision=sha256:<64 hex>"
            )
        return "url"
    if Path(dataset).exists():
        if revision:
            check_local_pin(Path(dataset), revision)
        return "local"
    if dataset.startswith(("/", ".", "~")) or not HUB_ID.match(dataset):
        raise SystemExit(
            f"nemoskills: dataset={dataset} is no existing path, hub id or URL"
        )
    if not COMMIT.match(revision):
        raise SystemExit(
            f"nemoskills: dataset={dataset} is a hub id: pin it with "
            "dataset_revision=<40-hex commit>"
        )
    return "hub"


@register
class Nemoskills(NemoSkillsBenchmark):
    id = "nemoskills"
    metric = "ns's own headline metric for the benchmark"

    # The pin every benchmark must carry: the ns checkout (the benchmark
    # configs, prompts and evaluators); the data is pinned by the recipe's
    # required dataset= / dataset_revision=.
    dataset = NS_REPO
    dataset_revision = NS_COMMIT

    def __init__(self, config):
        bench = config.options.get("benchmark", "").strip()
        if not bench:
            raise SystemExit(
                "nemoskills: benchmark= is required (an ns benchmark name, e.g. "
                "benchmark=math-500); the curated ids are the validated protocols"
            )
        self.data_form = dataset_form(
            config.dataset or "", config.dataset_revision or ""
        )
        super().__init__(config)
        self.ns_benchmark = bench
        if "metric" in config.options:
            self.ns_metric = config.options["metric"]
        if self.data_form == "url":
            sha = SHA256.match(config.dataset_revision).group(1)
            # The base class serves prepare.py's urlretrieve from this.
            self.pinned_urls = {config.dataset: (config.dataset, sha)}
        self.max_failed_frac = max_failed_frac(config.options.get("max_failed_frac"))

    def run(self, base_url: str, served_model_name: str) -> dict[str, Any]:
        """The base class's run, then the generation-failure threshold: a
        request that kept failing is an ns soft failure (``finish_reason:
        error``), which ns scores as an empty, wrong answer. Over
        ``max_failed_frac`` of all generations the run fails instead of
        reporting that score."""
        outcome = super().run(base_url, served_model_name)
        statuses = outcome.get("statuses") or {}
        failed, total = statuses.get("error", 0), sum(statuses.values())
        outcome["generations_failed"], outcome["generations_total"] = failed, total
        if failed and (not total or failed / total > self.max_failed_frac):
            raise SystemExit(
                f"nemoskills: generations_failed {failed}/{total} is above the allowed fraction "
                f"{self.max_failed_frac}: ns scores a failed request as an empty, wrong answer, and "
                "it resumes filled rows, so fix the server and rerun into a fresh run dir"
            )
        return outcome

    def split(self) -> str:
        return self.opt("split", "") or super().split()

    def ns_data_dir(self) -> Path:
        """Where the base class runs ns's prepare (its ``ddir``)."""
        return self.config.output_dir / "ns-data" / self.ns_benchmark.replace(".", "/")

    @contextlib.contextmanager
    def prepare_guards(self) -> Iterator[None]:
        """The guards (``GUARD_LIB``) in the prepare subprocess's environment:
        a ``wget`` on PATH, a ``sitecustomize`` on PYTHONPATH, the pinned URLs
        and the reads log they write."""
        with tempfile.TemporaryDirectory(prefix="granite-ns-guard-") as gdir:
            shim = Path(gdir) / "wget"
            shim.write_text(
                f"#!{sys.executable} -I\n{GUARD_LIB}\nwget_main(sys.argv[1:])\n"
            )
            shim.chmod(0o755)
            (Path(gdir) / "sitecustomize.py").write_text(
                f"{GUARD_LIB}\ninstall_guards()\n"
            )

            def prepend(var: str) -> str:
                rest = os.environ.get(var, "")
                return f"{gdir}{os.pathsep}{rest}" if rest else gdir

            env = {
                "PATH": prepend("PATH"),
                "PYTHONPATH": prepend("PYTHONPATH"),
                "GRANITE_NS_PINNED_URLS": json.dumps(
                    {url: list(pin) for url, pin in self.pinned_urls.items()}
                ),
                "GRANITE_NS_READS": str(self.ns_data_dir() / GUARD_READS),
            }
            saved = {key: os.environ.get(key) for key in env}
            os.environ.update(env)
            try:
                yield
            finally:
                for key, value in saved.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def read_the_pin(self, reads: list[dict]) -> bool:
        """Whether the recorded reads include the recipe's pin."""
        dataset, revision = self.config.dataset, self.config.dataset_revision
        if self.data_form == "hub":
            return any(
                (r.get("hf_dataset") or r.get("hf_repo")) == dataset
                and r.get("revision") == revision
                for r in reads
            )
        return any(r.get("url") == dataset for r in reads)

    def prepare_data(self) -> tuple[Path, dict[str, Any]]:
        if self.data_form == "local":
            return super().prepare_data()
        with self.prepare_guards():
            path, prov = super().prepare_data()
        log_path = self.ns_data_dir() / GUARD_READS
        logged = [
            json.loads(line)
            for line in (log_path.read_text() if log_path.exists() else "").splitlines()
            if line
        ]
        marker = guard["GUARD_MARKER"]
        # Fail closed: the urlopen/data_files guards and the redaction exist
        # only where the sitecustomize loaded; a prepare run without it (a
        # future python -I in the base class) is refused, not trusted.
        if not any(marker in entry for entry in logged):
            raise SystemExit(
                f"nemoskills: ns's {self.ns_benchmark} prepare ran without the bridge's "
                "guards (no sitecustomize marker in its reads log); refusing its data"
            )
        prov["reads"] = [
            *prov["reads"],
            *(entry for entry in logged if marker not in entry),
        ]
        if not self.read_the_pin(prov["reads"]):
            raise SystemExit(
                f"nemoskills: ns's {self.ns_benchmark} prepare never read the pinned "
                f"dataset={self.config.dataset}; its data came from somewhere the bridge "
                "does not check (give it a local dataset=)"
            )
        prov["pinned_by"] = {
            "dataset": self.config.dataset,
            "dataset_revision": self.config.dataset_revision,
        }
        return path, prov
