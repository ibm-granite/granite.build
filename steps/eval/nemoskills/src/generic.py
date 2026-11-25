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
  jsonl file), read directly -- ns's prepare does not run;
* a hub id with ``dataset_revision=<40-hex commit>``: the HF dataset ns's
  ``prepare.py`` reads, pinned to that commit (the base class's
  ``hf_revisions`` mechanism: its ``load_dataset``/``hf_hub_download``/
  ``snapshot_download`` reads get the revision);
* an ``http(s)`` URL with ``dataset_revision=sha256:<hex>``: the file ns's
  ``prepare.py`` downloads, pinned by content -- served to its ``urlretrieve``
  (the base class's ``pinned_urls``) and to its ``wget`` calls (a shim on the
  prepare's PATH: the images ship no wget, and a wget fetch is unpinned).
  ``recipes/evals/nemoskills/ns-math-500`` pins ns's math-500 download so.

ns's prepare runs with its HF reads, ``urlretrieve`` and ``wget`` checked: one
the recipe does not pin (another HF repo, another URL) fails the prepare,
naming it. There is no unpinned mode. A prepare that fetches some other way
(``requests``, ``git clone`` -- a few ns benchmarks do) is not intercepted:
give such a benchmark a local ``dataset=``. The run's provenance records each
checked read and its pin (``reads``, ``pinned_by``).

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
WGET_READS = "granite-wget-reads.jsonl"

# wget as ns prepare.py scripts call it (``wget [-q ...] -O <file> <url>``),
# served from the recipe's pinned URLs only. Written with the job's own
# interpreter as its shebang, so httpx (an image dependency) is importable.
WGET_SHIM = '''\
"""wget for ns prepare.py, granite's nemoskills bridge: serves only the URLs
the recipe pins (GRANITE_NS_PINNED_URLS: {url: [fetch url, sha256]}), each
checked against its sha256, and logs every read (GRANITE_NS_WGET_READS)."""
import hashlib, json, os, sys
from pathlib import Path

import httpx

args, out, urls = sys.argv[1:], None, []
i = 0
while i < len(args):
    a = args[i]
    if a in ("-O", "--output-document"):
        out, i = args[i + 1], i + 2
        continue
    if a.startswith("--output-document="):
        out = a.split("=", 1)[1]
    elif not a.startswith("-"):
        urls.append(a)
    i += 1
if len(urls) != 1:
    sys.exit(f"granite wget shim: expected one URL, got {urls}")
url = urls[0]
pins = json.loads(os.environ.get("GRANITE_NS_PINNED_URLS", "{}"))
if url not in pins:
    sys.exit(
        f"granite: ns prepare.py runs wget {url}, which the recipe does not pin "
        "(dataset=<that url> dataset_revision=sha256:<hex>)"
    )
fetch, sha = pins[url]
r = httpx.get(fetch, follow_redirects=True, timeout=300)
r.raise_for_status()
digest = hashlib.sha256(r.content).hexdigest()
if digest != sha:
    sys.exit(f"granite: {fetch}: sha256 {digest} != pinned {sha}")
Path(out or url.rsplit("/", 1)[-1]).write_bytes(r.content)
with open(os.environ["GRANITE_NS_WGET_READS"], "a") as f:
    f.write(json.dumps({"url": url, "pinned_url": fetch, "sha256": sha, "via": "wget"}) + "\\n")
'''


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

    def split(self) -> str:
        return self.opt("split", "") or super().split()

    def ns_data_dir(self) -> Path:
        """Where the base class runs ns's prepare (its ``ddir``)."""
        return self.config.output_dir / "ns-data" / self.ns_benchmark.replace(".", "/")

    @contextlib.contextmanager
    def wget_shim(self) -> Iterator[None]:
        """A ``wget`` on PATH for the prepare subprocess, serving only the
        pinned URLs (none in the local and hub forms: any wget fails, naming
        its URL)."""
        with tempfile.TemporaryDirectory(prefix="granite-wget-") as bindir:
            shim = Path(bindir) / "wget"
            shim.write_text(f"#!{sys.executable}\n{WGET_SHIM}")
            shim.chmod(0o755)
            env = {
                "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
                "GRANITE_NS_PINNED_URLS": json.dumps(
                    {url: list(pin) for url, pin in self.pinned_urls.items()}
                ),
                "GRANITE_NS_WGET_READS": str(self.ns_data_dir() / WGET_READS),
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

    def prepare_data(self) -> tuple[Path, dict[str, Any]]:
        if self.data_form == "local":
            return super().prepare_data()
        with self.wget_shim():
            path, prov = super().prepare_data()
        log_path = self.ns_data_dir() / WGET_READS
        if log_path.exists():
            prov["reads"] = [
                *prov["reads"],
                *(
                    json.loads(line)
                    for line in log_path.read_text().splitlines()
                    if line
                ),
            ]
        prov["pinned_by"] = {
            "dataset": self.config.dataset,
            "dataset_revision": self.config.dataset_revision,
        }
        return path, prov
