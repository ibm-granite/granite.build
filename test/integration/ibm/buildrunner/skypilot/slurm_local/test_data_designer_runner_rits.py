# Copyright LLM.build Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The data-designer step's runner against RITS, off-cluster.

Runs the step's ``src/dd_run.py`` in this process with a two-row config against RITS
granite-4.1-8b: credential resolution from the environment into the ``RITS_API_KEY``
header, Data Designer's validate + generate, the jsonl export, and the artifact marker
as the last line. The fastest check that the runner and the endpoint work, before
spending a cluster build on it.

**IBM-only** (``pytest.mark.ibm``): RITS is IBM-internal. Self-skips unless
``RITS_API_KEY`` is exported and the ``data-designer`` package is importable. It is
deliberately not in the repo venv: installing it there would replace the repo's pinned
duckdb, fsspec, pyarrow and opentelemetry. Run this file with a separate venv that has
``data-designer`` plus pytest, e.g. ``<venv>/bin/python -m pytest -c /dev/null <this file>``.
It imports nothing from gbserver, so the repo venv is not needed.
"""

import base64
import importlib.util
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.ibm

_RUNNER = (
    Path(__file__).resolve().parents[6]
    / "steps/data/data-designer/skypilot/src/dd_run.py"
)


def _load_runner():
    """Import the step's runner by path; it ships in the step, not as a package."""
    spec = importlib.util.spec_from_file_location("dd_run", _RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(
    not (
        os.environ.get("RITS_API_KEY")
        and importlib.util.find_spec("data_designer") is not None
    ),
    reason="needs RITS_API_KEY and the data-designer package",
)
def test_runner_generates_against_rits(tmp_path, monkeypatch, capsys):
    """Two rows from RITS granite-4.1-8b through the runner, end to end."""
    assert _RUNNER.is_file(), _RUNNER
    dd_run = _load_runner()
    monkeypatch.chdir(tmp_path)
    spec = {
        "model_providers": [
            {
                "name": "rits",
                "endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-4-1-8b/v1",
                "extra_headers_from_env": {"RITS_API_KEY": "RITS_API_KEY"},
            }
        ],
        "config": {
            "model_configs": [
                {
                    "alias": "text",
                    "model": "ibm-granite/granite-4.1-8b",
                    "provider": "rits",
                    "inference_parameters": {
                        "max_tokens": 64,
                        "max_parallel_requests": 2,
                    },
                }
            ],
            "columns": [
                {
                    "name": "topic",
                    "column_type": "sampler",
                    "sampler_type": "category",
                    "params": {"values": ["geology"]},
                },
                {
                    "name": "question",
                    "column_type": "llm-text",
                    "model_alias": "text",
                    "prompt": "Write one short trivia question about {{ topic }}.",
                },
            ],
        },
    }
    monkeypatch.setenv(
        dd_run.SPEC_ENV, base64.b64encode(json.dumps(spec).encode()).decode()
    )
    out = tmp_path / "out"
    rc = dd_run.main(
        [
            "--output-path",
            str(out),
            "--artifact-id",
            "questions",
            "--num-records",
            "2",
            "--resume",
            "never",
            "--min-records",
            "2",
            "--export-format",
            "jsonl",
        ]
    )
    stdout = capsys.readouterr().out
    assert rc == 0
    assert (
        stdout.rstrip().splitlines()[-1]
        == f"GB_ARTIFACT_ID:questions GB_ARTIFACT_PATH:{out}"
    )
    assert "GB_STEP_METADATA_KEY:resumed_records GB_STEP_METADATA_VALUE:0" in stdout
    rows = [json.loads(line) for line in (out / "out.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(r["question"] for r in rows)
    key = os.environ["RITS_API_KEY"]
    assert not [
        p for p in out.rglob("*") if p.is_file() and key in p.read_text(errors="ignore")
    ]
