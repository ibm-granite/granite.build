#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the eval/livecodebench-score test fixture.

The fixture is a tiny stand-in for the ``livecodebench/code_generation_lite``
dataset, in the same format: a loading script and a JSONL file whose rows carry
LiveCodeBench's fields, with the private tests encoded the way LiveCodeBench reads
them (base64 of zlib of a pickled JSON string). Plus a matching completions
directory, as eval/livecodebench-generate would write it.

The problems are written for this fixture, not taken from the real dataset, so the
fixture carries no third-party content. Each completion is chosen so the expected
score is exact:

==========  ========  ==========  ============================  ===========
problem     type      difficulty  samples                       pass@1
==========  ========  ==========  ============================  ===========
fx_sum      stdin     easy        correct, correct              1.0
fx_max      stdin     easy        correct, wrong                0.5
fx_rev      stdin     medium      wrong, no code (cut off)      0.0
fx_square   function  medium      correct, times out            0.5
==========  ========  ==========  ============================  ===========

Overall pass@1 = (1.0 + 0.5 + 0.0 + 0.5) / 4 = 0.5; easy 0.75, medium 0.25.

Run it to (re)write the fixture files next to this script, and the local SLURM
build test's ``build.yaml``: ``python make_fixture.py``. The output is
deterministic, and a unit test checks that the committed files match it.

The build test needs the fixture ON the cluster, and nothing copies repo files
there. So its build has a first target that writes the fixture into the cluster's
shared filesystem from one base64 string embedded in ``build.yaml`` (a tar.gz of
both directories, about 3 KB; base64 holds no characters that Jinja or the shell
would interpret), and registers the two directories as ``env://`` outputs that the
scoring target binds.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import pickle
import tarfile
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE / "code_generation_lite"
COMPLETIONS = HERE / "completions"
BUILD_YAML = HERE.parent / "slurm-cpu" / "build.yaml"

#: Where the staging target unpacks the fixture: the local SLURM cluster's shared
#: filesystem, mounted on every node. Per-run subdirectory, from GB_TARGETRUN_ID.
SHARED_ROOT = "/shared/lcb-score-fixture"

RELEASE = "release_v6"
# One contest day, so the date window selects exactly these problems.
DATE = "2099-01-01"

#: The expected score, pinned by the build test.
EXPECTED = {
    "pass@1": 0.5,
    "by_difficulty": {"easy": 0.75, "medium": 0.25},
    "problems": 4,
}

STDIN_FORMAT = (
    "Read the input from standard input and write the answer to standard output."
)


def _stdin(inp: str, out: str) -> dict:
    return {"input": inp, "output": out, "testtype": "stdin"}


def _func(args: list, result) -> dict:
    # Functional tests hold one JSON value per argument, one per line, and the result.
    return {
        "input": "\n".join(json.dumps(a) for a in args),
        "output": json.dumps(result),
        "testtype": "functional",
    }


PROBLEMS = [
    {
        "question_id": "fx_sum",
        "question_title": "Fixture Sum",
        "question_content": "Given two integers A and B on one line, print A + B.\n\n"
        + STDIN_FORMAT,
        "difficulty": "easy",
        "starter_code": "",
        "func_name": None,
        "public": [_stdin("1 2\n", "3\n")],
        "private": [
            _stdin("-5 5\n", "0\n"),
            _stdin("1000000000 1000000000\n", "2000000000\n"),
        ],
    },
    {
        "question_id": "fx_max",
        "question_title": "Fixture Maximum",
        "question_content": "Given N and then N integers on the next line, print the largest.\n\n"
        + STDIN_FORMAT,
        "difficulty": "easy",
        "starter_code": "",
        "func_name": None,
        "public": [_stdin("3\n4 9 2\n", "9\n")],
        "private": [_stdin("1\n-7\n", "-7\n"), _stdin("4\n-1 -2 -3 -4\n", "-1\n")],
    },
    {
        "question_id": "fx_rev",
        "question_title": "Fixture Reverse",
        "question_content": "Given a string S of lowercase letters, print S reversed.\n\n"
        + STDIN_FORMAT,
        "difficulty": "medium",
        "starter_code": "",
        "func_name": None,
        "public": [_stdin("abc\n", "cba\n")],
        "private": [
            _stdin("a\n", "a\n"),
            _stdin("racecar\n", "racecar\n"),
            _stdin("ab\n", "ba\n"),
        ],
    },
    {
        "question_id": "fx_square",
        "question_title": "Fixture Square",
        "question_content": "Return the square of the integer x.",
        "difficulty": "medium",
        "starter_code": "class Solution:\n    def square(self, x: int) -> int:\n        ",
        "func_name": "square",
        "public": [_func([3], 9)],
        "private": [_func([-4], 16), _func([0], 0)],
    },
]

# Two samples per problem; see the table in the module docstring.
_SOLUTION = (
    "class Solution:\n    def square(self, x: int) -> int:\n        return x * x\n"
)
SAMPLES = {
    "fx_sum": [
        "a, b = map(int, input().split())\nprint(a + b)",
        "print(sum(map(int, input().split())))",
    ],
    "fx_max": [
        "input()\nprint(max(map(int, input().split())))",
        "input()\nprint(min(map(int, input().split())))",
    ],
    # The second sample is the empty string the generate step writes when a sample
    # has no code block (for example, when it ran out of tokens).
    "fx_rev": ["print(input())", ""],
    "fx_square": [
        _SOLUTION,
        "class Solution:\n    def square(self, x: int) -> int:\n        while True:\n            pass\n",
    ],
}

LOADING_SCRIPT = '''# SPDX-License-Identifier: Apache-2.0
"""Loading script for the eval/livecodebench-score test fixture.

It mirrors the interface of livecodebench/code_generation_lite's loading script
(a `version_tag` config and the same features), so LiveCodeBench's loader reads
this fixture exactly as it reads the real dataset. Every version reads the one
fixture file. Generated by make_fixture.py.
"""

import json

import datasets

_FILES = ["test.jsonl"]
_VERSIONS = [
    "release_v1",
    "release_v2",
    "release_v3",
    "release_v4",
    "release_v5",
    "release_v6",
    "release_latest",
]
_FIELDS = [
    "question_title",
    "question_content",
    "platform",
    "question_id",
    "contest_id",
    "contest_date",
    "starter_code",
    "difficulty",
    "public_test_cases",
    "private_test_cases",
    "metadata",
]


class FixtureConfig(datasets.BuilderConfig):
    def __init__(self, *args, version_tag="release_latest", **kwargs):
        super().__init__(
            *args, name=version_tag, version=datasets.Version("1.0.0"), **kwargs
        )
        assert version_tag in _VERSIONS, f"{version_tag} not in {_VERSIONS}"
        self.version_tag = version_tag


class Fixture(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = FixtureConfig
    BUILDER_CONFIGS = [FixtureConfig(version_tag=v) for v in _VERSIONS]
    DEFAULT_CONFIG_NAME = "release_latest"

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({f: datasets.Value("string") for f in _FIELDS})
        )

    def _split_generators(self, dl_manager):
        paths = dl_manager.download(_FILES)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={"paths": paths}
            )
        ]

    def _generate_examples(self, paths):
        key = 0
        for path in paths:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    yield key, json.loads(line)
                    key += 1
'''


def encode_private(tests: list) -> str:
    """The encoding LiveCodeBench decodes: base64(zlib(pickle(json string)))."""
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(tests)))).decode()


def row(p: dict) -> dict:
    return {
        "question_title": p["question_title"],
        "question_content": p["question_content"],
        "platform": "leetcode" if p["func_name"] else "atcoder",
        "question_id": p["question_id"],
        "contest_id": "fixture",
        "contest_date": f"{DATE}T00:00:00",
        "starter_code": p["starter_code"],
        "difficulty": p["difficulty"],
        "public_test_cases": json.dumps(p["public"]),
        "private_test_cases": encode_private(p["private"]),
        "metadata": json.dumps({"func_name": p["func_name"]} if p["func_name"] else {}),
    }


def fixture_files() -> dict:
    """Every fixture file, as ``{relative path: text}``."""
    ids = [p["question_id"] for p in PROBLEMS]
    meta = {
        "release_version": RELEASE,
        "start_date": DATE,
        "end_date": DATE,
        "model": "fixture (hand-written samples)",
        "n": 2,
        "question_ids_sha256": hashlib.sha256(
            "\n".join(sorted(ids)).encode()
        ).hexdigest(),
        "expected": EXPECTED,
    }
    return {
        "code_generation_lite/code_generation_lite.py": LOADING_SCRIPT,
        "code_generation_lite/test.jsonl": "".join(
            json.dumps(row(p)) + "\n" for p in PROBLEMS
        ),
        "completions/completions.json": json.dumps(
            [{"question_id": i, "code_list": SAMPLES[i]} for i in ids], indent=1
        )
        + "\n",
        "completions/generation_meta.json": json.dumps(meta, indent=2) + "\n",
    }


def fixture_tarball_b64(files: dict) -> str:
    """A reproducible tar.gz of ``files``, base64-encoded (fixed mtimes and owners)."""
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for name in sorted(files):
                data = files[name].encode()
                info = tarfile.TarInfo(name)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                tar.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


BUILD_TEMPLATE = """\
# GENERATED by ../fixture/make_fixture.py; do not edit by hand. Re-run it after
# changing the fixture problems or samples (a unit test fails if this file is stale).
#
# eval/livecodebench-score on the local Docker SLURM cluster, CPU only, against a
# fixture written for this test (4 problems, 2 samples each), so the score is known
# exactly: pass@1 0.5 (easy 0.75, medium 0.25). See make_fixture.py for the table.
#
# WHAT REACHING SUCCESS PROVES:
#   1. setup's install works on a real node: the pinned LiveCodeBench checkout, CPU
#      torch, the pinned packages, LiveCodeBench --no-deps.
#   2. The two declared inputs reach the scorer as bindings, and the dataset shim
#      reads a local copy of the dataset offline.
#   3. LiveCodeBench's own scorer grades every case the fixture plants: correct,
#      wrong answer, empty code, a stdin problem and a function problem, and a
#      timeout. The `check` target reads summary.json and fails unless pass@1 and
#      the per-difficulty scores are exactly the expected ones.
#   4. The step refuses nothing it shouldn't: no HF_TOKEN reaches it.
#   5. The results directory is registered from the artifact marker.
#
# THREE targets, because nothing copies repo files onto the cluster:
#   - `stage` unpacks the fixture (one base64 tar.gz, below) into the shared
#     filesystem and registers its two directories as env:// outputs;
#   - `score` runs the step against them;
#   - `check` asserts the exact scores.
# env:// moves no bytes: it works here because /shared is mounted on every node.
# An env:// output registers the path in its URI (not the artifact marker's), so
# each URI below names exactly the directory its step writes.
granite.build:
  name: skypilot-slurm-livecodebench-score-cpu-test
  targets:

    stage:
      environment_uri: space://environments/skypilot/slurm
      outputs:
        dataset:
          uri: "env://{shared}/{{{{ run_metadata.targetrun_id }}}}/code_generation_lite"
          type: dataset
        completions:
          uri: "env://{shared}/{{{{ run_metadata.targetrun_id }}}}/completions"
          type: dataset
      steps:
        - step_uri: space://steps/command
          config:
            poll_interval_seconds: 5
            command_config:
              command: >-
                set -e;
                D="{shared}/$GB_TARGETRUN_ID";
                mkdir -p "$D";
                echo "{tarball}" | base64 -d | tar -xzf - -C "$D";
                ls -la "$D"/*;
                echo "GB_ARTIFACT_ID:dataset GB_ARTIFACT_PATH:$D/code_generation_lite";
                echo "GB_ARTIFACT_ID:completions GB_ARTIFACT_PATH:$D/completions"
            compute_config:
              num_cpus_per_node: 1
              num_gpus_per_node: 0
              total_memory_per_node: 1Gi

    score:
      environment_uri: space://environments/skypilot/slurm
      inputs:
        dataset:
          binding: stage.dataset
        completions:
          binding: stage.completions
      outputs:
        # env:// registers THIS path, not the artifact marker's, so it must be
        # exactly where output_path below puts the results.
        results:
          uri: "env://{shared}/results-{{{{ run_metadata.targetrun_id }}}}"
          type: dataset
      steps:
        - step_uri: space://steps/eval/livecodebench-score
          config:
            poll_interval_seconds: 5
            lcb_score_config:
              # No Pyxis on the local cluster: run on the bare node.
              image: ""
              release_version: "{release}"
              start_date: "{date}"
              end_date: "{date}"
              num_process_evaluate: 2
              # The results land on the shared filesystem, so `check` can read them.
              output_path: "{shared}/results-{{{{ run_metadata.targetrun_id }}}}"
            compute_config:
              num_cpus_per_node: 2
              num_gpus_per_node: 0

    check:
      environment_uri: space://environments/skypilot/slurm
      inputs:
        results:
          binding: score.results
      steps:
        - step_uri: space://steps/command
          config:
            poll_interval_seconds: 5
            command_config:
              command: >-
                set -e;
                S="{{{{ bindings.results.binding.path }}}}/summary.json";
                cat "$S";
                python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); got={{"pass@1": s["pass@1"], "problems": s["problems"], "by_difficulty": {{k: v["pass@1"] for k, v in s["by_difficulty"].items()}}}}; want={expected}; print("expected", want); print("got     ", got); sys.exit(0 if got == want else 1)' "$S"
            compute_config:
              num_cpus_per_node: 1
              num_gpus_per_node: 0
              total_memory_per_node: 1Gi
"""


def build_yaml(tarball: str) -> str:
    return BUILD_TEMPLATE.format(
        shared=SHARED_ROOT,
        tarball=tarball,
        release=RELEASE,
        date=DATE,
        expected=json.dumps(EXPECTED),
    )


def main() -> None:
    files = fixture_files()
    for name, text in files.items():
        path = HERE / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(text)
    BUILD_YAML.parent.mkdir(exist_ok=True)
    BUILD_YAML.write_text(build_yaml(fixture_tarball_b64(files)))
    print(
        f"wrote {len(PROBLEMS)} problems and their completions under {HERE}, and {BUILD_YAML}"
    )


if __name__ == "__main__":
    main()
