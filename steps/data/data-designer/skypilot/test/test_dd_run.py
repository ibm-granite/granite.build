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

"""Unit tests for the bundled ``src/dd_run.py``.

``make test`` puts ``src/`` on ``PYTHONPATH``, so the runner is imported directly. Its
pure logic needs no Data Designer install: spec decoding, credential resolution, the
seed-path override and the checks that run before any work. Cluster-agnostic, Mode 1
only. A live run of the runner against a real model endpoint is IBM-only, in
``test/integration/ibm/buildrunner/skypilot/slurm_local/test_data_designer_runner_rits.py``.
"""

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import dd_run
import pytest
from dd_run import StepError

# The runner's end-to-end tests import data-designer, which is deliberately NOT in the
# repo .venv that `make test` uses (installing it there replaces the repo's pinned
# duckdb, fsspec, pyarrow and opentelemetry). They skip there; run them with a venv
# that has it — see the step README.md, "Unit tests".
_RUNNER = Path(__file__).resolve().parents[1] / "src" / "dd_run.py"

_NEEDS_DD = (
    "needs data-designer (not in the repo .venv): "
    "<venv>/bin/python -m pytest -c /dev/null --noconftest test/test_dd_run.py"
)


def _b64(obj) -> str:
    return base64.b64encode(json.dumps(obj).encode()).decode()


_CONFIG = {"columns": [{"name": "a", "column_type": "sampler"}]}
_PROVIDERS = [{"name": "rits", "endpoint": "https://e/v1"}]


class TestLoadSpec:
    def test_round_trip(self):
        spec = dd_run.load_spec(
            {dd_run.SPEC_ENV: _b64({"config": _CONFIG, "model_providers": _PROVIDERS})}
        )
        assert spec == {
            "config": _CONFIG,
            "model_providers": _PROVIDERS,
            "run_config": {},
            "local_model": {},
        }

    def test_run_config_passes_through(self):
        rc = {"buffer_size": 2, "max_concurrent_row_groups": 1}
        spec = dd_run.load_spec(
            {
                dd_run.SPEC_ENV: _b64(
                    {"config": _CONFIG, "model_providers": _PROVIDERS, "run_config": rc}
                )
            }
        )
        assert spec["run_config"] == rc

    def test_run_config_must_be_mapping(self):
        with pytest.raises(StepError, match="run_config must be a mapping"):
            dd_run.load_spec(
                {
                    dd_run.SPEC_ENV: _b64(
                        {
                            "config": _CONFIG,
                            "model_providers": _PROVIDERS,
                            "run_config": [1],
                        }
                    )
                }
            )

    @pytest.mark.parametrize(
        "raw,needle",
        [
            ("", "is empty"),
            ("not base64!!", "not base64-encoded YAML"),
            (_b64([1, 2]), "must decode to an object"),
            (
                _b64({"config": {}, "model_providers": _PROVIDERS}),
                "config must be a non-empty",
            ),
            (
                _b64({"config": _CONFIG, "model_providers": []}),
                "model_providers must be a non-empty",
            ),
            (
                _b64(
                    {
                        "config": _CONFIG,
                        "model_providers": _PROVIDERS,
                        "local_model": [1],
                    }
                ),
                "local_model must be a mapping",
            ),
        ],
    )
    def test_refusals(self, raw, needle):
        with pytest.raises(StepError, match=needle):
            dd_run.load_spec({dd_run.SPEC_ENV: raw})


class TestLocalModel:
    _OLLAMA = {
        "runtime": "ollama",
        "model": "granite4:350m",
        "port": 11434,
        "name": "local",
    }

    def test_empty_providers_allowed_with_local_model(self):
        spec = dd_run.load_spec(
            {
                dd_run.SPEC_ENV: _b64(
                    {
                        "config": _CONFIG,
                        "model_providers": [],
                        "local_model": self._OLLAMA,
                    }
                )
            }
        )
        assert spec["model_providers"] == [] and spec["local_model"] == self._OLLAMA

    def test_provider_points_at_localhost(self):
        assert dd_run.local_model_provider(self._OLLAMA, []) == {
            "name": "local",
            "endpoint": "http://127.0.0.1:11434/v1",
            "api_key": "ollama",
        }

    def test_port_and_name_overrides(self):
        p = dd_run.local_model_provider(
            {**self._OLLAMA, "port": 12000, "name": "small"}, []
        )
        assert p["endpoint"] == "http://127.0.0.1:12000/v1" and p["name"] == "small"

    def test_off_returns_none(self):
        assert dd_run.local_model_provider({"runtime": ""}, _PROVIDERS) is None
        assert dd_run.local_model_provider({}, _PROVIDERS) is None

    def test_unknown_runtime(self):
        with pytest.raises(StepError, match="must be empty or 'ollama'"):
            dd_run.local_model_provider({**self._OLLAMA, "runtime": "vllm"}, [])

    def test_name_collision_with_a_provider(self):
        with pytest.raises(StepError, match="also a model_providers name"):
            dd_run.local_model_provider(
                self._OLLAMA, [{"name": "local", "endpoint": "e"}]
            )


class TestBuildProviders:
    def test_header_from_env_merges(self):
        p = [
            {
                "name": "rits",
                "endpoint": "e",
                "extra_headers": {"X-A": "1"},
                "extra_headers_from_env": {"RITS_API_KEY": "MY_KEY"},
            }
        ]
        out = dd_run.build_providers(p, {"MY_KEY": "s3cret"})
        assert out == [
            {
                "name": "rits",
                "endpoint": "e",
                "extra_headers": {"X-A": "1", "RITS_API_KEY": "s3cret"},
            }
        ]

    def test_api_key_env(self):
        out = dd_run.build_providers(
            [{"name": "n", "endpoint": "e", "api_key_env": "K"}], {"K": "v"}
        )
        assert out[0]["api_key"] == "v" and "api_key_env" not in out[0]

    def test_input_not_mutated(self):
        p = [{"name": "n", "endpoint": "e", "api_key_env": "K"}]
        dd_run.build_providers(p, {"K": "v"})
        assert p == [{"name": "n", "endpoint": "e", "api_key_env": "K"}]

    def test_missing_variable_names_it_but_not_a_value(self):
        p = [
            {
                "name": "rits",
                "endpoint": "e",
                "extra_headers_from_env": {"RITS_API_KEY": "RITS_API_KEY"},
            }
        ]
        with pytest.raises(StepError) as e:
            dd_run.build_providers(p, {"OTHER": "s3cret"})
        assert "$RITS_API_KEY" in str(e.value)
        assert "secret_names_to_use_as_env_variable" in str(e.value)
        assert "s3cret" not in str(e.value)

    @pytest.mark.parametrize(
        "entry,needle",
        [
            (
                {"name": "n", "endpoint": "e", "api_key": "x", "api_key_env": "K"},
                "both api_key and api_key_env",
            ),
            (
                {"name": "n", "endpoint": "e", "api_key_env": "bad-name"},
                "must name an environment variable",
            ),
            (
                {"name": "n", "endpoint": "e", "extra_headers_from_env": ["K"]},
                "must map header",
            ),
            ("not a mapping", "must be a mapping"),
        ],
    )
    def test_refusals(self, entry, needle):
        with pytest.raises(StepError, match=needle):
            dd_run.build_providers([entry], {"K": "v"})

    def test_plain_entries_pass_through(self):
        assert dd_run.build_providers(_PROVIDERS, {}) == _PROVIDERS


class TestApplySeedPath:
    def _seeded(self, seed_type="local", full=False):
        cfg = {
            "columns": [],
            "seed_config": {"source": {"seed_type": seed_type, "path": "orig.parquet"}},
        }
        return {"data_designer": cfg} if full else cfg

    def test_no_seed_is_identity(self):
        cfg = self._seeded()
        assert dd_run.apply_seed_path(cfg, "", "") is cfg

    @pytest.mark.parametrize("full", [False, True])
    def test_replaces_path_in_both_forms(self, full):
        out = dd_run.apply_seed_path(
            self._seeded(full=full), "/staged/seed/", "*.parquet"
        )
        inner = out["data_designer"] if full else out
        assert inner["seed_config"]["source"]["path"] == "/staged/seed/*.parquet"

    def test_does_not_mutate_input(self):
        cfg = self._seeded()
        dd_run.apply_seed_path(cfg, "/staged", "")
        assert cfg["seed_config"]["source"]["path"] == "orig.parquet"

    def test_bound_without_seed_config(self):
        with pytest.raises(StepError, match="no seed_config.source"):
            dd_run.apply_seed_path(_CONFIG, "/staged", "")

    def test_hub_source_refused(self):
        with pytest.raises(StepError, match="'hf'"):
            dd_run.apply_seed_path(self._seeded("hf"), "/staged", "")


class TestPreWorkChecks:
    def test_never_refuses_non_empty_dir(self, tmp_path):
        (tmp_path / "old.parquet").write_text("x")
        with pytest.raises(StepError, match="not empty and resume is 'never'"):
            dd_run.check_output_dir(tmp_path, "never")
        dd_run.check_output_dir(tmp_path, "if_possible")
        dd_run.check_output_dir(tmp_path / "absent", "never")

    def test_resumable_records_counts_checkpointed_rows(self, tmp_path):
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        batches = tmp_path / "out" / "parquet-files"
        batches.mkdir(parents=True)
        for i, n in enumerate([2, 2, 1]):
            pq.write_table(
                pa.table({"q": ["x"] * n}), batches / f"batch_{i:05d}.parquet"
            )
        out = tmp_path / "out"
        assert dd_run.count_resumable_records(out, True) == 5
        assert dd_run.count_resumable_records(tmp_path / "absent", True) == 0
        # batches without a kept metadata.json: no resume, count 0
        assert dd_run.count_resumable_records(out, False) == 0

    @pytest.mark.parametrize("bad", ["", "a b", 'a"b'])
    def test_marker_id(self, bad, tmp_path):
        with pytest.raises(StepError):
            dd_run.check_marker_values(bad, tmp_path)

    def test_marker_path_quote(self, tmp_path):
        with pytest.raises(StepError, match="double quote"):
            dd_run.check_marker_values("ok", tmp_path / 'a"b')

    def test_main_reports_one_error_line(self, capsys, monkeypatch, tmp_path):
        monkeypatch.delenv(dd_run.SPEC_ENV, raising=False)
        rc = dd_run.main(
            [
                "--output-path",
                str(tmp_path / "o"),
                "--artifact-id",
                "q",
                "--num-records",
                "1",
                "--resume",
                "never",
            ]
        )
        assert rc == 1
        err = capsys.readouterr().err
        assert err.startswith("data-designer: ERROR") and "Traceback" not in err


class TestClearPreviousRun:
    def test_removes_only_data_designer_artifacts(self, tmp_path):
        out = tmp_path / "out"
        for d in ("parquet-files", "tmp-partial-parquet-files", "images"):
            (out / d).mkdir(parents=True)
            (out / d / "x").write_text("x")
        for f in ("metadata.json", "builder_config.json", "out.jsonl", "out.csv"):
            (out / f).write_text("x")
        (out / "notes").mkdir()
        (out / "notes" / "mine.txt").write_text("keep me")
        (out / "README").write_text("keep me")
        dd_run.clear_previous_run(out)
        assert sorted(p.name for p in out.iterdir()) == ["README", "notes"]
        assert (out / "notes" / "mine.txt").read_text() == "keep me"

    def test_symlink_is_unlinked_not_followed(self, tmp_path):
        target = tmp_path / "elsewhere"
        target.mkdir()
        (target / "precious").write_text("x")
        out = tmp_path / "out"
        out.mkdir()
        (out / "parquet-files").symlink_to(target, target_is_directory=True)
        dd_run.clear_previous_run(out)
        assert (target / "precious").exists() and not (out / "parquet-files").exists()


class TestDuplicateProviders:
    def test_refused_before_any_secret_is_resolved(self):
        """Data Designer's own duplicate-name error printed the whole list, secrets too."""
        providers = [
            {"name": "p", "endpoint": "e"},
            {"name": "p", "endpoint": "e", "api_key_env": "K"},
        ]
        with pytest.raises(StepError, match="must be unique; repeated: p") as e:
            dd_run.build_providers(providers, {"K": "sk-DUP-SECRET"})
        assert "sk-DUP-SECRET" not in str(e.value)

    def test_unique_names_pass(self):
        assert len(dd_run.build_providers([{"name": "a"}, {"name": "b"}], {})) == 2


class TestForeignContents:
    def _dd_dir(self, tmp_path, **overrides):
        out = tmp_path / "out"
        out.mkdir()
        meta = {"config_hash": "h", "config_hash_version": 2, "target_num_records": 4}
        meta.update(overrides)
        (out / "metadata.json").write_text(json.dumps(meta))
        (out / "builder_config.json").write_text("{}")
        return out

    @staticmethod
    def _foreign(out):
        return dd_run.foreign_contents(out, dd_run.read_metadata(out))

    def test_data_designer_dir_is_not_foreign(self, tmp_path):
        assert self._foreign(self._dd_dir(tmp_path)) is None
        assert self._foreign(tmp_path / "absent") is None

    @pytest.mark.parametrize(
        "setup,needle",
        [
            (
                lambda o: (o / "metadata.json").write_text('{"owner": "me"}'),
                "not one Data Designer wrote",
            ),
            (lambda o: (o / "metadata.json").write_text("not json"), "not valid JSON"),
            (lambda o: (o / "metadata.json").write_text("[1]"), "not an object"),
            (lambda o: (o / "builder_config.json").unlink(), "no builder_config.json"),
        ],
        ids=["someone-elses-metadata", "corrupt", "not-an-object", "no-builder-config"],
    )
    def test_foreign_dirs_are_recognised(self, tmp_path, setup, needle):
        out = self._dd_dir(tmp_path)
        setup(out)
        assert needle in self._foreign(out)


class TestOrphanedOutputs:
    def test_an_early_interruption_is_not_orphaned(self, tmp_path):
        """What a run stopped before its first row group leaves: Data Designer clears
        the partials and rewrites builder_config.json, so a plain retry must work."""
        out = tmp_path / "out"
        (out / "tmp-partial-parquet-files").mkdir(parents=True)
        (out / "tmp-partial-parquet-files" / "b.parquet").write_text("x")
        (out / "builder_config.json").write_text("{}")
        (out / "images").mkdir()  # empty: nothing to mix in
        assert dd_run.orphaned_outputs(out, None) == []
        assert dd_run.orphaned_outputs(tmp_path / "absent", None) == []

    @pytest.mark.parametrize(
        "make,name",
        [
            (
                lambda o: _touch(o / "parquet-files" / "batch_00000.parquet"),
                "parquet-files/",
            ),
            (
                lambda o: _touch(o / "dropped-columns-parquet-files" / "b.parquet"),
                "dropped-columns-parquet-files/",
            ),
            (
                lambda o: _touch(o / "processors-files" / "p" / "b.parquet"),
                "processors-files/",
            ),
            (lambda o: _touch(o / "images" / "old.png"), "images/"),
            (lambda o: _touch(o / "scheduler_events.jsonl"), "scheduler_events.jsonl"),
            (lambda o: _touch(o / "out.jsonl"), "out.jsonl"),
        ],
        ids=[
            "row-groups",
            "dropped-columns",
            "processors",
            "images",
            "events",
            "export",
        ],
    )
    def test_every_other_output_counts(self, tmp_path, make, name):
        """Re-review of #543: only parquet-files/ was checked, so the others mixed in."""
        out = tmp_path / "out"
        out.mkdir()
        make(out)
        assert dd_run.orphaned_outputs(out, None) == [name]
        assert dd_run.orphaned_outputs(out, {"config_hash": "h"}) == []


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


class TestRedaction:
    def test_secrets_collected_longest_first(self):
        providers = [
            {"name": "a", "api_key": "KEYVAR", "extra_headers": {"X": "hdr-value-1"}},
            {"name": "b", "api_key": "sk-literal"},
            {"name": "c"},
        ]
        secrets = dd_run.provider_secrets(providers, {"KEYVAR": "from-env-value"})
        # A plain api_key naming a set variable: its VALUE is the secret, the name is not.
        assert set(secrets) == {"from-env-value", "hdr-value-1", "sk-literal"}
        assert [len(x) for x in secrets] == sorted(map(len, secrets), reverse=True)

    def test_errors_stay_readable(self):
        """Re-review of #543: the local placeholder key and the NAME a plain api_key
        points at were redacted everywhere, so ``dd_ollama.sh`` became
        ``dd_<redacted>.sh``."""
        local = dd_run.local_model_provider({"runtime": "ollama", "model": "m"}, [])
        providers = [local, {"name": "gw", "api_key": "OPENAI_API_KEY"}]
        secrets = dd_run.provider_secrets(
            providers, {"OPENAI_API_KEY": "sk-real-12345"}
        )
        assert secrets == ["sk-real-12345"]
        text = (
            "dd_ollama.sh: ollama/granite4:350m failed; set OPENAI_API_KEY "
            "(sk-real-12345)"
        )
        assert dd_run.redact(text, secrets) == text.replace(
            "sk-real-12345", "<redacted>"
        )

    def test_short_credentials_are_redacted(self):
        """Third review of #543: an 8-character cutoff let a short real key through.
        What counts is where a value came from, not its length."""
        env = {"KEY": "ab12cd", "HDR": "x9"}
        providers = dd_run.build_providers(
            [
                {
                    "name": "gw",
                    "endpoint": "https://e/v1",
                    "api_key_env": "KEY",
                    "extra_headers_from_env": {"X-Token": "HDR"},
                }
            ],
            env,
        )
        assert set(dd_run.provider_secrets(providers, env)) == {"ab12cd", "x9"}

    def test_every_literal_header_value_is_redacted(self):
        """Which headers carry a credential can't be told from their names, so every
        literal extra_headers value is redacted, auth or not."""
        providers = [
            {
                "name": "gw",
                "extra_headers": {
                    "Authorization": "Bearer abc",
                    "X-Custom-Thing": "q7",
                    "Content-Type": "application/json",
                },
            }
        ]
        assert set(dd_run.provider_secrets(providers, {})) == {
            "Bearer abc",
            "q7",
            "application/json",
        }

    def test_redaction_is_one_pass_and_keeps_text_readable(self):
        """Independent review of the #543 fixes: replacing secrets one after another let
        a short header value (``2``, ``e``) rewrite the ``<redacted>`` markers and the
        words around them."""
        secrets = dd_run.provider_secrets(
            [{"api_key": "sk-REAL-KEY-123", "extra_headers": {"V": "2", "F": "e"}}], {}
        )
        text = 'File "/opt/x.py", line 212, in create: key=sk-REAL-KEY-123 v 2 f e'
        assert dd_run.redact(text, secrets) == (
            'File "/opt/x.py", line 212, in create: key=<redacted> v <redacted> '
            "f <redacted>"
        )
        # A short value is redacted standing alone, not inside a longer word, number
        # or path (second independent review: "1.0" turned "11.05s" into noise)...
        assert dd_run.redact("nonexistent; api_key=none", ["none"]) == (
            "nonexistent; api_key=<redacted>"
        )
        assert dd_run.redact(
            'took 11.05s in "/usr/lib/python3.12/a.py"; lang en-US',
            ["1.0", "3.1", "/a", "en-US"],
        ) == ('took 11.05s in "/usr/lib/python3.12/a.py"; lang <redacted>')
        # ...but a real key is removed even glued to other text, however it's written.
        hexkey = "abcd" * 8  # a long, plain-alphanumeric stand-in for a hex key
        assert dd_run.redact(f"x{hexkey}y sk-1abcd!", [hexkey, "sk-1abcd"]) == (
            "x<redacted>y <redacted>!"
        )

    def test_repr_escaped_secrets_are_redacted(self):
        """Data Designer's ModelProvider repr escapes backslashes and quotes."""
        secret = "ab\\cd'\"q"
        assert "<redacted>" in dd_run.redact(repr({"api_key": secret}), [secret])
        assert secret not in dd_run.redact(repr({"api_key": secret}), [secret])
        assert repr(secret)[1:-1] not in dd_run.redact(
            repr({"api_key": secret}), [secret]
        )

    def test_errors_and_tracebacks_lose_every_secret(self, capsys):
        secrets = ["sk-SECRET-1", "hdr-SECRET-2"]
        with pytest.raises(StepError) as e:
            with dd_run.redacted_errors(secrets):
                raise RuntimeError(
                    "provider gw: api_key='sk-SECRET-1' X='hdr-SECRET-2'"
                )
        assert "SECRET" not in str(e.value) and "<redacted>" in str(e.value)
        assert e.value.__cause__ is None and e.value.__suppress_context__
        err = capsys.readouterr().err
        assert "Traceback" in err and "SECRET" not in err

    def test_step_errors_and_passthrough_are_untouched(self):
        with pytest.raises(StepError, match="^as is$"):
            with dd_run.redacted_errors(["x"]):
                raise StepError("as is")
        with pytest.raises(KeyError):
            with dd_run.redacted_errors(["x"], passthrough=(KeyError,)):
                raise KeyError("k")


class TestPostProcessedRefusal:
    def _meta(self, tmp_path, **meta):
        return {"target_num_records": 4, **meta}

    def test_extending_a_processed_dataset_is_refused(self, tmp_path):
        out = self._meta(tmp_path, post_generation_processed=True)
        assert "can't be extended to 6" in dd_run.post_processed_refusal(out, 6)
        assert dd_run.post_processed_refusal(out, 4) is None  # finished: nothing to do

    def test_interrupted_processing_is_refused(self, tmp_path):
        out = self._meta(tmp_path, post_generation_state="started")
        assert "did not finish" in dd_run.post_processed_refusal(out, 4)

    def test_unprocessed_dataset_is_fine(self, tmp_path):
        assert dd_run.post_processed_refusal(self._meta(tmp_path), 6) is None


class TestValidateProviders:
    def test_invalid_entry_never_shows_the_secret(self):
        pytest.importorskip("data_designer", reason=_NEEDS_DD)
        from data_designer.config.models import ModelProvider

        resolved = [
            {
                "name": "gw",
                "endpont": "http://e",
                "api_key": "sk-SECRET-1",
                "extra_headers": {"X-Key": "sk-SECRET-2"},
            }
        ]
        with pytest.raises(StepError) as e:
            dd_run.validate_providers(resolved, ModelProvider)
        msg = str(e.value)
        assert "endpoint" in msg and "sk-SECRET" not in msg
        assert e.value.__cause__ is None and e.value.__suppress_context__

    @pytest.mark.parametrize(
        "bad",
        [
            {"api_key": 12345},  # a non-string credential
            {"extra_headers": {"X-Org": 123, "X-Key": "sk-SECRET-3"}},
        ],
    )
    def test_wrong_type_credential_reported_by_location(self, bad):
        """Second review: a placeholder hid the type error from the first pass."""
        pytest.importorskip("data_designer", reason=_NEEDS_DD)
        from data_designer.config.models import ModelProvider

        with pytest.raises(StepError) as e:
            dd_run.validate_providers(
                [{"name": "gw", "endpoint": "http://e", **bad}], ModelProvider
            )
        msg = str(e.value)
        assert "12345" not in msg and "sk-SECRET-3" not in msg and "123" not in msg
        assert e.value.__cause__ is None and e.value.__suppress_context__

    def test_valid_entry_keeps_real_values(self):
        pytest.importorskip("data_designer", reason=_NEEDS_DD)
        from data_designer.config.models import ModelProvider

        [p] = dd_run.validate_providers(
            [
                {
                    "name": "gw",
                    "endpoint": "http://e",
                    "api_key": "k",
                    "extra_headers": {"H": "v"},
                }
            ],
            ModelProvider,
        )
        assert p.api_key == "k" and p.extra_headers == {"H": "v"}


# The runner end to end with a sampler-only config: no column calls a model, so this
# needs data-designer but no endpoint. The provider points at a closed port and is
# never contacted. Covers what used to be reachable only through the IBM RITS tests.
_SAMPLER_CONFIG = {
    "columns": [
        {
            "name": "t",
            "column_type": "sampler",
            "sampler_type": "category",
            "params": {"values": ["a", "b"]},
        }
    ]
}
_DEAD = [{"name": "p", "endpoint": "http://127.0.0.1:9/v1"}]


class _Run:
    """One subprocess run of the runner: exit code plus captured output."""

    def __init__(self):
        self.rc, self.out, self.err = None, "", ""


@pytest.fixture
def run_dd(tmp_path):
    """Run src/dd_run.py in a SUBPROCESS with its own DATA_DESIGNER_HOME, as on the node.

    In-process runs can't isolate Data Designer's settings: it reads DATA_DESIGNER_HOME
    once, at import, so every in-process test shared the developer's real
    ~/.data-designer. That home is already seeded with default model configs, which hid
    the builder-ordering bug the test below exists to catch (third review: with the fix
    reverted, the whole file still passed). Each run here starts a fresh interpreter,
    and ``fresh_home=True`` gives that run a brand-new home, as a new build gets.
    """
    pytest.importorskip("data_designer", reason=_NEEDS_DD)
    last = _Run()
    homes = iter(range(1, 1000))

    def run(
        num,
        resume="if_possible",
        run_config=None,
        config=None,
        providers=None,
        extra=(),
        env=None,
        fresh_home=False,
    ):
        spec = {
            "config": config or _SAMPLER_CONFIG,
            "model_providers": providers or _DEAD,
            "run_config": run_config or {},
        }
        home = tmp_path / (f"home{next(homes)}" if fresh_home else "home")
        child_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(tmp_path),
            "NEMO_TELEMETRY_ENABLED": "false",
            "DATA_DESIGNER_HOME": str(home),
            dd_run.SPEC_ENV: _b64(spec),
            **(env or {}),
        }
        proc = subprocess.run(
            [
                sys.executable,
                str(_RUNNER),
                "--output-path",
                str(tmp_path / "out"),
                "--artifact-id",
                "q",
                "--num-records",
                str(num),
                "--resume",
                resume,
                *extra,
            ],
            capture_output=True,
            text=True,
            env=child_env,
            cwd=tmp_path,
            timeout=300,
        )
        last.rc, last.out, last.err = proc.returncode, proc.stdout, proc.stderr
        return proc.returncode

    run.last = last
    return run


def _last_line(run_dd):
    return run_dd.last.out.rstrip().splitlines()[-1]


class TestRunEndToEnd:
    def test_marker_is_last_and_export_written(self, run_dd, tmp_path):
        assert run_dd(4, extra=("--export-format", "csv", "--min-records", "4")) == 0
        assert (
            _last_line(run_dd)
            == f"GB_ARTIFACT_ID:q GB_ARTIFACT_PATH:{tmp_path / 'out'}"
        )
        assert (
            len((tmp_path / "out" / "out.csv").read_text().splitlines()) == 5
        )  # header + 4

    def test_min_records_above_num_records_fails_before_work(self, run_dd, tmp_path):
        assert run_dd(2, extra=("--min-records", "3")) == 1
        assert "greater than num_records" in run_dd.last.err
        assert not (tmp_path / "out").exists()

    def test_resume_extends_a_finished_run(self, run_dd, tmp_path):
        assert run_dd(2, run_config={"buffer_size": 2}) == 0
        assert run_dd(4, run_config={"buffer_size": 2}) == 0
        out = run_dd.last.out
        assert "resuming — 2 record(s)" in out
        assert "GB_STEP_METADATA_KEY:resumed_records GB_STEP_METADATA_VALUE:2" in out

    _MISMATCHES = [
        {"run_config": {"buffer_size": 4}},  # was a hard failure
        {
            "run_config": {"buffer_size": 2, "preserve_dropped_columns": False}
        },  # was a sibling dir
        {
            "config": {
                "columns": [
                    {**_SAMPLER_CONFIG["columns"][0], "params": {"values": ["z"]}}
                ]
            },
            "run_config": {"buffer_size": 2},
        },
    ]

    @pytest.mark.parametrize("second", _MISMATCHES)
    def test_restart_starts_fresh_in_place_on_any_mismatch(
        self, run_dd, tmp_path, second
    ):
        assert run_dd(2, run_config={"buffer_size": 2}) == 0
        assert run_dd(2, resume="restart", **second) == 0
        out = run_dd.last.out
        assert "not resuming" in out and "resume: restart, so clearing" in out
        assert "resuming —" not in out
        assert "GB_STEP_METADATA_KEY:resumed_records GB_STEP_METADATA_VALUE:0" in out
        assert [
            p.name for p in tmp_path.iterdir() if p.name.startswith("out_")
        ] == [], "Data Designer wrote to a timestamped sibling dir"

    @pytest.mark.parametrize("second", _MISMATCHES)
    @pytest.mark.parametrize("resume", ["if_possible", "always"])
    def test_default_refuses_a_mismatch_and_removes_nothing(
        self, run_dd, tmp_path, second, resume
    ):
        """Deleting a previous run is opt-in (resume: restart). The default, and always,
        refuse a mismatch before any work and leave every file as it was."""
        assert run_dd(2, run_config={"buffer_size": 2}) == 0
        out = tmp_path / "out"
        before = {
            p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()
        }
        assert run_dd(2, resume=resume, **second) == 1
        err = run_dd.last.err
        assert "Nothing was removed" in err and "resume: restart" in err
        after = {
            p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()
        }
        assert after == before, "the previous run was modified"

    def test_always_refuses_a_mismatch_before_work(self, run_dd):
        assert run_dd(2, run_config={"buffer_size": 2}) == 0
        assert run_dd(2, resume="always", run_config={"buffer_size": 4}) == 1
        assert "buffer_size changed" in run_dd.last.err

    def test_always_on_a_first_run_starts_fresh(self, run_dd, tmp_path):
        """As Data Designer documents for ALWAYS: no checkpoint yet means a fresh start."""
        assert run_dd(2, resume="always") == 0
        assert "no checkpoint yet; starting fresh" in run_dd.last.out
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith("out_")] == []

    def test_always_after_an_interruption_before_any_checkpoint(self, run_dd, tmp_path):
        """Only partial batches and no metadata.json: retried under always, starts fresh."""
        partial = tmp_path / "out" / "tmp-partial-parquet-files"
        partial.mkdir(parents=True)
        (partial / "batch.parquet").write_text("x")
        assert run_dd(2, resume="always") == 0
        assert "starting fresh" in run_dd.last.out

    @pytest.mark.parametrize("resume", ["if_possible", "always"])
    def test_lower_num_records_is_refused_and_keeps_the_data(
        self, run_dd, tmp_path, resume
    ):
        assert run_dd(4, run_config={"buffer_size": 2}) == 0
        before = sorted(p.name for p in (tmp_path / "out" / "parquet-files").iterdir())
        assert run_dd(2, resume=resume, run_config={"buffer_size": 2}) == 1
        err = run_dd.last.err
        assert "below the 4" in err and "Nothing was removed" in err
        assert (
            sorted(p.name for p in (tmp_path / "out" / "parquet-files").iterdir())
            == before
        )

    @pytest.mark.parametrize(
        "config",
        [
            # No sampler column: Data Designer adds an internal row-id column on compile.
            {"columns": [{"name": "e", "column_type": "expression", "expr": "hello"}]},
            # A seed: Data Designer adds the seed's columns on compile.
            "SEEDED",
        ],
        ids=["expression-only", "seeded"],
    )
    @pytest.mark.parametrize(
        "home", ["same", "fresh"], ids=["same-home", "fresh-home-each-run"]
    )
    def test_unchanged_compiled_config_resumes(self, run_dd, tmp_path, config, home):
        """Same config twice must resume. Two bugs used to break this (second review):

        - the step fingerprinted the RAW config, but Data Designer stores the COMPILED
          one (it adds an internal row-id column, or the seed's columns);
        - the builder was made before DataDesigner(), which seeds Data Designer's
          default model configs into DATA_DESIGNER_HOME, so the first run in an empty
          home fingerprinted a different model list from every later run.

        "fresh-home-each-run" is how a new build looks on the node (each run gets its
        own ``./.data-designer-home``). Both cases fail if either fix is reverted.
        """
        if config == "SEEDED":
            seed = tmp_path / "seed.csv"
            seed.write_text("city\nParis\nLima\nOslo\nRome\nKyiv\nCairo\n")
            config = {
                "seed_config": {"source": {"seed_type": "local", "path": str(seed)}},
                "columns": [
                    {"name": "e", "column_type": "expression", "expr": "{{ city }}!"}
                ],
            }
        fresh = home == "fresh"
        rc = {"buffer_size": 2}
        assert (
            run_dd(2, config=config, run_config=rc, fresh_home=fresh) == 0
        ), run_dd.last.err
        assert (
            run_dd(4, config=config, run_config=rc, fresh_home=fresh) == 0
        ), run_dd.last.err
        out = run_dd.last.out
        assert "not resuming" not in out
        assert "resuming — 2 record(s)" in out
        assert "GB_STEP_METADATA_KEY:resumed_records GB_STEP_METADATA_VALUE:2" in out
        assert (
            run_dd(6, resume="always", config=config, run_config=rc, fresh_home=fresh)
            == 0
        )

    def test_scheduler_events_are_cleared_on_restart(self, run_dd, tmp_path):
        """Data Designer APPENDS to this file, so a restart that didn't clear it would
        leave the first run's events at its start. Checked by run count, not by content
        merely differing, which appending alone satisfies (third review)."""
        rc = {"buffer_size": 2, "write_scheduler_events": True}
        assert run_dd(2, run_config=rc) == 0
        events = tmp_path / "out" / "scheduler_events.jsonl"
        first = events.read_text()
        assert first.strip(), "Data Designer wrote no scheduler events"
        assert run_dd(2, resume="restart", run_config={**rc, "buffer_size": 4}) == 0
        assert not events.read_text().startswith(first), "previous run's events kept"

    def test_foreign_metadata_is_refused_and_nothing_removed(self, run_dd, tmp_path):
        """Third review: a user's own metadata.json made the step clear the dir."""
        out = tmp_path / "out"
        (out / "images").mkdir(parents=True)
        (out / "metadata.json").write_text('{"owner": "me"}')
        (out / "images" / "cat.png").write_text("png")
        (out / "out.csv").write_text("a,b")
        assert run_dd(1) == 1
        assert "Nothing was removed" in run_dd.last.err
        assert (out / "images" / "cat.png").exists() and (out / "out.csv").exists()

    def test_duplicate_provider_names_report_without_the_key(self, run_dd):
        dup = [
            {"name": "p", "endpoint": "http://127.0.0.1:9/v1"},
            {
                "name": "p",
                "endpoint": "http://127.0.0.1:9/v1",
                "extra_headers_from_env": {"X": "MYKEY"},
            },
        ]
        assert run_dd(1, providers=dup, env={"MYKEY": "sk-FAKE-DUP"}) == 1
        assert "sk-FAKE-DUP" not in run_dd.last.out + run_dd.last.err
        assert "Traceback" not in run_dd.last.err and "unique" in run_dd.last.err

    def test_resume_logs_no_phantom_sibling_dir(self, run_dd):
        """The fingerprint pre-check used to log "will be saved to out_<timestamp>"."""
        assert run_dd(2, run_config={"buffer_size": 2}) == 0
        assert run_dd(4, run_config={"buffer_size": 2}) == 0
        assert "will be saved to" not in run_dd.last.out + run_dd.last.err

    def test_user_files_survive_a_restart(self, run_dd, tmp_path):
        assert run_dd(2, run_config={"buffer_size": 2}) == 0
        (tmp_path / "out" / "notes.txt").write_text("keep me")
        assert run_dd(2, resume="restart", run_config={"buffer_size": 4}) == 0
        assert (tmp_path / "out" / "notes.txt").read_text() == "keep me"

    def test_bad_provider_reports_without_the_key(self, run_dd):
        bad = [
            {
                "name": "gw",
                "endpont": "http://e",
                "extra_headers_from_env": {"X-Key": "MYKEY"},
            }
        ]
        assert run_dd(1, providers=bad, env={"MYKEY": "sk-FAKE-SECRET"}) == 1
        captured = run_dd.last
        assert "sk-FAKE-SECRET" not in captured.out + captured.err
        assert "Traceback" not in captured.err and "endpoint" in captured.err

    @staticmethod
    def _orphan_batches(run_dd, tmp_path):
        """A finished run of ANOTHER config whose metadata.json is gone: batches remain."""
        other = {"columns": [{**_SAMPLER_CONFIG["columns"][0], "name": "zz"}]}
        assert run_dd(4, config=other, run_config={"buffer_size": 2}) == 0
        out = tmp_path / "out"
        (out / "metadata.json").unlink()
        return out

    @pytest.mark.parametrize("resume", ["if_possible", "always"])
    def test_row_groups_without_metadata_are_refused(self, run_dd, tmp_path, resume):
        """Found while checking the review of #543: Data Designer added the new rows
        beside the old ones ("generated 4 of 2 requested", rows from two configs)."""
        out = self._orphan_batches(run_dd, tmp_path)
        before = {
            p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()
        }
        assert run_dd(2, resume=resume, run_config={"buffer_size": 2}) == 1
        err = run_dd.last.err
        assert "but no metadata.json" in err and "Nothing was removed" in err
        after = {
            p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()
        }
        assert after == before

    def test_restart_clears_row_groups_without_metadata(self, run_dd, tmp_path):
        out = self._orphan_batches(run_dd, tmp_path)
        assert (
            run_dd(
                2,
                resume="restart",
                run_config={"buffer_size": 2},
                extra=("--export-format", "jsonl"),
            )
            == 0
        ), run_dd.last.err
        assert "generated 2 of 2" in run_dd.last.out
        rows = [json.loads(x) for x in (out / "out.jsonl").read_text().splitlines()]
        assert len(rows) == 2 and all(set(r) == {"t"} for r in rows)

    def test_restart_wont_clear_row_groups_without_builder_config(
        self, run_dd, tmp_path
    ):
        out = self._orphan_batches(run_dd, tmp_path)
        (out / "builder_config.json").unlink()
        assert run_dd(2, resume="restart", run_config={"buffer_size": 2}) == 1
        assert "won't clear it" in run_dd.last.err
        assert any((out / "parquet-files").iterdir())

    @pytest.mark.parametrize(
        "config",
        [
            # A provider name that isn't registered: raised by Data Designer inside
            # create(), not by the step's own provider checks.
            {
                "model_configs": [{"alias": "a", "model": "m", "provider": "gw-typo"}],
                "columns": [
                    {
                        "name": "q",
                        "column_type": "llm-text",
                        "model_alias": "a",
                        "prompt": "hi",
                    }
                ],
            },
            # An unknown model alias.
            {
                "model_configs": [{"alias": "a", "model": "m", "provider": "gw"}],
                "columns": [
                    {
                        "name": "q",
                        "column_type": "llm-text",
                        "model_alias": "nope",
                        "prompt": "hi",
                    }
                ],
            },
        ],
        ids=["unknown-provider", "unknown-alias"],
    )
    def test_data_designer_errors_are_one_line_without_secrets(self, run_dd, config):
        """Review of #543: validate()/create() errors were raw tracebacks, and Data
        Designer's ModelProvider prints api_key and extra_headers in clear."""
        providers = [
            {
                "name": "gw",
                "endpoint": "http://127.0.0.1:9/v1",
                "api_key_env": "K",
                "extra_headers_from_env": {"X-Key": "H"},
            }
        ]
        env = {"K": "sk-FAKE-KEY-123", "H": "hdr-FAKE-456"}
        assert run_dd(1, config=config, providers=providers, env=env) == 1
        both = run_dd.last.out + run_dd.last.err
        assert "sk-FAKE-KEY-123" not in both and "hdr-FAKE-456" not in both
        assert (
            run_dd.last.err.rstrip()
            .splitlines()[-1]
            .startswith("data-designer: ERROR Data Designer failed:")
        )

    @pytest.mark.parametrize(
        "folder", ["dropped-columns-parquet-files", "processors-files", "images"]
    )
    @pytest.mark.parametrize("resume", ["if_possible", "always"])
    def test_other_outputs_without_metadata_are_refused(
        self, run_dd, tmp_path, folder, resume
    ):
        """Re-review of #543: with no metadata.json, a previous run's dropped columns,
        processor outputs and images survived the next run and were registered with
        it (measured). Only parquet-files/ used to be checked."""
        out = tmp_path / "out"
        _touch(out / "builder_config.json")
        _touch(out / folder / "old" / "leftover.parquet")
        assert run_dd(2, resume=resume, run_config={"buffer_size": 2}) == 1
        err = run_dd.last.err
        assert f"{folder}/" in err and "Nothing was removed" in err
        assert (out / folder / "old" / "leftover.parquet").exists()

    def test_restart_clears_other_outputs_without_metadata(self, run_dd, tmp_path):
        out = tmp_path / "out"
        _touch(out / "builder_config.json")
        _touch(out / "dropped-columns-parquet-files" / "leftover.parquet")
        _touch(out / "notes.txt")
        assert (
            run_dd(2, resume="restart", run_config={"buffer_size": 2}) == 0
        ), run_dd.last.err
        assert not (out / "dropped-columns-parquet-files" / "leftover.parquet").exists()
        assert (out / "notes.txt").exists(), "a user file was removed"

    def test_a_retry_after_an_early_interruption_still_works(self, run_dd, tmp_path):
        out = tmp_path / "out"
        _touch(out / "builder_config.json")
        _touch(out / "tmp-partial-parquet-files" / "batch.parquet")
        assert run_dd(2, run_config={"buffer_size": 2}) == 0, run_dd.last.err

    def test_scheduler_events_left_by_an_early_interruption_need_restart(
        self, run_dd, tmp_path
    ):
        """Independent review of the #543 fixes: with write_scheduler_events, Data
        Designer starts scheduler_events.jsonl (appending) before the first row group,
        so an early interruption leaves it. It would carry over, so the default refuses
        the retry and restart clears it."""
        out = tmp_path / "out"
        _touch(out / "builder_config.json")
        _touch(out / "tmp-partial-parquet-files" / "batch.parquet")
        (out / "scheduler_events.jsonl").write_text('{"old": true}\n')
        rc = {"buffer_size": 2, "write_scheduler_events": True}
        assert run_dd(2, run_config=rc) == 1
        assert "scheduler_events.jsonl" in run_dd.last.err
        assert run_dd(2, resume="restart", run_config=rc) == 0, run_dd.last.err
        assert '{"old": true}' not in (out / "scheduler_events.jsonl").read_text()
