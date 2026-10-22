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

import dd_run
import pytest
from dd_run import StepError

# The runner's end-to-end tests import data-designer, which is deliberately NOT in the
# repo .venv that `make test` uses (installing it there replaces the repo's pinned
# duckdb, fsspec, pyarrow and opentelemetry). They skip there; run them with a venv
# that has it — see the step README.md, "Unit tests".
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
            ("not base64!!", "not base64-encoded JSON"),
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
        (out / "metadata.json").write_text("{}")
        assert dd_run.count_resumable_records(out, "if_possible") == 5
        assert dd_run.count_resumable_records(out, "always") == 5
        assert dd_run.count_resumable_records(out, "never") == 0
        assert dd_run.count_resumable_records(tmp_path / "absent", "if_possible") == 0
        (out / "metadata.json").unlink()  # batches without metadata: no resume, count 0
        assert dd_run.count_resumable_records(out, "if_possible") == 0

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


@pytest.fixture
def run_dd(tmp_path, monkeypatch):
    pytest.importorskip("data_designer", reason=_NEEDS_DD)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "false")
    monkeypatch.setenv("DATA_DESIGNER_HOME", str(tmp_path / "home"))

    def run(
        num,
        resume="if_possible",
        run_config=None,
        config=None,
        providers=None,
        extra=(),
        env=None,
    ):
        spec = {
            "config": config or _SAMPLER_CONFIG,
            "model_providers": providers or _DEAD,
            "run_config": run_config or {},
        }
        monkeypatch.setenv(dd_run.SPEC_ENV, _b64(spec))
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)
        return dd_run.main(
            [
                "--output-path",
                str(tmp_path / "out"),
                "--artifact-id",
                "q",
                "--num-records",
                str(num),
                "--resume",
                resume,
                *extra,
            ]
        )

    return run


def _last_line(capsys):
    return capsys.readouterr().out.rstrip().splitlines()[-1]


class TestRunEndToEnd:
    def test_marker_is_last_and_export_written(self, run_dd, tmp_path, capsys):
        assert run_dd(4, extra=("--export-format", "csv", "--min-records", "4")) == 0
        assert (
            _last_line(capsys)
            == f"GB_ARTIFACT_ID:q GB_ARTIFACT_PATH:{tmp_path / 'out'}"
        )
        assert (
            len((tmp_path / "out" / "out.csv").read_text().splitlines()) == 5
        )  # header + 4

    def test_min_records_above_num_records_fails_before_work(
        self, run_dd, tmp_path, capsys
    ):
        assert run_dd(2, extra=("--min-records", "3")) == 1
        assert "greater than num_records" in capsys.readouterr().err
        assert not (tmp_path / "out").exists()

    def test_resume_extends_a_finished_run(self, run_dd, tmp_path, capsys):
        assert run_dd(2, run_config={"buffer_size": 2}) == 0
        capsys.readouterr()
        assert run_dd(4, run_config={"buffer_size": 2}) == 0
        out = capsys.readouterr().out
        assert "resuming — 2 record(s)" in out
        assert "GB_STEP_METADATA_KEY:resumed_records GB_STEP_METADATA_VALUE:2" in out

    @pytest.mark.parametrize(
        "second",
        [
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
        ],
    )
    def test_if_possible_restarts_in_place_on_any_mismatch(
        self, run_dd, tmp_path, capsys, second
    ):
        assert run_dd(2, run_config={"buffer_size": 2}) == 0
        capsys.readouterr()
        assert run_dd(2, **second) == 0
        out = capsys.readouterr().out
        assert "not resuming" in out and "resuming —" not in out
        assert "GB_STEP_METADATA_KEY:resumed_records GB_STEP_METADATA_VALUE:0" in out
        assert [
            p.name for p in tmp_path.iterdir() if p.name.startswith("out_")
        ] == [], "Data Designer wrote to a timestamped sibling dir"

    def test_always_refuses_a_mismatch_before_work(self, run_dd, capsys):
        assert run_dd(2, run_config={"buffer_size": 2}) == 0
        capsys.readouterr()
        assert run_dd(2, resume="always", run_config={"buffer_size": 4}) == 1
        assert "buffer_size changed" in capsys.readouterr().err

    def test_always_on_a_first_run_starts_fresh(self, run_dd, tmp_path, capsys):
        """As Data Designer documents for ALWAYS: no checkpoint yet means a fresh start."""
        assert run_dd(2, resume="always") == 0
        assert "no checkpoint yet; starting fresh" in capsys.readouterr().out
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith("out_")] == []

    def test_always_after_an_interruption_before_any_checkpoint(
        self, run_dd, tmp_path, capsys
    ):
        """Only partial batches and no metadata.json: retried under always, starts fresh."""
        partial = tmp_path / "out" / "tmp-partial-parquet-files"
        partial.mkdir(parents=True)
        (partial / "batch.parquet").write_text("x")
        assert run_dd(2, resume="always") == 0
        assert "starting fresh" in capsys.readouterr().out

    @pytest.mark.parametrize("resume", ["if_possible", "always"])
    def test_lower_num_records_is_refused_and_keeps_the_data(
        self, run_dd, tmp_path, capsys, resume
    ):
        assert run_dd(4, run_config={"buffer_size": 2}) == 0
        before = sorted(p.name for p in (tmp_path / "out" / "parquet-files").iterdir())
        capsys.readouterr()
        assert run_dd(2, resume=resume, run_config={"buffer_size": 2}) == 1
        err = capsys.readouterr().err
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
    def test_unchanged_compiled_config_resumes(self, run_dd, tmp_path, capsys, config):
        """The second review's blocker: these always looked "changed" and were cleared.

        The step compared the build's raw config fingerprint with the COMPILED one Data
        Designer stores. Both runs here use the same config, so the second must resume.
        """
        if config == "SEEDED":
            seed = tmp_path / "seed.csv"
            seed.write_text("city\nParis\nLima\nOslo\nRome\n")
            config = {
                "seed_config": {"source": {"seed_type": "local", "path": str(seed)}},
                "columns": [
                    {"name": "e", "column_type": "expression", "expr": "{{ city }}!"}
                ],
            }
        assert run_dd(2, config=config, run_config={"buffer_size": 2}) == 0
        capsys.readouterr()
        assert run_dd(4, config=config, run_config={"buffer_size": 2}) == 0
        out = capsys.readouterr().out
        assert "not resuming" not in out
        assert "resuming — 2 record(s)" in out
        assert "GB_STEP_METADATA_KEY:resumed_records GB_STEP_METADATA_VALUE:2" in out
        assert (
            run_dd(6, resume="always", config=config, run_config={"buffer_size": 2})
            == 0
        )

    def test_scheduler_events_are_cleared_on_restart(self, run_dd, tmp_path):
        rc = {"buffer_size": 2, "write_scheduler_events": True}
        assert run_dd(2, run_config=rc) == 0
        events = tmp_path / "out" / "scheduler_events.jsonl"
        assert events.is_file()
        first = events.read_text()
        assert (
            run_dd(2, run_config={**rc, "buffer_size": 4}) == 0
        )  # mismatch -> restart
        assert events.read_text() != first or not events.exists()

    def test_user_files_survive_a_restart(self, run_dd, tmp_path):
        assert run_dd(2, run_config={"buffer_size": 2}) == 0
        (tmp_path / "out" / "notes.txt").write_text("keep me")
        assert run_dd(2, run_config={"buffer_size": 4}) == 0
        assert (tmp_path / "out" / "notes.txt").read_text() == "keep me"

    def test_bad_provider_reports_without_the_key(self, run_dd, capsys):
        bad = [
            {
                "name": "gw",
                "endpont": "http://e",
                "extra_headers_from_env": {"X-Key": "MYKEY"},
            }
        ]
        assert run_dd(1, providers=bad, env={"MYKEY": "sk-FAKE-SECRET"}) == 1
        captured = capsys.readouterr()
        assert "sk-FAKE-SECRET" not in captured.out + captured.err
        assert "Traceback" not in captured.err and "endpoint" in captured.err
