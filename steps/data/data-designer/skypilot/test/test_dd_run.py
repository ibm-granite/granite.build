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

import pytest

import dd_run
from dd_run import StepError


def _b64(obj) -> str:
    return base64.b64encode(json.dumps(obj).encode()).decode()


_CONFIG = {"columns": [{"name": "a", "column_type": "sampler"}]}
_PROVIDERS = [{"name": "rits", "endpoint": "https://e/v1"}]


class TestLoadSpec:
    def test_round_trip(self):
        spec = dd_run.load_spec({dd_run.SPEC_ENV: _b64({"config": _CONFIG, "model_providers": _PROVIDERS})})
        assert spec == {"config": _CONFIG, "model_providers": _PROVIDERS, "run_config": {}, "local_model": {}}

    def test_run_config_passes_through(self):
        rc = {"buffer_size": 2, "max_concurrent_row_groups": 1}
        spec = dd_run.load_spec(
            {dd_run.SPEC_ENV: _b64({"config": _CONFIG, "model_providers": _PROVIDERS, "run_config": rc})}
        )
        assert spec["run_config"] == rc

    def test_run_config_must_be_mapping(self):
        with pytest.raises(StepError, match="run_config must be a mapping"):
            dd_run.load_spec(
                {dd_run.SPEC_ENV: _b64({"config": _CONFIG, "model_providers": _PROVIDERS, "run_config": [1]})}
            )

    @pytest.mark.parametrize(
        "raw,needle",
        [
            ("", "is empty"),
            ("not base64!!", "not base64-encoded JSON"),
            (_b64([1, 2]), "must decode to an object"),
            (_b64({"config": {}, "model_providers": _PROVIDERS}), "config must be a non-empty"),
            (_b64({"config": _CONFIG, "model_providers": []}), "model_providers must be a non-empty"),
            (_b64({"config": _CONFIG, "model_providers": _PROVIDERS, "local_model": [1]}), "local_model must be a mapping"),
        ],
    )
    def test_refusals(self, raw, needle):
        with pytest.raises(StepError, match=needle):
            dd_run.load_spec({dd_run.SPEC_ENV: raw})


class TestLocalModel:
    _OLLAMA = {"runtime": "ollama", "model": "granite4:350m", "port": 11434, "name": "local"}

    def test_empty_providers_allowed_with_local_model(self):
        spec = dd_run.load_spec({dd_run.SPEC_ENV: _b64(
            {"config": _CONFIG, "model_providers": [], "local_model": self._OLLAMA})})
        assert spec["model_providers"] == [] and spec["local_model"] == self._OLLAMA

    def test_provider_points_at_localhost(self):
        assert dd_run.local_model_provider(self._OLLAMA, []) == {
            "name": "local", "endpoint": "http://127.0.0.1:11434/v1", "api_key": "ollama"}

    def test_port_and_name_overrides(self):
        p = dd_run.local_model_provider({**self._OLLAMA, "port": 12000, "name": "small"}, [])
        assert p["endpoint"] == "http://127.0.0.1:12000/v1" and p["name"] == "small"

    def test_off_returns_none(self):
        assert dd_run.local_model_provider({"runtime": ""}, _PROVIDERS) is None
        assert dd_run.local_model_provider({}, _PROVIDERS) is None

    def test_unknown_runtime(self):
        with pytest.raises(StepError, match="must be empty or 'ollama'"):
            dd_run.local_model_provider({**self._OLLAMA, "runtime": "vllm"}, [])

    def test_name_collision_with_a_provider(self):
        with pytest.raises(StepError, match="also a model_providers name"):
            dd_run.local_model_provider(self._OLLAMA, [{"name": "local", "endpoint": "e"}])


class TestBuildProviders:
    def test_header_from_env_merges(self):
        p = [{"name": "rits", "endpoint": "e", "extra_headers": {"X-A": "1"},
              "extra_headers_from_env": {"RITS_API_KEY": "MY_KEY"}}]
        out = dd_run.build_providers(p, {"MY_KEY": "s3cret"})
        assert out == [{"name": "rits", "endpoint": "e",
                        "extra_headers": {"X-A": "1", "RITS_API_KEY": "s3cret"}}]

    def test_api_key_env(self):
        out = dd_run.build_providers([{"name": "n", "endpoint": "e", "api_key_env": "K"}], {"K": "v"})
        assert out[0]["api_key"] == "v" and "api_key_env" not in out[0]

    def test_input_not_mutated(self):
        p = [{"name": "n", "endpoint": "e", "api_key_env": "K"}]
        dd_run.build_providers(p, {"K": "v"})
        assert p == [{"name": "n", "endpoint": "e", "api_key_env": "K"}]

    def test_missing_variable_names_it_but_not_a_value(self):
        p = [{"name": "rits", "endpoint": "e", "extra_headers_from_env": {"RITS_API_KEY": "RITS_API_KEY"}}]
        with pytest.raises(StepError) as e:
            dd_run.build_providers(p, {"OTHER": "s3cret"})
        assert "$RITS_API_KEY" in str(e.value)
        assert "secret_names_to_use_as_env_variable" in str(e.value)
        assert "s3cret" not in str(e.value)

    @pytest.mark.parametrize(
        "entry,needle",
        [
            ({"name": "n", "endpoint": "e", "api_key": "x", "api_key_env": "K"}, "both api_key and api_key_env"),
            ({"name": "n", "endpoint": "e", "api_key_env": "bad-name"}, "must name an environment variable"),
            ({"name": "n", "endpoint": "e", "extra_headers_from_env": ["K"]}, "must map header"),
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
        cfg = {"columns": [], "seed_config": {"source": {"seed_type": seed_type, "path": "orig.parquet"}}}
        return {"data_designer": cfg} if full else cfg

    def test_no_seed_is_identity(self):
        cfg = self._seeded()
        assert dd_run.apply_seed_path(cfg, "", "") is cfg

    @pytest.mark.parametrize("full", [False, True])
    def test_replaces_path_in_both_forms(self, full):
        out = dd_run.apply_seed_path(self._seeded(full=full), "/staged/seed/", "*.parquet")
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

    def _prior_run(self, tmp_path, config_hash="sha256:old"):
        out = tmp_path / "out"
        (out / "parquet-files").mkdir(parents=True)
        (out / "parquet-files" / "batch_00000.parquet").write_text("x")
        (out / "metadata.json").write_text(json.dumps({"config_hash": config_hash}))
        return out

    def test_changed_config_is_cleared_under_if_possible(self, tmp_path):
        """Data Designer would otherwise start fresh in a timestamped sibling dir."""
        out = self._prior_run(tmp_path)
        assert dd_run.clear_stale_output(out, "if_possible", "sha256:new") is True
        assert not out.exists()

    @pytest.mark.parametrize(
        "resume,current", [("if_possible", "sha256:old"), ("always", "sha256:new"), ("never", "sha256:new")]
    )
    def test_kept_otherwise(self, tmp_path, resume, current):
        """Same config resumes; `always` must reach Data Designer's own refusal."""
        out = self._prior_run(tmp_path)
        assert dd_run.clear_stale_output(out, resume, current) is False
        assert (out / "parquet-files" / "batch_00000.parquet").exists()

    def test_resumable_records_counts_checkpointed_rows(self, tmp_path):
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
        batches = tmp_path / "out" / "parquet-files"
        batches.mkdir(parents=True)
        for i, n in enumerate([2, 2, 1]):
            pq.write_table(pa.table({"q": ["x"] * n}), batches / f"batch_{i:05d}.parquet")
        out = tmp_path / "out"
        assert dd_run.count_resumable_records(out, "if_possible") == 5
        assert dd_run.count_resumable_records(out, "always") == 5
        assert dd_run.count_resumable_records(out, "never") == 0
        assert dd_run.count_resumable_records(tmp_path / "absent", "if_possible") == 0

    def test_non_data_designer_dir_never_cleared(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        (out / "precious.txt").write_text("x")
        assert dd_run.clear_stale_output(out, "if_possible", "sha256:new") is False
        assert (out / "precious.txt").exists()

    @pytest.mark.parametrize("bad", ["", "a b", 'a"b'])
    def test_marker_id(self, bad, tmp_path):
        with pytest.raises(StepError):
            dd_run.check_marker_values(bad, tmp_path)

    def test_marker_path_quote(self, tmp_path):
        with pytest.raises(StepError, match="double quote"):
            dd_run.check_marker_values("ok", tmp_path / 'a"b')

    def test_main_reports_one_error_line(self, capsys, monkeypatch, tmp_path):
        monkeypatch.delenv(dd_run.SPEC_ENV, raising=False)
        rc = dd_run.main(["--output-path", str(tmp_path / "o"), "--artifact-id", "q",
                          "--num-records", "1", "--resume", "never"])
        assert rc == 1
        err = capsys.readouterr().err
        assert err.startswith("data-designer: ERROR") and "Traceback" not in err
