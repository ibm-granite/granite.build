"""The bfcl step's FC output budget (bfcl_serving): when the served vLLM gets the
max_new_tokens default, and what results.json records. No network, no model."""

import json

import pytest

from granite_evals.benchmarks import bfcl, bfcl_serving

BUDGET = '--override-generation-config={"max_new_tokens":16384}'


@pytest.fixture
def ckpt(tmp_path):
    """A local checkpoint dir; write(config) sets its generation_config.json."""
    d = tmp_path / "ckpt"
    d.mkdir()

    def write(config):
        (d / "generation_config.json").write_text(json.dumps(config) if isinstance(config, dict) else config)
        return str(d)

    write.path = str(d)
    return write


@pytest.mark.parametrize("options, handler", [
    ("", None),
    ("handler=legacy-granite", "legacy-granite"),
    ("handler=legacy-granite temperature=0.001 handler=fc", "fc"),  # the last one wins, as the CLI
    ("handler=fc\thandler=legacy-granite", "legacy-granite"),  # any whitespace separates
    ("temperature=0.001\n  handler=legacy-granite\t", "legacy-granite"),
    ("handlers=legacy-granite", None),
])
def test_last_option_is_the_clis(options, handler):
    assert bfcl_serving.last_option(options, "handler") == handler


def test_fc_without_a_checkpoint_limit_gets_the_budget(ckpt):
    for config in ({}, {"temperature": 1.0, "top_p": 0.95}):  # granite-4.2-3b sets these, no limit
        assert bfcl_serving.plan(ckpt(config), "", "") == ("16384:step-default", [BUDGET])


def test_no_generation_config_file_gets_the_budget(tmp_path):
    (tmp_path / "ckpt").mkdir()
    assert bfcl_serving.plan(str(tmp_path / "ckpt"), "", "categories=simple") == ("16384:step-default", [BUDGET])


@pytest.mark.parametrize("options", ["handler=legacy-granite", "handler=fc handler=legacy-granite",
                                     "temperature=0.001\thandler=legacy-granite"])
def test_legacy_granite_is_never_served_the_budget(ckpt, options):
    assert bfcl_serving.plan(ckpt({}), "", options) == ("none:legacy-granite", [])


def test_the_last_handler_fc_is_served_the_budget(ckpt):
    assert bfcl_serving.plan(ckpt({}), "", "handler=legacy-granite temperature=0.001 handler=fc")[1] == [BUDGET]


def test_base_url_serves_nothing(ckpt):
    assert bfcl_serving.plan(ckpt({}), "http://host:8000/v1", "") == ("none:base_url", [])


def test_an_explicit_max_tokens_is_sent_not_served(ckpt):
    assert bfcl_serving.plan(ckpt({}), "", "max_tokens=100 temperature=0.5") == ("100:max_tokens-option", [])


def test_the_checkpoints_own_limit_is_kept(ckpt):
    # Both directions: a 4096 cap is not raised, a 65536 one not lowered.
    assert bfcl_serving.plan(ckpt({"max_new_tokens": 4096}), "", "") == ("4096:checkpoint-max_new_tokens", [])
    assert bfcl_serving.plan(ckpt({"max_new_tokens": 65536}), "", "") == ("65536:checkpoint-max_new_tokens", [])
    assert bfcl_serving.plan(ckpt({"max_length": 8192}), "", "") == ("context-left:checkpoint-max_length", [])


def test_an_unreadable_generation_config_is_recorded_not_guessed(ckpt):
    record, args = bfcl_serving.plan(ckpt("{not json"), "", "")
    assert record == "unknown:generation_config-unreadable-JSONDecodeError" and args == []


def test_a_hub_id_is_read_through_the_hub_cache(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub.errors import EntryNotFoundError

    f = tmp_path / "generation_config.json"
    f.write_text(json.dumps({"max_new_tokens": 2048}))
    seen = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name: seen.append((repo, name)) or str(f))
    assert bfcl_serving.plan("ibm-granite/some-model", "", "") == ("2048:checkpoint-max_new_tokens", [])
    assert seen == [("ibm-granite/some-model", "generation_config.json")]

    def missing(repo, name):
        raise EntryNotFoundError("no generation_config.json")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", missing)
    assert bfcl_serving.plan("ibm-granite/some-model", "", "") == ("16384:step-default", [BUDGET])

    def offline(repo, name):
        raise OSError("offline")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", offline)
    assert bfcl_serving.plan("ibm-granite/some-model", "", "") == ("unknown:generation_config-unreadable-OSError", [])


def test_main_prints_the_record_then_the_arguments(ckpt, capsys):
    assert bfcl_serving.main([ckpt({}), "", "handler=fc"]) == 0
    assert capsys.readouterr().out.splitlines() == ["fc_budget=16384:step-default", BUDGET]
    bfcl_serving.main([ckpt({}), "", "handler=legacy-granite"])
    assert capsys.readouterr().out.splitlines() == ["fc_budget=none:legacy-granite"]


@pytest.mark.parametrize("record, cap", [
    ("16384:step-default", 16384), ("4096:checkpoint-max_new_tokens", 4096), ("none:base_url", "none"),
    ("context-left:checkpoint-max_length", "context-left"), (None, "unknown"),
])
def test_describe(record, cap):
    out = bfcl_serving.describe(record)
    assert out["cap"] == cap and out["source"]


def test_the_score_phase_reports_what_generate_recorded(tmp_path):
    rec = bfcl.budget_record(tmp_path, {"fc_budget": "16384:step-default"}, generating=True)
    assert rec["cap"] == 16384 and "served default" in rec["source"]
    assert bfcl.budget_record(tmp_path, {}, generating=False) == rec  # score: no option, the file
    assert bfcl.budget_record(tmp_path, {"max_tokens": "100"}, generating=False)["cap"] == 100
    assert bfcl.budget_record(tmp_path / "nowhere", {}, generating=False)["cap"] == "unknown"
