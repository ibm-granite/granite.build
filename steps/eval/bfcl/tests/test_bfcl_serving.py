"""The bfcl step's FC output budget (bfcl_serving): when the served vLLM gets the
max_new_tokens default, and what results.json records. No network, no model."""

import json
import shlex

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
        assert bfcl_serving.plan("bfcl-v3", ckpt(config), "", "") == ("16384:step-default", [BUDGET])


def test_no_generation_config_file_gets_the_budget(tmp_path):
    (tmp_path / "ckpt").mkdir()
    assert bfcl_serving.plan("bfcl-v3", str(tmp_path / "ckpt"), "", "categories=simple") == ("16384:step-default", [BUDGET])


@pytest.mark.parametrize("options", ["handler=legacy-granite", "handler=fc handler=legacy-granite",
                                     "temperature=0.001\thandler=legacy-granite"])
def test_legacy_granite_is_never_served_the_budget(ckpt, options):
    assert bfcl_serving.plan("bfcl-v3", ckpt({}), "", options) == ("none:legacy-granite", [])


@pytest.mark.parametrize("options", ["handler=legacy-granite", "handler=fc", ""])
def test_bfcl_v4_has_no_handler_option_so_it_never_suppresses_the_budget(ckpt, options):
    # bfcl-v4 always runs FC (it never reads handler=): the budget stays.
    assert bfcl_serving.plan("bfcl-v4", ckpt({}), "", options) == ("16384:step-default", [BUDGET])


def test_the_last_handler_fc_is_served_the_budget(ckpt):
    assert bfcl_serving.plan("bfcl-v3", ckpt({}), "", "handler=legacy-granite temperature=0.001 handler=fc")[1] == [BUDGET]


def test_base_url_serves_nothing(ckpt):
    assert bfcl_serving.plan("bfcl-v3", ckpt({}), "http://host:8000/v1", "") == ("none:base_url", [])


def test_an_explicit_max_tokens_is_sent_not_served(ckpt):
    assert bfcl_serving.plan("bfcl-v3", ckpt({}), "", "max_tokens=100 temperature=0.5") == ("100:max_tokens-option", [])


def test_the_checkpoints_own_limit_is_kept(ckpt):
    # Both directions: a 4096 cap is not raised, a 65536 one not lowered.
    assert bfcl_serving.plan("bfcl-v3", ckpt({"max_new_tokens": 4096}), "", "") == ("4096:checkpoint-max_new_tokens", [])
    assert bfcl_serving.plan("bfcl-v3", ckpt({"max_new_tokens": 65536}), "", "") == ("65536:checkpoint-max_new_tokens", [])
    assert bfcl_serving.plan("bfcl-v3", ckpt({"max_length": 8192}), "", "") == ("context-left:checkpoint-max_length", [])


# Every form vLLM 0.27.1's serve parser reads as the generation config (checked with its
# FlexibleArgumentParser: underscores, unique prefixes, dotted keys, a --config file).
OVERRIDES = [
    '--override-generation-config={"temperature":0.5}',  # replaces the step's JSON override
    '--override-generation-config {"max_new_tokens":3}',
    "--override_generation_config={}",
    "--override-generation-config.max_new_tokens=4",
    "--override-generation-config.temperature 0.5",
    "--override-gen={}",
    "--generation-config=vllm",
    "--generation_config vllm",
    "--generation-conf=/models/other",
    "--config serve.yaml",
    "--config=serve.yaml",
]


@pytest.mark.parametrize("vllm_args", OVERRIDES)
def test_a_generation_config_in_vllm_args_is_recorded_not_overridden(ckpt, vllm_args):
    args = shlex.split(f"--max-num-seqs=4 {vllm_args} --enforce-eager")
    assert bfcl_serving.plan("bfcl-v3", ckpt({}), "", "", args) == ("unknown:vllm_args-override", [])
    assert bfcl_serving.describe("unknown:vllm_args-override")["source"].startswith("vllm_args")


@pytest.mark.parametrize("vllm_args", ["", "--max-num-seqs=4 --enforce-eager", "--config-format=hf",
                                       "--generation-config-x=1", "--max-model-len=8192"])
def test_other_vllm_args_keep_the_budget(ckpt, vllm_args):
    assert bfcl_serving.plan("bfcl-v3", ckpt({}), "", "", shlex.split(vllm_args)) == ("16384:step-default", [BUDGET])


def test_a_vllm_args_override_does_not_change_a_cap_sent_with_every_request(ckpt):
    over = ["--override-generation-config={}"]
    assert bfcl_serving.plan("bfcl-v3", ckpt({}), "", "max_tokens=100", over)[0] == "100:max_tokens-option"
    assert bfcl_serving.plan("bfcl-v3", ckpt({}), "http://h/v1", "", over)[0] == "none:base_url"


def test_an_unreadable_generation_config_is_recorded_not_guessed(ckpt):
    record, args = bfcl_serving.plan("bfcl-v3", ckpt("{not json"), "", "")
    assert record == "unknown:generation_config-unreadable-JSONDecodeError" and args == []


@pytest.mark.parametrize("text", ["[]", '"max_new_tokens"', "4096", "null"])
def test_a_generation_config_that_is_not_an_object_is_recorded_not_guessed(ckpt, text):
    assert bfcl_serving.plan("bfcl-v3", ckpt(text), "", "") == ("unknown:generation_config-unreadable-TypeError", [])


def _hub_raises(monkeypatch, exc):
    import huggingface_hub

    def download(repo, name):
        raise exc

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)


def test_a_hub_id_is_read_through_the_hub_cache(tmp_path, monkeypatch):
    import huggingface_hub

    f = tmp_path / "generation_config.json"
    f.write_text(json.dumps({"max_new_tokens": 2048}))
    seen = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name: seen.append((repo, name)) or str(f))
    assert bfcl_serving.plan("bfcl-v3", "ibm-granite/some-model", "", "") == ("2048:checkpoint-max_new_tokens", [])
    assert seen == [("ibm-granite/some-model", "generation_config.json")]


def test_only_a_file_the_hub_says_is_absent_means_no_config(monkeypatch):
    # The real classes: RemoteEntryNotFoundError is the Hub's 404 for the file (the repo has
    # none); LocalEntryNotFoundError, also an EntryNotFoundError, is raised when the Hub could
    # not be asked (offline, a timeout, HF_HUB_OFFLINE without the file cached), so the
    # checkpoint may well set a limit: unknown, never the budget.
    from types import SimpleNamespace

    from huggingface_hub.errors import LocalEntryNotFoundError, RemoteEntryNotFoundError

    response = SimpleNamespace(headers={}, request=None)  # all the constructor reads
    _hub_raises(monkeypatch, RemoteEntryNotFoundError("404 generation_config.json", response=response))
    assert bfcl_serving.plan("bfcl-v3", "ibm-granite/some-model", "", "") == ("16384:step-default", [BUDGET])
    _hub_raises(monkeypatch, LocalEntryNotFoundError("cannot reach the Hub, file not in the cache"))
    assert bfcl_serving.plan("bfcl-v3", "ibm-granite/some-model", "", "") == (
        "unknown:generation_config-unreadable-LocalEntryNotFoundError", [])
    _hub_raises(monkeypatch, OSError("offline"))
    assert bfcl_serving.plan("bfcl-v3", "ibm-granite/some-model", "", "") == ("unknown:generation_config-unreadable-OSError", [])


def test_main_records_the_decision_and_prints_the_arguments(ckpt, tmp_path, capsys):
    out = tmp_path / "out" / "fc_budget.json"
    out.parent.mkdir()
    assert bfcl_serving.main([str(out), "bfcl-v3", ckpt({}), "", "handler=fc"]) == 0
    assert capsys.readouterr().out.splitlines() == [BUDGET]
    assert json.loads(out.read_text()) == {"fc_budget": "16384:step-default"}
    bfcl_serving.main([str(out), "bfcl-v3", ckpt({}), "", "handler=legacy-granite"])  # a rerun replaces it
    assert capsys.readouterr().out == ""
    assert json.loads(out.read_text()) == {"fc_budget": "none:legacy-granite"}
    bfcl_serving.main([str(out), "bfcl-v3", ckpt({}), "", "", "--max-num-seqs=4", "--generation-config", "vllm"])
    assert capsys.readouterr().out == ""
    assert json.loads(out.read_text()) == {"fc_budget": "unknown:vllm_args-override"}


@pytest.mark.parametrize("record, cap", [
    ("16384:step-default", 16384), ("4096:checkpoint-max_new_tokens", 4096), ("none:base_url", "none"),
    ("context-left:checkpoint-max_length", "context-left"), (None, "unknown"),
])
def test_describe(record, cap):
    out = bfcl_serving.describe(record)
    assert out["cap"] == cap and out["source"]


def test_an_output_dir_without_a_record_says_why():
    # Generated by a bfcl step from before fc_budget.json, or by granite-evals run outside it.
    assert bfcl_serving.describe(None)["source"] == (
        "not recorded (generated by an earlier step version, or outside the bfcl step)")


def test_both_phases_report_the_recorded_decision(tmp_path):
    # Generate and score read the step's record from the output dir; nothing comes from an
    # option (the score phase's options would differ from the generation's).
    (tmp_path / bfcl.FC_BUDGET_FILE).write_text(json.dumps({"fc_budget": "16384:step-default"}))
    rec = bfcl.budget_record(tmp_path, {})
    assert rec["cap"] == 16384 and "served default" in rec["source"]
    assert bfcl.budget_record(tmp_path, {"fc_budget": "none:base_url"}) == rec  # not an input any more
    assert bfcl.budget_record(tmp_path, {"max_tokens": "100"})["cap"] == 100
    assert bfcl.budget_record(tmp_path / "nowhere", {})["cap"] == "unknown"
