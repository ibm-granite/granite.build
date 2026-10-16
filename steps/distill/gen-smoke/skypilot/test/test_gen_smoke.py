"""Unit tests for gen_smoke.py's detector and gate. Generation is injected, so none of
this needs torch, transformers or a GPU."""

import json

import pytest
from gen_smoke import (
    PROMPTS,
    adjacent_rate,
    final_rung_failure,
    looped_fraction,
    measure,
    parse_rungs,
    run,
)

_COLLAPSED = "\n".join(["// (true)"] * 50)
_HEALTHY_JAVA = """\
        int n = 0;
        for (int x : a) {
            if (x > 0) {
                n++;
            }
        }
        return n;
"""


def test_the_collapsed_sample_is_entirely_looped():
    assert looped_fraction(_COLLAPSED.splitlines()) == 1.0


def test_healthy_code_with_repeated_braces_is_not_looped():
    """The case that rules adjacent_rate out as a gate: legitimate repeated lines."""
    looped, adjacent = measure(_HEALTHY_JAVA)
    assert looped == 0.0
    assert adjacent == 0.0  # indentation kept, so the closing braces differ


def test_adjacent_rate_flags_a_pair_that_looped_fraction_tolerates():
    lines = ["}", "}", "x", "y"]
    assert adjacent_rate(lines) == pytest.approx(1 / 3)
    assert looped_fraction(lines) == 0.0


def test_a_run_of_three_counts_and_a_run_of_two_does_not():
    assert looped_fraction(["a", "a", "a", "b"]) == 0.75
    assert looped_fraction(["a", "a", "b", "b"]) == 0.0


def test_empty_and_single_line_outputs_measure_zero():
    assert measure("") == (0.0, 0.0)
    assert measure("one line") == (0.0, 0.0)


def test_blank_lines_are_ignored_and_trailing_space_is_not_a_difference():
    assert looped_fraction([ln.rstrip() for ln in ["x ", "x", "x  "]]) == 1.0
    assert measure("x\n\nx\n\nx\n")[0] == 1.0


@pytest.mark.parametrize("bad", ["500", "x:/p", "500:", ":/p"])
def test_a_malformed_rung_is_refused(bad):
    with pytest.raises(SystemExit, match="is not <step>:<path>"):
        parse_rungs([bad])


def test_no_rungs_is_refused():
    with pytest.raises(SystemExit, match="no rungs"):
        parse_rungs([])


def test_a_path_may_contain_colons():
    assert parse_rungs(["25:/a:b"]) == [("25", "/a:b")]


def _fake_generate(degenerate_paths):
    def generate(path, new_tokens):
        text = _COLLAPSED if path in degenerate_paths else _HEALTHY_JAVA
        return [text] * len(PROMPTS)

    return generate


def _argv(out, gate, *rungs):
    return ["x", str(out), "0.15", "16", gate, *rungs]


def test_only_the_final_rung_gates(tmp_path, capsys):
    out = tmp_path / "r.json"
    rc = run(_argv(out, "true", "25:/early", "500:/late"), _fake_generate({"/early"}))
    assert rc == 0
    report = json.loads(out.read_text())
    assert [r["degenerate"] for r in report["rungs"]] == [True, False]
    assert f"GB_ARTIFACT_ID:repetition_report GB_ARTIFACT_PATH:{out}" in (
        capsys.readouterr().out
    )


def test_a_degenerate_final_rung_fails_when_gated(tmp_path, capsys):
    rc = run(
        _argv(tmp_path / "r.json", "true", "25:/a", "500:/b"), _fake_generate({"/b"})
    )
    assert rc == 1
    assert "GEN-SMOKE FATAL: the final checkpoint (step 500)" in capsys.readouterr().err


def test_a_degenerate_final_rung_only_reports_when_not_gated(tmp_path, capsys):
    """distill-checkpoint-eval's mode: measure the whole ladder and act on none of it."""
    out = tmp_path / "r.json"
    rc = run(_argv(out, "false", "500:/b"), _fake_generate({"/b"}))
    assert rc == 0
    assert json.loads(out.read_text())["rungs"][0]["degenerate"] is True
    assert "FATAL" not in capsys.readouterr().err


def test_the_gate_switch_must_be_spelled_out(tmp_path):
    with pytest.raises(SystemExit, match="gate_final_rung must be true or false"):
        run(_argv(tmp_path / "r.json", "yes", "1:/a"), _fake_generate(set()))


def test_final_rung_failure_names_the_step_and_threshold():
    msg = final_rung_failure(
        [{"step": 7, "degenerate": True, "mean_looped_fraction": 0.5}], 0.15
    )
    assert "step 7" in msg and "50.0%" in msg and "15.0%" in msg
    assert final_rung_failure([{"degenerate": False}], 0.15) is None


def test_the_prompts_are_raw_completions():
    """MultiPL-E is where the collapse showed worst and it applies no chat template, so
    a templated prompt set would test a different thing from the benchmark that caught
    this."""
    for prompt in PROMPTS:
        assert "<|start_of_role|>" not in prompt
        assert "<|im_start|>" not in prompt


def test_decoding_is_greedy():
    """Sampling would hide the failure. Temperature is what lets a narrowed
    distribution still look varied, and df8512e0's MultiPL-E ran at temperature 0.2 on
    a model with an effective branching factor of 1.49 -- effectively greedy already,
    which is why it degenerated there and not elsewhere. generate_rung needs a GPU, so
    this reads its source."""
    import inspect

    import gen_smoke

    source = inspect.getsource(gen_smoke.generate_rung)
    assert "do_sample=False" in source
