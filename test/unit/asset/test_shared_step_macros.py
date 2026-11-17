#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

"""The shared macros of the steps/train SkyPilot templates: q.j2, which
openinstruct-sft, hf-peft-sft and fms-hf-tuning gb-include, and trainer_args.j2,
which openinstruct-sft and hf-peft-sft do. The steps' own render tests don't run
in CI, so the macros' behaviour is pinned here."""

import pathlib

import pytest

from gbserver.utils.template import fill_template

_SHARED = pathlib.Path(__file__).resolve().parents[3] / "steps" / "train" / "shared"
_Q = (_SHARED / "q.j2").read_text()
_TRAINER_ARGS = (_SHARED / "trainer_args.j2").read_text()


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, ""),
        ("", ""),
        ("a b", "a b"),
        ("it's", "it'\"'\"'s"),
        (2, "2"),
        (False, "False"),
    ],
)
def test_q_quotes_for_a_single_quoted_shell_word(value, expected):
    """null renders as empty (unset), not as the word None; a ' is closed,
    escaped and reopened."""
    assert fill_template(_Q + "{{- q(v) }}", {"v": value}, strict=True) == expected


def _validate_args(args):
    """Run validate_args as a step's validate macro calls it."""
    call = (
        '{{- validate_args("s", "s_config.args", args, ["output_dir"],'
        ' "trainer flag", "output_path", "the trainer") }}'
    )
    return fill_template(_Q + _TRAINER_ARGS + call, {"args": args}, strict=True)


def _refusal(args):
    with pytest.raises(RuntimeError) as exc:
        _validate_args(args)
    return str(exc.value.__cause__)


class TestArgsValuesThatStartWithADash:
    """Each value is its own argv word after the step's own flags, so the trainer's
    argparse would read a word like --output_dir as a flag and the last setting
    would win, overriding the step-owned one (#613)."""

    @pytest.mark.parametrize(
        "args,shown",
        [
            ({"dataset_transform_fn": ["a", "--output_dir", "/evil"]}, "--output_dir"),
            ({"use_flash_attn": "--output_dir=/evil"}, "--output_dir=/evil"),
            ({"x": "-o"}, "-o"),
            ({"x": "--"}, "--"),
            ({"x": ["-1", "-v"]}, "-v"),
        ],
    )
    def test_is_refused(self, args, shown):
        key = next(iter(args))
        assert _refusal(args) == (
            f"s: s_config.args.{key} has a value that starts with '-' ('{shown}'); "
            "the trainer would read it as a flag, which could override a "
            "step-owned one. Only a negative number may start with '-'."
        )

    @pytest.mark.parametrize(
        "value",
        [
            -1,
            -0.5,
            "-1",
            "-0.5",
            "-.5",
            "-",
            "a-b",
            "a --b",
            ["-1", "-.25"],
            True,
            None,
        ],
    )
    def test_negative_numbers_and_inner_dashes_pass(self, value):
        """argparse reads a negative number as a value, a lone '-' too, and a value
        is never split into words, so only a leading dash matters."""
        _validate_args({"x": value})
