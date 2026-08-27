"""Tests for autotune.rewards.gsm8k_reward — pure regex/numeric scoring logic."""

import pytest

from autotune.rewards.gsm8k_reward import (
    CORRECT_REWARD,
    FORMAT_BONUS,
    MAX_LENGTH_PENALTY,
    WRONG_REWARD,
    _normalize_number_str,
    _score_one,
    compute_score,
    extract_final_answer,
)


class TestNormalizeNumberStr:
    def test_plain_integer(self):
        assert _normalize_number_str("42") == "42"

    def test_negative(self):
        assert _normalize_number_str("-5") == "-5"

    def test_decimal(self):
        assert _normalize_number_str("3.14") == "3.14"

    def test_strips_commas(self):
        assert _normalize_number_str("1,234") == "1234"
        assert _normalize_number_str("1,234.56") == "1234.56"

    def test_strips_dollar_sign(self):
        assert _normalize_number_str("$100") == "100"

    def test_extracts_number_from_text(self):
        # If not a clean fullmatch, search for first number
        assert _normalize_number_str("answer is 42 dollars") == "42"

    def test_returns_none_for_no_number(self):
        assert _normalize_number_str("abc") is None

    def test_returns_none_for_none(self):
        assert _normalize_number_str(None) is None

    def test_accepts_int_input(self):
        assert _normalize_number_str(42) == "42"

    def test_accepts_float_input(self):
        assert _normalize_number_str(3.14) == "3.14"


class TestExtractFinalAnswer:
    def test_hash_marker(self):
        assert extract_final_answer("Reasoning blah blah\n#### 42") == "42"

    def test_hash_marker_negative(self):
        assert extract_final_answer("#### -5") == "-5"

    def test_hash_marker_with_commas(self):
        assert extract_final_answer("#### 1,234") == "1234"

    def test_fallback_last_number(self):
        # No #### marker → last number wins
        assert extract_final_answer("Step 1: 10. Step 2: 20. Final: 30.") == "30"

    def test_empty_string(self):
        assert extract_final_answer("") is None

    def test_none_input(self):
        assert extract_final_answer(None) is None

    def test_no_numbers_at_all(self):
        assert extract_final_answer("hello world") is None


class TestScoreOne:
    def test_correct_with_hash_format(self):
        score = _score_one("The answer is #### 42", "42")
        # Correct answer + format bonus minus tiny length penalty
        assert score > CORRECT_REWARD  # bonus pushed it above base reward
        assert score <= CORRECT_REWARD + FORMAT_BONUS

    def test_correct_without_hash_format(self):
        score = _score_one("The answer is 42", "42")
        # No format bonus
        assert score <= CORRECT_REWARD
        assert score > 0  # still positive overall (correct dominates short response)

    def test_wrong_answer(self):
        score = _score_one("The answer is 7", "42")
        assert score <= WRONG_REWARD + FORMAT_BONUS
        assert score < 0

    def test_no_prediction_extractable(self):
        score = _score_one("hello world", "42")
        # No prediction → wrong, no bonus, small length penalty
        assert score < 0
        assert score <= WRONG_REWARD

    def test_none_response(self):
        score = _score_one(None, "42")
        assert score == WRONG_REWARD  # no bonus, no length penalty

    def test_length_penalty_capped(self):
        long_response = "x" * 100_000  # very long, no #### or numbers
        score = _score_one(long_response, "42")
        # The penalty is capped at MAX_LENGTH_PENALTY
        assert score >= WRONG_REWARD - MAX_LENGTH_PENALTY - 1e-9


class TestComputeScore:
    def test_correct(self):
        score = compute_score(
            data_source="gsm8k",
            solution_str="#### 42",
            ground_truth="42",
        )
        assert score > 0

    def test_wrong(self):
        score = compute_score(
            data_source="gsm8k",
            solution_str="#### 7",
            ground_truth="42",
        )
        assert score < 0

    def test_none_ground_truth_returns_wrong(self):
        score = compute_score(
            data_source="gsm8k",
            solution_str="#### 42",
            ground_truth=None,
        )
        # gt is None, so prediction can't match → wrong
        assert score <= WRONG_REWARD + FORMAT_BONUS

    def test_extra_kwargs_accepted(self):
        # verl may pass extra kwargs we don't use
        compute_score(
            data_source="x",
            solution_str="#### 1",
            ground_truth="1",
            extra_info={"split": "test"},
            unknown_kwarg="ignored",
        )

    def test_int_ground_truth(self):
        score = compute_score(solution_str="#### 42", ground_truth=42)
        assert score > 0


@pytest.mark.parametrize(
    "response,gt,should_be_positive",
    [
        ("#### 100", "100", True),
        ("#### 100", 100, True),
        ("#### 100", "$100", True),
        ("#### 1,000", "1000", True),
        ("#### 99", "100", False),
    ],
)
def test_score_parametrized(response, gt, should_be_positive):
    score = _score_one(response, gt)
    if should_be_positive:
        assert score > 0
    else:
        assert score < 0
