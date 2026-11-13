# Copyright Granite.Build Authors
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

"""Pure parsing-strategy helpers: application + JS-regex compatibility."""

from __future__ import annotations

import pytest

from autotunex.core.exceptions import DomainValidationError
from autotunex.models.dataset_intelligence import ParsingStrategy
from autotunex.services.parsing_strategy import apply_parsing_strategy, validate_js_regex


def test_direct_mapping_pulls_input_and_output_fields() -> None:
    strategy = ParsingStrategy(type="direct_mapping", input_field="q", output_field="a")

    pairs, errors = apply_parsing_strategy([{"q": "hi", "a": "yo"}], strategy)

    assert pairs == [{"input": "hi", "output": "yo"}]
    assert errors == []


def test_direct_mapping_reports_missing_fields_without_raising() -> None:
    strategy = ParsingStrategy(type="direct_mapping", input_field="q", output_field="a")

    pairs, errors = apply_parsing_strategy([{"q": "hi"}], strategy)

    assert pairs == []
    assert len(errors) == 1


def test_direct_mapping_guards_non_dict_rows() -> None:
    strategy = ParsingStrategy(type="direct_mapping", input_field="q", output_field="a")

    pairs, errors = apply_parsing_strategy(["not a dict"], strategy)  # type: ignore[list-item]

    assert pairs == []
    assert len(errors) == 1


def test_regex_pairs_matches_over_text() -> None:
    strategy = ParsingStrategy(type="regex", input_pattern=r"Q: (\w+)", output_pattern=r"A: (\w+)")

    pairs, errors = apply_parsing_strategy("Q: hello\nA: world", strategy)

    assert pairs == [{"input": "hello", "output": "world"}]
    assert errors == []


def test_regex_reports_when_nothing_matches() -> None:
    strategy = ParsingStrategy(type="regex", input_pattern=r"Q: (\w+)", output_pattern=r"A: (\w+)")

    pairs, errors = apply_parsing_strategy("nothing here", strategy)

    assert pairs == []
    assert errors


def test_transformation_is_reported_unsupported() -> None:
    strategy = ParsingStrategy(type="transformation")

    pairs, errors = apply_parsing_strategy([{"q": "x"}], strategy)

    assert pairs == []
    assert any("not supported" in error for error in errors)


def test_validate_js_regex_accepts_a_portable_pattern() -> None:
    validate_js_regex(r"(\w+)\s+(\d+)")  # no raise


@pytest.mark.parametrize(
    "pattern",
    [
        r"(?P<name>\w+)",  # Python named group
        r"(?i)hello",  # inline global flag
        r"(?i:hello)",  # scoped inline flag
        r"\Astart",  # \A anchor
        r"end\Z",  # \Z anchor
    ],
)
def test_validate_js_regex_rejects_python_only_constructs(pattern: str) -> None:
    with pytest.raises(DomainValidationError):
        validate_js_regex(pattern)


def test_validate_js_regex_rejects_an_uncompilable_pattern() -> None:
    with pytest.raises(DomainValidationError):
        validate_js_regex(r"(unclosed")
