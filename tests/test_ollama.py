import pytest

from auto_answer.errors import SolverError
from auto_answer.ollama import (
    numeric_option_values,
    parse_answer_index,
    parse_structured_index,
    parse_structured_numeric_answer,
)


@pytest.mark.parametrize(("content", "expected"), [("0", 0), (" 3\n", 3)])
def test_parse_answer_index_accepts_only_single_index(content: str, expected: int) -> None:
    assert parse_answer_index(content) == expected


@pytest.mark.parametrize("content", ["", "4", "答案是1", "1。", None])
def test_parse_answer_index_rejects_non_strict_output(content: object) -> None:
    with pytest.raises(SolverError):
        parse_answer_index(content)


def test_detects_only_four_pure_numeric_options() -> None:
    assert numeric_option_values(("1", "-2", "3.5", "04")) is not None
    assert numeric_option_values(("1元", "2元", "3元", "4元")) is None
    assert numeric_option_values(("2+1", "4+1", "5+0", "2+3")) is None


def test_structured_numeric_answer_must_match_index_uniquely() -> None:
    index, calculation = parse_structured_numeric_answer(
        '{"calculation":"8*4=32","answer_value":32,"answer_index":1}',
        ("24", "32", "40", "16"),
    )
    assert index == 1
    assert calculation == "8*4=32"


@pytest.mark.parametrize(
    "content",
    [
        '{"calculation":"8*4=32","answer_value":32,"answer_index":2}',
        '{"calculation":"2+2=4","answer_value":4,"answer_index":0}',
        '{"calculation":"2+2=4","answer_value":4,"answer_index":4}',
        '{"answer_value":4,"answer_index":0}',
    ],
)
def test_structured_numeric_answer_rejects_inconsistent_or_ambiguous_output(
    content: str,
) -> None:
    with pytest.raises(SolverError):
        parse_structured_numeric_answer(content, ("4", "4", "3", "5"))


def test_structured_text_answer_accepts_only_index_object() -> None:
    assert parse_structured_index('{"answer_index":3}') == 3
    with pytest.raises(SolverError):
        parse_structured_index('{"answer_index":3,"answer_value":"正确"}')
    with pytest.raises(SolverError):
        parse_structured_index('{"answer_index":4}')
