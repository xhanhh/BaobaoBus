import json

import httpx
import pytest

from auto_answer.core.config import OllamaConfig
from auto_answer.core.errors import SolverError
from auto_answer.core.models import OCRBundle, OCRResult, Question
from auto_answer.solving.ollama import (
    OllamaClient,
    numeric_option_values,
    parse_answer_index,
    parse_structured_index,
    parse_structured_numeric_answer,
)


def question(text: str, options: tuple[str, str, str, str]) -> Question:
    empty = OCRResult("", 1.0, False)
    return Question(text, options, OCRBundle(empty, (empty, empty, empty, empty)))


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
        '{"calculation":"8*6=48","answer_value":9,"answer_index":1}',
        '{"calculation":"8*6=9","answer_value":9,"answer_index":1}',
        '{"calculation":"8*6","answer_value":9,"answer_index":1}',
        '{"calculation":"unknown=9","answer_value":9,"answer_index":1}',
    ],
)
def test_structured_numeric_answer_rejects_unverified_calculation(
    content: str,
) -> None:
    with pytest.raises(SolverError):
        parse_structured_numeric_answer(content, ("4", "9", "5", "7"))


def test_structured_numeric_answer_accepts_unicode_operators() -> None:
    index, calculation = parse_structured_numeric_answer(
        '{"calculation":"2×5+3×1=13","answer_value":13,"answer_index":1}',
        ("23", "13", "10", "50"),
    )
    assert index == 1
    assert calculation == "2×5+3×1=13"


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


def test_numeric_failure_retries_once_in_text_mode() -> None:
    requests: list[dict[str, object]] = []
    responses = iter(
        (
            '{"calculation":"9-4=5","answer_value":50,"answer_index":2}',
            '{"answer_index":2}',
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"message": {"content": next(responses)}},
        )

    client = OllamaClient(OllamaConfig(warmup_on_start=False))
    client._client.close()
    client._client = httpx.Client(  # noqa: SLF001
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    try:
        decision = client.solve(
            question(
                "在计数器上,用5颗珠子可以表示的最大两位数是().",
                ("32", "41", "50", "23"),
            )
        )
    finally:
        client.close()

    assert decision.answer_index == 2
    assert decision.source == "ollama"
    assert decision.reason.startswith("text retry after numeric failure:")
    assert len(requests) == 2
    assert requests[0]["format"] != requests[1]["format"]


def test_numeric_failure_does_not_retry_when_disabled() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": (
                        '{"calculation":"9-4=5","answer_value":50,"answer_index":2}'
                    )
                }
            },
        )

    client = OllamaClient(
        OllamaConfig(
            warmup_on_start=False,
            retry_numeric_as_text=False,
        )
    )
    client._client.close()
    client._client = httpx.Client(  # noqa: SLF001
        base_url="http://test",
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(SolverError):
            client.solve(question("未知题目", ("32", "41", "50", "23")))
    finally:
        client.close()
    assert calls == 1
