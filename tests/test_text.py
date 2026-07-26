import pytest

from auto_answer.core.errors import UnsafeOCRResult
from auto_answer.core.models import OCRBundle, OCRResult
from auto_answer.vision.text import assemble_question, normalize_question_text, normalize_text


def result(text: str, confidence: float = 0.99) -> OCRResult:
    return OCRResult(text, confidence, confidence < 0.6)


def test_normalize_common_math_symbols() -> None:
    assert normalize_text(" 12 × 3 ＝ 36。\n") == "12*3=36."


def test_ambiguous_ten_is_only_repaired_in_expression_context() -> None:
    assert normalize_question_text("与5十2得数相等的算式是()") == (
        "与5+2得数相等的算式是()"
    )
    assert normalize_question_text("8个十再添上多少是100") == "8个十再添上多少是100"


def test_ambiguous_one_is_repaired_as_minus_only_in_comparison_context() -> None:
    assert normalize_question_text("13一7□5,比较大小") == "13-7□5,比较大小"
    assert normalize_question_text("一共有7个") == "一共有7个"


def test_misread_left_parenthesis_is_repaired_only_for_blank_prompt() -> None:
    assert normalize_question_text("括号里的数是1)") == "括号里的数是()"
    assert normalize_question_text("这个数是1)") == "这个数是1)"


def test_expression_context_also_repairs_options() -> None:
    bundle = OCRBundle(
        result("下面算式中得数最大的是"),
        (result("5十2"), result("5-2"), result("3*2"), result("8/2")),
    )
    assert assemble_question(bundle).options == ("5+2", "5-2", "3*2", "8/2")


def test_assemble_preserves_option_mapping() -> None:
    bundle = OCRBundle(
        result("比 9 多 6 的数"),
        (result("16"), result("15"), result("3"), result("10")),
    )
    assembled = assemble_question(bundle)
    assert assembled.text == "比9多6的数"
    assert assembled.options == ("16", "15", "3", "10")


def test_low_confidence_blocks_question() -> None:
    bundle = OCRBundle(
        result("1+1", 0.5),
        (result("1"), result("2"), result("3"), result("4")),
    )
    with pytest.raises(UnsafeOCRResult):
        assemble_question(bundle)
