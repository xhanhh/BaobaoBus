import pytest
from PIL import Image, ImageDraw

from auto_answer.core.config import OCRConfig
from auto_answer.vision.ocr import PaddleOCRReader


def reader() -> PaddleOCRReader:
    instance = object.__new__(PaddleOCRReader)
    instance._config = OCRConfig()  # noqa: SLF001
    return instance


def test_tiny_isolated_fragment_is_filtered_before_confidence_aggregation() -> None:
    result = reader()._build(  # noqa: SLF001
        (
            "一个两位数，十位上的数字是4，个位上的数字是十",
            "位上的数字的一半，这个数是（）。",
            "O",
        ),
        (0.9996, 0.9936, 0.5839),
        (
            [214, 22, 841, 67],
            [214, 78, 841, 125],
            [871, 106, 885, 118],
        ),
    )

    assert result.lines == (
        "一个两位数，十位上的数字是4，个位上的数字是十",
        "位上的数字的一半，这个数是（）。",
    )
    assert result.line_confidences == (0.9996, 0.9936)
    assert result.confidence > 0.99
    assert result.low_confidence is False
    assert len(result.boxes) == 2


def test_confidence_is_character_weighted_instead_of_minimum_line_score() -> None:
    result = reader()._build(  # noqa: SLF001
        ("较长的主要文字内容", "补充"),
        (0.99, 0.50),
        ([0, 0, 180, 30], [0, 40, 40, 70]),
    )

    assert result.confidence == pytest.approx((0.99 * 9 + 0.50 * 2) / 11)
    assert result.confidence > min(result.line_confidences)


def test_normal_size_single_character_is_not_filtered() -> None:
    result = reader()._build(  # noqa: SLF001
        ("主要文字", "4"),
        (0.99, 0.95),
        ([0, 0, 160, 30], [0, 40, 20, 70]),
    )

    assert result.lines == ("主要文字", "4")


def test_empty_ocr_recovers_a_visible_thin_minus_option() -> None:
    image = Image.new("RGB", (444, 85), "white")
    ImageDraw.Draw(image).line((205, 44, 240, 44), fill=(0, 80, 160), width=2)
    empty = reader()._build((), (), ())  # noqa: SLF001

    recovered = reader()._recover_empty_option_symbol(empty, image)  # noqa: SLF001

    assert recovered.text == "-"
    assert recovered.low_confidence is False


def test_empty_ocr_recovers_a_visible_equals_option() -> None:
    image = Image.new("RGB", (444, 85), "white")
    draw = ImageDraw.Draw(image)
    draw.line((205, 38, 240, 38), fill=(0, 80, 160), width=3)
    draw.line((205, 49, 240, 49), fill=(0, 80, 160), width=3)
    empty = reader()._build((), (), ())  # noqa: SLF001

    recovered = reader()._recover_empty_option_symbol(empty, image)  # noqa: SLF001

    assert recovered.text == "="
    assert recovered.low_confidence is False


def test_comparison_symbol_uses_separate_confidence_threshold() -> None:
    low_symbol = reader()._build(  # noqa: SLF001
        (">",),
        (0.537,),
        ([200, 20, 245, 65],),
    )

    recovered = reader()._recover_empty_option_symbol(  # noqa: SLF001
        low_symbol,
        Image.new("RGB", (444, 85), "white"),
    )

    assert recovered.text == ">"
    assert recovered.low_confidence is False


def test_truly_blank_option_is_not_recovered() -> None:
    image = Image.new("RGB", (444, 85), "white")
    empty = reader()._build((), (), ())  # noqa: SLF001

    recovered = reader()._recover_empty_option_symbol(empty, image)  # noqa: SLF001

    assert recovered.text == ""
