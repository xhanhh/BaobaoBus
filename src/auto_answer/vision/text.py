"""OCR text normalization and question assembly."""

from __future__ import annotations

import re
import unicodedata

from ..core.errors import UnsafeOCRResult
from ..core.models import OCRBundle, Question

_MATH_REPLACEMENTS = str.maketrans(
    {
        "＋": "+",
        "﹢": "+",
        "－": "-",
        "—": "-",
        "–": "-",
        "×": "*",
        "✕": "*",
        "÷": "/",
        "／": "/",
        "＝": "=",
        "？": "?",
        "（": "(",
        "）": ")",
        "，": ",",
        "。": ".",
        "：": ":",
    }
)
_MATH_EXPRESSION_CONTEXT = re.compile(r"算式|等式|得数|计算|比较大小")
_AMBIGUOUS_PLUS = re.compile(r"(?<=\d)十(?=\d)")
_AMBIGUOUS_MINUS = re.compile(r"(?<=\d)一(?=\d)")
_MISREAD_BLANK_PAREN = re.compile(r"(括号里(?:的数)?是)[1lI]\)")


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).translate(_MATH_REPLACEMENTS)
    text = re.sub(r"[\t\r\n ]+", "", text)
    text = re.sub(r"[|丨]", "1", text)
    text = re.sub(r"(?<=\d)[oO](?=\d|$)|(?<![A-Za-z])[oO](?=\d)", "0", text)
    return text.strip()


def normalize_question_text(value: str) -> str:
    """Repair ambiguous OCR operators only when the question clearly discusses expressions."""
    text = normalize_text(value)
    text = _MISREAD_BLANK_PAREN.sub(r"\1()", text)
    if _MATH_EXPRESSION_CONTEXT.search(text):
        return _AMBIGUOUS_MINUS.sub("-", _AMBIGUOUS_PLUS.sub("+", text))
    return text


def assemble_question(bundle: OCRBundle) -> Question:
    question = normalize_question_text(bundle.question.text)
    math_expression_context = bool(_MATH_EXPRESSION_CONTEXT.search(question))
    options = tuple(
        _AMBIGUOUS_MINUS.sub("-", _AMBIGUOUS_PLUS.sub("+", normalize_text(item.text)))
        if math_expression_context
        else normalize_text(item.text)
        for item in bundle.options
    )
    if not question:
        raise UnsafeOCRResult("question OCR is empty")
    if len(options) != 4 or any(not option for option in options):
        raise UnsafeOCRResult(f"expected four non-empty option texts, got {options!r}")
    low = [
        ("question", bundle.question.confidence)
        if bundle.question.low_confidence
        else None,
        *[
            (f"option[{index}]", item.confidence) if item.low_confidence else None
            for index, item in enumerate(bundle.options)
        ],
    ]
    low = [item for item in low if item is not None]
    if low:
        details = ", ".join(f"{label}={score:.3f}" for label, score in low)
        raise UnsafeOCRResult(f"OCR confidence below threshold: {details}")
    return Question(
        text=question,
        options=options,  # type: ignore[arg-type]
        ocr=bundle,
    )
