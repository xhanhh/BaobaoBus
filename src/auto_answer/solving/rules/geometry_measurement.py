"""Conservative elementary geometry and measurement facts."""

from __future__ import annotations

import re
from decimal import Decimal

from ...core.models import Question, SolveDecision
from .common import NUMBER, match_unique, parse_number

_SQUARE_RIGHT_ANGLES = re.compile(
    rf"(?P<count>{NUMBER})个正方形一共有(?:\(\)|多少|几)个直角"
)
_BOTH_SHAPES_RIGHT_ANGLES = re.compile(
    r"长方形和正方形分别有(?:\(\)|多少|几)个直角"
)
_RIGHT_ANGLE_SIZE = re.compile(
    r"(?:三角板|学具)[^。]*?直角.*?(?:长方形|正方形)[^。]*?直角.*?相比"
)
_ANGLE_VERTEX_POSITION = re.compile(r"角的大小和顶点的位置")
_TRIANGLE_ANGLE_MAXIMUM = re.compile(
    r"一个三角形中最多可以有(?:\(\)|多少|几)个锐角[,，]"
    r"(?:\(\)|多少|几)个直角"
)
_RULER_ZERO = re.compile(
    r"用尺子量.*?长度[,，]一般应从(?:\(\)|什么|哪个)刻度开始"
)
_STREET_WIDTH_UNIT = re.compile(r"量一条街道的宽.*?用(?:\(\)|什么|哪个)作单位")
_STUDENT_HEIGHT_144 = re.compile(r"二年级学生的身高大约是144(?:\(\)|多少|几)")
_HOMEWORK_15 = re.compile(r"写一次作业用15(?:\(\)|多少|几)")
_FRONT_VIEW_CIRCLE = re.compile(r"几何体从正面看是圆形")


def solve_geometry_measurement(question: Question) -> SolveDecision | None:
    values = tuple(parse_number(option) for option in question.options)

    squares = _SQUARE_RIGHT_ANGLES.search(question.text)
    if squares:
        count = Decimal(squares.group("count"))
        return (
            match_unique(count * 4, values, "square right angles")
            if count >= 0
            else None
        )

    if _BOTH_SHAPES_RIGHT_ANGLES.search(question.text):
        return match_unique(Decimal(4), values, "rectangle and square right angles")

    if _RIGHT_ANGLE_SIZE.search(question.text):
        return _match_text("一样大", question, "all right angles are equal")

    if _ANGLE_VERTEX_POSITION.search(question.text):
        return _match_text("无关", question, "angle independent of vertex position")

    if _TRIANGLE_ANGLE_MAXIMUM.search(question.text):
        return _match_text("3,1", question, "triangle acute/right-angle maximum")

    if _RULER_ZERO.search(question.text):
        return _match_text("0", question, "ruler starts at zero")

    if _STREET_WIDTH_UNIT.search(question.text):
        return _match_text("米", question, "street width unit")

    if _STUDENT_HEIGHT_144.search(question.text):
        return _match_text("厘米", question, "student height unit")

    if _HOMEWORK_15.search(question.text):
        return _match_text("分钟", question, "homework duration unit")

    if _FRONT_VIEW_CIRCLE.search(question.text):
        return _match_text("球体", question, "sphere front view")
    return None


def _match_text(
    expected: str,
    question: Question,
    reason: str,
) -> SolveDecision | None:
    matches = [
        index for index, option in enumerate(question.options) if option == expected
    ]
    return (
        SolveDecision(matches[0], "rule", f"{reason}: {expected}")
        if len(matches) == 1
        else None
    )
