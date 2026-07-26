"""Rules for elementary clock-face and elapsed-hour questions."""

from __future__ import annotations

import re
from decimal import Decimal

from ...core.models import Question, SolveDecision
from .common import NUMBER, match_unique, parse_number

_CLOCK_HOUR = re.compile(rf"(?:钟面上)?是(?P<hour>{NUMBER})时")
_CLOCK_HANDS = re.compile(
    rf"时针指(?:着)?(?P<hour>{NUMBER})[,，]"
    rf"分针指(?:着)?(?P<minute>{NUMBER})"
)
_ELAPSED_HOURS = re.compile(
    rf"(?P<start>{NUMBER})时.*?(?P<end>{NUMBER})时.*?"
    rf"(?:后|经过)(?:\(\)|多少|几)小时"
)
_CLOCK_BETWEEN_HOURS = re.compile(
    rf"时针指在(?P<hour>{NUMBER})和(?P<next_hour>{NUMBER})之间[,，]"
    rf"分针指(?:着)?(?P<minute_pointer>{NUMBER})"
)
_DIGITAL_TIME = re.compile(
    r"(?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
)


def solve_time(question: Question) -> SolveDecision | None:
    between = _CLOCK_BETWEEN_HOURS.search(question.text)
    if between:
        hour = int(Decimal(between.group("hour")))
        next_hour = int(Decimal(between.group("next_hour")))
        minute_pointer = int(Decimal(between.group("minute_pointer")))
        if next_hour % 12 != (hour + 1) % 12 or not 1 <= minute_pointer <= 12:
            return None
        minute = 0 if minute_pointer == 12 else minute_pointer * 5
        matches: list[int] = []
        for index, option in enumerate(question.options):
            digital = _DIGITAL_TIME.fullmatch(option)
            if (
                digital is not None
                and int(digital.group("hour")) == hour
                and int(digital.group("minute")) == minute
                and int(digital.group("second") or "0") == 0
            ):
                matches.append(index)
        if len(matches) == 1:
            return SolveDecision(
                matches[0],
                "rule",
                f"clock hands: {hour:02d}:{minute:02d}",
            )
        return None

    clock = _CLOCK_HOUR.search(question.text)
    if clock:
        target_hour = Decimal(clock.group("hour"))
        matches: list[int] = []
        for index, option in enumerate(question.options):
            hands = _CLOCK_HANDS.fullmatch(option)
            if (
                hands is not None
                and Decimal(hands.group("hour")) == target_hour
                and Decimal(hands.group("minute")) == 12
            ):
                matches.append(index)
        if len(matches) == 1:
            return SolveDecision(
                matches[0],
                "rule",
                f"clock face at {target_hour}:00",
            )
        return None

    elapsed = _ELAPSED_HOURS.search(question.text)
    if elapsed:
        start = Decimal(elapsed.group("start"))
        end = Decimal(elapsed.group("end"))
        if end < start:
            return None
        values = tuple(parse_number(option) for option in question.options)
        return match_unique(end - start, values, "elapsed hours")
    return None
