"""Mechanically verifiable four-digit place-value and reading rules."""

from __future__ import annotations

import re
from decimal import Decimal

from ...core.models import Question, SolveDecision
from .common import NUMBER, match_unique, parse_number

_CLOSEST_NUMBER = re.compile(
    rf"(?:各数|数)中[,，]?(?:哪一个|哪个)?最接近(?P<target>{NUMBER})"
)
_COUNT_BY_HUNDREDS = re.compile(
    rf"(?:从(?P<start>{NUMBER})起[,，])?一百一百地数"
    rf"(?:[,，]数到(?P<current>{NUMBER}))?[,，]下一个数"
)
_COUNT_BY_HUNDREDS_FROM = re.compile(
    rf"从(?P<current>{NUMBER})起[,，]一百一百地数[,，]下一个数"
)
_MAXIMUM_FROM_DIGITS = re.compile(
    r"用(?P<digits>\d(?:[、,，]\d)+)可以组成的最大的"
    r"(?P<count>[四4])位数"
)
_DIGIT_AT_TENS = re.compile(r"[“\"']?(?P<digit>\d)[”\"']?在十位上的数")
_DIGIT_REPRESENTS = re.compile(
    r"(?P<number>\d{3,4})中的(?P<digit>\d)表示"
)
_DIGIT_HUNDREDS_CHOICE = re.compile(
    r"各数中的(?P<digit>\d)表示(?P=digit)个百"
)
_BOUNDED_NUMBER = re.compile(
    rf"比(?P<upper>{NUMBER})小[,，]比(?P<lower>{NUMBER})大"
)
_ZERO_READING_CHOICE = re.compile(
    r"(?:只读一个[“\"']?零[”\"']?|只读一个0|一个[“\"']?零[”\"']?也不读|"
    r"一个零都不读)"
)
_SPECIFIC_ZERO_READING = re.compile(r"(?P<number>\d{4})的两个[“\"']?0[”\"']?")


def solve_extended_place_value(question: Question) -> SolveDecision | None:
    values = tuple(parse_number(option) for option in question.options)

    closest = _CLOSEST_NUMBER.search(question.text)
    if closest:
        target = Decimal(closest.group("target"))
        candidates = [
            (index, value, abs(value - target))
            for index, value in enumerate(values)
            if value is not None
        ]
        if len(candidates) != 4:
            return None
        distance = min(item[2] for item in candidates)
        matches = [item[0] for item in candidates if item[2] == distance]
        return (
            SolveDecision(matches[0], "rule", f"closest number to {target}")
            if len(matches) == 1
            else None
        )

    hundreds = _COUNT_BY_HUNDREDS.search(question.text)
    if hundreds:
        current_text = hundreds.group("current") or hundreds.group("start")
        if current_text is not None:
            return match_unique(
                Decimal(current_text) + 100,
                values,
                "counting by hundreds",
            )

    hundreds_from = _COUNT_BY_HUNDREDS_FROM.search(question.text)
    if hundreds_from:
        return match_unique(
            Decimal(hundreds_from.group("current")) + 100,
            values,
            "counting by hundreds",
        )

    maximum = _MAXIMUM_FROM_DIGITS.search(question.text)
    if maximum:
        digits = re.split(r"[、,，]", maximum.group("digits"))
        if len(digits) != 4:
            return None
        target = Decimal("".join(sorted(digits, reverse=True)))
        return match_unique(target, values, "largest number from digits")

    tens = _DIGIT_AT_TENS.search(question.text)
    if tens:
        digit = int(tens.group("digit"))
        matches = [
            index
            for index, value in enumerate(values)
            if value is not None
            and value == value.to_integral_value()
            and (int(value) // 10) % 10 == digit
        ]
        return _unique_index(matches, f"digit {digit} in tens place")

    represents = _DIGIT_REPRESENTS.search(question.text)
    if represents:
        number = represents.group("number")
        digit = represents.group("digit")
        if number.count(digit) != 1:
            return None
        position = number.index(digit)
        place_names = ("千", "百", "十", "个")[-len(number) :]
        return _match_text(
            f"{digit}个{place_names[position]}",
            question,
            "digit place value",
        )

    hundreds_digit = _DIGIT_HUNDREDS_CHOICE.search(question.text)
    if hundreds_digit:
        digit = int(hundreds_digit.group("digit"))
        matches = [
            index
            for index, value in enumerate(values)
            if value is not None
            and value == value.to_integral_value()
            and (int(value) // 100) % 10 == digit
        ]
        return _unique_index(matches, f"digit {digit} in hundreds place")

    bounded = _BOUNDED_NUMBER.search(question.text)
    if bounded:
        lower = Decimal(bounded.group("lower"))
        upper = Decimal(bounded.group("upper"))
        matches = [
            index
            for index, value in enumerate(values)
            if value is not None and lower < value < upper
        ]
        return _unique_index(matches, f"number between {lower} and {upper}")

    specific_zero = _SPECIFIC_ZERO_READING.search(question.text)
    if specific_zero:
        count = _spoken_zero_count(int(specific_zero.group("number")))
        expected = "都不读" if count == 0 else "只读一个零"
        return _match_text(expected, question, "spoken zero count")

    if _ZERO_READING_CHOICE.search(question.text):
        wants_zero = "只读一个" in question.text
        wants_no_zero = "不读" in question.text or "一个零都不读" in question.text
        matches = []
        for index, value in enumerate(values):
            if value is None or value != value.to_integral_value():
                continue
            number = int(value)
            if not 1000 <= number <= 9999:
                continue
            count = _spoken_zero_count(number)
            if (wants_zero and count == 1) or (wants_no_zero and count == 0):
                matches.append(index)
        return _unique_index(matches, "four-digit zero reading")
    return None


def _spoken_zero_count(number: int) -> int:
    digits = tuple(int(character) for character in f"{number:04d}")
    _thousands, hundreds, tens, ones = digits
    speaks_zero = (
        (hundreds == 0 and (tens != 0 or ones != 0))
        or (tens == 0 and ones != 0)
    )
    return 1 if speaks_zero else 0


def _match_text(
    expected: str,
    question: Question,
    reason: str,
) -> SolveDecision | None:
    matches = [
        index for index, option in enumerate(question.options) if option == expected
    ]
    return _unique_index(matches, f"{reason}: {expected}")


def _unique_index(matches: list[int], reason: str) -> SolveDecision | None:
    return (
        SolveDecision(matches[0], "rule", reason)
        if len(matches) == 1
        else None
    )
