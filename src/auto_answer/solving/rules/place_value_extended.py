"""Mechanically verifiable four-digit place-value and reading rules."""

from __future__ import annotations

import re
from decimal import Decimal

from ...core.models import Question, SolveDecision
from .common import NUMBER, match_unique, parse_number

_CLOSEST_NUMBER = re.compile(
    rf"(?:各数|数)中[,，]?(?:哪一个|哪个)?最接近"
    rf"(?P<target>{NUMBER})(?P<unit>万|亿)?"
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
_CHINESE_NUMBER_WRITING = re.compile(
    r"(?P<number>[零一二两三四五六七八九十百千万亿]+)写作"
)
_COMPOSED_PLACE_VALUE = re.compile(
    r"(?:\d+个(?:亿|千万|百万|十万|万|千|百|十|一)[、,，]?){2,}组成的数"
)
_COMPOSED_TERM = re.compile(
    r"(?P<count>\d+)个(?P<unit>亿|千万|百万|十万|万|千|百|十|一)"
)
_PLACE_MULTIPLIERS = {
    "亿": 100_000_000,
    "千万": 10_000_000,
    "百万": 1_000_000,
    "十万": 100_000,
    "万": 10_000,
    "千": 1_000,
    "百": 100,
    "十": 10,
    "一": 1,
}
_CHINESE_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CHINESE_SMALL_UNITS = {"十": 10, "百": 100, "千": 1_000}
_CHINESE_LARGE_UNITS = {"万": 10_000, "亿": 100_000_000}


def solve_extended_place_value(question: Question) -> SolveDecision | None:
    values = tuple(parse_number(option) for option in question.options)

    writing = _CHINESE_NUMBER_WRITING.search(question.text)
    if writing:
        target = _parse_chinese_integer(writing.group("number"))
        if target is not None:
            return match_unique(Decimal(target), values, "Chinese number writing")

    if _COMPOSED_PLACE_VALUE.search(question.text):
        terms = tuple(_COMPOSED_TERM.finditer(question.text))
        if len(terms) >= 2:
            target = sum(
                (
                    Decimal(term.group("count"))
                    * _PLACE_MULTIPLIERS[term.group("unit")]
                    for term in terms
                ),
                Decimal(),
            )
            return match_unique(target, values, "composed place-value number")

    closest = _CLOSEST_NUMBER.search(question.text)
    if closest:
        multiplier = {
            None: Decimal(1),
            "万": Decimal(10_000),
            "亿": Decimal(100_000_000),
        }[closest.group("unit")]
        target = Decimal(closest.group("target")) * multiplier
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


def _parse_chinese_integer(text: str) -> int | None:
    total = 0
    section = 0
    digit = 0
    for character in text:
        if character in _CHINESE_DIGITS:
            digit = _CHINESE_DIGITS[character]
        elif character in _CHINESE_SMALL_UNITS:
            section += (digit or 1) * _CHINESE_SMALL_UNITS[character]
            digit = 0
        elif character in _CHINESE_LARGE_UNITS:
            section += digit
            total += section * _CHINESE_LARGE_UNITS[character]
            section = 0
            digit = 0
        else:
            return None
    return total + section + digit


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
