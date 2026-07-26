"""Rules for number neighbors, place value, and counters."""

from __future__ import annotations

import re
from decimal import Decimal

from ...core.models import Question, SolveDecision
from .common import NUMBER, match_unique

_NUMBER_NEIGHBORS = re.compile(
    rf"邻居(?:分别)?是(?P<first>{NUMBER})(?:和|、|,)(?P<second>{NUMBER})"
)
_COUNTER_TWO_DIGIT_EXTREME = re.compile(
    rf"计数器上.*?用(?P<count>{NUMBER})[颗棵个]珠子.*?"
    r"(?P<direction>最大|最小)(?:的)?两位数"
)
_PLACE_VALUE_COMPARE = re.compile(
    rf"(?:一个两位数|一个数).*?十位上(?:的数字)?是"
    rf"(?P<tens>最大的一位数|{NUMBER}).*?"
    rf"个位上的数字比十位上的数字(?P<direction>少|小|多|大)(?P<delta>{NUMBER}).*?"
    rf"这个数"
)
_PLACE_VALUE_OPERATION = re.compile(
    rf"一个两位数.*?十位上的数字是(?P<tens>最大的一位数|{NUMBER}).*?"
    rf"个位上的数字是十位上的数字(?P<operation>加上|减去)(?P<delta>{NUMBER}).*?"
    rf"这个数"
)
_ONES_TO_TENS_COMPARE = re.compile(
    rf"(?:一个数|一个两位数).*?个位上(?:的数(?:字)?)?是(?P<ones>{NUMBER}).*?"
    rf"十位上(?:的数(?:字)?)?比个位上(?:的数(?:字)?)?"
    rf"(?P<direction>少|小|多|大)(?P<delta>{NUMBER}).*?这个数"
)
_COUNTED_PLACE_UNITS = re.compile(
    rf"(?P<count>{NUMBER})个(?P<place>一|十|百|千)是(?:\(\)|多少|几)"
)
_PLACE_UNITS_MAKE_TOTAL = re.compile(
    rf"(?:\(\)|多少|几)个(?P<place>一|十|百|千)是(?P<total>{NUMBER})"
)
_DIRECT_ONES_TENS = re.compile(
    rf"个位上(?:的数字)?是(?P<ones>{NUMBER}).*?"
    rf"十位上(?:的数字)?是(?P<tens>{NUMBER}).*?这个数"
)
_DIRECT_TENS_ONES = re.compile(
    rf"十位上(?:的数字)?是(?P<tens>{NUMBER}).*?"
    rf"个位上(?:的数字)?是(?P<ones>{NUMBER}).*?这个数"
)
_DIGIT_SUM_EXTREME = re.compile(
    rf"(?:个位和十位|十位和个位)(?:上)?的数字(?:的)?和是(?P<sum>{NUMBER}).*?"
    rf"这个数(?P<direction>最大|最小)"
)
_CONSECUTIVE_NATURAL_SUM = re.compile(
    rf"有(?P<count>{NUMBER})个连续的自然数.*?和是(?P<sum>{NUMBER}).*?"
    rf"其中(?P<direction>最大|最小)的数"
)
_SAME_DIGITS_WITH_LOWER_BOUND = re.compile(
    rf"个位和十位上的数字相同.*?比(?P<lower>{NUMBER})大"
)
_BOUNDED_PARITY = re.compile(
    rf"比(?P<lower>{NUMBER})大.*?比(?P<upper>{NUMBER})小.*?"
    rf"(?:并且|而且|,|，).*?是(?P<parity>偶数|双数|奇数|单数)"
)
_EXTREME_DIGIT_SUM = re.compile(r"最大的一位数和最小的两位数的和")
_TWO_DIGIT_RELATION_CHOICE = re.compile(
    rf"一个两位数.*?十位上的数字比个位上的数字"
    rf"(?P<direction>少|小|多|大)(?P<delta>{NUMBER}).*?这个数可能是"
)
_EXTREME_NUMBER_OFFSET = re.compile(
    rf"比(?P<base>最小的两位数|最大的一位数)"
    rf"(?P<direction>多|少)(?P<delta>{NUMBER})(?:的数)?"
)
_NUMBER_AND_LATER = re.compile(
    rf"个位上(?:的数字)?是(?P<ones>{NUMBER})[,，]"
    rf"十位上(?:的数字)?是(?P<tens>{NUMBER})[,，]"
    rf"这个数是(?:\(\)|多少|几)[,，]"
    rf"它后面第(?P<offset>{NUMBER})个数是(?:\(\)|多少|几)"
)
_NUMBER_PAIR_OPTION = re.compile(
    rf"(?P<first>{NUMBER})(?:和|、|,)(?P<second>{NUMBER})"
)
_DIGIT_OCCURRENCES_IN_RANGE = re.compile(
    rf"从(?P<start>{NUMBER})写到(?P<end>{NUMBER})[,，]"
    rf"一共写了(?:\(\)|多少|几)个数字[“”\"']?"
    rf"(?P<digit>\d)[“”\"']?"
)


def solve_number_neighbor(
    text: str,
    values: tuple[Decimal | None, ...],
) -> SolveDecision | None:
    match = _NUMBER_NEIGHBORS.search(text)
    if match is None:
        return None
    first = Decimal(match.group("first"))
    second = Decimal(match.group("second"))
    if abs(first - second) != 2:
        return None
    target = (first + second) / 2
    return match_unique(target, values, "number between neighbors")


def solve_number_pair(question: Question) -> SolveDecision | None:
    match = _NUMBER_AND_LATER.search(question.text)
    if match is None:
        return None
    tens = Decimal(match.group("tens"))
    ones = Decimal(match.group("ones"))
    offset = Decimal(match.group("offset"))
    if (
        tens != tens.to_integral_value()
        or ones != ones.to_integral_value()
        or offset != offset.to_integral_value()
        or not 1 <= tens <= 9
        or not 0 <= ones <= 9
        or offset < 0
    ):
        return None
    first = tens * 10 + ones
    second = first + offset
    matches = []
    for index, option in enumerate(question.options):
        candidate = _NUMBER_PAIR_OPTION.fullmatch(option)
        if (
            candidate is not None
            and Decimal(candidate.group("first")) == first
            and Decimal(candidate.group("second")) == second
        ):
            matches.append(index)
    if len(matches) != 1:
        return None
    return SolveDecision(
        matches[0],
        "rule",
        f"place value and later number: {first}, {second}",
    )


def solve_counter_two_digit_extreme(
    text: str,
    values: tuple[Decimal | None, ...],
) -> SolveDecision | None:
    match = _COUNTER_TWO_DIGIT_EXTREME.search(text)
    if match is None:
        return None
    bead_count = Decimal(match.group("count"))
    if bead_count != bead_count.to_integral_value() or not 1 <= bead_count <= 18:
        return None

    if match.group("direction") == "最大":
        tens = min(Decimal(9), bead_count)
    else:
        tens = max(Decimal(1), bead_count - 9)
    ones = bead_count - tens
    target = tens * 10 + ones
    return match_unique(
        target,
        values,
        f"counter {match.group('direction')} two-digit number",
    )


def solve_place_value_number(
    text: str,
    values: tuple[Decimal | None, ...],
) -> SolveDecision | None:
    occurrences = _DIGIT_OCCURRENCES_IN_RANGE.search(text)
    if occurrences:
        start = Decimal(occurrences.group("start"))
        end = Decimal(occurrences.group("end"))
        if (
            start != start.to_integral_value()
            or end != end.to_integral_value()
            or start < 0
            or end < start
        ):
            return None
        digit = int(occurrences.group("digit"))
        target = _count_digit_through(int(end), digit) - _count_digit_through(
            int(start) - 1,
            digit,
        )
        return match_unique(
            Decimal(target),
            values,
            "digit occurrences in inclusive range",
        )

    relation = _TWO_DIGIT_RELATION_CHOICE.search(text)
    if relation and all(value is not None for value in values):
        delta = Decimal(relation.group("delta"))
        wants_tens_smaller = relation.group("direction") in {"少", "小"}
        matches = []
        for index, value in enumerate(values):
            assert value is not None
            if value != value.to_integral_value() or not 10 <= value <= 99:
                continue
            tens, ones = divmod(int(value), 10)
            actual = ones - tens if wants_tens_smaller else tens - ones
            if Decimal(actual) == delta:
                matches.append(index)
        if len(matches) == 1:
            return SolveDecision(matches[0], "rule", "two-digit digit relationship")
        return None

    offset = _EXTREME_NUMBER_OFFSET.search(text)
    if offset:
        base = (
            Decimal(10)
            if offset.group("base") == "最小的两位数"
            else Decimal(9)
        )
        delta = Decimal(offset.group("delta"))
        target = base + delta if offset.group("direction") == "多" else base - delta
        return match_unique(target, values, "extreme number offset")

    if _EXTREME_DIGIT_SUM.search(text):
        return match_unique(Decimal(19), values, "extreme digit sum")

    consecutive = _CONSECUTIVE_NATURAL_SUM.search(text)
    if consecutive:
        count = Decimal(consecutive.group("count"))
        total = Decimal(consecutive.group("sum"))
        if (
            count != count.to_integral_value()
            or count <= 0
            or total != total.to_integral_value()
        ):
            return None
        first = total / count - (count - 1) / 2
        if first != first.to_integral_value() or first < 0:
            return None
        first = first.to_integral_value()
        target = (
            first + count - 1
            if consecutive.group("direction") == "最大"
            else first
        )
        return match_unique(target, values, "consecutive natural numbers")

    digit_sum = _DIGIT_SUM_EXTREME.search(text)
    if digit_sum:
        total = Decimal(digit_sum.group("sum"))
        if total != total.to_integral_value() or not 1 <= total <= 18:
            return None
        if digit_sum.group("direction") == "最大":
            tens = min(Decimal(9), total)
        else:
            tens = max(Decimal(1), total - 9)
        ones = total - tens
        return match_unique(
            tens * 10 + ones,
            values,
            f"{digit_sum.group('direction')} two-digit number with digit sum",
        )

    direct = _DIRECT_ONES_TENS.search(text) or _DIRECT_TENS_ONES.search(text)
    if direct:
        tens = Decimal(direct.group("tens"))
        ones = Decimal(direct.group("ones"))
        if (
            tens == tens.to_integral_value()
            and ones == ones.to_integral_value()
            and 1 <= tens <= 9
            and 0 <= ones <= 9
        ):
            return match_unique(
                tens * 10 + ones,
                values,
                "direct two-digit place value",
            )
        return None

    place_units = _PLACE_UNITS_MAKE_TOTAL.search(text)
    if place_units:
        total = Decimal(place_units.group("total"))
        multiplier = {
            "一": Decimal(1),
            "十": Decimal(10),
            "百": Decimal(100),
            "千": Decimal(1000),
        }[place_units.group("place")]
        count, remainder = divmod(total, multiplier)
        if total < 0 or remainder:
            return None
        return match_unique(count, values, "place units making total")

    same_digits = _SAME_DIGITS_WITH_LOWER_BOUND.search(text)
    if same_digits and all(value is not None for value in values):
        lower = Decimal(same_digits.group("lower"))
        matches = [
            index
            for index, value in enumerate(values)
            if value is not None
            and value == value.to_integral_value()
            and 10 <= value <= 99
            and value > lower
            and int(value) // 10 == int(value) % 10
        ]
        if len(matches) == 1:
            return SolveDecision(matches[0], "rule", "same tens and ones digit")
        return None

    bounded_parity = _BOUNDED_PARITY.search(text)
    if bounded_parity and all(value is not None for value in values):
        lower = Decimal(bounded_parity.group("lower"))
        upper = Decimal(bounded_parity.group("upper"))
        wants_even = bounded_parity.group("parity") in {"偶数", "双数"}
        matches = [
            index
            for index, value in enumerate(values)
            if value is not None
            and value == value.to_integral_value()
            and lower < value < upper
            and (int(value) % 2 == 0) == wants_even
        ]
        if len(matches) == 1:
            return SolveDecision(matches[0], "rule", "bounded parity")
        return None

    match = _PLACE_VALUE_COMPARE.search(text)
    if match is not None:
        tens = _tens_digit(match.group("tens"))
        delta = Decimal(match.group("delta"))
        ones = (
            tens - delta
            if match.group("direction") in {"少", "小"}
            else tens + delta
        )
    else:
        match = _PLACE_VALUE_OPERATION.search(text)
        if match is not None:
            tens = _tens_digit(match.group("tens"))
            delta = Decimal(match.group("delta"))
            ones = (
                tens + delta
                if match.group("operation") == "加上"
                else tens - delta
            )
        else:
            reverse = _ONES_TO_TENS_COMPARE.search(text)
            if reverse is None:
                counted_units = _COUNTED_PLACE_UNITS.search(text)
                if counted_units is None:
                    return None
                count = Decimal(counted_units.group("count"))
                multiplier = {
                    "一": Decimal(1),
                    "十": Decimal(10),
                    "百": Decimal(100),
                    "千": Decimal(1000),
                }[counted_units.group("place")]
                return match_unique(
                    count * multiplier,
                    values,
                    "counted place units",
                )
            ones = Decimal(reverse.group("ones"))
            delta = Decimal(reverse.group("delta"))
            tens = (
                ones - delta
                if reverse.group("direction") in {"少", "小"}
                else ones + delta
            )

    if (
        tens != tens.to_integral_value()
        or ones != ones.to_integral_value()
        or not 1 <= tens <= 9
        or not 0 <= ones <= 9
    ):
        return None
    target = tens * 10 + ones
    return match_unique(target, values, "two-digit place value")


def _tens_digit(value: str) -> Decimal:
    return Decimal(9) if value == "最大的一位数" else Decimal(value)


def _count_digit_through(limit: int, digit: int) -> int:
    """Count one decimal digit in the usual representations of 0..limit."""
    if limit < 0:
        return 0
    if limit == 0:
        return 1 if digit == 0 else 0

    count = 1 if digit == 0 else 0  # The representation of the number zero.
    factor = 1
    while factor <= limit:
        lower = limit % factor
        current = (limit // factor) % 10
        higher = limit // (factor * 10)
        if digit == 0:
            if higher == 0:
                break
            count += (higher - 1) * factor
            count += lower + 1 if current == 0 else factor
        else:
            count += higher * factor
            if current > digit:
                count += factor
            elif current == digit:
                count += lower + 1
        factor *= 10
    return count
