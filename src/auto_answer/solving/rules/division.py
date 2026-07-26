"""Deterministic division, remainder, and capacity rules."""

from __future__ import annotations

import re
from decimal import Decimal

from ...core.models import Question, SolveDecision
from .common import NUMBER, match_unique, parse_number

_DIVISION_TERMS_FORMULA = re.compile(
    rf"除数是(?P<divisor>{NUMBER})[,，]被除数是(?P<dividend>{NUMBER})"
    rf".*?正确的算式"
)
_DIVISION_TERMS_QUOTIENT = re.compile(
    rf"被除数是(?P<dividend>{NUMBER})[,，]除数是(?P<divisor>{NUMBER})"
    rf"[,，]商是"
)
_HOW_MANY_DIVISION_FORMULA = re.compile(
    rf"求(?P<dividend>{NUMBER})里面有几个(?P<divisor>{NUMBER})[,，]列式"
)
_AVERAGE_DIVISION_FORMULA = re.compile(
    rf"把(?P<dividend>{NUMBER})个[^,，。]*?平均(?:放进|分成)"
    rf"(?P<divisor>{NUMBER})(?:个|份)[^。]*?算式"
)
_REMAINDER_DIVISOR_MINIMUM = re.compile(
    rf"=(?:[^。]*?余数(?:是)?|\D*?\.{{2,}})(?P<remainder>{NUMBER})"
    rf".*?除数最小"
)
_POSSIBLE_REMAINDERS = re.compile(
    rf"/(?P<divisor>{NUMBER})=[^。]*?余数可以是"
)
_OPTION_REMAINDER = re.compile(rf"余数是(?P<remainder>{NUMBER})")
_DIVISION_OPTION = re.compile(rf"(?P<dividend>{NUMBER})/(?P<divisor>{NUMBER})")
_OPERATION_ORDER = re.compile(
    rf"(?P<left>{NUMBER})\+(?P<first>{NUMBER})\*(?P<second>{NUMBER})"
    rf".*?应先算"
)
_ADD_TO_DIVISIBLE = re.compile(
    rf"要使(?P<total>{NUMBER})个[^,，。]*?正好平均分给"
    rf"(?P<divisor>{NUMBER})个[^,，。]*?[,，]?至少应该添加"
)
_ONE_CAPACITY = re.compile(
    rf"一(?:条船|辆车|张桌子|个笼子).*?(?:坐|关|容纳)"
    rf"(?P<per>{NUMBER})人[,，](?P<total>{NUMBER})人.*?至少"
)
_TOTAL_CAPACITY = re.compile(
    rf"(?P<total>{NUMBER})(?:个|名)?(?:学生|人|只兔子).*?"
    rf"每(?:条船|辆出租车|张桌子|个笼子).*?"
    rf"(?:最多|只能)?(?:坐|关|容纳)?(?P<per>{NUMBER})(?:人|只)?.*?至少"
)
_TEACHERS_AND_STUDENTS_CAPACITY = re.compile(
    rf"(?P<teachers>{NUMBER})个老师.*?(?P<students>{NUMBER})个同学"
    rf".*?每辆出租车坐(?P<per>{NUMBER})人.*?至少"
)
_PEOPLE_PER_TABLE = re.compile(
    rf"每(?P<per>{NUMBER})人一桌[,，]有(?P<total>{NUMBER})人.*?至少"
)
_MAXIMUM_SEGMENTS = re.compile(
    rf"(?P<total>{NUMBER})米[,，]每(?P<per>{NUMBER})米做"
    rf"[^,，。]*?[,，]?最多"
)
_MAXIMUM_PURCHASE = re.compile(
    rf"每(?:支|本|个)[^,，。]*?(?P<price>{NUMBER})元"
    rf"[^,，。]*[,，][^,，。]*?(?:拿|有)(?P<budget>{NUMBER})元"
    rf"[^。]*?最多能买"
)


def solve_division(question: Question) -> SolveDecision | None:
    values = tuple(parse_number(option) for option in question.options)

    formula = _DIVISION_TERMS_FORMULA.search(question.text)
    if formula:
        return _match_formula(
            f"{formula.group('dividend')}/{formula.group('divisor')}",
            question,
            "division terms formula",
        )

    quotient = _DIVISION_TERMS_QUOTIENT.search(question.text)
    if quotient:
        divisor = Decimal(quotient.group("divisor"))
        dividend = Decimal(quotient.group("dividend"))
        return (
            match_unique(dividend / divisor, values, "division quotient")
            if divisor != 0
            else None
        )

    how_many = _HOW_MANY_DIVISION_FORMULA.search(question.text)
    if how_many:
        return _match_formula(
            f"{how_many.group('dividend')}/{how_many.group('divisor')}",
            question,
            "how-many-groups formula",
        )

    average = _AVERAGE_DIVISION_FORMULA.search(question.text)
    if average:
        return _match_formula(
            f"{average.group('dividend')}/{average.group('divisor')}",
            question,
            "average division formula",
        )

    minimum_divisor = _REMAINDER_DIVISOR_MINIMUM.search(question.text)
    if minimum_divisor:
        return match_unique(
            Decimal(minimum_divisor.group("remainder")) + 1,
            values,
            "minimum divisor from remainder",
        )

    possible = _POSSIBLE_REMAINDERS.search(question.text)
    if possible:
        divisor = int(Decimal(possible.group("divisor")))
        expected = ",".join(str(value) for value in range(1, divisor))
        return _match_formula(expected, question, "possible remainders")

    remainder = _OPTION_REMAINDER.search(question.text)
    if remainder:
        target = int(Decimal(remainder.group("remainder")))
        matches = []
        for index, option in enumerate(question.options):
            candidate = _DIVISION_OPTION.fullmatch(option)
            if candidate is None:
                continue
            divisor = int(Decimal(candidate.group("divisor")))
            dividend = int(Decimal(candidate.group("dividend")))
            if divisor > 0 and dividend % divisor == target:
                matches.append(index)
        return _unique(matches, f"division remainder: {target}")

    order = _OPERATION_ORDER.search(question.text)
    if order:
        return _match_formula(
            f"{order.group('first')}*{order.group('second')}",
            question,
            "multiplication before addition",
        )

    add = _ADD_TO_DIVISIBLE.search(question.text)
    if add:
        total = int(Decimal(add.group("total")))
        divisor = int(Decimal(add.group("divisor")))
        if divisor <= 0:
            return None
        target = (-total) % divisor
        return match_unique(Decimal(target), values, "minimum addition to divide evenly")

    teachers = _TEACHERS_AND_STUDENTS_CAPACITY.search(question.text)
    if teachers:
        total = Decimal(teachers.group("teachers")) + Decimal(
            teachers.group("students")
        )
        return _match_ceiling(total, Decimal(teachers.group("per")), values)

    table = _PEOPLE_PER_TABLE.search(question.text)
    if table:
        return _match_ceiling(
            Decimal(table.group("total")),
            Decimal(table.group("per")),
            values,
        )

    capacity = _ONE_CAPACITY.search(question.text)
    if capacity:
        return _match_ceiling(
            Decimal(capacity.group("total")),
            Decimal(capacity.group("per")),
            values,
        )

    total_capacity = _TOTAL_CAPACITY.search(question.text)
    if total_capacity:
        return _match_ceiling(
            Decimal(total_capacity.group("total")),
            Decimal(total_capacity.group("per")),
            values,
        )

    segments = _MAXIMUM_SEGMENTS.search(question.text)
    if segments:
        return _match_floor(
            Decimal(segments.group("total")),
            Decimal(segments.group("per")),
            values,
            "maximum segments",
        )

    purchase = _MAXIMUM_PURCHASE.search(question.text)
    if purchase:
        return _match_floor(
            Decimal(purchase.group("budget")),
            Decimal(purchase.group("price")),
            values,
            "maximum purchase",
        )
    return None


def _match_ceiling(
    total: Decimal,
    per: Decimal,
    values: tuple[Decimal | None, ...],
) -> SolveDecision | None:
    if total < 0 or per <= 0:
        return None
    quotient, remainder = divmod(total, per)
    return match_unique(
        quotient + (1 if remainder else 0),
        values,
        "minimum capacity",
    )


def _match_floor(
    total: Decimal,
    per: Decimal,
    values: tuple[Decimal | None, ...],
    reason: str,
) -> SolveDecision | None:
    return match_unique(total // per, values, reason) if total >= 0 and per > 0 else None


def _match_formula(
    expected: str,
    question: Question,
    reason: str,
) -> SolveDecision | None:
    matches = [
        index
        for index, option in enumerate(question.options)
        if option.rstrip("=") == expected or option.partition("=")[0] == expected
    ]
    return _unique(matches, f"{reason}: {expected}")


def _unique(matches: list[int], reason: str) -> SolveDecision | None:
    return (
        SolveDecision(matches[0], "rule", reason)
        if len(matches) == 1
        else None
    )
