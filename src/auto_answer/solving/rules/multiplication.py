"""Deterministic multiplication meanings, mnemonics, and repeated sums."""

from __future__ import annotations

import re
from decimal import Decimal

from ...core.models import Question, SolveDecision
from .common import NUMBER, evaluate_expression, match_unique, parse_number

_PRODUCT_MEANING = re.compile(
    rf"(?P<first>{NUMBER})\*(?P<second>{NUMBER})表示"
)
_REPEATED_SUM_FORMULA = re.compile(
    rf"求(?P<count>{NUMBER})个(?P<value>{NUMBER})相加的和.*?列式"
)
_COMBINED_REPEATED_SUMS = re.compile(
    rf"(?P<first_count>{NUMBER})个(?P<value>{NUMBER})相加的和"
    rf"加上(?P<second_count>{NUMBER})个(?P=value)相加的和"
)
_MNEMONIC_FORMULA = re.compile(
    r"口诀[“\"']?(?P<mnemonic>[一二三四五六七八九]{2})"
    r"[一二三四五六七八九十百]+[”\"']?.*?口算"
)
_SAME_FACTOR_PRODUCT = re.compile(
    rf"两个因数都是(?P<factor>{NUMBER}).*?积是"
)
_TWO_SAME_NUMBERS_MULTIPLIED = re.compile(
    rf"(?P<count>{NUMBER})个(?P<factor>{NUMBER})相乘.*?列式"
)
_CONCRETE_FACTOR_CHANGE = re.compile(
    rf"在(?P<first>{NUMBER})\*(?P<second>{NUMBER})的算式中.*?"
    rf"第一个乘数增加(?P<first_add>{NUMBER}).*?"
    rf"另一个乘数减少(?P<second_sub>{NUMBER}).*?积会"
)
_REPEATED_ADDITION_MEANING = re.compile(
    rf"与(?P<sum>{NUMBER}(?:\+{NUMBER})+).*?意思不相同"
)
_CHOPSTICKS = re.compile(
    rf"每人需要一双筷子[,，](?P<count>{NUMBER})个人需要"
)
_ANIMAL_LEGS = re.compile(
    rf"一只(?P<animal>[\u4e00-\u9fff]{{1,6}})(?P<per>{NUMBER})条腿[,，]"
    rf"(?P<count>{NUMBER})只(?P=animal)(?:\(\)|多少|几)条腿"
)
_NOTEBOOK_TOTAL = re.compile(
    rf"每本[^,，。]*?(?P<price>{NUMBER})元[,，](?P<count>{NUMBER})本"
    rf"[^,，。]*?一共(?:\(\)|多少|几)元"
)
_DAILY_CONSUMPTION = re.compile(
    rf"一只[^,，。]*?每天吃(?P<per>{NUMBER})(?P<unit>根|个|颗)"
    rf"[^,，。]*[,，](?P<days>{NUMBER})天吃(?:\(\)|多少|几)(?P=unit)"
)
_EVEN_DISTRIBUTION_TOTAL = re.compile(
    rf"(?P<count>{NUMBER})个[^,，。]*?正好每人分到(?:了)?"
    rf"(?P<per>{NUMBER})个[^,，。]*[,，].*?一共(?:带了|有)"
)
_BOX_FILL_RESULT = re.compile(
    rf"每个盒子可以装(?P<per>{NUMBER})个[^,，。]*[,，]"
    rf"如果用(?P<boxes>{NUMBER})个这样的盒子来装(?P<items>{NUMBER})个"
)
_EQUAL_PLATE_NON_DIVISOR = re.compile(
    rf"把(?P<total>{NUMBER})个[^,，。]*?摆在盘子里[,，]"
    rf"每盘[^,，。]*?数量相同[,，].*?每盘[^,，。]*?不可能"
)
_MULTIPLICATION_OPERATION_CHOICE = re.compile(
    rf"一(?:块|本|支|个)[^,，。]*?(?P<per>{NUMBER})(?:元|角)[^,，。]*[,，]"
    rf"买(?P<count>{NUMBER})(?:块|本|支|个)[^。]*?用(?:\(\)|什么|哪种)计算"
)
_PRODUCT_COMPARE = re.compile(
    rf"每只[^,，。]*?(?P<per>{NUMBER})个[^,，。]*[,，]"
    rf"(?P<count>{NUMBER})只[^,，。]*?比(?P<target>{NUMBER})个"
)
_ONLY_ONE_MNEMONIC = re.compile(r"只能用来计算一个乘法算式")

_CHINESE_DIGITS = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}


def solve_multiplication_concept(question: Question) -> SolveDecision | None:
    """Solve only multiplication templates with one mechanically provable option."""
    meaning = _PRODUCT_MEANING.search(question.text)
    if meaning:
        first = _integer_text(meaning.group("first"))
        second = _integer_text(meaning.group("second"))
        if first is None or second is None:
            return None
        expected = f"{second}个{first}相加"
        return _match_text(expected, question, "multiplication meaning")

    formula = _REPEATED_SUM_FORMULA.search(question.text)
    if formula:
        count = _integer_text(formula.group("count"))
        value = _integer_text(formula.group("value"))
        if count is None or value is None:
            return None
        expected = {f"{count}*{value}", f"{value}*{count}"}
        matches = [
            index
            for index, option in enumerate(question.options)
            if option in expected
        ]
        return (
            SolveDecision(
                matches[0],
                "rule",
                f"repeated-addition formula: {count}*{value}",
            )
            if len(matches) == 1
            else None
        )

    combined = _COMBINED_REPEATED_SUMS.search(question.text)
    if combined:
        target = (
            Decimal(combined.group("first_count"))
            + Decimal(combined.group("second_count"))
        ) * Decimal(combined.group("value"))
        return _match_expression_value(target, question, "combined repeated sums")

    if "能用乘法算式表示" in question.text:
        matches = [
            index
            for index, option in enumerate(question.options)
            if _is_equal_addend_sum(option)
        ]
        if len(matches) == 1:
            return SolveDecision(matches[0], "rule", "equal-addend sum")
        return None

    mnemonic = _MNEMONIC_FORMULA.search(question.text)
    if mnemonic:
        digits = mnemonic.group("mnemonic")
        target = Decimal(_CHINESE_DIGITS[digits[0]] * _CHINESE_DIGITS[digits[1]])
        return _match_expression_value(target, question, "multiplication mnemonic")

    if _ONLY_ONE_MNEMONIC.search(question.text):
        matches = []
        for index, option in enumerate(question.options):
            digits = [
                character
                for character in option
                if character in _CHINESE_DIGITS
            ]
            if len(digits) >= 2 and digits[0] == digits[1]:
                matches.append(index)
        if len(matches) == 1:
            return SolveDecision(
                matches[0],
                "rule",
                "square multiplication mnemonic",
            )
        return None

    same_factor = _SAME_FACTOR_PRODUCT.search(question.text)
    if same_factor:
        factor = Decimal(same_factor.group("factor"))
        values = tuple(parse_number(option) for option in question.options)
        return match_unique(factor * factor, values, "same-factor product")

    two_same = _TWO_SAME_NUMBERS_MULTIPLIED.search(question.text)
    if two_same and Decimal(two_same.group("count")) == 2:
        factor = Decimal(two_same.group("factor"))
        return _match_expression_value(
            factor * factor,
            question,
            "two equal factors",
        )

    change = _CONCRETE_FACTOR_CHANGE.search(question.text)
    if change:
        old = Decimal(change.group("first")) * Decimal(change.group("second"))
        new = (
            Decimal(change.group("first")) + Decimal(change.group("first_add"))
        ) * (
            Decimal(change.group("second")) - Decimal(change.group("second_sub"))
        )
        expected = "变大" if new > old else "变小" if new < old else "不变"
        return _match_text(expected, question, "concrete product change")

    different_meaning = _REPEATED_ADDITION_MEANING.search(question.text)
    if different_meaning:
        addends = _addition_terms(different_meaning.group("sum"))
        if addends is None or len(set(addends)) != 1:
            return None
        value = addends[0]
        count = len(addends)
        matches = [
            index
            for index, option in enumerate(question.options)
            if not _same_repeated_addition_meaning(option, value, count)
        ]
        if len(matches) == 1:
            return SolveDecision(
                matches[0],
                "rule",
                "different repeated-addition meaning",
            )

    chopsticks = _CHOPSTICKS.search(question.text)
    if chopsticks:
        return _match_numeric(
            Decimal(chopsticks.group("count")) * 2,
            question,
            "pairs of chopsticks",
        )

    legs = _ANIMAL_LEGS.search(question.text)
    if legs:
        return _match_numeric(
            Decimal(legs.group("per")) * Decimal(legs.group("count")),
            question,
            "animal legs",
        )

    notebooks = _NOTEBOOK_TOTAL.search(question.text)
    if notebooks:
        return _match_numeric(
            Decimal(notebooks.group("price"))
            * Decimal(notebooks.group("count")),
            question,
            "notebook total",
        )

    daily = _DAILY_CONSUMPTION.search(question.text)
    if daily:
        return _match_numeric(
            Decimal(daily.group("per")) * Decimal(daily.group("days")),
            question,
            "daily consumption",
        )

    distribution = _EVEN_DISTRIBUTION_TOTAL.search(question.text)
    if distribution:
        return _match_numeric(
            Decimal(distribution.group("count"))
            * Decimal(distribution.group("per")),
            question,
            "even distribution total",
        )

    boxes = _BOX_FILL_RESULT.search(question.text)
    if boxes:
        capacity = Decimal(boxes.group("per")) * Decimal(boxes.group("boxes"))
        items = Decimal(boxes.group("items"))
        expected = (
            "正好装满"
            if capacity == items
            else "没有全部装满"
            if capacity > items
            else "装不下这些乒乓球"
        )
        return _match_text(expected, question, "box capacity result")

    non_divisor = _EQUAL_PLATE_NON_DIVISOR.search(question.text)
    if non_divisor:
        total = Decimal(non_divisor.group("total"))
        values = tuple(parse_number(option) for option in question.options)
        matches = [
            index
            for index, value in enumerate(values)
            if value is not None
            and value > 0
            and total % value != 0
        ]
        if len(matches) == 1:
            return SolveDecision(matches[0], "rule", "non-divisor group size")
        return None

    if _MULTIPLICATION_OPERATION_CHOICE.search(question.text):
        return _match_text("乘法", question, "equal groups use multiplication")

    comparison = _PRODUCT_COMPARE.search(question.text)
    if comparison:
        product = Decimal(comparison.group("per")) * Decimal(
            comparison.group("count")
        )
        target = Decimal(comparison.group("target"))
        expected = "多" if product > target else "少" if product < target else "一样多"
        return _match_text(expected, question, "product comparison")
    return None


def _match_expression_value(
    target: Decimal,
    question: Question,
    reason: str,
) -> SolveDecision | None:
    values = tuple(evaluate_expression(option) for option in question.options)
    return match_unique(target, values, reason)


def _match_numeric(
    target: Decimal,
    question: Question,
    reason: str,
) -> SolveDecision | None:
    values = tuple(parse_number(option) for option in question.options)
    return match_unique(target, values, reason)


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


def _addition_terms(expression: str) -> tuple[int, ...] | None:
    if re.fullmatch(r"\d+(?:\+\d+)+", expression) is None:
        return None
    return tuple(int(part) for part in expression.split("+"))


def _is_equal_addend_sum(option: str) -> bool:
    terms = _addition_terms(option)
    return terms is not None and len(terms) >= 2 and len(set(terms)) == 1


def _same_repeated_addition_meaning(option: str, value: int, count: int) -> bool:
    compact = option.replace(" ", "")
    if compact in {f"{value}*{count}", f"{count}*{value}"}:
        return True
    if compact == f"{count}个{value}相加":
        return True
    terms = _addition_terms(compact)
    return terms == (value,) * count


def _integer_text(value: str) -> int | None:
    number = Decimal(value)
    return int(number) if number == number.to_integral_value() and number >= 0 else None
