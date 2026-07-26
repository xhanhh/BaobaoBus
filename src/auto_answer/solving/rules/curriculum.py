"""Strict templates for recurring elementary-school question wording."""

from __future__ import annotations

import re
from decimal import Decimal

from ...core.models import Question, SolveDecision
from .common import NUMBER, match_unique, parse_number

_BLANK = r"(?:\(\)|多少|几)"
_NESTED_CAPACITY = re.compile(
    rf"一(?P<outer>壶|桶|瓶)[^,，。]*?装满(?P<middle_count>{NUMBER})个"
    rf"(?P<middle>水瓶|瓶子|杯子)[,，]"
    rf"一个(?P=middle)[^,，。]*?装满(?P<inner_count>{NUMBER})(?:杯|碗)"
    rf"[^,，。]*[,，]一(?P=outer)[^。]*?装满{_BLANK}(?:杯|碗)"
)
_UNKNOWN_ORIGINAL_AFTER_TRANSFER = re.compile(
    rf"(?P<giver>[\u4e00-\u9fff]{{1,4}})有(?P<initial>{NUMBER})"
    rf"[^,，。]*[,，](?:(?P=giver))?送给"
    rf"(?P<receiver>[\u4e00-\u9fff]{{1,4}})(?P<transfer>{NUMBER})"
    rf"[^,，。]*[,，]两人就一样多[,，](?P=receiver)原来有{_BLANK}"
)
_COMPARATIVE_TARGET = re.compile(
    rf"(?P<base>{NUMBER})(?P<unit>支|道|人|只|本|朵|个)"
    rf"[^,，。]*[,，][^,，。]*?比[^,，。]*?"
    rf"(?P<direction>少|多)(?:做了?)?(?P<delta>{NUMBER})(?P=unit)"
    rf"[^,，。]*[,，][^,，。]*?(?:做了|(?<!共)有){_BLANK}(?P=unit)"
)
_COMPARATIVE_TOTAL = re.compile(
    rf"(?:有|养了)(?P<base>{NUMBER})(?P<unit>只|人|支|本|个)"
    rf"(?P<first_kind>[\u4e00-\u9fff]{{1,6}})[,，]"
    rf"(?P<second_kind>[\u4e00-\u9fff]{{1,6}})比(?P=first_kind)"
    rf"(?P<direction>少|多)(?P<delta>{NUMBER})(?P=unit)"
    rf"[^。]*?一共[^。]*?{_BLANK}(?P=unit)"
)
_FINISH_ON_LAST_PERIOD = re.compile(
    rf"有(?P<total>{NUMBER})[^,，。]*[,，]"
    rf"第一天(?:吃了|用了)(?P<first>{NUMBER})[^,，。]*[,，]"
    rf"第二天(?:吃了|用了)(?P<second>{NUMBER})[^,，。]*[,，]"
    rf"第三天(?:吃了|用了){_BLANK}[^。]*?正好(?:吃|用)完"
)
_THREE_CATEGORY_REMAINDER = re.compile(
    rf"(?P<kinds>[^,，。]+)共(?P<total>{NUMBER})个[,，]"
    rf"其中[^,，。]*?有(?P<first>{NUMBER})个[,，]"
    rf"[^,，。]*?有(?P<second>{NUMBER})个[,，]"
    rf"[^,，。]*?有{_BLANK}个"
)
_HALF_REMAINING = re.compile(
    rf"(?:有|一共有)(?P<total>{NUMBER})(?P<unit>盒|个|支|本|块|只)"
    rf"[^,，。]*[,，](?:喝|吃|用|拿)(?:掉|去)?一半后还剩{_BLANK}(?P=unit)"
)
_PAIRWISE_GAMES = re.compile(
    rf"(?P<count>{NUMBER})个[^,，。]*?(?:下棋|比赛)[,，]"
    rf"每两个人都要(?:下|比)一(?:盘|场)[,，]"
    rf"一共要(?:下|比){_BLANK}(?:盘|场)"
)
_PACK_EQUAL_GROUPS = re.compile(
    rf"(?:买了|有)(?P<total>{NUMBER})(?:个|只|支|本|块)"
    rf"[^,，。]*[,，](?P<per>{NUMBER})个装一(?:袋|盒|笼)"
    rf"[^,，。]*[,，](?:可以|能)装{_BLANK}(?:袋|盒|笼)"
)
_CAPACITY_CEILING = re.compile(
    rf"(?:有|一共有)(?P<total>{NUMBER})[^,，。]*?(?:坐船|乘车)"
    rf"[,，]一(?:条船|辆车)最多(?:坐|容纳)(?P<per>{NUMBER})人"
    rf"[,，][^。]*?至少需要{_BLANK}(?:条船|辆车)"
)
_REPEATED_ADDITION = re.compile(
    rf"(?P<value>{NUMBER})连续加(?P<times>{NUMBER})次[,，]和是{_BLANK}"
)
_FUTURE_AGE_DIFFERENCE = re.compile(
    rf"(?P<older>{NUMBER})岁[,，][^,，。]*?(?P<younger>{NUMBER})岁[,，]"
    rf"(?P<years>{NUMBER})年后[^。]*?比[^。]*?大{_BLANK}岁"
)
_TWO_GROUP_DIFFERENCE = re.compile(
    rf"班有(?P<first>{NUMBER})个?学生[,，]"
    rf"[^,，。]*?班有(?P<second>{NUMBER})个?学生[,，]"
    rf"[^,，。]*?班比[^,，。]*?班(?P<direction>多|少){_BLANK}个学生"
)
_NET_CHANGE = re.compile(
    rf"(?:下去|下车)(?P<out>{NUMBER})人[,，]"
    rf"(?:上来|上车)(?P<incoming>{NUMBER})人[^。]*?"
    rf"与原来相比[,，]{_BLANK}了{_BLANK}人"
)
_NET_CHANGE_OPTION = re.compile(rf"(?P<direction>多|少)[,，](?P<amount>{NUMBER})")
_TWO_SQUARES_RECTANGLE = re.compile(
    r"用两个完全一样的正方形可以拼成一个(?:\(\)|什么|哪种图形)"
)
_ZERO_OBJECTS = re.compile(r"一个物体都没有用(?:\(\)|什么|哪个数)表示")


def solve_curriculum_template(question: Question) -> SolveDecision | None:
    """Solve only templates whose arithmetic or fact has one exact interpretation."""

    values = tuple(parse_number(option) for option in question.options)

    nested = _NESTED_CAPACITY.search(question.text)
    if nested:
        target = Decimal(nested.group("middle_count")) * Decimal(
            nested.group("inner_count")
        )
        return match_unique(target, values, "nested capacity")

    original = _UNKNOWN_ORIGINAL_AFTER_TRANSFER.search(question.text)
    if original:
        target = Decimal(original.group("initial")) - 2 * Decimal(
            original.group("transfer")
        )
        return (
            match_unique(target, values, "original amount before equal transfer")
            if target >= 0
            else None
        )

    comparative = _COMPARATIVE_TARGET.search(question.text)
    if comparative:
        base = Decimal(comparative.group("base"))
        delta = Decimal(comparative.group("delta"))
        target = base + delta if comparative.group("direction") == "多" else base - delta
        return (
            match_unique(target, values, "comparative quantity")
            if target >= 0
            else None
        )

    comparative_total = _COMPARATIVE_TOTAL.search(question.text)
    if comparative_total:
        base = Decimal(comparative_total.group("base"))
        delta = Decimal(comparative_total.group("delta"))
        other = (
            base + delta
            if comparative_total.group("direction") == "多"
            else base - delta
        )
        return (
            match_unique(base + other, values, "comparative category total")
            if other >= 0
            else None
        )

    finishing = _FINISH_ON_LAST_PERIOD.search(question.text)
    if finishing:
        target = (
            Decimal(finishing.group("total"))
            - Decimal(finishing.group("first"))
            - Decimal(finishing.group("second"))
        )
        return match_unique(target, values, "last period to finish") if target >= 0 else None

    categories = _THREE_CATEGORY_REMAINDER.search(question.text)
    if categories:
        target = (
            Decimal(categories.group("total"))
            - Decimal(categories.group("first"))
            - Decimal(categories.group("second"))
        )
        return match_unique(target, values, "third category remainder") if target >= 0 else None

    half = _HALF_REMAINING.search(question.text)
    if half:
        total = Decimal(half.group("total"))
        return match_unique(total / 2, values, "half remaining") if total >= 0 else None

    pairwise = _PAIRWISE_GAMES.search(question.text)
    if pairwise:
        count = Decimal(pairwise.group("count"))
        if count != count.to_integral_value() or count < 2:
            return None
        return match_unique(count * (count - 1) / 2, values, "pairwise games")

    packing = _PACK_EQUAL_GROUPS.search(question.text)
    if packing:
        total = Decimal(packing.group("total"))
        per = Decimal(packing.group("per"))
        quotient, remainder = divmod(total, per) if per > 0 else (Decimal(), Decimal(1))
        return (
            match_unique(quotient, values, "equal-size packages")
            if total >= 0 and per > 0 and remainder == 0
            else None
        )

    capacity = _CAPACITY_CEILING.search(question.text)
    if capacity:
        total = Decimal(capacity.group("total"))
        per = Decimal(capacity.group("per"))
        if total < 0 or per <= 0:
            return None
        quotient, remainder = divmod(total, per)
        return match_unique(
            quotient + (1 if remainder else 0),
            values,
            "minimum capacity units",
        )

    repeated = _REPEATED_ADDITION.search(question.text)
    if repeated:
        value = Decimal(repeated.group("value"))
        times = Decimal(repeated.group("times"))
        if times != times.to_integral_value() or times < 0:
            return None
        return match_unique(value * (times + 1), values, "repeated addition")

    future_difference = _FUTURE_AGE_DIFFERENCE.search(question.text)
    if future_difference:
        older = Decimal(future_difference.group("older"))
        younger = Decimal(future_difference.group("younger"))
        target = older - younger
        return match_unique(target, values, "age difference is invariant") if target >= 0 else None

    groups = _TWO_GROUP_DIFFERENCE.search(question.text)
    if groups:
        first = Decimal(groups.group("first"))
        second = Decimal(groups.group("second"))
        target = abs(first - second)
        return match_unique(target, values, "two-group difference")

    net = _NET_CHANGE.search(question.text)
    if net:
        change = Decimal(net.group("incoming")) - Decimal(net.group("out"))
        expected_direction = "多" if change > 0 else "少" if change < 0 else "不变"
        matches = []
        for index, option in enumerate(question.options):
            candidate = _NET_CHANGE_OPTION.fullmatch(option)
            if (
                candidate is not None
                and candidate.group("direction") == expected_direction
                and Decimal(candidate.group("amount")) == abs(change)
            ):
                matches.append(index)
        if len(matches) == 1:
            return SolveDecision(matches[0], "rule", f"net passenger change: {change}")
        return None

    if _TWO_SQUARES_RECTANGLE.search(question.text):
        matches = [
            index for index, option in enumerate(question.options) if option == "长方形"
        ]
        return (
            SolveDecision(matches[0], "rule", "two equal squares form a rectangle")
            if len(matches) == 1
            else None
        )

    if _ZERO_OBJECTS.search(question.text):
        return match_unique(Decimal(), values, "zero objects")
    return None
