"""Deterministic arithmetic word-problem rules."""

from __future__ import annotations

import operator
import re
from collections.abc import Callable
from decimal import Decimal, InvalidOperation

from .common import NUMBER

_MAKE_MAXIMUM_GROUPS = re.compile(
    rf"(?:做|制作)(?:一个|1个)?[^,，。]*?(?:要用|需要)"
    rf"(?P<per_group>{NUMBER})[^,，。]*?[,，]"
    rf"(?P<total>{NUMBER})[^,，。]*?最多(?:可以)?(?:做|制作)"
)
_REPEATED_ACTION = re.compile(
    rf"每次(?:拿|取|搬|吃|用)(?:掉|去)?(?P<per_action>{NUMBER})"
    rf"[^,，。]*?[,，][^,，。]*?(?:拿|取|搬|吃|用)(?:了)?"
    rf"(?P<times>{NUMBER})次[^,，。]*?[,，][^。]*?一共"
)
_COMPLETE_GROUPS = re.compile(
    rf"(?:有|共|一共有)(?P<total>{NUMBER})[^,，。]*?[,，]"
    rf"每(?P<per_group>{NUMBER})[^,，。]*?(?:装|放|分|扎)[^,，。]*?"
    rf"[,，](?:可以|能)?(?:装满|装|分成|扎成|扎)"
)
_CAPACITY_REQUIRED = re.compile(
    rf"(?:有|共|一共有)(?P<total>{NUMBER})[^,，。]*?[,，]"
    rf"每(?:条船|辆车|个盒子?|个袋子?|盒|袋)"
    rf"(?:坐|装|放|容纳)(?P<per_group>{NUMBER})"
    rf"[^,，。]*?[,，][^。]*?需要"
)
_INITIAL_INVENTORY = re.compile(
    rf"(?:原来有|原有|有|买来|买了)(?P<initial>{NUMBER})"
    r"(?:个|根|块|颗|本|张|只|米)"
)
_INVENTORY_ACTION = re.compile(
    rf"(?P<verb>吃掉了?|吃了|用去了?|拿走了?|送了|放飞了?|卖了|走了|下了|"
    rf"又买了|买来了?|增加了?|添了|上了)(?P<amount>{NUMBER})"
    r"(?:个|根|块|颗|本|张|只|米)?"
)
_NEGATIVE_ACTION = re.compile(
    rf"(?:吃掉了?|吃了|用去了?|拿走了?|送了|放飞了?|卖了|走了|下了)"
    rf"(?P<amount>{NUMBER})(?:个|根|块|颗|本|张|只|米)?"
)
_REMAINING_QUERY = re.compile(r"还剩|现在(?:还)?(?:有|剩)|目前(?:还)?(?:有|剩)")
_POSITIVE_ACTIONS = ("又买了", "买来", "增加", "添了", "上了")


def solve_word_problem(text: str) -> Decimal | None:
    repeated_action = _REPEATED_ACTION.search(text)
    if repeated_action:
        per_action = Decimal(repeated_action.group("per_action"))
        times = Decimal(repeated_action.group("times"))
        if per_action < 0 or times < 0:
            return None
        return per_action * times

    complete_groups = _COMPLETE_GROUPS.search(text)
    if complete_groups:
        total = Decimal(complete_groups.group("total"))
        per_group = Decimal(complete_groups.group("per_group"))
        if total < 0 or per_group <= 0:
            return None
        return total // per_group

    capacity_required = _CAPACITY_REQUIRED.search(text)
    if capacity_required:
        total = Decimal(capacity_required.group("total"))
        per_group = Decimal(capacity_required.group("per_group"))
        if total < 0 or per_group <= 0:
            return None
        full_groups, remainder = divmod(total, per_group)
        return full_groups + (1 if remainder else 0)

    maximum_groups = _MAKE_MAXIMUM_GROUPS.search(text)
    if maximum_groups:
        per_group = Decimal(maximum_groups.group("per_group"))
        total = Decimal(maximum_groups.group("total"))
        if per_group <= 0 or total < 0:
            return None
        return total // per_group

    inventory = _inventory_total(text)
    if inventory is not None:
        return inventory

    if "短了" in text or "一共用去" in text:
        reductions = [
            Decimal(match.group("amount"))
            for match in _NEGATIVE_ACTION.finditer(text)
        ]
        if reductions:
            return sum(reductions, Decimal())

    currency_exchange = re.search(
        rf"(?:(\d+)张)?({NUMBER})元可以换(?:\(\)|多少|几)?张({NUMBER})元",
        text,
    )
    if currency_exchange:
        note_count = Decimal(currency_exchange.group(1) or "1")
        source_value = Decimal(currency_exchange.group(2))
        target_value = Decimal(currency_exchange.group(3))
        if target_value == 0:
            return None
        return note_count * source_value / target_value

    queue = re.search(
        rf"(?:有|一共有)?({NUMBER})个.*?排队.*?"
        rf"从左数.*?第({NUMBER})个.*?从右数.*?第(?:\(\)|多少|几)",
        text,
    )
    if queue:
        return Decimal(queue.group(1)) - Decimal(queue.group(2)) + 1

    younger = re.search(
        rf"({NUMBER})岁.*?比(?:爸爸|妈妈).*?小({NUMBER})岁.*?"
        rf"(?:爸爸|妈妈).*?\(\)岁",
        text,
    )
    if younger:
        return Decimal(younger.group(1)) + Decimal(younger.group(2))

    patterns: tuple[tuple[str, Callable[[Decimal, Decimal], Decimal]], ...] = (
        (rf"比({NUMBER})多({NUMBER})(?:的数)?", operator.add),
        (rf"比({NUMBER})少({NUMBER})(?:的数)?", operator.sub),
    )
    for pattern, operation in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return operation(Decimal(match.group(1)), Decimal(match.group(2)))
            except (InvalidOperation, ArithmeticError):
                return None
    return None


def _inventory_total(text: str) -> Decimal | None:
    if _REMAINING_QUERY.search(text) is None:
        return None
    initial_match = _INITIAL_INVENTORY.search(text)
    if initial_match is None:
        return None

    total = Decimal(initial_match.group("initial"))
    actions = list(_INVENTORY_ACTION.finditer(text, initial_match.end()))
    if not actions:
        return None
    for action in actions:
        amount = Decimal(action.group("amount"))
        verb = action.group("verb")
        if any(verb.startswith(token) for token in _POSITIVE_ACTIONS):
            total += amount
        else:
            total -= amount
    return total
