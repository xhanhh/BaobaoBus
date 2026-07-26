"""Rules and normalization for RMB amounts."""

from __future__ import annotations

import re
from decimal import Decimal

from ...core.models import Question, SolveDecision
from .common import NUMBER

_COUNTED_MONEY = re.compile(
    rf"(?P<count>{NUMBER})张(?P<amount>{NUMBER})(?P<unit>元|角)"
)
_MONEY_TOTAL_QUERY = re.compile(r"一共|共有|共计")
_MONEY_CHANGE = re.compile(
    rf"(?:价格是|一个|一件|一本|一张).*?"
    rf"(?P<price_yuan>{NUMBER})元(?:(?P<price_jiao>{NUMBER})角)?"
    rf"[^。]*?(?:付出|付了|给了).*?(?P<paid_yuan>{NUMBER})元"
    rf"[^。]*?(?:应|应该)?找回"
)
_YUAN_TO_FEN = re.compile(rf"(?P<yuan>{NUMBER})元=(?:\(\)|多少|几)分")
_MIXED_NOTE_EXCHANGE = re.compile(
    rf"(?:一张|1张)?(?P<total>{NUMBER})元可以换"
    rf"(?:\(\)|多少|几)张(?P<first>{NUMBER})元和"
    rf"(?:\(\)|多少|几)张(?P<second>{NUMBER})元"
)
_MIXED_NOTE_OPTION = re.compile(
    rf"(?P<first_count>{NUMBER})张(?P<first>{NUMBER})元和"
    rf"(?P<second_count>{NUMBER})张(?P<second>{NUMBER})元"
)
_MINIMUM_NOTES_FOR_TWO_ITEMS = re.compile(
    rf"[^,，。]*?(?P<first>{NUMBER})元[,，]"
    rf"[^,，。]*?(?P<second>{NUMBER})元[,，]"
    rf"买这两样东西[,，].*?至少要付(?:\(\)|多少|几)张"
    rf"(?P<denomination>{NUMBER})元"
)
_MIXED_UNIT_SUBTRACTION = re.compile(
    rf"(?P<yuan>{NUMBER})元-(?P<jiao>{NUMBER})角="
    rf"(?:\(\)|多少|几)角"
)
_QUANTITY_PRICE_CHANGE = re.compile(
    rf"买了(?P<count>{NUMBER})(?P<unit>斤|个|本|支|盒|袋)[^,，。]*[,，]"
    rf"每(?P=unit)(?P<price>{NUMBER})元[^,，。]*[,，]"
    rf"(?:付了|付出|给了)(?P<paid>{NUMBER})元[^。]*?找回"
)
_AVAILABLE_MONEY_CHANGE = re.compile(
    rf"有(?P<paid>{NUMBER})元钱[^,，。]*[,，]"
    rf"买了(?:一个|一件|一本|一张)(?:[^0-9,，。]*)"
    rf"(?P<price>{NUMBER})元[^。]*?找回"
)
_TWO_COIN_DENOMINATIONS = re.compile(
    rf"有(?P<first_value>{NUMBER})(?P<first_unit>元|角)和"
    rf"(?P<second_value>{NUMBER})(?P<second_unit>元|角)的硬币"
    rf"共(?P<total>{NUMBER})元[,，]"
    rf"其中(?P=first_value)(?P=first_unit)硬币有(?P<first_count>{NUMBER})枚[,，]"
    rf"(?P=second_value)(?P=second_unit)硬币有(?:\(\)|多少|几)枚"
)


def solve_money(question: Question) -> SolveDecision | None:
    coins = _TWO_COIN_DENOMINATIONS.search(question.text)
    if coins:
        first_value = _amount_to_jiao(
            Decimal(coins.group("first_value")),
            coins.group("first_unit"),
        )
        second_value = _amount_to_jiao(
            Decimal(coins.group("second_value")),
            coins.group("second_unit"),
        )
        remaining = (
            Decimal(coins.group("total")) * 10
            - first_value * Decimal(coins.group("first_count"))
        )
        count, remainder = (
            divmod(remaining, second_value)
            if second_value > 0
            else (Decimal(), Decimal(1))
        )
        numeric_options = tuple(
            Decimal(option) if re.fullmatch(NUMBER, option) else None
            for option in question.options
        )
        return (
            _match_numeric_choice(count, numeric_options, "coin denomination count")
            if remaining >= 0 and remainder == 0
            else None
        )

    mixed_subtraction = _MIXED_UNIT_SUBTRACTION.search(question.text)
    if mixed_subtraction:
        target = (
            Decimal(mixed_subtraction.group("yuan")) * 10
            - Decimal(mixed_subtraction.group("jiao"))
        )
        if target < 0:
            return None
        numeric_options = tuple(
            Decimal(option) if re.fullmatch(NUMBER, option) else None
            for option in question.options
        )
        matches = [
            index for index, value in enumerate(numeric_options) if value == target
        ]
        if len(matches) == 1:
            return SolveDecision(
                matches[0],
                "rule",
                f"mixed money subtraction: {target}角",
            )
        return None

    quantity_change = _QUANTITY_PRICE_CHANGE.search(question.text)
    if quantity_change:
        cost = Decimal(quantity_change.group("count")) * Decimal(
            quantity_change.group("price")
        )
        paid = Decimal(quantity_change.group("paid"))
        return _match_bare_yuan_change(paid - cost, question)

    available_change = _AVAILABLE_MONEY_CHANGE.search(question.text)
    if available_change:
        paid = Decimal(available_change.group("paid"))
        price = Decimal(available_change.group("price"))
        return _match_bare_yuan_change(paid - price, question)

    minimum_notes = _MINIMUM_NOTES_FOR_TWO_ITEMS.search(question.text)
    if minimum_notes:
        first = Decimal(minimum_notes.group("first"))
        second = Decimal(minimum_notes.group("second"))
        denomination = Decimal(minimum_notes.group("denomination"))
        if first < 0 or second < 0 or denomination <= 0:
            return None
        full_notes, remainder = divmod(first + second, denomination)
        target = full_notes + (1 if remainder else 0)
        numeric_options = tuple(
            Decimal(option) if re.fullmatch(NUMBER, option) else None
            for option in question.options
        )
        matches = [
            index for index, value in enumerate(numeric_options) if value == target
        ]
        if len(matches) == 1:
            return SolveDecision(
                matches[0],
                "rule",
                f"minimum notes required: {target}",
            )
        return None

    mixed_exchange = _MIXED_NOTE_EXCHANGE.search(question.text)
    if mixed_exchange:
        total = Decimal(mixed_exchange.group("total"))
        first = Decimal(mixed_exchange.group("first"))
        second = Decimal(mixed_exchange.group("second"))
        matches: list[int] = []
        for index, option in enumerate(question.options):
            candidate = _MIXED_NOTE_OPTION.fullmatch(option)
            if candidate is None:
                continue
            first_count = Decimal(candidate.group("first_count"))
            second_count = Decimal(candidate.group("second_count"))
            if (
                Decimal(candidate.group("first")) == first
                and Decimal(candidate.group("second")) == second
                and first_count >= 0
                and second_count >= 0
                and first_count == first_count.to_integral_value()
                and second_count == second_count.to_integral_value()
                and first_count * first + second_count * second == total
            ):
                matches.append(index)
        if len(matches) == 1:
            return SolveDecision(
                matches[0],
                "rule",
                f"mixed note exchange: {total}元",
            )
        return None

    change = _MONEY_CHANGE.search(question.text)
    if change:
        paid_jiao = Decimal(change.group("paid_yuan")) * 10
        price_jiao = Decimal(change.group("price_yuan")) * 10 + Decimal(
            change.group("price_jiao") or "0"
        )
        if price_jiao < 0 or paid_jiao < price_jiao:
            return None
        target_jiao = paid_jiao - price_jiao
        decision = _match_money_option(
            target_jiao,
            question.options,
            "money change",
        )
        if decision is not None:
            return decision
        if re.search(r"找回(?:\(\)|多少|几)元", question.text):
            bare_yuan = tuple(
                Decimal(option) * 10 if re.fullmatch(NUMBER, option) else None
                for option in question.options
            )
            matches = [
                index
                for index, value in enumerate(bare_yuan)
                if value == target_jiao
            ]
            if len(matches) == 1:
                return SolveDecision(
                    matches[0],
                    "rule",
                    f"money change: {target_jiao}角",
                )
        return None

    yuan_to_fen = _YUAN_TO_FEN.search(question.text)
    if yuan_to_fen:
        target = Decimal(yuan_to_fen.group("yuan")) * 100
        numeric_options = tuple(
            Decimal(option) if re.fullmatch(NUMBER, option) else None
            for option in question.options
        )
        matches = [
            index for index, value in enumerate(numeric_options) if value == target
        ]
        if len(matches) == 1:
            return SolveDecision(matches[0], "rule", f"yuan to fen: {target}")
        return None

    counted_terms = list(_COUNTED_MONEY.finditer(question.text))
    if counted_terms and _MONEY_TOTAL_QUERY.search(question.text):
        if any(
            Decimal(term.group("count")) <= 0
            or Decimal(term.group("amount")) < 0
            for term in counted_terms
        ):
            return None
        target_jiao = sum(
            (
                Decimal(term.group("count"))
                * _amount_to_jiao(
                    Decimal(term.group("amount")),
                    term.group("unit"),
                )
                for term in counted_terms
            ),
            Decimal(),
        )
        return _match_money_option(target_jiao, question.options, "counted money sum")

    match = re.search(
        rf"(?:买)?.*?({NUMBER})角.*?({NUMBER})角.*?一共",
        question.text,
    )
    if not match:
        return None
    target_jiao = Decimal(match.group(1)) + Decimal(match.group(2))
    return _match_money_option(target_jiao, question.options, "money sum")


def _match_money_option(
    target_jiao: Decimal,
    options: tuple[str, str, str, str],
    reason: str,
) -> SolveDecision | None:
    option_values = tuple(_money_to_jiao(value) for value in options)
    matches = [
        index for index, value in enumerate(option_values) if value == target_jiao
    ]
    if len(matches) != 1:
        return None
    return SolveDecision(matches[0], "rule", f"{reason}: {target_jiao}角")


def _money_to_jiao(value: str) -> Decimal | None:
    match = re.fullmatch(
        rf"(?:(?P<yuan>{NUMBER})元)?(?:(?P<jiao>{NUMBER})角)?",
        value.strip(),
    )
    if not match:
        return None
    yuan = match.group("yuan")
    jiao = match.group("jiao")
    if yuan is None and jiao is None:
        return None
    return (
        _amount_to_jiao(Decimal(yuan), "元") if yuan is not None else Decimal()
    ) + (Decimal(jiao) if jiao is not None else Decimal())


def _amount_to_jiao(amount: Decimal, unit: str) -> Decimal:
    return amount * 10 if unit == "元" else amount


def _match_bare_yuan_change(
    target_yuan: Decimal,
    question: Question,
) -> SolveDecision | None:
    if target_yuan < 0:
        return None
    values = tuple(
        Decimal(option) if re.fullmatch(NUMBER, option) else None
        for option in question.options
    )
    matches = [index for index, value in enumerate(values) if value == target_yuan]
    if len(matches) != 1:
        return None
    return SolveDecision(matches[0], "rule", f"money change: {target_yuan}元")


def _match_numeric_choice(
    target: Decimal,
    values: tuple[Decimal | None, ...],
    reason: str,
) -> SolveDecision | None:
    matches = [index for index, value in enumerate(values) if value == target]
    if len(matches) != 1:
        return None
    return SolveDecision(matches[0], "rule", f"{reason}: {target}")
