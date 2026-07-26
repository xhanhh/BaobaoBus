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


def solve_money(question: Question) -> SolveDecision | None:
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
