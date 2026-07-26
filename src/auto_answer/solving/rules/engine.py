"""Ordered conservative rule engine."""

from __future__ import annotations

from ...core.models import Question, SolveDecision
from .arithmetic import (
    solve_arithmetic_expression,
    solve_arithmetic_sequence,
    solve_between,
    solve_comparison_symbol,
    solve_extreme,
    solve_inequality_blank,
    solve_operation_change,
    solve_option_expression,
    solve_threshold,
)
from .common import match_unique, parse_number
from .counting import solve_counting_problem
from .money import solve_money
from .number_concepts import (
    solve_counter_two_digit_extreme,
    solve_number_neighbor,
    solve_place_value_number,
)
from .time_concepts import solve_time
from .word_problems import solve_counting_choice, solve_word_problem


class RuleEngine:
    """Return None whenever a unique, explainable answer cannot be proven."""

    def solve(self, question: Question) -> SolveDecision | None:
        option_values = tuple(parse_number(value) for value in question.options)

        decision = solve_number_neighbor(question.text, option_values)
        if decision is not None:
            return decision

        decision = solve_counter_two_digit_extreme(question.text, option_values)
        if decision is not None:
            return decision

        decision = solve_place_value_number(question.text, option_values)
        if decision is not None:
            return decision

        decision = solve_money(question)
        if decision is not None:
            return decision

        decision = solve_time(question)
        if decision is not None:
            return decision

        decision = solve_comparison_symbol(question)
        if decision is not None:
            return decision

        decision = solve_arithmetic_sequence(question, option_values)
        if decision is not None:
            return decision

        decision = solve_operation_change(question)
        if decision is not None:
            return decision

        decision = solve_option_expression(question)
        if decision is not None:
            return decision

        decision = solve_counting_choice(question)
        if decision is not None:
            return decision

        target = solve_word_problem(question.text)
        if target is not None:
            return match_unique(target, option_values, "word arithmetic")

        target = solve_counting_problem(question.text)
        if target is not None:
            return match_unique(target, option_values, "counting")

        target = solve_arithmetic_expression(question.text)
        if target is not None:
            return match_unique(target, option_values, "arithmetic expression")

        remaining_rules = (
            solve_inequality_blank,
            solve_extreme,
            solve_threshold,
            solve_between,
        )
        for solve_rule in remaining_rules:
            decision = solve_rule(question.text, option_values)
            if decision is not None:
                return decision
        return None
