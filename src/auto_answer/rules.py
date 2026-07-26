"""Conservative deterministic solver for elementary arithmetic questions."""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable
from decimal import Decimal, InvalidOperation

from .models import Question, SolveDecision

_NUMBER = r"[-+]?\d+(?:\.\d+)?"
_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Decimal, Decimal], Decimal]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_SYMBOL_OPERATORS: dict[str, Callable[[Decimal, Decimal], Decimal]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}
_BLANK = r"(?:\(\)|__|□)"
_BLANK_LEFT_INEQUALITY = re.compile(
    rf"(?P<blank>{_BLANK})(?P<operator>[+\-*/])(?P<number>{_NUMBER})"
    rf"(?P<comparator>[<>])(?P<bound>{_NUMBER})"
)
_BLANK_RIGHT_INEQUALITY = re.compile(
    rf"(?P<number>{_NUMBER})(?P<operator>[+\-*/])(?P<blank>{_BLANK})"
    rf"(?P<comparator>[<>])(?P<bound>{_NUMBER})"
)
_DIRECT_EXTREME_QUESTION = re.compile(
    r"(?:下面|下列|这些|哪个|哪一个|四个选项).*?"
    r"(?:最大|最小|最多|最少|最高|最低)"
    r"|(?:最大|最小|最高|最低)(?:的)?(?:数|数字)(?:是|为|有)"
)
_MAKE_MAXIMUM_GROUPS = re.compile(
    rf"(?:做|制作)(?:一个|1个)?[^,，。]*?(?:要用|需要)"
    rf"(?P<per_group>{_NUMBER})[^,，。]*?[,，]"
    rf"(?P<total>{_NUMBER})[^,，。]*?最多(?:可以)?(?:做|制作)"
)
_REPEATED_ACTION = re.compile(
    rf"每次(?:拿|取|搬|吃|用)(?:掉|去)?(?P<per_action>{_NUMBER})"
    rf"[^,，。]*?[,，][^,，。]*?(?:拿|取|搬|吃|用)(?:了)?"
    rf"(?P<times>{_NUMBER})次[^,，。]*?[,，][^。]*?一共"
)
_COMPLETE_GROUPS = re.compile(
    rf"(?:有|共|一共有)(?P<total>{_NUMBER})[^,，。]*?[,，]"
    rf"每(?P<per_group>{_NUMBER})[^,，。]*?(?:装|放|分|扎)[^,，。]*?"
    rf"[,，](?:可以|能)?(?:装满|装|分成|扎成|扎)"
)
_CAPACITY_REQUIRED = re.compile(
    rf"(?:有|共|一共有)(?P<total>{_NUMBER})[^,，。]*?[,，]"
    rf"每(?:条船|辆车|个盒子?|个袋子?|盒|袋)"
    rf"(?:坐|装|放|容纳)(?P<per_group>{_NUMBER})"
    rf"[^,，。]*?[,，][^。]*?需要"
)
_INITIAL_INVENTORY = re.compile(
    rf"(?:原来有|原有|有|买来|买了)(?P<initial>{_NUMBER})"
    r"(?:个|根|块|颗|本|张|只|米)"
)
_INVENTORY_ACTION = re.compile(
    rf"(?P<verb>吃掉了?|吃了|用去了?|拿走了?|送了|放飞了?|卖了|走了|下了|"
    rf"又买了|买来了?|增加了?|添了|上了)(?P<amount>{_NUMBER})"
    r"(?:个|根|块|颗|本|张|只|米)?"
)
_NEGATIVE_ACTION = re.compile(
    rf"(?:吃掉了?|吃了|用去了?|拿走了?|送了|放飞了?|卖了|走了|下了)"
    rf"(?P<amount>{_NUMBER})(?:个|根|块|颗|本|张|只|米)?"
)
_REMAINING_QUERY = re.compile(r"还剩|现在(?:还)?(?:有|剩)|目前(?:还)?(?:有|剩)")
_POSITIVE_ACTIONS = ("又买了", "买来", "增加", "添了", "上了")
_EXPRESSION = r"[-+]?\d+(?:\.\d+)?(?:[+\-*/]\d+(?:\.\d+)?)+"
_EQUIVALENT_EXPRESSION = re.compile(
    rf"与(?P<target>{_EXPRESSION})(?:结果|得数)(?:相等|相同)"
)
_EXPRESSION_NEAR_THRESHOLD = re.compile(
    rf"得数比?(?P<threshold>{_NUMBER})(?P<direction>小一些|小一点|大一些|大一点)"
)
_EXPRESSION_THRESHOLD = re.compile(
    rf"得数(?P<direction>小于|大于)(?P<threshold>{_NUMBER})的算式"
)
_COMPARISON_BLANK = re.compile(
    rf"(?P<left>{_EXPRESSION}|{_NUMBER})(?:□|口|\(\)|__)"
    rf"(?P<right>{_EXPRESSION}|{_NUMBER})"
)


class RuleEngine:
    """Return None whenever a unique, explainable answer cannot be proven."""

    def solve(self, question: Question) -> SolveDecision | None:
        option_values = tuple(self._number(value) for value in question.options)

        money = self._money_sum(question)
        if money is not None:
            return money

        comparison_symbol = self._comparison_symbol(question)
        if comparison_symbol is not None:
            return comparison_symbol

        option_expression = self._option_expression(question)
        if option_expression is not None:
            return option_expression

        target = self._word_problem(question.text)
        if target is not None:
            return self._match_unique(target, option_values, "word arithmetic")

        target = self._arithmetic_expression(question.text)
        if target is not None:
            return self._match_unique(target, option_values, "arithmetic expression")

        inequality = self._inequality_blank(question.text, option_values)
        if inequality is not None:
            return inequality

        comparison = self._extreme(question.text, option_values)
        if comparison is not None:
            return comparison

        threshold = self._threshold(question.text, option_values)
        if threshold is not None:
            return threshold

        between = self._between(question.text, option_values)
        if between is not None:
            return between
        return None

    def _comparison_symbol(self, question: Question) -> SolveDecision | None:
        if "比较大小" not in question.text and "比大小" not in question.text:
            return None
        match = _COMPARISON_BLANK.search(question.text)
        if match is None:
            return None
        left = self._evaluate_numeric_form(match.group("left"))
        right = self._evaluate_numeric_form(match.group("right"))
        if left is None or right is None:
            return None
        target = "<" if left < right else ">" if left > right else "="
        matches = [
            index for index, option in enumerate(question.options) if option == target
        ]
        if len(matches) != 1:
            return None
        return SolveDecision(
            matches[0],
            "rule",
            f"comparison expression: {left}{target}{right}",
        )

    def _evaluate_numeric_form(self, text: str) -> Decimal | None:
        if re.fullmatch(_NUMBER, text):
            try:
                return Decimal(text)
            except InvalidOperation:
                return None
        return self._evaluate_complete_expression(text)

    def _option_expression(self, question: Question) -> SolveDecision | None:
        values = tuple(self._evaluate_complete_expression(option) for option in question.options)
        if any(value is None for value in values):
            return None
        numeric_values = tuple(value for value in values if value is not None)

        equivalent = _EQUIVALENT_EXPRESSION.search(question.text)
        if equivalent:
            target = self._evaluate_complete_expression(equivalent.group("target"))
            if target is None:
                return None
            matches = [
                index for index, value in enumerate(numeric_values) if value == target
            ]
            if len(matches) == 1:
                return SolveDecision(
                    matches[0],
                    "rule",
                    f"equivalent option expression: {target}",
                )
            return None

        near = _EXPRESSION_NEAR_THRESHOLD.search(question.text)
        if near:
            threshold = Decimal(near.group("threshold"))
            below = near.group("direction").startswith("小")
            candidates = [
                (index, value)
                for index, value in enumerate(numeric_values)
                if (value < threshold if below else value > threshold)
            ]
            if not candidates:
                return None
            target = (
                max(value for _, value in candidates)
                if below
                else min(value for _, value in candidates)
            )
            matches = [index for index, value in candidates if value == target]
            if len(matches) == 1:
                return SolveDecision(
                    matches[0],
                    "rule",
                    f"nearest option expression: {target}",
                )
            return None

        threshold_match = _EXPRESSION_THRESHOLD.search(question.text)
        if threshold_match:
            threshold = Decimal(threshold_match.group("threshold"))
            predicate = (
                operator.lt
                if threshold_match.group("direction") == "小于"
                else operator.gt
            )
            matches = [
                index
                for index, value in enumerate(numeric_values)
                if predicate(value, threshold)
            ]
            if len(matches) == 1:
                return SolveDecision(
                    matches[0],
                    "rule",
                    "unique option expression threshold",
                )
        return None

    def _evaluate_complete_expression(self, text: str) -> Decimal | None:
        expression = text.strip().rstrip(".。")
        if re.fullmatch(_EXPRESSION, expression) is None:
            return None
        try:
            return self._eval(ast.parse(expression, mode="eval").body)
        except (SyntaxError, ValueError, ArithmeticError, InvalidOperation):
            return None

    def _word_problem(self, text: str) -> Decimal | None:
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

        inventory = self._inventory_total(text)
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
            rf"(?:(\d+)张)?({_NUMBER})元可以换(?:\(\)|多少|几)?张({_NUMBER})元",
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
            rf"(?:有|一共有)?({_NUMBER})个.*?排队.*?"
            rf"从左数.*?第({_NUMBER})个.*?从右数.*?第(?:\(\)|多少|几)",
            text,
        )
        if queue:
            return Decimal(queue.group(1)) - Decimal(queue.group(2)) + 1

        younger = re.search(
            rf"({_NUMBER})岁.*?比(?:爸爸|妈妈).*?小({_NUMBER})岁.*?"
            rf"(?:爸爸|妈妈).*?\(\)岁",
            text,
        )
        if younger:
            return Decimal(younger.group(1)) + Decimal(younger.group(2))

        patterns: tuple[tuple[str, Callable[[Decimal, Decimal], Decimal]], ...] = (
            (rf"比({_NUMBER})多({_NUMBER})(?:的数)?", operator.add),
            (rf"比({_NUMBER})少({_NUMBER})(?:的数)?", operator.sub),
        )
        for pattern, operation in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return operation(Decimal(match.group(1)), Decimal(match.group(2)))
                except (InvalidOperation, ArithmeticError):
                    return None
        return None

    @staticmethod
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

    def _arithmetic_expression(self, text: str) -> Decimal | None:
        normalized = (
            text.replace("加", "+")
            .replace("减", "-")
            .replace("乘以", "*")
            .replace("乘", "*")
            .replace("除以", "/")
            .replace("除", "/")
        )
        expression_pattern = (
            r"(?<![\d.])[-+]?\d+(?:\.\d+)?"
            r"(?:[+\-*/]\d+(?:\.\d+)?)+(?![\d.])"
        )
        matches = re.findall(expression_pattern, normalized)
        if len(matches) != 1:
            return None
        try:
            node = ast.parse(matches[0], mode="eval").body
            return self._eval(node)
        except (SyntaxError, ValueError, ArithmeticError, InvalidOperation):
            return None

    def _eval(self, node: ast.AST) -> Decimal:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = self._eval(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(node.op)](self._eval(node.left), self._eval(node.right))
        raise ValueError("unsupported expression")

    def _extreme(
        self,
        text: str,
        values: tuple[Decimal | None, ...],
    ) -> SolveDecision | None:
        if _DIRECT_EXTREME_QUESTION.search(text) is None:
            return None
        if any(value is None for value in values):
            return None
        numeric = tuple(value for value in values if value is not None)
        choose_max = any(token in text for token in ("最大", "最多", "最高"))
        choose_min = any(token in text for token in ("最小", "最少", "最低"))
        if choose_max == choose_min:
            return None
        target = max(numeric) if choose_max else min(numeric)
        return self._match_unique(target, values, "numeric comparison")

    def _inequality_blank(
        self,
        text: str,
        values: tuple[Decimal | None, ...],
    ) -> SolveDecision | None:
        choose_max = "最大" in text
        choose_min = "最小" in text
        if choose_max == choose_min or any(value is None for value in values):
            return None

        match = _BLANK_LEFT_INEQUALITY.search(text)
        blank_on_left = match is not None
        if match is None:
            match = _BLANK_RIGHT_INEQUALITY.search(text)
        if match is None:
            return None

        operation = _SYMBOL_OPERATORS[match.group("operator")]
        fixed = Decimal(match.group("number"))
        bound = Decimal(match.group("bound"))
        comparator = operator.lt if match.group("comparator") == "<" else operator.gt

        matches: list[tuple[int, Decimal]] = []
        for index, value in enumerate(values):
            assert value is not None
            try:
                result = operation(value, fixed) if blank_on_left else operation(fixed, value)
            except (ArithmeticError, InvalidOperation):
                continue
            if comparator(result, bound):
                matches.append((index, value))
        if not matches:
            return None

        target = (
            max(value for _, value in matches)
            if choose_max
            else min(value for _, value in matches)
        )
        target_indexes = [index for index, value in matches if value == target]
        if len(target_indexes) != 1:
            return None
        return SolveDecision(
            target_indexes[0],
            "rule",
            f"inequality {'maximum' if choose_max else 'minimum'}: {target}",
        )

    def _threshold(
        self,
        text: str,
        values: tuple[Decimal | None, ...],
    ) -> SolveDecision | None:
        match = re.search(rf"(大于|小于)({_NUMBER})(?:的数)?", text)
        if not match or any(value is None for value in values):
            return None
        threshold = Decimal(match.group(2))
        predicate = operator.gt if match.group(1) == "大于" else operator.lt
        matches = [index for index, value in enumerate(values) if predicate(value, threshold)]
        if len(matches) != 1:
            return None
        return SolveDecision(matches[0], "rule", "unique threshold comparison")

    def _between(
        self,
        text: str,
        values: tuple[Decimal | None, ...],
    ) -> SolveDecision | None:
        match = re.search(rf"比({_NUMBER})大.*?比({_NUMBER})小", text)
        if not match or any(value is None for value in values):
            return None
        lower = Decimal(match.group(1))
        upper = Decimal(match.group(2))
        if lower >= upper:
            return None
        matches = [
            index
            for index, value in enumerate(values)
            if value is not None and lower < value < upper
        ]
        if len(matches) != 1:
            return None
        return SolveDecision(matches[0], "rule", "unique value between bounds")

    def _money_sum(self, question: Question) -> SolveDecision | None:
        match = re.search(
            rf"(?:买)?.*?({_NUMBER})角.*?({_NUMBER})角.*?一共",
            question.text,
        )
        if not match:
            return None
        target_jiao = Decimal(match.group(1)) + Decimal(match.group(2))
        option_values = tuple(self._money_to_jiao(value) for value in question.options)
        matches = [
            index for index, value in enumerate(option_values) if value == target_jiao
        ]
        if len(matches) != 1:
            return None
        return SolveDecision(matches[0], "rule", f"money sum: {target_jiao}角")

    @staticmethod
    def _number(value: str) -> Decimal | None:
        cleaned = value.strip().rstrip(".。")
        if not re.fullmatch(_NUMBER, cleaned):
            return None
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None

    @staticmethod
    def _money_to_jiao(value: str) -> Decimal | None:
        match = re.fullmatch(rf"({_NUMBER})(元|角)", value.strip())
        if not match:
            return None
        amount = Decimal(match.group(1))
        return amount * 10 if match.group(2) == "元" else amount

    @staticmethod
    def _match_unique(
        target: Decimal,
        options: tuple[Decimal | None, ...],
        reason: str,
    ) -> SolveDecision | None:
        matches = [index for index, value in enumerate(options) if value == target]
        if len(matches) != 1:
            return None
        return SolveDecision(matches[0], "rule", f"{reason}: {target}")
