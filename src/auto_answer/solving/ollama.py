"""Persistent, strict Ollama chat client."""

from __future__ import annotations

import ast
import json
import logging
import operator
import re
import threading
import time
from collections.abc import Callable
from decimal import Decimal, InvalidOperation
from typing import Any

import httpx

from ..core.config import OllamaConfig
from ..core.errors import SolverError
from ..core.models import Question, SolveDecision

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
_CALCULATION_OPERATORS: dict[
    type[ast.operator],
    Callable[[Decimal, Decimal], Decimal],
] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_NUMERIC_SYSTEM_PROMPT = (
    "你是小学数学选择题答题器。准确识别题目真正询问的量，"
    "用一个最短算式计算，再把最终数值匹配到选项。"
    "calculation必须是以最终数值结尾的短算式，例如8*4=32；"
    "确需两步时可用逗号分隔两个完整等式，例如6+4=10,10+5=15；"
    "answer_value必须填写JSON数值32，不能填写字符串或算式；"
    "answer_index只能填写该数值对应选项前面的0、1、2、3零基序号。"
)
_TEXT_SYSTEM_PROMPT = (
    "你是小学选择题答题器。准确理解题目后选择唯一正确选项。"
    "answer_index只能填写选项前面的0、1、2、3零基序号。"
)
_NUMERIC_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "calculation": {
            "type": "string",
            "description": "One short equation ending with the final numeric value.",
        },
        "answer_value": {
            "type": "number",
            "description": "Final numeric value only, without units or an equation.",
        },
        "answer_index": {
            "type": "integer",
            "enum": [0, 1, 2, 3],
            "description": "Zero-based index of the option equal to answer_value.",
        },
    },
    "required": ["calculation", "answer_value", "answer_index"],
}
_TEXT_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer_index": {
            "type": "integer",
            "enum": [0, 1, 2, 3],
            "description": "Zero-based index of the correct option.",
        }
    },
    "required": ["answer_index"],
}
_NUMERIC_USER_INSTRUCTION = (
    "\n先确定题目问的是哪个量。calculation写最短算式，"
    "最后一个等式的右侧必须是题目所问单位的最终数值，并与answer_value完全相同；"
    "answer_value写不带引号的最终数值，answer_index写它对应的选项序号。"
)
_TEXT_USER_INSTRUCTION = "\n只选择唯一正确选项，并返回它前面的零基序号。"


def numeric_option_values(
    options: tuple[str, str, str, str],
) -> tuple[Decimal, Decimal, Decimal, Decimal] | None:
    values: list[Decimal] = []
    for option in options:
        cleaned = option.strip()
        if _NUMBER.fullmatch(cleaned) is None:
            return None
        try:
            values.append(Decimal(cleaned))
        except InvalidOperation:
            return None
    return tuple(values)  # type: ignore[return-value]


def _structured_object(content: object) -> dict[str, Any]:
    if not isinstance(content, str) or not content.strip():
        raise SolverError("Ollama returned an empty structured answer")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as exc:
        raise SolverError(f"Ollama returned invalid structured JSON: {content!r}") from exc
    if not isinstance(value, dict):
        raise SolverError(f"Ollama structured answer must be an object: {value!r}")
    return value


def _validated_index(value: object) -> int:
    if type(value) is not int or value not in range(4):
        raise SolverError(f"Ollama returned invalid answer_index: {value!r}")
    return value


def parse_structured_index(content: object) -> int:
    body = _structured_object(content)
    if set(body) != {"answer_index"}:
        raise SolverError(f"Ollama text answer has unexpected fields: {sorted(body)!r}")
    return _validated_index(body["answer_index"])


def parse_structured_numeric_answer(
    content: object,
    options: tuple[str, str, str, str],
) -> tuple[int, str]:
    option_values = numeric_option_values(options)
    if option_values is None:
        raise SolverError("numeric answer validation requires four pure numeric options")

    body = _structured_object(content)
    expected_fields = {"calculation", "answer_value", "answer_index"}
    if set(body) != expected_fields:
        raise SolverError(f"Ollama numeric answer has unexpected fields: {sorted(body)!r}")

    calculation = body["calculation"]
    if not isinstance(calculation, str) or not calculation.strip() or len(calculation) > 80:
        raise SolverError(f"Ollama returned invalid short calculation: {calculation!r}")
    answer_value = body["answer_value"]
    if type(answer_value) not in (int, float):
        raise SolverError(f"Ollama returned invalid numeric answer_value: {answer_value!r}")
    try:
        numeric_answer = Decimal(str(answer_value))
    except InvalidOperation as exc:
        raise SolverError(
            f"Ollama returned invalid numeric answer_value: {answer_value!r}"
        ) from exc

    calculation_text = calculation.strip().replace("×", "*").replace("÷", "/")
    equations = re.split(r"\s*[,，;；]\s*", calculation_text)
    if not equations or len(equations) > 4 or any(not equation for equation in equations):
        raise SolverError(f"Ollama returned invalid calculation chain: {calculation!r}")

    final_value: Decimal | None = None
    for equation in equations:
        equation_parts = equation.split("=")
        if len(equation_parts) != 2 or not all(equation_parts):
            raise SolverError(
                "Ollama calculation must contain complete equations: "
                f"{calculation!r}"
            )
        try:
            left_value = _evaluate_calculation_expression(equation_parts[0])
            right_value = _evaluate_calculation_expression(equation_parts[1])
        except (SyntaxError, ValueError, ArithmeticError, InvalidOperation) as exc:
            raise SolverError(
                f"Ollama returned invalid calculation: {calculation!r}"
            ) from exc
        if left_value != right_value:
            raise SolverError(
                "Ollama calculation equation is false: "
                f"{equation!r} ({left_value} != {right_value})"
            )
        final_value = right_value

    if final_value != numeric_answer:
        raise SolverError(
            "Ollama calculation result does not match answer_value: "
            f"calculation={calculation!r}, value={answer_value!r}"
        )

    answer_index = _validated_index(body["answer_index"])
    matching_indexes = [
        index for index, option_value in enumerate(option_values)
        if option_value == numeric_answer
    ]
    if matching_indexes != [answer_index]:
        raise SolverError(
            "Ollama answer_value does not uniquely match answer_index: "
            f"value={answer_value!r}, index={answer_index}, matches={matching_indexes}"
        )
    return answer_index, calculation.strip()


def _evaluate_calculation_expression(expression: str) -> Decimal:
    expression = expression.strip()
    if not expression:
        raise ValueError("empty calculation expression")
    node = ast.parse(expression, mode="eval").body

    def evaluate(current: ast.AST) -> Decimal:
        if isinstance(current, ast.Constant) and type(current.value) in (int, float):
            return Decimal(str(current.value))
        if isinstance(current, ast.UnaryOp) and isinstance(current.op, (ast.UAdd, ast.USub)):
            value = evaluate(current.operand)
            return value if isinstance(current.op, ast.UAdd) else -value
        if isinstance(current, ast.BinOp) and type(current.op) in _CALCULATION_OPERATORS:
            return _CALCULATION_OPERATORS[type(current.op)](
                evaluate(current.left),
                evaluate(current.right),
            )
        raise ValueError("unsupported calculation expression")

    return evaluate(node)


def parse_answer_index(content: object) -> int:
    if not isinstance(content, str) or not content.strip():
        raise SolverError("Ollama returned an empty answer")
    match = re.fullmatch(r"\s*([0-3])\s*", content)
    if match is None:
        raise SolverError(f"Ollama returned an invalid answer index: {content!r}")
    return int(match.group(1))


class OllamaClient:
    def __init__(self, config: OllamaConfig) -> None:
        self._config = config
        self._logger = logging.getLogger(__name__)
        self._warmup_thread: threading.Thread | None = None
        self._client = httpx.Client(
            base_url=config.base_url,
            timeout=httpx.Timeout(config.timeout_seconds),
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=1),
        )

    def start_warmup(self) -> None:
        if not self._config.warmup_on_start or self._warmup_thread is not None:
            return
        self._warmup_thread = threading.Thread(
            target=self._warmup,
            name="ollama-warmup",
            daemon=True,
        )
        self._warmup_thread.start()

    def solve(self, question: Question) -> SolveDecision:
        numeric_mode = numeric_option_values(question.options) is not None
        if not numeric_mode:
            return self._solve_mode(question, numeric_mode=False)

        try:
            return self._solve_mode(question, numeric_mode=True)
        except SolverError as numeric_error:
            if not self._config.retry_numeric_as_text:
                raise
            self._logger.warning(
                "numeric Ollama response failed validation; retrying once in text mode: %s",
                numeric_error,
            )
            try:
                decision = self._solve_mode(question, numeric_mode=False)
            except SolverError as text_error:
                raise SolverError(
                    "numeric Ollama attempt failed and text-mode retry also failed: "
                    f"numeric={numeric_error}; text={text_error}"
                ) from text_error
            return SolveDecision(
                decision.answer_index,
                "ollama",
                f"text retry after numeric failure: {numeric_error}",
            )

    def _solve_mode(
        self,
        question: Question,
        *,
        numeric_mode: bool,
    ) -> SolveDecision:
        payload = {
            "model": self._config.model,
            "stream": False,
            "think": False,
            "keep_alive": self._config.keep_alive,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        _NUMERIC_SYSTEM_PROMPT if numeric_mode else _TEXT_SYSTEM_PROMPT
                    ),
                },
                {
                    "role": "user",
                    "content": question.as_prompt()
                    + (
                        _NUMERIC_USER_INSTRUCTION
                        if numeric_mode
                        else _TEXT_USER_INSTRUCTION
                    ),
                },
            ],
            "format": (
                _NUMERIC_RESPONSE_SCHEMA if numeric_mode else _TEXT_RESPONSE_SCHEMA
            ),
            "options": {
                "temperature": 0,
                "num_predict": self._config.num_predict,
            },
        }
        try:
            response = self._client.post("/api/chat", json=payload)
            response.raise_for_status()
            body: Any = response.json()
        except httpx.TimeoutException as exc:
            raise SolverError(
                f"Ollama timed out after {self._config.timeout_seconds:.1f}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            raise SolverError(
                f"Ollama HTTP {exc.response.status_code}: {detail or exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise SolverError(f"Ollama HTTP request failed: {exc}") from exc
        except ValueError as exc:
            raise SolverError("Ollama returned invalid JSON") from exc

        self._log_response_timing(body, numeric_mode=numeric_mode)
        try:
            content = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise SolverError(f"Ollama response has no message.content: {body!r}") from exc
        if numeric_mode:
            answer_index, calculation = parse_structured_numeric_answer(
                content,
                question.options,
            )
            reason = f"verified numeric response: {calculation}"
        else:
            answer_index = parse_structured_index(content)
            reason = "structured text response"
        return SolveDecision(answer_index, "ollama", reason)

    def _log_response_timing(
        self,
        body: object,
        *,
        numeric_mode: bool,
    ) -> None:
        if not isinstance(body, dict):
            return

        def milliseconds(key: str) -> float:
            value = body.get(key, 0)
            return float(value) / 1_000_000 if isinstance(value, int | float) else 0.0

        total_ms = milliseconds("total_duration")
        if total_ms <= 0:
            return
        self._logger.info(
            "OLLAMA_TIMING mode=%s total_ms=%.0f load_ms=%.0f "
            "prompt_eval_ms=%.0f eval_ms=%.0f prompt_tokens=%s eval_tokens=%s",
            "numeric" if numeric_mode else "text",
            total_ms,
            milliseconds("load_duration"),
            milliseconds("prompt_eval_duration"),
            milliseconds("eval_duration"),
            body.get("prompt_eval_count", "?"),
            body.get("eval_count", "?"),
        )

    def close(self) -> None:
        self._client.close()

    def _warmup(self) -> None:
        started = time.perf_counter()
        payload = {
            "model": self._config.model,
            "prompt": "",
            "stream": False,
            "think": False,
            "keep_alive": self._config.keep_alive,
            "options": {"temperature": 0, "num_predict": 1},
        }
        try:
            with httpx.Client(
                base_url=self._config.base_url,
                timeout=httpx.Timeout(self._config.timeout_seconds),
            ) as client:
                response = client.post("/api/generate", json=payload)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            self._logger.warning("Ollama background warmup failed: %s", exc)
            return
        self._logger.info(
            "Ollama model warmed in %.0fms",
            (time.perf_counter() - started) * 1000,
        )
