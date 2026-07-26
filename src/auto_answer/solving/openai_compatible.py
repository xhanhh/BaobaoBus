"""Strict OpenAI-compatible chat client for Alibaba Cloud Model Studio."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from ..core.config import OpenAICompatibleConfig
from ..core.errors import ConfigurationError, SolverError
from ..core.models import Question, SolveDecision
from .ollama import (
    _NUMERIC_SYSTEM_PROMPT,
    _NUMERIC_USER_INSTRUCTION,
    _TEXT_SYSTEM_PROMPT,
    _TEXT_USER_INSTRUCTION,
    numeric_option_values,
    parse_structured_index,
    parse_structured_numeric_answer,
)


class OpenAICompatibleClient:
    """Call Qwen through `/chat/completions` while preserving local validation."""

    def __init__(self, config: OpenAICompatibleConfig) -> None:
        self._config = config
        self._logger = logging.getLogger(__name__)
        api_key = config.api_key.strip()
        if not api_key and config.api_key_env.strip():
            api_key = os.getenv(config.api_key_env, "").strip()
        if not api_key:
            raise ConfigurationError(
                "openai_compatible.api_key is empty and environment variable "
                f"{config.api_key_env!r} is unavailable"
            )
        self._client = httpx.Client(
            base_url=config.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(config.timeout_seconds),
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=1),
        )

    def start_warmup(self) -> None:
        """Cloud models do not require a billable warmup request."""
        return None

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
                "numeric Aliyun response failed validation; "
                "retrying once in text mode: %s",
                numeric_error,
            )
            try:
                decision = self._solve_mode(question, numeric_mode=False)
            except SolverError as text_error:
                raise SolverError(
                    "numeric Aliyun attempt failed and text-mode retry also failed: "
                    f"numeric={numeric_error}; text={text_error}"
                ) from text_error
            return SolveDecision(
                decision.answer_index,
                "aliyun-openai",
                f"text retry after numeric failure: {numeric_error}",
            )

    def _solve_mode(
        self,
        question: Question,
        *,
        numeric_mode: bool,
    ) -> SolveDecision:
        system_prompt = (
            _NUMERIC_SYSTEM_PROMPT if numeric_mode else _TEXT_SYSTEM_PROMPT
        )
        user_instruction = (
            _NUMERIC_USER_INSTRUCTION if numeric_mode else _TEXT_USER_INSTRUCTION
        )
        payload = {
            "model": self._config.model,
            "stream": False,
            "temperature": 0,
            "enable_thinking": self._config.enable_thinking,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": question.as_prompt() + user_instruction,
                },
            ],
        }
        started = time.perf_counter()
        try:
            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            body: Any = response.json()
        except httpx.TimeoutException as exc:
            raise SolverError(
                f"Aliyun OpenAI API timed out after "
                f"{self._config.timeout_seconds:.1f}s"
            ) from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip()
            raise SolverError(
                f"Aliyun OpenAI API HTTP {exc.response.status_code}: "
                f"{detail or exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise SolverError(f"Aliyun OpenAI API request failed: {exc}") from exc
        except ValueError as exc:
            raise SolverError("Aliyun OpenAI API returned invalid JSON") from exc

        elapsed_ms = (time.perf_counter() - started) * 1000
        try:
            content = body["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError) as exc:
            raise SolverError(
                f"Aliyun OpenAI response has no choices[0].message.content: {body!r}"
            ) from exc
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        self._logger.info(
            "OPENAI_TIMING provider=aliyun model=%s mode=%s total_ms=%.0f "
            "prompt_tokens=%s completion_tokens=%s",
            self._config.model,
            "numeric" if numeric_mode else "text",
            elapsed_ms,
            usage.get("prompt_tokens", "?") if isinstance(usage, dict) else "?",
            usage.get("completion_tokens", "?") if isinstance(usage, dict) else "?",
        )

        try:
            if numeric_mode:
                answer_index, calculation = parse_structured_numeric_answer(
                    content,
                    question.options,
                )
                reason = f"verified numeric response: {calculation}"
            else:
                answer_index = parse_structured_index(content)
                reason = "structured text response"
        except SolverError as exc:
            raise SolverError(str(exc).replace("Ollama", "Aliyun OpenAI")) from exc
        return SolveDecision(answer_index, "aliyun-openai", reason)

    def close(self) -> None:
        self._client.close()
