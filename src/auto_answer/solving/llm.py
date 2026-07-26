"""LLM client protocol and ordered provider fallback."""

from __future__ import annotations

import logging
from typing import Protocol

from ..core.config import AppConfig
from ..core.errors import ConfigurationError, SolverError
from ..core.models import Question, SolveDecision
from .ollama import OllamaClient
from .openai_compatible import OpenAICompatibleClient


class LLMClient(Protocol):
    def start_warmup(self) -> None: ...

    def solve(self, question: Question) -> SolveDecision: ...

    def close(self) -> None: ...


class RoutedLLMClient:
    def __init__(self, providers: tuple[tuple[str, LLMClient], ...]) -> None:
        if not providers:
            raise ConfigurationError("no enabled LLM provider is available")
        self._providers = providers
        self._logger = logging.getLogger(__name__)

    @property
    def provider_names(self) -> tuple[str, ...]:
        return tuple(name for name, _client in self._providers)

    def start_warmup(self) -> None:
        for _name, client in self._providers:
            client.start_warmup()

    def solve(self, question: Question) -> SolveDecision:
        errors: list[str] = []
        for index, (name, client) in enumerate(self._providers):
            try:
                return client.solve(question)
            except SolverError as exc:
                errors.append(f"{name}={exc}")
                if index + 1 < len(self._providers):
                    self._logger.warning(
                        "LLM provider %s failed; falling back to %s: %s",
                        name,
                        self._providers[index + 1][0],
                        exc,
                    )
        raise SolverError("all LLM providers failed: " + "; ".join(errors))

    def close(self) -> None:
        for name, client in self._providers:
            try:
                client.close()
            except Exception as exc:
                self._logger.warning("failed to close LLM provider %s: %s", name, exc)


def build_llm_client(config: AppConfig) -> RoutedLLMClient:
    providers: list[tuple[str, LLMClient]] = []
    for name in config.llm.provider_order:
        if name == "ollama":
            providers.append((name, OllamaClient(config.ollama)))
        elif name == "aliyun" and config.openai_compatible.enabled:
            providers.append(
                (name, OpenAICompatibleClient(config.openai_compatible))
            )
    return RoutedLLMClient(tuple(providers))
