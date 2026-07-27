"""In-memory counters for the current auto-answer process."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SessionStats:
    rounds: int = 0
    questions: int = 0
    rule_answers: int = 0
    llm_answers: int = 0
    fallback_answers: int = 0
    victories: int = 0
    failures: int = 0

    def record_question(self, source: str) -> None:
        self.questions += 1
        if source == "rule":
            self.rule_answers += 1
        elif source in {"aliyun-openai", "ollama"}:
            self.llm_answers += 1
        else:
            self.fallback_answers += 1

    def record_round(self, *, won: bool) -> None:
        self.rounds += 1
        if won:
            self.victories += 1
        else:
            self.failures += 1

    def as_log_message(self) -> str:
        return (
            f"SESSION_STATS rounds={self.rounds} questions={self.questions} "
            f"rule={self.rule_answers} llm={self.llm_answers} "
            f"fallback={self.fallback_answers} victories={self.victories} "
            f"failures={self.failures}"
        )
