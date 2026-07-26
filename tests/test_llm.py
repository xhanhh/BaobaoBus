import json

import httpx
import pytest

from auto_answer.core.config import OpenAICompatibleConfig
from auto_answer.core.errors import ConfigurationError, SolverError
from auto_answer.core.models import OCRBundle, OCRResult, Question, SolveDecision
from auto_answer.solving.llm import RoutedLLMClient
from auto_answer.solving.openai_compatible import OpenAICompatibleClient


def question(text: str, options: tuple[str, str, str, str]) -> Question:
    empty = OCRResult("", 1.0, False)
    return Question(text, options, OCRBundle(empty, (empty, empty, empty, empty)))


def test_aliyun_openai_numeric_request_and_strict_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-secret")
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"calculation":"8*4=32",'
                                '"answer_value":32,"answer_index":1}'
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 80, "completion_tokens": 18},
            },
        )

    client = OpenAICompatibleClient(
        OpenAICompatibleConfig(
            enabled=True,
            base_url="https://example.invalid/compatible-mode/v1",
            model="qwen3.7-flash",
            retry_numeric_as_text=False,
        )
    )
    client._client.close()  # noqa: SLF001
    client._client = httpx.Client(  # noqa: SLF001
        base_url="https://example.invalid/compatible-mode/v1",
        headers={"Authorization": "Bearer test-secret"},
        transport=httpx.MockTransport(handler),
    )
    try:
        decision = client.solve(
            question("每盒8支,4盒一共多少支?", ("24", "32", "40", "16"))
        )
    finally:
        client.close()

    assert decision.answer_index == 1
    assert decision.source == "aliyun-openai"
    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert requests[0].url.path.endswith("/v1/chat/completions")
    assert payload["model"] == "qwen3.7-flash"
    assert payload["enable_thinking"] is False
    assert payload["temperature"] == 0
    assert payload["response_format"] == {"type": "json_object"}
    assert "max_tokens" not in payload
    assert "JSON" in payload["messages"][0]["content"]


def test_aliyun_provider_requires_key_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    with pytest.raises(ConfigurationError):
        OpenAICompatibleClient(OpenAICompatibleConfig(enabled=True))


def test_aliyun_provider_prefers_key_from_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "environment-secret")
    client = OpenAICompatibleClient(
        OpenAICompatibleConfig(
            enabled=True,
            base_url="https://example.invalid/compatible-mode/v1",
            api_key="config-secret",
        )
    )
    try:
        assert client._client.headers["Authorization"] == "Bearer config-secret"  # noqa: SLF001
    finally:
        client.close()


class _FakeClient:
    def __init__(
        self,
        *,
        decision: SolveDecision | None = None,
        error: str | None = None,
    ) -> None:
        self.decision = decision
        self.error = error
        self.closed = False

    def start_warmup(self) -> None:
        return None

    def solve(self, _question: Question) -> SolveDecision:
        if self.error is not None:
            raise SolverError(self.error)
        assert self.decision is not None
        return self.decision

    def close(self) -> None:
        self.closed = True


def test_llm_router_falls_back_in_configured_order() -> None:
    cloud = _FakeClient(error="cloud unavailable")
    local = _FakeClient(decision=SolveDecision(2, "ollama", "local answer"))
    router = RoutedLLMClient((("aliyun", cloud), ("ollama", local)))
    try:
        decision = router.solve(question("未知题", ("1", "2", "3", "4")))
    finally:
        router.close()
    assert decision.answer_index == 2
    assert decision.source == "ollama"
    assert cloud.closed and local.closed
