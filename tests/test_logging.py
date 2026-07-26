import logging

from auto_answer.cli import _AliyunWorkspaceRedactionFilter


def test_aliyun_workspace_is_redacted_from_formatted_httpx_log() -> None:
    record = logging.LogRecord(
        "httpx",
        logging.INFO,
        "",
        0,
        'HTTP Request: POST %s "HTTP/1.1 200 OK"',
        (
            "https://llm-secret-workspace.cn-beijing.maas.aliyuncs.com/"
            "compatible-mode/v1/chat/completions",
        ),
        None,
    )

    assert _AliyunWorkspaceRedactionFilter().filter(record)
    assert "llm-secret-workspace" not in record.getMessage()
    assert (
        "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/"
        "compatible-mode/v1/chat/completions"
        in record.getMessage()
    )


def test_redaction_filter_leaves_local_ollama_url_unchanged() -> None:
    record = logging.LogRecord(
        "httpx",
        logging.INFO,
        "",
        0,
        "HTTP Request: POST %s",
        ("http://localhost:11434/api/chat",),
        None,
    )

    assert _AliyunWorkspaceRedactionFilter().filter(record)
    assert record.getMessage().endswith("http://localhost:11434/api/chat")
