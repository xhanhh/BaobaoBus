import logging
import sys
from pathlib import Path
from types import SimpleNamespace

from pytest import MonkeyPatch

from auto_answer.cli import (
    _AliyunWorkspaceRedactionFilter,
    _configure_logging,
    _ConsoleFormatter,
)


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
        "https://workspace-id.cn-beijing.maas.aliyuncs.com/"
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


def test_console_handler_uses_stdout_instead_of_idea_red_stderr(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        logging,
        "basicConfig",
        lambda **kwargs: captured.update(kwargs),
    )

    _configure_logging(
        SimpleNamespace(
            log_file=tmp_path / "auto-answer.log",
            log_level="DEBUG",
        )
    )

    handlers = captured["handlers"]
    assert isinstance(handlers, list)
    assert captured["level"] == logging.DEBUG
    console = handlers[0]
    assert type(console) is logging.StreamHandler
    assert console.stream is sys.stdout
    for handler in handlers:
        handler.close()


def test_stats_color_is_console_only() -> None:
    record = logging.LogRecord(
        "auto_answer.runtime.scheduler",
        logging.INFO,
        "",
        0,
        "SESSION_STATS rounds=1",
        (),
        None,
    )
    record.console_green = True  # type: ignore[attr-defined]
    console_text = _ConsoleFormatter("%(message)s").format(record)
    file_text = logging.Formatter("%(message)s").format(record)

    assert console_text == "\033[32mSESSION_STATS rounds=1\033[0m"
    assert file_text == "SESSION_STATS rounds=1"
