"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

from .core.config import AppConfig, load_config
from .core.errors import AutoAnswerError, ConfigurationError
from .device.adb import ADBController
from .runtime.debug import DebugRecorder
from .runtime.post_challenge import PostChallengeController
from .runtime.scheduler import AnswerScheduler
from .solving.llm import build_llm_client
from .solving.rules import RuleEngine
from .vision.capture import ImageFrameSource, WindowsScreenFrameSource
from .vision.ocr import PaddleOCRReader
from .vision.state import PageStateDetector

_ALIYUN_WORKSPACE_HOST = re.compile(
    r"https://[^./\s]+(?P<suffix>\.[a-z0-9-]+\.maas\.aliyuncs\.com)",
    re.IGNORECASE,
)


class _AliyunWorkspaceRedactionFilter(logging.Filter):
    """Hide the workspace identifier while retaining useful endpoint details."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = _ALIYUN_WORKSPACE_HOST.sub(
            r"https://workspace-id\g<suffix>",
            message,
        )
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


class _ConsoleFormatter(logging.Formatter):
    """Apply optional ANSI color only to console output, never the log file."""

    _GREEN = "\033[32m"
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return (
            f"{self._GREEN}{message}{self._RESET}"
            if getattr(record, "console_green", False)
            else message
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="scrcpy + PaddleOCR + routed LLM auto answer"
    )
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument(
        "--image",
        type=Path,
        help="read a static image instead of the Windows screen (requires --dry-run)",
    )
    parser.add_argument("--dry-run", action="store_true", help="solve one question without ADB tap")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate TOML and exit without loading OCR",
    )
    return parser.parse_args()


def _configure_logging(config: AppConfig) -> None:
    config.log_file.parent.mkdir(parents=True, exist_ok=True)
    format_string = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(
        format_string,
        datefmt=date_format,
    )
    # StreamHandler defaults to stderr. IDEA renders stderr in red regardless of
    # the record's logging level, so normal INFO output looked like an error.
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(_ConsoleFormatter(format_string, datefmt=date_format))
    console.addFilter(_AliyunWorkspaceRedactionFilter())
    file_handler = logging.FileHandler(config.log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_AliyunWorkspaceRedactionFilter())
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        handlers=[console, file_handler],
        force=True,
    )


def main() -> int:
    args = _arguments()
    try:
        if args.image is not None and not args.dry_run:
            raise ConfigurationError("--image is only allowed with --dry-run")
        if args.max_questions is not None and args.max_questions <= 0:
            raise ConfigurationError("--max-questions must be positive")

        config = load_config(args.config)
        config.validate(require_live_coordinates=not args.dry_run and not args.check_config)
        _configure_logging(config)
        if (
            config.fallback.random_on_ocr_failure
            or config.fallback.random_on_llm_failure
        ):
            logging.getLogger(__name__).warning(
                "UNSAFE random fallback enabled: ocr_failure=%s llm_failure=%s",
                config.fallback.random_on_ocr_failure,
                config.fallback.random_on_llm_failure,
            )
        if args.check_config:
            logging.getLogger(__name__).info("configuration is valid: %s", args.config.resolve())
            return 0

        source = (
            ImageFrameSource(args.image)
            if args.image is not None
            else WindowsScreenFrameSource(config.capture.screen_rect)
        )
        adb = None
        if not args.dry_run:
            adb = ADBController(config.adb)
            device = adb.check_device()
            logging.getLogger(__name__).info("ADB device ready: %s", device)

        llm_client = build_llm_client(config)
        logging.getLogger(__name__).info(
            "LLM provider order: %s",
            " -> ".join(llm_client.provider_names),
        )
        llm_client.start_warmup()
        ocr_reader = PaddleOCRReader(config.ocr)
        state_detector = PageStateDetector(
            config.state,
            config.regions,
            ocr_reader.recognize_single,
        )
        post_challenge = (
            PostChallengeController(
                config.post_challenge,
                source,
                adb,
                state_detector.ready_page_visible,
            )
            if config.post_challenge.enabled and adb is not None
            else None
        )
        scheduler = AnswerScheduler(
            config=config,
            source=source,
            ocr=ocr_reader,
            rules=RuleEngine(),
            llm=llm_client,
            state_detector=state_detector,
            recorder=DebugRecorder(config.debug),
            adb=adb,
            post_challenge=post_challenge,
        )
        decision = scheduler.run(
            dry_run=args.dry_run,
            max_questions=1 if args.dry_run else args.max_questions,
        )
        if decision is not None:
            print(decision.answer_index)
        return 0
    except (AutoAnswerError, OSError, ValueError) as exc:
        logging.getLogger(__name__).critical("%s", exc)
        return 1
    except Exception:
        logging.getLogger(__name__).exception("unexpected fatal error")
        return 2
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("stopped by Ctrl+C")
        return 130


if __name__ == "__main__":
    sys.exit(main())
