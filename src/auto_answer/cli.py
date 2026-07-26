"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .adb import ADBController
from .capture import ImageFrameSource, WindowsScreenFrameSource
from .config import AppConfig, load_config
from .debug import DebugRecorder
from .errors import AutoAnswerError, ConfigurationError
from .ocr import PaddleOCRReader
from .ollama import OllamaClient
from .rules import RuleEngine
from .scheduler import AnswerScheduler
from .state import PageStateDetector


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="scrcpy + PaddleOCR + Ollama auto answer")
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
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(config.log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[console, file_handler], force=True)


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

        ollama_client = OllamaClient(config.ollama)
        ollama_client.start_warmup()
        ocr_reader = PaddleOCRReader(config.ocr)
        scheduler = AnswerScheduler(
            config=config,
            source=source,
            ocr=ocr_reader,
            rules=RuleEngine(),
            ollama=ollama_client,
            state_detector=PageStateDetector(
                config.state,
                config.regions,
                ocr_reader.recognize_single,
            ),
            recorder=DebugRecorder(config.debug),
            adb=adb,
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
