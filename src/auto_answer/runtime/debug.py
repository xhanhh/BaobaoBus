"""Failure artifact recorder."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.config import DebugConfig
from ..core.models import CroppedFrame, OCRBundle, Question


class DebugRecorder:
    def __init__(self, config: DebugConfig) -> None:
        self._config = config

    def save(
        self,
        label: str,
        *,
        crops: CroppedFrame | None = None,
        ocr: OCRBundle | None = None,
        question: Question | None = None,
        error: BaseException | None = None,
    ) -> Path | None:
        if not self._config.enabled:
            return None
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        directory = self._config.output_dir / f"{stamp}-{self._safe_label(label)}"
        directory.mkdir(parents=True, exist_ok=False)
        if crops is not None:
            crops.full_frame.save(directory / "full-frame.png")
            crops.question.save(directory / "question.png")
            for index, option in enumerate(crops.options):
                option.save(directory / f"option-{index}.png")
        if ocr is not None:
            (directory / "ocr.json").write_text(
                json.dumps(self._ocr_data(ocr), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if question is not None:
            (directory / "question.txt").write_text(
                question.as_prompt() + "\n",
                encoding="utf-8",
            )
        if error is not None:
            (directory / "error.txt").write_text(
                f"{type(error).__name__}: {error}\n",
                encoding="utf-8",
            )
        return directory

    @staticmethod
    def _ocr_data(bundle: OCRBundle) -> dict[str, Any]:
        def item(result: Any) -> dict[str, Any]:
            return {
                "text": result.text,
                "confidence": result.confidence,
                "low_confidence": result.low_confidence,
                "lines": list(result.lines),
                "line_confidences": list(result.line_confidences),
                "boxes": list(result.boxes),
            }

        return {
            "question": item(bundle.question),
            "options": [item(value) for value in bundle.options],
        }

    @staticmethod
    def _safe_label(value: str) -> str:
        result = "".join(char if char.isalnum() or char in "-_" else "-" for char in value)
        return result[:48] or "capture"
