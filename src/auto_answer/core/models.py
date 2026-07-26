"""Shared immutable data models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from PIL import Image


@dataclass(frozen=True, slots=True)
class Rect:
    """A left/top/width/height rectangle."""

    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.width / 2, self.top + self.height / 2)


@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class CroppedFrame:
    full_frame: Image.Image
    question: Image.Image
    options: tuple[Image.Image, Image.Image, Image.Image, Image.Image]


@dataclass(frozen=True, slots=True)
class OCRResult:
    text: str
    confidence: float
    low_confidence: bool
    lines: tuple[str, ...] = ()
    line_confidences: tuple[float, ...] = ()
    boxes: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class OCRBundle:
    question: OCRResult
    options: tuple[OCRResult, OCRResult, OCRResult, OCRResult]


@dataclass(frozen=True, slots=True)
class Question:
    text: str
    options: tuple[str, str, str, str]
    ocr: OCRBundle

    def as_prompt(self) -> str:
        choices = "\n".join(f"{index}. {value}" for index, value in enumerate(self.options))
        return f"题目：{self.text}\n{choices}"


@dataclass(frozen=True, slots=True)
class SolveDecision:
    answer_index: int
    source: str
    reason: str


class SchedulerState(Enum):
    WAITING_FOR_QUESTION_PAGE = auto()
    WAITING_FOR_STABLE_FRAME = auto()
    CAPTURING = auto()
    OCR = auto()
    VALIDATING_OCR = auto()
    SOLVING_BY_RULE = auto()
    SOLVING_BY_LLM = auto()
    TAPPING = auto()
    WAITING_FOR_TRANSITION = auto()
    WAITING_FOR_NEXT_QUESTION = auto()
    HANDLING_POST_CHALLENGE = auto()
    WAITING_FOR_READY = auto()
    ERROR = auto()
    STOPPED = auto()
