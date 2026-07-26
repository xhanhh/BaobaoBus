"""Question-page recognition, visual stability, and transition detection."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps

from ..core.config import RegionConfig, StateConfig
from ..core.errors import CaptureError, OCRError, StateDetectionError
from ..core.models import OCRResult, Rect
from .capture import FrameSource
from .text import normalize_text


@dataclass(frozen=True, slots=True)
class PageObservation:
    detected_question_number: int | None
    effective_question_number: int | None
    title_text: str
    option_boxes_present: bool
    option_white_ratios: tuple[float, float, float, float]

    @property
    def is_question_page(self) -> bool:
        return self.effective_question_number is not None and self.option_boxes_present


@dataclass(frozen=True, slots=True)
class ReadyQuestionPage:
    frame: Image.Image
    question_number: int
    observation: PageObservation


def normalized_image_difference(first: Image.Image, second: Image.Image) -> float:
    """Mean absolute grayscale difference normalized to [0, 1]."""
    size = (160, 90)
    first_array = np.asarray(ImageOps.grayscale(first).resize(size), dtype=np.float32)
    second_array = np.asarray(ImageOps.grayscale(second).resize(size), dtype=np.float32)
    return float(np.mean(np.abs(first_array - second_array)) / 255.0)


def crop_for_state(frame: Image.Image, roi: Rect) -> Image.Image:
    if (
        roi.left < 0
        or roi.top < 0
        or roi.right > frame.width
        or roi.bottom > frame.height
    ):
        raise StateDetectionError(
            f"state ROI {roi} exceeds captured frame {frame.width}x{frame.height}"
        )
    return frame.crop((roi.left, roi.top, roi.right, roi.bottom))


def extract_question_number(result: OCRResult) -> int | None:
    if result.low_confidence or not result.text:
        return None
    match = re.search(r"第(\d+)题", normalize_text(result.text))
    return int(match.group(1)) if match else None


def white_pixel_ratio(image: Image.Image, threshold: int) -> float:
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    white = np.all(pixels >= threshold, axis=2)
    return float(np.mean(white))


def _content_vector(frame: Image.Image, regions: RegionConfig) -> np.ndarray:
    """Use only question and option text centers; exclude timers and animations."""
    rois = (regions.question, *regions.options)
    vectors: list[np.ndarray] = []
    for index, roi in enumerate(rois):
        size = (320, 64) if index == 0 else (160, 48)
        crop = crop_for_state(frame, roi)
        vectors.append(
            np.asarray(ImageOps.grayscale(crop).resize(size), dtype=np.float32).reshape(-1)
        )
    return np.concatenate(vectors)


def normalized_content_difference(
    first: Image.Image,
    second: Image.Image,
    regions: RegionConfig,
) -> float:
    first_vector = _content_vector(first, regions)
    second_vector = _content_vector(second, regions)
    return float(np.mean(np.abs(first_vector - second_vector)) / 255.0)


class PageStateDetector:
    """Recognize a question page using both title OCR and four white option boxes."""

    def __init__(
        self,
        config: StateConfig,
        regions: RegionConfig,
        title_reader: Callable[[Image.Image], OCRResult],
    ) -> None:
        self._config = config
        self._regions = regions
        self._title_reader = title_reader
        self._last_question_number: int | None = None
        self._last_question_seen_at = 0.0
        self._last_title_warning_at = 0.0
        self._logger = logging.getLogger(__name__)

    def observe(
        self,
        frame: Image.Image,
        *,
        force_title_ocr: bool = True,
    ) -> PageObservation:
        ratios = tuple(
            white_pixel_ratio(
                crop_for_state(frame, roi),
                self._config.white_pixel_threshold,
            )
            for roi in self._regions.option_boxes
        )
        boxes_present = all(ratio >= self._config.min_white_ratio for ratio in ratios)
        detected_number: int | None = None
        title_text = ""
        now = time.monotonic()

        # White boxes are a cheap gate. Avoid running title OCR on obvious non-question pages.
        attempted_title_ocr = boxes_present and (
            force_title_ocr
            or self._last_question_number is None
            or now - self._last_question_seen_at
            > self._config.question_number_cache_seconds
        )
        if attempted_title_ocr:
            try:
                title_result = self._title_reader(
                    crop_for_state(frame, self._regions.question_number)
                )
            except OCRError as exc:
                if now - self._last_title_warning_at >= 2.0:
                    self._logger.warning("question number OCR temporarily failed: %s", exc)
                    self._last_title_warning_at = now
            else:
                title_text = title_result.text
                detected_number = extract_question_number(title_result)
                if detected_number is not None:
                    self._last_question_number = detected_number
                    self._last_question_seen_at = now
                elif now - self._last_title_warning_at >= 2.0:
                    self._logger.warning(
                        "four option boxes are present but question number is unreadable"
                    )
                    self._last_title_warning_at = now

        cache_is_fresh = (
            self._last_question_number is not None
            and now - self._last_question_seen_at
            <= self._config.question_number_cache_seconds
        )
        effective_number = (
            detected_number
            if detected_number is not None
            else self._last_question_number if cache_is_fresh else None
        )
        return PageObservation(
            detected_question_number=detected_number,
            effective_question_number=effective_number,
            title_text=title_text,
            option_boxes_present=boxes_present,
            option_white_ratios=ratios,  # type: ignore[arg-type]
        )

    def wait_for_question_page(
        self,
        source: FrameSource,
        *,
        expected_number: int | None = None,
        excluded_number: int | None = None,
        timeout_seconds: float | None = None,
    ) -> ReadyQuestionPage | None:
        return self._wait_for_ready_page(
            source,
            expected_number=expected_number,
            excluded_number=excluded_number,
            timeout_seconds=timeout_seconds or self._config.page_wait_timeout_seconds,
            baseline_frame=None,
            require_fresh_number_frames=False,
        )

    def wait_for_next_question(
        self,
        source: FrameSource,
        *,
        old_question_number: int,
        baseline_frame: Image.Image,
    ) -> ReadyQuestionPage | None:
        return self._wait_for_ready_page(
            source,
            expected_number=None,
            excluded_number=old_question_number,
            timeout_seconds=self._config.transition_timeout_seconds,
            baseline_frame=baseline_frame,
            require_fresh_number_frames=True,
        )

    def content_remained_stable(
        self,
        ocr_frame: Image.Image,
        latest_frame: Image.Image,
    ) -> bool:
        boxes_present, _ = self.option_boxes_present(latest_frame)
        if not boxes_present:
            return False
        difference = normalized_content_difference(
            ocr_frame,
            latest_frame,
            self._regions,
        )
        return difference <= self._config.stable_threshold

    def option_boxes_present(
        self,
        frame: Image.Image,
    ) -> tuple[bool, tuple[float, float, float, float]]:
        ratios = tuple(
            white_pixel_ratio(
                crop_for_state(frame, roi),
                self._config.white_pixel_threshold,
            )
            for roi in self._regions.option_boxes
        )
        return (
            all(ratio >= self._config.min_white_ratio for ratio in ratios),
            ratios,  # type: ignore[return-value]
        )

    def _wait_for_ready_page(
        self,
        source: FrameSource,
        *,
        expected_number: int | None,
        excluded_number: int | None,
        timeout_seconds: float,
        baseline_frame: Image.Image | None,
        require_fresh_number_frames: bool,
    ) -> ReadyQuestionPage | None:
        deadline = time.monotonic() + timeout_seconds
        candidate_number: int | None = None
        option_box_count = 0
        fresh_number_count = 0
        stable_count = 0
        previous_stable_frame: Image.Image | None = None
        visual_change_seen = baseline_frame is None

        while time.monotonic() < deadline:
            frame = self.capture_with_retry(source)
            if baseline_frame is not None and not visual_change_seen:
                difference = normalized_content_difference(
                    baseline_frame,
                    frame,
                    self._regions,
                )
                visual_change_seen = difference >= self._config.change_threshold

            boxes_present, _ratios = self.option_boxes_present(frame)
            if boxes_present:
                option_box_count += 1
                if previous_stable_frame is not None:
                    content_difference = normalized_content_difference(
                        previous_stable_frame,
                        frame,
                        self._regions,
                    )
                    stable_count = (
                        stable_count + 1
                        if content_difference <= self._config.stable_threshold
                        else 1
                    )
                else:
                    stable_count = 1
                previous_stable_frame = frame

                visual_ready = (
                    visual_change_seen
                    and option_box_count >= self._config.page_confirm_frames
                    and stable_count >= self._config.required_stable_frames
                )
                if visual_ready:
                    observation = self.observe(
                        frame,
                        force_title_ocr=(
                            require_fresh_number_frames
                            or candidate_number is None
                            or expected_number is not None
                            or excluded_number is not None
                        ),
                    )
                    number = observation.effective_question_number
                    valid_number = (
                        number is not None
                        and (expected_number is None or number == expected_number)
                        and (excluded_number is None or number != excluded_number)
                    )
                    if valid_number:
                        if number != candidate_number:
                            candidate_number = number
                            fresh_number_count = 0
                        if observation.detected_question_number == candidate_number:
                            fresh_number_count += 1
                        elif require_fresh_number_frames:
                            fresh_number_count = 0

                        required_number_frames = (
                            self._config.new_question_confirm_frames
                            if require_fresh_number_frames
                            else 1
                        )
                        if fresh_number_count >= required_number_frames:
                            assert candidate_number is not None
                            return ReadyQuestionPage(frame, candidate_number, observation)
            else:
                candidate_number = None
                option_box_count = 0
                fresh_number_count = 0
                stable_count = 0
                previous_stable_frame = None

            time.sleep(self._config.poll_interval_seconds)
        return None

    def capture_with_retry(self, source: FrameSource) -> Image.Image:
        last_error: CaptureError | None = None
        for attempt in range(1, self._config.max_capture_failures + 1):
            try:
                return source.capture()
            except CaptureError as exc:
                last_error = exc
                self._logger.warning(
                    "frame capture failed (%d/%d): %s",
                    attempt,
                    self._config.max_capture_failures,
                    exc,
                )
                if attempt < self._config.max_capture_failures:
                    time.sleep(self._config.poll_interval_seconds)
        assert last_error is not None
        raise last_error
