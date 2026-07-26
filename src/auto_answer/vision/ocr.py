"""Single-instance PaddleOCR adapter with batch-first inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from ..core.config import OCRConfig
from ..core.errors import OCRError
from ..core.models import CroppedFrame, OCRBundle, OCRResult


@dataclass(frozen=True, slots=True)
class _RecognizedLine:
    text: str
    score: float
    box: Any
    area: float
    height: float


class PaddleOCRReader:
    """Initialize PaddleOCR once and reuse it for every question."""

    def __init__(self, config: OCRConfig) -> None:
        self._config = config
        try:
            if config.use_gpu:
                import paddle

                if not paddle.device.is_compiled_with_cuda():
                    raise OCRError(
                        "OCR is configured for GPU, but Paddle was not compiled with CUDA"
                    )
                if paddle.device.cuda.device_count() < 1:
                    raise OCRError(
                        "OCR is configured for GPU, but Paddle cannot see a CUDA device"
                    )

            from paddleocr import PaddleOCR

            arguments: dict[str, object] = {
                "device": "gpu:0" if config.use_gpu else "cpu",
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
                "enable_mkldnn": config.enable_mkldnn,
                "cpu_threads": config.cpu_threads,
            }
            if config.text_detection_model_name:
                arguments["text_detection_model_name"] = config.text_detection_model_name
            if config.text_recognition_model_name:
                arguments["text_recognition_model_name"] = config.text_recognition_model_name
            if not config.text_detection_model_name and not config.text_recognition_model_name:
                arguments["lang"] = config.language
            self._engine = PaddleOCR(
                **arguments,
            )
        except Exception as exc:
            raise OCRError(f"cannot initialize PaddleOCR: {exc}") from exc

    def recognize(self, crops: CroppedFrame) -> OCRBundle:
        parsed = self._recognize_images([crops.question, *crops.options])
        options = tuple(
            self._recover_empty_option_symbol(result, image)
            for result, image in zip(parsed[1:], crops.options, strict=True)
        )
        return OCRBundle(
            question=parsed[0],
            options=options,  # type: ignore[arg-type]
        )

    def recognize_with_title(
        self,
        crops: CroppedFrame,
        title_image: Image.Image,
    ) -> tuple[OCRBundle, OCRResult]:
        """Batch the title, question, and options in one inference call."""
        parsed = self._recognize_images([title_image, crops.question, *crops.options])
        options = tuple(
            self._recover_empty_option_symbol(result, image)
            for result, image in zip(parsed[2:], crops.options, strict=True)
        )
        return (
            OCRBundle(
                question=parsed[1],
                options=options,  # type: ignore[arg-type]
            ),
            parsed[0],
        )

    def recognize_single(self, image: Image.Image) -> OCRResult:
        """Recognize a small page-feature ROI while reusing the same model."""
        return self._recognize_images([image])[0]

    def _recognize_images(self, images: Sequence[Image.Image]) -> tuple[OCRResult, ...]:
        # PaddleOCR's ndarray input follows OpenCV's BGR channel convention.
        arrays = [np.asarray(image.convert("RGB"))[:, :, ::-1].copy() for image in images]
        try:
            if hasattr(self._engine, "predict"):
                raw_results = list(self._engine.predict(arrays))
            else:
                # Compatibility with PaddleOCR 2.x. This still reuses one heavy model.
                raw_results = [self._engine.ocr(array, cls=False) for array in arrays]
        except Exception as exc:
            raise OCRError(f"PaddleOCR inference failed: {exc}") from exc
        if len(raw_results) != len(images):
            raise OCRError(
                f"PaddleOCR returned {len(raw_results)} results for {len(images)} regions"
            )
        return tuple(self._parse_result(item) for item in raw_results)

    def _parse_result(self, raw: Any) -> OCRResult:
        mapping = self._as_mapping(raw)
        if mapping is not None and "rec_texts" in mapping:
            texts = [str(item).strip() for item in mapping.get("rec_texts", [])]
            scores = [float(item) for item in mapping.get("rec_scores", [])]
            boxes = tuple(self._to_serializable(item) for item in mapping.get("rec_boxes", []))
            return self._build(texts, scores, boxes)

        texts: list[str] = []
        scores: list[float] = []
        boxes: list[Any] = []
        for item in self._flatten_legacy_lines(raw):
            try:
                box, recognition = item
                text, score = recognition
                texts.append(str(text).strip())
                scores.append(float(score))
                boxes.append(self._to_serializable(box))
            except (TypeError, ValueError):
                continue
        return self._build(texts, scores, tuple(boxes))

    def _build(
        self,
        texts: Sequence[str],
        scores: Sequence[float],
        boxes: tuple[Any, ...],
    ) -> OCRResult:
        if len(scores) < len(texts):
            scores = [*scores, *([0.0] * (len(texts) - len(scores)))]
        lines: list[_RecognizedLine] = []
        for index, (text, score) in enumerate(zip(texts, scores, strict=False)):
            if not text:
                continue
            box = boxes[index] if index < len(boxes) else None
            area, height = self._box_metrics(box)
            lines.append(_RecognizedLine(text, score, box, area, height))

        lines = self._filter_isolated_fragments(lines)
        clean_texts = tuple(item.text for item in lines)
        clean_scores = tuple(item.score for item in lines)
        clean_boxes = tuple(item.box for item in lines if item.box is not None)
        weights = tuple(max(len("".join(item.text.split())), 1) for item in lines)
        weight_sum = sum(weights)
        confidence = (
            sum(item.score * weight for item, weight in zip(lines, weights, strict=True))
            / weight_sum
            if weight_sum
            else 0.0
        )
        return OCRResult(
            text="\n".join(clean_texts),
            confidence=confidence,
            low_confidence=confidence < self._config.confidence_threshold,
            lines=clean_texts,
            line_confidences=clean_scores,
            boxes=clean_boxes,
        )

    def _filter_isolated_fragments(
        self,
        lines: list[_RecognizedLine],
    ) -> list[_RecognizedLine]:
        """Discard tiny, short detections without hiding normal-size single characters."""
        measurable = [item for item in lines if item.area > 0 and item.height > 0]
        if len(measurable) < 2:
            return lines
        max_area = max(item.area for item in measurable)
        max_height = max(item.height for item in measurable)
        if max_area <= 0 or max_height <= 0:
            return lines

        kept: list[_RecognizedLine] = []
        for item in lines:
            compact_length = len("".join(item.text.split()))
            short = compact_length <= self._config.isolated_fragment_max_chars
            tiny_area = (
                item.area > 0
                and item.area / max_area
                <= self._config.isolated_fragment_max_area_ratio
            )
            tiny_height = (
                item.height > 0
                and item.height / max_height
                <= self._config.isolated_fragment_max_height_ratio
            )
            if short and tiny_area and tiny_height:
                continue
            kept.append(item)
        return kept

    def _recover_empty_option_symbol(
        self,
        result: OCRResult,
        image: Image.Image,
    ) -> OCRResult:
        """Recover only a clearly visible, thin horizontal minus missed by OCR."""
        if result.text.strip() or not self._config.recover_thin_minus:
            return result

        grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
        height, width = grayscale.shape
        left = int(width * 0.25)
        right = int(width * 0.75)
        top = int(height * 0.15)
        bottom = int(height * 0.85)
        center = grayscale[top:bottom, left:right]
        foreground = center < self._config.symbol_foreground_threshold
        ys, xs = np.where(foreground)
        if len(xs) < 12:
            return result

        stroke_left = int(xs.min())
        stroke_right = int(xs.max())
        stroke_top = int(ys.min())
        stroke_bottom = int(ys.max())
        stroke_width = stroke_right - stroke_left + 1
        stroke_height = stroke_bottom - stroke_top + 1
        if stroke_width < 12 or stroke_height > 4 or stroke_width < stroke_height * 6:
            return result

        confidence = 0.90
        return OCRResult(
            text="-",
            confidence=confidence,
            low_confidence=confidence < self._config.confidence_threshold,
            lines=("-",),
            line_confidences=(confidence,),
            boxes=(
                [
                    left + stroke_left,
                    top + stroke_top,
                    left + stroke_right + 1,
                    top + stroke_bottom + 1,
                ],
            ),
        )

    @staticmethod
    def _box_metrics(box: Any) -> tuple[float, float]:
        if box is None:
            return 0.0, 0.0
        try:
            points = np.asarray(box, dtype=float)
        except (TypeError, ValueError):
            return 0.0, 0.0
        if points.shape == (4,):
            width = max(points[2] - points[0], 0.0)
            height = max(points[3] - points[1], 0.0)
            return float(width * height), float(height)
        if points.ndim == 2 and points.shape[0] >= 3 and points.shape[1] == 2:
            x = points[:, 0]
            y = points[:, 1]
            area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
            height = float(max(y) - min(y))
            return area, max(height, 0.0)
        return 0.0, 0.0

    @staticmethod
    def _as_mapping(raw: Any) -> Mapping[str, Any] | None:
        if isinstance(raw, Mapping):
            return raw
        json_value = getattr(raw, "json", None)
        if callable(json_value):
            json_value = json_value()
        if isinstance(json_value, Mapping):
            result = json_value.get("res", json_value)
            return result if isinstance(result, Mapping) else None
        result = getattr(raw, "res", None)
        return result if isinstance(result, Mapping) else None

    @staticmethod
    def _flatten_legacy_lines(raw: Any) -> list[Any]:
        if not isinstance(raw, (list, tuple)):
            return []
        if len(raw) == 1 and isinstance(raw[0], (list, tuple)):
            return list(raw[0])
        return list(raw)

    @staticmethod
    def _to_serializable(value: Any) -> Any:
        return value.tolist() if hasattr(value, "tolist") else value
