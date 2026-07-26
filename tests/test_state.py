from PIL import Image

from auto_answer.core.config import RegionConfig, StateConfig
from auto_answer.core.models import OCRResult, Rect
from auto_answer.vision.capture import FrameSource
from auto_answer.vision.state import (
    PageStateDetector,
    extract_question_number,
    normalized_content_difference,
    normalized_image_difference,
    ready_indicator_ratios,
)


def test_normalized_difference_bounds_and_identity() -> None:
    black = Image.new("RGB", (20, 20), "black")
    white = Image.new("RGB", (20, 20), "white")
    assert normalized_image_difference(black, black) == 0.0
    assert normalized_image_difference(black, white) == 1.0


def test_extract_question_number_requires_confident_title() -> None:
    assert extract_question_number(OCRResult("第 12 题", 0.99, False)) == 12
    assert extract_question_number(OCRResult("第12题", 0.20, True)) is None
    assert extract_question_number(OCRResult("", 0.0, True)) is None


def test_page_observation_requires_title_and_all_four_white_boxes() -> None:
    frame = Image.new("RGB", (200, 200), "black")
    option_boxes = (
        Rect(0, 50, 50, 40),
        Rect(50, 50, 50, 40),
        Rect(0, 90, 50, 40),
        Rect(50, 90, 50, 40),
    )
    for roi in option_boxes:
        frame.paste("white", (roi.left, roi.top, roi.right, roi.bottom))
    regions = RegionConfig(
        question_number=Rect(0, 0, 100, 40),
        question=Rect(100, 0, 100, 40),
        options=option_boxes,
        option_boxes=option_boxes,
    )
    detector = PageStateDetector(
        StateConfig(white_pixel_threshold=210, min_white_ratio=0.9),
        regions,
        lambda _image: OCRResult("第3题", 0.99, False),
    )
    observation = detector.observe(frame)
    assert observation.is_question_page
    assert observation.detected_question_number == 3

    frame.paste("black", (0, 50, 50, 90))
    assert not detector.observe(frame).is_question_page


def test_title_visibility_gate_skips_ocr_during_fade() -> None:
    frame = Image.new("RGB", (200, 200), "black")
    option_boxes = (
        Rect(0, 50, 50, 40),
        Rect(50, 50, 50, 40),
        Rect(0, 90, 50, 40),
        Rect(50, 90, 50, 40),
    )
    for roi in option_boxes:
        frame.paste("white", (roi.left, roi.top, roi.right, roi.bottom))
    regions = RegionConfig(
        question_number=Rect(0, 0, 100, 40),
        question=Rect(100, 0, 100, 40),
        options=option_boxes,
        option_boxes=option_boxes,
    )
    calls = 0

    def title_reader(_image: Image.Image) -> OCRResult:
        nonlocal calls
        calls += 1
        return OCRResult("第3题", 0.99, False)

    detector = PageStateDetector(
        StateConfig(
            white_pixel_threshold=210,
            min_white_ratio=0.9,
            title_min_white_ratio=0.02,
            title_probe_interval_seconds=10.0,
        ),
        regions,
        title_reader,
    )
    frame.paste("white", (10, 10, 30, 20))
    assert detector.observe(frame).detected_question_number == 3
    assert calls == 1
    frame.paste("black", (0, 0, 100, 40))
    faded = detector.observe(frame)
    assert faded.detected_question_number is None
    assert calls == 1


def test_ready_indicator_uses_colored_text_on_purple_background() -> None:
    ready = Image.new("RGB", (100, 50), (100, 80, 240))
    ready.paste((240, 180, 50), (20, 10, 80, 30))
    text_ratio, purple_ratio = ready_indicator_ratios(ready)
    assert text_ratio >= 0.15
    assert purple_ratio >= 0.60

    versus = Image.new("RGB", (100, 50), (220, 80, 40))
    versus.paste((240, 180, 50), (20, 10, 80, 30))
    _text_ratio, purple_ratio = ready_indicator_ratios(versus)
    assert purple_ratio < 0.60


def test_ready_page_arms_first_question_without_separate_title_ocr() -> None:
    option_boxes = (
        Rect(0, 40, 50, 20),
        Rect(50, 40, 50, 20),
        Rect(0, 60, 50, 20),
        Rect(50, 60, 50, 20),
    )
    regions = RegionConfig(
        question_number=Rect(0, 0, 20, 20),
        question=Rect(20, 0, 80, 20),
        options=option_boxes,
        option_boxes=option_boxes,
        ready_indicator=Rect(20, 20, 60, 20),
    )
    ready = Image.new("RGB", (100, 100), (100, 80, 240))
    ready.paste((240, 180, 50), (30, 22, 70, 30))
    entering = Image.new("RGB", (100, 100), "black")
    entering.paste("white", (0, 0, 20, 20))
    entering.paste("white", (0, 40, 50, 60))
    answer = entering.copy()
    for roi in option_boxes:
        answer.paste("white", (roi.left, roi.top, roi.right, roi.bottom))
    title_calls = 0

    def title_reader(_image: Image.Image) -> OCRResult:
        nonlocal title_calls
        title_calls += 1
        return OCRResult("第1题", 0.99, False)

    detector = PageStateDetector(
        StateConfig(
            poll_interval_seconds=0.001,
            ready_poll_interval_seconds=0.001,
            ready_fast_window_seconds=1.0,
            ready_confirm_frames=2,
            ready_min_text_color_ratio=0.15,
            ready_min_purple_ratio=0.60,
            required_stable_frames=2,
            page_confirm_frames=2,
            page_wait_timeout_seconds=0.2,
            min_white_ratio=0.9,
            overlap_ocr_with_stability=True,
        ),
        regions,
        title_reader,
    )
    result = detector.wait_for_question_page(
        SequenceSource([ready, ready, entering, answer])
    )
    assert result is not None
    assert result.question_number == 1
    assert result.question_number_inferred
    assert result.ready_to_answer_ms is not None
    assert result.answer_to_confirm_ms is not None
    assert title_calls == 0


def test_content_difference_uses_question_and_option_text_regions() -> None:
    regions = RegionConfig(
        question_number=Rect(0, 0, 20, 20),
        question=Rect(0, 20, 100, 20),
        options=(
            Rect(0, 40, 50, 20),
            Rect(50, 40, 50, 20),
            Rect(0, 60, 50, 20),
            Rect(50, 60, 50, 20),
        ),
        option_boxes=(
            Rect(0, 40, 50, 20),
            Rect(50, 40, 50, 20),
            Rect(0, 60, 50, 20),
            Rect(50, 60, 50, 20),
        ),
    )
    first = Image.new("RGB", (100, 100), "black")
    second = first.copy()
    second.paste("white", (0, 80, 100, 100))
    assert normalized_content_difference(first, second, regions) == 0.0
    second.paste("white", (0, 20, 100, 40))
    assert normalized_content_difference(first, second, regions) > 0.0


class SequenceSource(FrameSource):
    def __init__(self, frames: list[Image.Image]) -> None:
        self.frames = frames
        self.index = 0

    def capture(self) -> Image.Image:
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return frame.copy()


def test_overlap_starts_ocr_candidate_one_confirmation_frame_early() -> None:
    option_boxes = (
        Rect(0, 40, 50, 20),
        Rect(50, 40, 50, 20),
        Rect(0, 60, 50, 20),
        Rect(50, 60, 50, 20),
    )
    regions = RegionConfig(
        question_number=Rect(0, 0, 20, 20),
        question=Rect(20, 0, 80, 20),
        options=option_boxes,
        option_boxes=option_boxes,
    )
    frame = Image.new("RGB", (100, 100), "black")
    for roi in option_boxes:
        frame.paste("white", (roi.left, roi.top, roi.right, roi.bottom))
    source = SequenceSource([frame, frame])
    detector = PageStateDetector(
        StateConfig(
            poll_interval_seconds=0.001,
            required_stable_frames=2,
            page_confirm_frames=2,
            page_wait_timeout_seconds=0.1,
            min_white_ratio=0.9,
            overlap_ocr_with_stability=True,
        ),
        regions,
        lambda _image: OCRResult("第1题", 0.99, False),
    )
    result = detector.wait_for_question_page(source)
    assert result is not None
    assert source.index == 1


def test_next_question_requires_change_repeated_new_number_and_stability() -> None:
    option_boxes = (
        Rect(0, 40, 50, 20),
        Rect(50, 40, 50, 20),
        Rect(0, 60, 50, 20),
        Rect(50, 60, 50, 20),
    )
    regions = RegionConfig(
        question_number=Rect(0, 0, 20, 20),
        question=Rect(20, 0, 80, 20),
        options=option_boxes,
        option_boxes=option_boxes,
    )

    def page(title_color: str, question_color: str, *, boxes: bool = True) -> Image.Image:
        frame = Image.new("RGB", (100, 100), "black")
        frame.paste(title_color, (0, 0, 20, 20))
        frame.paste(question_color, (20, 0, 100, 20))
        if boxes:
            for roi in option_boxes:
                frame.paste("white", (roi.left, roi.top, roi.right, roi.bottom))
        return frame

    old = page("red", "black")
    transition = page("black", "black", boxes=False)
    new = page("blue", "white")

    def title_reader(image: Image.Image) -> OCRResult:
        red, _green, blue = image.getpixel((0, 0))
        if red > blue:
            return OCRResult("第1题", 0.99, False)
        if blue > red:
            return OCRResult("第2题", 0.99, False)
        return OCRResult("", 0.0, True)

    detector = PageStateDetector(
        StateConfig(
            stable_threshold=0.01,
            change_threshold=0.05,
            poll_interval_seconds=0.001,
            required_stable_frames=2,
            page_confirm_frames=2,
            new_question_confirm_frames=2,
            transition_timeout_seconds=0.2,
            min_white_ratio=0.9,
            title_min_white_ratio=0.0,
        ),
        regions,
        title_reader,
    )
    result = detector.wait_for_next_question(
        SequenceSource([transition, new, new, new]),
        old_question_number=1,
        baseline_frame=old,
    )
    assert result is not None
    assert result.question_number == 2
