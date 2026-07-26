from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from auto_answer.capture import FrameSource
from auto_answer.config import DebugConfig, RegionConfig
from auto_answer.debug import DebugRecorder
from auto_answer.models import OCRBundle, OCRResult, Rect
from auto_answer.rules import RuleEngine
from auto_answer.scheduler import AnswerScheduler
from auto_answer.state import PageObservation, ReadyQuestionPage


class StaticSource(FrameSource):
    def __init__(self, frame: Image.Image) -> None:
        self.frame = frame

    def capture(self) -> Image.Image:
        return self.frame.copy()


class RetryOCR:
    def __init__(self) -> None:
        self.calls = 0

    def recognize_with_title(
        self,
        _crops: object,
        _title: object,
    ) -> tuple[OCRBundle, OCRResult]:
        self.calls += 1
        valid = OCRResult("2", 0.99, False)
        empty = OCRResult("", 0.0, True)
        options = (valid, empty, valid, valid) if self.calls == 1 else (
            OCRResult("1", 0.99, False),
            OCRResult("2", 0.99, False),
            OCRResult("3", 0.99, False),
            OCRResult("4", 0.99, False),
        )
        return (
            OCRBundle(OCRResult("1+1=?", 0.99, False), options),
            OCRResult("第1题", 0.99, False),
        )


class FakeDetector:
    def __init__(self, page: ReadyQuestionPage) -> None:
        self.page = page

    def wait_for_question_page(self, *_args: object, **_kwargs: object) -> ReadyQuestionPage:
        return self.page

    def capture_with_retry(self, source: StaticSource) -> Image.Image:
        return source.capture()

    def content_remained_stable(self, _first: Image.Image, _second: Image.Image) -> bool:
        return True


class FakeOllama:
    def close(self) -> None:
        return None


def test_transient_incomplete_ocr_is_retried_without_stopping(
    tmp_path: Path,
    caplog: object,
) -> None:
    frame = Image.new("RGB", (100, 100), "white")
    option_rois = (
        Rect(0, 40, 50, 20),
        Rect(50, 40, 50, 20),
        Rect(0, 60, 50, 20),
        Rect(50, 60, 50, 20),
    )
    regions = RegionConfig(
        question_number=Rect(0, 0, 50, 20),
        question=Rect(0, 20, 100, 20),
        options=option_rois,
        option_boxes=option_rois,
    )
    page = ReadyQuestionPage(
        frame,
        1,
        PageObservation(1, 1, "第1题", True, (1.0, 1.0, 1.0, 1.0)),
    )
    config = SimpleNamespace(
        regions=regions,
        state=SimpleNamespace(
            ocr_retry_attempts=2,
            ocr_retry_interval_seconds=0.0,
            stable_timeout_seconds=1.0,
            page_wait_timeout_seconds=1.0,
            transition_timeout_seconds=1.0,
        ),
        debug=SimpleNamespace(save_each_question=False),
    )
    ocr = RetryOCR()
    scheduler = AnswerScheduler(
        config=config,
        source=StaticSource(frame),
        ocr=ocr,
        rules=RuleEngine(),
        ollama=FakeOllama(),
        state_detector=FakeDetector(page),
        recorder=DebugRecorder(DebugConfig(enabled=False, output_dir=tmp_path)),
        adb=None,
    )
    caplog.set_level("INFO")  # type: ignore[attr-defined]
    decision = scheduler.run(dry_run=True)
    assert decision is not None
    assert decision.answer_index == 1
    assert ocr.calls == 2
    assert "recognize_to_decision_ms=" in caplog.text  # type: ignore[attr-defined]
