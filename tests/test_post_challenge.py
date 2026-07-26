from PIL import Image, ImageDraw

from auto_answer.core.config import PostChallengeConfig
from auto_answer.core.models import Point, Rect
from auto_answer.runtime.post_challenge import (
    PostChallengeController,
    PostChallengeOutcome,
)
from auto_answer.vision.capture import FrameSource
from auto_answer.vision.post_challenge import observe_post_challenge


def post_config() -> PostChallengeConfig:
    return PostChallengeConfig(
        enabled=True,
        success_banner=Rect(0, 0, 20, 20),
        continue_button=Rect(20, 0, 20, 20),
        ranking_panel=Rect(0, 20, 40, 20),
        ranking_close_button=Rect(40, 20, 20, 20),
        ranking_close_tap=Point(1980, 300),
        continue_challenge_tap=Point(1660, 960),
        poll_interval_seconds=0.001,
        detection_timeout_seconds=0.05,
        ranking_popup_wait_seconds=0.05,
        ranking_close_timeout_seconds=0.05,
        ready_timeout_seconds=0.05,
        confirm_frames=2,
    )


def success_frame() -> Image.Image:
    image = Image.new("RGB", (100, 60), (80, 80, 180))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 19, 19), fill=(0, 210, 20))
    draw.rectangle((20, 0, 39, 19), fill=(255, 140, 0))
    return image


def failure_frame() -> Image.Image:
    image = Image.new("RGB", (100, 60), (80, 80, 180))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 19, 19), fill=(80, 110, 190))
    draw.rectangle((20, 0, 39, 19), fill=(255, 140, 0))
    return image


def popup_frame() -> Image.Image:
    image = Image.new("RGB", (100, 60), (20, 20, 40))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 19, 19), fill=(255, 0, 120))
    draw.rectangle((0, 20, 39, 39), fill=(250, 250, 245))
    draw.rectangle((0, 20, 19, 39), fill=(255, 0, 120))
    draw.rectangle((40, 20, 59, 39), fill=(250, 250, 245))
    draw.rectangle((40, 20, 44, 39), fill=(255, 140, 0))
    return image


def ready_frame() -> Image.Image:
    image = Image.new("RGB", (100, 60), (80, 80, 180))
    image.putpixel((99, 59), (1, 2, 3))
    return image


class SequenceSource(FrameSource):
    def __init__(self, frames: list[Image.Image]) -> None:
        self.frames = frames
        self.index = 0

    def capture(self) -> Image.Image:
        index = min(self.index, len(self.frames) - 1)
        self.index += 1
        return self.frames[index].copy()


class RecordingADB:
    def __init__(self) -> None:
        self.taps: list[tuple[Point, str]] = []

    def tap_point(self, point: Point, *, purpose: str) -> None:
        self.taps.append((point, purpose))


def test_success_and_popup_require_combined_features() -> None:
    config = post_config()
    success = observe_post_challenge(success_frame(), config)
    failure = observe_post_challenge(failure_frame(), config)
    popup = observe_post_challenge(popup_frame(), config)
    unrelated = observe_post_challenge(Image.new("RGB", (100, 60), "white"), config)

    assert success.success_page_visible
    assert not success.failure_page_visible
    assert not success.ranking_popup_visible
    assert failure.failure_page_visible
    assert not failure.success_page_visible
    assert not failure.ranking_popup_visible
    assert popup.ranking_popup_visible
    assert not popup.success_page_visible
    assert not unrelated.success_page_visible
    assert not unrelated.failure_page_visible
    assert not unrelated.ranking_popup_visible


def test_controller_closes_popup_then_continues_and_confirms_ready() -> None:
    source = SequenceSource(
        [
            success_frame(),
            success_frame(),
            popup_frame(),
            popup_frame(),
            success_frame(),
            success_frame(),
            ready_frame(),
            ready_frame(),
        ]
    )
    adb = RecordingADB()
    controller = PostChallengeController(
        post_config(),
        source,
        adb,  # type: ignore[arg-type]
        lambda frame: frame.getpixel((99, 59)) == (1, 2, 3),
    )

    outcome = controller.handle_if_present()

    assert outcome is PostChallengeOutcome.READY_CONFIRMED
    assert adb.taps == [
        (Point(1980, 300), "ranking popup close"),
        (Point(1660, 960), "continue challenge"),
    ]


def test_controller_continues_directly_from_failure_page() -> None:
    source = SequenceSource(
        [
            failure_frame(),
            failure_frame(),
            failure_frame(),
            failure_frame(),
            ready_frame(),
            ready_frame(),
        ]
    )
    adb = RecordingADB()
    controller = PostChallengeController(
        post_config(),
        source,
        adb,  # type: ignore[arg-type]
        lambda frame: frame.getpixel((99, 59)) == (1, 2, 3),
    )

    outcome = controller.handle_if_present()

    assert outcome is PostChallengeOutcome.READY_CONFIRMED
    assert adb.taps == [(Point(1660, 960), "continue challenge")]
