"""Frame source abstraction and Windows desktop implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image, ImageGrab

from ..core.errors import CaptureError
from ..core.models import CroppedFrame, Rect


class FrameSource(ABC):
    """Replaceable source of the latest complete scrcpy frame."""

    @abstractmethod
    def capture(self) -> Image.Image:
        """Return a full RGB frame."""

    def close(self) -> None:
        """Release optional source resources."""
        return None


class WindowsScreenFrameSource(FrameSource):
    """Capture a fixed desktop rectangle containing only the scrcpy video."""

    def __init__(self, screen_rect: Rect) -> None:
        self._rect = screen_rect

    def capture(self) -> Image.Image:
        bbox = (
            self._rect.left,
            self._rect.top,
            self._rect.right,
            self._rect.bottom,
        )
        try:
            return ImageGrab.grab(bbox=bbox, all_screens=True).convert("RGB")
        except Exception as exc:
            raise CaptureError(f"Windows screen capture failed for {bbox}: {exc}") from exc


class ImageFrameSource(FrameSource):
    """Static image source for configuration and offline smoke tests."""

    def __init__(self, image_path: str | Path) -> None:
        self._path = Path(image_path)
        try:
            with Image.open(self._path) as image:
                self._image = image.convert("RGB")
        except OSError as exc:
            raise CaptureError(f"cannot open image source {self._path}: {exc}") from exc

    def capture(self) -> Image.Image:
        return self._image.copy()


def crop_regions(
    frame: Image.Image,
    question_roi: Rect,
    option_rois: tuple[Rect, Rect, Rect, Rect],
) -> CroppedFrame:
    """Crop all five regions from exactly the same captured frame."""
    width, height = frame.size
    for label, roi in (
        ("question", question_roi),
        *((f"option[{index}]", rect) for index, rect in enumerate(option_rois)),
    ):
        if (
            roi.left < 0
            or roi.top < 0
            or roi.width <= 0
            or roi.height <= 0
            or roi.right > width
            or roi.bottom > height
        ):
            raise CaptureError(f"{label} ROI {roi} is outside frame {width}x{height}")

    return CroppedFrame(
        full_frame=frame,
        question=frame.crop(
            (question_roi.left, question_roi.top, question_roi.right, question_roi.bottom)
        ),
        options=tuple(
            frame.crop((roi.left, roi.top, roi.right, roi.bottom)) for roi in option_rois
        ),  # type: ignore[arg-type]
    )
