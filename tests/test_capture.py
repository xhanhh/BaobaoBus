from PIL import Image

from auto_answer.core.models import Rect
from auto_answer.vision.capture import crop_regions


def test_all_regions_are_cropped_from_same_frame() -> None:
    frame = Image.new("RGB", (100, 100), "white")
    result = crop_regions(
        frame,
        Rect(0, 0, 50, 20),
        (
            Rect(0, 20, 50, 20),
            Rect(50, 20, 50, 20),
            Rect(0, 40, 50, 20),
            Rect(50, 40, 50, 20),
        ),
    )
    assert result.full_frame is frame
    assert result.question.size == (50, 20)
    assert all(option.size == (50, 20) for option in result.options)
