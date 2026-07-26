from pathlib import Path

from auto_answer.config import load_config


def test_options_are_sorted_by_visual_order(tmp_path: Path) -> None:
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
[capture]
left=0
top=0
width=400
height=400
[regions.question_number]
left=100
top=0
width=100
height=50
[regions.question]
left=0
top=0
width=100
height=50
[[regions.options]]
left=200
top=200
width=100
height=50
[[regions.options]]
left=200
top=100
width=100
height=50
[[regions.options]]
left=0
top=200
width=100
height=50
[[regions.options]]
left=0
top=100
width=100
height=50
[adb]
[[adb.tap_points]]
x=-1
y=-1
[[adb.tap_points]]
x=-1
y=-1
[[adb.tap_points]]
x=-1
y=-1
[[adb.tap_points]]
x=-1
y=-1
""",
        encoding="utf-8",
    )
    config = load_config(config_file)
    assert config.ocr.use_gpu is True
    assert config.ocr.enable_mkldnn is False
    assert config.ocr.cpu_threads == 8
    assert config.ocr.text_detection_model_name == "PP-OCRv6_medium_det"
    assert config.ocr.text_recognition_model_name == "PP-OCRv6_medium_rec"
    assert config.state.new_question_confirm_frames == 1
    assert [(item.left, item.top) for item in config.regions.options] == [
        (0, 100),
        (200, 100),
        (0, 200),
        (200, 200),
    ]
