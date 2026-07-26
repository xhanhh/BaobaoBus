from pathlib import Path

from auto_answer.core.config import load_config


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
    assert config.state.overlap_ocr_with_stability is True
    assert config.state.title_white_pixel_threshold == 210
    assert config.state.title_min_white_ratio == 0.02
    assert config.state.title_probe_interval_seconds == 0.75
    assert config.state.ready_poll_interval_seconds == 0.005
    assert config.state.ready_fast_window_seconds == 6.0
    assert config.state.infer_first_question_number_after_ready is True
    assert config.state.infer_sequential_question_number is True
    assert config.regions.ready_indicator is None
    assert config.ollama.retry_numeric_as_text is True
    assert config.fallback.random_on_ocr_failure is False
    assert config.fallback.random_on_llm_failure is False
    assert config.adb.persistent_shell is True
    assert [(item.left, item.top) for item in config.regions.options] == [
        (0, 100),
        (200, 100),
        (0, 200),
        (200, 200),
    ]
