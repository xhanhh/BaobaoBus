"""TOML configuration loading and validation."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .models import Point, Rect


def _rect(
    data: dict[str, Any],
    label: str,
    *,
    allow_negative_origin: bool = False,
) -> Rect:
    try:
        value = Rect(
            left=int(data["left"]),
            top=int(data["top"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label} must contain integer left/top/width/height") from exc
    invalid_origin = not allow_negative_origin and (value.left < 0 or value.top < 0)
    if invalid_origin or value.width <= 0 or value.height <= 0:
        origin_rule = "any integer" if allow_negative_origin else "non-negative"
        raise ConfigurationError(
            f"{label} origin must be {origin_rule} and size must be positive"
        )
    return value


def _point(data: dict[str, Any], label: str) -> Point:
    try:
        return Point(x=int(data["x"]), y=int(data["y"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"{label} must contain integer x/y") from exc


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    screen_rect: Rect


@dataclass(frozen=True, slots=True)
class RegionConfig:
    question_number: Rect
    question: Rect
    options: tuple[Rect, Rect, Rect, Rect]
    option_boxes: tuple[Rect, Rect, Rect, Rect]
    ready_indicator: Rect | None = None


@dataclass(frozen=True, slots=True)
class OCRConfig:
    language: str = "ch"
    confidence_threshold: float = 0.60
    option_symbol_confidence_threshold: float = 0.45
    use_gpu: bool = True
    enable_mkldnn: bool = False
    cpu_threads: int = 8
    text_detection_model_name: str | None = "PP-OCRv6_medium_det"
    text_recognition_model_name: str | None = "PP-OCRv6_medium_rec"
    isolated_fragment_max_chars: int = 1
    isolated_fragment_max_area_ratio: float = 0.02
    isolated_fragment_max_height_ratio: float = 0.50
    recover_thin_minus: bool = True
    symbol_foreground_threshold: int = 180


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    base_url: str = "http://localhost:11434"
    model: str = "qwen3.5:9b"
    timeout_seconds: float = 20.0
    keep_alive: str = "30m"
    num_predict: int = 64
    warmup_on_start: bool = True
    retry_numeric_as_text: bool = True


@dataclass(frozen=True, slots=True)
class OpenAICompatibleConfig:
    enabled: bool = False
    base_url: str = (
        "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    )
    model: str = "qwen3.7-flash"
    api_key: str = ""
    api_key_env: str = "DASHSCOPE_API_KEY"
    timeout_seconds: float = 10.0
    enable_thinking: bool = False
    retry_numeric_as_text: bool = True


@dataclass(frozen=True, slots=True)
class LLMRoutingConfig:
    provider_order: tuple[str, ...] = ("aliyun", "ollama")


@dataclass(frozen=True, slots=True)
class ADBConfig:
    executable: Path
    tap_points: tuple[Point, Point, Point, Point]
    serial: str | None = None
    timeout_seconds: float = 5.0
    persistent_shell: bool = True


@dataclass(frozen=True, slots=True)
class StateConfig:
    stable_threshold: float = 0.010
    change_threshold: float = 0.060
    poll_interval_seconds: float = 0.15
    required_stable_frames: int = 3
    stable_timeout_seconds: float = 8.0
    change_timeout_seconds: float = 8.0
    page_confirm_frames: int = 2
    new_question_confirm_frames: int = 1
    page_wait_timeout_seconds: float = 10.0
    transition_timeout_seconds: float = 12.0
    question_number_cache_seconds: float = 1.0
    ocr_retry_attempts: int = 3
    ocr_retry_interval_seconds: float = 0.25
    white_pixel_threshold: int = 210
    min_white_ratio: float = 0.55
    max_capture_failures: int = 3
    overlap_ocr_with_stability: bool = True
    title_white_pixel_threshold: int = 210
    title_min_white_ratio: float = 0.02
    title_probe_interval_seconds: float = 0.75
    ready_poll_interval_seconds: float = 0.005
    ready_fast_window_seconds: float = 6.0
    ready_confirm_frames: int = 2
    ready_min_text_color_ratio: float = 0.15
    ready_min_purple_ratio: float = 0.60
    infer_first_question_number_after_ready: bool = True
    infer_sequential_question_number: bool = True


@dataclass(frozen=True, slots=True)
class PostChallengeConfig:
    enabled: bool = False
    success_banner: Rect | None = None
    continue_button: Rect | None = None
    ranking_panel: Rect | None = None
    ranking_close_button: Rect | None = None
    ranking_close_tap: Point | None = None
    continue_challenge_tap: Point | None = None
    poll_interval_seconds: float = 0.10
    detection_timeout_seconds: float = 3.0
    ranking_popup_wait_seconds: float = 4.0
    ranking_close_timeout_seconds: float = 10.0
    ready_timeout_seconds: float = 30.0
    confirm_frames: int = 3
    max_capture_failures: int = 3
    success_green_ratio: float = 0.35
    failure_blue_ratio: float = 0.55
    continue_orange_ratio: float = 0.30
    popup_banner_pink_ratio: float = 0.35
    popup_panel_pink_ratio: float = 0.10
    popup_panel_white_ratio: float = 0.30
    popup_close_white_ratio: float = 0.20
    popup_close_orange_ratio: float = 0.03


@dataclass(frozen=True, slots=True)
class DebugConfig:
    enabled: bool = True
    output_dir: Path = Path("artifacts/failures")
    save_each_question: bool = False


@dataclass(frozen=True, slots=True)
class FallbackConfig:
    random_on_ocr_failure: bool = False
    random_on_llm_failure: bool = False


@dataclass(frozen=True, slots=True)
class AppConfig:
    capture: CaptureConfig
    regions: RegionConfig
    ocr: OCRConfig
    ollama: OllamaConfig
    openai_compatible: OpenAICompatibleConfig
    llm: LLMRoutingConfig
    adb: ADBConfig
    state: StateConfig
    post_challenge: PostChallengeConfig
    debug: DebugConfig
    fallback: FallbackConfig
    log_file: Path = Path("artifacts/auto-answer.log")

    def validate(self, *, require_live_coordinates: bool = True) -> None:
        frame = self.capture.screen_rect
        region_entries = [
            ("regions.question_number", self.regions.question_number),
            ("regions.question", self.regions.question),
            *(
                (f"regions.options[{i}]", rect)
                for i, rect in enumerate(self.regions.options)
            ),
            *(
                (f"regions.option_boxes[{i}]", rect)
                for i, rect in enumerate(self.regions.option_boxes)
            ),
        ]
        if self.regions.ready_indicator is not None:
            region_entries.append(
                ("regions.ready_indicator", self.regions.ready_indicator)
            )
        post_challenge_regions = (
            ("post_challenge.success_banner", self.post_challenge.success_banner),
            ("post_challenge.continue_button", self.post_challenge.continue_button),
            ("post_challenge.ranking_panel", self.post_challenge.ranking_panel),
            (
                "post_challenge.ranking_close_button",
                self.post_challenge.ranking_close_button,
            ),
        )
        region_entries.extend(
            (label, roi)
            for label, roi in post_challenge_regions
            if roi is not None
        )
        for label, roi in region_entries:
            if roi.right > frame.width or roi.bottom > frame.height:
                raise ConfigurationError(
                    f"{label}={roi} exceeds captured frame size {frame.width}x{frame.height}"
                )
        if not 0 <= self.ocr.confidence_threshold <= 1:
            raise ConfigurationError("ocr.confidence_threshold must be between 0 and 1")
        if not 0 <= self.ocr.option_symbol_confidence_threshold <= 1:
            raise ConfigurationError(
                "ocr.option_symbol_confidence_threshold must be between 0 and 1"
            )
        if self.ocr.cpu_threads <= 0:
            raise ConfigurationError("ocr.cpu_threads must be positive")
        if self.ocr.isolated_fragment_max_chars < 0:
            raise ConfigurationError("ocr.isolated_fragment_max_chars must be non-negative")
        if not 0 <= self.ocr.isolated_fragment_max_area_ratio <= 1:
            raise ConfigurationError(
                "ocr.isolated_fragment_max_area_ratio must be between 0 and 1"
            )
        if not 0 <= self.ocr.isolated_fragment_max_height_ratio <= 1:
            raise ConfigurationError(
                "ocr.isolated_fragment_max_height_ratio must be between 0 and 1"
            )
        if not 0 <= self.ocr.symbol_foreground_threshold <= 255:
            raise ConfigurationError(
                "ocr.symbol_foreground_threshold must be between 0 and 255"
            )
        if not 0 <= self.state.stable_threshold < self.state.change_threshold <= 1:
            raise ConfigurationError(
                "state thresholds must satisfy 0 <= stable_threshold < change_threshold <= 1"
            )
        if self.state.required_stable_frames < 2:
            raise ConfigurationError("state.required_stable_frames must be at least 2")
        if self.state.page_confirm_frames < 2:
            raise ConfigurationError("state.page_confirm_frames must be at least 2")
        if self.state.new_question_confirm_frames < 1:
            raise ConfigurationError("state.new_question_confirm_frames must be at least 1")
        if self.state.ocr_retry_attempts <= 0:
            raise ConfigurationError("state.ocr_retry_attempts must be positive")
        if self.state.max_capture_failures <= 0:
            raise ConfigurationError("state.max_capture_failures must be positive")
        if not 0 <= self.state.white_pixel_threshold <= 255:
            raise ConfigurationError("state.white_pixel_threshold must be in [0, 255]")
        if not 0 <= self.state.min_white_ratio <= 1:
            raise ConfigurationError("state.min_white_ratio must be in [0, 1]")
        if not 0 <= self.state.title_white_pixel_threshold <= 255:
            raise ConfigurationError(
                "state.title_white_pixel_threshold must be in [0, 255]"
            )
        if not 0 <= self.state.title_min_white_ratio <= 1:
            raise ConfigurationError("state.title_min_white_ratio must be in [0, 1]")
        if self.state.ready_confirm_frames < 1:
            raise ConfigurationError("state.ready_confirm_frames must be at least 1")
        if not 0 <= self.state.ready_min_text_color_ratio <= 1:
            raise ConfigurationError(
                "state.ready_min_text_color_ratio must be in [0, 1]"
            )
        if not 0 <= self.state.ready_min_purple_ratio <= 1:
            raise ConfigurationError("state.ready_min_purple_ratio must be in [0, 1]")
        if self.post_challenge.confirm_frames < 2:
            raise ConfigurationError(
                "post_challenge.confirm_frames must be at least 2"
            )
        if self.post_challenge.max_capture_failures <= 0:
            raise ConfigurationError(
                "post_challenge.max_capture_failures must be positive"
            )
        ratio_values = {
            "post_challenge.success_green_ratio": (
                self.post_challenge.success_green_ratio
            ),
            "post_challenge.failure_blue_ratio": (
                self.post_challenge.failure_blue_ratio
            ),
            "post_challenge.continue_orange_ratio": (
                self.post_challenge.continue_orange_ratio
            ),
            "post_challenge.popup_banner_pink_ratio": (
                self.post_challenge.popup_banner_pink_ratio
            ),
            "post_challenge.popup_panel_pink_ratio": (
                self.post_challenge.popup_panel_pink_ratio
            ),
            "post_challenge.popup_panel_white_ratio": (
                self.post_challenge.popup_panel_white_ratio
            ),
            "post_challenge.popup_close_white_ratio": (
                self.post_challenge.popup_close_white_ratio
            ),
            "post_challenge.popup_close_orange_ratio": (
                self.post_challenge.popup_close_orange_ratio
            ),
        }
        for label, value in ratio_values.items():
            if not 0 <= value <= 1:
                raise ConfigurationError(f"{label} must be in [0, 1]")
        positive_timeouts = {
            "ollama.timeout_seconds": self.ollama.timeout_seconds,
            "openai_compatible.timeout_seconds": (
                self.openai_compatible.timeout_seconds
            ),
            "adb.timeout_seconds": self.adb.timeout_seconds,
            "state.poll_interval_seconds": self.state.poll_interval_seconds,
            "state.stable_timeout_seconds": self.state.stable_timeout_seconds,
            "state.change_timeout_seconds": self.state.change_timeout_seconds,
            "state.page_wait_timeout_seconds": self.state.page_wait_timeout_seconds,
            "state.transition_timeout_seconds": self.state.transition_timeout_seconds,
            "state.question_number_cache_seconds": self.state.question_number_cache_seconds,
            "state.ocr_retry_interval_seconds": self.state.ocr_retry_interval_seconds,
            "state.title_probe_interval_seconds": (
                self.state.title_probe_interval_seconds
            ),
            "state.ready_poll_interval_seconds": (
                self.state.ready_poll_interval_seconds
            ),
            "state.ready_fast_window_seconds": self.state.ready_fast_window_seconds,
            "post_challenge.poll_interval_seconds": (
                self.post_challenge.poll_interval_seconds
            ),
            "post_challenge.detection_timeout_seconds": (
                self.post_challenge.detection_timeout_seconds
            ),
            "post_challenge.ranking_popup_wait_seconds": (
                self.post_challenge.ranking_popup_wait_seconds
            ),
            "post_challenge.ranking_close_timeout_seconds": (
                self.post_challenge.ranking_close_timeout_seconds
            ),
            "post_challenge.ready_timeout_seconds": (
                self.post_challenge.ready_timeout_seconds
            ),
        }
        for label, value in positive_timeouts.items():
            if value <= 0:
                raise ConfigurationError(f"{label} must be positive")
        if self.ollama.num_predict <= 0:
            raise ConfigurationError("ollama.num_predict must be positive")
        supported_providers = {"ollama", "aliyun"}
        if not self.llm.provider_order:
            raise ConfigurationError("llm.provider_order must not be empty")
        if len(set(self.llm.provider_order)) != len(self.llm.provider_order):
            raise ConfigurationError("llm.provider_order must not contain duplicates")
        unknown_providers = set(self.llm.provider_order) - supported_providers
        if unknown_providers:
            raise ConfigurationError(
                f"llm.provider_order contains unsupported providers: "
                f"{sorted(unknown_providers)!r}"
            )
        if self.openai_compatible.enabled:
            if not self.openai_compatible.base_url.startswith(("http://", "https://")):
                raise ConfigurationError(
                    "openai_compatible.base_url must be an HTTP(S) URL"
                )
            if not self.openai_compatible.model.strip():
                raise ConfigurationError("openai_compatible.model must not be empty")
            if (
                not self.openai_compatible.api_key.strip()
                and not self.openai_compatible.api_key_env.strip()
            ):
                raise ConfigurationError(
                    "openai_compatible.api_key or api_key_env must not be empty"
                )
        if self.post_challenge.enabled:
            if self.regions.ready_indicator is None:
                raise ConfigurationError(
                    "enabled post_challenge requires regions.ready_indicator"
                )
            required_regions = {
                label: value for label, value in post_challenge_regions
            }
            missing_regions = [
                label for label, value in required_regions.items() if value is None
            ]
            if missing_regions:
                raise ConfigurationError(
                    "enabled post_challenge requires regions: "
                    f"{missing_regions!r}"
                )
            required_taps = {
                "post_challenge.ranking_close_tap": (
                    self.post_challenge.ranking_close_tap
                ),
                "post_challenge.continue_challenge_tap": (
                    self.post_challenge.continue_challenge_tap
                ),
            }
            invalid_taps = [
                label
                for label, point in required_taps.items()
                if point is None or point.x < 0 or point.y < 0
            ]
            if invalid_taps:
                raise ConfigurationError(
                    "enabled post_challenge requires non-negative phone tap "
                    f"coordinates: {invalid_taps!r}"
                )
        if require_live_coordinates:
            for index, point in enumerate(self.adb.tap_points):
                if point.x < 0 or point.y < 0:
                    raise ConfigurationError(
                        f"adb.tap_points[{index}] is a placeholder; fill phone coordinates"
                    )


def _ordered_options(options: list[Rect]) -> tuple[Rect, Rect, Rect, Rect]:
    """Order options by visual rows, then left-to-right within each row."""
    if len(options) != 4:
        raise ConfigurationError(f"exactly four option ROIs are required, got {len(options)}")
    remaining = sorted(options, key=lambda item: item.center[1])
    rows: list[list[Rect]] = []
    for rect in remaining:
        for row in rows:
            average_y = sum(item.center[1] for item in row) / len(row)
            tolerance = min([rect.height, *(item.height for item in row)]) * 0.5
            if abs(rect.center[1] - average_y) <= tolerance:
                row.append(rect)
                break
        else:
            rows.append([rect])
    ordered = [rect for row in rows for rect in sorted(row, key=lambda item: item.center[0])]
    return tuple(ordered)  # type: ignore[return-value]


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    try:
        with config_path.open("rb") as file:
            raw = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"cannot read configuration {config_path}: {exc}") from exc

    base_dir = config_path.parent
    capture = raw.get("capture", {})
    regions = raw.get("regions", {})
    ocr = raw.get("ocr", {})
    ollama = raw.get("ollama", {})
    openai_compatible = raw.get("openai_compatible", {})
    llm = raw.get("llm", {})
    adb = raw.get("adb", {})
    state = raw.get("state", {})
    debug = raw.get("debug", {})
    fallback = raw.get("fallback", {})
    post_challenge = raw.get("post_challenge", {})
    runtime = raw.get("runtime", {})

    option_rects = [_rect(item, f"regions.options[{i}]") for i, item in enumerate(
        regions.get("options", [])
    )]
    option_box_items = regions.get("option_boxes", regions.get("options", []))
    option_box_rects = [
        _rect(item, f"regions.option_boxes[{i}]")
        for i, item in enumerate(option_box_items)
    ]
    tap_points = [_point(item, f"adb.tap_points[{i}]") for i, item in enumerate(
        adb.get("tap_points", [])
    )]
    if len(tap_points) != 4:
        raise ConfigurationError(f"exactly four ADB tap points are required, got {len(tap_points)}")

    adb_path = Path(adb.get("executable", "tools/android/adb.exe"))
    output_dir = Path(debug.get("output_dir", "artifacts/failures"))
    log_file = Path(runtime.get("log_file", "artifacts/auto-answer.log"))
    configured_serial = str(adb.get("serial", "")).strip() or None
    detection_model = str(
        ocr.get("text_detection_model_name", "PP-OCRv6_medium_det")
    ).strip()
    recognition_model = str(
        ocr.get("text_recognition_model_name", "PP-OCRv6_medium_rec")
    ).strip()

    result = AppConfig(
        capture=CaptureConfig(
            screen_rect=_rect(capture, "capture", allow_negative_origin=True)
        ),
        regions=RegionConfig(
            question_number=_rect(
                regions.get("question_number", {}),
                "regions.question_number",
            ),
            question=_rect(regions.get("question", {}), "regions.question"),
            options=_ordered_options(option_rects),
            option_boxes=_ordered_options(option_box_rects),
            ready_indicator=(
                _rect(regions["ready_indicator"], "regions.ready_indicator")
                if "ready_indicator" in regions
                else None
            ),
        ),
        ocr=OCRConfig(
            language=str(ocr.get("language", "ch")),
            confidence_threshold=float(ocr.get("confidence_threshold", 0.60)),
            option_symbol_confidence_threshold=float(
                ocr.get("option_symbol_confidence_threshold", 0.45)
            ),
            use_gpu=bool(ocr.get("use_gpu", True)),
            enable_mkldnn=bool(ocr.get("enable_mkldnn", False)),
            cpu_threads=int(ocr.get("cpu_threads", 8)),
            text_detection_model_name=detection_model or None,
            text_recognition_model_name=recognition_model or None,
            isolated_fragment_max_chars=int(
                ocr.get("isolated_fragment_max_chars", 1)
            ),
            isolated_fragment_max_area_ratio=float(
                ocr.get("isolated_fragment_max_area_ratio", 0.02)
            ),
            isolated_fragment_max_height_ratio=float(
                ocr.get("isolated_fragment_max_height_ratio", 0.50)
            ),
            recover_thin_minus=bool(ocr.get("recover_thin_minus", True)),
            symbol_foreground_threshold=int(
                ocr.get("symbol_foreground_threshold", 180)
            ),
        ),
        ollama=OllamaConfig(
            base_url=str(ollama.get("base_url", "http://localhost:11434")).rstrip("/"),
            model=str(ollama.get("model", "qwen3.5:9b")),
            timeout_seconds=float(ollama.get("timeout_seconds", 20.0)),
            keep_alive=str(ollama.get("keep_alive", "30m")),
            num_predict=int(ollama.get("num_predict", 64)),
            warmup_on_start=bool(ollama.get("warmup_on_start", True)),
            retry_numeric_as_text=bool(ollama.get("retry_numeric_as_text", True)),
        ),
        openai_compatible=OpenAICompatibleConfig(
            enabled=bool(openai_compatible.get("enabled", False)),
            base_url=str(
                openai_compatible.get(
                    "base_url",
                    "https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/"
                    "compatible-mode/v1",
                )
            ).rstrip("/"),
            model=str(openai_compatible.get("model", "qwen3.7-flash")),
            api_key=str(openai_compatible.get("api_key", "")),
            api_key_env=str(
                openai_compatible.get("api_key_env", "DASHSCOPE_API_KEY")
            ),
            timeout_seconds=float(
                openai_compatible.get("timeout_seconds", 10.0)
            ),
            enable_thinking=bool(
                openai_compatible.get("enable_thinking", False)
            ),
            retry_numeric_as_text=bool(
                openai_compatible.get("retry_numeric_as_text", True)
            ),
        ),
        llm=LLMRoutingConfig(
            provider_order=tuple(
                str(provider)
                for provider in llm.get("provider_order", ["aliyun", "ollama"])
            )
        ),
        adb=ADBConfig(
            executable=(base_dir / adb_path).resolve() if not adb_path.is_absolute() else adb_path,
            tap_points=tuple(tap_points),  # type: ignore[arg-type]
            serial=configured_serial,
            timeout_seconds=float(adb.get("timeout_seconds", 5.0)),
            persistent_shell=bool(adb.get("persistent_shell", True)),
        ),
        state=StateConfig(
            stable_threshold=float(state.get("stable_threshold", 0.010)),
            change_threshold=float(state.get("change_threshold", 0.060)),
            poll_interval_seconds=float(state.get("poll_interval_seconds", 0.15)),
            required_stable_frames=int(state.get("required_stable_frames", 3)),
            stable_timeout_seconds=float(state.get("stable_timeout_seconds", 8.0)),
            change_timeout_seconds=float(state.get("change_timeout_seconds", 8.0)),
            page_confirm_frames=int(state.get("page_confirm_frames", 2)),
            new_question_confirm_frames=int(state.get("new_question_confirm_frames", 1)),
            page_wait_timeout_seconds=float(state.get("page_wait_timeout_seconds", 10.0)),
            transition_timeout_seconds=float(state.get("transition_timeout_seconds", 12.0)),
            question_number_cache_seconds=float(
                state.get("question_number_cache_seconds", 1.0)
            ),
            ocr_retry_attempts=int(state.get("ocr_retry_attempts", 3)),
            ocr_retry_interval_seconds=float(
                state.get("ocr_retry_interval_seconds", 0.25)
            ),
            white_pixel_threshold=int(state.get("white_pixel_threshold", 210)),
            min_white_ratio=float(state.get("min_white_ratio", 0.55)),
            max_capture_failures=int(state.get("max_capture_failures", 3)),
            overlap_ocr_with_stability=bool(
                state.get("overlap_ocr_with_stability", True)
            ),
            title_white_pixel_threshold=int(
                state.get("title_white_pixel_threshold", 210)
            ),
            title_min_white_ratio=float(
                state.get("title_min_white_ratio", 0.02)
            ),
            title_probe_interval_seconds=float(
                state.get("title_probe_interval_seconds", 0.75)
            ),
            ready_poll_interval_seconds=float(
                state.get("ready_poll_interval_seconds", 0.005)
            ),
            ready_fast_window_seconds=float(
                state.get("ready_fast_window_seconds", 6.0)
            ),
            ready_confirm_frames=int(state.get("ready_confirm_frames", 2)),
            ready_min_text_color_ratio=float(
                state.get("ready_min_text_color_ratio", 0.15)
            ),
            ready_min_purple_ratio=float(
                state.get("ready_min_purple_ratio", 0.60)
            ),
            infer_first_question_number_after_ready=bool(
                state.get("infer_first_question_number_after_ready", True)
            ),
            infer_sequential_question_number=bool(
                state.get("infer_sequential_question_number", True)
            ),
        ),
        post_challenge=PostChallengeConfig(
            enabled=bool(post_challenge.get("enabled", False)),
            success_banner=(
                _rect(
                    post_challenge["success_banner"],
                    "post_challenge.success_banner",
                )
                if "success_banner" in post_challenge
                else None
            ),
            continue_button=(
                _rect(
                    post_challenge["continue_button"],
                    "post_challenge.continue_button",
                )
                if "continue_button" in post_challenge
                else None
            ),
            ranking_panel=(
                _rect(
                    post_challenge["ranking_panel"],
                    "post_challenge.ranking_panel",
                )
                if "ranking_panel" in post_challenge
                else None
            ),
            ranking_close_button=(
                _rect(
                    post_challenge["ranking_close_button"],
                    "post_challenge.ranking_close_button",
                )
                if "ranking_close_button" in post_challenge
                else None
            ),
            ranking_close_tap=(
                _point(
                    post_challenge["ranking_close_tap"],
                    "post_challenge.ranking_close_tap",
                )
                if "ranking_close_tap" in post_challenge
                else None
            ),
            continue_challenge_tap=(
                _point(
                    post_challenge["continue_challenge_tap"],
                    "post_challenge.continue_challenge_tap",
                )
                if "continue_challenge_tap" in post_challenge
                else None
            ),
            poll_interval_seconds=float(
                post_challenge.get("poll_interval_seconds", 0.10)
            ),
            detection_timeout_seconds=float(
                post_challenge.get("detection_timeout_seconds", 3.0)
            ),
            ranking_popup_wait_seconds=float(
                post_challenge.get("ranking_popup_wait_seconds", 4.0)
            ),
            ranking_close_timeout_seconds=float(
                post_challenge.get("ranking_close_timeout_seconds", 10.0)
            ),
            ready_timeout_seconds=float(
                post_challenge.get("ready_timeout_seconds", 30.0)
            ),
            confirm_frames=int(post_challenge.get("confirm_frames", 3)),
            max_capture_failures=int(
                post_challenge.get("max_capture_failures", 3)
            ),
            success_green_ratio=float(
                post_challenge.get("success_green_ratio", 0.35)
            ),
            failure_blue_ratio=float(
                post_challenge.get("failure_blue_ratio", 0.55)
            ),
            continue_orange_ratio=float(
                post_challenge.get("continue_orange_ratio", 0.30)
            ),
            popup_banner_pink_ratio=float(
                post_challenge.get("popup_banner_pink_ratio", 0.35)
            ),
            popup_panel_pink_ratio=float(
                post_challenge.get("popup_panel_pink_ratio", 0.10)
            ),
            popup_panel_white_ratio=float(
                post_challenge.get("popup_panel_white_ratio", 0.30)
            ),
            popup_close_white_ratio=float(
                post_challenge.get("popup_close_white_ratio", 0.20)
            ),
            popup_close_orange_ratio=float(
                post_challenge.get("popup_close_orange_ratio", 0.03)
            ),
        ),
        debug=DebugConfig(
            enabled=bool(debug.get("enabled", True)),
            output_dir=(base_dir / output_dir).resolve()
            if not output_dir.is_absolute()
            else output_dir,
            save_each_question=bool(debug.get("save_each_question", False)),
        ),
        fallback=FallbackConfig(
            random_on_ocr_failure=bool(
                fallback.get("random_on_ocr_failure", False)
            ),
            random_on_llm_failure=bool(
                fallback.get("random_on_llm_failure", False)
            ),
        ),
        log_file=(base_dir / log_file).resolve() if not log_file.is_absolute() else log_file,
    )
    result.validate(require_live_coordinates=False)
    return result
