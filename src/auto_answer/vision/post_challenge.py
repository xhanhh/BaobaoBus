"""Cheap visual features for the challenge-success flow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..core.config import PostChallengeConfig
from .state import crop_for_state


@dataclass(frozen=True, slots=True)
class PostChallengeObservation:
    success_page_visible: bool
    failure_page_visible: bool
    ranking_popup_visible: bool
    banner_green_ratio: float
    banner_blue_ratio: float
    continue_orange_ratio: float
    banner_pink_ratio: float
    panel_pink_ratio: float
    panel_white_ratio: float
    close_white_ratio: float
    close_orange_ratio: float


@dataclass(frozen=True, slots=True)
class _ColorRatios:
    green: float
    blue: float
    orange: float
    pink: float
    white: float


def observe_post_challenge(
    frame: Image.Image,
    config: PostChallengeConfig,
) -> PostChallengeObservation:
    """Combine four independent ROIs; no single color is enough to trigger a tap."""
    if (
        config.success_banner is None
        or config.continue_button is None
        or config.ranking_panel is None
        or config.ranking_close_button is None
    ):
        return PostChallengeObservation(
            False, False, False, 0, 0, 0, 0, 0, 0, 0, 0
        )

    banner = _color_ratios(crop_for_state(frame, config.success_banner))
    continue_button = _color_ratios(
        crop_for_state(frame, config.continue_button)
    )
    panel = _color_ratios(crop_for_state(frame, config.ranking_panel))
    close_button = _color_ratios(
        crop_for_state(frame, config.ranking_close_button)
    )
    success_visible = (
        banner.green >= config.success_green_ratio
        and continue_button.orange >= config.continue_orange_ratio
    )
    failure_visible = (
        banner.blue >= config.failure_blue_ratio
        and continue_button.orange >= config.continue_orange_ratio
    )
    popup_visible = (
        banner.pink >= config.popup_banner_pink_ratio
        and panel.pink >= config.popup_panel_pink_ratio
        and panel.white >= config.popup_panel_white_ratio
        and close_button.white >= config.popup_close_white_ratio
        and close_button.orange >= config.popup_close_orange_ratio
    )
    return PostChallengeObservation(
        success_page_visible=success_visible,
        failure_page_visible=failure_visible,
        ranking_popup_visible=popup_visible,
        banner_green_ratio=banner.green,
        banner_blue_ratio=banner.blue,
        continue_orange_ratio=continue_button.orange,
        banner_pink_ratio=banner.pink,
        panel_pink_ratio=panel.pink,
        panel_white_ratio=panel.white,
        close_white_ratio=close_button.white,
        close_orange_ratio=close_button.orange,
    )


def _color_ratios(image: Image.Image) -> _ColorRatios:
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    red = pixels[:, :, 0]
    green = pixels[:, :, 1]
    blue = pixels[:, :, 2]
    maximum = np.max(pixels, axis=2)
    minimum = np.min(pixels, axis=2)

    green_mask = (
        (green >= 130)
        & (green >= red + 15)
        & (green >= blue + 10)
    )
    blue_mask = (
        (blue >= 130)
        & (blue >= red + 25)
        & (blue >= green + 10)
        & (red <= 180)
    )
    orange_mask = (
        (red >= 190)
        & (green >= 80)
        & (green <= 210)
        & (blue <= 100)
        & (red >= green + 25)
    )
    pink_mask = (
        (red >= 170)
        & (red >= green + 50)
        & (blue >= 60)
        & (blue <= 210)
    )
    white_mask = (maximum - minimum <= 65) & (maximum >= 200)
    return _ColorRatios(
        green=float(np.mean(green_mask)),
        blue=float(np.mean(blue_mask)),
        orange=float(np.mean(orange_mask)),
        pink=float(np.mean(pink_mask)),
        white=float(np.mean(white_mask)),
    )
