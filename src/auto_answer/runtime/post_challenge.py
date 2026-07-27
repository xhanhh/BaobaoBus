"""Conservative automation for challenge result pages and ranking popups."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from enum import Enum, auto

from PIL import Image

from ..core.config import PostChallengeConfig
from ..core.errors import CaptureError
from ..device.adb import ADBController
from ..vision.capture import FrameSource
from ..vision.post_challenge import (
    PostChallengeObservation,
    observe_post_challenge,
)


class PostChallengeOutcome(Enum):
    NOT_FOUND = auto()
    CONTINUE_TAPPED = auto()
    READY_CONFIRMED = auto()


class PostChallengeController:
    """Handle success/failure pages and safely start the next challenge."""

    def __init__(
        self,
        config: PostChallengeConfig,
        source: FrameSource,
        adb: ADBController,
        ready_visible: Callable[[Image.Image], bool],
    ) -> None:
        self._config = config
        self._source = source
        self._adb = adb
        self._ready_visible = ready_visible
        self._logger = logging.getLogger(__name__)
        self.last_result_won: bool | None = None

    def handle_if_present(self) -> PostChallengeOutcome:
        self.last_result_won = None
        initial = self._wait_for_result_or_popup(
            self._config.detection_timeout_seconds
        )
        if initial is None:
            return PostChallengeOutcome.NOT_FOUND

        if initial.failure_page_visible:
            self.last_result_won = False
            self._logger.info(
                "challenge-failure flow detected; no ranking popup expected"
            )
            failure = self._wait_for_failure_without_popup(
                self._config.ranking_close_timeout_seconds
            )
            if failure is None:
                self._logger.warning(
                    "challenge-failure page was not stable; "
                    "continue button will not be tapped"
                )
                return PostChallengeOutcome.NOT_FOUND
            return self._tap_continue_and_wait_for_ready()

        self.last_result_won = True
        popup_visible = initial.ranking_popup_visible
        self._logger.info(
            "challenge-success flow detected: ranking_popup=%s",
            popup_visible,
        )
        if not popup_visible:
            popup_visible = self._wait_for_popup(
                self._config.ranking_popup_wait_seconds
            )

        if popup_visible:
            close_point = self._config.ranking_close_tap
            assert close_point is not None
            self._adb.tap_point(close_point, purpose="ranking popup close")
            self._logger.info("ranking popup close tapped")
            success = self._wait_for_success_without_popup(
                self._config.ranking_close_timeout_seconds
            )
            if success is None:
                self._logger.warning(
                    "ranking popup did not close to a confirmed success page; "
                    "continue button will not be tapped"
                )
                return PostChallengeOutcome.NOT_FOUND
            self._logger.info("ranking popup closed; challenge-success page restored")
        else:
            success = self._wait_for_success_without_popup(
                self._config.ranking_close_timeout_seconds
            )
            if success is None:
                self._logger.warning(
                    "challenge-success page was not stable after popup wait; "
                    "continue button will not be tapped"
                )
                return PostChallengeOutcome.NOT_FOUND

        return self._tap_continue_and_wait_for_ready()

    def _tap_continue_and_wait_for_ready(self) -> PostChallengeOutcome:
        continue_point = self._config.continue_challenge_tap
        assert continue_point is not None
        self._adb.tap_point(continue_point, purpose="continue challenge")
        self._logger.info("continue challenge tapped; waiting for Ready")

        if self._wait_for_ready(self._config.ready_timeout_seconds):
            self._logger.info("Ready page confirmed after continuing challenge")
            return PostChallengeOutcome.READY_CONFIRMED
        self._logger.warning(
            "Ready page was not confirmed within %.1fs after continue tap; "
            "returning to normal page detection without tapping again",
            self._config.ready_timeout_seconds,
        )
        return PostChallengeOutcome.CONTINUE_TAPPED

    def _wait_for_result_or_popup(
        self,
        timeout_seconds: float,
    ) -> PostChallengeObservation | None:
        success_count = 0
        failure_count = 0
        popup_count = 0
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            observation = self._capture_observation()
            success_count = (
                success_count + 1
                if observation.success_page_visible
                and not observation.ranking_popup_visible
                else 0
            )
            failure_count = (
                failure_count + 1
                if observation.failure_page_visible
                and not observation.ranking_popup_visible
                else 0
            )
            popup_count = (
                popup_count + 1 if observation.ranking_popup_visible else 0
            )
            if (
                success_count >= self._config.confirm_frames
                or failure_count >= self._config.confirm_frames
                or popup_count >= self._config.confirm_frames
            ):
                return observation
            time.sleep(self._config.poll_interval_seconds)
        return None

    def _wait_for_failure_without_popup(
        self,
        timeout_seconds: float,
    ) -> PostChallengeObservation | None:
        count = 0
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            observation = self._capture_observation()
            visible = (
                observation.failure_page_visible
                and not observation.ranking_popup_visible
            )
            count = count + 1 if visible else 0
            if count >= self._config.confirm_frames:
                return observation
            time.sleep(self._config.poll_interval_seconds)
        return None

    def _wait_for_popup(self, timeout_seconds: float) -> bool:
        count = 0
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            observation = self._capture_observation()
            count = count + 1 if observation.ranking_popup_visible else 0
            if count >= self._config.confirm_frames:
                return True
            time.sleep(self._config.poll_interval_seconds)
        return False

    def _wait_for_success_without_popup(
        self,
        timeout_seconds: float,
    ) -> PostChallengeObservation | None:
        count = 0
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            observation = self._capture_observation()
            visible = (
                observation.success_page_visible
                and not observation.ranking_popup_visible
            )
            count = count + 1 if visible else 0
            if count >= self._config.confirm_frames:
                return observation
            time.sleep(self._config.poll_interval_seconds)
        return None

    def _wait_for_ready(self, timeout_seconds: float) -> bool:
        count = 0
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            frame = self._capture()
            count = count + 1 if self._ready_visible(frame) else 0
            if count >= self._config.confirm_frames:
                return True
            time.sleep(self._config.poll_interval_seconds)
        return False

    def _capture_observation(self) -> PostChallengeObservation:
        return observe_post_challenge(self._capture(), self._config)

    def _capture(self) -> Image.Image:
        last_error: CaptureError | None = None
        for _ in range(self._config.max_capture_failures):
            try:
                return self._source.capture()
            except CaptureError as exc:
                last_error = exc
                self._logger.warning(
                    "temporary capture failure in post-challenge flow: %s",
                    exc,
                )
                time.sleep(self._config.poll_interval_seconds)
        assert last_error is not None
        raise last_error
