"""Recoverable central state machine for question-page automation."""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

from ..core.config import AppConfig
from ..core.errors import ADBError, OCRError, SolverError, UnsafeOCRResult
from ..core.models import (
    CroppedFrame,
    OCRBundle,
    OCRResult,
    Question,
    SchedulerState,
    SolveDecision,
)
from ..device.adb import ADBController
from ..solving.ollama import OllamaClient
from ..solving.rules import RuleEngine
from ..vision.capture import FrameSource, crop_regions
from ..vision.ocr import PaddleOCRReader
from ..vision.state import (
    PageStateDetector,
    ReadyQuestionPage,
    crop_for_state,
    extract_question_number,
)
from ..vision.text import assemble_question
from .debug import DebugRecorder


@dataclass(frozen=True, slots=True)
class _QuestionRead:
    page: ReadyQuestionPage
    crops: CroppedFrame
    ocr: OCRBundle
    title_ocr: OCRResult
    question: Question


class AnswerScheduler:
    def __init__(
        self,
        config: AppConfig,
        source: FrameSource,
        ocr: PaddleOCRReader,
        rules: RuleEngine,
        ollama: OllamaClient,
        state_detector: PageStateDetector,
        recorder: DebugRecorder,
        adb: ADBController | None,
        random_source: random.Random | None = None,
    ) -> None:
        self._config = config
        self._source = source
        self._ocr = ocr
        self._rules = rules
        self._ollama = ollama
        self._state_detector = state_detector
        self._recorder = recorder
        self._adb = adb
        self._random = random_source or random.SystemRandom()
        self.state = SchedulerState.STOPPED
        self._logger = logging.getLogger(__name__)

    def run(
        self,
        *,
        dry_run: bool = False,
        max_questions: int | None = None,
    ) -> SolveDecision | None:
        if not dry_run and self._adb is None:
            raise ValueError("live scheduler requires an ADB controller")
        completed = 0
        ready_page: ReadyQuestionPage | None = None
        blocked_after_tap: int | None = None
        deferred_after_failure: int | None = None
        last_read: _QuestionRead | None = None
        page_confirm_ms = 0.0

        try:
            while max_questions is None or completed < max_questions:
                if ready_page is None:
                    self._transition(SchedulerState.WAITING_FOR_QUESTION_PAGE)
                    excluded = (
                        blocked_after_tap
                        if blocked_after_tap is not None
                        else deferred_after_failure
                    )
                    page_wait_started = time.perf_counter()
                    try:
                        ready_page = self._state_detector.wait_for_question_page(
                            self._source,
                            excluded_number=excluded,
                        )
                    except OCRError as exc:
                        self._logger.warning(
                            "question number OCR temporarily failed; continuing to wait: %s",
                            exc,
                        )
                        time.sleep(self._config.state.ocr_retry_interval_seconds)
                        continue
                    page_confirm_ms = (
                        time.perf_counter() - page_wait_started
                    ) * 1000
                    if ready_page is None:
                        self._logger.info(
                            "question page not confirmed within %.1fs; continuing to wait",
                            self._config.state.page_wait_timeout_seconds,
                        )
                        # Keep a failed question excluded until the user advances it or a
                        # different question appears. This is the manual-handling mode.
                        continue
                    self._logger.info(
                        "question page %d confirmed in %.0fms",
                        ready_page.question_number,
                        page_confirm_ms,
                    )
                    if ready_page.ready_to_answer_ms is not None:
                        self._logger.info(
                            "FIRST_PAGE_TIMING ready_to_answer_ms=%.0f "
                            "answer_layout_to_confirm_ms=%.0f number_source=%s",
                            ready_page.ready_to_answer_ms,
                            ready_page.answer_to_confirm_ms or 0.0,
                            (
                                "ready-inferred"
                                if ready_page.question_number_inferred
                                else "title-ocr"
                            ),
                        )

                if (
                    blocked_after_tap is not None
                    and ready_page.question_number != blocked_after_tap
                ):
                    blocked_after_tap = None
                deferred_after_failure = None

                recognition_started = time.perf_counter()
                last_read = self._read_question_with_retries(ready_page)
                if last_read is None:
                    if not self._config.fallback.random_on_ocr_failure:
                        deferred_after_failure = ready_page.question_number
                        ready_page = None
                        continue
                    try:
                        confirmed_page = self._state_detector.wait_for_question_page(
                            self._source,
                            expected_number=ready_page.question_number,
                            timeout_seconds=self._config.state.stable_timeout_seconds,
                        )
                    except OCRError as exc:
                        self._logger.warning(
                            "OCR random fallback cancelled because the question page "
                            "could not be reconfirmed: %s",
                            exc,
                        )
                        confirmed_page = None
                    if confirmed_page is None:
                        deferred_after_failure = ready_page.question_number
                        ready_page = None
                        continue
                    ready_page = confirmed_page
                    decision = self._random_decision("OCR failed after all retries")
                    question_number = ready_page.question_number
                    baseline_frame = ready_page.frame
                    ocr_ms = (time.perf_counter() - recognition_started) * 1000
                    solve_ms = 0.0
                    self._logger.warning(
                        "OCR failed for question %d; configured random fallback selected %d",
                        question_number,
                        decision.answer_index,
                    )
                else:
                    ocr_ms = (time.perf_counter() - recognition_started) * 1000
                    question = last_read.question
                    question_number = last_read.page.question_number
                    baseline_frame = last_read.crops.full_frame
                    self._logger.info(
                        "question_number=%d question=%s options=%s",
                        question_number,
                        question.text,
                        question.options,
                    )
                    if self._config.debug.save_each_question:
                        self._recorder.save(
                            "question",
                            crops=last_read.crops,
                            ocr=last_read.ocr,
                            question=question,
                        )

                    solve_started = time.perf_counter()
                    try:
                        self._transition(SchedulerState.SOLVING_BY_RULE)
                        decision = self._rules.solve(question)
                        if decision is None:
                            self._transition(SchedulerState.SOLVING_BY_LLM)
                            decision = self._ollama.solve(question)
                        self._validate_decision(decision)
                    except SolverError as exc:
                        self._record_recoverable_error("solver-failed", last_read, exc)
                        if not self._config.fallback.random_on_llm_failure:
                            self._logger.error(
                                "solver failed for question %d; random fallback is "
                                "disabled, waiting for manual handling: %s",
                                question_number,
                                exc,
                            )
                            deferred_after_failure = question_number
                            ready_page = None
                            continue
                        decision = self._random_decision(
                            f"rule/LLM failed after retries: {exc}"
                        )
                        self._logger.warning(
                            "solver failed for question %d; configured random fallback "
                            "selected %d: %s",
                            question_number,
                            decision.answer_index,
                            exc,
                        )
                    solve_ms = (time.perf_counter() - solve_started) * 1000
                recognize_to_decision_ms = (
                    time.perf_counter() - recognition_started
                ) * 1000

                self._logger.info(
                    "answer=%d source=%s reason=%s",
                    decision.answer_index,
                    decision.source,
                    decision.reason,
                )
                self._logger.info(
                    "TIMING question=%d page_confirm_ms=%.0f ocr_ms=%.0f "
                    "solve_ms=%.0f recognize_to_decision_ms=%.0f",
                    question_number,
                    page_confirm_ms,
                    ocr_ms,
                    solve_ms,
                    recognize_to_decision_ms,
                )
                if dry_run:
                    self._transition(SchedulerState.STOPPED)
                    return decision

                self._transition(SchedulerState.TAPPING)
                assert self._adb is not None
                tap_started = time.perf_counter()
                try:
                    self._adb.tap(decision.answer_index)
                except ADBError as exc:
                    if last_read is not None:
                        self._record_recoverable_error("adb-tap-failed", last_read, exc)
                    else:
                        self._recorder.save("adb-tap-failed", error=exc)
                    self._logger.error(
                        "ADB tap failed once; no immediate retry will occur: %s",
                        exc,
                    )
                    deferred_after_failure = question_number
                    ready_page = None
                    continue
                tap_ms = (time.perf_counter() - tap_started) * 1000
                self._logger.info(
                    "TIMING question=%d tap_ms=%.0f recognize_to_tap_ms=%.0f",
                    question_number,
                    tap_ms,
                    (time.perf_counter() - recognition_started) * 1000,
                )

                old_number = question_number
                blocked_after_tap = old_number
                completed += 1
                if max_questions is not None and completed >= max_questions:
                    self._transition(SchedulerState.STOPPED)
                    return None

                self._transition(SchedulerState.WAITING_FOR_TRANSITION)
                self._transition(SchedulerState.WAITING_FOR_NEXT_QUESTION)
                transition_started = time.perf_counter()
                try:
                    ready_page = self._state_detector.wait_for_next_question(
                        self._source,
                        old_question_number=old_number,
                        baseline_frame=baseline_frame,
                    )
                except OCRError as exc:
                    self._logger.warning(
                        "question number temporarily unreadable during transition: %s",
                        exc,
                    )
                    ready_page = None
                transition_ms = (time.perf_counter() - transition_started) * 1000
                if ready_page is None:
                    self._logger.warning(
                        "next question was not confirmed within %.1fs; "
                        "returning to question-page detection",
                        self._config.state.transition_timeout_seconds,
                    )
                else:
                    page_confirm_ms = transition_ms
                    self._logger.info(
                        "question transition confirmed: %d -> %d in %.0fms "
                        "number_source=%s",
                        old_number,
                        ready_page.question_number,
                        transition_ms,
                        (
                            "sequence-inferred"
                            if ready_page.question_number_inferred
                            else "title-ocr"
                        ),
                    )
                last_read = None

            self._transition(SchedulerState.STOPPED)
            return None
        except KeyboardInterrupt:
            self._logger.info("Ctrl+C received; stopping safely before any further tap")
            self._transition(SchedulerState.STOPPED)
            return None
        except Exception as exc:
            self._transition(SchedulerState.ERROR)
            directory = self._recorder.save(
                "critical-error",
                crops=last_read.crops if last_read else None,
                ocr=last_read.ocr if last_read else None,
                question=last_read.question if last_read else None,
                error=exc,
            )
            self._logger.critical(
                "unrecoverable scheduler error: %s; artifacts=%s",
                exc,
                directory,
            )
            raise
        finally:
            self._source.close()
            self._ollama.close()
            if self._adb is not None:
                self._adb.close()
            if self.state is not SchedulerState.ERROR:
                self.state = SchedulerState.STOPPED

    def _read_question_with_retries(
        self,
        initial_page: ReadyQuestionPage,
    ) -> _QuestionRead | None:
        page = initial_page
        last_crops: CroppedFrame | None = None
        last_ocr: OCRBundle | None = None
        last_question: Question | None = None
        last_error: Exception | None = None

        for attempt in range(1, self._config.state.ocr_retry_attempts + 1):
            self._transition(SchedulerState.CAPTURING)
            last_crops = crop_regions(
                page.frame,
                self._config.regions.question,
                self._config.regions.options,
            )
            title_image = crop_for_state(
                page.frame,
                self._config.regions.question_number,
            )
            try:
                self._transition(SchedulerState.OCR)
                last_ocr, title_ocr = self._ocr.recognize_with_title(
                    last_crops,
                    title_image,
                )
                recognized_number = extract_question_number(title_ocr)
                if recognized_number is None:
                    raise UnsafeOCRResult("question number is temporarily unreadable")
                if recognized_number != page.question_number:
                    raise UnsafeOCRResult(
                        f"question number changed during OCR: "
                        f"{page.question_number} -> {recognized_number}"
                    )
                last_question = assemble_question(last_ocr)

                self._transition(SchedulerState.VALIDATING_OCR)
                latest_frame = self._state_detector.capture_with_retry(self._source)
                if not self._state_detector.content_remained_stable(
                    page.frame,
                    latest_frame,
                ):
                    raise UnsafeOCRResult(
                        "question content changed or option boxes disappeared during OCR"
                    )
                return _QuestionRead(
                    page=page,
                    crops=last_crops,
                    ocr=last_ocr,
                    title_ocr=title_ocr,
                    question=last_question,
                )
            except (UnsafeOCRResult, OCRError) as exc:
                last_error = exc
                self._logger.warning(
                    "OCR result invalid for question %d (%d/%d): %s",
                    page.question_number,
                    attempt,
                    self._config.state.ocr_retry_attempts,
                    exc,
                )
                if attempt >= self._config.state.ocr_retry_attempts:
                    break
                time.sleep(self._config.state.ocr_retry_interval_seconds)
                try:
                    next_page = self._state_detector.wait_for_question_page(
                        self._source,
                        expected_number=page.question_number,
                        timeout_seconds=self._config.state.stable_timeout_seconds,
                    )
                except OCRError as title_error:
                    last_error = title_error
                    continue
                if next_page is None:
                    last_error = UnsafeOCRResult(
                        "same stable question page was not found before OCR retry"
                    )
                    break
                page = next_page

        assert last_error is not None
        directory = self._recorder.save(
            "ocr-failed",
            crops=last_crops,
            ocr=last_ocr,
            question=last_question,
            error=last_error,
        )
        self._logger.error(
            "question %d failed OCR repeatedly; saved artifacts=%s; "
            "returning to question-page detection",
            initial_page.question_number,
            directory,
        )
        return None

    def _record_recoverable_error(
        self,
        label: str,
        read: _QuestionRead,
        error: Exception,
    ) -> None:
        self._recorder.save(
            label,
            crops=read.crops,
            ocr=read.ocr,
            question=read.question,
            error=error,
        )

    def _transition(self, state: SchedulerState) -> None:
        self._logger.debug("state %s -> %s", self.state.name, state.name)
        self.state = state

    def _random_decision(self, reason: str) -> SolveDecision:
        return SolveDecision(
            answer_index=self._random.randrange(4),
            source="random-fallback",
            reason=reason,
        )

    @staticmethod
    def _validate_decision(decision: SolveDecision) -> None:
        if decision.answer_index not in range(4):
            raise SolverError(f"solver returned invalid answer index {decision.answer_index}")
