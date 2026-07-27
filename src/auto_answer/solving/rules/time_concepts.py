"""Rules for elementary clock-face and elapsed-hour questions."""

from __future__ import annotations

import re
from decimal import Decimal

from ...core.models import Question, SolveDecision
from .common import NUMBER, match_unique, parse_number

_CLOCK_HOUR = re.compile(rf"(?:钟面上)?是(?P<hour>{NUMBER})时")
_CLOCK_HANDS = re.compile(
    rf"时针指(?:着)?(?P<hour>{NUMBER})[,，]"
    rf"分针指(?:着)?(?P<minute>{NUMBER})"
)
_ELAPSED_HOURS = re.compile(
    rf"(?P<start>{NUMBER})时.*?(?P<end>{NUMBER})时.*?"
    rf"(?:后|经过)(?:\(\)|多少|几)小时"
)
_CLOCK_BETWEEN_HOURS = re.compile(
    rf"时针(?:指)?在(?P<hour>{NUMBER})和(?P<next_hour>{NUMBER})之间[,，]"
    rf"分针指(?:着)?(?P<minute_pointer>{NUMBER})"
)
_DIGITAL_TIME = re.compile(
    r"(?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?"
)
_CLOCK_OVERLAP_AT_HOUR = re.compile(
    r"(?:\(\)|多少|几)时整.*?分针和时针重合(?:在一起)?"
)
_MINUTE_HAND_LARGE_GRIDS = re.compile(
    rf"分针走(?P<grids>{NUMBER})大格.*?秒针走(?:\(\)|多少|几)圈"
)
_MOVIE_END_TIME = re.compile(
    rf"电影(?P<start_hour>{NUMBER})时"
    rf"(?:(?P<start_minute>{NUMBER})分)?开始.*?"
    rf"时长(?:(?P<hours>{NUMBER})小时)?"
    rf"(?:(?P<minutes>{NUMBER})分(?:钟)?)?.*?结束时间"
)
_HOUR_MINUTE_OPTION = re.compile(
    rf"(?P<hour>{NUMBER})时(?:(?P<minute>{NUMBER})分)?"
)
_MINUTE_CIRCLE_HOUR_GRIDS = re.compile(
    r"分针走一圈[,，]时针走(?:\(\)|多少|几)(?:个)?大格"
)
_HOUR_HAND_DISTANCE = re.compile(
    rf"时针从[“\"']?(?P<start>{NUMBER})[”\"']?走到"
    rf"[“\"']?(?P<end>{NUMBER})[”\"']?[,，]走了(?:\(\)|多少|几)小时"
)
_HOURS_AGO = re.compile(
    rf"现在是(?:上午|下午|晚上)?(?P<hour>{NUMBER})时[,，]"
    rf"(?P<elapsed>{NUMBER})小时前是(?:\(\)|多少|几)时"
)
_QUARTER_HOUR_LATER = re.compile(
    rf"(?:现在是|[\u4e00-\u9fff]+)?"
    rf"(?P<hour>{NUMBER})(?::|时)(?P<minute>{NUMBER})(?:分)?"
    rf".*?(?:过一刻钟|一刻钟后)"
)
_QUARTER_HOUR_IS = re.compile(r"^一刻钟是(?:\(\)|多少|几)")
_MINUTE_HAND_ROTATION = re.compile(
    r"从(?P<start_hour>\d{1,2}):(?P<start_minute>\d{2})走到"
    r"(?P<end_hour>\d{1,2}):(?P<end_minute>\d{2})[,，]分针转动了"
)


def solve_time(question: Question) -> SolveDecision | None:
    values = tuple(parse_number(option) for option in question.options)

    rotation = _MINUTE_HAND_ROTATION.search(question.text)
    if rotation:
        start = int(rotation.group("start_hour")) * 60 + int(
            rotation.group("start_minute")
        )
        end = int(rotation.group("end_hour")) * 60 + int(rotation.group("end_minute"))
        elapsed = (end - start) % (12 * 60)
        target = Decimal(elapsed * 6)
        degree_values = tuple(
            parse_number(option.rstrip("°度")) for option in question.options
        )
        return match_unique(target, degree_values, "minute-hand rotation")

    if _QUARTER_HOUR_IS.search(question.text):
        matches = [
            index
            for index, option in enumerate(question.options)
            if option in {"15分", "15分钟"}
        ]
        if len(matches) == 1:
            return SolveDecision(matches[0], "rule", "quarter hour: 15 minutes")
        return None

    quarter_later = _QUARTER_HOUR_LATER.search(question.text)
    if quarter_later:
        hour = int(Decimal(quarter_later.group("hour")))
        minute = int(Decimal(quarter_later.group("minute")))
        if not 0 <= hour < 24 or not 0 <= minute < 60:
            return None
        total_minutes = (hour * 60 + minute + 15) % (24 * 60)
        matches = [
            index
            for index, option in enumerate(question.options)
            if _option_minutes(option) == total_minutes
        ]
        if len(matches) == 1:
            return SolveDecision(
                matches[0],
                "rule",
                f"quarter hour later: {total_minutes // 60:02d}:{total_minutes % 60:02d}",
            )
        return None

    if _MINUTE_CIRCLE_HOUR_GRIDS.search(question.text):
        return match_unique(Decimal(1), values, "minute circle to hour-hand grid")

    hand_distance = _HOUR_HAND_DISTANCE.search(question.text)
    if hand_distance:
        start = Decimal(hand_distance.group("start"))
        end = Decimal(hand_distance.group("end"))
        distance = (end - start) % 12
        return match_unique(distance, values, "hour-hand distance")

    hours_ago = _HOURS_AGO.search(question.text)
    if hours_ago:
        hour = Decimal(hours_ago.group("hour"))
        elapsed = Decimal(hours_ago.group("elapsed"))
        target = (hour - elapsed) % 12
        if target == 0:
            target = Decimal(12)
        return match_unique(target, values, "hours ago")

    if _CLOCK_OVERLAP_AT_HOUR.search(question.text):
        return match_unique(Decimal(12), values, "clock hands overlap at whole hour")

    large_grids = _MINUTE_HAND_LARGE_GRIDS.search(question.text)
    if large_grids:
        grids = Decimal(large_grids.group("grids"))
        if grids < 0:
            return None
        return match_unique(grids * 5, values, "second-hand rotations")

    movie = _MOVIE_END_TIME.search(question.text)
    if movie and (movie.group("hours") or movie.group("minutes")):
        start_hour = Decimal(movie.group("start_hour"))
        start_minute = Decimal(movie.group("start_minute") or "0")
        hours = Decimal(movie.group("hours") or "0")
        minutes = Decimal(movie.group("minutes") or "0")
        if any(
            value != value.to_integral_value() or value < 0
            for value in (start_hour, start_minute, hours, minutes)
        ):
            return None
        total_minutes = (
            int(start_hour) * 60
            + int(start_minute)
            + int(hours) * 60
            + int(minutes)
        ) % (24 * 60)
        matches = [
            index
            for index, option in enumerate(question.options)
            if _option_minutes(option) == total_minutes
        ]
        if len(matches) == 1:
            return SolveDecision(
                matches[0],
                "rule",
                f"movie end time: {total_minutes // 60:02d}:{total_minutes % 60:02d}",
            )
        return None

    between = _CLOCK_BETWEEN_HOURS.search(question.text)
    if between:
        hour = int(Decimal(between.group("hour")))
        next_hour = int(Decimal(between.group("next_hour")))
        minute_pointer = int(Decimal(between.group("minute_pointer")))
        if next_hour % 12 != (hour + 1) % 12 or not 1 <= minute_pointer <= 12:
            return None
        minute = 0 if minute_pointer == 12 else minute_pointer * 5
        matches: list[int] = []
        for index, option in enumerate(question.options):
            digital = _DIGITAL_TIME.fullmatch(option)
            if (
                digital is not None
                and int(digital.group("hour")) == hour
                and int(digital.group("minute")) == minute
                and int(digital.group("second") or "0") == 0
            ):
                matches.append(index)
        if len(matches) == 1:
            return SolveDecision(
                matches[0],
                "rule",
                f"clock hands: {hour:02d}:{minute:02d}",
            )
        return None

    clock = _CLOCK_HOUR.search(question.text)
    if clock:
        target_hour = Decimal(clock.group("hour"))
        matches: list[int] = []
        for index, option in enumerate(question.options):
            hands = _CLOCK_HANDS.fullmatch(option)
            if (
                hands is not None
                and Decimal(hands.group("hour")) == target_hour
                and Decimal(hands.group("minute")) == 12
            ):
                matches.append(index)
        if len(matches) == 1:
            return SolveDecision(
                matches[0],
                "rule",
                f"clock face at {target_hour}:00",
            )
        return None

    elapsed = _ELAPSED_HOURS.search(question.text)
    if elapsed:
        start = Decimal(elapsed.group("start"))
        end = Decimal(elapsed.group("end"))
        if end < start:
            return None
        values = tuple(parse_number(option) for option in question.options)
        return match_unique(end - start, values, "elapsed hours")
    return None


def _option_minutes(option: str) -> int | None:
    digital = _DIGITAL_TIME.fullmatch(option)
    if digital is not None:
        if int(digital.group("second") or "0") != 0:
            return None
        hour = int(digital.group("hour"))
        minute = int(digital.group("minute"))
    else:
        hour_minute = _HOUR_MINUTE_OPTION.fullmatch(option)
        if hour_minute is None:
            return None
        hour = int(Decimal(hour_minute.group("hour")))
        minute = int(Decimal(hour_minute.group("minute") or "0"))
    if not 0 <= hour < 24 or not 0 <= minute < 60:
        return None
    return hour * 60 + minute
