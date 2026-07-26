from pathlib import Path

from auto_answer.core.config import ADBConfig
from auto_answer.core.errors import ADBError
from auto_answer.core.models import Point
from auto_answer.device.adb import ADBController


def adb_config(*, persistent_shell: bool) -> ADBConfig:
    return ADBConfig(
        executable=Path("adb.exe"),
        tap_points=(
            Point(10, 20),
            Point(30, 40),
            Point(50, 60),
            Point(70, 80),
        ),
        serial="device",
        persistent_shell=persistent_shell,
    )


def test_nonpersistent_tap_uses_one_shot_command(monkeypatch: object) -> None:
    controller = ADBController(adb_config(persistent_shell=False))
    commands: list[tuple[str | None, ...]] = []
    monkeypatch.setattr(  # type: ignore[attr-defined]
        controller,
        "_run",
        lambda *arguments: commands.append(arguments) or "",
    )
    controller.tap(2)
    assert commands == [
        ("-s", "device", "shell", "input", "tap", "50", "60")
    ]


def test_persistent_failure_retries_once_with_one_shot_command(
    monkeypatch: object,
) -> None:
    controller = ADBController(adb_config(persistent_shell=True))
    commands: list[tuple[str | None, ...]] = []

    def fail_persistent(_x: int, _y: int) -> None:
        raise ADBError("pipe closed")

    monkeypatch.setattr(  # type: ignore[attr-defined]
        controller,
        "_tap_persistent",
        fail_persistent,
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        controller,
        "_stop_persistent_shell",
        lambda: None,
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        controller,
        "_warm_persistent_shell",
        lambda: None,
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        controller,
        "_run",
        lambda *arguments: commands.append(arguments) or "",
    )
    controller.tap(1)
    assert commands == [
        ("-s", "device", "shell", "input", "tap", "30", "40")
    ]
