"""Safe ADB device selection and tap control."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ..core.config import ADBConfig
from ..core.errors import ADBError, ConfigurationError


class ADBController:
    def __init__(self, config: ADBConfig) -> None:
        self._config = config
        self._executable = Path(config.executable)
        self._serial: str | None = config.serial

    def check_device(self) -> str:
        if not self._executable.is_file():
            raise ConfigurationError(f"ADB executable does not exist: {self._executable}")
        output = self._run("devices")
        devices: list[str] = []
        unusable: list[str] = []
        for line in output.splitlines()[1:]:
            if not line.strip() or "\t" not in line:
                continue
            serial, status = line.split("\t", 1)
            if status.strip() == "device":
                devices.append(serial.strip())
            else:
                unusable.append(f"{serial.strip()} ({status.strip()})")
        if self._serial is not None:
            if self._serial not in devices:
                raise ADBError(
                    f"configured device {self._serial!r} is not ready; "
                    f"ready={devices}, other={unusable}"
                )
        elif len(devices) == 1:
            self._serial = devices[0]
        elif not devices:
            raise ADBError(f"no ready ADB device; other={unusable}")
        else:
            raise ADBError(f"multiple ADB devices found; configure adb.serial: {devices}")
        self._validate_tap_points_against_device()
        return self._serial

    def tap(self, answer_index: int) -> None:
        if answer_index not in range(4):
            raise ADBError(f"refusing invalid answer index {answer_index}")
        point = self._config.tap_points[answer_index]
        if point.x < 0 or point.y < 0:
            raise ConfigurationError(
                f"ADB tap point {answer_index} is still a placeholder: ({point.x}, {point.y})"
            )
        if self._serial is None:
            self.check_device()
        self._run("-s", self._serial, "shell", "input", "tap", str(point.x), str(point.y))

    def _validate_tap_points_against_device(self) -> None:
        output = self._run("-s", self._serial, "shell", "wm", "size")
        sizes = [
            (int(width), int(height))
            for width, height in re.findall(r"(?:Physical|Override) size:\s*(\d+)x(\d+)", output)
        ]
        if not sizes:
            raise ADBError(f"could not parse phone screen size from: {output.strip()!r}")
        width, height = sizes[-1]
        for index, point in enumerate(self._config.tap_points):
            if point.x < 0 or point.y < 0:
                raise ConfigurationError(
                    f"ADB tap point {index} is still a placeholder: ({point.x}, {point.y})"
                )
            portrait_fit = point.x < width and point.y < height
            landscape_fit = point.x < height and point.y < width
            if not portrait_fit and not landscape_fit:
                raise ConfigurationError(
                    f"ADB tap point {index}=({point.x}, {point.y}) is outside device "
                    f"size {width}x{height} in both orientations"
                )

    def _run(self, *arguments: str | None) -> str:
        command = [str(self._executable), *(arg for arg in arguments if arg is not None)]
        startup_info = None
        creation_flags = 0
        if hasattr(subprocess, "CREATE_NO_WINDOW"):
            creation_flags = subprocess.CREATE_NO_WINDOW
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_seconds,
                startupinfo=startup_info,
                creationflags=creation_flags,
            )
        except subprocess.TimeoutExpired as exc:
            raise ADBError(
                f"ADB command timed out after {self._config.timeout_seconds:.1f}s"
            ) from exc
        except OSError as exc:
            raise ADBError(f"cannot execute ADB: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise ADBError(f"ADB exited with {result.returncode}: {detail}")
        return result.stdout
