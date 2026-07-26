"""Safe ADB device selection and tap control."""

from __future__ import annotations

import itertools
import logging
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

from ..core.config import ADBConfig
from ..core.errors import ADBError, ConfigurationError


class ADBController:
    def __init__(self, config: ADBConfig) -> None:
        self._config = config
        self._executable = Path(config.executable)
        self._serial: str | None = config.serial
        self._logger = logging.getLogger(__name__)
        self._shell_process: subprocess.Popen[str] | None = None
        self._shell_output: queue.Queue[str | None] = queue.Queue()
        self._shell_reader: threading.Thread | None = None
        self._shell_lock = threading.Lock()
        self._command_ids = itertools.count(1)

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
        if self._config.persistent_shell:
            self._warm_persistent_shell()
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
        if not self._config.persistent_shell:
            self._run(
                "-s",
                self._serial,
                "shell",
                "input",
                "tap",
                str(point.x),
                str(point.y),
            )
            return

        try:
            self._tap_persistent(point.x, point.y)
        except ADBError as exc:
            self._logger.warning(
                "persistent ADB shell failed; retrying this tap with one-shot ADB: %s",
                exc,
            )
            self._stop_persistent_shell()
            self._run(
                "-s",
                self._serial,
                "shell",
                "input",
                "tap",
                str(point.x),
                str(point.y),
            )
            self._warm_persistent_shell()

    def close(self) -> None:
        with self._shell_lock:
            self._stop_persistent_shell()

    def _tap_persistent(self, x: int, y: int) -> None:
        self._run_persistent_command(f"input tap {x} {y}")

    def _run_persistent_command(self, command: str) -> None:
        with self._shell_lock:
            self._ensure_persistent_shell()
            process = self._shell_process
            if process is None or process.stdin is None:
                raise ADBError("persistent ADB shell has no stdin")

            marker = f"__AUTO_ANSWER_DONE_{next(self._command_ids)}__"
            try:
                process.stdin.write(f"{command}; echo {marker}:$?\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise ADBError(f"cannot write to persistent ADB shell: {exc}") from exc

            deadline = time.monotonic() + self._config.timeout_seconds
            diagnostics: list[str] = []
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ADBError(
                        "persistent ADB tap timed out; output="
                        f"{diagnostics[-3:]!r}"
                    )
                try:
                    line = self._shell_output.get(timeout=remaining)
                except queue.Empty as exc:
                    raise ADBError(
                        "persistent ADB tap timed out; output="
                        f"{diagnostics[-3:]!r}"
                    ) from exc
                if line is None:
                    raise ADBError(
                        "persistent ADB shell exited before acknowledging tap"
                    )
                cleaned = line.strip()
                if cleaned.startswith(f"{marker}:"):
                    status = cleaned.removeprefix(f"{marker}:").strip()
                    if status != "0":
                        raise ADBError(
                            f"persistent ADB tap exited with {status!r}; "
                            f"output={diagnostics[-3:]!r}"
                        )
                    return
                if cleaned:
                    diagnostics.append(cleaned)

    def _warm_persistent_shell(self) -> None:
        try:
            self._run_persistent_command(":")
        except ADBError as exc:
            self._logger.warning(
                "could not prestart persistent ADB shell; one-shot fallback remains available: %s",
                exc,
            )

    def _ensure_persistent_shell(self) -> None:
        if self._shell_process is not None and self._shell_process.poll() is None:
            return
        self._stop_persistent_shell()
        command = [str(self._executable), "-s", str(self._serial), "shell"]
        creation_flags = (
            subprocess.CREATE_NO_WINDOW
            if hasattr(subprocess, "CREATE_NO_WINDOW")
            else 0
        )
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
            )
        except OSError as exc:
            raise ADBError(f"cannot start persistent ADB shell: {exc}") from exc
        if process.stdin is None or process.stdout is None:
            process.terminate()
            raise ADBError("persistent ADB shell did not expose pipes")

        self._shell_process = process
        output_queue: queue.Queue[str | None] = queue.Queue()
        self._shell_output = output_queue

        def read_output() -> None:
            assert process.stdout is not None
            try:
                for line in process.stdout:
                    output_queue.put(line)
            finally:
                output_queue.put(None)

        self._shell_reader = threading.Thread(
            target=read_output,
            name="adb-shell-reader",
            daemon=True,
        )
        self._shell_reader.start()

    def _stop_persistent_shell(self) -> None:
        process = self._shell_process
        self._shell_process = None
        self._shell_reader = None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.5)

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
