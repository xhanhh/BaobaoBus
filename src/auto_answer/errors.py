"""Domain exceptions used to keep orchestration independent of implementations."""


class AutoAnswerError(Exception):
    """Base class for expected application failures."""


class ConfigurationError(AutoAnswerError):
    """Configuration is missing, inconsistent, or unsafe."""


class CaptureError(AutoAnswerError):
    """A frame could not be captured or cropped."""


class OCRError(AutoAnswerError):
    """OCR initialization or inference failed."""


class UnsafeOCRResult(OCRError):
    """OCR output is empty or too uncertain to permit a tap."""


class SolverError(AutoAnswerError):
    """A remote or local solver failed to return a safe answer."""


class ADBError(AutoAnswerError):
    """ADB device discovery or input failed."""


class StateDetectionError(AutoAnswerError):
    """The page did not change or stabilize in time."""
