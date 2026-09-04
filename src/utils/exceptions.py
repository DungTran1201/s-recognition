"""
Custom exceptions module for STT Application.
Defines clean exception hierarchy for audio processing, model inference, and configuration errors.
"""

from typing import Optional, Tuple


class STTBaseException(Exception):
    """Base exception class for all Speech-To-Text module errors."""

    def __init__(self, message: str, details: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class AudioProcessingError(STTBaseException):
    """Raised when audio loading, decoding, resampling, or normalization fails."""

    pass


class ModelInferenceError(STTBaseException):
    """Raised when Sherpa-ONNX recognizer inference or stream decoding fails."""

    pass


class LanguageNotSupportedError(STTBaseException):
    """Raised when an unsupported language code is selected."""

    def __init__(self, language: str, supported: Tuple[str, ...] = ("vi", "en", "ja")) -> None:
        message = f"Language '{language}' is not supported. Supported languages are: {', '.join(supported)}"
        super().__init__(message)
        self.language = language
        self.supported = supported


class ModelNotFoundError(STTBaseException):
    """Raised when required ONNX model files or tokens file cannot be located on disk."""

    def __init__(self, language: str, path: str) -> None:
        message = f"Model file for language '{language}' not found at: {path}"
        super().__init__(message)
        self.language = language
        self.path = path
