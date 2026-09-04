"""
Structured Logger Module for STT Application.
Provides colored console output and performance metric logging (Execution Time, Audio Duration, RTF).
"""

import logging
import sys
from typing import Optional


def setup_logger(name: str = "stt_app", level: int = logging.INFO) -> logging.Logger:
    """
    Create and configure system logger with standardized formatting.

    :param name: Name of the logger instance.
    :param level: Logging severity level.
    :return: Configured logging.Logger object.
    """
    logger_inst = logging.getLogger(name)

    if not logger_inst.handlers:
        logger_inst.setLevel(level)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        stream = sys.stdout
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

        console_handler = logging.StreamHandler(stream)
        console_handler.setFormatter(formatter)
        logger_inst.addHandler(console_handler)

    return logger_inst


logger: logging.Logger = setup_logger()


def log_transcription_stats(
    language: str,
    audio_duration: float,
    inference_time: float,
    transcription_text: Optional[str] = None,
) -> float:
    """
    Calculate Real-Time Factor (RTF) and log execution stats.

    :param language: Language code processed ('vi', 'en', 'ja').
    :param audio_duration: Audio duration in seconds.
    :param inference_time: Inference execution time in seconds.
    :param transcription_text: Optional decoded text snippet for preview.
    :return: Calculated Real-Time Factor (RTF).
    """
    rtf = inference_time / audio_duration if audio_duration > 0 else 0.0
    text_preview = (
        (transcription_text[:50] + "...") if transcription_text and len(transcription_text) > 50 else (transcription_text or "")
    )

    logger.info(
        f"Transcribed [{language.upper()}] | "
        f"Audio: {audio_duration:.2f}s | "
        f"Inference: {inference_time:.3f}s | "
        f"RTF: {rtf:.4f} | "
        f"Text Preview: '{text_preview}'"
    )

    return rtf
