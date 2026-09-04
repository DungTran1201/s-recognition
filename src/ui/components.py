"""
UI Components and Helpers Module for Gradio Interface.
Provides file size/format validation, metrics HTML rendering, and text download file generation.
"""

import os
import tempfile
from pathlib import Path
from typing import Set, Union

from src.utils.exceptions import AudioProcessingError

MAX_FILE_SIZE_MB: float = 50.0
ALLOWED_AUDIO_EXTENSIONS: Set[str] = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def validate_audio_file(
    file_path: Union[str, Path],
    max_size_mb: float = MAX_FILE_SIZE_MB,
    allowed_extensions: Set[str] = ALLOWED_AUDIO_EXTENSIONS,
) -> Path:
    """
    Validate audio file path, format extension, and file size limit.

    :param file_path: Path to audio file.
    :param max_size_mb: Maximum allowed file size in Megabytes (default: 50MB).
    :param allowed_extensions: Set of allowed file extensions.
    :return: Validated Path object.
    :raises AudioProcessingError: If file is missing, unsupported format, or exceeds size limit.
    """
    if not file_path:
        raise AudioProcessingError("Vui lòng cung cấp file âm thanh hoặc thu âm từ Microphone.")

    path = Path(file_path)

    if not path.is_file():
        raise AudioProcessingError(f"File âm thanh không tồn tại tại đường dẫn: {file_path}")

    ext = path.suffix.lower()
    if ext not in allowed_extensions:
        raise AudioProcessingError(
            f"Định dạng file '{ext}' không được hỗ trợ. "
            f"Các định dạng cho phép: {', '.join(sorted(allowed_extensions))}"
        )

    file_size_bytes = path.stat().st_size
    max_bytes = max_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        file_size_mb = file_size_bytes / (1024 * 1024)
        raise AudioProcessingError(
            f"Kích thước file ({file_size_mb:.2f} MB) vượt quá giới hạn cho phép ({max_size_mb} MB)."
        )

    return path


def create_metrics_html(duration: float, inference_time: float, rtf: float) -> str:
    """
    Generate styled HTML metrics banner for Gradio UI.

    :param duration: Audio duration in seconds.
    :param inference_time: AI inference time in seconds.
    :param rtf: Real-Time Factor (inference_time / duration).
    :return: HTML string.
    """
    # Color badge based on RTF efficiency (RTF < 1.0 means faster than real time)
    rtf_color = "#10B981" if rtf <= 0.5 else "#F59E0B" if rtf <= 1.0 else "#EF4444"

    return f"""
    <div style="
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border-radius: 12px;
        padding: 16px 24px;
        color: #F8FAFC;
        font-family: system-ui, -apple-system, sans-serif;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        margin-top: 10px;
    ">
        <div style="font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94A3B8; margin-bottom: 8px;">
            📊 Chỉ số hiệu năng suy luận AI (Performance Metrics)
        </div>
        <div style="display: flex; gap: 24px; flex-wrap: wrap; align-items: center;">
            <div>
                <span style="font-size: 12px; color: #CBD5E1;">Thời lượng âm thanh:</span><br/>
                <strong style="font-size: 18px; color: #38BDF8;">{duration:.2f}s</strong>
            </div>
            <div style="border-left: 1px solid #334155; padding-left: 24px;">
                <span style="font-size: 12px; color: #CBD5E1;">Thời gian xử lý AI:</span><br/>
                <strong style="font-size: 18px; color: #A78BFA;">{inference_time:.3f}s</strong>
            </div>
            <div style="border-left: 1px solid #334155; padding-left: 24px;">
                <span style="font-size: 12px; color: #CBD5E1;">Real-Time Factor (RTF):</span><br/>
                <strong style="font-size: 18px; color: {rtf_color};">{rtf:.4f}</strong>
                <span style="font-size: 11px; color: #94A3B8; margin-left: 4px;">({"Nhanh hơn thời gian thực" if rtf < 1.0 else "Chậm hơn thời gian thực"})</span>
            </div>
        </div>
    </div>
    """


def create_download_file(text: str, filename: str = "transcription.txt") -> str:
    """
    Save text string into a temporary .txt file for Gradio download component.

    :param text: Transcribed text content.
    :param filename: Desired download file name.
    :return: Path string to the created temporary file.
    """
    temp_dir = Path(tempfile.gettempdir()) / "stt_downloads"
    temp_dir.mkdir(parents=True, exist_ok=True)

    file_path = temp_dir / filename
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text or "")

    return str(file_path)
