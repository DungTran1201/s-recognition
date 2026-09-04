"""
Main Gradio Application UI Module for Multilingual Speech-To-Text.
Features non-blocking UI layout, mic recorder & file uploader, language selector (VI, EN, JA),
1-click text copy, file download, and real-time performance metrics display.
"""

from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import gradio_client.utils as client_utils

# Patch Gradio 4.x schema bug where boolean additionalProperties breaks OpenAPI schema generation
_orig_schema_to_type = client_utils._json_schema_to_python_type


def _safe_json_schema_to_python_type(schema: Any, defs: Any = None) -> str:
    if isinstance(schema, bool):
        return "Any" if schema else "None"
    if not isinstance(schema, dict):
        return "Any"
    return _orig_schema_to_type(schema, defs)


client_utils._json_schema_to_python_type = _safe_json_schema_to_python_type

from src.core.stt_engine import STTEngine
from src.ui.components import (
    ALLOWED_AUDIO_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    create_download_file,
    create_metrics_html,
    validate_audio_file,
)
from src.utils.config import init_ffmpeg
from src.utils.downloader import check_models_status
from src.utils.exceptions import STTBaseException
from src.utils.logger import logger


def get_language_choices() -> List[Tuple[str, str]]:
    """Generate dynamic language choices showing model availability status."""
    status = check_models_status()
    return [
        (f"Tiếng Việt (VI) {'[Sẵn sàng ✅]' if status.get('vi') else '[Thiếu model ⚠️]'}", "vi"),
        (f"English (EN) {'[Sẵn sàng ✅]' if status.get('en') else '[Thiếu model ⚠️]'}", "en"),
        (f"日本語 (JA) {'[Sẵn sàng ✅]' if status.get('ja') else '[Thiếu model ⚠️]'}", "ja"),
    ]


def transcribe_audio_ui_handler(
    mic_audio: Optional[str],
    file_audio: Optional[str],
    selected_language: str,
    active_tab_index: int,
) -> Tuple[str, Any, str]:
    """
    UI Callback handler for transcription requests.

    :param mic_audio: File path string from Microphone tab.
    :param file_audio: File path string from File Upload tab.
    :param selected_language: Language code selected ('vi', 'en', 'ja').
    :param active_tab_index: Index of currently active tab (0 for mic, 1 for upload).
    :return: Tuple containing (transcribed_text, gr.File update, metrics_html).
    """
    # Check if target language model exists before processing
    status = check_models_status()
    if not status.get(selected_language):
        warning_msg = (
            f"Mô hình ONNX cho ngôn ngữ [{selected_language.upper()}] chưa có trong thư mục models/.\n"
            f"👉 Vui lòng chạy lệnh: python scripts/download_models.py --lang {selected_language}\n"
            f"👉 Hoặc khởi động lại với cờ: python main.py --auto-download"
        )
        gr.Warning(warning_msg)
        raise gr.Error(warning_msg)

    audio_path = mic_audio if active_tab_index == 0 else file_audio

    # If active tab audio is empty, fallback to non-empty audio source if present
    if not audio_path:
        audio_path = file_audio if mic_audio is None else mic_audio

    if not audio_path:
        raise gr.Error("Vui lòng thu âm hoặc chọn file âm thanh trước khi nhấn Trích xuất!")

    try:
        # Step 1: Validate input file format and size
        validated_path = validate_audio_file(audio_path, max_size_mb=MAX_FILE_SIZE_MB)

        # Step 2: Transcribe via STTEngine singleton
        engine = STTEngine()
        result: Dict[str, Any] = engine.transcribe_file(
            file_path=validated_path,
            language=selected_language,
        )

        text: str = result.get("text", "")
        duration: float = result.get("duration", 0.0)
        inference_time: float = result.get("inference_time", 0.0)
        rtf: float = result.get("rtf", 0.0)

        # Step 3: Render metrics HTML and prepare download file
        metrics_html = create_metrics_html(duration, inference_time, rtf)
        download_path = create_download_file(text, filename=f"transcription_{selected_language}.txt")

        return text, gr.update(value=download_path, visible=True), metrics_html

    except STTBaseException as stt_err:
        logger.warning(f"UI STT Error: {stt_err}")
        raise gr.Error(str(stt_err))
    except Exception as err:
        logger.error(f"Unexpected UI error: {err}", exc_info=True)
        raise gr.Error(f"Đã xảy ra lỗi hệ thống: {err}")


def create_app() -> gr.Blocks:
    """
    Build and configure Gradio Blocks interface.

    :return: Configured gr.Blocks application instance.
    """
    init_ffmpeg()
    language_choices = get_language_choices()

    custom_css = """
    .main-title {
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        color: #64748B;
        margin-bottom: 20px;
    }
    """

    with gr.Blocks(title="Speech-To-Text AI Multi-language (VI/EN/JA)", css=custom_css) as app:
        active_tab_state = gr.State(value=0)  # 0 for Mic, 1 for File Upload

        gr.Markdown(
            """
            # 🎙️ Nhận Dạng Tiếng Nói Đa Ngôn Ngữ (Sherpa-ONNX STT)
            ### Hỗ trợ **Tiếng Việt (VI)**, **Tiếng Anh (EN)** và **Tiếng Nhật (JA)** với tốc độ siêu nhanh
            """
        )

        with gr.Row():
            # Left Column: Control Panel & Audio Inputs
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Cấu hình đầu vào")
                language_dropdown = gr.Dropdown(
                    choices=language_choices,
                    value="vi",
                    label="Chọn Ngôn ngữ Nhận dạng",
                    interactive=True,
                )

                gr.Markdown("### 🔊 Nguồn Âm thanh")
                with gr.Tabs() as tabs:
                    with gr.TabItem("🎤 Thu âm trực tiếp", id=0) as tab_mic:
                        mic_input = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="Ghi âm qua Microphone",
                        )

                    with gr.TabItem("📁 Tải lên File", id=1) as tab_file:
                        file_input = gr.Audio(
                            sources=["upload"],
                            type="filepath",
                            label=f"File âm thanh ({', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}) - Tối đa {MAX_FILE_SIZE_MB}MB",
                        )

                transcribe_btn = gr.Button(
                    "⚡ Trích xuất văn bản (Transcribe)",
                    variant="primary",
                    size="lg",
                )

            # Right Column: Output & Performance Metrics
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Kết quả Trích xuất")
                output_text = gr.Textbox(
                    label="Văn bản kết quả (Hỗ trợ Sao chép 1-Click)",
                    lines=10,
                    placeholder="Văn bản nhận dạng sẽ xuất hiện ở đây...",
                    show_copy_button=True,
                    interactive=False,
                )

                download_file = gr.File(
                    label="💾 Tải xuống kết quả (.txt)",
                    visible=False,
                    interactive=False,
                )

                metrics_output = gr.HTML(
                    value="""
                    <div style="background: #F1F5F9; border-radius: 8px; padding: 12px; color: #64748B; font-size: 13px; text-align: center;">
                        Vui lòng tải lên file hoặc thu âm và nhấn 'Trích xuất văn bản' để xem chỉ số hiệu năng RTF.
                    </div>
                    """
                )

        # Tab switch callbacks to keep track of active tab
        tab_mic.select(fn=lambda: 0, outputs=[active_tab_state])
        tab_file.select(fn=lambda: 1, outputs=[active_tab_state])

        # Main transcription button callback
        transcribe_btn.click(
            fn=transcribe_audio_ui_handler,
            inputs=[mic_input, file_input, language_dropdown, active_tab_state],
            outputs=[output_text, download_file, metrics_output],
        )

    return app


def launch_app(
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
) -> None:
    """
    Launch Gradio web server instance.

    :param host: Server host IP address (default: 127.0.0.1).
    :param port: Server port number (default: 7860).
    :param share: Whether to generate a public Gradio share link.
    """
    app = create_app()

    # Flexible server_name binding to prevent Windows socket localhost errors
    bind_host = "0.0.0.0" if host in ("127.0.0.1", "localhost") else host

    logger.info(f"Starting Gradio web server at http://{host}:{port} (bound to {bind_host}:{port}, share={share})...")

    app.launch(
        server_name=bind_host,
        server_port=port,
        share=share,
        show_api=False,
        inbrowser=False,
    )


if __name__ == "__main__":
    launch_app()
