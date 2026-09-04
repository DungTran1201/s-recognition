"""
Main Entrypoint Script for Multilingual Speech-To-Text (STT) Application.
Handles CLI arguments, directory validation, configuration loading, and Gradio server startup.
"""

import argparse
import sys
from pathlib import Path

from src.ui.app import launch_app
from src.utils.config import get_config, init_ffmpeg
from src.utils.logger import logger


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for server setup.

    :return: Namespace object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Multilingual Speech-To-Text (STT) Server (VI, EN, JA)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port number to bind server (default: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Generate a public Gradio share URL",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug logging mode",
    )
    parser.add_argument(
        "--auto-download",
        action="store_true",
        default=False,
        help="Automatically download missing Zipformer ONNX models on startup",
    )
    return parser.parse_args()


MODEL_DOWNLOAD_GUIDE = {
    "vi": "Model Tiếng Việt: Vui lòng tải các file Zipformer ONNX (encoder, decoder, joiner, tokens.txt) và đặt tại 'models/vietnamese-zipformer/'. Hoặc chạy: python scripts/download_models.py --lang vi",
    "en": "English Model: Download Zipformer ONNX files (encoder, decoder, joiner, tokens.txt) and place at 'models/english-zipformer/'. Hoặc chạy: python scripts/download_models.py --lang en",
    "ja": "Japanese Model: Download Zipformer ONNX files (encoder, decoder, joiner, tokens.txt) and place at 'models/japanese-zipformer/'. Hoặc chạy: python scripts/download_models.py --lang ja",
}


def prepare_environment(auto_download: bool = False) -> None:
    """
    Ensure project asset directories exist, initialize static-ffmpeg, and validate config setup.
    Logs helpful guidance if model files are missing without crashing.
    """
    init_ffmpeg()
    config = get_config()

    # Ensure required directories exist
    dirs_to_create = [
        Path("assets/icons"),
        Path("assets/sample_audio"),
        config.models_dir,
    ]

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"Models directory set to: '{config.models_dir.resolve()}'")

    from src.utils.downloader import download_model_for_language

    missing_languages = []
    # Check status of model paths for supported languages
    for lang in config.supported_languages:
        try:
            model_cfg = config.get_model_config(lang)
            if model_cfg.exists():
                logger.info(f"Language [{lang.upper()}] model files status: READY [OK]")
            else:
                if auto_download:
                    logger.info(f"Auto-download enabled. Attempting to download [{lang.upper()}] model...")
                    try:
                        download_model_for_language(lang)
                        logger.info(f"Language [{lang.upper()}] model files status: READY [OK]")
                    except Exception as dl_err:
                        logger.error(f"Auto-download failed for [{lang.upper()}]: {dl_err}")
                        missing_languages.append(lang.upper())
                else:
                    missing_languages.append(lang.upper())
                    guide = MODEL_DOWNLOAD_GUIDE.get(lang, "")
                    logger.warning(
                        f"Language [{lang.upper()}] model files missing in '{model_cfg.encoder_path.parent}'.\n"
                        f"    [INFO] {guide}"
                    )
        except Exception as err:
            logger.warning(f"Could not validate model status for [{lang.upper()}]: {err}")

    if missing_languages:
        logger.info(
            f"[NOTE] Web UI van khoi chay binh thuong. "
            f"Cac ngon ngu chua co model ({', '.join(missing_languages)}) se bao loi tren UI khi nhan Trich xuat.\n"
            f"       De tu dong tai tat ca model, chay: python main.py --auto-download"
        )


def main() -> None:
    """
    Application entrypoint.
    """
    args = parse_arguments()

    if args.debug:
        logger.setLevel(10)  # DEBUG level
        logger.debug("Debug logging enabled.")

    logger.info("Starting Speech-To-Text Application...")
    prepare_environment(auto_download=args.auto_download)

    try:
        launch_app(host=args.host, port=args.port, share=args.share)
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt). Exiting cleanly.")
        sys.exit(0)
    except Exception as err:
        logger.critical(f"Fatal error starting application: {err}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
