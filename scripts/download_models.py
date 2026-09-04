"""
Standalone Model Downloader Script for Sherpa-ONNX Zipformer models (VI, EN, JA).
Usage:
    python scripts/download_models.py --lang vi
    python scripts/download_models.py --lang all
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.config import get_config
from src.utils.downloader import download_model_for_language
from src.utils.logger import logger


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for model download script."""
    parser = argparse.ArgumentParser(
        description="Download Sherpa-ONNX Zipformer ASR models for Vietnamese, English, and Japanese."
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="all",
        choices=["all", "vi", "en", "ja"],
        help="Language model to download: 'vi', 'en', 'ja', or 'all' (default: 'all')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-download even if model files already exist",
    )
    return parser.parse_args()


def main() -> None:
    """Download specified models."""
    args = parse_args()
    config = get_config()

    languages = (
        config.supported_languages
        if args.lang == "all"
        else (args.lang.lower(),)
    )

    logger.info(f"Starting model download task for: {', '.join(languages)}")

    success_count = 0
    for lang in languages:
        try:
            download_model_for_language(lang, force=args.force)
            success_count += 1
        except Exception as err:
            logger.error(f"Failed to download model for [{lang.upper()}]: {err}")

    if success_count == len(languages):
        logger.info("All requested models downloaded successfully! [OK]")
    else:
        logger.warning(f"Downloaded {success_count}/{len(languages)} models successfully.")


if __name__ == "__main__":
    main()
