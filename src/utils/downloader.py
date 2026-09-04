"""
Auto-downloader utility for Sherpa-ONNX Zipformer models (VI, EN, JA).
Downloads pre-trained ASR model archives from official releases and extracts them to the models/ directory.
"""

import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional

import requests
from tqdm import tqdm

from src.utils.config import get_config
from src.utils.exceptions import LanguageNotSupportedError, ModelNotFoundError
from src.utils.logger import logger

MODEL_DOWNLOAD_URLS: Dict[str, Dict[str, str]] = {
    "vi": {
        "name": "vietnamese-zipformer",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09.tar.bz2",
        "target_dir_name": "vietnamese-zipformer",
    },
    "en": {
        "name": "english-zipformer",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2",
        "target_dir_name": "english-zipformer",
    },
    "ja": {
        "name": "japanese-zipformer",
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01.tar.bz2",
        "target_dir_name": "japanese-zipformer",
    },
}


def download_file_with_progress(url: str, output_path: Path) -> Path:
    """
    Download file from URL with tqdm progress bar.

    :param url: Download source URL.
    :param output_path: Path to save downloaded file.
    :return: Saved file Path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading model archive from: {url}")

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192

    with open(output_path, "wb") as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    return output_path


def extract_archive_to_model_dir(archive_path: Path, target_dir: Path) -> None:
    """
    Extract tar.bz2 / zip archive and place model files into target_dir.

    :param archive_path: Path to downloaded archive.
    :param target_dir: Path to model target directory.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_extract_dir:
        temp_dir_path = Path(temp_extract_dir)

        if archive_path.name.endswith(".tar.bz2") or archive_path.name.endswith(".tar.gz") or archive_path.name.endswith(".tgz"):
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(path=temp_dir_path)
        elif archive_path.name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(path=temp_dir_path)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path.name}")

        # Find top extracted directory or files
        subdirs = [p for p in temp_dir_path.iterdir() if p.is_dir()]
        source_folder = subdirs[0] if len(subdirs) == 1 else temp_dir_path

        # Copy all files (.onnx, .txt, etc.) directly into target_dir
        for item in source_folder.glob("*"):
            if item.is_file():
                shutil.copy2(item, target_dir / item.name)

    logger.info(f"Successfully extracted model files to: '{target_dir.resolve()}'")


def download_model_for_language(language: str, force: bool = False) -> bool:
    """
    Check and auto-download Sherpa-ONNX model files for specified language.

    :param language: Language code ('vi', 'en', 'ja').
    :param force: If True, re-downloads model even if files exist.
    :return: True if model is available/downloaded successfully.
    """
    config = get_config()
    lang = config.validate_language(language)

    if lang not in MODEL_DOWNLOAD_URLS:
        raise LanguageNotSupportedError(lang, config.supported_languages)

    model_info = MODEL_DOWNLOAD_URLS[lang]
    target_dir = config.models_dir / model_info["target_dir_name"]
    model_cfg = config.get_model_config(lang)

    if not force and model_cfg.exists():
        logger.info(f"Language [{lang.upper()}] model already exists at '{target_dir}'. Skipping download.")
        return True

    url = model_info["url"]
    filename = url.split("/")[-1]

    with tempfile.TemporaryDirectory() as temp_dl_dir:
        temp_archive_path = Path(temp_dl_dir) / filename
        try:
            download_file_with_progress(url, temp_archive_path)
            extract_archive_to_model_dir(temp_archive_path, target_dir)
            logger.info(f"Language [{lang.upper()}] model download complete!")
            return True
        except Exception as err:
            logger.error(f"Failed to download model for language [{lang.upper()}]: {err}")
            raise ModelNotFoundError(
                lang,
                f"Auto-download failed from {url}. Error: {err}",
            ) from err


def check_models_status() -> Dict[str, bool]:
    """
    Check model existence status for all supported languages.

    :return: Dict mapping language code to boolean readiness.
    """
    config = get_config()
    status = {}

    for lang in config.supported_languages:
        try:
            model_cfg = config.get_model_config(lang)
            status[lang] = model_cfg.exists()
        except Exception:
            status[lang] = False

    return status
