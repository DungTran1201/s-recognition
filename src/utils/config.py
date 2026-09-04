"""
Configuration settings module for Speech-To-Text application.
Provides dataclass configurations for model paths, sample rates, and multi-language mappings.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

from src.utils.exceptions import LanguageNotSupportedError, ModelNotFoundError


@dataclass
class ModelPathConfig:
    """Dataclass holding paths to Sherpa-ONNX model files for a specific language."""

    encoder_path: Path
    decoder_path: Path
    joiner_path: Path
    tokens_path: Path

    def _resolve_file(self, primary: Path, glob_pattern: str) -> Optional[Path]:
        if primary.is_file():
            return primary
        parent = primary.parent
        if parent.is_dir():
            candidates = sorted(list(parent.glob(glob_pattern)))
            if candidates:
                return candidates[0]
        return None

    @property
    def resolved_encoder_path(self) -> Path:
        return self._resolve_file(self.encoder_path, "encoder*.onnx") or self.encoder_path

    @property
    def resolved_decoder_path(self) -> Path:
        return self._resolve_file(self.decoder_path, "decoder*.onnx") or self.decoder_path

    @property
    def resolved_joiner_path(self) -> Path:
        return self._resolve_file(self.joiner_path, "joiner*.onnx") or self.joiner_path

    @property
    def resolved_tokens_path(self) -> Path:
        return self._resolve_file(self.tokens_path, "tokens.txt") or self.tokens_path

    def exists(self) -> bool:
        """Check if all required model files exist on disk."""
        return (
            self.resolved_encoder_path.is_file()
            and self.resolved_decoder_path.is_file()
            and self.resolved_joiner_path.is_file()
            and self.resolved_tokens_path.is_file()
        )

    def validate(self, language: str) -> None:
        """
        Validate that model files exist, raising ModelNotFoundError if any file is missing.

        :param language: Language code being validated.
        :raises ModelNotFoundError: If any of the 4 model files is missing.
        """
        for name, path in [
            ("encoder", self.resolved_encoder_path),
            ("decoder", self.resolved_decoder_path),
            ("joiner", self.resolved_joiner_path),
            ("tokens", self.resolved_tokens_path),
        ]:
            if not path.is_file():
                raise ModelNotFoundError(language, f"{name} file missing: {path}")


@dataclass
class STTConfig:
    """Global configuration settings for STT processing and model management."""

    target_sample_rate: int = 16000
    default_language: str = "vi"
    supported_languages: Tuple[str, ...] = ("vi", "en", "ja")
    num_threads: int = field(default_factory=lambda: max(1, min(4, os.cpu_count() or 1)))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    max_audio_duration_seconds: float = 600.0  # 10 minutes max per file

    models: Dict[str, ModelPathConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize default model paths mapping for supported languages if not provided."""
        if not self.models:
            base_dir = self.models_dir
            self.models = {
                "vi": ModelPathConfig(
                    encoder_path=base_dir / "vietnamese-zipformer" / "encoder-epoch-99-avg-1.onnx",
                    decoder_path=base_dir / "vietnamese-zipformer" / "decoder-epoch-99-avg-1.onnx",
                    joiner_path=base_dir / "vietnamese-zipformer" / "joiner-epoch-99-avg-1.onnx",
                    tokens_path=base_dir / "vietnamese-zipformer" / "tokens.txt",
                ),
                "en": ModelPathConfig(
                    encoder_path=base_dir / "english-zipformer" / "encoder-epoch-99-avg-1.onnx",
                    decoder_path=base_dir / "english-zipformer" / "decoder-epoch-99-avg-1.onnx",
                    joiner_path=base_dir / "english-zipformer" / "joiner-epoch-99-avg-1.onnx",
                    tokens_path=base_dir / "english-zipformer" / "tokens.txt",
                ),
                "ja": ModelPathConfig(
                    encoder_path=base_dir / "japanese-zipformer" / "encoder-epoch-99-avg-1.onnx",
                    decoder_path=base_dir / "japanese-zipformer" / "decoder-epoch-99-avg-1.onnx",
                    joiner_path=base_dir / "japanese-zipformer" / "joiner-epoch-99-avg-1.onnx",
                    tokens_path=base_dir / "japanese-zipformer" / "tokens.txt",
                ),
            }

    def validate_language(self, language: str) -> str:
        """
        Validate and normalize requested language code.

        :param language: Language code (e.g. 'vi', 'en', 'ja').
        :return: Normalized lowercase language code.
        :raises LanguageNotSupportedError: If language code is invalid.
        """
        lang = language.lower().strip()
        if lang not in self.supported_languages:
            raise LanguageNotSupportedError(lang, self.supported_languages)
        return lang

    def get_model_config(self, language: str) -> ModelPathConfig:
        """
        Retrieve model paths for a specific language after validation.

        :param language: Language code ('vi', 'en', 'ja').
        :return: ModelPathConfig instance for the specified language.
        """
        lang = self.validate_language(language)
        return self.models[lang]


_config_instance: STTConfig = STTConfig()


def get_config() -> STTConfig:
    """Get global STTConfig singleton instance."""
    return _config_instance


def init_ffmpeg() -> bool:
    """
    Auto-initialize ffmpeg and ffprobe binary paths via static-ffmpeg.

    :return: True if ffmpeg and ffprobe are loaded into PATH, False otherwise.
    """
    try:
        import static_ffmpeg
        from src.utils.logger import logger

        static_ffmpeg.add_paths()
        logger.info("FFmpeg & FFprobe binary paths initialized via static-ffmpeg.")
        return True
    except Exception as err:
        try:
            from src.utils.logger import logger
            logger.warning(
                f"Could not auto-initialize static-ffmpeg: {err}.\n"
                f"  [INFO] Co the cai dat ffmpeg thu cong qua Windows Winget: 'winget install ffmpeg'"
            )
        except Exception:
            pass
        return False
