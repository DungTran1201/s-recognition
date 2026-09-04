"""
Multilingual Speech-To-Text Engine Module using Sherpa-ONNX.
Applies Singleton Pattern for efficient model instance management across Vietnamese (VI),
English (EN), and Japanese (JA).
"""

import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, Union

import numpy as np
import sherpa_onnx

from src.core.audio_processor import AudioProcessor
from src.utils.config import STTConfig, get_config
from src.utils.exceptions import AudioProcessingError, ModelInferenceError, ModelNotFoundError
from src.utils.logger import logger, log_transcription_stats


class STTEngine:
    """
    Singleton Wrapper for Sherpa-ONNX Offline Recognizer supporting multi-language models.
    """

    _instance: Optional["STTEngine"] = None
    _lock: Lock = Lock()

    def __new__(cls, config: Optional[STTConfig] = None) -> "STTEngine":
        """
        Thread-safe Singleton instance creation.

        :param config: Optional STTConfig override.
        :return: Shared STTEngine singleton instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self, config: Optional[STTConfig] = None) -> None:
        """
        Initialize STTEngine instance with configuration and model registry.

        :param config: Optional STTConfig configuration.
        """
        if getattr(self, "_initialized", False):
            return

        self.config: STTConfig = config or get_config()
        self._recognizers: Dict[str, sherpa_onnx.OfflineRecognizer] = {}
        self._initialized: bool = True
        logger.info(
            f"STTEngine initialized | "
            f"Default Lang: '{self.config.default_language}' | "
            f"Threads: {self.config.num_threads} | "
            f"Sample Rate: {self.config.target_sample_rate}Hz"
        )

    def get_recognizer(self, language: str) -> sherpa_onnx.OfflineRecognizer:
        """
        Get or lazy-load the Sherpa-ONNX OfflineRecognizer for the target language.

        :param language: Language code ('vi', 'en', 'ja').
        :return: Initialized sherpa_onnx.OfflineRecognizer.
        :raises LanguageNotSupportedError: If language code is unsupported.
        :raises ModelNotFoundError: If model files are missing from disk.
        :raises ModelInferenceError: If model initialization fails.
        """
        lang = self.config.validate_language(language)

        if lang in self._recognizers:
            return self._recognizers[lang]

        with self._lock:
            if lang in self._recognizers:
                return self._recognizers[lang]

            model_cfg = self.config.get_model_config(lang)
            model_cfg.validate(lang)

            logger.info(f"Loading Sherpa-ONNX model for language [{lang.upper()}]...")

            try:
                recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                    encoder=str(model_cfg.resolved_encoder_path),
                    decoder=str(model_cfg.resolved_decoder_path),
                    joiner=str(model_cfg.resolved_joiner_path),
                    tokens=str(model_cfg.resolved_tokens_path),
                    num_threads=self.config.num_threads,
                    sample_rate=self.config.target_sample_rate,
                    feature_dim=80,
                    decoding_method="greedy_search",
                    debug=False,
                )
                self._recognizers[lang] = recognizer
                logger.info(f"Successfully loaded model for [{lang.upper()}]")
                return recognizer
            except ModelNotFoundError:
                raise
            except Exception as err:
                raise ModelInferenceError(
                    f"Failed to initialize Sherpa-ONNX model for language '{lang}'",
                    details=str(err),
                ) from err

    def transcribe_samples(
        self,
        samples: np.ndarray,
        sample_rate: int = 16000,
        language: str = "vi",
    ) -> Dict[str, Any]:
        """
        Transcribe raw floating point 1D audio sample array.

        :param samples: 1D NumPy array with dtype float32 normalized to [-1.0, 1.0].
        :param sample_rate: Sample rate of the audio (default: 16000).
        :param language: Language code ('vi', 'en', 'ja').
        :return: Dictionary containing transcription results and metrics:
                 {'text': str, 'language': str, 'duration': float, 'inference_time': float, 'rtf': float}.
        :raises AudioProcessingError: If audio array is invalid or empty.
        :raises ModelInferenceError: If decoding fails.
        """
        if not isinstance(samples, np.ndarray) or samples.ndim != 1 or len(samples) == 0:
            raise AudioProcessingError("Input audio samples must be a non-empty 1D NumPy array.")

        lang = self.config.validate_language(language)
        recognizer = self.get_recognizer(lang)

        audio_duration = len(samples) / float(sample_rate)

        try:
            start_time = time.perf_counter()
            stream = recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples.astype(np.float32))
            recognizer.decode_stream(stream)
            inference_time = time.perf_counter() - start_time

            result_text = stream.result.text.strip()
            rtf = log_transcription_stats(lang, audio_duration, inference_time, result_text)

            return {
                "text": result_text,
                "language": lang,
                "duration": round(audio_duration, 3),
                "inference_time": round(inference_time, 4),
                "rtf": round(rtf, 4),
            }
        except ModelNotFoundError:
            raise
        except Exception as err:
            raise ModelInferenceError(
                f"Inference error during transcription for language '{lang}'",
                details=str(err),
            ) from err

    def transcribe_file(
        self,
        file_path: Union[str, Path],
        language: str = "vi",
    ) -> Dict[str, Any]:
        """
        Load audio file from disk and perform speech recognition.

        :param file_path: Path to audio file.
        :param language: Language code ('vi', 'en', 'ja').
        :return: Dictionary containing transcription results and performance metrics.
        """
        samples, sr = AudioProcessor.load_audio(
            file_path=file_path,
            target_sample_rate=self.config.target_sample_rate,
        )

        try:
            result = self.transcribe_samples(
                samples=samples,
                sample_rate=sr,
                language=language,
            )
            return result
        finally:
            AudioProcessor.cleanup_memory(samples)
