"""
Audio Preprocessor Module for Speech-To-Text processing.
Handles fast audio loading (soundfile + librosa fallback), mono conversion,
16kHz resampling, amplitude normalization (float32), and memory optimization.
"""

import gc
from pathlib import Path
from typing import Tuple, Union

import librosa
import numpy as np
import soundfile as sf

from src.utils.exceptions import AudioProcessingError
from src.utils.logger import logger


class AudioProcessor:
    """High-performance audio processing utility class for STT engines."""

    @staticmethod
    def load_audio(
        file_path: Union[str, Path],
        target_sample_rate: int = 16000,
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file, convert to mono, resample to target_sample_rate (16000 Hz),
        and normalize amplitude to float32 range [-1.0, 1.0].

        Luôn ưu tiên dùng soundfile.read() trước, fallback sang librosa.load() khi xảy ra lỗi.

        :param file_path: Path to input audio file.
        :param target_sample_rate: Target sample rate in Hz (default: 16000).
        :return: Tuple containing (1D float32 np.ndarray mono audio, target_sample_rate).
        :raises AudioProcessingError: If audio file cannot be loaded or decoded.
        """
        file_str = str(file_path)
        if not Path(file_str).is_file():
            raise AudioProcessingError(f"Audio file does not exist: {file_str}")

        audio: np.ndarray
        sample_rate: int

        # Step 1: Attempt fast read via soundfile
        try:
            audio, sample_rate = sf.read(file_str, dtype="float32")
        except Exception as sf_err:
            logger.warning(f"SoundFile read failed for '{file_str}' ({sf_err}). Falling back to librosa.")
            try:
                # Librosa fallback (returns mono float32 by default if mono=True)
                audio, sample_rate = librosa.load(file_str, sr=None, mono=False, dtype=np.float32)
            except Exception as librosa_err:
                raise AudioProcessingError(
                    f"Failed to load audio file with both SoundFile and Librosa: {file_str}",
                    details=f"SoundFile: {sf_err} | Librosa: {librosa_err}",
                ) from librosa_err

        # Step 2: Convert to Mono 1D array if stereo/multichannel
        audio = AudioProcessor.to_mono(audio)

        # Step 3: Resample to target_sample_rate if needed
        if sample_rate != target_sample_rate:
            audio = AudioProcessor.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
            sample_rate = target_sample_rate

        # Step 4: Normalize amplitude
        audio = AudioProcessor.normalize(audio)

        # Step 5: Enforce float32 1D array
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        return audio, sample_rate

    @staticmethod
    def to_mono(audio: np.ndarray) -> np.ndarray:
        """
        Convert multi-channel audio to single-channel (mono) by averaging channels.

        :param audio: Input NumPy array (1D or 2D).
        :return: 1D NumPy array representing mono audio.
        """
        if audio.ndim == 1:
            return audio

        # SoundFile returns shape (samples, channels), Librosa (channels, samples)
        if audio.shape[0] < audio.shape[1] and audio.ndim == 2:
            # Librosa format (channels, samples)
            audio = np.mean(audio, axis=0)
        else:
            # SoundFile format (samples, channels)
            audio = np.mean(audio, axis=1)

        return audio

    @staticmethod
    def resample(audio: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
        """
        Resample audio array from orig_sr to target_sr.

        :param audio: 1D NumPy float32 array.
        :param orig_sr: Original sample rate.
        :param target_sr: Target sample rate (default: 16000).
        :return: Resampled 1D NumPy float32 array.
        """
        if orig_sr == target_sr:
            return audio

        resampled_audio = librosa.resample(y=audio, orig_sr=orig_sr, target_sr=target_sr)
        return resampled_audio.astype(np.float32)

    @staticmethod
    def normalize(audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio peak amplitude to range [-1.0, 1.0].

        :param audio: Input 1D NumPy array.
        :return: Peak-normalized 1D NumPy array.
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0.0:
            return audio / max_val
        return audio

    @staticmethod
    def cleanup_memory(*arrays: np.ndarray) -> None:
        """
        Explicitly delete large audio NumPy arrays and run garbage collection.

        :param arrays: Variable number of NumPy arrays to release.
        """
        for arr in arrays:
            del arr
        gc.collect()
