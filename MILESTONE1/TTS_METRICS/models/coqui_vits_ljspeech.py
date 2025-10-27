"""
Coqui TTS - VITS model trained on LJSpeech dataset.

Fast neural TTS with single female US English voice.
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

from .base_tts import BaseTTS, TTSModelInfo


class CoquiVITSLJSpeech(BaseTTS):
    """Coqui TTS VITS model trained on LJSpeech dataset."""

    MODEL_NAME = "tts_models/en/ljspeech/vits"

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize Coqui VITS LJSpeech model."""
        super().__init__(cache_dir)
        self.tts = None
        self.model_info_cache = None

    def load(self) -> None:
        """Load Coqui TTS model."""
        if self._is_loaded:
            return

        try:
            # Import here to avoid loading TTS library until needed
            from TTS.api import TTS

            # Set cache directory via environment variable
            os.environ['COQUI_TTS_CACHE_DIR'] = str(self.cache_dir)

            print(f"Loading {self.MODEL_NAME}...")
            self.tts = TTS(model_name=self.MODEL_NAME, progress_bar=False)

            # Move to CPU (we're not using GPU for this evaluation)
            # Note: Coqui TTS uses CPU by default if CUDA not available

            self._is_loaded = True
            print(f"✓ Loaded {self.MODEL_NAME}")

        except ImportError as e:
            raise RuntimeError(f"TTS library not installed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text.

        Args:
            text: Input text

        Returns:
            Tuple of (audio_samples, sample_rate)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            # Coqui TTS returns numpy array directly
            audio = self.tts.tts(text=text)

            # Convert to numpy array if it isn't already
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)

            # Ensure float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize to [-1, 1]
            audio = self._normalize_audio(audio)

            # Get sample rate from model
            sample_rate = self.tts.synthesizer.output_sample_rate

            # Validate
            self._validate_audio(audio, sample_rate)

            return audio, sample_rate

        except Exception as e:
            raise RuntimeError(f"Synthesis failed: {e}")

    def get_model_info(self) -> TTSModelInfo:
        """Get model metadata."""
        if self.model_info_cache is None:
            # Calculate disk size if model is cached
            disk_size_mb = 0.0
            if self._is_loaded and self.tts:
                # Estimate based on model files in cache
                model_cache = self.cache_dir / "tts_models--en--ljspeech--vits"
                if model_cache.exists():
                    disk_size_mb = sum(
                        f.stat().st_size for f in model_cache.rglob('*') if f.is_file()
                    ) / (1024 * 1024)
                else:
                    disk_size_mb = 85.0  # Approximate size

            self.model_info_cache = TTSModelInfo(
                name="coqui_vits_ljspeech",
                display_name="Coqui VITS (LJSpeech)",
                model_type="neural",
                architecture="VITS",
                language="en-US",
                voice_description="Female, US English, clear prosody",
                disk_size_mb=disk_size_mb if disk_size_mb > 0 else 85.0,
                sample_rate=22050,
                license="Mozilla Public License 2.0"
            )

        return self.model_info_cache

    def unload(self) -> None:
        """Unload model from memory."""
        if self.tts is not None:
            del self.tts
            self.tts = None
        self._is_loaded = False
