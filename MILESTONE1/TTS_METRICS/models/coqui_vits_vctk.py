"""
Coqui TTS - VITS model trained on VCTK dataset.

Multi-speaker model, we'll use a single speaker for consistency.
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

from .base_tts import BaseTTS, TTSModelInfo


class CoquiVITSVCTK(BaseTTS):
    """Coqui TTS VITS model trained on VCTK multi-speaker dataset."""

    MODEL_NAME = "tts_models/en/vctk/vits"
    # Use speaker p226 (female, English) for consistency
    SPEAKER_ID = "p226"

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize Coqui VITS VCTK model."""
        super().__init__(cache_dir)
        self.tts = None
        self.model_info_cache = None

    def load(self) -> None:
        """Load Coqui TTS model."""
        if self._is_loaded:
            return

        try:
            from TTS.api import TTS

            os.environ['COQUI_TTS_CACHE_DIR'] = str(self.cache_dir)

            print(f"Loading {self.MODEL_NAME} (speaker: {self.SPEAKER_ID})...")
            self.tts = TTS(model_name=self.MODEL_NAME, progress_bar=False)

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
            # VCTK is multi-speaker, specify speaker_name
            audio = self.tts.tts(text=text, speaker=self.SPEAKER_ID)

            if not isinstance(audio, np.ndarray):
                audio = np.array(audio)

            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            audio = self._normalize_audio(audio)

            sample_rate = self.tts.synthesizer.output_sample_rate

            self._validate_audio(audio, sample_rate)

            return audio, sample_rate

        except Exception as e:
            raise RuntimeError(f"Synthesis failed: {e}")

    def get_model_info(self) -> TTSModelInfo:
        """Get model metadata."""
        if self.model_info_cache is None:
            disk_size_mb = 0.0
            if self._is_loaded and self.tts:
                model_cache = self.cache_dir / "tts_models--en--vctk--vits"
                if model_cache.exists():
                    disk_size_mb = sum(
                        f.stat().st_size for f in model_cache.rglob('*') if f.is_file()
                    ) / (1024 * 1024)
                else:
                    disk_size_mb = 100.0

            self.model_info_cache = TTSModelInfo(
                name="coqui_vits_vctk",
                display_name="Coqui VITS (VCTK)",
                model_type="neural",
                architecture="VITS",
                language="en-UK",
                voice_description=f"Multi-speaker (using {self.SPEAKER_ID}: female, British English)",
                disk_size_mb=disk_size_mb if disk_size_mb > 0 else 100.0,
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
