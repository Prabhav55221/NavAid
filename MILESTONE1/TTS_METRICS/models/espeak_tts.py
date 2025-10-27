"""
eSpeak-NG TTS - Ultra-fast formant synthesis.

Robotic but very low latency. Good baseline for minimum RTF.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import soundfile as sf

from .base_tts import BaseTTS, TTSModelInfo


class ESpeakTTS(BaseTTS):
    """eSpeak-NG formant synthesis TTS."""

    # Voice settings for US English
    VOICE = "en-us"
    SPEED = 175  # words per minute (default 175)
    PITCH = 50   # 0-99 range (default 50)

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize eSpeak-NG TTS."""
        super().__init__(cache_dir)
        self.espeak_available = False
        self.model_info_cache = None

    def _check_espeak_installed(self) -> bool:
        """Check if espeak-ng is installed."""
        try:
            result = subprocess.run(
                ['espeak-ng', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    def load(self) -> None:
        """Check if eSpeak-NG is available."""
        if self._is_loaded:
            return

        if not self._check_espeak_installed():
            raise RuntimeError(
                "eSpeak-NG not installed.\n"
                "macOS: brew install espeak-ng\n"
                "Linux: sudo apt-get install espeak-ng"
            )

        self.espeak_available = True
        self._is_loaded = True
        print(f"✓ eSpeak-NG ready (voice: {self.VOICE})")

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text using eSpeak-NG.

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
            # Use temporary file for output
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                output_path = Path(tmp_file.name)

            try:
                # Run espeak-ng command
                cmd = [
                    'espeak-ng',
                    '-v', self.VOICE,
                    '-s', str(self.SPEED),
                    '-p', str(self.PITCH),
                    '-w', str(output_path),
                    text
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode != 0:
                    raise RuntimeError(f"eSpeak-NG failed: {result.stderr}")

                # Read generated audio
                audio, sample_rate = sf.read(output_path, dtype='float32')

                # Ensure 1D array
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                # Normalize
                audio = self._normalize_audio(audio)

                # Validate
                self._validate_audio(audio, sample_rate)

                return audio, sample_rate

            finally:
                # Clean up temp file
                if output_path.exists():
                    output_path.unlink()

        except subprocess.TimeoutExpired:
            raise RuntimeError("eSpeak-NG synthesis timed out")
        except Exception as e:
            raise RuntimeError(f"Synthesis failed: {e}")

    def get_model_info(self) -> TTSModelInfo:
        """Get model metadata."""
        if self.model_info_cache is None:
            self.model_info_cache = TTSModelInfo(
                name="espeak",
                display_name="eSpeak-NG",
                model_type="formant",
                architecture="Formant synthesis (rule-based)",
                language="en-US",
                voice_description="Robotic, highly intelligible",
                disk_size_mb=2.0,  # Very small - system binary
                sample_rate=22050,
                license="GPL-3.0"
            )

        return self.model_info_cache

    def unload(self) -> None:
        """Unload (nothing to unload for subprocess-based approach)."""
        self._is_loaded = False
