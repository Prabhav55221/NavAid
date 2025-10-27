"""
Piper TTS - Fast ONNX-optimized neural TTS.

Designed for edge devices and assistive technology.
Uses piper_tts Python library directly (not CLI binary).
"""

import io
import wave
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

from .base_tts import BaseTTS, TTSModelInfo


class PiperTTS(BaseTTS):
    """Piper TTS with ONNX runtime via Python API."""

    # Using en_US-lessac-medium voice (balanced quality/speed)
    VOICE_NAME = "en_US-lessac-medium"

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize Piper TTS."""
        super().__init__(cache_dir)
        self.voice = None
        self.voice_model_path = None
        self.voice_config_path = None
        self.model_info_cache = None

    def _download_voice_model(self) -> None:
        """Download Piper voice model if not cached."""
        import requests

        voice_dir = self.cache_dir / "piper_voices" / self.VOICE_NAME
        voice_dir.mkdir(parents=True, exist_ok=True)

        model_file = voice_dir / f"{self.VOICE_NAME}.onnx"
        config_file = voice_dir / f"{self.VOICE_NAME}.onnx.json"

        # Base URL for Piper voice models
        base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium"

        if not model_file.exists():
            print(f"Downloading Piper voice model: {self.VOICE_NAME}...")
            model_url = f"{base_url}/en_US-lessac-medium.onnx"
            response = requests.get(model_url, timeout=300)
            response.raise_for_status()
            model_file.write_bytes(response.content)
            print(f"✓ Downloaded model to {model_file}")

        if not config_file.exists():
            print(f"Downloading Piper voice config...")
            config_url = f"{base_url}/en_US-lessac-medium.onnx.json"
            response = requests.get(config_url, timeout=60)
            response.raise_for_status()
            config_file.write_bytes(response.content)
            print(f"✓ Downloaded config to {config_file}")

        self.voice_model_path = model_file
        self.voice_config_path = config_file

    def load(self) -> None:
        """Load Piper TTS via Python library."""
        if self._is_loaded:
            return

        try:
            # Import piper_tts
            try:
                from piper import PiperVoice
            except ImportError:
                raise RuntimeError(
                    "piper-tts not installed. Install with: pip install piper-tts"
                )

            # Download voice model if needed
            self._download_voice_model()

            # Load voice
            print(f"Loading Piper voice: {self.VOICE_NAME}...")
            self.voice = PiperVoice.load(
                str(self.voice_model_path),
                config_path=str(self.voice_config_path)
            )

            self._is_loaded = True
            print(f"✓ Piper TTS ready with voice: {self.VOICE_NAME}")

        except Exception as e:
            raise RuntimeError(f"Failed to load Piper TTS: {e}")

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text using Piper Python API.

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
            # Try multiple API methods (piper-tts API varies by version)
            audio_samples = []

            # Method 1: synthesize_stream_raw (some versions)
            if hasattr(self.voice, 'synthesize_stream_raw'):
                for audio_chunk in self.voice.synthesize_stream_raw(text):
                    audio_samples.extend(audio_chunk)

            # Method 2: synthesize (common method)
            elif hasattr(self.voice, 'synthesize'):
                wav_bytes = io.BytesIO()
                self.voice.synthesize(text, wav_bytes)

                # Read WAV data
                wav_bytes.seek(0)
                import wave
                with wave.open(wav_bytes, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    audio_bytes = wav_file.readframes(wav_file.getnframes())
                    audio_samples = np.frombuffer(audio_bytes, dtype=np.int16)

            # Method 3: Direct call (if voice is callable)
            else:
                # Fallback to __call__
                wav_bytes = io.BytesIO()
                self.voice(text, wav_bytes)

                wav_bytes.seek(0)
                import wave
                with wave.open(wav_bytes, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    audio_bytes = wav_file.readframes(wav_file.getnframes())
                    audio_samples = np.frombuffer(audio_bytes, dtype=np.int16)

            # Convert to numpy array if needed
            if not isinstance(audio_samples, np.ndarray):
                audio = np.array(audio_samples, dtype=np.int16)
            else:
                audio = audio_samples

            # Convert to float32 [-1, 1]
            audio = audio.astype(np.float32) / 32768.0

            # Get sample rate from voice config
            if not 'sample_rate' in locals():
                sample_rate = self.voice.config.sample_rate

            # Normalize
            audio = self._normalize_audio(audio)

            # Validate
            self._validate_audio(audio, sample_rate)

            return audio, sample_rate

        except Exception as e:
            raise RuntimeError(f"Synthesis failed: {e}")

    def get_model_info(self) -> TTSModelInfo:
        """Get model metadata."""
        if self.model_info_cache is None:
            disk_size_mb = 0.0
            if self.voice_model_path and self.voice_model_path.exists():
                disk_size_mb = self.voice_model_path.stat().st_size / (1024 * 1024)
            else:
                disk_size_mb = 50.0  # Approximate

            self.model_info_cache = TTSModelInfo(
                name="piper",
                display_name="Piper TTS (ONNX)",
                model_type="neural",
                architecture="VITS (ONNX quantized)",
                language="en-US",
                voice_description="Lessac voice (optimized for speed)",
                disk_size_mb=disk_size_mb,
                sample_rate=22050,
                license="MIT"
            )

        return self.model_info_cache

    def unload(self) -> None:
        """Unload model (nothing to unload for subprocess-based approach)."""
        self._is_loaded = False
