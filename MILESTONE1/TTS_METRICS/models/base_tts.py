"""
Abstract base class for TTS models.

Defines the interface that all TTS implementations must follow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import numpy as np


@dataclass
class TTSModelInfo:
    """Metadata about a TTS model."""

    name: str                       # Model identifier (e.g., "coqui_vits_ljspeech")
    display_name: str               # Human-readable name
    model_type: str                 # Type (e.g., "neural", "formant")
    architecture: str               # Architecture (e.g., "VITS", "Tacotron2", "formant")
    language: str = "en"            # Language code
    voice_description: str = ""     # Description of voice (e.g., "Female, US English")
    disk_size_mb: float = 0.0       # Model size on disk (MB)
    sample_rate: int = 22050        # Audio sample rate (Hz)
    license: str = "Unknown"        # License information


class BaseTTS(ABC):
    """
    Abstract base class for TTS engines.

    All TTS implementations must inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize TTS model.

        Args:
            cache_dir: Directory for caching model weights
        """
        self.cache_dir = cache_dir or Path(__file__).parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """
        Load model into memory.

        This method should:
        1. Download model weights if not cached
        2. Load model into memory
        3. Perform any necessary initialization
        4. Set self._is_loaded = True

        Raises:
            RuntimeError: If model fails to load
        """
        pass

    @abstractmethod
    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech from text.

        Args:
            text: Input text to synthesize

        Returns:
            Tuple of (audio_samples, sample_rate)
            - audio_samples: 1D numpy array of float32 audio samples
            - sample_rate: Sample rate in Hz (e.g., 22050)

        Raises:
            RuntimeError: If model not loaded or synthesis fails
        """
        pass

    @abstractmethod
    def get_model_info(self) -> TTSModelInfo:
        """
        Get model metadata.

        Returns:
            TTSModelInfo object with model details

        Note:
            This can be called before load() to get static information.
        """
        pass

    def is_loaded(self) -> bool:
        """
        Check if model is loaded into memory.

        Returns:
            True if model loaded, False otherwise
        """
        return self._is_loaded

    def unload(self) -> None:
        """
        Unload model from memory (optional, for memory management).

        Default implementation does nothing. Subclasses can override
        to free GPU/CPU memory.
        """
        self._is_loaded = False

    def _validate_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """
        Validate synthesized audio output.

        Args:
            audio: Audio samples array
            sample_rate: Sample rate

        Raises:
            ValueError: If audio format is invalid
        """
        if not isinstance(audio, np.ndarray):
            raise ValueError(f"Audio must be numpy array, got {type(audio)}")

        if audio.dtype != np.float32:
            raise ValueError(f"Audio must be float32, got {audio.dtype}")

        if audio.ndim != 1:
            raise ValueError(f"Audio must be 1D array, got shape {audio.shape}")

        if len(audio) == 0:
            raise ValueError("Audio array is empty")

        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")

        # Check for NaN or Inf
        if not np.isfinite(audio).all():
            raise ValueError("Audio contains NaN or Inf values")

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.

        Args:
            audio: Input audio array

        Returns:
            Normalized audio array
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio

    def get_cache_path(self, filename: str) -> Path:
        """
        Get path for cached model file.

        Args:
            filename: Name of cached file

        Returns:
            Full path in cache directory
        """
        return self.cache_dir / filename

    def __repr__(self) -> str:
        """String representation of model."""
        info = self.get_model_info()
        loaded_str = "loaded" if self._is_loaded else "not loaded"
        return f"<{info.display_name} ({loaded_str})>"


class DummyTTS(BaseTTS):
    """
    Dummy TTS implementation for testing.

    Generates silence instead of actual speech.
    """

    def load(self) -> None:
        """Load dummy model (instant)."""
        self._is_loaded = True

    def synthesize(self, text: str) -> Tuple[np.ndarray, int]:
        """Generate silence of appropriate duration."""
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Generate ~1 second of silence per 10 words (rough estimate)
        word_count = len(text.split())
        duration_seconds = max(0.5, word_count * 0.1)
        sample_rate = 22050

        n_samples = int(duration_seconds * sample_rate)
        audio = np.zeros(n_samples, dtype=np.float32)

        return audio, sample_rate

    def get_model_info(self) -> TTSModelInfo:
        """Get dummy model info."""
        return TTSModelInfo(
            name="dummy",
            display_name="Dummy TTS (Testing)",
            model_type="test",
            architecture="silence",
            voice_description="No voice (silence)",
            disk_size_mb=0.001,
            sample_rate=22050,
            license="Public Domain"
        )


# Utility functions

def list_available_models() -> list:
    """
    List all available TTS model classes.

    Returns:
        List of model class names (to be populated as models are implemented)
    """
    # This will be populated as we implement models
    # For now, just return the dummy
    return ["DummyTTS"]


def create_model(model_name: str, cache_dir: Optional[Path] = None) -> BaseTTS:
    """
    Factory function to create TTS model instances.

    Args:
        model_name: Name of model to create
        cache_dir: Cache directory for model weights

    Returns:
        Instance of requested TTS model

    Raises:
        ValueError: If model_name not recognized
    """
    # Map of model names to classes (will be expanded as we implement models)
    MODEL_REGISTRY = {
        "dummy": DummyTTS,
    }

    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    model_class = MODEL_REGISTRY[model_name]
    return model_class(cache_dir=cache_dir)
