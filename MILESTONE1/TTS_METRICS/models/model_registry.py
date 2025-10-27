"""
Model registry for all TTS implementations.

Provides factory functions to create and list available models.
"""

from pathlib import Path
from typing import Dict, List, Type, Optional

from .base_tts import BaseTTS
from .coqui_vits_ljspeech import CoquiVITSLJSpeech
from .coqui_vits_vctk import CoquiVITSVCTK
from .coqui_tacotron2 import CoquiTacotron2
from .piper_tts import PiperTTS
from .espeak_tts import ESpeakTTS


# Registry mapping model names to classes
MODEL_REGISTRY: Dict[str, Type[BaseTTS]] = {
    "coqui_vits_ljspeech": CoquiVITSLJSpeech,
    "coqui_vits_vctk": CoquiVITSVCTK,
    "coqui_tacotron2": CoquiTacotron2,
    "piper": PiperTTS,
    "espeak": ESpeakTTS,
}


def list_available_models() -> List[str]:
    """
    List all available TTS model names.

    Returns:
        List of model identifier strings
    """
    return list(MODEL_REGISTRY.keys())


def get_model_info_all() -> Dict[str, dict]:
    """
    Get metadata for all models without loading them.

    Returns:
        Dictionary mapping model names to their info dictionaries
    """
    info = {}
    for name, model_class in MODEL_REGISTRY.items():
        # Create temporary instance to get info
        temp_instance = model_class()
        model_info = temp_instance.get_model_info()
        info[name] = {
            'name': model_info.name,
            'display_name': model_info.display_name,
            'model_type': model_info.model_type,
            'architecture': model_info.architecture,
            'voice_description': model_info.voice_description,
        }
    return info


def create_model(model_name: str, cache_dir: Optional[Path] = None) -> BaseTTS:
    """
    Factory function to create TTS model instances.

    Args:
        model_name: Name of model to create (see list_available_models())
        cache_dir: Optional cache directory for model weights

    Returns:
        Instance of requested TTS model

    Raises:
        ValueError: If model_name not recognized

    Example:
        >>> model = create_model("coqui_vits_ljspeech")
        >>> model.load()
        >>> audio, sr = model.synthesize("Turn left ahead")
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {available}"
        )

    model_class = MODEL_REGISTRY[model_name]
    return model_class(cache_dir=cache_dir)


def create_all_models(cache_dir: Optional[Path] = None) -> Dict[str, BaseTTS]:
    """
    Create instances of all available models.

    Args:
        cache_dir: Optional cache directory for model weights

    Returns:
        Dictionary mapping model names to model instances

    Example:
        >>> models = create_all_models()
        >>> for name, model in models.items():
        ...     model.load()
        ...     print(f"Loaded: {name}")
    """
    models = {}
    for name in MODEL_REGISTRY.keys():
        models[name] = create_model(name, cache_dir)
    return models


def print_model_summary():
    """Print a summary table of all available models."""
    print("\n" + "=" * 80)
    print("AVAILABLE TTS MODELS")
    print("=" * 80)
    print(f"{'Name':<25} {'Type':<10} {'Architecture':<20} {'Voice':<25}")
    print("-" * 80)

    for name, model_class in MODEL_REGISTRY.items():
        temp_instance = model_class()
        info = temp_instance.get_model_info()
        print(f"{info.display_name:<25} {info.model_type:<10} "
              f"{info.architecture:<20} {info.voice_description[:24]:<25}")

    print("=" * 80)
    print(f"Total models: {len(MODEL_REGISTRY)}\n")


if __name__ == "__main__":
    # Print summary when run directly
    print_model_summary()
