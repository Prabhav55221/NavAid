"""
TTS models module.

Provides abstract interface and implementations for various TTS engines.
"""

from .base_tts import BaseTTS, TTSModelInfo

__all__ = ['BaseTTS', 'TTSModelInfo']
