"""
Data module for TTS evaluation.

Handles dataset download, preprocessing, and sample management.
"""

from .download import download_datasets
from .preprocess import preprocess_datasets, load_samples

__all__ = ['download_datasets', 'preprocess_datasets', 'load_samples']
