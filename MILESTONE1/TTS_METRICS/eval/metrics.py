"""
Core evaluation metrics: RTF, latency, and footprint measurement.
"""

import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import psutil
import os

from models import BaseTTS


class RTFMetrics:
    """Real-Time Factor and latency metrics."""

    @staticmethod
    def measure_synthesis(model: BaseTTS, text: str) -> Dict:
        """
        Measure single synthesis operation.

        Args:
            model: Loaded TTS model
            text: Text to synthesize

        Returns:
            Dictionary with timing metrics
        """
        # Measure synthesis time
        start_time = time.perf_counter()
        audio, sample_rate = model.synthesize(text)
        end_time = time.perf_counter()

        synthesis_time = end_time - start_time
        audio_duration = len(audio) / sample_rate
        rtf = synthesis_time / audio_duration if audio_duration > 0 else float('inf')

        return {
            'synthesis_time_ms': synthesis_time * 1000,
            'audio_duration_s': audio_duration,
            'rtf': rtf,
            'sample_rate': sample_rate,
            'audio_length': len(audio)
        }

    @staticmethod
    def measure_batch(model: BaseTTS, texts: List[str]) -> Dict:
        """
        Measure synthesis across multiple samples.

        Args:
            model: Loaded TTS model
            texts: List of texts to synthesize

        Returns:
            Aggregated metrics dictionary
        """
        results = []

        for text in texts:
            try:
                result = RTFMetrics.measure_synthesis(model, text)
                results.append(result)
            except Exception as e:
                print(f"Warning: Synthesis failed for text: {text[:50]}... Error: {e}")
                continue

        if not results:
            return {
                'count': 0,
                'rtf_median': None,
                'rtf_p95': None,
                'rtf_max': None,
                'rtf_mean': None,
                'synthesis_time_median_ms': None,
                'synthesis_time_p95_ms': None,
                'audio_duration_mean_s': None
            }

        rtfs = [r['rtf'] for r in results]
        synthesis_times = [r['synthesis_time_ms'] for r in results]
        audio_durations = [r['audio_duration_s'] for r in results]

        return {
            'count': len(results),
            'rtf_median': float(np.median(rtfs)),
            'rtf_p95': float(np.percentile(rtfs, 95)),
            'rtf_max': float(np.max(rtfs)),
            'rtf_mean': float(np.mean(rtfs)),
            'rtf_std': float(np.std(rtfs)),
            'synthesis_time_median_ms': float(np.median(synthesis_times)),
            'synthesis_time_p95_ms': float(np.percentile(synthesis_times, 95)),
            'synthesis_time_max_ms': float(np.max(synthesis_times)),
            'audio_duration_mean_s': float(np.mean(audio_durations)),
            'raw_rtfs': rtfs,  # For plotting
            'raw_synthesis_times': synthesis_times
        }


class FootprintMetrics:
    """Model size, memory, and cold start metrics."""

    @staticmethod
    def measure_disk_size(cache_dir: Path, model_name: str) -> float:
        """
        Measure disk space used by model.

        Args:
            cache_dir: Model cache directory
            model_name: Model identifier

        Returns:
            Size in MB
        """
        if not cache_dir.exists():
            return 0.0

        total_bytes = 0

        # Search for model files
        for file_path in cache_dir.rglob('*'):
            if file_path.is_file():
                # Check if file is related to this model
                if model_name.replace('_', '-') in str(file_path).lower():
                    total_bytes += file_path.stat().st_size

        # If no specific files found, try to estimate from entire cache
        if total_bytes == 0:
            # For Coqui models
            coqui_patterns = {
                'coqui_vits_ljspeech': 'tts_models--en--ljspeech--vits',
                'coqui_vits_vctk': 'tts_models--en--vctk--vits',
                'coqui_tacotron2': 'tts_models--en--ljspeech--tacotron2'
            }

            if model_name in coqui_patterns:
                pattern_dir = cache_dir / coqui_patterns[model_name]
                if pattern_dir.exists():
                    total_bytes = sum(
                        f.stat().st_size for f in pattern_dir.rglob('*') if f.is_file()
                    )

            # For Piper
            elif model_name == 'piper':
                piper_dir = cache_dir / 'piper_voices'
                if piper_dir.exists():
                    total_bytes = sum(
                        f.stat().st_size for f in piper_dir.rglob('*') if f.is_file()
                    )

        return total_bytes / (1024 * 1024)  # Convert to MB

    @staticmethod
    def measure_cold_start(model_class, cache_dir: Optional[Path] = None) -> float:
        """
        Measure cold start time (model loading).

        Args:
            model_class: TTS model class
            cache_dir: Cache directory

        Returns:
            Load time in milliseconds
        """
        model = model_class(cache_dir=cache_dir)

        start_time = time.perf_counter()
        model.load()
        end_time = time.perf_counter()

        model.unload()

        return (end_time - start_time) * 1000  # Convert to ms

    @staticmethod
    def measure_peak_memory(model: BaseTTS, sample_text: str) -> Dict:
        """
        Measure peak memory usage during synthesis.

        Args:
            model: Loaded TTS model
            sample_text: Text to synthesize

        Returns:
            Memory metrics dictionary
        """
        process = psutil.Process(os.getpid())

        # Get baseline memory
        baseline_mem = process.memory_info().rss / (1024 * 1024)  # MB

        # Track memory during synthesis
        tracemalloc.start()

        try:
            # Perform synthesis
            audio, sr = model.synthesize(sample_text)

            # Get peak memory
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Get process memory after synthesis
            post_mem = process.memory_info().rss / (1024 * 1024)  # MB

            return {
                'baseline_mb': baseline_mem,
                'peak_mb': post_mem,
                'delta_mb': post_mem - baseline_mem,
                'tracemalloc_peak_mb': peak / (1024 * 1024)
            }

        except Exception as e:
            tracemalloc.stop()
            raise e

    @staticmethod
    def measure_all_footprint(model: BaseTTS, model_name: str,
                             sample_text: str, cache_dir: Path) -> Dict:
        """
        Measure all footprint metrics for a model.

        Args:
            model: TTS model instance (can be loaded or unloaded)
            model_name: Model identifier
            sample_text: Text for memory measurement
            cache_dir: Cache directory

        Returns:
            Complete footprint metrics
        """
        # Disk size
        disk_size_mb = FootprintMetrics.measure_disk_size(cache_dir, model_name)

        # Cold start time (if model not loaded)
        if not model.is_loaded():
            cold_start_ms = FootprintMetrics.measure_cold_start(
                type(model), cache_dir
            )
            # Reload after cold start measurement
            model.load()
        else:
            # Model already loaded, estimate from info
            cold_start_ms = None

        # Peak memory
        try:
            memory_metrics = FootprintMetrics.measure_peak_memory(model, sample_text)
        except Exception as e:
            print(f"Warning: Memory measurement failed: {e}")
            memory_metrics = {
                'baseline_mb': 0,
                'peak_mb': 0,
                'delta_mb': 0,
                'tracemalloc_peak_mb': 0
            }

        return {
            'disk_size_mb': disk_size_mb,
            'cold_start_ms': cold_start_ms,
            'memory_baseline_mb': memory_metrics['baseline_mb'],
            'memory_peak_mb': memory_metrics['peak_mb'],
            'memory_delta_mb': memory_metrics['delta_mb']
        }


def compute_all_metrics(model: BaseTTS, model_name: str,
                       texts: List[str], cache_dir: Path) -> Dict:
    """
    Compute all metrics for a model.

    Args:
        model: Loaded TTS model
        model_name: Model identifier
        texts: List of test texts
        cache_dir: Cache directory

    Returns:
        Complete metrics dictionary
    """
    print(f"\n--- Computing metrics for {model_name} ---")

    # RTF metrics
    print("  Measuring RTF across samples...")
    rtf_metrics = RTFMetrics.measure_batch(model, texts)

    # Footprint metrics
    print("  Measuring footprint...")
    sample_text = texts[0] if texts else "Turn left at the next intersection."
    footprint_metrics = FootprintMetrics.measure_all_footprint(
        model, model_name, sample_text, cache_dir
    )

    # Combine
    return {
        'rtf': rtf_metrics,
        'footprint': footprint_metrics
    }
