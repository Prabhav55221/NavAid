"""
Word Error Rate (WER) evaluation via Whisper ASR.

Performs ASR round-trip: TTS → audio → Whisper → text → compare to original.
"""

from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import soundfile as sf
import tempfile
from jiwer import wer, cer


class WhisperEvaluator:
    """Evaluate TTS intelligibility using Whisper ASR."""

    def __init__(self, model_size: str = "tiny"):
        """
        Initialize Whisper evaluator.

        Args:
            model_size: Whisper model size ("tiny", "base", "small", etc.)
                       tiny is fastest, good enough for relative comparison
        """
        self.model_size = model_size
        self.model = None

    def load(self):
        """Load Whisper model."""
        if self.model is not None:
            return

        try:
            import whisper
            print(f"Loading Whisper model: {self.model_size}...")
            self.model = whisper.load_model(self.model_size)
            print(f"✓ Whisper {self.model_size} loaded")

        except ImportError:
            raise RuntimeError(
                "Whisper not installed. Install with: pip install openai-whisper"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper: {e}")

    def transcribe_audio(self, audio: np.ndarray, sample_rate: int) -> str:
        """
        Transcribe audio using Whisper.

        Args:
            audio: Audio samples (float32, 1D)
            sample_rate: Sample rate in Hz

        Returns:
            Transcribed text
        """
        if self.model is None:
            raise RuntimeError("Whisper model not loaded. Call load() first.")

        try:
            # Whisper expects 16kHz audio
            if sample_rate != 16000:
                # Resample to 16kHz
                import librosa
                audio = librosa.resample(
                    audio,
                    orig_sr=sample_rate,
                    target_sr=16000
                )

            # Transcribe
            result = self.model.transcribe(
                audio,
                language='en',
                fp16=False  # Use FP32 for CPU
            )

            return result['text'].strip()

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}")

    def compute_wer(self, reference: str, hypothesis: str) -> Dict:
        """
        Compute Word Error Rate and Character Error Rate.

        Args:
            reference: Original text
            hypothesis: Transcribed text

        Returns:
            Dictionary with WER and CER metrics
        """
        # Normalize texts (lowercase, strip)
        ref_normalized = reference.lower().strip()
        hyp_normalized = hypothesis.lower().strip()

        # Compute WER and CER
        try:
            wer_score = wer(ref_normalized, hyp_normalized)
            cer_score = cer(ref_normalized, hyp_normalized)

            return {
                'wer': wer_score * 100,  # Convert to percentage
                'cer': cer_score * 100,
                'reference': reference,
                'hypothesis': hypothesis,
                'reference_normalized': ref_normalized,
                'hypothesis_normalized': hyp_normalized
            }

        except Exception as e:
            print(f"Warning: WER computation failed: {e}")
            return {
                'wer': 100.0,  # Worst case
                'cer': 100.0,
                'reference': reference,
                'hypothesis': hypothesis,
                'reference_normalized': ref_normalized,
                'hypothesis_normalized': hyp_normalized
            }

    def evaluate_tts_output(self, audio: np.ndarray, sample_rate: int,
                           original_text: str) -> Dict:
        """
        Full evaluation: transcribe audio and compute WER.

        Args:
            audio: Synthesized audio
            sample_rate: Audio sample rate
            original_text: Original text that was synthesized

        Returns:
            Dictionary with transcription and WER metrics
        """
        # Transcribe
        transcription = self.transcribe_audio(audio, sample_rate)

        # Compute WER
        wer_metrics = self.compute_wer(original_text, transcription)

        return {
            'transcription': transcription,
            'wer': wer_metrics['wer'],
            'cer': wer_metrics['cer'],
            'reference': original_text,
            'hypothesis': transcription
        }

    def evaluate_batch(self, audio_samples: List[Tuple[np.ndarray, int]],
                      original_texts: List[str]) -> Dict:
        """
        Evaluate multiple TTS outputs.

        Args:
            audio_samples: List of (audio, sample_rate) tuples
            original_texts: Corresponding original texts

        Returns:
            Aggregated WER metrics
        """
        if len(audio_samples) != len(original_texts):
            raise ValueError("Number of audio samples must match number of texts")

        results = []

        for (audio, sample_rate), text in zip(audio_samples, original_texts):
            try:
                result = self.evaluate_tts_output(audio, sample_rate, text)
                results.append(result)
            except Exception as e:
                print(f"Warning: Evaluation failed for text: {text[:50]}... Error: {e}")
                continue

        if not results:
            return {
                'count': 0,
                'wer_mean': None,
                'wer_std': None,
                'wer_min': None,
                'wer_max': None,
                'cer_mean': None
            }

        wers = [r['wer'] for r in results]
        cers = [r['cer'] for r in results]

        return {
            'count': len(results),
            'wer_mean': float(np.mean(wers)),
            'wer_std': float(np.std(wers)),
            'wer_min': float(np.min(wers)),
            'wer_max': float(np.max(wers)),
            'wer_median': float(np.median(wers)),
            'cer_mean': float(np.mean(cers)),
            'cer_std': float(np.std(cers)),
            'raw_wers': wers,  # For plotting
            'raw_cers': cers,
            'examples': results[:5]  # Save first 5 examples
        }


def evaluate_wer_for_model(model, texts: List[str],
                           whisper_size: str = "tiny") -> Dict:
    """
    Convenience function to evaluate WER for a TTS model.

    Args:
        model: Loaded TTS model
        texts: List of test texts
        whisper_size: Whisper model size

    Returns:
        WER metrics dictionary
    """
    print(f"  Evaluating WER with Whisper ({whisper_size})...")

    evaluator = WhisperEvaluator(model_size=whisper_size)
    evaluator.load()

    audio_samples = []

    # Synthesize all texts
    for text in texts:
        try:
            audio, sr = model.synthesize(text)
            audio_samples.append((audio, sr))
        except Exception as e:
            print(f"Warning: Synthesis failed for: {text[:50]}... Error: {e}")
            continue

    # Evaluate
    wer_metrics = evaluator.evaluate_batch(audio_samples, texts)

    return wer_metrics
