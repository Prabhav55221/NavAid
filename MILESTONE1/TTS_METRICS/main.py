#!/usr/bin/env python3
"""
Main entry point for TTS evaluation pipeline.

Orchestrates dataset download, preprocessing, synthesis, evaluation, and visualization.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import logging
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import download_datasets, preprocess_datasets, load_samples
from models import BaseTTS, create_model, list_available_models
from eval import compute_all_metrics, evaluate_wer_for_model, select_and_prepare_mos, generate_all_plots
import soundfile as sf
import json
from tqdm import tqdm


# Configure logging
def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


class TTSEvaluationPipeline:
    """Main pipeline for TTS evaluation."""

    def __init__(self, project_dir: Path):
        """
        Initialize pipeline.

        Args:
            project_dir: Root directory of the project
        """
        self.project_dir = project_dir
        self.data_dir = project_dir / "data"
        self.models_dir = project_dir / "models"
        self.outputs_dir = project_dir / "outputs"
        self.results_dir = project_dir / "results"

        # Create directories
        self.outputs_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        (self.results_dir / "plots").mkdir(exist_ok=True)
        (self.results_dir / "mos_samples").mkdir(exist_ok=True)

        self.logger = logging.getLogger(__name__)

    def run_download(self) -> bool:
        """
        Stage 1: Download datasets.

        Returns:
            True if successful
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 1: Dataset Download")
        self.logger.info("=" * 60)

        success = download_datasets(data_dir=self.data_dir / "raw")

        if success:
            self.logger.info("✓ Dataset download completed")
        else:
            self.logger.error("✗ Dataset download failed")

        return success

    def run_preprocess(self, n_samples: int = 100) -> bool:
        """
        Stage 2: Preprocess datasets and create test corpus.

        Args:
            n_samples: Number of samples to extract

        Returns:
            True if successful
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 2: Dataset Preprocessing")
        self.logger.info("=" * 60)

        success = preprocess_datasets(
            data_dir=self.data_dir / "raw",
            output_dir=self.data_dir,
            n_samples=n_samples
        )

        if success:
            self.logger.info("✓ Dataset preprocessing completed")
            samples_path = self.data_dir / "samples.json"
            if samples_path.exists():
                samples = load_samples(samples_path)
                self.logger.info(f"✓ Created {len(samples)} test samples")
        else:
            self.logger.error("✗ Dataset preprocessing failed")

        return success

    def run_synthesize(self, models: Optional[List[str]] = None,
                      skip_existing: bool = False) -> bool:
        """
        Stage 3: Synthesize audio for all models.

        Args:
            models: List of model names to synthesize (None = all)
            skip_existing: Skip if output already exists

        Returns:
            True if successful
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 3: Audio Synthesis")
        self.logger.info("=" * 60)

        # Load samples
        samples_path = self.data_dir / "samples.json"
        if not samples_path.exists():
            self.logger.error("samples.json not found. Run preprocess first.")
            return False

        samples = load_samples(samples_path)
        self.logger.info(f"Loaded {len(samples)} test samples")

        # Determine which models to use
        if models is None:
            models = list_available_models()

        self.logger.info(f"Models to evaluate: {', '.join(models)}")

        # Synthesize for each model
        all_success = True

        for model_name in models:
            self.logger.info(f"\n--- Synthesizing with {model_name} ---")

            try:
                # Create model
                model = create_model(model_name, cache_dir=self.models_dir / "cache")

                # Load model
                self.logger.info(f"Loading {model_name}...")
                model.load()

                # Create output directory
                model_output_dir = self.outputs_dir / model_name
                model_output_dir.mkdir(parents=True, exist_ok=True)

                # Synthesize each sample
                for sample in tqdm(samples, desc=f"Synthesizing {model_name}"):
                    sample_id = sample['id']
                    text = sample['text']
                    output_path = model_output_dir / f"{sample_id}.wav"

                    # Skip if exists
                    if skip_existing and output_path.exists():
                        continue

                    try:
                        # Synthesize
                        audio, sr = model.synthesize(text)

                        # Save audio
                        sf.write(output_path, audio, sr)

                    except Exception as e:
                        self.logger.warning(f"Failed to synthesize {sample_id}: {e}")
                        continue

                self.logger.info(f"✓ {model_name} synthesis complete")

                # Unload model to free memory
                model.unload()

            except Exception as e:
                self.logger.error(f"✗ {model_name} failed: {e}")
                all_success = False
                continue

        return all_success

    def run_evaluate(self) -> bool:
        """
        Stage 4: Compute evaluation metrics.

        Returns:
            True if successful
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 4: Metric Evaluation")
        self.logger.info("=" * 60)

        # Load samples
        samples_path = self.data_dir / "samples.json"
        if not samples_path.exists():
            self.logger.error("samples.json not found")
            return False

        samples = load_samples(samples_path)
        texts = [s['text'] for s in samples]

        # Get list of models to evaluate
        models = list_available_models()
        all_metrics = {}

        for model_name in models:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Evaluating: {model_name}")
            self.logger.info(f"{'='*60}")

            try:
                # Create and load model
                model = create_model(model_name, cache_dir=self.models_dir / "cache")
                model.load()

                # Compute all metrics (RTF + footprint)
                metrics = compute_all_metrics(
                    model, model_name, texts,
                    cache_dir=self.models_dir / "cache"
                )

                # Compute WER
                self.logger.info(f"Computing WER for {model_name}...")
                wer_metrics = evaluate_wer_for_model(model, texts, whisper_size="tiny")
                metrics['wer'] = wer_metrics

                all_metrics[model_name] = metrics

                self.logger.info(f"✓ {model_name} evaluation complete")

                # Unload model
                model.unload()

            except Exception as e:
                self.logger.error(f"✗ {model_name} evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save metrics to JSON
        metrics_path = self.results_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

        self.logger.info(f"\n✓ Metrics saved to {metrics_path}")

        return True

    def run_mos_selection(self, n_samples: int = 30) -> bool:
        """
        Stage 5: Select samples for MOS rating.

        Args:
            n_samples: Number of samples to select (6 per model for 5 models)

        Returns:
            True if successful
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 5: MOS Sample Selection")
        self.logger.info("=" * 60)

        # Load samples
        samples_path = self.data_dir / "samples.json"
        if not samples_path.exists():
            self.logger.error("samples.json not found")
            return False

        samples = load_samples(samples_path)

        # Collect all synthesized samples
        all_samples_data = []

        for model_name in list_available_models():
            model_output_dir = self.outputs_dir / model_name
            if not model_output_dir.exists():
                self.logger.warning(f"No outputs found for {model_name}")
                continue

            for sample in samples:
                audio_path = model_output_dir / f"{sample['id']}.wav"
                if audio_path.exists():
                    all_samples_data.append({
                        'model': model_name,
                        'sample_id': sample['id'],
                        'text': sample['text'],
                        'audio_path': str(audio_path)
                    })

        if not all_samples_data:
            self.logger.error("No synthesized samples found. Run synthesis first.")
            return False

        self.logger.info(f"Found {len(all_samples_data)} synthesized samples")

        # Select and prepare MOS samples
        mos_output_dir = self.results_dir / "mos_samples"
        success = select_and_prepare_mos(
            self.outputs_dir,
            all_samples_data,
            mos_output_dir,
            n_samples=n_samples
        )

        return success

    def run_visualize(self) -> bool:
        """
        Stage 6: Generate visualization plots.

        Returns:
            True if successful
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 6: Visualization")
        self.logger.info("=" * 60)

        metrics_path = self.results_dir / "metrics.json"
        plots_dir = self.results_dir / "plots"

        success = generate_all_plots(metrics_path, plots_dir)

        return success

    def run_all(self, n_samples: int = 100, models: Optional[List[str]] = None,
                skip_existing: bool = False) -> bool:
        """
        Run complete pipeline.

        Args:
            n_samples: Number of test samples
            models: List of models to evaluate (None = all)
            skip_existing: Skip existing outputs

        Returns:
            True if all stages successful
        """
        start_time = datetime.now()

        self.logger.info("\n" + "=" * 60)
        self.logger.info("TTS EVALUATION PIPELINE - FULL RUN")
        self.logger.info("=" * 60 + "\n")

        # Stage 1: Download
        if not self.run_download():
            return False

        # Stage 2: Preprocess
        if not self.run_preprocess(n_samples):
            return False

        # Stage 3: Synthesize
        if not self.run_synthesize(models, skip_existing):
            return False

        # Stage 4: Evaluate
        if not self.run_evaluate():
            return False

        # Stage 5: MOS selection
        if not self.run_mos_selection():
            return False

        # Stage 6: Visualize
        if not self.run_visualize():
            return False

        # Summary
        elapsed = datetime.now() - start_time
        self.logger.info("\n" + "=" * 60)
        self.logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info(f"✓ Total time: {elapsed}")
        self.logger.info("=" * 60 + "\n")

        self.logger.info("Results saved to:")
        self.logger.info(f"  - {self.results_dir / 'metrics.json'}")
        self.logger.info(f"  - {self.results_dir / 'comparison.csv'}")
        self.logger.info(f"  - {self.results_dir / 'summary.md'}")
        self.logger.info(f"  - {self.results_dir / 'plots/'}")
        self.logger.info(f"  - {self.results_dir / 'mos_samples/'}")

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TTS Evaluation Pipeline for NavAid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py --mode all

  # Run individual stages
  python main.py --mode download
  python main.py --mode preprocess
  python main.py --mode synthesize
  python main.py --mode evaluate
  python main.py --mode visualize

  # Test specific models only
  python main.py --mode all --models coqui_vits_ljspeech piper

  # Skip existing outputs
  python main.py --mode synthesize --skip-existing
        """
    )

    # Mode selection
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['all', 'download', 'preprocess', 'synthesize', 'evaluate', 'mos', 'visualize'],
        help='Pipeline stage to run'
    )

    # Options
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Models to evaluate (default: all)'
    )

    parser.add_argument(
        '--n-samples',
        type=int,
        default=100,
        help='Number of test samples to create (default: 100)'
    )

    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip synthesis if output already exists'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Create pipeline
    pipeline = TTSEvaluationPipeline(PROJECT_ROOT)

    # Run requested stage
    success = False

    if args.mode == 'all':
        success = pipeline.run_all(
            n_samples=args.n_samples,
            models=args.models,
            skip_existing=args.skip_existing
        )
    elif args.mode == 'download':
        success = pipeline.run_download()
    elif args.mode == 'preprocess':
        success = pipeline.run_preprocess(n_samples=args.n_samples)
    elif args.mode == 'synthesize':
        success = pipeline.run_synthesize(
            models=args.models,
            skip_existing=args.skip_existing
        )
    elif args.mode == 'evaluate':
        success = pipeline.run_evaluate()
    elif args.mode == 'mos':
        success = pipeline.run_mos_selection()
    elif args.mode == 'visualize':
        success = pipeline.run_visualize()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
