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
from models import BaseTTS


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

        # This will be implemented in Day 2-3 when we have model implementations
        self.logger.warning("⚠ Synthesis not yet implemented (Day 2-3)")
        self.logger.info("   Models to implement:")
        self.logger.info("   - coqui_vits_ljspeech")
        self.logger.info("   - coqui_vits_vctk")
        self.logger.info("   - coqui_tacotron")
        self.logger.info("   - piper")
        self.logger.info("   - espeak")

        return True

    def run_evaluate(self) -> bool:
        """
        Stage 4: Compute evaluation metrics.

        Returns:
            True if successful
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 4: Metric Evaluation")
        self.logger.info("=" * 60)

        # This will be implemented in Day 4 when we have evaluation code
        self.logger.warning("⚠ Evaluation not yet implemented (Day 4)")
        self.logger.info("   Metrics to compute:")
        self.logger.info("   - Real-Time Factor (RTF)")
        self.logger.info("   - Word Error Rate (WER) via Whisper")
        self.logger.info("   - Model footprint (disk, RAM, cold start)")

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

        # This will be implemented in Day 4
        self.logger.warning("⚠ MOS selection not yet implemented (Day 4)")
        self.logger.info(f"   Will select {n_samples} samples for human rating")

        return True

    def run_visualize(self) -> bool:
        """
        Stage 6: Generate visualization plots.

        Returns:
            True if successful
        """
        self.logger.info("=" * 60)
        self.logger.info("STAGE 6: Visualization")
        self.logger.info("=" * 60)

        # This will be implemented in Day 4
        self.logger.warning("⚠ Visualization not yet implemented (Day 4)")
        self.logger.info("   Plots to generate:")
        self.logger.info("   - RTF comparison (bar chart)")
        self.logger.info("   - WER comparison (bar chart)")
        self.logger.info("   - Latency distribution (box plot)")
        self.logger.info("   - Footprint comparison (grouped bar)")

        return True

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
