#!/usr/bin/env python3
"""
Visualization for hazard detection evaluation results.

Creates:
1. Confusion Matrix
2. Precision/Recall/F1 comparison
3. Per-type performance (bar chart)
4. Confidence distribution (correct vs incorrect)
5. Latency distribution (histogram + percentiles)
6. CHMR gauge (safety metric)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class EvaluationVisualizer:
    """Create visualizations from evaluation results."""

    def __init__(self, results_path: Path):
        """Load evaluation results."""
        with open(results_path, 'r') as f:
            self.results = json.load(f)

        self.detection = self.results['detection_performance']
        self.latency = self.results.get('latency_performance', {})

    def plot_confusion_matrix(self, ax=None):
        """Plot confusion matrix as heatmap."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))

        bm = self.detection['binary_metrics']
        confusion = np.array([
            [bm['tp'], bm['fp']],
            [bm['fn'], bm['tn']]
        ])

        # Create heatmap
        sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
                    cbar_kws={'label': 'Count'},
                    xticklabels=['Predicted Hazard', 'Predicted No Hazard'],
                    yticklabels=['Actual Hazard', 'Actual No Hazard'],
                    ax=ax, vmin=0, linewidths=1, linecolor='black')

        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('Actual Label', fontsize=12)

        return ax

    def plot_metrics_comparison(self, ax=None):
        """Plot Precision, Recall, F1, Accuracy as bar chart."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        bm = self.detection['binary_metrics']
        metrics = {
            'Precision': bm['precision'],
            'Recall': bm['recall'],
            'F1-Score': bm['f1'],
            'Accuracy': bm['accuracy']
        }

        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        bars = ax.bar(metrics.keys(), [v * 100 for v in metrics.values()],
                      color=colors, alpha=0.8, edgecolor='black')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # Add target line at 80%
        ax.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target (80%)')

        ax.set_ylim(0, 105)
        ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('Detection Performance Metrics', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(axis='y', alpha=0.3)

        return ax

    def plot_per_type_performance(self, ax=None):
        """Plot per-hazard-type Precision and Recall."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        type_metrics = self.detection['type_metrics']

        # Filter to types with ground truth data
        types_with_gt = {k: v for k, v in type_metrics.items() if v['ground_truth_count'] > 0}

        if not types_with_gt:
            ax.text(0.5, 0.5, 'No type-level metrics available',
                   ha='center', va='center', fontsize=14)
            return ax

        # Sort by ground truth count (most common first)
        sorted_types = sorted(types_with_gt.items(),
                            key=lambda x: x[1]['ground_truth_count'],
                            reverse=True)

        types = [t[0] for t in sorted_types]
        precisions = [t[1]['precision'] * 100 for t in sorted_types]
        recalls = [t[1]['recall'] * 100 for t in sorted_types]
        f1s = [t[1]['f1'] * 100 for t in sorted_types]
        counts = [t[1]['ground_truth_count'] for t in sorted_types]

        x = np.arange(len(types))
        width = 0.25

        # Create bars
        bars1 = ax.bar(x - width, precisions, width, label='Precision',
                      color='#3498db', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x, recalls, width, label='Recall',
                      color='#2ecc71', alpha=0.8, edgecolor='black')
        bars3 = ax.bar(x + width, f1s, width, label='F1-Score',
                      color='#e74c3c', alpha=0.8, edgecolor='black')

        ax.set_xlabel('Hazard Type', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
        ax.set_title('Per-Type Detection Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}\n(n={c})" for t, c in zip(types, counts)], rotation=0)
        ax.legend(loc='upper right')
        ax.set_ylim(0, 105)
        ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(axis='y', alpha=0.3)

        return ax

    def plot_confidence_distribution(self, ax=None):
        """Plot confidence distribution for correct vs incorrect predictions."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        cs = self.detection['confidence_stats']

        # Get individual confidences from error cases if available
        error_cases = self.detection.get('error_analysis', {}).get('error_cases', [])

        # Approximate distributions based on means and stds
        mean_correct = cs['mean_confidence']
        mean_incorrect = cs['mean_confidence_incorrect']
        std = cs['std_confidence']

        # If we have error cases, use actual confidences
        incorrect_confs = [e['confidence'] for e in error_cases]

        # Generate approximate data for correct (since we don't store individual correct confidences)
        n_correct = cs['num_correct']
        correct_confs = np.random.normal(mean_correct, std, n_correct)
        correct_confs = np.clip(correct_confs, 0, 1)  # Clip to [0, 1]

        # Plot histograms
        bins = np.linspace(0, 1, 21)
        ax.hist(correct_confs, bins=bins, alpha=0.6, label='Correct Predictions',
               color='#2ecc71', edgecolor='black')
        if incorrect_confs:
            ax.hist(incorrect_confs, bins=bins, alpha=0.6, label='Incorrect Predictions',
                   color='#e74c3c', edgecolor='black')

        # Add mean lines
        ax.axvline(mean_correct, color='#27ae60', linestyle='--', linewidth=2,
                  label=f'Mean (Correct): {mean_correct:.2%}')
        if incorrect_confs:
            ax.axvline(mean_incorrect, color='#c0392b', linestyle='--', linewidth=2,
                      label=f'Mean (Incorrect): {mean_incorrect:.2%}')

        ax.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        return ax

    def plot_latency_distribution(self, ax=None):
        """Plot latency distribution with percentiles."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))

        lat = self.latency.get('aggregate_stats', {})
        if not lat:
            ax.text(0.5, 0.5, 'No latency data available',
                   ha='center', va='center', fontsize=14)
            return ax

        # Key metrics
        mean = lat.get('mean', 0)
        median = lat.get('median', 0)
        p95 = lat.get('p95', 0)
        p99 = lat.get('p99', 0)
        min_lat = lat.get('min', 0)
        max_lat = lat.get('max', 0)

        # Create box-like visualization
        metrics = {
            'Min': min_lat,
            'Median\n(P50)': median,
            'Mean': mean,
            'P95': p95,
            'P99': p99,
            'Max': max_lat
        }

        colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#c0392b', '#8e44ad']
        bars = ax.bar(metrics.keys(), metrics.values(), color=colors,
                     alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 20,
                   f'{int(height)}ms', ha='center', va='bottom',
                   fontsize=10, fontweight='bold')

        # Add target lines
        ax.axhline(y=2000, color='orange', linestyle='--', linewidth=2,
                  alpha=0.7, label='Acceptable (2000ms)')
        ax.axhline(y=1000, color='green', linestyle='--', linewidth=2,
                  alpha=0.7, label='Excellent (1000ms)')

        ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
        ax.set_title('Latency Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(axis='y', alpha=0.3)

        return ax

    def plot_chmr_gauge(self, ax=None):
        """Plot CHMR as a gauge/speedometer chart."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        chmr = self.detection['binary_metrics']['chmr'] * 100

        # Create gauge
        ax.barh([0], [chmr], height=0.5, color='#e74c3c' if chmr > 5 else '#2ecc71',
               alpha=0.8, edgecolor='black', linewidth=2)
        ax.barh([0], [100 - chmr], height=0.5, left=chmr, color='#ecf0f1',
               alpha=0.5, edgecolor='black', linewidth=2)

        # Add markers
        ax.axvline(x=5, color='orange', linestyle='--', linewidth=2, label='Target (5%)')
        ax.axvline(x=10, color='red', linestyle='--', linewidth=2, label='Unsafe (10%)')

        # Add text
        status = '✅ SAFE' if chmr < 5 else '⚠️ MARGINAL' if chmr < 10 else '❌ UNSAFE'
        ax.text(chmr/2, 0, f'{chmr:.1f}%\n{status}',
               ha='center', va='center', fontsize=14, fontweight='bold')

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Critical Hazard Miss Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('🚨 CHMR - Safety Metric', fontsize=14, fontweight='bold')
        ax.set_yticks([])
        ax.legend(loc='upper right')
        ax.grid(axis='x', alpha=0.3)

        return ax

    def create_full_dashboard(self, output_path: Path):
        """Create comprehensive dashboard with all plots."""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Row 1: Confusion Matrix, Metrics Comparison, CHMR Gauge
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_confusion_matrix(ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_metrics_comparison(ax2)

        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_chmr_gauge(ax3)

        # Row 2: Per-Type Performance (spans 2 columns), Confidence Distribution
        ax4 = fig.add_subplot(gs[1, :2])
        self.plot_per_type_performance(ax4)

        ax5 = fig.add_subplot(gs[1, 2])
        self.plot_confidence_distribution(ax5)

        # Row 3: Latency Distribution (spans all columns)
        ax6 = fig.add_subplot(gs[2, :])
        self.plot_latency_distribution(ax6)

        # Add overall title
        fig.suptitle('NavAid Hazard Detection - Full Evaluation Dashboard',
                    fontsize=18, fontweight='bold', y=0.98)

        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dashboard saved to: {output_path}")

        return fig

    def create_individual_plots(self, output_dir: Path):
        """Create individual plot files."""
        output_dir.mkdir(parents=True, exist_ok=True)

        plots = [
            ('confusion_matrix.png', self.plot_confusion_matrix),
            ('metrics_comparison.png', self.plot_metrics_comparison),
            ('per_type_performance.png', self.plot_per_type_performance),
            ('confidence_distribution.png', self.plot_confidence_distribution),
            ('latency_distribution.png', self.plot_latency_distribution),
            ('chmr_gauge.png', self.plot_chmr_gauge)
        ]

        for filename, plot_func in plots:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_func(ax)
            output_path = output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ Saved: {filename}")

        print(f"\n✓ All individual plots saved to: {output_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Create visualizations from evaluation results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create full dashboard
  python eval/visualize.py -i results/evaluation_v2.0.json -o results/plots/dashboard.png

  # Create individual plots
  python eval/visualize.py -i results/evaluation_v2.0.json --individual -o results/plots
        """
    )
    ap.add_argument("--input", "-i", type=Path, required=True,
                    help="Path to evaluation JSON (e.g., results/evaluation_v2.0.json)")
    ap.add_argument("--output", "-o", type=Path, required=True,
                    help="Output path (file for dashboard, directory for individual plots)")
    ap.add_argument("--individual", action="store_true",
                    help="Create individual plot files instead of dashboard")
    args = ap.parse_args()

    # Validate input
    if not args.input.exists():
        raise FileNotFoundError(f"Evaluation results not found: {args.input}")

    # Create visualizer
    viz = EvaluationVisualizer(args.input)

    # Generate plots
    if args.individual:
        viz.create_individual_plots(args.output)
    else:
        viz.create_full_dashboard(args.output)

    print("\n✅ Visualization complete!")


if __name__ == "__main__":
    main()
