"""
Visualization module for TTS evaluation metrics.

Generates comparison plots for RTF, WER, footprint, and latency distributions.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def load_metrics(metrics_path: Path) -> Dict:
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def plot_rtf_comparison(metrics: Dict, output_path: Path):
    """
    Generate RTF comparison bar chart.

    Args:
        metrics: Dictionary of metrics per model
        output_path: Path to save plot
    """
    # Extract RTF data
    models = []
    rtf_medians = []
    rtf_p95s = []

    for model_name, model_metrics in metrics.items():
        if 'rtf' in model_metrics and model_metrics['rtf'].get('rtf_median') is not None:
            models.append(model_name)
            rtf_medians.append(model_metrics['rtf']['rtf_median'])
            rtf_p95s.append(model_metrics['rtf']['rtf_p95'])

    if not models:
        print("No RTF data available for plotting")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, rtf_medians, width, label='Median RTF', color='steelblue')
    bars2 = ax.bar(x + width/2, rtf_p95s, width, label='95th Percentile RTF', color='coral')

    # Add threshold line
    ax.axhline(y=0.1, color='green', linestyle='--', linewidth=2, label='Target RTF < 0.1', alpha=0.7)
    ax.axhline(y=0.2, color='orange', linestyle='--', linewidth=2, label='Acceptable RTF < 0.2', alpha=0.7)

    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Real-Time Factor', fontweight='bold')
    ax.set_title('TTS Model Real-Time Factor Comparison\n(Lower is Faster)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved RTF comparison: {output_path}")


def plot_wer_comparison(metrics: Dict, output_path: Path):
    """
    Generate WER comparison bar chart.

    Args:
        metrics: Dictionary of metrics per model
        output_path: Path to save plot
    """
    # Extract WER data
    models = []
    wer_means = []
    wer_stds = []

    for model_name, model_metrics in metrics.items():
        if 'wer' in model_metrics and model_metrics['wer'].get('wer_mean') is not None:
            models.append(model_name)
            wer_means.append(model_metrics['wer']['wer_mean'])
            wer_stds.append(model_metrics['wer'].get('wer_std', 0))

    if not models:
        print("No WER data available for plotting")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    bars = ax.bar(x, wer_means, yerr=wer_stds, capsize=5, color='lightcoral', edgecolor='darkred', linewidth=1.5)

    # Add threshold lines
    ax.axhline(y=10, color='green', linestyle='--', linewidth=2, label='Target WER < 10%', alpha=0.7)
    ax.axhline(y=15, color='orange', linestyle='--', linewidth=2, label='Acceptable WER < 15%', alpha=0.7)

    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Word Error Rate (%)', fontweight='bold')
    ax.set_title('TTS Model Intelligibility (Word Error Rate)\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, mean_val) in enumerate(zip(bars, wer_means)):
        ax.text(bar.get_x() + bar.get_width()/2., mean_val,
               f'{mean_val:.1f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved WER comparison: {output_path}")


def plot_footprint_comparison(metrics: Dict, output_path: Path):
    """
    Generate footprint comparison grouped bar chart.

    Args:
        metrics: Dictionary of metrics per model
        output_path: Path to save plot
    """
    # Extract footprint data
    models = []
    disk_sizes = []
    memory_peaks = []
    cold_starts = []

    for model_name, model_metrics in metrics.items():
        if 'footprint' in model_metrics:
            fp = model_metrics['footprint']
            models.append(model_name)
            disk_sizes.append(fp.get('disk_size_mb', 0) or 0)
            memory_peaks.append(fp.get('memory_peak_mb', 0) or 0)
            # Handle None for cold_start_ms
            cold_start_val = fp.get('cold_start_ms')
            cold_starts.append((cold_start_val / 1000) if cold_start_val is not None else 0)

    if not models:
        print("No footprint data available for plotting")
        return

    # Create subplot with 3 panels
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    x = np.arange(len(models))

    # Plot 1: Disk Size
    bars1 = ax1.bar(x, disk_sizes, color='mediumpurple', edgecolor='darkviolet', linewidth=1.5)
    ax1.set_xlabel('Model', fontweight='bold')
    ax1.set_ylabel('Disk Size (MB)', fontweight='bold')
    ax1.set_title('Model Disk Size', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars1, disk_sizes):
        ax1.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.0f}MB',
                ha='center', va='bottom', fontsize=9)

    # Plot 2: Memory Peak
    bars2 = ax2.bar(x, memory_peaks, color='lightseagreen', edgecolor='darkgreen', linewidth=1.5)
    ax2.set_xlabel('Model', fontweight='bold')
    ax2.set_ylabel('Peak Memory (MB)', fontweight='bold')
    ax2.set_title('Runtime Memory Usage', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars2, memory_peaks):
        ax2.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.0f}MB',
                ha='center', va='bottom', fontsize=9)

    # Plot 3: Cold Start Time
    bars3 = ax3.bar(x, cold_starts, color='sandybrown', edgecolor='darkorange', linewidth=1.5)
    ax3.set_xlabel('Model', fontweight='bold')
    ax3.set_ylabel('Cold Start Time (s)', fontweight='bold')
    ax3.set_title('Model Load Time', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars3, cold_starts):
        ax3.text(bar.get_x() + bar.get_width()/2., val,
                f'{val:.1f}s',
                ha='center', va='bottom', fontsize=9)

    plt.suptitle('TTS Model Footprint Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved footprint comparison: {output_path}")


def plot_latency_distribution(metrics: Dict, output_path: Path):
    """
    Generate latency distribution box plot.

    Args:
        metrics: Dictionary of metrics per model
        output_path: Path to save plot
    """
    # Extract latency data
    data_for_plot = []

    for model_name, model_metrics in metrics.items():
        if 'rtf' in model_metrics and 'raw_synthesis_times' in model_metrics['rtf']:
            times = model_metrics['rtf']['raw_synthesis_times']
            for time_ms in times:
                data_for_plot.append({
                    'Model': model_name,
                    'Synthesis Time (ms)': time_ms
                })

    if not data_for_plot:
        print("No latency distribution data available for plotting")
        return

    df = pd.DataFrame(data_for_plot)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.boxplot(data=df, x='Model', y='Synthesis Time (ms)', ax=ax, palette='Set2')
    sns.stripplot(data=df, x='Model', y='Synthesis Time (ms)', ax=ax,
                 color='black', alpha=0.3, size=2)

    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('Synthesis Time (ms)', fontweight='bold')
    ax.set_title('TTS Synthesis Time Distribution\n(Per Sample)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved latency distribution: {output_path}")


def generate_all_plots(metrics_path: Path, output_dir: Path) -> bool:
    """
    Generate all visualization plots.

    Args:
        metrics_path: Path to metrics.json
        output_dir: Directory to save plots

    Returns:
        True if successful
    """
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATION PLOTS")
    print("=" * 60)

    if not metrics_path.exists():
        print(f"✗ Metrics file not found: {metrics_path}")
        print("  Run evaluation first: python main.py --mode evaluate")
        return False

    # Load metrics
    print(f"\nLoading metrics from: {metrics_path}")
    metrics = load_metrics(metrics_path)
    print(f"✓ Loaded metrics for {len(metrics)} models")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate each plot
    try:
        plot_rtf_comparison(metrics, output_dir / "rtf_comparison.png")
        plot_wer_comparison(metrics, output_dir / "wer_comparison.png")
        plot_footprint_comparison(metrics, output_dir / "footprint_comparison.png")
        plot_latency_distribution(metrics, output_dir / "latency_distribution.png")

        print("\n" + "=" * 60)
        print("✓ ALL PLOTS GENERATED SUCCESSFULLY")
        print("=" * 60)
        print(f"\nPlots saved to: {output_dir}")
        print("  - rtf_comparison.png")
        print("  - wer_comparison.png")
        print("  - footprint_comparison.png")
        print("  - latency_distribution.png")

        return True

    except Exception as e:
        print(f"\n✗ Plot generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
