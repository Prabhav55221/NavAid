#!/usr/bin/env python3
"""
Complete evaluation pipeline for hazard detection.

Combines:
- Detection metrics (TP, FP, FN, Precision, Recall, F1)
- CHMR (Critical Hazard Miss Rate)
- Per-type analysis
- Latency statistics
- Confidence calibration
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

from eval.metrics import HazardEvaluator


def load_latency_stats(predictions_dir: Path) -> dict:
    """Load aggregate latency statistics if available."""
    stats_file = predictions_dir / "aggregate_statistics.json"
    if stats_file.exists():
        with open(stats_file, 'r') as f:
            return json.load(f)
    return {}


def extract_per_image_latencies(predictions_dir: Path) -> dict:
    """Extract latency from each prediction file."""
    latencies = {}
    for pred_file in predictions_dir.glob("*__g15flash__v*.json"):
        image_name = pred_file.stem.split("__")[0] + ".png"
        with open(pred_file, 'r') as f:
            data = json.load(f)
            if '_meta' in data and 'latency' in data['_meta']:
                latencies[image_name] = data['_meta']['latency']
    return latencies


def compile_full_report(detection_results: dict, latency_stats: dict,
                       per_image_latencies: dict) -> dict:
    """Combine all evaluation results."""
    report = {
        "metadata": {
            "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
            "num_images_evaluated": (
                detection_results['binary_metrics']['tp'] +
                detection_results['binary_metrics']['fp'] +
                detection_results['binary_metrics']['fn'] +
                detection_results['binary_metrics']['tn']
            )
        },
        "detection_performance": detection_results,
        "latency_performance": {
            "aggregate_stats": latency_stats.get('latency_ms', {}),
            "per_image_count": len(per_image_latencies),
            "config": latency_stats.get('config', {})
        }
    }
    return report


def print_full_summary(report: dict):
    """Print comprehensive summary with all metrics."""
    print("\n" + "="*70)
    print(" " * 20 + "NAVAID HAZARD DETECTION EVALUATION")
    print("="*70)

    # Detection Performance
    bm = report['detection_performance']['binary_metrics']
    print("\n📊 BINARY DETECTION PERFORMANCE")
    print("-"*70)
    print(f"  Confusion Matrix:")
    print(f"    TP: {bm['tp']:<3}  FP: {bm['fp']:<3}")
    print(f"    FN: {bm['fn']:<3}  TN: {bm['tn']:<3}")
    print()
    print(f"  Precision:  {bm['precision']:>6.2%}  (How many detections were correct?)")
    print(f"  Recall:     {bm['recall']:>6.2%}  (How many hazards did we catch?)")
    print(f"  F1-Score:   {bm['f1']:>6.2%}  (Harmonic mean)")
    print(f"  Accuracy:   {bm['accuracy']:>6.2%}  (Overall correctness)")

    # CHMR - Safety Critical
    print(f"\n🚨 SAFETY-CRITICAL METRIC")
    print("-"*70)
    print(f"  CHMR (Critical Hazard Miss Rate): {bm['chmr']:>6.2%}")
    print(f"    → This means {bm['chmr']*100:.1f}% of hazardous scenes were COMPLETELY MISSED")
    if bm['chmr'] < 0.05:
        print(f"    ✅ PASS: < 5% target (catches ≥95% of hazards)")
    elif bm['chmr'] < 0.10:
        print(f"    ⚠️  CAUTION: 5-10% miss rate (marginally acceptable)")
    else:
        print(f"    ❌ FAIL: ≥10% miss rate (UNSAFE for blind navigation)")

    # Confidence Calibration
    cs = report['detection_performance']['confidence_stats']
    print(f"\n📈 CONFIDENCE CALIBRATION")
    print("-"*70)
    print(f"  Mean Confidence (All):       {cs['mean_confidence']:>6.2%}")
    print(f"  Mean Confidence (Correct):   {cs['mean_confidence_correct']:>6.2%}")
    print(f"  Mean Confidence (Incorrect): {cs['mean_confidence_incorrect']:>6.2%}")

    conf_gap = cs['mean_confidence_correct'] - cs['mean_confidence_incorrect']
    print(f"\n  Confidence Gap (Correct - Incorrect): {conf_gap:+.2%}")
    if conf_gap > 0.10:
        print(f"    ✅ Good calibration (model is more confident when correct)")
    elif conf_gap > 0:
        print(f"    ⚠️  Weak calibration (small confidence difference)")
    else:
        print(f"    ❌ Poor calibration (model overconfident on errors!)")

    # Per-Type Performance
    print(f"\n🎯 PER-HAZARD-TYPE PERFORMANCE")
    print("-"*70)
    print(f"{'Hazard Type':<15} {'GT Count':<10} {'Precision':<12} {'Recall':<12} {'F1':<10}")
    print("-"*70)
    type_metrics = report['detection_performance']['type_metrics']
    for type_name, tm in sorted(type_metrics.items(), key=lambda x: x[1]['ground_truth_count'], reverse=True):
        print(f"{type_name:<15} {tm['ground_truth_count']:<10} "
              f"{tm['precision']:>6.2%}{'':>5} {tm['recall']:>6.2%}{'':>5} {tm['f1']:>6.2%}")

    # Latency Performance
    if 'aggregate_stats' in report['latency_performance'] and report['latency_performance']['aggregate_stats']:
        lat = report['latency_performance']['aggregate_stats']
        print(f"\n⏱️  LATENCY PERFORMANCE")
        print("-"*70)
        print(f"  Mean:   {lat.get('mean', 0):>7.0f}ms  (Average response time)")
        print(f"  Median: {lat.get('median', 0):>7.0f}ms  (Typical response time)")
        print(f"  P95:    {lat.get('p95', 0):>7.0f}ms  (95% faster than this)")
        print(f"  P99:    {lat.get('p99', 0):>7.0f}ms  (99% faster than this)")
        print(f"  Range:  {lat.get('min', 0):>7.0f}ms - {lat.get('max', 0):>7.0f}ms")

        p95 = lat.get('p95', 0)
        print(f"\n  Real-time Viability:")
        if p95 < 1000:
            print(f"    ✅ EXCELLENT: P95 < 1000ms (sub-second response)")
        elif p95 < 2000:
            print(f"    ✅ GOOD: P95 < 2000ms (acceptable for navigation)")
        elif p95 < 3000:
            print(f"    ⚠️  MARGINAL: P95 2-3s (may feel sluggish)")
        else:
            print(f"    ❌ SLOW: P95 > 3s (too slow for real-time use)")

    # Error Analysis
    ea = report['detection_performance']['error_analysis']
    print(f"\n🔍 ERROR ANALYSIS")
    print("-"*70)
    print(f"  Total Errors:      {ea['total_errors']}")
    print(f"    False Positives: {ea['false_positives']} (detected hazard when none exists)")
    print(f"    False Negatives: {ea['false_negatives']} (missed actual hazards)")

    if ea['error_cases']:
        print(f"\n  Sample Errors:")
        for i, err in enumerate(ea['error_cases'][:3], 1):
            print(f"\n  {i}. {err['error_type']}: {err['image_name']}")
            print(f"     Ground Truth: {err['gt_types'] if err['gt_types'] else 'No hazard'}")
            print(f"     Predicted:    {err['pred_types'] if err['pred_types'] else 'No hazard'}")
            print(f"     Confidence:   {err['confidence']:.2%}")
            if err['notes']:
                print(f"     Notes:        {err['notes'][:60]}...")

    print("\n" + "="*70)
    print(" " * 25 + "END OF EVALUATION REPORT")
    print("="*70 + "\n")


def main():
    ap = argparse.ArgumentParser(
        description="Complete evaluation pipeline for hazard detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python evaluate.py \\
    --ground_truth data/ground_truth_labels.json \\
    --predictions OUTPUTS \\
    --output results/evaluation_report.json
        """
    )
    ap.add_argument("--ground_truth", "-g", type=Path, required=True,
                    help="Path to ground_truth_labels.json")
    ap.add_argument("--predictions", "-p", type=Path, required=True,
                    help="Directory containing prediction JSON files")
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="Output JSON file for full report (optional)")
    args = ap.parse_args()

    # Validate inputs
    if not args.ground_truth.exists():
        raise FileNotFoundError(f"Ground truth not found: {args.ground_truth}")
    if not args.predictions.is_dir():
        raise NotADirectoryError(f"Predictions directory not found: {args.predictions}")

    # Run detection evaluation
    print("Running hazard detection evaluation...")
    evaluator = HazardEvaluator(args.ground_truth, args.predictions)
    detection_results = evaluator.evaluate_all()

    # Load latency data
    print("Loading latency statistics...")
    latency_stats = load_latency_stats(args.predictions)
    per_image_latencies = extract_per_image_latencies(args.predictions)

    # Compile full report
    full_report = compile_full_report(detection_results, latency_stats, per_image_latencies)

    # Print comprehensive summary
    print_full_summary(full_report)

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(full_report, f, indent=2)
        print(f"✓ Full evaluation report saved to: {args.output}")

    # Quick decision summary
    bm = detection_results['binary_metrics']
    lat = latency_stats.get('latency_ms', {})

    print("\n" + "="*70)
    print("QUICK DECISION SUMMARY")
    print("="*70)
    print(f"  Detection Quality:  {'✅ GOOD' if bm['f1'] > 0.80 else '⚠️  NEEDS IMPROVEMENT' if bm['f1'] > 0.60 else '❌ POOR'}")
    print(f"  Safety (CHMR):      {'✅ SAFE' if bm['chmr'] < 0.05 else '⚠️  MARGINAL' if bm['chmr'] < 0.10 else '❌ UNSAFE'}")
    if lat.get('p95'):
        print(f"  Latency:            {'✅ FAST' if lat['p95'] < 2000 else '⚠️  ACCEPTABLE' if lat['p95'] < 3000 else '❌ SLOW'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
