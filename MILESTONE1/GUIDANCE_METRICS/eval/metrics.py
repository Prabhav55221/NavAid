#!/usr/bin/env python3
"""
Evaluation metrics for hazard detection.

Computes:
- Binary detection metrics (TP, FP, FN, TN, Precision, Recall, F1, Accuracy)
- Critical Hazard Miss Rate (CHMR)
- Per-class metrics
- Confidence calibration
- Error analysis
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


@dataclass
class BinaryMetrics:
    """Binary classification metrics for hazard detection."""
    tp: int = 0  # True Positives
    fp: int = 0  # False Positives
    fn: int = 0  # False Negatives
    tn: int = 0  # True Negatives

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall (Sensitivity) = TP / (TP + FN)"""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        """F1-Score = 2 * (Precision * Recall) / (Precision + Recall)"""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Accuracy = (TP + TN) / Total"""
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    @property
    def chmr(self) -> float:
        """Critical Hazard Miss Rate = FN / (TP + FN)"""
        denom = self.tp + self.fn
        return self.fn / denom if denom > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "chmr": round(self.chmr, 4)
        }


@dataclass
class TypeMetrics:
    """Per-hazard-type metrics."""
    type_name: str
    tp: int = 0  # Correctly detected this type
    fp: int = 0  # Falsely detected this type
    fn: int = 0  # Missed this type
    ground_truth_count: int = 0  # Total instances in GT

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "type": self.type_name,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "ground_truth_count": self.ground_truth_count,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4)
        }


@dataclass
class ConfidenceStats:
    """Confidence calibration statistics."""
    confidences_correct: List[float] = field(default_factory=list)
    confidences_incorrect: List[float] = field(default_factory=list)
    all_confidences: List[float] = field(default_factory=list)

    def add(self, confidence: float, is_correct: bool):
        self.all_confidences.append(confidence)
        if is_correct:
            self.confidences_correct.append(confidence)
        else:
            self.confidences_incorrect.append(confidence)

    def to_dict(self) -> Dict:
        return {
            "mean_confidence": round(np.mean(self.all_confidences), 4) if self.all_confidences else 0,
            "mean_confidence_correct": round(np.mean(self.confidences_correct), 4) if self.confidences_correct else 0,
            "mean_confidence_incorrect": round(np.mean(self.confidences_incorrect), 4) if self.confidences_incorrect else 0,
            "std_confidence": round(np.std(self.all_confidences), 4) if self.all_confidences else 0,
            "num_correct": len(self.confidences_correct),
            "num_incorrect": len(self.confidences_incorrect)
        }


@dataclass
class ErrorCase:
    """Individual error case for analysis."""
    image_name: str
    error_type: str  # 'FP' or 'FN'
    gt_has_hazard: bool
    gt_hazard_types: List[str]
    pred_has_hazard: bool
    pred_hazard_types: List[str]
    confidence: float
    notes: str = ""


class HazardEvaluator:
    """Evaluates hazard detection performance."""

    # Mapping from Gemini types to ground truth types
    TYPE_MAPPING = {
        # Gemini -> Ground Truth
        "pole": "column",
        "person": "creature",
        "car": "vehicle",
        "truck": "vehicle",
        "van": "vehicle",
        "bike": "bicycle",
        "sign": "trafficsign",
        # Direct mappings
        "trafficcone": "trafficcone",
        "vehicle": "vehicle",
        "creature": "creature",
        "column": "column",
        "wall": "wall",
    }

    def __init__(self, ground_truth_path: Path, predictions_dir: Path):
        """
        Initialize evaluator.

        Args:
            ground_truth_path: Path to ground_truth_labels.json
            predictions_dir: Directory containing prediction JSON files
        """
        self.ground_truth_path = ground_truth_path
        self.predictions_dir = predictions_dir

        # Load ground truth
        with open(ground_truth_path, 'r') as f:
            self.ground_truth = {item['image_name']: item for item in json.load(f)}

        # Load predictions
        self.predictions = {}
        for pred_file in predictions_dir.glob("*__g15flash__v*.json"):
            # Extract image name from filename
            # Format: {image_name}__g15flash__v2.0.json
            image_name = pred_file.stem.split("__")[0] + ".png"
            with open(pred_file, 'r') as f:
                self.predictions[image_name] = json.load(f)

        # Results
        self.binary_metrics = BinaryMetrics()
        self.type_metrics: Dict[str, TypeMetrics] = defaultdict(lambda: TypeMetrics("unknown"))
        self.confidence_stats = ConfidenceStats()
        self.error_cases: List[ErrorCase] = []

    def normalize_type(self, pred_type: str) -> str:
        """Map predicted type to ground truth type."""
        return self.TYPE_MAPPING.get(pred_type.lower(), pred_type.lower())

    def evaluate_all(self) -> Dict:
        """Run full evaluation."""
        print("="*60)
        print("HAZARD DETECTION EVALUATION")
        print("="*60)

        # Find common images
        common_images = set(self.ground_truth.keys()) & set(self.predictions.keys())
        print(f"\nGround truth images: {len(self.ground_truth)}")
        print(f"Prediction images: {len(self.predictions)}")
        print(f"Common images: {len(common_images)}")

        if not common_images:
            raise ValueError("No common images found between GT and predictions!")

        # Evaluate each image
        for image_name in sorted(common_images):
            self._evaluate_image(image_name)

        # Compute type-level metrics
        self._compute_type_metrics(common_images)

        return self._compile_results()

    def _evaluate_image(self, image_name: str):
        """Evaluate single image."""
        gt = self.ground_truth[image_name]
        pred = self.predictions[image_name]['result']

        gt_has_hazard = gt['has_hazard']
        pred_has_hazard = pred['hazard_detected']
        confidence = pred['confidence']

        # Binary classification
        if gt_has_hazard and pred_has_hazard:
            self.binary_metrics.tp += 1
            is_correct = True
        elif not gt_has_hazard and not pred_has_hazard:
            self.binary_metrics.tn += 1
            is_correct = True
        elif not gt_has_hazard and pred_has_hazard:
            self.binary_metrics.fp += 1
            is_correct = False
            self._record_error(image_name, "FP", gt, pred)
        else:  # gt_has_hazard and not pred_has_hazard
            self.binary_metrics.fn += 1
            is_correct = False
            self._record_error(image_name, "FN", gt, pred)

        # Confidence tracking
        self.confidence_stats.add(confidence, is_correct)

    def _compute_type_metrics(self, common_images: Set[str]):
        """Compute per-type precision/recall."""
        for image_name in common_images:
            gt = self.ground_truth[image_name]
            pred = self.predictions[image_name]['result']

            # Get ground truth types
            gt_types = set()
            for hazard in gt['critical_hazards']:
                gt_types.add(hazard['class'])

            # Get predicted types (normalized)
            pred_types = set()
            for pred_type in pred['hazard_types']:
                normalized = self.normalize_type(pred_type)
                pred_types.add(normalized)

            # Initialize type metrics
            for gt_type in gt_types:
                if gt_type not in self.type_metrics:
                    self.type_metrics[gt_type] = TypeMetrics(gt_type)

            # Count TP, FP, FN per type
            for gt_type in gt_types:
                self.type_metrics[gt_type].ground_truth_count += 1
                if gt_type in pred_types:
                    self.type_metrics[gt_type].tp += 1
                else:
                    self.type_metrics[gt_type].fn += 1

            for pred_type in pred_types:
                if pred_type not in gt_types:
                    if pred_type not in self.type_metrics:
                        self.type_metrics[pred_type] = TypeMetrics(pred_type)
                    self.type_metrics[pred_type].fp += 1

    def _record_error(self, image_name: str, error_type: str, gt: Dict, pred: Dict):
        """Record error case for analysis."""
        gt_types = [h['class'] for h in gt['critical_hazards']]
        pred_types = pred['hazard_types']

        error = ErrorCase(
            image_name=image_name,
            error_type=error_type,
            gt_has_hazard=gt['has_hazard'],
            gt_hazard_types=gt_types,
            pred_has_hazard=pred['hazard_detected'],
            pred_hazard_types=pred_types,
            confidence=pred['confidence'],
            notes=pred.get('notes', '')
        )
        self.error_cases.append(error)

    def _compile_results(self) -> Dict:
        """Compile all results into structured dict."""
        results = {
            "binary_metrics": self.binary_metrics.to_dict(),
            "confidence_stats": self.confidence_stats.to_dict(),
            "type_metrics": {
                name: metrics.to_dict()
                for name, metrics in sorted(self.type_metrics.items())
            },
            "error_analysis": {
                "total_errors": len(self.error_cases),
                "false_positives": len([e for e in self.error_cases if e.error_type == "FP"]),
                "false_negatives": len([e for e in self.error_cases if e.error_type == "FN"]),
                "error_cases": [
                    {
                        "image_name": e.image_name,
                        "error_type": e.error_type,
                        "gt_types": e.gt_hazard_types,
                        "pred_types": e.pred_hazard_types,
                        "confidence": round(e.confidence, 2),
                        "notes": e.notes
                    }
                    for e in self.error_cases
                ]
            }
        }
        return results

    def print_summary(self, results: Dict):
        """Print human-readable summary."""
        print("\n" + "="*60)
        print("BINARY DETECTION METRICS")
        print("="*60)
        bm = results['binary_metrics']
        print(f"  True Positives:  {bm['tp']}")
        print(f"  False Positives: {bm['fp']}")
        print(f"  False Negatives: {bm['fn']}")
        print(f"  True Negatives:  {bm['tn']}")
        print(f"\n  Precision: {bm['precision']:.2%}")
        print(f"  Recall:    {bm['recall']:.2%}")
        print(f"  F1-Score:  {bm['f1']:.2%}")
        print(f"  Accuracy:  {bm['accuracy']:.2%}")
        print(f"\n  🚨 CHMR (Critical Hazard Miss Rate): {bm['chmr']:.2%}")
        if bm['chmr'] < 0.05:
            print(f"     ✅ PASS (< 5% target)")
        else:
            print(f"     ⚠️  WARNING (≥ 5%, needs improvement)")

        print("\n" + "="*60)
        print("CONFIDENCE CALIBRATION")
        print("="*60)
        cs = results['confidence_stats']
        print(f"  Mean Confidence (All):       {cs['mean_confidence']:.2%}")
        print(f"  Mean Confidence (Correct):   {cs['mean_confidence_correct']:.2%}")
        print(f"  Mean Confidence (Incorrect): {cs['mean_confidence_incorrect']:.2%}")
        print(f"  Std Deviation:               {cs['std_confidence']:.4f}")

        print("\n" + "="*60)
        print("PER-TYPE METRICS")
        print("="*60)
        print(f"{'Type':<15} {'GT Count':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print("-"*60)
        for type_name, tm in sorted(results['type_metrics'].items()):
            print(f"{type_name:<15} {tm['ground_truth_count']:<10} "
                  f"{tm['precision']:.2%}{'':>6} {tm['recall']:.2%}{'':>6} {tm['f1']:.2%}")

        print("\n" + "="*60)
        print("ERROR ANALYSIS")
        print("="*60)
        ea = results['error_analysis']
        print(f"  Total Errors: {ea['total_errors']}")
        print(f"    False Positives: {ea['false_positives']}")
        print(f"    False Negatives: {ea['false_negatives']}")

        if ea['error_cases']:
            print(f"\n  Sample Errors (first 5):")
            for err in ea['error_cases'][:5]:
                print(f"    {err['error_type']}: {err['image_name']}")
                print(f"       GT: {err['gt_types'] if err['gt_types'] else 'No hazard'}")
                print(f"       Pred: {err['pred_types'] if err['pred_types'] else 'No hazard'} (conf={err['confidence']:.2f})")


def main():
    """Main evaluation script."""
    import argparse

    ap = argparse.ArgumentParser(description="Evaluate hazard detection performance.")
    ap.add_argument("--ground_truth", "-g", type=Path, required=True,
                    help="Path to ground_truth_labels.json")
    ap.add_argument("--predictions", "-p", type=Path, required=True,
                    help="Directory containing prediction JSON files")
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="Output JSON file for metrics (optional)")
    args = ap.parse_args()

    # Run evaluation
    evaluator = HazardEvaluator(args.ground_truth, args.predictions)
    results = evaluator.evaluate_all()

    # Print summary
    evaluator.print_summary(results)

    # Save results
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
