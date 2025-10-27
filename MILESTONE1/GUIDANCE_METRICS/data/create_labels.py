#!/usr/bin/env python3
"""
Create ground truth labels for hazard detection evaluation.

Processes all annotation files and creates binary labels (hazard/no-hazard).
"""

import json
import glob
from pathlib import Path
from typing import Dict, List, Set
import csv


# Hazard classification rules
CRITICAL_HAZARDS = {
    'vehicle',        # Moving vehicles - collision risk
    'trafficcone',    # Obstacles in path
    'creature',       # Pedestrians, animals
    'column',         # Static obstacles
    'wall'            # Barriers, obstacles
}

NON_HAZARDS = {
    'bump',           # Road surface feature (not a collision hazard)
    'dent',           # Road surface feature (not a collision hazard)
    'weed',           # Minor environmental feature
}

INFORMATIONAL = {
    'trafficsign',    # Important but not immediate collision risk
    'zebracrossing'   # Important context
}


class GroundTruthLabeler:
    """Create ground truth labels from annotations."""

    def __init__(self, annotations_dir: Path, images_dir: Path):
        """
        Initialize labeler.

        Args:
            annotations_dir: Directory containing annotation JSON files
            images_dir: Directory containing images
        """
        self.annotations_dir = Path(annotations_dir)
        self.images_dir = Path(images_dir)

    def classify_hazard_level(self, class_title: str) -> str:
        """
        Classify object into hazard level.

        Args:
            class_title: Object class name

        Returns:
            Hazard level: 'critical', 'informational', or 'non-hazard'
        """
        if class_title in CRITICAL_HAZARDS:
            return 'critical'
        elif class_title in INFORMATIONAL:
            return 'informational'
        elif class_title in NON_HAZARDS:
            return 'non-hazard'
        else:
            # Unknown classes treated as potential hazards
            return 'unknown'

    def process_annotation(self, annotation_path: Path) -> Dict:
        """
        Process single annotation file.

        Args:
            annotation_path: Path to annotation JSON

        Returns:
            Dictionary with image metadata and hazard labels
        """
        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        image_name = annotation_path.stem  # Remove .json extension
        image_path = self.images_dir / image_name

        # Extract all objects and classify
        objects = annotation.get('objects', [])

        critical_hazards = []
        informational_objects = []
        non_hazards = []
        unknown_objects = []

        for obj in objects:
            class_title = obj['classTitle']
            hazard_level = self.classify_hazard_level(class_title)

            obj_info = {
                'class': class_title,
                'bbox': obj['points']['exterior'],
                'id': obj['id']
            }

            if hazard_level == 'critical':
                critical_hazards.append(obj_info)
            elif hazard_level == 'informational':
                informational_objects.append(obj_info)
            elif hazard_level == 'non-hazard':
                non_hazards.append(obj_info)
            else:
                unknown_objects.append(obj_info)

        # Determine if image contains hazards
        has_hazard = len(critical_hazards) > 0
        has_critical_hazard = has_hazard  # Same for now

        return {
            'image_name': image_name,
            'image_path': str(image_path),
            'width': annotation['size']['width'],
            'height': annotation['size']['height'],
            'has_hazard': has_hazard,
            'has_critical_hazard': has_critical_hazard,
            'num_critical_hazards': len(critical_hazards),
            'num_informational': len(informational_objects),
            'num_non_hazards': len(non_hazards),
            'num_unknown': len(unknown_objects),
            'critical_hazards': critical_hazards,
            'informational_objects': informational_objects,
            'non_hazards': non_hazards,
            'unknown_objects': unknown_objects,
            'total_objects': len(objects)
        }

    def create_all_labels(self) -> List[Dict]:
        """
        Process all annotations.

        Returns:
            List of label dictionaries
        """
        annotation_files = sorted(self.annotations_dir.glob('*.json'))

        print(f"Found {len(annotation_files)} annotation files")

        all_labels = []

        for ann_path in annotation_files:
            try:
                label = self.process_annotation(ann_path)
                all_labels.append(label)
            except Exception as e:
                print(f"Error processing {ann_path.name}: {e}")
                continue

        return all_labels

    def save_labels(self, labels: List[Dict], output_path: Path):
        """
        Save labels to JSON file.

        Args:
            labels: List of label dictionaries
            output_path: Path to save JSON
        """
        with open(output_path, 'w') as f:
            json.dump(labels, f, indent=2)

        print(f"✓ Saved labels to {output_path}")

    def save_summary_csv(self, labels: List[Dict], output_path: Path):
        """
        Save summary CSV (for easy inspection).

        Args:
            labels: List of label dictionaries
            output_path: Path to save CSV
        """
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'image_name',
                'has_hazard',
                'num_critical_hazards',
                'critical_classes',
                'num_informational',
                'num_non_hazards',
                'total_objects'
            ])

            # Rows
            for label in labels:
                critical_classes = ', '.join(sorted(set(h['class'] for h in label['critical_hazards'])))

                writer.writerow([
                    label['image_name'],
                    label['has_hazard'],
                    label['num_critical_hazards'],
                    critical_classes if critical_classes else 'None',
                    label['num_informational'],
                    label['num_non_hazards'],
                    label['total_objects']
                ])

        print(f"✓ Saved summary CSV to {output_path}")

    def print_statistics(self, labels: List[Dict]):
        """Print dataset statistics."""
        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)

        total_images = len(labels)
        images_with_hazards = sum(1 for l in labels if l['has_hazard'])
        images_without_hazards = total_images - images_with_hazards

        total_critical = sum(l['num_critical_hazards'] for l in labels)
        total_informational = sum(l['num_informational'] for l in labels)
        total_non_hazards = sum(l['num_non_hazards'] for l in labels)

        print(f"\nTotal images: {total_images}")
        print(f"  Images with hazards: {images_with_hazards} ({images_with_hazards/total_images*100:.1f}%)")
        print(f"  Images without hazards: {images_without_hazards} ({images_without_hazards/total_images*100:.1f}%)")

        print(f"\nTotal objects:")
        print(f"  Critical hazards: {total_critical}")
        print(f"  Informational: {total_informational}")
        print(f"  Non-hazards: {total_non_hazards}")

        # Count by class
        class_counts = {}
        for label in labels:
            for hazard in label['critical_hazards']:
                class_name = hazard['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

        print(f"\nCritical hazard breakdown:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count}")


def main():
    """Main execution."""
    # Paths
    data_dir = Path(__file__).parent
    annotations_dir = data_dir / "Annotations"
    images_dir = data_dir / "Images"
    output_dir = data_dir

    print("=" * 60)
    print("GROUND TRUTH LABEL GENERATION")
    print("=" * 60)

    # Create labeler
    labeler = GroundTruthLabeler(annotations_dir, images_dir)

    # Process all annotations
    print(f"\nProcessing annotations from: {annotations_dir}")
    labels = labeler.create_all_labels()

    # Print statistics
    labeler.print_statistics(labels)

    # Save outputs
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    labels_json_path = output_dir / "ground_truth_labels.json"
    labeler.save_labels(labels, labels_json_path)

    summary_csv_path = output_dir / "ground_truth_summary.csv"
    labeler.save_summary_csv(labels, summary_csv_path)

    print("\n" + "=" * 60)
    print("✓ LABELING COMPLETE")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"  1. Full labels: {labels_json_path}")
    print(f"  2. Summary CSV: {summary_csv_path}")
    print(f"\nReady for Gemini evaluation!")


if __name__ == "__main__":
    main()
