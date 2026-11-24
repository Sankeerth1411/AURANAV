#!/usr/bin/env python3
"""
AURANAV - Dataset Validator
Comprehensive validation of YOLO format dataset
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter

import yaml
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from src.config_loader import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validate YOLO format dataset"""

    def __init__(self, dataset_path: Path):
        """
        Initialize validator

        Args:
            dataset_path: Path to dataset root directory
        """
        self.config = get_config()
        self.dataset_path = Path(dataset_path)
        self.data_yaml = self._load_yaml()
        self.classes = self._parse_classes()
        self.num_classes = len(self.classes)

    def _load_yaml(self) -> Dict:
        """Load data.yaml file"""
        yaml_path = self.dataset_path / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"data.yaml not found at {yaml_path}")

        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    def _parse_classes(self) -> List[str]:
        """Parse class names from data.yaml"""
        classes = self.data_yaml.get('names', [])
        if isinstance(classes, dict):
            classes = [classes[i] for i in sorted(classes.keys())]
        return classes

    def validate_all(self) -> Dict:
        """
        Run all validation checks

        Returns:
            Dictionary with validation results
        """
        logger.info("=" * 60)
        logger.info("DATASET VALIDATION")
        logger.info("=" * 60)

        results = {
            'dataset_path': str(self.dataset_path),
            'num_classes': self.num_classes,
            'classes': self.classes,
            'splits': {},
            'label_errors': [],
            'missing_labels': [],
            'class_distribution': {},
            'issues': [],
            'passed': False
        }

        # Validate splits
        for split in ['train', 'valid', 'test']:
            logger.info(f"\nValidating {split.upper()} split...")
            split_results = self.validate_split(split)
            results['splits'][split] = split_results

            if split_results['label_errors']:
                results['label_errors'].extend(
                    [(split, e) for e in split_results['label_errors']]
                )

            if split_results['missing_labels']:
                results['missing_labels'].extend(
                    [(split, f) for f in split_results['missing_labels']]
                )

        # Check minimum images per split
        min_images = self.config.get('validation', 'min_images_per_split')
        for split, min_count in min_images.items():
            split_data = results['splits'].get(split, {})
            actual_count = split_data.get('num_images', 0)

            if actual_count < min_count:
                issue = f"{split} split has only {actual_count} images (minimum: {min_count})"
                results['issues'].append(issue)
                logger.warning(f"⚠️  {issue}")

        # Check total errors
        max_errors = self.config.get('validation', 'max_annotation_errors', default=100)
        if len(results['label_errors']) > max_errors:
            issue = f"Too many annotation errors: {len(results['label_errors'])} (max: {max_errors})"
            results['issues'].append(issue)

        # Compute class distribution
        results['class_distribution'] = self._compute_class_distribution(results['splits'])

        # Check for class imbalance
        self._check_class_balance(results)

        # Final verdict
        results['passed'] = len(results['issues']) == 0

        logger.info("\n" + "=" * 60)
        if results['passed']:
            logger.info("✅ VALIDATION PASSED")
        else:
            logger.error(f"❌ VALIDATION FAILED - {len(results['issues'])} issue(s)")
            for issue in results['issues']:
                logger.error(f"   - {issue}")
        logger.info("=" * 60)

        return results

    def validate_split(self, split: str) -> Dict:
        """
        Validate a single dataset split

        Args:
            split: Split name (train, valid, test)

        Returns:
            Dictionary with split validation results
        """
        results = {
            'split': split,
            'exists': False,
            'images_dir': None,
            'labels_dir': None,
            'num_images': 0,
            'num_labels': 0,
            'label_errors': [],
            'missing_labels': [],
            'class_counts': Counter()
        }

        split_dir = self.dataset_path / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            return results

        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            logger.warning(f"Missing images or labels directory in {split}")
            return results

        results['exists'] = True
        results['images_dir'] = str(images_dir)
        results['labels_dir'] = str(labels_dir)

        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [
            p for p in images_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ]
        results['num_images'] = len(image_files)

        # Validate labels
        for image_file in image_files:
            label_file = labels_dir / f"{image_file.stem}.txt"

            if not label_file.exists():
                results['missing_labels'].append(image_file.name)
                continue

            # Validate label format
            errors = self._validate_label_file(label_file)
            if errors:
                results['label_errors'].extend(
                    [(label_file.name, error) for error in errors]
                )

            # Count classes
            class_ids = self._extract_class_ids(label_file)
            results['class_counts'].update(class_ids)

        results['num_labels'] = sum(
            1 for _ in labels_dir.glob('*.txt')
        )

        logger.info(f"  Images: {results['num_images']}")
        logger.info(f"  Labels: {results['num_labels']}")
        logger.info(f"  Errors: {len(results['label_errors'])}")
        logger.info(f"  Missing labels: {len(results['missing_labels'])}")

        return results

    def _validate_label_file(self, label_file: Path) -> List[str]:
        """
        Validate YOLO format label file

        Args:
            label_file: Path to label file

        Returns:
            List of error messages
        """
        errors = []

        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                # Check number of values
                if len(parts) != 5:
                    errors.append(
                        f"Line {line_num}: Expected 5 values, got {len(parts)}"
                    )
                    continue

                # Parse values
                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                except ValueError:
                    errors.append(f"Line {line_num}: Non-numeric value")
                    continue

                # Validate class ID
                if not 0 <= class_id < self.num_classes:
                    errors.append(
                        f"Line {line_num}: Invalid class_id {class_id} "
                        f"(valid range: 0-{self.num_classes-1})"
                    )

                # Validate coordinates
                if not (0 <= x <= 1 and 0 <= y <= 1 and
                       0 <= w <= 1 and 0 <= h <= 1):
                    errors.append(
                        f"Line {line_num}: Coordinates outside [0,1] range"
                    )

                # Check for valid bounding box
                if w <= 0 or h <= 0:
                    errors.append(
                        f"Line {line_num}: Invalid bbox dimensions (w={w}, h={h})"
                    )

        except Exception as e:
            errors.append(f"Failed to read file: {str(e)}")

        return errors

    def _extract_class_ids(self, label_file: Path) -> List[int]:
        """Extract class IDs from label file"""
        class_ids = []

        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_ids.append(int(parts[0]))
        except Exception:
            pass

        return class_ids

    def _compute_class_distribution(self, splits: Dict) -> Dict:
        """Compute class distribution across all splits"""
        total_counts = Counter()

        for split_name, split_data in splits.items():
            if 'class_counts' in split_data:
                total_counts.update(split_data['class_counts'])

        distribution = {
            class_id: {
                'name': self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}",
                'count': count
            }
            for class_id, count in total_counts.items()
        }

        return distribution

    def _check_class_balance(self, results: Dict):
        """Check for severe class imbalance"""
        distribution = results['class_distribution']

        if not distribution:
            results['issues'].append("No class distribution data available")
            return

        counts = [d['count'] for d in distribution.values()]
        if not counts:
            return

        max_count = max(counts)
        min_count = min(counts)

        # Check for missing classes
        for class_id in range(self.num_classes):
            if class_id not in distribution or distribution[class_id]['count'] == 0:
                class_name = self.classes[class_id]
                issue = f"Class '{class_name}' (id={class_id}) has no annotations"
                results['issues'].append(issue)
                logger.warning(f"⚠️  {issue}")

        # Check for severe imbalance
        if max_count > 0 and min_count > 0:
            ratio = max_count / min_count
            if ratio > 100:
                issue = f"Severe class imbalance detected (ratio: {ratio:.1f}:1)"
                results['issues'].append(issue)
                logger.warning(f"⚠️  {issue}")

    def visualize_distribution(self, results: Dict, save_path: Path = None):
        """
        Visualize class distribution

        Args:
            results: Validation results dictionary
            save_path: Path to save plot (optional)
        """
        distribution = results['class_distribution']

        if not distribution:
            logger.warning("No distribution data to visualize")
            return

        # Sort by count
        sorted_dist = sorted(
            distribution.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )

        class_names = [d[1]['name'] for d in sorted_dist]
        counts = [d[1]['count'] for d in sorted_dist]

        # Create plot
        plt.figure(figsize=(14, 6))
        plt.bar(range(len(class_names)), counts)
        plt.xticks(range(len(class_names)), class_names, rotation=75, ha='right')
        plt.ylabel('Number of annotations')
        plt.title('AURANAV Dataset - Class Distribution')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Distribution plot saved to {save_path}")

        plt.close()


def main():
    """Main entry point"""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='Validate AURANAV YOLO dataset'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset root (default: from config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save validation report JSON'
    )
    parser.add_argument(
        '--plot',
        type=str,
        help='Path to save distribution plot'
    )
    args = parser.parse_args()

    try:
        config = get_config()

        # Determine dataset path
        if args.dataset:
            dataset_path = Path(args.dataset)
        else:
            # Use raw dataset from config
            raw_path = config.get_path('raw_dataset')
            rf_config = config.roboflow_config
            dataset_path = raw_path

        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            logger.info("Run dataset_downloader.py first to download the dataset")
            sys.exit(1)

        # Run validation
        validator = DatasetValidator(dataset_path)
        results = validator.validate_all()

        # Save report if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert Counter objects to dicts for JSON serialization
            for split_data in results['splits'].values():
                if 'class_counts' in split_data:
                    split_data['class_counts'] = dict(split_data['class_counts'])

            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nValidation report saved to {output_path}")

        # Create distribution plot if requested
        if args.plot:
            validator.visualize_distribution(results, Path(args.plot))

        # Exit with appropriate code
        sys.exit(0 if results['passed'] else 1)

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
