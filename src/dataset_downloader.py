#!/usr/bin/env python3
"""
AURANAV - Dataset Downloader
Downloads and inspects Roboflow dataset with comprehensive validation
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from roboflow import Roboflow

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.config_loader import get_config


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Download and inspect Roboflow dataset"""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize dataset downloader

        Args:
            api_key: Roboflow API key. If None, reads from environment.
        """
        self.config = get_config()
        self.api_key = api_key or self._get_api_key()
        self.rf = Roboflow(api_key=self.api_key)

    def _get_api_key(self) -> str:
        """Get API key from environment or Google Colab secrets"""
        # Try environment variable first
        api_key = os.environ.get('ROBOFLOW_API_KEY')

        if api_key:
            logger.info("API key loaded from environment variable")
            return api_key

        # Try Google Colab secrets
        try:
            from google.colab import userdata
            api_key = userdata.get('ROBOFLOW_API_KEY')
            logger.info("API key loaded from Google Colab secrets")
            return api_key
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Could not load from Colab secrets: {e}")

        # Try .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get('ROBOFLOW_API_KEY')
            if api_key:
                logger.info("API key loaded from .env file")
                return api_key
        except ImportError:
            pass

        raise ValueError(
            "ROBOFLOW_API_KEY not found. Set it as:\n"
            "1. Environment variable: export ROBOFLOW_API_KEY=your_key\n"
            "2. Google Colab secret: userdata.set('ROBOFLOW_API_KEY', 'your_key')\n"
            "3. .env file: ROBOFLOW_API_KEY=your_key"
        )

    def download_dataset(self, force: bool = False) -> Path:
        """
        Download dataset from Roboflow

        Args:
            force: If True, re-download even if dataset exists

        Returns:
            Path to downloaded dataset root
        """
        rf_config = self.config.roboflow_config
        raw_dataset_path = self.config.get_path('raw_dataset')

        workspace = rf_config['workspace']
        project = rf_config['project']
        version = rf_config['version']
        format_type = rf_config['format']

        dataset_dir = raw_dataset_path / f"{project}-{version}"

        # Check if already downloaded
        if dataset_dir.exists() and not force:
            logger.info(f"Dataset already exists at {dataset_dir}")
            logger.info("Use force=True to re-download")
            return dataset_dir

        logger.info(f"Downloading dataset from Roboflow...")
        logger.info(f"Workspace: {workspace}")
        logger.info(f"Project: {project}")
        logger.info(f"Version: {version}")
        logger.info(f"Format: {format_type}")

        try:
            # Get project
            project_obj = self.rf.workspace(workspace).project(project)

            # Download dataset
            dataset = project_obj.version(version).download(
                format_type,
                location=str(raw_dataset_path)
            )

            dataset_path = Path(dataset.location)
            logger.info(f"Dataset downloaded to: {dataset_path}")

            return dataset_path

        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise

    def inspect_dataset(self, dataset_path: Path) -> Dict:
        """
        Inspect downloaded dataset structure and contents

        Args:
            dataset_path: Path to dataset root

        Returns:
            Dictionary with inspection results
        """
        logger.info("=" * 60)
        logger.info("DATASET INSPECTION")
        logger.info("=" * 60)

        inspection = {
            'path': str(dataset_path),
            'exists': dataset_path.exists(),
            'files': [],
            'splits': {},
            'data_yaml': None,
            'classes': [],
            'num_classes': 0,
            'issues': []
        }

        if not dataset_path.exists():
            inspection['issues'].append(f"Dataset path does not exist: {dataset_path}")
            return inspection

        # List top-level contents
        logger.info(f"\nDataset root: {dataset_path}")
        logger.info("\nTop-level contents:")
        for item in sorted(dataset_path.iterdir()):
            inspection['files'].append(item.name)
            logger.info(f"  {item.name}")

        # Check data.yaml
        yaml_path = dataset_path / "data.yaml"
        if yaml_path.exists():
            import yaml
            with open(yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)

            inspection['data_yaml'] = data_config

            # Extract classes
            if 'names' in data_config:
                classes = data_config['names']
                if isinstance(classes, dict):
                    classes = [classes[i] for i in sorted(classes.keys())]
                inspection['classes'] = classes
                inspection['num_classes'] = len(classes)

            logger.info(f"\ndata.yaml found")
            logger.info(f"Number of classes: {inspection['num_classes']}")
            logger.info(f"Classes: {inspection['classes']}")

            # Check expected vs actual classes
            expected_classes = self.config.roboflow_config.get('expected_classes', 28)
            if inspection['num_classes'] != expected_classes:
                issue = f"Expected {expected_classes} classes, but found {inspection['num_classes']}"
                inspection['issues'].append(issue)
                logger.warning(f"⚠️  {issue}")

        else:
            issue = "data.yaml not found"
            inspection['issues'].append(issue)
            logger.error(f"❌ {issue}")

        # Check splits
        logger.info("\nDataset splits:")
        for split in ['train', 'valid', 'test']:
            split_info = self._inspect_split(dataset_path, split)
            inspection['splits'][split] = split_info

            if split_info['exists']:
                logger.info(f"  {split.upper()}: {split_info['num_images']} images, "
                           f"{split_info['num_labels']} labels")
            else:
                logger.warning(f"  {split.upper()}: NOT FOUND")

        # Identify missing splits
        missing_splits = [s for s, info in inspection['splits'].items()
                         if not info['exists']]
        if missing_splits:
            issue = f"Missing splits: {', '.join(missing_splits)}"
            inspection['issues'].append(issue)
            logger.error(f"❌ {issue}")

        # Check for training data
        if not inspection['splits'].get('train', {}).get('exists'):
            issue = "No training split found - cannot train model"
            inspection['issues'].append(issue)
            logger.error(f"❌ {issue}")

        logger.info("\n" + "=" * 60)
        logger.info(f"TOTAL ISSUES FOUND: {len(inspection['issues'])}")
        logger.info("=" * 60)

        return inspection

    def _inspect_split(self, dataset_path: Path, split: str) -> Dict:
        """Inspect a single dataset split"""
        split_info = {
            'exists': False,
            'images_dir': None,
            'labels_dir': None,
            'num_images': 0,
            'num_labels': 0
        }

        split_dir = dataset_path / split
        if not split_dir.exists():
            return split_info

        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        split_info['exists'] = images_dir.exists() and labels_dir.exists()
        split_info['images_dir'] = str(images_dir)
        split_info['labels_dir'] = str(labels_dir)

        if images_dir.exists():
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            split_info['num_images'] = sum(
                1 for p in images_dir.iterdir()
                if p.suffix.lower() in image_extensions
            )

        if labels_dir.exists():
            split_info['num_labels'] = sum(
                1 for p in labels_dir.glob('*.txt')
            )

        return split_info

    def print_roboflow_instructions(self):
        """Print instructions for fixing dataset in Roboflow"""
        logger.info("\n" + "=" * 60)
        logger.info("HOW TO FIX DATASET IN ROBOFLOW")
        logger.info("=" * 60)
        logger.info("""
The current dataset version has issues that must be fixed in Roboflow:

PROBLEMS:
1. Only TEST split exists (no train/valid splits)
2. 27 classes instead of expected 28 (one class missing)
3. Malformed annotations in some files

SOLUTION - Generate New Version with Proper Splits:

1. Go to Roboflow: https://universe.roboflow.com/cracksdetection/merge_final-iixxm

2. Click "Generate" to create a new version

3. In "Train/Valid/Test Split" settings:
   - Train: 70%
   - Valid: 20%
   - Test: 10%

4. Review preprocessing and augmentation settings

5. Click "Generate" to create the new version

6. Update config.yaml with new version number:
   roboflow:
     version: 2  # or whatever version was generated

7. Re-run this script to download the corrected dataset

For the missing 28th class:
- Review your dataset annotations in Roboflow
- Ensure all intended classes are present
- May need to add missing class annotations

For annotation errors:
- Roboflow's generation process should validate annotations
- Check the "Health Check" in Roboflow for details
""")
        logger.info("=" * 60 + "\n")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Download and inspect AURANAV dataset from Roboflow'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if dataset exists'
    )
    parser.add_argument(
        '--api-key',
        type=str,
        help='Roboflow API key (or set ROBOFLOW_API_KEY env var)'
    )
    args = parser.parse_args()

    try:
        # Initialize downloader
        downloader = DatasetDownloader(api_key=args.api_key)

        # Download dataset
        dataset_path = downloader.download_dataset(force=args.force)

        # Inspect dataset
        inspection = downloader.inspect_dataset(dataset_path)

        # Print summary
        if inspection['issues']:
            logger.error(f"\n❌ Dataset has {len(inspection['issues'])} issue(s)")
            for issue in inspection['issues']:
                logger.error(f"   - {issue}")

            # Print fix instructions
            downloader.print_roboflow_instructions()

            sys.exit(1)
        else:
            logger.info("\n✅ Dataset downloaded and validated successfully!")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
