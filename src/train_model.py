#!/usr/bin/env python3
"""
AURANAV - Model Trainer
Train YOLOv8 model for obstacle detection
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict
import time

import torch
from ultralytics import YOLO

sys.path.append(str(Path(__file__).parent.parent))
from src.config_loader import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train YOLO model for AURANAV"""

    def __init__(self):
        """Initialize trainer"""
        self.config = get_config()
        self.model_config = self.config.model_config
        self.training_config = self.config.training_config

        # Set device
        self.device = self._setup_device()
        logger.info(f"Using device: {self.device}")

    def _setup_device(self) -> str:
        """Setup computation device"""
        device_config = self.training_config.get('device', 'auto')

        if device_config == 'auto':
            if torch.cuda.is_available():
                device = '0'
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                logger.warning("CUDA not available, using CPU (training will be slow)")
        else:
            device = str(device_config)

        return device

    def prepare_data_yaml(self, dataset_path: Path) -> Path:
        """
        Prepare data.yaml for training with absolute paths

        Args:
            dataset_path: Path to dataset root

        Returns:
            Path to prepared data.yaml
        """
        import yaml

        # Read original data.yaml
        original_yaml = dataset_path / "data.yaml"
        with open(original_yaml, 'r') as f:
            data_config = yaml.safe_load(f)

        # Update paths to absolute
        data_config['path'] = str(dataset_path.absolute())
        data_config['train'] = 'train/images'
        data_config['val'] = 'valid/images'
        data_config['test'] = 'test/images'

        # Save to artifacts
        artifacts_path = self.config.get_path('artifacts')
        prepared_yaml = artifacts_path / 'data_prepared.yaml'

        with open(prepared_yaml, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

        logger.info(f"Prepared data.yaml saved to {prepared_yaml}")
        return prepared_yaml

    def train(self,
              data_yaml: Path,
              resume: bool = False,
              pretrained: bool = True) -> Dict:
        """
        Train YOLO model

        Args:
            data_yaml: Path to data.yaml file
            resume: Resume from last checkpoint
            pretrained: Use pretrained weights

        Returns:
            Dictionary with training results
        """
        logger.info("=" * 60)
        logger.info("STARTING MODEL TRAINING")
        logger.info("=" * 60)

        # Model architecture
        architecture = self.model_config['architecture']
        logger.info(f"Architecture: {architecture}")

        # Initialize model
        if resume:
            # Resume from last checkpoint
            weights_path = self.config.get_path('weights') / 'last.pt'
            if weights_path.exists():
                logger.info(f"Resuming from checkpoint: {weights_path}")
                model = YOLO(str(weights_path))
            else:
                logger.warning(f"Checkpoint not found at {weights_path}, starting fresh")
                model = YOLO(f'{architecture}.pt' if pretrained else f'{architecture}.yaml')
        else:
            # Start fresh
            if pretrained:
                logger.info(f"Loading pretrained weights: {architecture}.pt")
                model = YOLO(f'{architecture}.pt')
            else:
                logger.info(f"Training from scratch: {architecture}.yaml")
                model = YOLO(f'{architecture}.yaml')

        # Training parameters
        epochs = self.training_config['epochs']
        batch_size = self.training_config['batch_size']
        patience = self.training_config['patience']
        workers = self.training_config['workers']
        amp = self.training_config['amp']
        save_period = self.training_config.get('save_period', 10)

        # Get image size
        img_size = self.model_config.get('input_size', 640)

        # Project and name for saving
        artifacts_path = self.config.get_path('artifacts')
        project_path = artifacts_path / 'runs' / 'train'

        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Image size: {img_size}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Workers: {workers}")
        logger.info(f"AMP: {amp}")
        logger.info(f"Patience: {patience}")

        # Start training
        start_time = time.time()

        try:
            results = model.train(
                data=str(data_yaml),
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                device=self.device,
                workers=workers,
                patience=patience,
                save=True,
                save_period=save_period,
                project=str(project_path),
                name='auranav',
                exist_ok=True,
                amp=amp,
                verbose=True
            )

            training_time = time.time() - start_time
            logger.info(f"\nTraining completed in {training_time/3600:.2f} hours")

            # Save best model to standard location
            best_weights = project_path / 'auranav' / 'weights' / 'best.pt'
            last_weights = project_path / 'auranav' / 'weights' / 'last.pt'

            weights_dir = self.config.get_path('weights')

            if best_weights.exists():
                import shutil
                dest_best = weights_dir / 'auranav_best.pt'
                shutil.copy(best_weights, dest_best)
                logger.info(f"Best weights saved to: {dest_best}")

            if last_weights.exists():
                import shutil
                dest_last = weights_dir / 'last.pt'
                shutil.copy(last_weights, dest_last)
                logger.info(f"Last checkpoint saved to: {dest_last}")

            return {
                'success': True,
                'training_time': training_time,
                'best_weights': str(dest_best) if best_weights.exists() else None,
                'results_dir': str(project_path / 'auranav')
            }

        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def evaluate(self, weights_path: Path, data_yaml: Path) -> Dict:
        """
        Evaluate trained model

        Args:
            weights_path: Path to model weights
            data_yaml: Path to data.yaml

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("=" * 60)
        logger.info("EVALUATING MODEL")
        logger.info("=" * 60)

        if not weights_path.exists():
            logger.error(f"Weights not found: {weights_path}")
            return {'success': False, 'error': 'Weights not found'}

        try:
            # Load model
            model = YOLO(str(weights_path))

            # Validate on test set
            logger.info("Running validation on test set...")
            results = model.val(
                data=str(data_yaml),
                split='test',
                device=self.device,
                verbose=True
            )

            # Extract metrics
            metrics = {
                'success': True,
                'mAP50': float(results.box.map50),
                'mAP50-95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'per_class_AP': {}
            }

            # Per-class AP
            if hasattr(results.box, 'ap_class_index'):
                for idx, ap in zip(results.box.ap_class_index, results.box.ap):
                    metrics['per_class_AP'][int(idx)] = float(ap)

            logger.info(f"\nEvaluation Results:")
            logger.info(f"  mAP@0.5: {metrics['mAP50']:.4f}")
            logger.info(f"  mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")

            return metrics

        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def benchmark_inference(self, weights_path: Path, num_iterations: int = 100) -> Dict:
        """
        Benchmark inference speed

        Args:
            weights_path: Path to model weights
            num_iterations: Number of inference iterations

        Returns:
            Dictionary with benchmark results
        """
        logger.info("=" * 60)
        logger.info("BENCHMARKING INFERENCE")
        logger.info("=" * 60)

        if not weights_path.exists():
            logger.error(f"Weights not found: {weights_path}")
            return {'success': False, 'error': 'Weights not found'}

        try:
            import numpy as np

            # Load model
            model = YOLO(str(weights_path))
            model.to(self.device)

            # Create dummy input
            img_size = self.model_config.get('input_size', 640)
            dummy_img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)

            # Warmup
            logger.info("Warming up...")
            for _ in range(10):
                model(dummy_img, verbose=False)

            # Benchmark
            logger.info(f"Running {num_iterations} iterations...")
            times = []

            for _ in range(num_iterations):
                start = time.time()
                model(dummy_img, verbose=False)
                times.append(time.time() - start)

            times = np.array(times)

            results = {
                'success': True,
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times)),
                'fps': float(1.0 / np.mean(times)),
                'device': self.device
            }

            logger.info(f"\nBenchmark Results:")
            logger.info(f"  Mean time: {results['mean_time']*1000:.2f} ms")
            logger.info(f"  Std dev: {results['std_time']*1000:.2f} ms")
            logger.info(f"  FPS: {results['fps']:.1f}")

            return results

        except Exception as e:
            logger.error(f"Benchmark failed: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}


def main():
    """Main entry point"""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='Train AURANAV YOLO model'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset root (default: from config)'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Train from scratch without pretrained weights'
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only evaluate existing model'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Benchmark inference speed'
    )
    args = parser.parse_args()

    try:
        config = get_config()
        trainer = ModelTrainer()

        # Determine dataset path
        if args.dataset:
            dataset_path = Path(args.dataset)
        else:
            raw_path = config.get_path('raw_dataset')
            rf_config = config.roboflow_config
            dataset_path = raw_path / f"{rf_config['project']}-{rf_config['version']}"

        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            logger.info("Run dataset_downloader.py first")
            sys.exit(1)

        # Prepare data.yaml
        data_yaml = trainer.prepare_data_yaml(dataset_path)

        # Weights path
        weights_path = config.get_path('weights') / 'auranav_best.pt'

        if args.eval_only:
            # Evaluation only
            if not weights_path.exists():
                logger.error(f"No trained model found at {weights_path}")
                sys.exit(1)

            metrics = trainer.evaluate(weights_path, data_yaml)

            if args.benchmark:
                bench_results = trainer.benchmark_inference(weights_path)
                metrics['benchmark'] = bench_results

            # Save results
            results_dir = config.get_path('test_results')
            results_file = results_dir / 'evaluation_results.json'
            with open(results_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"\nResults saved to {results_file}")

        else:
            # Training
            results = trainer.train(
                data_yaml,
                resume=args.resume,
                pretrained=not args.no_pretrained
            )

            if results['success']:
                logger.info("\n✅ Training completed successfully!")

                # Auto-evaluate
                logger.info("\nRunning automatic evaluation...")
                metrics = trainer.evaluate(weights_path, data_yaml)

                if args.benchmark:
                    bench_results = trainer.benchmark_inference(weights_path)
                    metrics['benchmark'] = bench_results

                # Save combined results
                results_dir = config.get_path('test_results')
                results_file = results_dir / 'training_results.json'

                combined = {
                    'training': results,
                    'evaluation': metrics
                }

                with open(results_file, 'w') as f:
                    json.dump(combined, f, indent=2)
                logger.info(f"\nResults saved to {results_file}")

                sys.exit(0)
            else:
                logger.error("\n❌ Training failed")
                sys.exit(1)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
