#!/usr/bin/env python3
"""
AURANAV - Configuration Loader
Loads and validates configuration from config.yaml
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager for AURANAV project"""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration

        Args:
            config_path: Path to config.yaml file. If None, searches in standard locations.
        """
        if config_path is None:
            config_path = self._find_config()

        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        self._create_directories()

    def _find_config(self) -> Path:
        """Find config.yaml in standard locations"""
        search_paths = [
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent / "config.yaml",
            Path.home() / ".auranav" / "config.yaml",
        ]

        for path in search_paths:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"config.yaml not found in any of: {search_paths}"
        )

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _validate_config(self):
        """Validate required configuration sections exist"""
        required_sections = [
            'roboflow', 'paths', 'model', 'training',
            'inference', 'camera', 'hazard', 'audio'
        ]

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")

    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        base_dir = Path(__file__).parent.parent

        for path_key, path_value in self.config['paths'].items():
            full_path = base_dir / path_value
            full_path.mkdir(parents=True, exist_ok=True)

    def get(self, *keys, default=None):
        """
        Get configuration value by keys

        Args:
            *keys: Nested keys to traverse
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            config.get('model', 'architecture')  # Returns 'yolov8n'
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def get_path(self, path_key: str) -> Path:
        """
        Get absolute path for a configured path

        Args:
            path_key: Key in paths section (e.g., 'raw_dataset', 'weights')

        Returns:
            Absolute Path object
        """
        base_dir = Path(__file__).parent.parent
        relative_path = self.config['paths'].get(path_key)

        if relative_path is None:
            raise ValueError(f"Path key not found: {path_key}")

        return base_dir / relative_path

    @property
    def roboflow_config(self) -> Dict[str, Any]:
        """Get Roboflow configuration"""
        return self.config['roboflow']

    @property
    def model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config['model']

    @property
    def training_config(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self.config['training']

    @property
    def inference_config(self) -> Dict[str, Any]:
        """Get inference configuration"""
        return self.config['inference']

    @property
    def camera_config(self) -> Dict[str, Any]:
        """Get camera configuration"""
        return self.config['camera']

    @property
    def tracking_config(self) -> Dict[str, Any]:
        """Get tracking configuration"""
        return self.config['tracking']

    @property
    def hazard_config(self) -> Dict[str, Any]:
        """Get hazard scoring configuration"""
        return self.config['hazard']

    @property
    def audio_config(self) -> Dict[str, Any]:
        """Get audio configuration"""
        return self.config['audio']

    def __repr__(self) -> str:
        return f"Config(config_path={self.config_path})"


# Singleton instance
_config_instance = None


def get_config(config_path: str = None) -> Config:
    """
    Get or create singleton Config instance

    Args:
        config_path: Path to config.yaml (only used on first call)

    Returns:
        Config instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = Config(config_path)

    return _config_instance


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()

    print("Configuration loaded successfully!")
    print(f"Config path: {config.config_path}")
    print(f"Model architecture: {config.get('model', 'architecture')}")
    print(f"Training epochs: {config.get('training', 'epochs')}")
    print(f"Weights directory: {config.get_path('weights')}")
