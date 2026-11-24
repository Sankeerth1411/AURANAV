#!/usr/bin/env python3
"""
AURANAV Setup Script
Helper script for initial project setup and validation
"""

import os
import sys
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher required")
        return False

    print("✅ Python version OK")
    return True


def check_dependencies():
    """Check if key dependencies are installed"""
    print("\nChecking dependencies...")

    required = [
        'torch',
        'cv2',
        'numpy',
        'ultralytics',
        'roboflow',
        'yaml',
        'pyttsx3'
    ]

    missing = []

    for package in required:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing.append(package)

    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    print("\n✅ All dependencies installed")
    return True


def check_config():
    """Check if config file exists and is valid"""
    print("\nChecking configuration...")

    config_path = Path('config.yaml')

    if not config_path.exists():
        print("❌ config.yaml not found")
        return False

    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        required_sections = ['roboflow', 'paths', 'model', 'training',
                           'inference', 'camera', 'hazard', 'audio']

        missing = [s for s in required_sections if s not in config]

        if missing:
            print(f"❌ Missing config sections: {', '.join(missing)}")
            return False

        print("✅ config.yaml is valid")
        return True

    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False


def check_api_key():
    """Check if Roboflow API key is set"""
    print("\nChecking Roboflow API key...")

    # Check environment variable
    key = os.environ.get('ROBOFLOW_API_KEY')

    if key:
        print("✅ ROBOFLOW_API_KEY found in environment")
        return True

    # Check .env file
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('ROBOFLOW_API_KEY'):
                    print("✅ ROBOFLOW_API_KEY found in .env file")
                    return True

    print("⚠️  ROBOFLOW_API_KEY not found")
    print("Set it with: export ROBOFLOW_API_KEY=your_key")
    print("Or create .env file with: ROBOFLOW_API_KEY=your_key")
    return False


def check_directories():
    """Check/create project directories"""
    print("\nChecking project directories...")

    directories = [
        'data/raw',
        'data/prepared',
        'artifacts/weights',
        'artifacts/logs',
        'artifacts/runs',
        'artifacts/test_results',
        'src',
        'tests'
    ]

    for dir_path in directories:
        path = Path(dir_path)
        if not path.exists():
            print(f"Creating {dir_path}...")
            path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✅ {dir_path}")

    return True


def check_gpu():
    """Check GPU availability"""
    print("\nChecking GPU availability...")

    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print(f"✅ CUDA available: {gpu_name}")
            print(f"   GPU count: {gpu_count}")

            # Check CUDA version
            cuda_version = torch.version.cuda
            print(f"   CUDA version: {cuda_version}")

            return True
        else:
            print("⚠️  No GPU detected, will use CPU")
            print("   (Training will be much slower)")
            return False

    except Exception as e:
        print(f"⚠️  Could not check GPU: {e}")
        return False


def check_camera():
    """Check camera availability"""
    print("\nChecking camera availability...")

    try:
        import cv2

        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                print(f"✅ Camera detected: {width}x{height}")
                cap.release()
                return True
            else:
                print("⚠️  Camera opened but cannot read frames")
                cap.release()
                return False
        else:
            print("⚠️  No camera detected at index 0")
            print("   (Required for real-time detection)")
            return False

    except Exception as e:
        print(f"⚠️  Could not check camera: {e}")
        return False


def test_audio():
    """Test audio system"""
    print("\nTesting audio system...")

    try:
        import pyttsx3
        engine = pyttsx3.init()
        print("✅ Audio engine initialized")

        # Quick test
        print("Speaking test message...")
        engine.say("Audio test")
        engine.runAndWait()
        print("✅ Audio test complete")

        return True

    except Exception as e:
        print(f"⚠️  Audio test failed: {e}")
        print("   (Audio warnings may not work)")
        return False


def print_next_steps(has_api_key, has_gpu, has_camera):
    """Print recommended next steps"""
    print_header("NEXT STEPS")

    print("1. Download Dataset:")
    if has_api_key:
        print("   ✅ API key configured")
        print("   Run: python src/dataset_downloader.py")
    else:
        print("   ⚠️  Set ROBOFLOW_API_KEY first")
        print("   Then run: python src/dataset_downloader.py")

    print("\n2. Generate Proper Dataset in Roboflow:")
    print("   ⚠️  Current version has no train/valid splits")
    print("   Follow instructions in README.md to generate new version")
    print("   Update config.yaml with new version number")

    print("\n3. Validate Dataset:")
    print("   Run: python src/dataset_validator.py")

    print("\n4. Train Model:")
    if has_gpu:
        print("   ✅ GPU available for training")
    else:
        print("   ⚠️  No GPU - training will be slow")
    print("   Run: python src/train_model.py")

    print("\n5. Test Real-Time Detection:")
    if has_camera:
        print("   ✅ Camera available")
    else:
        print("   ⚠️  No camera detected")
    print("   Run: python src/realtime_detector.py")

    print("\n📖 See README.md for detailed instructions")


def main():
    """Run all setup checks"""
    print_header("AURANAV SETUP VALIDATION")

    checks = []

    # Critical checks
    checks.append(("Python Version", check_python_version()))
    checks.append(("Dependencies", check_dependencies()))
    checks.append(("Configuration", check_config()))

    # Important but not critical
    has_api_key = check_api_key()
    checks.append(("API Key", has_api_key))

    checks.append(("Directories", check_directories()))

    has_gpu = check_gpu()
    checks.append(("GPU", has_gpu))

    has_camera = check_camera()
    checks.append(("Camera", has_camera))

    checks.append(("Audio", test_audio()))

    # Summary
    print_header("SETUP SUMMARY")

    passed = sum(1 for _, result in checks if result)
    total = len(checks)

    for name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All checks passed!")
    elif passed >= total - 2:
        print("\n⚠️  Setup mostly complete, some optional features unavailable")
    else:
        print("\n❌ Setup incomplete, please address issues above")

    # Next steps
    print_next_steps(has_api_key, has_gpu, has_camera)


if __name__ == "__main__":
    main()
