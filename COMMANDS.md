# AURANAV - Command Reference Sheet
# Quick copy-paste commands for each step

## ============================================================
## STEP 1: INSTALLATION
## ============================================================

# Navigate to project
cd C:\Users\rsath\AppData\Local\Claude-3p\local-agent-mode-sessions\7083ffb1\00000000\fc7f191d\outputs\auranav

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\activate

# Activate (Windows CMD)
venv\Scripts\activate.bat

# Activate (Linux/macOS/WSL)
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt


## ============================================================
## STEP 2: SET API KEY
## ============================================================

# Windows PowerShell
$env:ROBOFLOW_API_KEY="paste_your_api_key_here"

# Windows CMD
set ROBOFLOW_API_KEY=paste_your_api_key_here

# Linux/macOS/WSL
export ROBOFLOW_API_KEY="paste_your_api_key_here"

# Or create .env file
echo ROBOFLOW_API_KEY=paste_your_api_key_here > .env


## ============================================================
## STEP 3: VALIDATE SETUP
## ============================================================

python setup.py


## ============================================================
## STEP 4: FIX DATASET (Do this in browser first!)
## ============================================================

# 1. Go to: https://universe.roboflow.com/cracksdetection/merge_final-iixxm
# 2. Click "Generate" → Set splits to 70/20/10 → Generate
# 3. Wait for completion (5-10 minutes)

# Then update config.yaml version number:

# Windows PowerShell
(Get-Content config.yaml) -replace 'version: 1', 'version: 2' | Set-Content config.yaml

# Linux/macOS/WSL
sed -i 's/version: 1/version: 2/' config.yaml

# Or manually edit config.yaml and change line 5 from "version: 1" to "version: 2"


## ============================================================
## STEP 5: DOWNLOAD DATASET
## ============================================================

python src/dataset_downloader.py


## ============================================================
## STEP 6: VALIDATE DATASET
## ============================================================

python src/dataset_validator.py


## ============================================================
## STEP 7: TRAIN MODEL (1-4 hours)
## ============================================================

# Default training (GPU auto-detected)
python src/train_model.py

# Resume from checkpoint
python src/train_model.py --resume

# Train from scratch (no pretrained weights)
python src/train_model.py --no-pretrained


## ============================================================
## STEP 8: EVALUATE MODEL
## ============================================================

# Basic evaluation
python src/train_model.py --eval-only

# With speed benchmark
python src/train_model.py --eval-only --benchmark


## ============================================================
## STEP 9: TEST AUDIO
## ============================================================

# Test all alert types
python src/audio_system.py --test-all

# Test custom message
python src/audio_system.py --message "Testing AURANAV audio system"


## ============================================================
## STEP 10: RUN REAL-TIME DETECTION
## ============================================================

# Default (webcam 0)
python src/realtime_detector.py

# Use different camera
python src/realtime_detector.py --camera 1

# Use video file
python src/realtime_detector.py --camera path/to/video.mp4

# Audio only (no display)
python src/realtime_detector.py --no-display

# Test with limited frames
python src/realtime_detector.py --max-frames 100


## ============================================================
## STEP 11: RUN TESTS (Optional)
## ============================================================

# Run all tests
python tests/test_auranav.py

# Or with pytest
pytest tests/ -v


## ============================================================
## INDIVIDUAL COMPONENT TESTS
## ============================================================

# Test configuration
python src/config_loader.py

# Test hazard scorer
python src/hazard_scorer.py

# Show help for any module
python src/dataset_downloader.py --help
python src/train_model.py --help
python src/realtime_detector.py --help


## ============================================================
## TROUBLESHOOTING COMMANDS
## ============================================================

# Check Python version
python --version

# Check installed packages
pip list | grep -E "ultralytics|torch|opencv"

# Test camera manually
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Failed'); cap.release()"

# Test audio manually
python -c "import pyttsx3; engine = pyttsx3.init(); engine.say('Audio test'); engine.runAndWait(); print('Audio OK')"

# Check CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt


## ============================================================
## CONFIGURATION CHANGES (edit config.yaml)
## ============================================================

# Use CPU instead of GPU
# training:
#   device: "cpu"

# Reduce batch size (if out of memory)
# training:
#   batch_size: 8

# Use smaller model
# model:
#   architecture: "yolov8n"  # Already smallest

# Reduce input size (faster inference)
# model:
#   input_size: 416

# Disable tracking (faster)
# tracking:
#   enabled: false

# Disable audio
# audio:
#   enabled: false


## ============================================================
## FILE LOCATIONS
## ============================================================

# Configuration
config.yaml

# Source code
src/dataset_downloader.py
src/dataset_validator.py
src/train_model.py
src/realtime_detector.py
src/audio_system.py
src/hazard_scorer.py
src/object_tracker.py
src/config_loader.py

# Documentation
README.md              # Complete guide
QUICKSTART.md          # Fast setup
EXECUTION_GUIDE.md     # This file (step-by-step)
PROJECT_STATUS.md      # Project analysis
DELIVERY_SUMMARY.md    # What was delivered

# Data directories
data/raw/              # Downloaded datasets
artifacts/weights/     # Model weights
artifacts/runs/        # Training logs
artifacts/test_results/  # Evaluation results


## ============================================================
## QUICK START (COPY ALL AT ONCE)
## ============================================================

# Windows PowerShell (run line by line)
cd C:\Users\rsath\AppData\Local\Claude-3p\local-agent-mode-sessions\7083ffb1\00000000\fc7f191d\outputs\auranav
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
$env:ROBOFLOW_API_KEY="your_key_here"
python setup.py
# (Fix dataset in Roboflow browser, then continue)
python src/dataset_downloader.py
python src/dataset_validator.py
python src/train_model.py
python src/realtime_detector.py


## ============================================================
## IMPORTANT URLS
## ============================================================

# Roboflow Project
https://universe.roboflow.com/cracksdetection/merge_final-iixxm

# Roboflow API Key
https://app.roboflow.com/settings/api

# Google Colab (your original notebook)
https://colab.research.google.com/drive/1VD_6LHxAo0HBPNF0URFwz3rChHZzuaIc


## ============================================================
## KEYBOARD SHORTCUTS
## ============================================================

# During real-time detection
q - Quit application


## ============================================================
## EXPECTED OUTPUTS
## ============================================================

# setup.py → "✅ All checks passed!"
# dataset_downloader.py → "✅ Dataset downloaded successfully"
# dataset_validator.py → "✅ VALIDATION PASSED"
# train_model.py → "✅ Training completed successfully!"
# realtime_detector.py → Shows live video with detections

## ============================================================
## NEXT STEPS AFTER SETUP
## ============================================================

1. ✅ Run python setup.py - Validate environment
2. ⚠️  Fix dataset in Roboflow (CRITICAL!)
3. 📥 python src/dataset_downloader.py
4. ✅ python src/dataset_validator.py
5. 🎓 python src/train_model.py (1-4 hours)
6. 🎥 python src/realtime_detector.py


## ============================================================
## GETTING HELP
## ============================================================

# Read full documentation
README.md                 # Complete guide (recommended)
EXECUTION_GUIDE.md        # Step-by-step instructions
QUICKSTART.md             # Fast setup

# Check logs
artifacts/logs/auranav.log

# Get command help
python src/train_model.py --help
python src/realtime_detector.py --help

# Run tests
python tests/test_auranav.py
