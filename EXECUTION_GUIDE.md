# AURANAV - Complete Step-by-Step Execution Guide
# Follow these steps in order

## 📍 STEP 0: Navigate to Project Directory

```bash
cd C:\Users\rsath\AppData\Local\Claude-3p\local-agent-mode-sessions\7083ffb1\00000000\fc7f191d\outputs\auranav
```

Or if you're in Linux/WSL:
```bash
cd /sessions/confident-serene-cray/mnt/outputs/auranav
```

---

## 🔧 STEP 1: Install Dependencies (5 minutes)

### Create Virtual Environment (Recommended)

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Windows Command Prompt:**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**Linux/macOS/WSL:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Required Packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- ultralytics (YOLOv8)
- torch, torchvision
- opencv-python
- roboflow
- pyttsx3 (audio)
- filterpy, scipy (tracking)
- pandas, matplotlib
- pytest

**Expected time: 3-5 minutes**

---

## 🔑 STEP 2: Set API Key

### Option A: Environment Variable (Recommended)

**Windows PowerShell:**
```powershell
$env:ROBOFLOW_API_KEY="your_actual_api_key_here"
```

**Windows Command Prompt:**
```cmd
set ROBOFLOW_API_KEY=your_actual_api_key_here
```

**Linux/macOS/WSL:**
```bash
export ROBOFLOW_API_KEY="your_actual_api_key_here"
```

### Option B: Create .env File

```bash
# Copy example file
cp .env.example .env

# Edit .env file and add your key
# ROBOFLOW_API_KEY=your_actual_api_key_here
```

**Get your API key from:** https://app.roboflow.com/settings/api

---

## ✅ STEP 3: Validate Setup (1 minute)

```bash
python setup.py
```

**What this checks:**
- ✅ Python version (3.8+)
- ✅ All dependencies installed
- ✅ config.yaml is valid
- ✅ API key is set
- ✅ GPU availability
- ✅ Camera access
- ✅ Audio system

**Expected output:**
```
============================================================
  AURANAV SETUP VALIDATION
============================================================

Checking Python version...
Python 3.10.x
✅ Python version OK

Checking dependencies...
✅ torch
✅ cv2
✅ numpy
...

✅ All checks passed!
```

**If any checks fail, see troubleshooting at the end.**

---

## 🚨 STEP 4: Fix Dataset in Roboflow (15 minutes) - CRITICAL!

**Your current dataset CANNOT be used for training!** It only has a test split.

### 4.1: Generate New Version in Roboflow

1. **Go to your project:**
   https://universe.roboflow.com/cracksdetection/merge_final-iixxm

2. **Click "Generate" button** (top right)

3. **Configure Train/Valid/Test Split:**
   ```
   Train:      70%
   Validation: 20%
   Test:       10%
   ```

4. **Review Preprocessing:**
   - Auto-Orient: ✅ Yes
   - Resize: 640x640 (recommended)

5. **Review Augmentation (Optional but Recommended):**
   - Flip: Horizontal ✅
   - Rotation: ±15°
   - Brightness: ±15%
   - (Keep other defaults)

6. **Click "Generate"** and wait (5-10 minutes)

### 4.2: Update Config with New Version

```bash
# Open config.yaml and change version number
# From:
#   version: 1
# To:
#   version: 2  (or whatever version number Roboflow assigned)
```

**Or use this command:**

**Windows:**
```powershell
(Get-Content config.yaml) -replace 'version: 1', 'version: 2' | Set-Content config.yaml
```

**Linux/macOS:**
```bash
sed -i 's/version: 1/version: 2/' config.yaml
```

---

## 📥 STEP 5: Download Dataset (5 minutes)

```bash
python src/dataset_downloader.py
```

**What this does:**
- Downloads dataset from Roboflow
- Inspects structure
- Reports any issues
- Saves to `data/raw/merge_final-{version}/`

**Expected output:**
```
============================================================
DATASET INSPECTION
============================================================

Dataset root: data/raw/merge_final-2

Top-level contents:
  data.yaml
  test
  train
  valid

data.yaml found
Number of classes: 28 (or 27)

Dataset splits:
  TRAIN: 3221 images, 3221 labels
  VALID: 920 images, 920 labels
  TEST: 462 images, 462 labels

✅ Dataset downloaded and validated successfully!
```

---

## 🔍 STEP 6: Validate Dataset (2 minutes)

```bash
python src/dataset_validator.py
```

**What this checks:**
- All splits exist (train/valid/test)
- Label format is correct
- No missing labels
- Class distribution
- Annotation errors

**Expected output:**
```
============================================================
DATASET VALIDATION
============================================================

Validating TRAIN split...
  Images: 3221
  Labels: 3221
  Errors: 0
  Missing labels: 0

Validating VALID split...
  Images: 920
  Labels: 920
  Errors: 0
  Missing labels: 0

Validating TEST split...
  Images: 462
  Labels: 462
  Errors: 0
  Missing labels: 0

✅ VALIDATION PASSED
```

**⚠️ If validation fails, DO NOT proceed to training!**
Fix issues in Roboflow and regenerate dataset.

---

## 🎓 STEP 7: Train Model (1-4 hours)

### Option A: Default Training (GPU Recommended)

```bash
python src/train_model.py
```

### Option B: Custom Settings

```bash
# Train on CPU (slower)
# Edit config.yaml: training.device: "cpu"

# Use smaller batch size if out of memory
# Edit config.yaml: training.batch_size: 8

# Quick test with fewer epochs
# Edit config.yaml: training.epochs: 10
```

### Option C: Resume from Checkpoint

```bash
python src/train_model.py --resume
```

**What happens during training:**
- Loads YOLOv8 nano model with pretrained weights
- Trains for 100 epochs (default)
- Shows progress with loss curves
- Saves best model to `artifacts/weights/auranav_best.pt`
- Saves checkpoints every 10 epochs
- Auto-evaluates on test set when done

**Expected output:**
```
============================================================
STARTING MODEL TRAINING
============================================================

Architecture: yolov8n
Loading pretrained weights: yolov8n.pt
Model loaded successfully (device: cuda:0)

Epochs: 100
Batch size: 16
Image size: 640
Device: 0
Workers: 4

Epoch   GPU_mem   box_loss   cls_loss   dfl_loss  Instances  Size
  1/100    2.1G      1.234      1.456      1.234        100   640
  2/100    2.1G      1.123      1.345      1.123        100   640
  ...

Training completed in 2.34 hours

Best weights saved to: artifacts/weights/auranav_best.pt
```

**Training time estimates:**
- GPU (RTX 3060+): 1-2 hours
- GPU (GTX 1660+): 2-4 hours
- CPU: 10-20 hours (not recommended)

---

## 📊 STEP 8: Evaluate Model (2 minutes)

```bash
python src/train_model.py --eval-only
```

**With speed benchmark:**
```bash
python src/train_model.py --eval-only --benchmark
```

**Expected output:**
```
============================================================
EVALUATING MODEL
============================================================

Running validation on test set...

Evaluation Results:
  mAP@0.5: 0.6234
  mAP@0.5:0.95: 0.4123
  Precision: 0.7234
  Recall: 0.6543

============================================================
BENCHMARKING INFERENCE
============================================================

Warming up...
Running 100 iterations...

Benchmark Results:
  Mean time: 23.45 ms
  Std dev: 2.34 ms
  FPS: 42.6

Results saved to artifacts/test_results/evaluation_results.json
```

---

## 🎥 STEP 9: Test Audio System (1 minute)

```bash
python src/audio_system.py --test-all
```

**Expected output:**
```
Testing audio system...
Audio engine initialized: pyttsx3 (rate=180, volume=1.0)
Audio worker thread started

Testing: Pothole ahead
Spoke: 'Pothole ahead'

Testing: Vehicle approaching from the left
Spoke: 'Vehicle approaching from the left'

Testing: Obstacle close on your right
Spoke: 'Obstacle close on your right'

✅ Audio test completed successfully
```

**You should hear the spoken messages!**

---

## 🚀 STEP 10: Run Real-Time Detection (Main Application!)

### Basic Usage (Default Webcam)

```bash
python src/realtime_detector.py
```

**What you'll see:**
- Live video feed from webcam
- Bounding boxes around detected objects
- Object IDs and class names
- Risk scores (0-100) with color coding:
  - 🔴 Red: Critical (70+)
  - 🟠 Orange: Warning (50-70)
  - 🟢 Green: Low risk (<50)
- FPS counter
- Object count

**What you'll hear:**
- Audio warnings for hazardous objects
- "Pothole ahead"
- "Vehicle on your left"
- "Person ahead"

**Press 'q' to quit**

### Advanced Options

```bash
# Use different camera
python src/realtime_detector.py --camera 1

# Use video file for testing
python src/realtime_detector.py --camera test_video.mp4

# Audio only (no visual display)
python src/realtime_detector.py --no-display

# Process limited frames for testing
python src/realtime_detector.py --max-frames 500

# Use specific model weights
python src/realtime_detector.py --weights artifacts/weights/custom_model.pt
```

**Expected output:**
```
Loading model from artifacts/weights/auranav_best.pt
Model loaded successfully (device: cuda:0)
Classes: 28
Object tracker initialized

Camera opened: 640x480 @ 30.0 FPS

Starting detection loop (press 'q' to quit)

Processed 100 frames (avg 28.4 FPS)
Processed 200 frames (avg 27.9 FPS)
...
```

---

## 🧪 STEP 11: Run Automated Tests (Optional)

```bash
python tests/test_auranav.py
```

Or using pytest:
```bash
pytest tests/ -v
```

**Expected output:**
```
tests/test_auranav.py::TestConfigLoader::test_config_loads PASSED
tests/test_auranav.py::TestConfigLoader::test_config_sections_exist PASSED
tests/test_auranav.py::TestHazardScorer::test_scorer_initialization PASSED
tests/test_auranav.py::TestHazardScorer::test_class_severity PASSED
...

============ 12 passed in 2.34s ============
```

---

## 🎯 Testing Each Component Individually

### Test Configuration Loading
```bash
python src/config_loader.py
```

### Test Hazard Scorer
```bash
python src/hazard_scorer.py
```

### Test Dataset Downloader (without actual download)
```bash
python src/dataset_downloader.py --help
```

### Test Tracker (needs dataset)
```bash
python src/object_tracker.py
```

---

## 📝 Summary of Commands in Order

```bash
# 1. Setup
cd auranav/
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Configure
export ROBOFLOW_API_KEY="your_key"  # or set in .env

# 3. Validate
python setup.py

# 4. Fix dataset in Roboflow (web browser)
# Update config.yaml version: 1 → 2

# 5. Download & Validate Dataset
python src/dataset_downloader.py
python src/dataset_validator.py

# 6. Train
python src/train_model.py

# 7. Evaluate
python src/train_model.py --eval-only --benchmark

# 8. Test Audio
python src/audio_system.py --test-all

# 9. Run Detection
python src/realtime_detector.py

# 10. Run Tests (optional)
pytest tests/ -v
```

---

## 🐛 Troubleshooting

### "Module not found" errors
```bash
pip install -r requirements.txt
# Or reinstall specific package:
pip install ultralytics opencv-python torch
```

### "ROBOFLOW_API_KEY not found"
```bash
# Set environment variable
export ROBOFLOW_API_KEY="your_key"

# Or create .env file
echo "ROBOFLOW_API_KEY=your_key" > .env
```

### "Camera failed to open"
```bash
# Try different camera index
python src/realtime_detector.py --camera 1

# On Linux, check permissions
sudo usermod -a -G video $USER
# Log out and back in
```

### "CUDA out of memory"
```bash
# Edit config.yaml:
training:
  batch_size: 8  # Reduce from 16
  
# Or force CPU:
training:
  device: "cpu"
```

### "Dataset validation failed"
```bash
# Regenerate dataset in Roboflow with proper splits
# Make sure version number matches in config.yaml
# Re-download: python src/dataset_downloader.py
```

### Audio not working
```bash
# Windows: Install audio drivers
# Linux: sudo apt install espeak

# Test manually
python -c "import pyttsx3; engine = pyttsx3.init(); engine.say('test'); engine.runAndWait()"

# Or disable audio in config.yaml:
audio:
  enabled: false
```

### Low FPS / Slow detection
```bash
# Use smaller model in config.yaml:
model:
  architecture: "yolov8n"  # Already smallest
  input_size: 416  # Reduce from 640

# Disable tracking:
tracking:
  enabled: false
```

---

## 📊 Expected Results

### After Training
- **mAP@0.5**: 0.50-0.70 (depends on dataset quality)
- **Model size**: ~6 MB (YOLOv8n)
- **Training time**: 1-4 hours (GPU)

### During Real-Time Detection
- **FPS**: 20-30 (GPU), 3-5 (CPU)
- **Latency**: 30-50 ms per frame
- **Detections**: 0-10 objects per frame (typical)
- **Audio delay**: <500 ms

---

## 🎓 What Each Module Does

| Module | Purpose | When to Run |
|--------|---------|-------------|
| `setup.py` | Validates environment | Once at start |
| `dataset_downloader.py` | Downloads from Roboflow | After fixing dataset |
| `dataset_validator.py` | Checks dataset quality | Before training |
| `train_model.py` | Trains YOLO model | After validation passes |
| `realtime_detector.py` | Main application | After training |
| `audio_system.py` | Tests audio | Before real-time |
| `hazard_scorer.py` | Shows scoring logic | Anytime |
| `config_loader.py` | Tests configuration | If config issues |

---

## 🎯 Quick Start (TL;DR)

```bash
# 1. Install
cd auranav && pip install -r requirements.txt

# 2. API Key
export ROBOFLOW_API_KEY="your_key"

# 3. Validate
python setup.py

# 4. Fix dataset in Roboflow (browser), update config.yaml

# 5. Download
python src/dataset_downloader.py

# 6. Train
python src/train_model.py

# 7. Run!
python src/realtime_detector.py
```

---

## 📖 Additional Resources

- **Full Documentation**: `README.md`
- **Quick Start**: `QUICKSTART.md`
- **Project Status**: `PROJECT_STATUS.md`
- **Delivery Summary**: `DELIVERY_SUMMARY.md`

---

## 💡 Tips

1. **Always run validation** before training
2. **Monitor training logs** in `artifacts/runs/train/auranav/`
3. **Test with video file** before live camera
4. **Adjust hazard thresholds** in `config.yaml` based on testing
5. **Keep training data backed up**

---

**You're all set! Start with `python setup.py` to validate everything works! 🚀**
