# AURANAV — Smart Glasses AI System for Visual Assistance

> **AURANAV 2.0 development branch:** this repository is an experimental assistive-vision prototype. The primary detector remains the safety-alert source. An optional YOLO-World discovery model is available for research, disabled by default, and kept separate from spoken hazard alerts. See [Safety architecture](docs/SAFETY_ARCHITECTURE.md) and [Open-vocabulary discovery](docs/OPEN_VOCABULARY.md).

## Current verified baseline and limitations

The repository's saved evaluation report records mAP@0.5 of 61.01%, mAP@0.5:0.95 of 41.23%, precision of 63.73%, recall of 58.16%, and approximately 11.13 FPS on the machine used for that benchmark. These are historical evaluation results, not guaranteed live performance. Per-class performance varies and weak obstacle classes need more labelled data.

Bounding-box size is not metric distance. Hazard scores are relative alert-priority heuristics, not calibrated collision probabilities or time-to-collision. The detector cannot recognise every object and the experimental discovery model is not safety validated.

**Safety:** AURANAV is not certified and must not replace a white cane, guide dog, or orientation-and-mobility training. Test only in controlled environments with a sighted supervisor.

## Development-branch improvements

- Class-aware IoU tracking, invalid-box filtering, track confirmation and stale-track expiry.
- Conservative hazard prioritisation and critical-alert queue priority.
- Cross-platform offline speech through pyttsx3, subject to the operating system speech backend.
- Optional prompt-based object discovery, disabled by default and not connected to hazard alerts.
- Regression tests for tracker and discovery helpers.

---

## Dataset quality audit (new)

Before training on a Roboflow export, run the read-only dataset audit:

```bash
python src/dataset_audit.py --data path/to/data.yaml --output artifacts/test_results/dataset_audit.json
```

To make the command return a failing exit code when annotation errors or cross-split duplicate images are found:

```bash
python src/dataset_audit.py --data path/to/data.yaml --fail-on-errors
```

The report checks configured split paths, image/label pairing, malformed YOLO rows, class IDs, normalised bounding boxes, class counts and byte-identical images shared across splits. Review its output before training; it does not automatically repair annotations or guarantee dataset quality.

## Legacy project guide

The detailed setup and dataset instructions below are retained from the original project documentation. Some dataset-version notes, class counts, dependency versions and performance expectations may refer to an earlier project state; verify them against the current `config.yaml`, dataset export and evaluation report before relying on them.


**Real-time AI-powered obstacle detection and hazard warning system for visually impaired users.**

## 🎯 Project Overview

AURANAV is an intelligent navigation assistance system that uses computer vision and edge AI to help visually impaired people navigate safely. The system identifies nearby objects, assesses collision risks, tracks movement, and provides spoken warnings about potential hazards.

### Key Features

- **Real-time Object Detection**: YOLOv8-based detection of 28 object classes
- **Multi-Object Tracking**: Persistent tracking across frames with motion analysis
- **Intelligent Hazard Scoring**: Risk assessment based on object type, proximity, position, and movement
- **Audio Warnings**: Spoken alerts with smart cooldown to prevent spam
- **Edge AI Ready**: Optimized for resource-constrained devices like Raspberry Pi 5

## 📋 Current Status

### ⚠️ Known Issues with Dataset (Version 1)

The current Roboflow dataset version has critical issues that **prevent training**:

1. **Only TEST split exists** - No training or validation data
2. **27 classes instead of 28** - One class is missing
3. **Malformed annotations** - 50+ files with format errors

**You must regenerate the dataset in Roboflow with proper train/valid/test splits before training.**

See detailed instructions in [Dataset Setup](#dataset-setup) section below.

## 🚀 Installation

### Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU
- Webcam or USB camera
- Linux, macOS, or Windows

### Step 1: Clone or Extract Project

```bash
# If you have the project as a zip
cd auranav/
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Linux/macOS
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- Ultralytics YOLOv8
- PyTorch and OpenCV
- Roboflow SDK
- Audio libraries (pyttsx3, pygame)
- Tracking libraries (filterpy, scipy)

### Step 4: Configure Environment

Create a `.env` file in the project root:

```bash
# Roboflow API Key (required for dataset download)
ROBOFLOW_API_KEY=your_api_key_here
```

Or set as environment variable:

```bash
export ROBOFLOW_API_KEY=your_api_key_here
```

## 📊 Dataset Setup

### Current Dataset Issues

The current dataset version (v1) downloaded in your notebook has these problems:

```
Dataset: merge_final-1
├── test/          ✅ 4,603 images (but with annotation errors)
├── train/         ❌ MISSING
├── valid/         ❌ MISSING
└── data.yaml      ⚠️  Lists 27 classes (expected: 28)
```

### Fix: Generate New Dataset Version in Roboflow

**Follow these steps to create a properly split dataset:**

1. **Go to Roboflow Project**
   - URL: https://universe.roboflow.com/cracksdetection/merge_final-iixxm

2. **Click "Generate" to create new version**
   - This will be Version 2 (or next available)

3. **Configure Train/Valid/Test Split**
   ```
   Train:      70%  (recommended)
   Validation: 20%
   Test:       10%
   ```

4. **Review preprocessing settings**
   - Auto-orient: Yes
   - Resize: 640x640 (or keep current)

5. **Review augmentation settings** (optional but recommended)
   - Flip: Horizontal
   - Rotation: ±15°
   - Brightness: ±15%
   - Keep other defaults

6. **Click "Generate" and wait for completion**

7. **Update config.yaml with new version number**
   ```yaml
   roboflow:
     workspace: "cracksdetection"
     project: "merge_final-iixxm"
     version: 2  # ← Update this to your new version
     format: "yolov8"
   ```

8. **Verify the 28th class**
   - Check class list in Roboflow
   - Ensure all intended classes are annotated
   - The current 27 classes are listed in `config.yaml` under `hazard.class_severity`

### Download Dataset

Once you've generated a proper version:

```bash
python src/dataset_downloader.py
```

This will:
- Download dataset from Roboflow
- Inspect structure and contents
- Report any issues
- Save to `data/raw/merge_final-{version}/`

### Validate Dataset

```bash
python src/dataset_validator.py
```

This will:
- Check all splits exist
- Validate annotation format
- Check for missing labels
- Analyze class distribution
- Generate validation report

**Only proceed to training after validation passes!**

## 🎓 Training

### Train Model

```bash
python src/train_model.py
```

**Configuration** (in `config.yaml`):

```yaml
model:
  architecture: "yolov8n"  # nano model for speed
  input_size: 640

training:
  epochs: 100
  batch_size: 16  # Adjust based on GPU memory
  patience: 20    # Early stopping
  device: "auto"  # auto-detects GPU/CPU
```

**Options:**

```bash
# Resume from checkpoint
python src/train_model.py --resume

# Train from scratch (no pretrained weights)
python src/train_model.py --no-pretrained

# Use custom dataset path
python src/train_model.py --dataset /path/to/dataset
```

**Training Output:**
- Model weights: `artifacts/weights/auranav_best.pt`
- Training logs: `artifacts/runs/train/auranav/`
- Checkpoints: `artifacts/weights/last.pt`

### Evaluate Model

```bash
python src/train_model.py --eval-only
```

**With speed benchmark:**

```bash
python src/train_model.py --eval-only --benchmark
```

This reports:
- mAP@0.5 and mAP@0.5:0.95
- Precision and Recall
- Per-class AP
- Inference speed (FPS)

## 🎥 Real-Time Detection

### Run Live Camera Detection

```bash
python src/realtime_detector.py
```

**This will:**
1. Load trained model
2. Open default webcam
3. Detect objects in real-time
4. Track objects across frames
5. Score hazard risks
6. Speak audio warnings for dangerous obstacles

**Press 'q' to quit**

### Options

```bash
# Use specific camera
python src/realtime_detector.py --camera 1

# Use video file instead of camera
python src/realtime_detector.py --camera video.mp4

# Audio only (no display)
python src/realtime_detector.py --no-display

# Process limited frames for testing
python src/realtime_detector.py --max-frames 100

# Use specific model weights
python src/realtime_detector.py --weights path/to/weights.pt
```

### Camera Access Troubleshooting

**Linux:**
```bash
# Check camera devices
ls /dev/video*

# Add user to video group
sudo usermod -a -G video $USER
```

**Windows:**
- Ensure camera is not in use by another app
- Check camera permissions in Windows Settings

**macOS:**
- Grant terminal camera access in System Preferences → Security & Privacy

## 🧪 Testing

### Test Audio System

```bash
python src/audio_system.py
```

**Test all alert types:**
```bash
python src/audio_system.py --test-all
```

**Custom message:**
```bash
python src/audio_system.py --message "Testing AURANAV audio"
```

### Test Hazard Scorer

```bash
python src/hazard_scorer.py
```

Shows class severity map and scoring multipliers.

### Test Configuration

```bash
python src/config_loader.py
```

Validates configuration file and shows loaded settings.

## ⚙️ Configuration

All settings are in `config.yaml`. Key sections:

### Model Settings
```yaml
model:
  architecture: "yolov8n"  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
  input_size: 640
```

### Hazard Scoring
```yaml
hazard:
  class_severity:
    pothole: 90          # Most dangerous
    "bad road": 85
    vehicle: 80
    # ... customize per class
    
  alert:
    critical_threshold: 70  # Immediate warning
    warning_threshold: 50   # Cautionary alert
    info_threshold: 30      # Low priority
```

### Audio Warnings
```yaml
audio:
  enabled: true
  engine: "pyttsx3"  # or "gtts"
  rate: 180          # Words per minute
  
  cooldown:
    same_object: 5.0   # Seconds before repeating same object
    same_class: 3.0    # Seconds before alerting same class
    any_alert: 1.0     # Minimum time between any alerts
```

### Camera Settings
```yaml
camera:
  source: 0      # Camera index or video file path
  width: 640
  height: 480
  fps: 30
```

## 📁 Project Structure

```
auranav/
├── config.yaml                   # Main configuration file
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── src/                          # Source code
│   ├── config_loader.py          # Configuration management
│   ├── dataset_downloader.py     # Download from Roboflow
│   ├── dataset_validator.py      # Validate dataset integrity
│   ├── train_model.py            # Model training pipeline
│   ├── object_tracker.py         # Multi-object tracking (SORT-like)
│   ├── hazard_scorer.py          # Risk scoring engine
│   ├── audio_system.py           # Text-to-speech warnings
│   └── realtime_detector.py      # Main detection application
│
├── data/                         # Dataset storage
│   ├── raw/                      # Downloaded raw datasets
│   └── prepared/                 # Processed datasets
│
├── artifacts/                    # Training outputs
│   ├── weights/                  # Model weights
│   │   └── auranav_best.pt      # Best trained model
│   ├── logs/                     # Training logs
│   ├── runs/                     # Training runs
│   └── test_results/             # Evaluation results
│
└── tests/                        # Automated tests
    └── (to be added)
```

## 🎯 28 Detected Object Classes

The system is designed to detect these 28 classes (current dataset has 27):

1. bad road
2. bench
3. bicycle
4. bird
5. bollard
6. bus
7. car
8. cat
9. crosswalk
10. dog
11. down stairs
12. fence barrier
13. fire hydrant
14. motorcycle
15. parking meter
16. person
17. pole
18. pothole
19. scooter
20. stairs
21. stop sign
22. traffic cone
23. traffic light
24. trash bin
25. tree
26. truck
27. up stairs
28. **[MISSING CLASS]** ← Needs to be identified and added

## 🔧 Raspberry Pi 5 Deployment (Optional)

### Requirements
- Raspberry Pi 5 (8GB recommended)
- Raspberry Pi Camera Module or USB webcam
- microSD card (32GB+)
- Active cooling recommended

### Setup

1. **Install Raspberry Pi OS (64-bit)**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   ```

2. **Install dependencies**
   ```bash
   # System packages
   sudo apt install -y python3-pip python3-opencv portaudio19-dev

   # Python packages
   pip3 install -r requirements.txt
   ```

3. **For Pi Camera Module**
   ```bash
   # Install picamera2
   pip3 install picamera2
   ```

4. **Enable camera**
   ```bash
   sudo raspi-config
   # Interface Options → Camera → Enable
   ```

5. **Optimize config.yaml for Pi**
   ```yaml
   model:
     architecture: "yolov8n"  # Use nano model
     input_size: 416          # Smaller for speed
   
   training:
     device: "cpu"            # Pi doesn't have CUDA
   
   inference:
     half_precision: false    # CPU doesn't support FP16
   
   camera:
     width: 416
     height: 416
     fps: 15                  # Lower FPS for stability
   ```

6. **Run on Pi**
   ```bash
   python3 src/realtime_detector.py
   ```

### Performance Expectations

- **Pi 5 + YOLOv8n + 416x416**: ~5-10 FPS
- **Pi 4**: ~2-5 FPS (slower)
- GPU acceleration not available on Pi

### Tips for Pi Optimization

- Use model quantization (INT8) for 2-3x speedup
- Lower input resolution to 320x320 if needed
- Disable tracking if too slow
- Use hardware-accelerated camera access

## 🚨 Important Limitations

### Distance Estimation
- **Bounding box size is NOT accurate distance**
- System reports "close" vs "far" based on bbox size
- This is a proxy for proximity, not metric distance
- Monocular depth estimation could improve this (future work)

### Camera Calibration
- No camera calibration performed
- Direction estimates (left/center/right) are relative to frame
- Not calibrated to user's actual walking path

### Detection Limitations
- Depends on training data quality
- May miss objects at extreme angles
- May confuse similar objects
- Performance degrades in low light
- Small or distant objects may not be detected

### Real-Time Requirements
- Requires sufficient computational resources
- Audio latency possible on slow devices
- Camera frame drops may occur

## 🐛 Troubleshooting

### "Model weights not found"
→ Train the model first: `python src/train_model.py`

### "Dataset not found"
→ Download dataset: `python src/dataset_downloader.py`

### "ROBOFLOW_API_KEY not found"
→ Set environment variable or create `.env` file

### "Failed to open camera"
- Check camera is connected
- Try different camera index: `--camera 1`
- Check permissions (Linux: add to video group)
- Close other apps using camera

### Audio not working
- Check pyttsx3 is installed: `pip install pyttsx3`
- On Linux: `sudo apt install espeak`
- Test audio: `python src/audio_system.py --test-all`
- Disable in config if not needed: `audio.enabled: false`

### Low FPS / Slow inference
- Use smaller model: `yolov8n` instead of `yolov8m`
- Reduce input size in config: `model.input_size: 416`
- Disable tracking: `tracking.enabled: false`
- Check GPU is being used (check logs)

### CUDA out of memory
- Reduce batch size: `training.batch_size: 8`
- Use smaller model architecture
- Close other GPU applications

## 📚 Next Steps

### Before You Can Train:
1. ✅ Generate proper dataset in Roboflow with train/valid/test splits
2. ✅ Update `config.yaml` with new version number
3. ✅ Download and validate dataset
4. ✅ Identify and add the missing 28th class

### After Training:
1. Run evaluation to check model quality
2. Test on live camera with various objects
3. Fine-tune hazard scoring thresholds
4. Adjust audio warning cooldowns based on usage
5. Consider model quantization for deployment
6. Add more training data for underrepresented classes

### Future Enhancements:
- Depth estimation for accurate distance measurement
- GPS integration for outdoor navigation
- Path planning and route guidance
- Open-vocabulary recognition for unknown objects
- Multi-camera support for wider field of view
- Cloud sync for model updates
- Mobile app interface

## 📝 Development Notes

### What Has Been Tested:
- ✅ Configuration loading
- ✅ Dataset structure inspection
- ✅ Validation logic
- ⚠️ Training pipeline (blocked by dataset issues)
- ⚠️ Real-time detection (requires trained model)

### What Requires Physical Testing:
- Camera access on target device
- Audio output quality and timing
- Inference speed on target hardware (CPU/GPU/Pi)
- User experience with actual obstacles
- Alert cooldown tuning for real usage

### Known Dataset Issues to Fix:
1. Generate new version with proper splits (70/20/10)
2. Fix annotation format errors in affected files
3. Verify all 28 classes are present and annotated
4. Check for class imbalance and add samples if needed

## 🤝 Contributing

This is a research/development project. Key areas for contribution:

- Dataset annotation and quality improvement
- Model optimization for edge devices
- Depth estimation integration
- User testing and feedback
- Documentation improvements

## 📄 License

CC BY 4.0 (as per Roboflow dataset license)

## ⚠️ Disclaimer

**This is a prototype system for research and development purposes.**

- Not a replacement for white cane, guide dog, or professional mobility training
- Not certified medical device
- Users should receive proper orientation and mobility training
- System limitations must be understood before real-world use
- Always use in safe environments during testing

---

**Built with:** YOLOv8 • OpenCV • PyTorch • Roboflow • pyttsx3

**For questions or issues, refer to the troubleshooting section above.**
