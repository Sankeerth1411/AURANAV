# AURANAV — Offline AI assistance for safer mobility

AURANAV is a research prototype for smart glasses that combines camera-based object detection, short-term tracking, hazard prioritisation, and offline spoken alerts. It is designed to assist—not replace—a white cane, guide dog, or professional orientation and mobility training.

## Current implementation

- YOLO-based closed-set detector trained on the classes in the configured dataset.
- Class-aware multi-object tracking with confirmation and stale-track expiry.
- Relative hazard prioritisation using class severity, image position, bounding-box area trend, and detector confidence.
- Offline text-to-speech through `pyttsx3` and the host operating system's speech engine.
- Dataset download, validation, training, evaluation, and benchmark scripts.

See [Safety architecture and validation notes](docs/SAFETY_ARCHITECTURE.md) for limitations, recommended upgrades, and field-testing gates.

## Important limitations

- The detector recognises only the classes it was trained to recognise. It does not identify every possible object.
- The committed evaluation report is a baseline, not a guarantee: mAP@0.5 0.6101, mAP@0.5:0.95 0.4123, precision 0.6373, recall 0.5816, and CPU benchmark 11.13 FPS on the machine used for that run. Class-level performance varies substantially.
- Bounding-box size is not metric distance. Hazard scores are relative priorities, not collision probabilities or time-to-collision.
- The current software does not provide validated free-space mapping, calibrated depth, GPS navigation, or certified collision avoidance.

## Requirements

- Python 3.10 or 3.11 recommended.
- Webcam or compatible camera.
- CPU inference is supported; a CUDA GPU can accelerate training/inference where available.
- Offline speech requires an installed OS speech backend. On Raspberry Pi OS/Linux install eSpeak NG (for example `sudo apt install espeak-ng`).

## Install

```bash
git clone https://github.com/Sankeerth1411/AURANAV.git
cd AURANAV
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For inference, place your trained model at `artifacts/weights/auranav_best.pt` (weights are intentionally not committed). Configure `config.yaml` for your camera and target device.

## Run live detection

```bash
python src/realtime_detector.py --camera 0
```

Useful options:

```bash
python src/realtime_detector.py --camera 1
python src/realtime_detector.py --camera path/to/test_video.mp4
python src/realtime_detector.py --no-display --max-frames 300
python src/realtime_detector.py --weights path/to/model.pt
```

Press `q` to stop the display window.

## Dataset and training

Dataset download requires your own Roboflow API key in a local `.env` file. Never commit API keys.

```bash
python src/dataset_downloader.py
python src/dataset_validator.py
python src/train_model.py
python src/train_model.py --eval-only --benchmark
```

Review the generated validation report before training. Verify that dataset class names and IDs match the model's labels. Do not use test data for training or tuning.

## Tests

```bash
pytest -q
```

## Security

- Copy `.env.example` to `.env` and add credentials locally.
- Rotate any API key that was previously committed. Removing a key from the latest file does not remove it from Git history.
- Model weights, datasets, and local environment files should remain untracked.

## Safety notice

AURANAV is an experimental assistive-technology prototype, not a certified medical or mobility device. Test only in controlled environments with a sighted supervisor. Never ask a person to navigate traffic, stairs, or unfamiliar spaces relying on this system alone.
