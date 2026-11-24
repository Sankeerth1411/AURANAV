# AURANAV - Quick Start Guide

## 🚀 Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Roboflow API Key

```bash
export ROBOFLOW_API_KEY=your_key_here
```

### 3. Run Setup Validation

```bash
python setup.py
```

This checks your installation and shows what's ready.

## ⚠️ CRITICAL: Dataset Must Be Fixed First

**Your current dataset (v1) cannot be used for training.** It only has a test split.

### Fix the Dataset:

1. Go to: https://universe.roboflow.com/cracksdetection/merge_final-iixxm
2. Click **"Generate"** to create Version 2
3. Set splits:
   - Train: 70%
   - Valid: 20%
   - Test: 10%
4. Click **"Generate"** and wait
5. Update `config.yaml`:
   ```yaml
   roboflow:
     version: 2  # Change from 1 to 2
   ```

### Then Download Dataset:

```bash
python src/dataset_downloader.py
```

### Validate Dataset:

```bash
python src/dataset_validator.py
```

**Only proceed to training after validation passes!**

## 🎓 Train the Model

```bash
python src/train_model.py
```

This takes 1-4 hours depending on GPU.

## 🎥 Run Real-Time Detection

```bash
python src/realtime_detector.py
```

Press 'q' to quit.

## 📚 Full Documentation

See [README.md](README.md) for complete instructions.

## 🐛 Common Issues

**"Dataset not found"**
→ Run: `python src/dataset_downloader.py`

**"Model weights not found"**
→ Train first: `python src/train_model.py`

**"Camera failed to open"**
→ Try: `python src/realtime_detector.py --camera 1`

**"CUDA out of memory"**
→ Reduce batch size in config.yaml: `batch_size: 8`

## 📁 Project Files

```
auranav/
├── src/
│   ├── dataset_downloader.py  # Download from Roboflow
│   ├── dataset_validator.py   # Validate dataset
│   ├── train_model.py          # Train YOLO model
│   └── realtime_detector.py    # Main application
├── config.yaml                 # All settings
├── requirements.txt            # Dependencies
├── setup.py                    # Setup validation
└── README.md                   # Full documentation
```

## 🎯 Expected Results

After proper training on a corrected dataset:

- **Detection**: 28 object classes
- **Speed**: 20-30 FPS on GPU, 3-5 FPS on CPU
- **Accuracy**: mAP@0.5 > 0.50 (depends on data quality)
- **Warnings**: Intelligent audio alerts for hazards

## 💡 Tips

1. **Start with validation**: Always run `dataset_validator.py` before training
2. **Use GPU**: Training on CPU takes 10-20x longer
3. **Monitor training**: Check logs in `artifacts/runs/train/auranav/`
4. **Test incrementally**: Test each component before the full system
5. **Adjust thresholds**: Fine-tune hazard scores in config.yaml after testing

---

**Status**: Ready to use once dataset is fixed and model is trained.
