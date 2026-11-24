# AURANAV - Python Version Fix Guide

## Problem Detected
Your Python version is too new for the required packages.

## Check Your Python Version

```powershell
python --version
```

If you see **Python 3.13.x**, that's the issue!

---

## Solution Options (Choose One)

### ✅ OPTION 1: Use Python 3.11 (RECOMMENDED)

#### Step 1: Download Python 3.11
Go to: https://www.python.org/downloads/release/python-31110/
Download: **Windows installer (64-bit)**

#### Step 2: Install Python 3.11
- Run installer
- ✅ Check "Add Python 3.11 to PATH"
- Click "Install Now"

#### Step 3: Use Python 3.11 for AURANAV

```powershell
# Delete old virtual environment
Remove-Item -Recurse -Force venv

# Create new venv with Python 3.11
py -3.11 -m venv venv

# Or if that doesn't work, use full path:
# C:\Users\rsath\AppData\Local\Programs\Python\Python311\python.exe -m venv venv

# Activate
.\venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

---

### ✅ OPTION 2: Install PyTorch for Python 3.13 First

If you want to keep Python 3.13, install compatible versions:

```powershell
# Make sure venv is activated
.\venv\Scripts\activate

# Install PyTorch with CUDA 12.1 support (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or CPU-only version (no GPU)
pip install torch torchvision torchaudio

# Then install ultralytics (will use newer version compatible with 3.13)
pip install ultralytics>=8.3.0

# Install rest of requirements
pip install opencv-python numpy pillow roboflow pyyaml
pip install filterpy scikit-learn scipy
pip install pyttsx3 gTTS pygame
pip install pandas matplotlib seaborn tqdm
pip install pytest pytest-cov python-dotenv colorama
```

---

## Quick Fix Commands (Copy-Paste)

### For Python 3.11 (After Installing):

```powershell
# Remove old venv
Remove-Item -Recurse -Force venv

# Create new with Python 3.11
py -3.11 -m venv venv

# Activate
.\venv\Scripts\activate

# Install
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### For Python 3.13 (Manual Installation):

```powershell
# Activate venv
.\venv\Scripts\activate

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Ultralytics
pip install ultralytics>=8.3.0

# Install other packages
pip install opencv-python numpy pillow roboflow pyyaml filterpy scikit-learn scipy pyttsx3 gTTS pygame pandas matplotlib seaborn tqdm pytest pytest-cov python-dotenv colorama
```

---

## Verify Installation

After installation completes:

```powershell
# Check Python version
python --version

# Check key packages
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"

# If all work, continue to setup validation
python setup.py
```

---

## Which Option Should You Choose?

**Use OPTION 1 (Python 3.11)** if:
- ✅ You want the most stable setup
- ✅ You don't mind installing another Python version
- ✅ You want to match tested configuration

**Use OPTION 2 (Python 3.13)** if:
- ✅ You don't want to install another Python version
- ✅ You're okay with newer package versions
- ✅ You understand some features might be different

---

## After Installing Successfully

Continue with:

```powershell
# Set API key
$env:ROBOFLOW_API_KEY="your_api_key_here"

# Validate setup
python setup.py
```

Then follow EXECUTION_GUIDE.md from Step 4 onwards!
