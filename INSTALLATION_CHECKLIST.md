# ✅ Installation Checklist - Enhanced Surveillance System

## Pre-Installation Check

### System Requirements
- [ ] Windows 10/11, Linux, or macOS
- [ ] Python 3.8 or higher installed
- [ ] 4GB+ RAM available
- [ ] 2GB+ free disk space
- [ ] Internet connection (for installation and optional Gemini API)

### Verify Python Installation
```bash
python --version
# Should show: Python 3.8.x or higher
```

If Python not installed:
- **Windows**: Download from https://www.python.org/downloads/
- **Linux**: `sudo apt install python3 python3-pip`
- **macOS**: `brew install python3`

---

## Step-by-Step Installation

### ✅ Step 1: Navigate to Project Directory
```bash
cd d:\final_hack
```

### ✅ Step 2: Install Dependencies
```bash
pip install -r requirements_enhanced_v2.txt
```

**Expected output:**
```
Collecting streamlit==1.29.0
Collecting ultralytics==8.3.9
Collecting opencv-python-headless==4.8.0.76
...
Successfully installed [packages]
```

**If errors occur:**
```bash
# Try with --upgrade
pip install -r requirements_enhanced_v2.txt --upgrade

# Or force reinstall
pip install -r requirements_enhanced_v2.txt --force-reinstall
```

### ✅ Step 3: Verify Installations
```bash
python -c "import streamlit; print('Streamlit:', streamlit.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import ultralytics; print('Ultralytics: OK')"
python -c "import deepface; print('DeepFace: OK')"
python -c "import google.generativeai; print('Gemini: OK')"
```

**All should print without errors**

### ✅ Step 4: Check Required Files
```bash
# Verify these files exist:
- [ ] main_enhanced.py
- [ ] yolov8n.pt (YOLO model)
- [ ] requirements_enhanced_v2.txt
- [ ] run_enhanced.bat (Windows)
```

**If yolov8n.pt missing:**
- It will auto-download on first run
- Or manually download from Ultralytics

### ✅ Step 5: (Optional) Get Gemini API Key
- [ ] Go to: https://makersuite.google.com/app/apikey
- [ ] Sign in with Google account
- [ ] Click "Create API Key"
- [ ] Copy the key (starts with: AIzaSy...)
- [ ] Save for later (paste in UI)

### ✅ Step 6: Test Basic Installation
```bash
streamlit run main_enhanced.py
```

**Expected:**
- Browser opens automatically
- Streamlit UI loads
- Sidebar shows configuration options
- "Model loaded" message appears

### ✅ Step 7: Test Video Sources

#### Test with Webcam
- [ ] Select "Webcam" in sidebar
- [ ] Click "▶️ Start Detection"
- [ ] Verify video feed appears
- [ ] Check FPS metric shows 15+ FPS

#### Test with Sample Video (if available)
- [ ] Select "Sample video" in sidebar
- [ ] Click "▶️ Start Detection"
- [ ] Verify video plays
- [ ] Check detections appear

### ✅ Step 8: Test Alert System
- [ ] Place weapon-like object in view (scissors, etc.)
- [ ] Verify bounding box appears (red)
- [ ] Check if red alert banner shows
- [ ] Confirm alert logged in right panel

### ✅ Step 9: Test Emotion Detection
- [ ] Ensure person visible in frame
- [ ] Verify face detection box appears
- [ ] Check emotion label shows (angry, happy, etc.)
- [ ] Test different facial expressions

### ✅ Step 10: Test Data Export
- [ ] Let system run for 30+ seconds
- [ ] Click "📊 Download Detections CSV"
- [ ] Verify CSV file downloads
- [ ] Open CSV and check data format

---

## Post-Installation Verification

### Functional Tests

#### Test 1: Basic Detection ✅
```
Expected: Person and objects detected
Actual: ___________
Status: [ ] Pass [ ] Fail
```

#### Test 2: Weapon Alert ✅
```
Expected: Red alert when weapon detected
Actual: ___________
Status: [ ] Pass [ ] Fail
```

#### Test 3: Emotion Analysis ✅
```
Expected: Emotion shown on person detection
Actual: ___________
Status: [ ] Pass [ ] Fail
```

#### Test 4: Timestamp Logging ✅
```
Expected: Logs saved in alert_logs/ folder
Actual: ___________
Status: [ ] Pass [ ] Fail
```

#### Test 5: CSV Export ✅
```
Expected: CSV downloads with data
Actual: ___________
Status: [ ] Pass [ ] Fail
```

---

## Troubleshooting Common Issues

### Issue: "Module not found" errors

**Symptom:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
pip install -r requirements_enhanced_v2.txt --force-reinstall
```

---

### Issue: TensorFlow/DeepFace errors

**Symptom:**
```
Could not load dynamic library 'cudart64_110.dll'
```

**Solution:**
This is a warning, not an error. System will use CPU mode.
- To fix: Install CUDA toolkit (optional, for GPU)
- Or ignore: System works fine on CPU

---

### Issue: OpenCV errors

**Symptom:**
```
ImportError: libGL.so.1: cannot open shared object file
```

**Solution (Linux):**
```bash
sudo apt-get install libgl1-mesa-glx
```

---

### Issue: Webcam not opening

**Symptom:**
No video feed when webcam selected

**Solution:**
1. Check webcam permissions
2. Close other apps using webcam
3. Try different camera index in code
4. Use video file instead

---

### Issue: Slow performance

**Symptom:**
FPS < 10, laggy video

**Solution:**
1. Reduce inference size (320px)
2. Increase frame skip (3-4)
3. Disable blur enhancement
4. Close other applications
5. Use GPU if available

---

### Issue: Gemini API not working

**Symptom:**
```
google.api_core.exceptions.PermissionDenied
```

**Solution:**
1. Verify API key is correct
2. Check API is enabled
3. Verify internet connection
4. Try regenerating API key
5. Check quota limits

---

## Performance Benchmarks

### Expected Performance (CPU)

| Hardware | FPS | Quality |
|----------|-----|---------|
| Intel i5-8th Gen | 15-20 | Good |
| Intel i7-10th Gen | 25-30 | Excellent |
| AMD Ryzen 5 | 18-22 | Good |
| AMD Ryzen 7 | 28-32 | Excellent |

### Expected Performance (GPU)

| GPU | FPS | Quality |
|-----|-----|---------|
| NVIDIA GTX 1060 | 35-40 | Excellent |
| NVIDIA RTX 2060 | 50-60 | Excellent |
| NVIDIA RTX 3060 | 60-70 | Excellent |

**Note:** With all features enabled (emotion + enhancement), expect 30-40% FPS reduction

---

## Final Checklist

### Core Features Working
- [ ] YOLO model loads successfully
- [ ] Video feed displays
- [ ] Object detection working
- [ ] Person detection working
- [ ] Weapon detection working
- [ ] Bounding boxes show correctly

### Enhanced Features Working
- [ ] Red alert banner appears
- [ ] Timestamps accurate (millisecond precision)
- [ ] Face detection working
- [ ] Emotion analysis showing
- [ ] Threat level displays
- [ ] Recent alerts panel updates

### Data & Export Working
- [ ] alert_logs/ directory created
- [ ] JSON logs saving
- [ ] Screenshots saving on alerts
- [ ] Detections CSV downloads
- [ ] Alerts CSV downloads

### UI Elements Working
- [ ] Sidebar controls functional
- [ ] Metrics updating (FPS, Threat, Detections)
- [ ] Color coding correct (Red/Green/Orange)
- [ ] Alert messages displaying
- [ ] Settings persist during session

### Optional Features
- [ ] Gemini API configured (if using)
- [ ] Blur enhancement working (if enabled)
- [ ] Sample videos loading (if available)

---

## Success Criteria

### Minimum Viable System ✅
- Basic weapon detection
- Person tracking
- Alert generation
- CSV export

### Full Featured System ✅
- All minimum features
- Face expression analysis
- Red alert system
- Timestamp logging
- Screenshot capture
- Threat scoring

### Production Ready System ✅
- All full features
- Gemini API integration
- Stable performance (15+ FPS)
- No critical errors
- Complete documentation

---

## Installation Complete! 🎉

If all checkboxes are marked, your system is ready for:
- ✅ Real-time surveillance
- ✅ Weapon detection
- ✅ Threat assessment
- ✅ Incident logging
- ✅ Data analysis

### Next Steps:
1. Read QUICK_START.md for usage tips
2. Configure optimal settings for your hardware
3. Test with real scenarios
4. Set up regular log backups
5. Monitor system performance

---

## Support Resources

### Documentation
- `ENHANCED_SETUP_GUIDE.md` - Complete guide
- `QUICK_START.md` - Quick reference
- `GEMINI_API_SETUP.md` - API setup
- `FEATURE_COMPARISON.md` - Feature details
- `README_ENHANCED.md` - Overview

### Commands Reference
```bash
# Run system
streamlit run main_enhanced.py

# Check installations
pip list | grep -E "streamlit|ultralytics|opencv|deepface"

# Update dependencies
pip install -r requirements_enhanced_v2.txt --upgrade

# Clear cache (if issues)
streamlit cache clear
```

---

## Installation Record

**Date:** _______________

**Python Version:** _______________

**OS:** _______________

**Hardware:** _______________

**Issues Encountered:** 
_________________________________
_________________________________
_________________________________

**Resolution:** 
_________________________________
_________________________________
_________________________________

**Final Status:** [ ] Success [ ] Partial [ ] Failed

**Notes:**
_________________________________
_________________________________
_________________________________

---

**Ready to start securing your environment! 🛡️**

```bash
streamlit run main_enhanced.py
```
