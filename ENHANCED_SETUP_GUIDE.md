# 🚨 Enhanced AI Surveillance System - Setup Guide

## 🎯 New Features

### 1. **Red Alert System**
- Visual red alert banner when weapons are detected
- Animated blinking effect for critical threats
- Color-coded alerts based on severity levels
- Alert cooldown system to prevent spam

### 2. **Timestamp Tracking**
- Every suspicious detection is logged with precise timestamp
- Date and time tracking for all events
- JSON log files stored in `alert_logs/` directory
- Screenshots automatically saved with alerts
- Downloadable CSV reports with timestamps

### 3. **Face Expression Analysis**
- Real-time emotion detection using DeepFace
- Tracks 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral
- Emotions displayed alongside person detections
- Enhanced threat scoring when weapon + angry/fear emotion detected
- Critical alerts when person with weapon shows aggressive emotion

### 4. **Gemini API Integration**
- AI-powered image enhancement for blurry/low-quality frames
- Automatic blur reduction and sharpening
- Denoising for better weapon detection accuracy
- Optional feature (can be enabled/disabled)
- Processes frames periodically to reduce API costs

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Webcam** or video files for testing
3. **Google Gemini API Key** (optional, for blur enhancement)
   - Get free API key from: https://makersuite.google.com/app/apikey

## 🚀 Installation Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements_enhanced_v2.txt
```

This will install:
- Streamlit (UI framework)
- Ultralytics YOLO (weapon detection)
- OpenCV (video processing)
- DeepFace + TensorFlow (emotion analysis)
- Google Generative AI (image enhancement)
- Pillow, NumPy, Pandas (utilities)

### Step 2: Download YOLO Model

The YOLOv8n model should be in your project directory as `yolov8n.pt`.
If not present, it will be automatically downloaded on first run.

### Step 3: Get Gemini API Key (Optional)

1. Go to https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the API key

### Step 4: Prepare Video Samples

Place your demo videos in the project root:
- `demo_weapon.mp4`
- `demo_video.mp4`
- Or use webcam for live testing

## ▶️ Running the Enhanced System

### Option 1: Using Batch File (Windows)

```bash
run_enhanced.bat
```

### Option 2: Manual Command

```bash
streamlit run main_enhanced.py
```

### Option 3: Python Command

```python
python -m streamlit run main_enhanced.py
```

## 🎛️ Configuration Guide

### 1. **Video Source Selection**
- **Webcam**: Use built-in or external camera
- **Upload Video**: Upload MP4/AVI files
- **Sample Video**: Use demo videos from project

### 2. **Detection Settings**
- **Inference Size**: Lower = faster, Higher = more accurate
  - 320px: Fast, lower quality
  - 640px: Slower, higher quality
- **Process Every N Frames**: Skip frames to increase FPS
  - 1 = process all frames (slower)
  - 3-4 = good balance
- **Confidence Threshold**: 0.2-0.9
  - Lower = more detections (may have false positives)
  - Higher = fewer, more accurate detections

### 3. **Advanced Features**
- **Enable Blur Enhancement**: Toggle Gemini API usage
  - ✅ Better detection on blurry videos
  - ⚠️ Requires API key and increases processing time
- **Enable Emotion Tracking**: Toggle face expression analysis
  - ✅ Provides emotion context
  - ⚠️ Increases processing time

### 4. **Alert Settings**
- **Alert Cooldown**: Time between duplicate alerts
  - Lower = more frequent alerts
  - Higher = less notification spam

## 📊 Understanding the Dashboard

### Main Video Feed
- **Red boxes**: Weapons detected
- **Green boxes**: Persons detected
- **Orange boxes**: Persons with suspicious emotions
- **Labels**: Show object type, ID, confidence, and emotion

### Alert Status Panel
- **Red Alert Banner**: Active when weapons detected
- **Critical Alerts**: Weapon + Person proximity + Aggressive emotion
- **High Alerts**: Weapon + Person proximity
- **Medium Alerts**: Suspicious emotions detected

### Metrics
- **FPS**: Frames per second (processing speed)
- **Threat Level**: 0-100% threat score
  - 0-29%: Low threat (green)
  - 30-69%: Medium threat (orange)
  - 70-100%: High threat (red)
- **Detections**: Total objects detected

### Recent Alerts
- Shows last 5 alerts with timestamps
- Timestamp format: HH:MM:SS
- Color-coded by severity

## 📥 Export & Logging

### Automatic Logging
All alerts are automatically saved to:
- **JSON logs**: `alert_logs/alerts_YYYYMMDD.json`
- **Screenshots**: `alert_logs/alert_YYYYMMDD_HHMMSS.jpg`

### Manual Export
1. **Detections CSV**: All object detections with timestamps
2. **Alerts CSV**: All alert events with details

CSV files include:
- Timestamp (date + time with milliseconds)
- Object type and confidence
- Emotion data (if person)
- Bounding box coordinates
- Frame number

## 🔧 Troubleshooting

### Issue: "Module not found" errors
**Solution**: Reinstall dependencies
```bash
pip install -r requirements_enhanced_v2.txt --force-reinstall
```

### Issue: Slow performance
**Solutions**:
1. Decrease inference size (320px)
2. Increase "Process every N frames" to 3-4
3. Disable blur enhancement
4. Disable emotion tracking temporarily

### Issue: Gemini API errors
**Solutions**:
1. Verify API key is correct
2. Check internet connection
3. Disable blur enhancement if not needed
4. Check API quota limits

### Issue: No faces detected
**Solutions**:
1. Ensure persons are visible in frame
2. Check lighting conditions
3. Increase confidence threshold
4. Verify OpenCV cascade file is loaded

### Issue: False weapon detections
**Solutions**:
1. Increase confidence threshold (0.5-0.7)
2. Train model on your specific dataset
3. Review SUSPICIOUS_CLASSES in config

## 🎓 Usage Tips

1. **Start with default settings** and adjust based on performance
2. **Test with sample videos** before using webcam
3. **Monitor FPS metric** - aim for 15+ FPS for smooth operation
4. **Review alert logs** regularly for pattern analysis
5. **Export CSV reports** for detailed incident analysis
6. **Use blur enhancement** only on low-quality videos
7. **Enable emotion tracking** for high-security scenarios

## 📈 Performance Optimization

### For Real-Time Detection (Webcam)
```python
Inference Size: 320px
Process Every N Frames: 3
Blur Enhancement: OFF
Emotion Tracking: ON
```

### For Accurate Analysis (Video Files)
```python
Inference Size: 640px
Process Every N Frames: 1
Blur Enhancement: ON
Emotion Tracking: ON
```

### For Low-End Hardware
```python
Inference Size: 320px
Process Every N Frames: 4
Blur Enhancement: OFF
Emotion Tracking: OFF
```

## 🔒 Security Considerations

1. **API Keys**: Never commit API keys to version control
2. **Alert Logs**: Secure `alert_logs/` directory with appropriate permissions
3. **Screenshots**: May contain sensitive data - handle appropriately
4. **Network**: Consider running on isolated network for security operations
5. **Data Privacy**: Comply with local surveillance and privacy laws

## 📞 Support

For issues or questions:
1. Check this guide thoroughly
2. Review error messages in console
3. Check Streamlit logs
4. Verify all dependencies are installed correctly

## 🆕 Version History

### Version 2.0 (Enhanced)
- ✅ Red alert system with visual indicators
- ✅ Timestamp tracking for all events
- ✅ Face expression analysis integration
- ✅ Gemini API for image enhancement
- ✅ Automated logging and screenshots
- ✅ Enhanced CSV export with timestamps
- ✅ Threat level scoring system
- ✅ Improved UI with color coding

### Version 1.0 (Original)
- Basic weapon detection
- Person tracking
- Simple alert system

## 📄 License

This is a prototype surveillance system. Ensure compliance with local laws and regulations regarding surveillance and facial recognition.

---

**Ready to start? Run the enhanced system:**
```bash
streamlit run main_enhanced.py
```

🛡️ **Stay Safe. Stay Secure.**
