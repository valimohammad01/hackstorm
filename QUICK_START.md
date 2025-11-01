# ⚡ Quick Start Guide - Enhanced Surveillance System

## 🚀 Get Started in 3 Minutes

### 1️⃣ Install (1 minute)
```bash
pip install -r requirements_enhanced_v2.txt
```

### 2️⃣ Run (30 seconds)
```bash
streamlit run main_enhanced.py
```
or double-click: `run_enhanced.bat`

### 3️⃣ Configure (1 minute)
1. *(Optional)* Paste Gemini API key in sidebar
2. Select video source (Webcam/Upload/Sample)
3. Click **"▶️ Start Detection"**

**Done! 🎉 System is now running.**

---

## 🎯 Essential Features at a Glance

### 🚨 Red Alert System
```
WEAPON DETECTED → 🚨 RED ALERT BANNER
```
- Impossible to miss
- Animated visual warning
- Auto-screenshots saved

### 📍 Timestamp Tracking
```
14:23:45.123 - WEAPON DETECTED: GUN
14:23:46.789 - Person approaches weapon
14:23:48.456 - CRITICAL THREAT
```
- Millisecond precision
- All events logged
- Export to CSV

### 😠 Face Expression
```
Person ID:5 [angry] + GUN = 🚨 CRITICAL
Person ID:3 [neutral] + knife = ⚠️ HIGH
```
- 7 emotions tracked
- Smart threat scoring
- Real-time analysis

### 🎨 Gemini AI Enhancement
```
Blurry Video → AI Enhancement → Better Detection
```
- Automatically sharpens frames
- Reduces noise
- Improves accuracy

---

## ⚙️ Recommended Settings

### 🎮 For Beginners (Balanced)
```
Inference Size: 384px
Process Every N Frames: 2
Confidence Threshold: 0.40
Blur Enhancement: OFF (start without)
Emotion Tracking: ON
```

### 🏃 For Speed (Fast Performance)
```
Inference Size: 320px
Process Every N Frames: 3
Confidence Threshold: 0.45
Blur Enhancement: OFF
Emotion Tracking: OFF
```

### 🎯 For Accuracy (Best Detection)
```
Inference Size: 640px
Process Every N Frames: 1
Confidence Threshold: 0.35
Blur Enhancement: ON (with API key)
Emotion Tracking: ON
```

---

## 📱 Dashboard Overview

```
┌─────────────────────────────────────────┐
│ 🚨 RED ALERT: THREATS DETECTED!         │ ← Red Alert Banner
├─────────────────────────────────────────┤
│ 📹 Live Video Feed                      │
│ ┌─────────────────────────────────────┐ │
│ │ [Video with bounding boxes]         │ │
│ │ Red = Weapons                        │ │
│ │ Green = Persons                      │ │
│ │ Orange = Suspicious Emotions         │ │
│ └─────────────────────────────────────┘ │
├─────────────────────────────────────────┤
│ 📊 Metrics                              │
│ FPS: 25.3 | Threat: 75% | Det: 145     │
├─────────────────────────────────────────┤
│ 📋 Recent Alerts (Last 5)               │
│ 14:30:45 - CRITICAL: Weapon + Angry    │
│ 14:30:12 - HIGH: Weapon near person    │
│ 14:29:58 - MEDIUM: Suspicious emotion  │
└─────────────────────────────────────────┘
```

---

## 🎨 Color Coding

### Bounding Boxes
- 🔴 **Red** = Weapon detected
- 🟢 **Green** = Person detected
- 🟠 **Orange** = Person with suspicious emotion

### Threat Levels
- 🟢 **0-29%** = Low (Safe)
- 🟠 **30-69%** = Medium (Monitor)
- 🔴 **70-100%** = High (Action Required)

### Emotions
- 😠 **Angry** = High Risk
- 😨 **Fear** = High Risk
- 😢 **Sad** = Medium Risk
- 😐 **Neutral** = Low Risk
- 😊 **Happy** = Low Risk

---

## 📥 Export Data

### During Session
1. Detection data auto-saves to `alert_logs/`
2. Screenshots auto-save on alerts
3. JSON logs created daily

### After Session
1. Click **"📊 Download Detections CSV"**
2. Click **"🚨 Download Alerts CSV"**
3. Files saved with timestamp

---

## 🔑 Gemini API (Optional)

### Get Free API Key (2 minutes)
1. Visit: https://makersuite.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy and paste in sidebar

### When to Use
- ✅ Blurry video footage
- ✅ Low-quality cameras
- ✅ Night vision recordings
- ✅ Long-distance views
- ❌ Not needed for HD cameras

---

## ⚡ Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Stop system | Ctrl+C (in terminal) |
| Restart page | F5 or Ctrl+R |
| Toggle sidebar | Click > on left |
| Full screen | F11 |

---

## 🆘 Quick Troubleshooting

### Problem: Slow performance
**Solution:** Increase "Process every N frames" to 3-4

### Problem: Too many false alerts
**Solution:** Increase confidence threshold to 0.5-0.6

### Problem: Missing detections
**Solution:** Decrease confidence threshold to 0.3-0.35

### Problem: No faces detected
**Solution:** Ensure good lighting and persons face camera

### Problem: Gemini API error
**Solution:** Check API key or disable blur enhancement

---

## 📊 What Gets Logged?

### Detections Log (detections.csv)
- Every object detected
- Timestamp (milliseconds)
- Confidence scores
- Bounding box coordinates
- Emotions (if person)

### Alerts Log (alerts.csv)
- Only critical events
- Weapon detections
- Proximity warnings
- Suspicious behavior

### Screenshots (alert_logs/)
- Auto-saved on alerts
- Timestamped filenames
- JPG format

---

## 🎓 Best Practices

### 1. Testing
- Start with sample videos
- Test settings before live use
- Verify alerts are working

### 2. Configuration
- Adjust settings based on camera quality
- Balance speed vs accuracy
- Monitor FPS metric

### 3. Monitoring
- Keep threat level below 30% normally
- Investigate any red alerts immediately
- Review logs regularly

### 4. Maintenance
- Clear old alert logs weekly
- Update model periodically
- Check API usage limits

---

## 📈 Performance Tips

### Increase FPS
1. Lower inference size (320px)
2. Increase frame skip (4-5)
3. Disable blur enhancement
4. Use GPU if available

### Improve Accuracy
1. Higher inference size (640px)
2. Lower confidence threshold (0.3)
3. Enable blur enhancement
4. Process all frames (1)

---

## 🔄 System Status Indicators

### Green ✅
- System running normally
- No threats detected
- All features working

### Orange ⚠️
- Medium threat level
- Some suspicious activity
- Monitor situation

### Red 🚨
- High threat detected
- Weapon near person
- Immediate attention needed

---

## 🎯 Common Scenarios

### Scenario 1: Office Security
```
Settings:
- Inference: 384px
- Frame skip: 2
- Confidence: 0.40
- Emotion: ON
```

### Scenario 2: Warehouse Monitoring
```
Settings:
- Inference: 320px
- Frame skip: 3
- Confidence: 0.45
- Emotion: OFF (for speed)
```

### Scenario 3: High-Security Area
```
Settings:
- Inference: 640px
- Frame skip: 1
- Confidence: 0.30
- Emotion: ON
- Enhancement: ON
```

---

## 📞 Need More Help?

### Documentation Files
- `ENHANCED_SETUP_GUIDE.md` - Complete setup guide
- `GEMINI_API_SETUP.md` - Gemini API details
- `FEATURE_COMPARISON.md` - Feature comparison
- `QUICK_START.md` - This file

### Common Commands
```bash
# Run enhanced system
streamlit run main_enhanced.py

# Run on different port
streamlit run main_enhanced.py --server.port 8502

# Enable debug mode
streamlit run main_enhanced.py --logger.level=debug
```

---

## ✅ Quick Checklist

Before going live:
- [ ] Dependencies installed
- [ ] Model file (yolov8n.pt) present
- [ ] Test video/webcam working
- [ ] Settings configured
- [ ] Alert system tested
- [ ] Export functionality verified
- [ ] Gemini API (if using) tested

---

## 🎉 You're Ready!

```bash
# Start the system
streamlit run main_enhanced.py

# Or use batch file
run_enhanced.bat
```

**System Features:**
✅ Red alert system
✅ Timestamp tracking  
✅ Face expression analysis
✅ AI image enhancement
✅ Automatic logging
✅ CSV export

**🛡️ Stay Safe. Stay Secure.**

---

**Quick Links:**
- Get API Key: https://makersuite.google.com/app/apikey
- Full Guide: See `ENHANCED_SETUP_GUIDE.md`
- Troubleshooting: See main guide Section 🔧
