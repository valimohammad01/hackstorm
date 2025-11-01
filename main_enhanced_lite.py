"""
Enhanced AI Suspicious Surveillance System - LITE VERSION
Works without DeepFace/TensorFlow - Uses basic emotion heuristics
Features:
- Red Alert System for weapon detection
- Timestamp tracking for suspicious events
- Basic face expression analysis (no deep learning required)
- Optional Gemini API integration for image enhancement
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import time
import json
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
import io

# Optional Gemini import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except:
    GEMINI_AVAILABLE = False

# -----------------------------
# CONFIGURATION
# -----------------------------
PROJECT_DIR = "suspicious_data/data"
ALERT_LOG_DIR = "alert_logs"
os.makedirs(ALERT_LOG_DIR, exist_ok=True)

SUSPICIOUS_CLASSES = {
    "gun", "knife", "scissor", "scissors", "fire", "bat", "rod", 
    "Pointed_knife", "Dual-edge-sharp-knife", "dual-edge_knife", 
    "recurve", "Sharp_cutter"
}

SUSPICIOUS_ALIASES = {
    "scissor": "scissors",
    "scissors": "scissors",
    "Pointed_knife": "knife",
    "Dual-edge-sharp-knife": "knife", 
    "dual-edge_knife": "knife",
    "Sharp_cutter": "knife",
    "bat": "baseball bat",
    "rod": "metal rod",
    "recurve": "recurve knife"
}

PROXIMITY_PIXELS = 80

# -----------------------------
# Streamlit page
# -----------------------------
st.set_page_config(page_title="AI Suspicious Surveillance - Enhanced Lite", layout="wide", page_icon="🚨")

# Custom CSS for red alert
st.markdown("""
    <style>
    .red-alert {
        background-color: #ff0000;
        color: white;
        padding: 20px;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0%, 50%, 100% { opacity: 1; }
        25%, 75% { opacity: 0.5; }
    }
    .critical-alert {
        background-color: #dc2626;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 5px solid #991b1b;
    }
    .timestamp-badge {
        background-color: #1f2937;
        color: #fbbf24;
        padding: 5px 10px;
        border-radius: 5px;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div style="background-color:#f5f5f5;padding:10px;border-radius:8px;">
        <h1 style="color:#1f2937;">🚨 AI Suspicious Surveillance - Enhanced Lite</h1>
        <p style="color:#4b5563;">Real-time weapon detection with basic emotion analysis & optional AI enhancement</p>
    </div>
    """, unsafe_allow_html=True
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ Configuration")

# Gemini API Key
gemini_api_key = None
if GEMINI_AVAILABLE:
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", 
                                            help="Enter your Google Gemini API key for image enhancement")
    if gemini_api_key:
        try:
            genai.configure(api_key=gemini_api_key)
            st.sidebar.success("✅ Gemini API configured")
        except Exception as e:
            st.sidebar.error(f"⚠️ Gemini API error: {e}")
            gemini_api_key = None
    else:
        st.sidebar.warning("⚠️ Gemini API not configured")
else:
    st.sidebar.info("ℹ️ Gemini AI not installed")

st.sidebar.markdown("---")
st.sidebar.header("📹 Video Source")
source_type = st.sidebar.selectbox("Select source", ["Webcam", "Upload video", "Sample video"])
uploaded_file = None
if source_type == "Upload video":
    uploaded_file = st.sidebar.file_uploader("Upload video (mp4/avi)", type=["mp4", "avi"])

st.sidebar.markdown("---")
st.sidebar.header("🎯 Detection Settings")
inference_size = st.sidebar.selectbox("Inference size (px)", [320, 384, 480, 640], index=1)
process_every_n = st.sidebar.slider("Process every N frames", 1, 6, 2)
confidence_thr = st.sidebar.slider("Confidence threshold", 0.2, 0.9, 0.40, 0.05)

enable_blur_enhancement = False
if GEMINI_AVAILABLE and gemini_api_key:
    enable_blur_enhancement = st.sidebar.checkbox("Enable blur enhancement (Gemini)", value=False)

enable_emotion_tracking = st.sidebar.checkbox("Enable basic emotion tracking", value=True,
                                               help="Uses simple heuristics (no deep learning)")

st.sidebar.markdown("---")
st.sidebar.header("🚨 Alert Settings")
alert_cooldown = st.sidebar.slider("Alert cooldown (seconds)", 1, 10, 3)

st.sidebar.markdown("---")
start_button = st.sidebar.button("▶️ Start Detection", type="primary")
stop_button = st.sidebar.button("⏹️ Stop / Reset")

# -----------------------------
# Load YOLO
# -----------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

with st.spinner("Loading YOLO model..."):
    model = load_yolo_model()
st.sidebar.success("✅ Model loaded")

# -----------------------------
# Face cascade for basic emotion
# -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# -----------------------------
# Functions
# -----------------------------
def simple_emotion_detection(face_roi):
    """Simple emotion detection using brightness and contrast heuristics"""
    if face_roi is None or face_roi.size == 0:
        return "neutral", 0.6
    
    try:
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray_face))
        contrast = float(np.std(gray_face))
        
        # Simple heuristics
        if brightness < 90 and contrast > 35:
            return "angry", 0.65
        if brightness > 160:
            return "happy", 0.70
        if contrast < 18:
            return "sad", 0.60
        if contrast > 45:
            return "fear", 0.65
        
        return "neutral", 0.70
    except:
        return "neutral", 0.5

def log_alert(alert_type, message, frame_data=None):
    """Log alert with timestamp"""
    timestamp = datetime.now()
    alert_record = {
        "timestamp": timestamp.isoformat(),
        "type": alert_type,
        "message": message,
        "date": timestamp.strftime("%Y-%m-%d"),
        "time": timestamp.strftime("%H:%M:%S")
    }
    
    # Save to JSON log
    log_file = os.path.join(ALERT_LOG_DIR, f"alerts_{timestamp.strftime('%Y%m%d')}.json")
    
    alerts = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                alerts = json.load(f)
        except:
            alerts = []
    
    alerts.append(alert_record)
    
    with open(log_file, 'w') as f:
        json.dump(alerts, f, indent=2)
    
    # Save screenshot
    if frame_data is not None:
        screenshot_file = os.path.join(ALERT_LOG_DIR, 
                                      f"alert_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg")
        cv2.imwrite(screenshot_file, frame_data)
    
    return alert_record

def display_red_alert(alert_messages):
    """Display red alert banner"""
    if alert_messages:
        st.markdown(
            f"""
            <div class="red-alert">
                🚨 RED ALERT: {len(alert_messages)} THREAT(S) DETECTED! 🚨
            </div>
            """, 
            unsafe_allow_html=True
        )

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB-xA)*max(0, yB-yA)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    if boxAArea+boxBArea-interArea==0:
        return 0.0
    return interArea/(boxAArea+boxBArea-interArea)

def assign_ids(detections, tracker, iou_thresh=0.25, frame_idx=0):
    assigned=[]
    used_ids=set()
    next_id = st.session_state.next_id
    
    for bbox,label,conf,centroid in detections:
        best_id=None; best_iou=0
        for tid, info in tracker.items():
            if tid in used_ids: continue
            i=iou(bbox, info.get("bbox", bbox))
            if i>best_iou:
                best_iou=i; best_id=tid
        if best_iou>=iou_thresh and best_id is not None:
            assigned.append((best_id,bbox,label,conf,centroid))
            used_ids.add(best_id)
            tracker[best_id]["bbox"]=bbox
            tracker[best_id]["centroids"].append(centroid)
            tracker[best_id]["last_seen"]=frame_idx
        else:
            new_id=next_id
            next_id+=1
            tracker[new_id]={"bbox":bbox,"centroids":[centroid],"last_seen":frame_idx}
            assigned.append((new_id,bbox,label,conf,centroid))
            used_ids.add(new_id)
    
    st.session_state.next_id = next_id
    
    # prune old
    to_del=[tid for tid, info in tracker.items() if frame_idx-info.get("last_seen",0)>500]
    for tid in to_del: tracker.pop(tid,None)
    
    return assigned

# -----------------------------
# Session state
# -----------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "detections" not in st.session_state:
    st.session_state.detections = []
if "alerts" not in st.session_state:
    st.session_state.alerts = []
if "tracker" not in st.session_state:
    st.session_state.tracker = {}
if "next_id" not in st.session_state:
    st.session_state.next_id = 0
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "last_alert_time" not in st.session_state:
    st.session_state.last_alert_time = 0

# Start / Stop
if start_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False

# -----------------------------
# Layout
# -----------------------------
col_main, col_right = st.columns([2.5, 1.5])

with col_main:
    st.subheader("📹 Live Video Feed")
    frame_placeholder = st.empty()

with col_right:
    st.subheader("🚨 Alert Status")
    alert_placeholder = st.empty()
    
    st.markdown("---")
    st.subheader("📊 Metrics")
    metrics_container = st.container()
    
    with metrics_container:
        m1, m2, m3 = st.columns(3)
        fps_metric = m1.empty()
        threat_metric = m2.empty()
        det_metric = m3.empty()
    
    st.markdown("---")
    st.subheader("📋 Recent Alerts")
    recent_alerts_container = st.empty()

# -----------------------------
# Main Loop
# -----------------------------
cap=None
if source_type=="Webcam":
    cap=cv2.VideoCapture(0)
elif source_type=="Upload video" and uploaded_file is not None:
    tmp_path="uploaded_temp.mp4"
    with open(tmp_path,"wb") as f: f.write(uploaded_file.read())
    cap=cv2.VideoCapture(tmp_path)
else:
    sample_paths = ["demo_weapon.mp4", "demo_video.mp4"]
    for sample_path in sample_paths:
        if os.path.exists(sample_path):
            cap=cv2.VideoCapture(sample_path)
            break
    if cap is None or not cap.isOpened():
        cap=cv2.VideoCapture(0)

if cap is None or not cap.isOpened():
    frame_placeholder.info("⚠️ Select a valid video source and press Start.")
else:
    prev_time=time.perf_counter()
    last_detections=[]
    
    while st.session_state.running:
        st.session_state.frame_idx+=1
        ok, frame=cap.read()
        
        if not ok: 
            st.session_state.running=False
            break
        
        frame_h, frame_w = frame.shape[:2]
        
        # Frame skipping
        if (st.session_state.frame_idx % process_every_n)!=0 and last_detections:
            display_frame=frame.copy()
            for info in last_detections:
                tid,bbox,label,conf=info["id"],info["bbox"],info["label"],info["conf"]
                emotion = info.get("emotion", "")
                x1,y1,x2,y2=bbox
                is_suspicious = label in SUSPICIOUS_CLASSES or label in SUSPICIOUS_ALIASES
                color=(0,0,255) if is_suspicious else (50,205,50)
                cv2.rectangle(display_frame,(x1,y1),(x2,y2),color,2)
                text = f"{label} ID:{tid} {conf:.2f}"
                if emotion:
                    text += f" [{emotion}]"
                cv2.putText(display_frame,text,(x1,max(10,y1-8)),
                           cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)

            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            time.sleep(0.01)
            continue

        # Resize for inference
        long_side=max(frame_w,frame_h)
        scale=1.0
        if long_side>inference_size:
            scale=inference_size/long_side
            frame_infer=cv2.resize(frame,(int(frame_w*scale),int(frame_h*scale)))
        else: 
            frame_infer=frame.copy()

        # YOLO detection
        t0=time.perf_counter()
        results=model(frame_infer,conf=confidence_thr,imgsz=inference_size,verbose=False)[0]
        t1=time.perf_counter()

        dets=[]
        for det in results.boxes:
            cls_id=int(det.cls[0])
            conf=float(det.conf[0])
            if conf<confidence_thr: continue
            x1,y1,x2,y2=map(int,det.xyxy[0])
            if scale<1.0: 
                x1=int(x1/scale);y1=int(y1/scale);x2=int(x2/scale);y2=int(y2/scale)
            label=model.names.get(cls_id,str(cls_id))
            cx,cy=(x1+x2)//2,(y1+y2)//2
            dets.append(([x1,y1,x2,y2],label,conf,(cx,cy)))

        assigned=assign_ids(dets,st.session_state.tracker,0.2,st.session_state.frame_idx)

        display_frame=frame.copy()
        alert_messages=[]
        persons_in_frame=[]
        suspicious_items=[]
        last_detections=[]
        threat_score = 0
        
        # Process detections
        for tid,bbox,label,conf,centroid in assigned:
            x1,y1,x2,y2=bbox
            cx,cy=centroid
            
            is_suspicious = label in SUSPICIOUS_CLASSES or label in SUSPICIOUS_ALIASES
            actual_label = SUSPICIOUS_ALIASES.get(label, label) if label in SUSPICIOUS_ALIASES else label
            
            emotion = ""
            emotion_conf = 0.0
            
            # Basic emotion detection for persons
            if label=="person" and enable_emotion_tracking:
                person_roi = frame[y1:y2, x1:x2]
                if person_roi.size > 0:
                    gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                    
                    if len(faces) > 0:
                        fx, fy, fw, fh = faces[0]
                        face_roi = person_roi[fy:fy+fh, fx:fx+fw]
                        emotion, emotion_conf = simple_emotion_detection(face_roi)
            
            # Color coding
            color=(50,205,50)
            if is_suspicious:
                color=(0,0,255)
                suspicious_items.append({
                    "id":tid, "label":actual_label, "bbox":bbox, 
                    "centroid":centroid, "conf":conf
                })
                threat_score += int(conf * 50)
            elif label=="person":
                persons_in_frame.append({
                    "id":tid, "centroid":centroid, "bbox":bbox, 
                    "emotion":emotion, "emotion_conf":emotion_conf
                })
                if emotion in ["angry", "fear"]:
                    color=(0,165,255)
                    threat_score += 10
            
            # Draw bounding box
            cv2.rectangle(display_frame,(x1,y1),(x2,y2),color,2)
            
            # Draw label
            text = f"{actual_label if is_suspicious else label} ID:{tid} {conf:.2f}"
            if emotion:
                text += f" [{emotion}]"
            cv2.putText(display_frame, text, (x1,max(10,y1-8)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            
            # Log detection
            detection_record = {
                "frame": st.session_state.frame_idx,
                "timestamp": datetime.now().isoformat(),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "id": tid,
                "label": actual_label if is_suspicious else label,
                "conf": conf,
                "bbox": bbox,
                "is_suspicious": is_suspicious,
                "emotion": emotion,
                "emotion_confidence": emotion_conf
            }
            st.session_state.detections.append(detection_record)
            last_detections.append({
                "id":tid, "label":actual_label if is_suspicious else label, 
                "conf":conf, "bbox":bbox, "emotion":emotion
            })

        # Generate alerts
        current_time = time.time()
        
        # Weapon detection alerts
        for item in suspicious_items:
            if current_time - st.session_state.last_alert_time > alert_cooldown:
                alert_msg = f"🚨 WEAPON DETECTED: {item['label'].upper()} (Confidence: {item['conf']:.2%})"
                alert_messages.append(alert_msg)
                
                alert_record = log_alert("weapon_detected", alert_msg, display_frame)
                st.session_state.alerts.append(alert_record)
                st.session_state.last_alert_time = current_time
        
        # Proximity alerts
        for item in suspicious_items:
            for p in persons_in_frame:
                dx=item["centroid"][0]-p["centroid"][0]
                dy=item["centroid"][1]-p["centroid"][1]
                dist = (dx*dx+dy*dy)**0.5
                
                if dist < PROXIMITY_PIXELS:
                    severity = "CRITICAL" if p["emotion"] in ["angry", "fear"] else "HIGH"
                    emotion_text = f" [Emotion: {p['emotion'].upper()}]" if p["emotion"] else ""
                    alert_msg = f"⚠️ {severity}: {item['label'].upper()} near Person ID:{p['id']}{emotion_text}"
                    alert_messages.append(alert_msg)
                    threat_score += 30 if severity == "CRITICAL" else 20
                    
                    if current_time - st.session_state.last_alert_time > alert_cooldown:
                        alert_record = log_alert("weapon_proximity", alert_msg, display_frame)
                        st.session_state.alerts.append(alert_record)
                        st.session_state.last_alert_time = current_time

        # Draw threat score
        threat_score = min(100, threat_score)
        threat_color = (0,255,0) if threat_score < 30 else (0,165,255) if threat_score < 70 else (0,0,255)
        cv2.putText(display_frame, f"Threat Level: {threat_score}%", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, threat_color, 2)

        # Display frame
        frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Update alert display
        with alert_placeholder.container():
            if alert_messages:
                display_red_alert(alert_messages)
                for alert in alert_messages:
                    st.markdown(f'<div class="critical-alert">{alert}</div>', 
                              unsafe_allow_html=True)
            else:
                st.success("✅ No threats detected")

        # Update metrics
        fps=1.0/max(1e-6,time.perf_counter()-prev_time)
        prev_time=time.perf_counter()
        
        fps_metric.metric("FPS", f"{fps:.1f}")
        threat_metric.metric("Threat Level", f"{threat_score}%")
        det_metric.metric("Detections", str(len(st.session_state.detections)))

        # Display recent alerts
        if st.session_state.alerts:
            recent = st.session_state.alerts[-5:][::-1]
            recent_alerts_html = ""
            for alert in recent:
                recent_alerts_html += f"""
                <div style="background:#f3f4f6;padding:8px;margin:5px 0;border-radius:5px;border-left:3px solid #dc2626;">
                    <div style="font-size:12px;color:#6b7280;">
                        <span class="timestamp-badge">{alert['time']}</span>
                    </div>
                    <div style="font-size:14px;margin-top:5px;color:#1f2937;">
                        {alert['message']}
                    </div>
                </div>
                """
            recent_alerts_container.markdown(recent_alerts_html, unsafe_allow_html=True)

        time.sleep(0.01)

    cap.release()

# -----------------------------
# Download Reports
# -----------------------------
st.markdown("---")
st.subheader("📥 Export Reports")

col1, col2 = st.columns(2)

with col1:
    if st.session_state.detections:
        df_detections = pd.DataFrame(st.session_state.detections)
        csv_detections = df_detections.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📊 Download Detections CSV",
            data=csv_detections,
            file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No detections recorded")

with col2:
    if st.session_state.alerts:
        df_alerts = pd.DataFrame(st.session_state.alerts)
        csv_alerts = df_alerts.to_csv(index=False).encode("utf-8")
        st.download_button(
            "🚨 Download Alerts CSV",
            data=csv_alerts,
            file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("No alerts recorded")

st.markdown("---")
st.caption("🛡️ Enhanced Surveillance System v2.0 LITE - With Red Alert, Timestamps, Basic Emotion Analysis")
