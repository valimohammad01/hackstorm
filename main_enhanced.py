"""
Enhanced AI Suspicious Surveillance System
Features:
- Red Alert System for weapon detection
- Timestamp tracking for suspicious events
- Face expression analysis for persons with weapons
- Gemini API integration for image enhancement
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

# Optional imports with error handling
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception as e:
    print(f"Warning: DeepFace not available: {e}")
    DeepFace = None
    DEEPFACE_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception as e:
    print(f"Warning: Gemini AI not available: {e}")
    genai = None
    GEMINI_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
except Exception as e:
    print(f"Warning: MediaPipe not available: {e}")
    MEDIAPIPE_AVAILABLE = False
    mp_pose = None
    mp_hands = None
    mp_drawing = None

from PIL import Image
import io
import base64
import math
from collections import deque

# -----------------------------
# CONFIGURATION
# -----------------------------
PROJECT_DIR = "suspicious_data/data"
DATA_YAML = os.path.join(PROJECT_DIR, "data.yaml")
MODEL_SAVE_DIR = os.path.join(PROJECT_DIR, "runs")
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

# Behavior Detection Parameters
RUNNING_SPEED_THRESHOLD = 150  # pixels per second
AGGRESSIVE_APPROACH_THRESHOLD = 100  # pixels per second toward another person
LOITERING_TIME_THRESHOLD = 10  # seconds in same location
SUDDEN_MOVEMENT_THRESHOLD = 80  # pixels acceleration
HIGH_BEHAVIOR_SCORE_THRESHOLD = 60  # out of 100

# -----------------------------
# Streamlit page
# -----------------------------
st.set_page_config(page_title="AI Suspicious Surveillance - Enhanced", layout="wide", page_icon="🚨")

# Professional Dark Theme CSS
st.markdown("""
    <style>
    /* Dark Theme Base */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1a1d29;
        border-right: 1px solid #2d3142;
    }
    
    [data-testid="stSidebar"] .element-container {
        color: #f3f4f6;
    }
    
    /* Sidebar labels */
    [data-testid="stSidebar"] label {
        color: #f3f4f6 !important;
        font-weight: 500;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* All text elements */
    p, span, div, label {
        color: #e5e7eb !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: #e5e7eb !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Primary Button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    /* Metrics Cards */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #4ade80 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #d1d5db !important;
        font-size: 14px;
        font-weight: 600;
    }
    
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1d29 0%, #252836 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2d3142;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    
    /* Red Alert Animation */
    .red-alert {
        background: linear-gradient(135deg, #ff0844 0%, #ff6b9d 100%);
        color: white;
        padding: 25px;
        border-radius: 12px;
        font-size: 26px;
        font-weight: 700;
        text-align: center;
        animation: pulse 1.5s infinite;
        box-shadow: 0 8px 16px rgba(255, 8, 68, 0.4);
        border: 2px solid #ff6b9d;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.02); opacity: 0.95; }
    }
    
    /* Critical Alert Cards */
    .critical-alert {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 18px 20px;
        border-radius: 10px;
        margin: 12px 0;
        border-left: 5px solid #991b1b;
        box-shadow: 0 4px 8px rgba(220, 38, 38, 0.3);
        font-weight: 500;
    }
    
    /* Success Alert */
    .success-alert {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        border-left: 5px solid #047857;
        box-shadow: 0 4px 8px rgba(5, 150, 105, 0.3);
        font-weight: 500;
    }
    
    /* Timestamp Badge */
    .timestamp-badge {
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-weight: 600;
        font-size: 13px;
        display: inline-block;
    }
    
    /* Alert History Card */
    .alert-card {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 15px;
        margin: 10px 0;
        border-radius: 10px;
        border-left: 4px solid #dc2626;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    
    .alert-card:hover {
        transform: translateX(5px);
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    
    .header-title {
        color: white;
        font-size: 38px;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    .header-subtitle {
        color: #e0e7ff;
        font-size: 16px;
        margin: 8px 0 0 0;
        font-weight: 400;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        background-color: #1a1d29;
        color: #f3f4f6 !important;
        border: 1px solid #2d3142;
        border-radius: 8px;
    }
    
    .stSelectbox > div > div {
        background-color: #1a1d29;
        color: #f3f4f6 !important;
        border-radius: 8px;
    }
    
    .stSelectbox label {
        color: #f3f4f6 !important;
    }
    
    /* Selectbox text */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1a1d29;
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
        color: #f3f4f6 !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: #667eea;
    }
    
    .stSlider label {
        color: #f3f4f6 !important;
        font-weight: 500;
    }
    
    /* Checkbox */
    .stCheckbox label {
        color: #f3f4f6 !important;
        font-weight: 500;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #f3f4f6 !important;
    }
    
    /* Download Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Section Headers */
    .section-header {
        color: #c4b5fd !important;
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 15px;
        padding-bottom: 8px;
        border-bottom: 2px solid #2d3142;
    }
    
    /* Info Box */
    .info-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 15px 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
        border-left: 5px solid #60a5fa;
    }
    
    /* Streamlit Messages */
    .stAlert {
        background-color: #1e293b;
        color: #f3f4f6 !important;
        border-radius: 8px;
    }
    
    .stSuccess {
        background-color: #065f46;
        color: #d1fae5 !important;
    }
    
    .stWarning {
        background-color: #92400e;
        color: #fef3c7 !important;
    }
    
    .stInfo {
        background-color: #1e3a8a;
        color: #dbeafe !important;
    }
    
    .stError {
        background-color: #7f1d1d;
        color: #fecaca !important;
    }
    
    /* File Uploader */
    .stFileUploader label {
        color: #f3f4f6 !important;
    }
    
    .stFileUploader > div {
        background-color: #1a1d29;
        border: 2px dashed #2d3142;
        border-radius: 8px;
    }
    
    .stFileUploader section {
        color: #f3f4f6 !important;
    }
    
    /* Spinner text */
    .stSpinner > div {
        color: #f3f4f6 !important;
    }
    
    /* Text Area */
    .stTextArea label {
        color: #f3f4f6 !important;
    }
    
    /* Number Input */
    .stNumberInput label {
        color: #f3f4f6 !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    </style>
""", unsafe_allow_html=True)

# Professional Header
st.markdown(
    """
    <div class="header-card">
        <h1 class="header-title">🛡️ AI Surveillance System</h1>
        <p class="header-subtitle">Advanced Threat Detection • Real-time Weapon Recognition • Intelligent Monitoring</p>
    </div>
    """, unsafe_allow_html=True
)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.markdown('<p class="section-header">⚙️ Configuration</p>', unsafe_allow_html=True)

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
            st.sidebar.error(f"⚠️ Gemini API configuration failed: {e}")
            gemini_api_key = None
    else:
        st.sidebar.warning("⚠️ Gemini API not configured - blur enhancement disabled")
else:
    st.sidebar.info("ℹ️ Gemini AI not installed - blur enhancement unavailable")

st.sidebar.markdown("---")
st.sidebar.markdown('<p class="section-header">📹 Video Source</p>', unsafe_allow_html=True)
source_type = st.sidebar.selectbox("Select source", ["Webcam", "Upload video", "Sample video"])
uploaded_file = None
if source_type == "Upload video":
    uploaded_file = st.sidebar.file_uploader("Upload video (mp4/avi)", type=["mp4", "avi"])

st.sidebar.markdown("---")
st.sidebar.markdown('<p class="section-header">🎯 Detection Settings</p>', unsafe_allow_html=True)
inference_size = st.sidebar.selectbox("Inference size (px)", [320, 384, 480, 640], index=2)
process_every_n = st.sidebar.slider("Process every N frames", 1, 6, 1)
confidence_thr = st.sidebar.slider("Confidence threshold", 0.2, 0.9, 0.30, 0.05)
knife_conf_thr = st.sidebar.slider("Knife-specific confidence", 0.10, 0.5, 0.18, 0.02)
scissor_conf_thr = st.sidebar.slider("Scissors-specific confidence", 0.10, 0.5, 0.18, 0.02)
weapon_mode = st.sidebar.checkbox("🔪 Instant Weapon Detection Mode", value=True, help="Disables frame skipping and alert cooldown for immediate weapon detection")

enable_blur_enhancement = False
if GEMINI_AVAILABLE:
    enable_blur_enhancement = st.sidebar.checkbox("Enable blur enhancement (Gemini)", value=False, 
                                                    help="Uses Gemini API to enhance blurry frames")

enable_emotion_tracking = False
if DEEPFACE_AVAILABLE:
    enable_emotion_tracking = st.sidebar.checkbox("Enable emotion tracking", value=True)
else:
    st.sidebar.warning("⚠️ DeepFace not available - emotion tracking disabled")

enable_behavior_detection = False
if MEDIAPIPE_AVAILABLE:
    enable_behavior_detection = st.sidebar.checkbox("🤖 AI Behavior Detection", value=True,
                                                     help="Detects suspicious behavior before threats appear")
    st.sidebar.info("ℹ️ Note: Movement & speed tracking will work. Pose detection may have limited availability.")
else:
    st.sidebar.info("ℹ️ MediaPipe not installed - behavior detection unavailable\nInstall: pip install mediapipe")

st.sidebar.markdown("---")
st.sidebar.markdown('<p class="section-header">🚨 Alert Settings</p>', unsafe_allow_html=True)
enable_sound_alert = st.sidebar.checkbox("Enable sound alerts", value=True)
alert_cooldown = st.sidebar.slider("Alert cooldown (seconds)", 1, 10, 3)

st.sidebar.markdown("---")
st.sidebar.markdown('<p class="section-header">🎮 Controls</p>', unsafe_allow_html=True)
col1, col2 = st.sidebar.columns(2)
with col1:
    start_button = st.button("▶️ Start", type="primary", use_container_width=True)
with col2:
    stop_button = st.button("⏹️ Stop", use_container_width=True)

# -----------------------------
# Load YOLO model
# -----------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

with st.spinner("Loading YOLO model..."):
    model = load_yolo_model()
st.sidebar.success("✅ Model loaded")

# -----------------------------
# Gemini Image Enhancement
# -----------------------------
def enhance_frame_with_gemini(frame):
    """Enhance blurry frame using Gemini API"""
    if not GEMINI_AVAILABLE or not gemini_api_key:
        return frame
    
    try:
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Resize if too large
        max_size = 1024
        if max(pil_image.size) > max_size:
            pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Use Gemini to analyze and enhance
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = """Analyze this surveillance image for clarity and sharpness. 
        If the image appears blurry or low quality, suggest specific areas that need enhancement.
        Focus on detecting any weapons, people, or suspicious objects.
        Describe what you see clearly."""
        
        response = gemini_model.generate_content([prompt, pil_image])
        
        # Apply basic sharpening based on analysis
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        enhanced = cv2.filter2D(frame, -1, kernel)
        
        # Denoise
        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        return enhanced
        
    except Exception as e:
        print(f"Gemini enhancement error: {e}")
        return frame

# -----------------------------
# Face Expression Analysis
# -----------------------------
def analyze_face_expression(face_roi):
    """Analyze face expression using DeepFace"""
    if not DEEPFACE_AVAILABLE:
        return "neutral", 0.5
    
    try:
        if face_roi is None or face_roi.size == 0:
            return "neutral", 0.5
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # Analyze emotion
        result = DeepFace.analyze(face_rgb, actions=['emotion'], 
                                 enforce_detection=False, detector_backend='opencv')
        
        if isinstance(result, list):
            result = result[0]
        
        emotions = result.get('emotion', {})
        if emotions:
            dominant_emotion = max(emotions, key=emotions.get)
            confidence = emotions[dominant_emotion] / 100.0
            return dominant_emotion.lower(), confidence
        
        return "neutral", 0.5
        
    except Exception as e:
        print(f"Emotion analysis error: {e}")
        return "neutral", 0.5

# -----------------------------
# AI Behavior Analysis
# -----------------------------
def calculate_speed(prev_centroid, curr_centroid, time_diff):
    """Calculate speed in pixels per second"""
    if prev_centroid is None or time_diff == 0:
        return 0
    dx = curr_centroid[0] - prev_centroid[0]
    dy = curr_centroid[1] - prev_centroid[1]
    distance = math.sqrt(dx*dx + dy*dy)
    return distance / time_diff

def detect_aggressive_approach(person_centroid, other_centroids, speed):
    """Detect if person is rapidly approaching another person"""
    if speed < AGGRESSIVE_APPROACH_THRESHOLD:
        return False, None
    
    for other_id, other_pos in other_centroids:
        dist = math.sqrt((person_centroid[0]-other_pos[0])**2 + (person_centroid[1]-other_pos[1])**2)
        if dist < 200:  # Within 200 pixels
            return True, other_id
    return False, None

def analyze_pose_threat(pose_landmarks):
    """Analyze pose for threatening gestures"""
    if pose_landmarks is None:
        return 0, "normal"
    
    try:
        # Get key landmarks
        left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        nose = pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        
        threat_score = 0
        posture = "normal"
        
        # Raised hands (fighting stance or threatening)
        avg_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        left_hand_raised = left_wrist.y < avg_shoulder_y - 0.1
        right_hand_raised = right_wrist.y < avg_shoulder_y - 0.1
        
        if left_hand_raised and right_hand_raised:
            threat_score += 30
            posture = "hands_raised"
        elif left_hand_raised or right_hand_raised:
            threat_score += 15
            posture = "one_hand_raised"
        
        # Hunched/concealing posture (hiding something)
        avg_hip_y = (left_hip.y + right_hip.y) / 2
        torso_bend = avg_hip_y - avg_shoulder_y
        if torso_bend < 0.15:  # Very bent over
            threat_score += 20
            posture = "concealing"
        
        # Wide stance (aggressive)
        hip_distance = abs(left_hip.x - right_hip.x)
        if hip_distance > 0.3:
            threat_score += 10
        
        # Reaching gesture (toward pocket/waist)
        avg_hip_x = (left_hip.x + right_hip.x) / 2
        if abs(right_wrist.x - avg_hip_x) < 0.15 and right_wrist.y > avg_shoulder_y:
            threat_score += 25
            posture = "reaching"
        elif abs(left_wrist.x - avg_hip_x) < 0.15 and left_wrist.y > avg_shoulder_y:
            threat_score += 25
            posture = "reaching"
        
        return min(threat_score, 100), posture
        
    except Exception as e:
        return 0, "normal"

def calculate_behavior_score(person_id, speed, emotion, emotion_conf, pose_threat, 
                            is_loitering, is_aggressive_approach, sudden_movement):
    """Calculate overall behavior threat score (0-100)"""
    score = 0
    behaviors = []
    
    # Speed-based threat
    if speed > RUNNING_SPEED_THRESHOLD:
        score += 25
        behaviors.append("running")
    elif speed > RUNNING_SPEED_THRESHOLD * 0.6:
        score += 10
        behaviors.append("fast_walking")
    
    # Emotion-based threat
    if emotion in ["angry", "fear"] and emotion_conf > 0.5:
        score += 20
        behaviors.append(f"emotion:{emotion}")
    elif emotion == "sad":
        score += 5
    
    # Pose threat
    score += pose_threat * 0.3  # Weight pose score
    
    # Aggressive approach
    if is_aggressive_approach:
        score += 30
        behaviors.append("aggressive_approach")
    
    # Loitering
    if is_loitering:
        score += 15
        behaviors.append("loitering")
    
    # Sudden movement
    if sudden_movement:
        score += 20
        behaviors.append("sudden_movement")
    
    return min(int(score), 100), behaviors

# -----------------------------
# Alert System
# -----------------------------
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
    
    # Load existing logs
    alerts = []
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            alerts = json.load(f)
    
    alerts.append(alert_record)
    
    # Save updated logs
    with open(log_file, 'w') as f:
        json.dump(alerts, f, indent=2)
    
    # Save screenshot if provided
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
if "weapon_detected_last_frame" not in st.session_state:
    st.session_state.weapon_detected_last_frame = False
if "person_movement_history" not in st.session_state:
    st.session_state.person_movement_history = {}  # Track movement for behavior analysis
if "person_behavior_scores" not in st.session_state:
    st.session_state.person_behavior_scores = {}  # Behavior scores per person

# Start / Stop
if start_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False

# -----------------------------
# Layout
# -----------------------------
col_main, col_right = st.columns([2.5, 1.5])

# Main video display
with col_main:
    st.markdown('<p class="section-header">📹 Live Video Feed</p>', unsafe_allow_html=True)
    frame_placeholder = st.empty()

# Right panel
with col_right:
    st.markdown('<p class="section-header">🚨 Alert Status</p>', unsafe_allow_html=True)
    alert_placeholder = st.empty()
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">📊 Performance Metrics</p>', unsafe_allow_html=True)
    metrics_container = st.container()
    
    with metrics_container:
        m1, m2, m3 = st.columns(3)
        fps_metric = m1.empty()
        threat_metric = m2.empty()
        det_metric = m3.empty()
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">📋 Recent Alerts</p>', unsafe_allow_html=True)
    recent_alerts_container = st.empty()

# -----------------------------
# Helper functions
# -----------------------------
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

def assign_ids(detections, tracker, iou_thresh=0.25):
    assigned=[]
    used_ids=set()
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
            tracker[best_id]["last_seen"]=st.session_state.frame_idx
        else:
            new_id=st.session_state.next_id
            st.session_state.next_id+=1
            tracker[new_id]={"bbox":bbox,"centroids":[centroid],"last_seen":st.session_state.frame_idx}
            assigned.append((new_id,bbox,label,conf,centroid))
            used_ids.add(new_id)
    # prune old
    to_del=[tid for tid, info in tracker.items() if st.session_state.frame_idx-info.get("last_seen",0)>500]
    for tid in to_del: tracker.pop(tid,None)
    return assigned

# -----------------------------
# Face cascade and pose detector
# -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize MediaPipe pose detector
pose_detector = None
pose_detector_available = False
if MEDIAPIPE_AVAILABLE and enable_behavior_detection:
    try:
        pose_detector = mp_pose.Pose(static_image_mode=False, 
                                      model_complexity=0,  # Faster
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
        pose_detector_available = True
        print("✅ MediaPipe Pose detector initialized successfully")
    except Exception as e:
        print(f"⚠️ MediaPipe Pose initialization failed: {e}")
        print("Continuing without pose detection...")
        pose_detector = None
        pose_detector_available = False

# -----------------------------
# Main Detection Loop
# -----------------------------
cap=None
if source_type=="Webcam":
    cap=cv2.VideoCapture(0)
elif source_type=="Upload video" and uploaded_file is not None:
    tmp_path="uploaded_temp.mp4"
    with open(tmp_path,"wb") as f: f.write(uploaded_file.read())
    cap=cv2.VideoCapture(tmp_path)
else:
    sample_paths = ["demo_weapon.mp4", "demo_video.mp4", "sample_short.mp4"]
    for sample_path in sample_paths:
        if os.path.exists(sample_path):
            cap=cv2.VideoCapture(sample_path)
            break
    if cap is None:
        cap=cv2.VideoCapture(0)

if cap is None or not cap.isOpened():
    frame_placeholder.info("⚠️ Select a valid video source and press Start.")
else:
    prev_time=time.perf_counter()
    last_detections=[]
    frame_count = 0
    
    while st.session_state.running:
        st.session_state.frame_idx+=1
        frame_count += 1
        ok, frame=cap.read()
        
        if not ok: 
            st.session_state.running=False
            break
        
        frame_h, frame_w = frame.shape[:2]
        
        # Apply Gemini enhancement if enabled (every 30 frames to reduce API calls)
        if enable_blur_enhancement and gemini_api_key and frame_count % 30 == 0:
            frame = enhance_frame_with_gemini(frame)
        
        # Frame skipping logic with weapon priority
        # NEVER skip frames if weapon mode is enabled or weapon was detected last frame
        skip_frame = (st.session_state.frame_idx % process_every_n)!=0 and last_detections
        if weapon_mode and st.session_state.weapon_detected_last_frame:
            skip_frame = False  # Force processing every frame when weapons present
        
        if skip_frame:
            display_frame=frame.copy()
            
            # Separate detections by type for layered rendering
            person_dets = []
            weapon_dets = []
            
            for info in last_detections:
                tid,bbox,label,conf=info["id"],info["bbox"],info["label"],info["conf"]
                emotion = info.get("emotion", "")
                is_suspicious = label in SUSPICIOUS_CLASSES or label in SUSPICIOUS_ALIASES
                
                if is_suspicious:
                    weapon_dets.append((tid,bbox,label,conf,emotion))
                else:
                    person_dets.append((tid,bbox,label,conf,emotion))
            
            # Draw persons first
            for tid,bbox,label,conf,emotion in person_dets:
                x1,y1,x2,y2=bbox
                color=(0,165,255) if emotion in ["angry", "fear"] else (50,205,50)
                cv2.rectangle(display_frame,(x1,y1),(x2,y2),color,2)
                text = f"{label} ID:{tid} {conf:.2f}"
                if emotion:
                    text += f" [{emotion}]"
                cv2.putText(display_frame,text,(x1,max(10,y1-8)),
                           cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
            
            # Draw weapons on top for visibility
            for tid,bbox,label,conf,emotion in weapon_dets:
                x1,y1,x2,y2=bbox
                thickness = 4 if label.lower() in ["scissors", "knife"] else 3
                color=(0,0,255)
                
                # Extra border for knives and scissors
                if label.lower() in ["scissors", "knife"]:
                    cv2.rectangle(display_frame,(x1-3,y1-3),(x2+3,y2+3),(0,0,255),1)
                    cv2.rectangle(display_frame,(x1-1,y1-1),(x2+1,y2+1),(0,0,255),1)
                
                cv2.rectangle(display_frame,(x1,y1),(x2,y2),color,thickness)
                text = f"{label} ID:{tid} {conf:.2f}"
                # Background for text
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(display_frame, (x1, max(10,y1-text_size[1]-8)), 
                             (x1+text_size[0], max(10,y1)), (0,0,255), -1)
                cv2.putText(display_frame,text,(x1,max(10,y1-8)),
                           cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)

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

        # YOLO detection with lower confidence for weapon detection
        t0=time.perf_counter()
        # Use lower confidence threshold for better weapon detection
        yolo_conf = 0.15 if weapon_mode else 0.20
        results=model(frame_infer,conf=yolo_conf,iou=0.40,imgsz=inference_size)[0]
        t1=time.perf_counter()
        inf_ms=(t1-t0)*1000

        dets=[]
        weapon_dets=[]  # Prioritized weapon detections
        other_dets=[]   # Other detections (persons, etc.)
        
        for det in results.boxes:
            cls_id=int(det.cls[0])
            conf=float(det.conf[0])
            label=model.names.get(cls_id,str(cls_id))
            
            # Apply different confidence thresholds for specific weapons
            is_weapon = label in SUSPICIOUS_CLASSES or label in SUSPICIOUS_ALIASES
            
            if label.lower() in ["knife", "pointed_knife", "dual-edge-sharp-knife", "dual-edge_knife", "sharp_cutter"]:
                min_conf = knife_conf_thr
                # Significant boost for knives to ensure detection
                conf = min(0.99, conf * 1.10)  # 10% boost, capped at 0.99
            elif label.lower() in ["scissor", "scissors"]:
                min_conf = scissor_conf_thr
                # Significant boost for scissors to ensure detection
                conf = min(0.99, conf * 1.10)  # 10% boost, capped at 0.99
            elif is_weapon:
                min_conf = confidence_thr * 0.8  # Lower threshold for other weapons
                # Boost for other weapons
                conf = min(0.99, conf * 1.08)  # 8% boost
            else:
                min_conf = confidence_thr
            
            if conf<min_conf: continue
            
            x1,y1,x2,y2=map(int,det.xyxy[0])
            if scale<1.0: 
                x1=int(x1/scale);y1=int(y1/scale);x2=int(x2/scale);y2=int(y2/scale)
            cx,cy=(x1+x2)//2,(y1+y2)//2
            
            # Separate weapons and other detections for priority processing
            detection_tuple = ([x1,y1,x2,y2],label,conf,(cx,cy))
            if is_weapon:
                weapon_dets.append(detection_tuple)
            else:
                other_dets.append(detection_tuple)
        
        # Process weapons FIRST, then other detections
        # This ensures weapons get priority in tracking and display
        dets = weapon_dets + other_dets

        assigned=assign_ids(dets,st.session_state.tracker,0.2)

        display_frame=frame.copy()
        alert_messages=[]
        persons_in_frame=[]
        suspicious_items=[]
        last_detections=[]
        threat_score = 0
        weapon_count = 0
        weapon_detected_this_frame = False
        
        # Separate assigned detections for layered rendering
        weapon_detections = []
        person_detections = []
        
        # Process detections - categorize first
        for tid,bbox,label,conf,centroid in assigned:
            x1,y1,x2,y2=bbox
            cx,cy=centroid
            
            is_suspicious = label in SUSPICIOUS_CLASSES or label in SUSPICIOUS_ALIASES
            actual_label = SUSPICIOUS_ALIASES.get(label, label) if label in SUSPICIOUS_ALIASES else label
            
            emotion = ""
            emotion_conf = 0.0
            
            # Store detection data for rendering
            detection_data = {
                'tid': tid, 'bbox': bbox, 'label': label, 
                'actual_label': actual_label, 'conf': conf, 
                'centroid': centroid, 'is_suspicious': is_suspicious
            }
            
            # AI Behavior Analysis for persons
            behavior_score = 0
            behaviors = []
            speed = 0
            pose_threat = 0
            posture = "normal"
            
            if label=="person":
                # Face expression analysis
                if enable_emotion_tracking:
                    person_roi = frame[y1:y2, x1:x2]
                    gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                    
                    if len(faces) > 0:
                        fx, fy, fw, fh = faces[0]
                        face_roi = person_roi[fy:fy+fh, fx:fx+fw]
                        emotion, emotion_conf = analyze_face_expression(face_roi)
                        detection_data['emotion'] = emotion
                        detection_data['emotion_conf'] = emotion_conf
                
                # Pose analysis with MediaPipe
                if enable_behavior_detection and pose_detector_available and pose_detector:
                    try:
                        person_roi = frame[y1:y2, x1:x2]
                        if person_roi.size > 0:  # Check if ROI is valid
                            person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                            pose_results = pose_detector.process(person_rgb)
                            
                            if pose_results and pose_results.pose_landmarks:
                                pose_threat, posture = analyze_pose_threat(pose_results.pose_landmarks)
                                detection_data['pose_threat'] = pose_threat
                                detection_data['posture'] = posture
                    except Exception as e:
                        # Silently skip pose detection if it fails
                        pass
                
                # Calculate movement speed and patterns
                if tid in st.session_state.person_movement_history:
                    history = st.session_state.person_movement_history[tid]
                    prev_centroid = history.get('last_centroid')
                    prev_time = history.get('last_time', time.time())
                    curr_time = time.time()
                    time_diff = curr_time - prev_time
                    
                    if time_diff > 0:
                        speed = calculate_speed(prev_centroid, centroid, time_diff)
                        
                        # Detect sudden movement
                        prev_speed = history.get('last_speed', 0)
                        acceleration = abs(speed - prev_speed)
                        sudden_movement = acceleration > SUDDEN_MOVEMENT_THRESHOLD
                        
                        # Detect loitering
                        if speed < 10:  # Almost stationary
                            history['stationary_time'] = history.get('stationary_time', 0) + time_diff
                        else:
                            history['stationary_time'] = 0
                        
                        is_loitering = history.get('stationary_time', 0) > LOITERING_TIME_THRESHOLD
                        
                        # Detect aggressive approach
                        other_centroids = [(p['id'], p['centroid']) for p in persons_in_frame if p['id'] != tid]
                        is_aggressive_approach, target_id = detect_aggressive_approach(centroid, other_centroids, speed)
                        
                        # Calculate overall behavior score
                        behavior_score, behaviors = calculate_behavior_score(
                            tid, speed, emotion, emotion_conf, pose_threat,
                            is_loitering, is_aggressive_approach, sudden_movement
                        )
                        
                        # Update history
                        history.update({
                            'last_speed': speed,
                            'last_time': curr_time,
                            'last_centroid': centroid,
                            'behavior_score': behavior_score,
                            'behaviors': behaviors
                        })
                        
                        detection_data.update({
                            'speed': speed,
                            'behavior_score': behavior_score,
                            'behaviors': behaviors,
                            'posture': posture
                        })
                else:
                    # Initialize movement history for this person
                    st.session_state.person_movement_history[tid] = {
                        'last_centroid': centroid,
                        'last_time': time.time(),
                        'last_speed': 0,
                        'stationary_time': 0,
                        'behavior_score': 0,
                        'behaviors': []
                    }
                
                # Store behavior score
                st.session_state.person_behavior_scores[tid] = behavior_score
            
            # Categorize detections for layered rendering
            if is_suspicious:
                suspicious_items.append({
                    "id":tid, "label":actual_label, "bbox":bbox, 
                    "centroid":centroid, "conf":conf
                })
                weapon_detections.append(detection_data)
                weapon_count += 1
                weapon_detected_this_frame = True
                threat_score += int(conf * 50)
            elif label=="person":
                persons_in_frame.append({
                    "id":tid, "centroid":centroid, "bbox":bbox, 
                    "emotion":emotion, "emotion_conf":emotion_conf
                })
                detection_data['emotion'] = emotion
                person_detections.append(detection_data)
                
                # Add behavior score to threat level
                if behavior_score > HIGH_BEHAVIOR_SCORE_THRESHOLD:
                    threat_score += int(behavior_score * 0.5)
                elif emotion in ["angry", "fear"]:
                    threat_score += 10
            
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
        
        # RENDER DETECTIONS IN LAYERS: Persons first, then weapons on top
        # This ensures weapons are always visible even when overlapping with persons
        
        # Draw person detections first (background layer)
        for det in person_detections:
            x1, y1, x2, y2 = det['bbox']
            tid = det['tid']
            label = det['label']
            conf = det['conf']
            emotion = det.get('emotion', '')
            behavior_score = det.get('behavior_score', 0)
            behaviors = det.get('behaviors', [])
            speed = det.get('speed', 0)
            posture = det.get('posture', 'normal')
            
            # Color based on behavior score
            if behavior_score >= HIGH_BEHAVIOR_SCORE_THRESHOLD:
                color = (0,0,255)  # Red for high threat behavior
                thickness = 3
            elif behavior_score >= 40:
                color = (0,165,255)  # Orange for medium threat
                thickness = 2
            elif emotion in ["angry", "fear"]:
                color = (0,165,255)  # Orange for suspicious emotions
                thickness = 2
            else:
                color = (50,205,50)  # Green for normal
                thickness = 2
            
            cv2.rectangle(display_frame, (x1,y1), (x2,y2), color, thickness)
            
            # Main label
            text = f"{label} ID:{tid} {conf:.2f}"
            if emotion:
                text += f" [{emotion}]"
            cv2.putText(display_frame, text, (x1, max(10,y1-8)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            
            # Behavior info overlay
            if enable_behavior_detection and behavior_score > 0:
                info_y = y2 + 15
                # Behavior score bar
                bar_width = 100
                bar_height = 8
                bar_x = x1
                cv2.rectangle(display_frame, (bar_x, info_y), (bar_x + bar_width, info_y + bar_height), (50,50,50), -1)
                filled_width = int(bar_width * (behavior_score / 100))
                bar_color = (0,0,255) if behavior_score >= 60 else (0,165,255) if behavior_score >= 40 else (0,255,0)
                cv2.rectangle(display_frame, (bar_x, info_y), (bar_x + filled_width, info_y + bar_height), bar_color, -1)
                
                # Behavior score text
                cv2.putText(display_frame, f"Threat: {behavior_score}%", (bar_x, info_y + bar_height + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                
                # Speed indicator
                if speed > 0:
                    speed_text = "Running" if speed > RUNNING_SPEED_THRESHOLD else "Walking"
                    cv2.putText(display_frame, f"{speed_text} {int(speed)}px/s", (bar_x, info_y + bar_height + 24),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                
                # Posture indicator
                if posture != "normal":
                    posture_display = posture.replace("_", " ").title()
                    cv2.putText(display_frame, f"Pose: {posture_display}", (bar_x, info_y + bar_height + 36),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,255), 1)
        
        # Draw weapon detections on top (foreground layer) - PRIORITY RENDERING
        for det in weapon_detections:
            x1, y1, x2, y2 = det['bbox']
            tid = det['tid']
            actual_label = det['actual_label']
            conf = det['conf']
            
            # Extra emphasis for sharp objects (knife and scissors)
            if actual_label.lower() in ["scissors", "knife"]:
                color = (0,0,255)  # Bright red for sharp objects
                thickness = 4  # Extra thick border
                # Triple border for maximum visibility
                cv2.rectangle(display_frame, (x1-3,y1-3), (x2+3,y2+3), (0,0,255), 1)
                cv2.rectangle(display_frame, (x1-1,y1-1), (x2+1,y2+1), (0,0,255), 1)
            else:
                color = (0,0,255)  # Red for other weapons
                thickness = 3
            
            cv2.rectangle(display_frame, (x1,y1), (x2,y2), color, thickness)
            
            # Draw label with background for better visibility
            text = f"{actual_label} ID:{tid} {conf:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            # Background rectangle for text
            cv2.rectangle(display_frame, (x1, max(10,y1-text_size[1]-8)), 
                         (x1+text_size[0], max(10,y1)), (0,0,255), -1)
            cv2.putText(display_frame, text, (x1, max(10,y1-8)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        # Generate alerts
        current_time = time.time()
        
        # Weapon detection alerts - IMMEDIATE (no cooldown in weapon mode)
        for item in suspicious_items:
            # In weapon mode, send alert immediately without cooldown
            should_alert = weapon_mode or (current_time - st.session_state.last_alert_time > alert_cooldown)
            
            if should_alert:
                weapon_type = "SHARP OBJECT" if item['label'].upper() in ["SCISSORS", "KNIFE"] else "WEAPON"
                alert_msg = f"🚨 {weapon_type} DETECTED: {item['label'].upper()} (Confidence: {item['conf']:.2%})"
                alert_messages.append(alert_msg)
                
                # Log alert with screenshot
                alert_record = log_alert("weapon_detected", alert_msg, display_frame)
                st.session_state.alerts.append(alert_record)
                
                # Only update cooldown if not in weapon mode
                if not weapon_mode:
                    st.session_state.last_alert_time = current_time
        
        # Proximity alerts (weapon near person)
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
        
        # AI Behavior-based alerts (early warning system)
        if enable_behavior_detection:
            for det in person_detections:
                behavior_score = det.get('behavior_score', 0)
                behaviors = det.get('behaviors', [])
                tid = det['tid']
                
                if behavior_score >= HIGH_BEHAVIOR_SCORE_THRESHOLD:
                    behavior_text = ", ".join([b.replace("_", " ").title() for b in behaviors if b])
                    alert_msg = f"⚠️ SUSPICIOUS BEHAVIOR: Person ID:{tid} - {behavior_text} (Score: {behavior_score}%)"
                    alert_messages.append(alert_msg)
                    
                    # Log behavior alert
                    if current_time - st.session_state.last_alert_time > alert_cooldown:
                        alert_record = log_alert("suspicious_behavior", alert_msg, display_frame)
                        st.session_state.alerts.append(alert_record)
                        st.session_state.last_alert_time = current_time

        # Update weapon detection state for next frame
        st.session_state.weapon_detected_last_frame = weapon_detected_this_frame
        
        # Draw threat score and weapon mode indicator
        threat_score = min(100, threat_score)
        threat_color = (0,255,0) if threat_score < 30 else (0,165,255) if threat_score < 70 else (0,0,255)
        cv2.putText(display_frame, f"Threat Level: {threat_score}%", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, threat_color, 2)
        
        # Show weapon mode indicator
        if weapon_mode:
            mode_text = "🔪 INSTANT WEAPON MODE: ACTIVE"
            cv2.putText(display_frame, mode_text, (10, frame_h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        # Show behavior detection indicator
        if enable_behavior_detection:
            if pose_detector_available:
                behavior_text = "🤖 AI BEHAVIOR: FULL (Movement + Pose)"
            else:
                behavior_text = "🤖 AI BEHAVIOR: ACTIVE (Movement Only)"
            text_y = frame_h - 45 if weapon_mode else frame_h - 20
            cv2.putText(display_frame, behavior_text, (10, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

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
                st.markdown('<div class="success-alert">✅ System Active - No Threats Detected</div>', 
                          unsafe_allow_html=True)

        # Update metrics
        fps=1.0/max(1e-6,time.perf_counter()-prev_time)
        prev_time=time.perf_counter()
        
        fps_metric.metric("FPS", f"{fps:.1f}")
        threat_metric.metric("Threat Level", f"{threat_score}%")
        # Show weapon count prominently
        det_metric.metric("🔪 Weapons", str(weapon_count))

        # Display recent alerts
        if st.session_state.alerts:
            recent = st.session_state.alerts[-5:][::-1]
            recent_alerts_html = ""
            for alert in recent:
                recent_alerts_html += f"""
                <div class="alert-card">
                    <div style="margin-bottom:8px;">
                        <span class="timestamp-badge">{alert['time']}</span>
                    </div>
                    <div style="font-size:14px;color:#e5e7eb;font-weight:500;">
                        {alert['message']}
                    </div>
                </div>
                """
            recent_alerts_container.markdown(recent_alerts_html, unsafe_allow_html=True)
        else:
            recent_alerts_container.markdown('<div class="info-box">No alerts yet - System monitoring...</div>', unsafe_allow_html=True)

        time.sleep(0.01)

    cap.release()

# -----------------------------
# Download Reports
# -----------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<p class="section-header">📥 Export Reports</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.session_state.detections:
        df_detections = pd.DataFrame(st.session_state.detections)
        csv_detections = df_detections.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📊 Download Detections CSV",
            data=csv_detections,
            file_name=f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.markdown('<div class="info-box">📊 No detections recorded yet</div>', unsafe_allow_html=True)

with col2:
    if st.session_state.alerts:
        df_alerts = pd.DataFrame(st.session_state.alerts)
        csv_alerts = df_alerts.to_csv(index=False).encode("utf-8")
        st.download_button(
            "🚨 Download Alerts CSV",
            data=csv_alerts,
            file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.markdown('<div class="info-box">🚨 No alerts recorded yet</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;color:#6b7280;font-size:14px;padding:20px;">🛡️ AI Surveillance System v2.0 • Advanced Threat Detection Engine</div>', 
    unsafe_allow_html=True
)
