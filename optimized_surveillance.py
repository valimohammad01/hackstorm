# Optimized Surveillance AI for Smooth Laptop Performance
import streamlit as st
import cv2
import numpy as np
try:
    import pandas as pd
except Exception:
    pd = None
import os
import time
import gc
import threading
import queue
from datetime import datetime
from ultralytics import YOLO
import torch
import psutil
import platform
from detection_engine import DetectionEngine

# -----------------------------
# PERFORMANCE CONFIGURATION
# -----------------------------
class PerformanceConfig:
    def __init__(self):
        self.detect_hardware()
        self.setup_optimization()
    
    def detect_hardware(self):
        """Auto-detect available hardware and configure accordingly"""
        self.cpu_count = psutil.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # GPU Detection
        self.has_gpu = torch.cuda.is_available()
        self.device = 'cuda' if self.has_gpu else 'cpu'
        
        # Performance tier based on hardware
        if self.memory_gb >= 16 and self.has_gpu:
            self.tier = "high"
        elif self.memory_gb >= 8:
            self.tier = "medium"
        else:
            self.tier = "low"
    
    def setup_optimization(self):
        """Configure optimization settings based on hardware"""
        if self.tier == "high":
            self.inference_size = 640
            self.frame_skip = 2
            self.target_fps = 20
            self.batch_size = 4
        elif self.tier == "medium":
            self.inference_size = 480
            self.frame_skip = 3
            self.target_fps = 15
            self.batch_size = 2
        else:  # low tier
            self.inference_size = 320
            self.frame_skip = 4
            self.target_fps = 10
            self.batch_size = 1
        
        # Memory management
        self.gc_interval = 100  # frames between garbage collection
        self.max_detections = 1000  # limit stored detections

# Initialize performance config
perf_config = PerformanceConfig()

# -----------------------------
# DETECTION ENGINE LOADING
# -----------------------------
@st.cache_resource
def load_detection_engine():
    """Load DetectionEngine with optional custom weights and optimizations"""
    try:
        # Prefer custom-trained weights if available
        custom_weights = os.path.join("suspicious_data", "data", "runs", "suspicious_model", "weights", "best.pt")
        engine = DetectionEngine()
        # If custom weights exist, replace YOLO model inside engine
        if os.path.exists(custom_weights):
            try:
                engine.yolo_model = YOLO(custom_weights)
                if engine.device == 'cuda':
                    engine.yolo_model.to('cuda')
            except Exception:
                pass
        return engine
    except Exception as e:
        st.error(f"Detection engine loading failed: {e}")
        return None

# -----------------------------
# THREADED VIDEO PROCESSOR
# -----------------------------
class VideoProcessor:
    def __init__(self, engine, config):
        self.engine = engine
        self.config = config
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.processing = False
        self.thread = None
        
    def start_processing(self):
        """Start background processing thread"""
        self.processing = True
        self.thread = threading.Thread(target=self._process_frames, daemon=True)
        self.thread.start()
    
    def stop_processing(self):
        """Stop background processing"""
        self.processing = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def add_frame(self, frame):
        """Add frame to processing queue"""
        if not self.frame_queue.full():
            self.frame_queue.put(frame)
    
    def get_result(self):
        """Get processed result if available"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _process_frames(self):
        """Background frame processing"""
        while self.processing:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                result = self._process_single_frame(frame)
                if not self.result_queue.full():
                    self.result_queue.put(result)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def _process_single_frame(self, frame):
        """Analyze frame via DetectionEngine and return annotated frame and analysis"""
        try:
            # Optionally resize input for speed while keeping display at original size
            h, w = frame.shape[:2]
            scale = min(self.config.inference_size / w, self.config.inference_size / h)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                proc_frame = cv2.resize(frame, (new_w, new_h))
            else:
                proc_frame = frame

            analysis = self.engine.analyze_frame(proc_frame, confidence_threshold=0.35)
            annotated = self.engine.draw_detections(frame, analysis)  # draw on original for stable display

            return {
                'analysis': analysis,
                'annotated_frame': annotated,
                'timestamp': time.time()
            }
        except Exception as e:
            return {'analysis': {'objects': [], 'faces': [], 'alerts': [], 'processing_time_ms': 0}, 'error': str(e), 'timestamp': time.time()}

# -----------------------------
# DEMO MODE SETUP
# -----------------------------
def setup_demo_mode():
    """Setup demo videos and configurations"""
    demo_videos = {
        "Weapon Detection Demo": "demo_weapon.mp4",
        "Suspicious Behavior Demo": "demo_behavior.mp4", 
        "Multi-Person Tracking": "demo_tracking.mp4"
    }
    
    # Create demo video if not exists (placeholder)
    for name, filename in demo_videos.items():
        if not os.path.exists(filename):
            # Create a simple demo video using OpenCV
            create_demo_video(filename)
    
    return demo_videos

def create_demo_video(filename):
    """Create a simple demo video for testing"""
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
        
        for i in range(300):  # 15 seconds at 20 FPS
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            # Add some simple shapes to simulate objects
            cv2.rectangle(frame, (100+i, 100), (200+i, 200), (0, 255, 0), 2)
            cv2.circle(frame, (300, 200+int(50*np.sin(i/10))), 30, (255, 0, 0), -1)
            out.write(frame)
        
        out.release()
    except Exception as e:
        print(f"Demo video creation failed: {e}")

# -----------------------------
# STREAMLIT UI SETUP
# -----------------------------
st.set_page_config(
    page_title="Optimized AI Surveillance", 
    layout="wide", 
    page_icon="🛡️",
    initial_sidebar_state="expanded"
)

# Custom CSS for better performance display
st.markdown("""
<style>
.metric-container {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
.alert-box {
    background-color: #ff4b4b;
    color: white;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
}
.performance-good { color: #00ff00; }
.performance-warning { color: #ffaa00; }
.performance-critical { color: #ff0000; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(f"""
<div style="background: linear-gradient(90deg, #1f2937 0%, #374151 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px;">
    <h1 style="color: white; margin: 0;">🛡️ Optimized AI Surveillance</h1>
    <p style="color: #d1d5db; margin: 5px 0 0 0;">
        Hardware: {perf_config.tier.title()} Performance | 
        Device: {perf_config.device.upper()} | 
        RAM: {perf_config.memory_gb:.1f}GB | 
        Target FPS: {perf_config.target_fps}
    </p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("🎮 Control Panel")

# Demo mode selection
demo_videos = setup_demo_mode()
demo_mode = st.sidebar.checkbox("🎬 Demo Mode", value=False)

if demo_mode:
    demo_selection = st.sidebar.selectbox("Select Demo", list(demo_videos.keys()))
    source_type = "Demo"
else:
    source_type = st.sidebar.selectbox("Video Source", ["Webcam", "Upload Video"])

uploaded_file = None
if source_type == "Upload Video" and not demo_mode:
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Performance Settings")

# Performance settings with auto-optimization
auto_optimize = st.sidebar.checkbox("🚀 Auto Optimize", value=True)

if not auto_optimize:
    inference_size = st.sidebar.selectbox("Inference Size", [320, 480, 640], 
                                        index=[320, 480, 640].index(perf_config.inference_size))
    frame_skip = st.sidebar.slider("Frame Skip", 1, 6, perf_config.frame_skip)
    target_fps = st.sidebar.slider("Target FPS", 5, 30, perf_config.target_fps)
else:
    inference_size = perf_config.inference_size
    frame_skip = perf_config.frame_skip
    target_fps = perf_config.target_fps
    
    st.sidebar.info(f"Auto-optimized for {perf_config.tier} performance:\n"
                   f"• Inference: {inference_size}px\n"
                   f"• Frame skip: {frame_skip}\n"
                   f"• Target FPS: {target_fps}")

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("🚨 Alert Settings")
proximity_threshold = st.sidebar.slider("Proximity Alert (px)", 50, 200, 80)
loiter_threshold = st.sidebar.slider("Loiter Frames", 30, 200, 90)

# Control buttons
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
start_button = col1.button("▶️ Start", type="primary")
stop_button = col2.button("⏹️ Stop", type="secondary")

# Emergency stop
emergency_stop = st.sidebar.button("🚨 Emergency Stop", type="secondary")

# -----------------------------
# SESSION STATE INITIALIZATION
# -----------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "processor" not in st.session_state:
    st.session_state.processor = None
if "detections" not in st.session_state:
    st.session_state.detections = []
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0
if "last_gc" not in st.session_state:
    st.session_state.last_gc = 0
if "performance_stats" not in st.session_state:
    st.session_state.performance_stats = {
        'fps': 0, 'inference_ms': 0, 'memory_mb': 0, 'cpu_percent': 0
    }

# Handle control buttons
if start_button:
    st.session_state.running = True
if stop_button or emergency_stop:
    st.session_state.running = False
    if st.session_state.processor:
        st.session_state.processor.stop_processing()

# -----------------------------
# MAIN LAYOUT
# -----------------------------
col_main, col_sidebar = st.columns([2.5, 1])

with col_main:
    st.subheader("📹 Live Feed")
    frame_placeholder = st.empty()

with col_sidebar:
    st.subheader("📊 Performance Monitor")
    perf_container = st.container()
    
    st.subheader("🚨 Alert Center")
    alert_container = st.container()
    
    st.subheader("📈 Statistics")
    stats_container = st.container()

# -----------------------------
# MAIN PROCESSING LOOP
# -----------------------------
if st.session_state.running:
    # Load detection engine
    with st.spinner("🔄 Loading detection engine..."):
        engine = load_detection_engine()
    
    if engine is None:
        st.error("❌ Failed to load detection engine. Please check your installation.")
        st.session_state.running = False
    else:
        # Initialize video processor
        if st.session_state.processor is None:
            st.session_state.processor = VideoProcessor(engine, perf_config)
            st.session_state.processor.start_processing()
        
        # Setup video capture
        cap = None
        try:
            if demo_mode:
                video_path = demo_videos[demo_selection]
                cap = cv2.VideoCapture(video_path)
            elif source_type == "Webcam":
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, target_fps)
            elif uploaded_file is not None:
                temp_path = "temp_upload.mp4"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                cap = cv2.VideoCapture(temp_path)
            
            if cap is None or not cap.isOpened():
                st.error("❌ Failed to open video source")
                st.session_state.running = False
            else:
                # Main processing loop
                fps_counter = 0
                fps_start_time = time.time()
                last_process_time = 0
                
                while st.session_state.running:
                    loop_start = time.time()
                    
                    # Read frame
                    ret, frame = cap.read()
                    if not ret:
                        if demo_mode:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop demo
                            continue
                        else:
                            break
                    
                    st.session_state.frame_count += 1
                    
                    # Frame skipping logic
                    should_process = (st.session_state.frame_count % frame_skip) == 0
                    
                    if should_process:
                        # Add frame to processing queue
                        st.session_state.processor.add_frame(frame.copy())
                        last_process_time = time.time()
                    
                    # Get processing results
                    result = st.session_state.processor.get_result()
                    
                    # Draw frame using annotated output
                    display_frame = frame.copy()
                    current_analysis = None
                    if result and 'annotated_frame' in result:
                        display_frame = result['annotated_frame']
                        current_analysis = result.get('analysis', None)
                    
                    # Update display
                    frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Performance monitoring
                    fps_counter += 1
                    current_time = time.time()
                    
                    if current_time - fps_start_time >= 1.0:
                        current_fps = fps_counter / (current_time - fps_start_time)
                        fps_counter = 0
                        fps_start_time = current_time
                        
                        # Update performance stats
                        process = psutil.Process()
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        cpu_percent = process.cpu_percent()
                        
                        inf_ms = 0
                        if current_analysis and 'processing_time_ms' in current_analysis:
                            inf_ms = current_analysis['processing_time_ms']
                        st.session_state.performance_stats = {
                            'fps': current_fps,
                            'inference_ms': inf_ms,
                            'memory_mb': memory_mb,
                            'cpu_percent': cpu_percent
                        }
                    
                    # Update UI components
                    with perf_container:
                        stats = st.session_state.performance_stats
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            fps_color = "performance-good" if stats['fps'] >= target_fps * 0.8 else "performance-warning"
                            st.markdown(f"**FPS:** <span class='{fps_color}'>{stats['fps']:.1f}</span>", 
                                      unsafe_allow_html=True)
                            st.markdown(f"**Memory:** {stats['memory_mb']:.0f} MB")
                        
                        with col2:
                            st.markdown(f"**Inference:** {stats['inference_ms']:.1f} ms")
                            st.markdown(f"**CPU:** {stats['cpu_percent']:.1f}%")
                    
                    # Alert system
                    with alert_container:
                        if current_analysis and current_analysis.get('alerts'):
                            # Show most recent alerts
                            for alert in current_analysis['alerts'][-3:]:
                                st.markdown(f"""
                                <div class="alert-box">
                                    🚨 {alert['message']} (conf {alert['confidence']:.2f})
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.info("👁️ Monitoring...")
                    
                    # Statistics
                    with stats_container:
                        st.metric("Total Detections", len(st.session_state.detections))
                        st.metric("Frames Processed", st.session_state.frame_count)
                        
                        if current_analysis and current_analysis.get('objects'):
                            detection_labels = [o['class'] for o in current_analysis['objects']]
                            unique_labels = list(set(detection_labels))
                            st.write("**Current Objects:**")
                            for label in unique_labels:
                                count = detection_labels.count(label)
                                st.write(f"• {label}: {count}")
                    
                    # Memory management
                    if st.session_state.frame_count - st.session_state.last_gc > perf_config.gc_interval:
                        gc.collect()
                        st.session_state.last_gc = st.session_state.frame_count
                        
                        # Limit stored detections
                        if len(st.session_state.detections) > perf_config.max_detections:
                            st.session_state.detections = st.session_state.detections[-perf_config.max_detections//2:]
                    
                    # Frame rate control
                    loop_time = time.time() - loop_start
                    target_loop_time = 1.0 / target_fps
                    if loop_time < target_loop_time:
                        time.sleep(target_loop_time - loop_time)
                
                cap.release()
                
        except Exception as e:
            st.error(f"❌ Processing error: {e}")
            if cap:
                cap.release()
        
        finally:
            if st.session_state.processor:
                st.session_state.processor.stop_processing()
                st.session_state.processor = None

else:
    # Idle state
    with col_main:
        st.info("🎯 Ready to start surveillance. Click 'Start' to begin.")
        
        # Show hardware info when idle
        st.markdown("### 💻 System Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Performance Tier", perf_config.tier.title())
            st.metric("CPU Cores", perf_config.cpu_count)
        
        with col2:
            st.metric("Memory", f"{perf_config.memory_gb:.1f} GB")
            st.metric("Device", perf_config.device.upper())
        
        with col3:
            st.metric("Target FPS", perf_config.target_fps)
            st.metric("Inference Size", f"{perf_config.inference_size}px")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    🛡️ <strong>Optimized AI Surveillance</strong> | 
    Designed for smooth laptop performance | 
    Auto-optimized for your hardware
</div>
""", unsafe_allow_html=True)

# Download detection data
if st.session_state.detections:
    if pd is not None:
        df = pd.DataFrame(st.session_state.detections)
        csv_data = df.to_csv(index=False)
    else:
        # Minimal CSV serialization without pandas
        import io, csv
        buf = io.StringIO()
        fieldnames = list(st.session_state.detections[0].keys()) if st.session_state.detections else []
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for row in st.session_state.detections:
            writer.writerow(row)
        csv_data = buf.getvalue()
    st.download_button(
        "📥 Download Detection Report",
        data=csv_data,
        file_name=f"surveillance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
