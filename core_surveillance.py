"""
Enhanced Surveillance AI - Core Stable Version
Addresses: Video flickering, basic detection, smooth performance
"""

import streamlit as st
import cv2
import numpy as np
import time
import threading
import queue
from datetime import datetime
import gc
import os
from detection_engine import DetectionEngine
import io
import wave

# Core configuration
class SurveillanceConfig:
    def __init__(self):
        self.target_fps = 20
        self.display_width = 640
        self.display_height = 480
        self.inference_size = 640
        self.frame_skip = 2
        self.buffer_size = 5
        self.confidence_threshold = 0.5

config = SurveillanceConfig()

# Stable video processor with proper threading
class StableVideoProcessor:
    def __init__(self):
        self.frame_queue = queue.Queue(maxsize=config.buffer_size)
        self.display_queue = queue.Queue(maxsize=config.buffer_size)
        self.running = False
        self.capture_thread = None
        self.process_thread = None
        self.cap = None
        self.frame_count = 0
        self.last_display_frame = None
        self.detection_engine = DetectionEngine()
        self.last_analysis = None
        
    def start(self, source=0):
        """Start video processing with stable pipeline"""
        try:
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                return False
                
            # Configure camera for stability
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.display_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.display_height)
            self.cap.set(cv2.CAP_PROP_FPS, config.target_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to prevent lag
            
            self.running = True
            
            # Start threads
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.process_thread = threading.Thread(target=self._process_frames, daemon=True)
            
            self.capture_thread.start()
            self.process_thread.start()
            
            return True
            
        except Exception as e:
            st.error(f"Failed to start video: {e}")
            return False
    
    def stop(self):
        """Stop video processing"""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        if self.process_thread:
            self.process_thread.join(timeout=1.0)
            
        if self.cap:
            self.cap.release()
            
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
                
        while not self.display_queue.empty():
            try:
                self.display_queue.get_nowait()
            except queue.Empty:
                break
    
    def _capture_frames(self):
        """Capture frames in separate thread"""
        while self.running and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                # Add to queue if not full
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                    
                time.sleep(1.0 / config.target_fps)  # Control capture rate
                
            except Exception as e:
                print(f"Capture error: {e}")
                continue
    
    def _process_frames(self):
        """Process frames in separate thread"""
        while self.running:
            try:
                # Get frame from capture queue
                frame = self.frame_queue.get(timeout=0.1)
                self.frame_count += 1
                
                # Process every nth frame to maintain performance
                if self.frame_count % config.frame_skip == 0:
                    processed_frame = self._process_single_frame(frame)
                else:
                    # Use last processed frame for display consistency
                    processed_frame = frame.copy()
                
                # Add to display queue
                if not self.display_queue.full():
                    self.display_queue.put(processed_frame)
                else:
                    # Remove old frame and add new one
                    try:
                        self.display_queue.get_nowait()
                        self.display_queue.put(processed_frame)
                    except queue.Empty:
                        pass
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
                continue
    
    def _process_single_frame(self, frame):
        """Process single frame with detection logic"""
        try:
            # Run detection analysis
            analysis = self.detection_engine.analyze_frame(frame, config.confidence_threshold)
            self.last_analysis = analysis
            
            # Draw detections on frame
            annotated_frame = self.detection_engine.draw_detections(frame, analysis)
            
            return annotated_frame
            
        except Exception as e:
            print(f"Detection processing error: {e}")
            return frame
    
    def get_display_frame(self):
        """Get frame for display - non-blocking"""
        try:
            frame = self.display_queue.get_nowait()
            self.last_display_frame = frame
            return frame
        except queue.Empty:
            # Return last frame to prevent flickering
            return self.last_display_frame
    
    def get_stats(self):
        """Get processing statistics"""
        stats = {
            'frame_count': self.frame_count,
            'capture_queue_size': self.frame_queue.qsize(),
            'display_queue_size': self.display_queue.qsize(),
            'running': self.running
        }
        
        # Add detection stats if available
        if self.last_analysis:
            stats.update({
                'objects_detected': len(self.last_analysis['objects']),
                'faces_detected': len(self.last_analysis['faces']),
                'alerts_active': len(self.last_analysis['alerts']),
                'processing_time_ms': self.last_analysis['processing_time_ms']
            })
        
        return stats
    
    def get_latest_analysis(self):
        """Get latest detection analysis"""
        return self.last_analysis

# Initialize Streamlit
st.set_page_config(
    page_title="Enhanced Surveillance AI",
    layout="wide",
    page_icon="🛡️"
)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    padding: 20px;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}
.metric-box {
    background: #f0f2f6;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
.alert-high {
    background: #ff4444;
    color: white;
    padding: 15px;
    border-radius: 8px;
    font-weight: bold;
    text-align: center;
}
.alert-medium {
    background: #ffaa00;
    color: white;
    padding: 15px;
    border-radius: 8px;
    font-weight: bold;
    text-align: center;
}
.status-good {
    color: #00aa00;
    font-weight: bold;
}
.status-warning {
    color: #ffaa00;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🛡️ Enhanced Surveillance AI</h1>
    <p>Stable Video Processing • Real-time Detection • Smart Alerts</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'processor' not in st.session_state:
    st.session_state.processor = None
if 'running' not in st.session_state:
    st.session_state.running = False
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'last_sound_time' not in st.session_state:
    st.session_state.last_sound_time = 0.0
if 'police_timer' not in st.session_state:
    st.session_state.police_timer = None  # {'start': ts, 'duration': seconds}

# Sidebar controls
st.sidebar.header("🎮 Control Panel")

# Video source selection
source_type = st.sidebar.selectbox(
    "Video Source",
    ["Webcam", "Demo Video", "Upload File"]
)

# Settings
st.sidebar.header("⚙️ Settings")
config.confidence_threshold = st.sidebar.slider(
    "Detection Confidence", 0.1, 0.9, 0.5, 0.1
)
config.frame_skip = st.sidebar.slider(
    "Frame Skip (Performance)", 1, 5, 2
)

# Control buttons
col1, col2 = st.sidebar.columns(2)
start_btn = col1.button("▶️ Start", type="primary")
stop_btn = col2.button("⏹️ Stop", type="secondary")

# Emergency stop
emergency_btn = st.sidebar.button("🚨 Emergency Stop", type="secondary")

# Main layout
col_video, col_info = st.columns([2, 1])

with col_video:
    st.subheader("📹 Live Feed")
    video_placeholder = st.empty()

with col_info:
    st.subheader("📊 System Status")
    status_container = st.container()
    
    st.subheader("🚨 Alerts")
    alert_container = st.container()
    st.subheader("📝 Alert History")
    history_container = st.container()
    
    st.subheader("📈 Statistics")
    stats_container = st.container()

# Utility: simple beep tone as WAV bytes
def _make_beep(duration_ms=250, freq=880, volume=0.5, sample_rate=16000):
    t = np.linspace(0, duration_ms/1000.0, int(sample_rate*duration_ms/1000.0), False)
    tone = (np.sin(2*np.pi*freq*t) * volume).astype(np.float32)
    # Convert to 16-bit PCM
    audio = (tone * 32767).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)
    return buf.getvalue()

# Handle control buttons
if start_btn and not st.session_state.running:
    st.session_state.processor = StableVideoProcessor()
    
    # Determine video source
    source = 0  # Default webcam
    if source_type == "Demo Video":
        # Create a simple demo video if it doesn't exist
        demo_path = "demo_video.mp4"
        if not os.path.exists(demo_path):
            # Create simple demo video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(demo_path, fourcc, 20.0, (640, 480))
            for i in range(200):  # 10 seconds
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                # Add some moving objects
                cv2.rectangle(frame, (100+i, 100), (200+i, 200), (0, 255, 0), 2)
                cv2.putText(frame, f"Demo Frame {i}", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                out.write(frame)
            out.release()
        source = demo_path
    
    if st.session_state.processor.start(source):
        st.session_state.running = True
        st.success("✅ Surveillance started successfully!")
    else:
        st.error("❌ Failed to start surveillance system")
        st.session_state.processor = None

if stop_btn or emergency_btn:
    if st.session_state.processor:
        st.session_state.processor.stop()
        st.session_state.processor = None
    st.session_state.running = False
    st.info("⏹️ Surveillance stopped")

# Main processing loop
if st.session_state.running and st.session_state.processor:
    # Get and display frame
    frame = st.session_state.processor.get_display_frame()
    
    if frame is not None:
        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
    
    # Update status
    stats = st.session_state.processor.get_stats()
    analysis = st.session_state.processor.get_latest_analysis()
    
    with status_container:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <strong>Status:</strong> <span class="status-good">RUNNING</span><br>
                <strong>Frames:</strong> {stats['frame_count']}<br>
                <strong>Source:</strong> {source_type}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            processing_time = stats.get('processing_time_ms', 0)
            performance_status = "GOOD" if processing_time < 100 else "SLOW"
            performance_class = "status-good" if processing_time < 100 else "status-warning"
            
            st.markdown(f"""
            <div class="metric-box">
                <strong>Objects:</strong> {stats.get('objects_detected', 0)}<br>
                <strong>Faces:</strong> {stats.get('faces_detected', 0)}<br>
                <strong>Performance:</strong> <span class="{performance_class}">{performance_status}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with alert_container:
        # Display current alerts and manage sound + police timer
        if analysis and analysis['alerts']:
            now_ts = time.time()
            play_sound = any(a['severity'] in ['high', 'critical'] for a in analysis['alerts'])
            if play_sound and (now_ts - st.session_state.last_sound_time) > 1.5:
                st.audio(_make_beep(200, 1000, 0.6), format='audio/wav')
                st.session_state.last_sound_time = now_ts

            # Start or refresh police response timer on critical alert
            if any(a['severity'] == 'critical' for a in analysis['alerts']):
                st.session_state.police_timer = {'start': now_ts, 'duration': 30}

            # Show active alerts
            for alert in analysis['alerts'][-5:]:
                alert_class = f"alert-{alert['severity']}" if alert['severity'] in ['high', 'medium'] else "alert-high"
                st.markdown(f"""
                <div class="{alert_class}">🚨 {alert['message']} ({alert['confidence']:.2f})</div>
                """, unsafe_allow_html=True)

        else:
            st.success("✅ No alerts - System monitoring")

        # Police response simulation panel
        if st.session_state.police_timer:
            elapsed = time.time() - st.session_state.police_timer['start']
            remaining = int(max(0, st.session_state.police_timer['duration'] - elapsed))
            if remaining > 0:
                st.info(f"🚓 Police alerted. ETA: {remaining}s")
            else:
                st.success("🚓 Police arrived.")
                st.session_state.police_timer = None
    
    with stats_container:
        # Update detection count from analysis
        if analysis:
            total_detections = len(analysis['objects']) + len(analysis['faces'])
            st.session_state.detections.extend(analysis['objects'])
        else:
            total_detections = 0
        
        processing_fps = f"{1000/stats.get('processing_time_ms', 100):.1f}" if stats.get('processing_time_ms', 0) > 0 else "N/A"
        
        st.metric("Objects Detected", stats.get('objects_detected', 0))
        st.metric("Processing Speed", f"{processing_fps} FPS")
        st.metric("Active Alerts", stats.get('alerts_active', 0))
        if analysis and 'threat_score' in analysis:
            st.metric("Threat Score", int(analysis['threat_score']))

    with history_container:
        if analysis and analysis['alerts']:
            # Append unique messages with timestamp
            ts = datetime.now().strftime('%H:%M:%S')
            for a in analysis['alerts']:
                msg = f"[{ts}] {a['message']} ({a['confidence']:.2f})"
                if not st.session_state.alert_history or st.session_state.alert_history[-1] != msg:
                    st.session_state.alert_history.append(msg)
            st.session_state.alert_history = st.session_state.alert_history[-20:]

        if st.session_state.alert_history:
            for m in reversed(st.session_state.alert_history[-10:]):
                st.write(f"- {m}")
        else:
            st.caption("No alerts yet.")
    
    # Auto-refresh the page to update video
    time.sleep(0.05)  # Small delay to prevent excessive CPU usage
    st.rerun()

else:
    # Idle state
    with video_placeholder:
        st.info("🎯 Ready to start surveillance. Select source and click 'Start'")
    
    with status_container:
        st.markdown("""
        <div class="metric-box">
            <strong>Status:</strong> <span class="status-warning">IDLE</span><br>
            <strong>System:</strong> Ready<br>
            <strong>Camera:</strong> Not connected
        </div>
        """, unsafe_allow_html=True)

    # Sidebar legend for emotions
    with st.sidebar:
        st.markdown("### 🎨 Emotion Legend")
        st.write("- Angry: RED")
        st.write("- Fear: ORANGE")
        st.write("- Happy: GREEN")
        st.write("- Sad: BLUE")
        st.write("- Surprise: YELLOW")
        st.write("- Disgust: PURPLE")
        st.write("- Neutral: GRAY")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 10px;">
    🛡️ <strong>Enhanced Surveillance AI v1.0</strong> | 
    Stable Video Pipeline | No Flickering | Ready for Enhancement
</div>
""", unsafe_allow_html=True)

# Cleanup on app termination
if not st.session_state.running and st.session_state.processor:
    st.session_state.processor.stop()
    st.session_state.processor = None
    gc.collect()  # Clean up memory
