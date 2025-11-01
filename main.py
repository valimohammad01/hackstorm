# suspicious_ai.py
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from ultralytics import YOLO

# -----------------------------
# CONFIGURATION
# -----------------------------
PROJECT_DIR = "suspicious_data/data"
DATA_YAML = os.path.join(PROJECT_DIR, "data.yaml")
MODEL_SAVE_DIR = os.path.join(PROJECT_DIR, "runs")
SUSPICIOUS_CLASSES = {
    "gun", "knife", "scissor", "scissors", "fire", "bat", "rod", 
    "Pointed_knife", "Dual-edge-sharp-knife", "dual-edge_knife", 
    "recurve", "Sharp_cutter"
}

# MAP ALIASES TO STANDARD NAMES FOR BETTER ORGANIZATION
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

PROXIMITY_PIXELS = 80  # pixels between suspicious object and person for alert

# -----------------------------
# Streamlit page
# -----------------------------
st.set_page_config(page_title="AI Suspicious Surveillance", layout="wide", page_icon="🛡️")
st.markdown(
    """
    <div style="background-color:#f5f5f5;padding:10px;border-radius:8px;">
        <h1 style="color:#1f2937;">🛡️ AI Suspicious Surveillance</h1>
        <p style="color:#4b5563;">Real-time detection of suspicious items and behavior</p>
    </div>
    """, unsafe_allow_html=True
)
st.sidebar.header("Input & Model")

# -----------------------------
# Sidebar Inputs
# -----------------------------
source_type = st.sidebar.selectbox("Video source", ["Webcam", "Upload video", "Sample video"])
uploaded_file = None
if source_type == "Upload video":
    uploaded_file = st.sidebar.file_uploader("Upload video (mp4/avi)", type=["mp4", "avi"])

st.sidebar.markdown("---")
st.sidebar.header("Inference Settings")
inference_size = st.sidebar.selectbox("Inference size (px)", [320, 384, 480, 640], index=0)
process_every_n = st.sidebar.slider("Process every N frames", 1, 6, 2)
confidence_thr = st.sidebar.slider("Confidence threshold", 0.2, 0.9, 0.35, 0.05)

st.sidebar.markdown("---")
st.sidebar.header("Behavioral Settings")
loiter_frames_thresh = st.sidebar.slider("Loiter frames threshold", 30, 300, 90, 10)
loiter_movement_px = st.sidebar.slider("Loiter movement radius (px)", 10, 80, 30, 5)

st.sidebar.markdown("---")
start_button = st.sidebar.button("Start Detection")
stop_button = st.sidebar.button("Stop / Reset")
train_button = st.sidebar.button("Train Model")

# -----------------------------
# Load or train YOLO model
# -----------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("yolov8n.pt")

if train_button:
    st.info("Training model on suspicious items dataset...")
    model = YOLO("yolov8n.pt")
    model.train(
        data=DATA_YAML,
        epochs=50,
        batch=8,
        imgsz=640,
        save=True,
        project=MODEL_SAVE_DIR,
        name="suspicious_model",
        exist_ok=True
    )
    st.success(f"Training finished! Model saved in {MODEL_SAVE_DIR}/suspicious_model/weights")

else:
    with st.spinner("Loading YOLO model..."):
        model = load_yolo_model()
    st.sidebar.success("Model loaded (yolov8n.pt)")

# -----------------------------
# Session state
# -----------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "detections" not in st.session_state:
    st.session_state.detections = []
if "tracker" not in st.session_state:
    st.session_state.tracker = {}
if "next_id" not in st.session_state:
    st.session_state.next_id = 0
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "last_alert_text" not in st.session_state:
    st.session_state.last_alert_text = ""
if "last_alert_frame" not in st.session_state:
    st.session_state.last_alert_frame = 0
if "alert_cooldown_frames" not in st.session_state:
    st.session_state.alert_cooldown_frames = 20

# Start / Stop
if start_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False

# -----------------------------
# Layout
# -----------------------------
col_main, col_right = st.columns((2.2, 1))
frame_placeholder = col_main.empty()
alerts_box = col_right.empty()
metrics_row = col_right.container()
download_box = col_right.empty()

# Metrics
with metrics_row:
    c1, c2, c3 = st.columns(3)
    fps_metric = c1.metric("FPS", "0")
    inf_metric = c2.metric("Inference (ms)", "0")
    det_metric = c3.metric("Total Detections", "0")

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
# Video Capture
# -----------------------------
cap=None
if source_type=="Webcam":
    cap=cv2.VideoCapture(0)
elif source_type=="Upload video" and uploaded_file is not None:
    tmp_path="uploaded_temp.mp4"
    with open(tmp_path,"wb") as f: f.write(uploaded_file.read())
    cap=cv2.VideoCapture(tmp_path)
else:
    sample_path="sample_short.mp4"
    if os.path.exists(sample_path):
        cap=cv2.VideoCapture(sample_path)
    else:
        cap=cv2.VideoCapture(0)

if cap is None or not cap.isOpened():
    frame_placeholder.info("Select a valid video source and press Start.")
else:
    prev_time=time.perf_counter()
    last_detections=[]
    while st.session_state.running:
        st.session_state.frame_idx+=1
        ok, frame=cap.read()
        if not ok: st.session_state.running=False; break
        frame_h, frame_w = frame.shape[:2]

        # frame skipping
        if (st.session_state.frame_idx % process_every_n)!=0 and last_detections:
            display_frame=frame.copy()
            for info in last_detections:
                tid,bbox,label,conf=info["id"],info["bbox"],info["label"],info["conf"]
                x1,y1,x2,y2=bbox
                is_suspicious = label in SUSPICIOUS_CLASSES or label in SUSPICIOUS_ALIASES
                color=(0,0,255) if is_suspicious else (50,205,50)
                cv2.rectangle(display_frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(display_frame,f"{label} ID:{tid} {conf:.2f}",(x1,max(10,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)

            # Fix display size (e.g., 640x480)
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", output_format="JPEG", width=640)

            time.sleep(0.01)
            continue

        # resize
        long_side=max(frame_w,frame_h)
        scale=1.0
        if long_side>inference_size:
            scale=inference_size/long_side
            frame_infer=cv2.resize(frame,(int(frame_w*scale),int(frame_h*scale)))
        else: frame_infer=frame.copy()

        # detection
        t0=time.perf_counter()
        results=model(frame_infer,conf=confidence_thr,imgsz=inference_size)[0]
        t1=time.perf_counter()
        inf_ms=(t1-t0)*1000

        dets=[]
        for det in results.boxes:
            cls_id=int(det.cls[0])
            conf=float(det.conf[0])
            if conf<confidence_thr: continue
            x1,y1,x2,y2=map(int,det.xyxy[0])
            if scale<1.0: x1=int(x1/scale);y1=int(y1/scale);x2=int(x2/scale);y2=int(y2/scale)
            label=model.names.get(cls_id,str(cls_id))
            cx,cy=(x1+x2)//2,(y1+y2)//2
            dets.append(([x1,y1,x2,y2],label,conf,(cx,cy)))

        assigned=assign_ids(dets,st.session_state.tracker,0.2)

        display_frame=frame.copy()
        alert_messages=[]
        persons_in_frame=[]
        suspicious_items=[]
        last_detections=[]
        for tid,bbox,label,conf,centroid in assigned:
            x1,y1,x2,y2=bbox
            cx,cy=centroid
            
            # Check if this is a suspicious item (direct match or alias)
            is_suspicious = label in SUSPICIOUS_CLASSES or label in SUSPICIOUS_ALIASES
            actual_label = SUSPICIOUS_ALIASES.get(label, label) if label in SUSPICIOUS_ALIASES else label
            
            color=(50,205,50) if label!="person" else (50,205,50)
            if is_suspicious: 
                color=(0,0,255)
                suspicious_items.append({"id":tid,"label":actual_label,"bbox":bbox,"centroid":centroid})
            if label=="person": persons_in_frame.append({"id":tid,"centroid":centroid,"bbox":bbox})
            cv2.rectangle(display_frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(display_frame,f"{label} ID:{tid} {conf:.2f}",(x1,max(10,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,2)
            st.session_state.detections.append({"frame":st.session_state.frame_idx,"timestamp":datetime.now().isoformat(),"id":tid,"label":label,"conf":conf,"bbox":bbox})
            last_detections.append({"id":tid,"label":label,"conf":conf,"bbox":bbox})

        # heuristic alerts
        for item in suspicious_items:
            for p in persons_in_frame:
                dx=item["centroid"][0]-p["centroid"][0]
                dy=item["centroid"][1]-p["centroid"][1]
                if (dx*dx+dy*dy)**0.5<PROXIMITY_PIXELS:
                    alert_messages.append(f"ALERT: {item['label']} near Person ID:{p['id']}")

        for p in persons_in_frame:
            info=st.session_state.tracker.get(p["id"],{})
            centroids=info.get("centroids",[])
            if len(centroids)>=loiter_frames_thresh:
                tail=centroids[-loiter_frames_thresh:]
                xs=[c[0] for c in tail]; ys=[c[1] for c in tail]
                if max(max(xs)-min(xs), max(ys)-min(ys))<=loiter_movement_px:
                    alert_messages.append(f"LOITERING ALERT: Person ID:{p['id']}")

        # update alerts
        alert_text="\n".join(alert_messages) if alert_messages else "No critical alerts"
        if alert_text!=st.session_state.last_alert_text or st.session_state.frame_idx-st.session_state.last_alert_frame>=st.session_state.alert_cooldown_frames:
            if alert_messages:
                now=datetime.now().strftime("%H:%M:%S")
                alerts_box.markdown(f"**Last update: {now} | Alerts: {len(alert_messages)}**")
                for a in alert_messages: alerts_box.warning(a)
            else: alerts_box.info("No critical alerts")
            st.session_state.last_alert_text=alert_text
            st.session_state.last_alert_frame=st.session_state.frame_idx

        # display frame
        frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        # metrics
        fps=1.0/max(1e-6,time.perf_counter()-prev_time)
        prev_time=time.perf_counter()
        fps_metric.metric("FPS",f"{fps:.1f}")
        inf_metric.metric("Inference (ms)",f"{inf_ms:.1f}")
        det_metric.metric("Total Detections",str(len(st.session_state.detections)))

        time.sleep(0.01)

    cap.release()

# -----------------------------
# Download CSV
# -----------------------------
if st.session_state.detections:
    df=pd.DataFrame(st.session_state.detections)
    csv_bytes=df.to_csv(index=False).encode("utf-8")
    download_box.download_button("Download detections CSV", data=csv_bytes, file_name="detections_report.csv", mime="text/csv")
else:
    download_box.info("No detections captured.")

st.caption("Prototype — trained on suspicious items, light UI, interactive alerts.")
