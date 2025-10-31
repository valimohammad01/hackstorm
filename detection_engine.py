"""
Detection Engine for Enhanced Surveillance AI
Handles YOLO weapon detection and emotion detection with optional DeepFace
"""

import cv2
import numpy as np
try:
    import torch
    _HAS_TORCH = True
except Exception:
    torch = None
    _HAS_TORCH = False
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    YOLO = None
    _HAS_YOLO = False
import time

try:
    from deepface import DeepFace  # Optional, heavy dependency
    _HAS_DEEPFACE = True
except Exception:
    DeepFace = None
    _HAS_DEEPFACE = False


class DetectionEngine:
    def __init__(self):
        self.yolo_model = None
        self.face_cascade = None
        self.weapon_classes = {'knife', 'gun', 'scissors', 'bottle', 'baseball bat'}
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.device = 'cuda' if (_HAS_TORCH and torch.cuda.is_available()) else 'cpu'
        self.use_half = self.device == 'cuda'

        # Emotion throttling
        self._last_emotion_time = 0.0
        self._emotion_interval_s = 0.5  # analyze emotions at most twice per second

        self.load_models()

    def load_models(self):
        """Load YOLO and emotion detection models"""
        try:
            # Load YOLOv8n for object detection if available
            if _HAS_YOLO:
                self.yolo_model = YOLO("yolov8n.pt")
                if self.device == 'cuda':
                    self.yolo_model.to('cuda')
                print("✅ YOLO model loaded successfully")
            else:
                self.yolo_model = None
                print("ℹ️ Ultralytics/YOLO not available - running faces-only mode")

            # Enable OpenCV optimizations
            try:
                cv2.setUseOptimized(True)
                if _HAS_TORCH and hasattr(torch, 'get_num_threads'):
                    cv2.setNumThreads(max(1, torch.get_num_threads()))
            except Exception:
                pass

            # Load face cascade for face detection (fast CPU fallback)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print("✅ Face detection loaded successfully")

            # DeepFace availability
            if _HAS_DEEPFACE:
                print("✅ DeepFace available for emotion analysis (throttled)")
            else:
                print("ℹ️ DeepFace not installed - using basic emotion heuristic")

        except Exception as e:
            print(f"❌ Model loading error: {e}")

    def detect_objects(self, frame, confidence_threshold=0.5):
        """Detect objects using YOLO"""
        if self.yolo_model is None:
            return []

        try:
            # Run YOLO inference
            results = self.yolo_model(frame, conf=confidence_threshold, verbose=False)[0]

            detections = []
            if results and results.boxes is not None:
                for box in results.boxes:
                    # Extract box information
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.yolo_model.names[class_id]

                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class': class_name,
                        'is_weapon': class_name in self.weapon_classes,
                        'is_person': class_name == 'person'
                    }
                    detections.append(detection)

            return detections

        except Exception as e:
            print(f"Object detection error: {e}")
            return []

    def _deepface_emotion(self, face_bgr):
        """Run DeepFace emotion if available, else fallback"""
        if not _HAS_DEEPFACE or face_bgr is None or face_bgr.size == 0:
            return self._simple_emotion_detection(face_bgr), 0.6
        try:
            # Convert BGR to RGB for DeepFace
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            analysis = DeepFace.analyze(face_rgb, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
            if isinstance(analysis, list):
                analysis = analysis[0]
            emotions = analysis.get('emotion', {})
            if emotions:
                # Pick max emotion
                emotion = max(emotions, key=emotions.get).lower()
                conf = float(emotions.get(emotion.capitalize(), emotions.get(emotion, 0.0))) / 100.0 if emotions else 0.6
                # Normalize label
                emotion = emotion if emotion in self.emotion_labels else 'neutral'
                return emotion, conf
            return 'neutral', 0.5
        except Exception:
            return self._simple_emotion_detection(face_bgr), 0.6

    def detect_faces_and_emotions(self, frame, person_rois=None):
        """Detect faces and estimate emotions. If person_rois provided, only search within them."""
        if self.face_cascade is None:
            return []

        face_detections = []
        regions = person_rois if person_rois else [(0, 0, frame.shape[1], frame.shape[0])]

        # Emotion throttling
        now = time.time()
        allow_emotion = (now - self._last_emotion_time) >= self._emotion_interval_s
        if allow_emotion:
            self._last_emotion_time = now

        try:
            for (rx1, ry1, rx2, ry2) in regions:
                rx1, ry1 = max(0, rx1), max(0, ry1)
                rx2, ry2 = min(frame.shape[1], rx2), min(frame.shape[0], ry2)
                roi = frame[ry1:ry2, rx1:rx2]

                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                for (x, y, w, h) in faces:
                    x1, y1, x2, y2 = rx1 + x, ry1 + y, rx1 + x + w, ry1 + y + h
                    face_roi = frame[y1:y2, x1:x2]

                    if allow_emotion:
                        emotion, emo_conf = self._deepface_emotion(face_roi)
                    else:
                        # Use fast heuristic when throttled
                        emotion = self._simple_emotion_detection(face_roi)
                        emo_conf = 0.6

                    face_detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'emotion': emotion,
                        'confidence': float(emo_conf)
                    })

            return face_detections

        except Exception as e:
            print(f"Face detection error: {e}")
            return []

    def _simple_emotion_detection(self, face_roi):
        """Simple fallback emotion detection when DeepFace is unavailable or throttled"""
        if face_roi is None or face_roi.size == 0:
            return 'neutral'
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray_face))
        contrast = float(np.std(gray_face))
        if brightness < 90 and contrast > 35:
            return 'angry'
        if brightness > 160:
            return 'happy'
        if contrast < 18:
            return 'sad'
        return 'neutral'

    def analyze_frame(self, frame, confidence_threshold=0.5):
        """Complete frame analysis with objects, faces, emotions, and alerts"""
        start_time = time.time()

        # Detect objects
        objects = self.detect_objects(frame, confidence_threshold)

        # Person ROIs for focused face search
        person_rois = [tuple(obj['bbox']) for obj in objects if obj.get('is_person')]
        faces = self.detect_faces_and_emotions(frame, person_rois if person_rois else None)

        processing_time = (time.time() - start_time) * 1000

        # Alerts and threat scoring
        alerts, threat_score = self._generate_alerts(objects, faces)

        return {
            'objects': objects,
            'faces': faces,
            'alerts': alerts,
            'processing_time_ms': processing_time,
            'threat_score': threat_score
        }

    def _generate_alerts(self, objects, faces):
        """Generate alerts based on detections and compute threat score"""
        alerts = []

        weapons = [obj for obj in objects if obj['is_weapon']]
        persons = [obj for obj in objects if obj['is_person']]

        # Basic weapon alerts
        for weapon in weapons:
            alerts.append({
                'type': 'weapon',
                'severity': 'high',
                'message': f"WEAPON DETECTED: {weapon['class'].upper()}",
                'confidence': weapon['confidence']
            })

        # Emotion alerts
        for face in faces:
            if face['emotion'] in ['angry', 'fear']:
                alerts.append({
                    'type': 'emotion',
                    'severity': 'medium',
                    'message': f"SUSPICIOUS EMOTION: {face['emotion'].upper()}",
                    'confidence': face['confidence']
                })

        # Proximity: weapon near person + aggressive emotion
        def _centroid(b):
            x1, y1, x2, y2 = b
            return ((x1 + x2) // 2, (y1 + y2) // 2)

        proximity_px = 100
        high_threat = False
        for w in weapons:
            wx, wy = _centroid(w['bbox'])
            for p in persons:
                px, py = _centroid(p['bbox'])
                dist = ((wx - px) ** 2 + (wy - py) ** 2) ** 0.5
                if dist < proximity_px:
                    # Check any face inside person bbox with suspicious emotion
                    x1, y1, x2, y2 = p['bbox']
                    emo_suspicious = any(
                        f['emotion'] in ['angry', 'fear'] and
                        f['bbox'][0] >= x1 and f['bbox'][1] >= y1 and f['bbox'][2] <= x2 and f['bbox'][3] <= y2
                        for f in faces
                    )
                    severity = 'critical' if emo_suspicious else 'high'
                    alerts.append({
                        'type': 'threat',
                        'severity': severity,
                        'message': 'CRITICAL: WEAPON NEAR PERSON' if emo_suspicious else 'WARNING: WEAPON NEAR PERSON',
                        'confidence': 0.9 if emo_suspicious else 0.75
                    })
                    if emo_suspicious:
                        high_threat = True

        # Threat score
        threat_score = 0
        threat_score += min(100, int(sum(w['confidence'] for w in weapons) * 40))
        threat_score += min(40, 20 * sum(1 for f in faces if f['emotion'] in ['angry', 'fear']))
        if high_threat:
            threat_score = max(threat_score, 80)

        return alerts, min(100, threat_score)

    def draw_detections(self, frame, analysis_result):
        """Draw detection results on frame"""
        annotated_frame = frame.copy()

        # Draw object detections
        for obj in analysis_result['objects']:
            x1, y1, x2, y2 = obj['bbox']

            # Color based on object type
            if obj['is_weapon']:
                color = (0, 0, 255)  # Red for weapons
                thickness = 3
            elif obj['is_person']:
                color = (0, 255, 0)  # Green for persons
                thickness = 2
            else:
                color = (255, 255, 0)  # Yellow for other objects
                thickness = 2

            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

            # Draw label
            label = f"{obj['class']} {obj['confidence']:.2f}"
            cv2.putText(annotated_frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw face detections
        for face in analysis_result['faces']:
            x1, y1, x2, y2 = face['bbox']

            # Color based on emotion
            emotion_colors = {
                'angry': (0, 0, 255),    # Red
                'fear': (0, 100, 255),   # Orange
                'sad': (255, 0, 0),      # Blue
                'happy': (0, 255, 0),    # Green
                'surprise': (255, 255, 0), # Yellow
                'disgust': (128, 0, 128),  # Purple
                'neutral': (128, 128, 128) # Gray
            }

            color = emotion_colors.get(face['emotion'], (128, 128, 128))

            # Draw face box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Draw emotion label
            emotion_label = f"{face['emotion']} {face['confidence']:.2f}"
            cv2.putText(annotated_frame, emotion_label, (x1, y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw alerts on frame
        alert_y = 30
        for alert in analysis_result['alerts']:
            if alert['severity'] == 'critical':
                color = (0, 0, 255)  # Red
            elif alert['severity'] == 'high':
                color = (0, 100, 255)  # Orange
            else:
                color = (0, 255, 255)  # Yellow

            cv2.putText(annotated_frame, alert['message'], (10, alert_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            alert_y += 30

        # Threat score display
        if 'threat_score' in analysis_result:
            cv2.putText(annotated_frame, f"Threat Score: {int(analysis_result['threat_score'])}", (10, alert_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated_frame
