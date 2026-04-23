import streamlit as st
import cv2
import numpy as np
import math
import pandas as pd
import tempfile
import os
import time
import shutil
from datetime import datetime
from pathlib import Path
from pymongo import MongoClient
from collections import defaultdict
import torch
import gc
import subprocess


# Memory optimization for Torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
torch.set_grad_enabled(False)


# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartCity — Vehicle Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0a0c10;
    --surface:  #111318;
    --border:   #1e2330;
    --accent:   #00e5ff;
    --green:    #00ff88;
    --red:      #ff3b5c;
    --amber:    #ffb300;
    --text:     #e8eaf0;
    --muted:    #5a6070;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg);
    color: var(--text);
}

/* Hide default streamlit header */
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1400px; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #111827 50%, #0a0f1a 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(0,229,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: -0.5px;
    margin: 0 0 0.4rem;
}
.hero-sub {
    font-size: 0.95rem;
    color: var(--muted);
    margin: 0;
    letter-spacing: 0.5px;
}

/* Metric cards */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; flex-wrap: wrap; }
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    flex: 1; min-width: 140px;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
}
.metric-card.accent::after  { background: var(--accent); }
.metric-card.green::after   { background: var(--green); }
.metric-card.red::after     { background: var(--red); }
.metric-card.amber::after   { background: var(--amber); }

.metric-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 0.3rem;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    line-height: 1;
}
.metric-card.accent .metric-value { color: var(--accent); }
.metric-card.green  .metric-value { color: var(--green); }
.metric-card.red    .metric-value { color: var(--red); }
.metric-card.amber  .metric-value { color: var(--amber); }

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.6rem;
    margin: 1.8rem 0 1rem;
}

/* Status badge */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    font-family: 'Space Mono', monospace;
}
.badge-green { background: rgba(0,255,136,0.12); color: var(--green); border: 1px solid rgba(0,255,136,0.3); }
.badge-red   { background: rgba(255,59,92,0.12);  color: var(--red);   border: 1px solid rgba(255,59,92,0.3); }
.badge-amber { background: rgba(255,179,0,0.12);  color: var(--amber); border: 1px solid rgba(255,179,0,0.3); }

/* Upload zone */
.upload-zone {
    background: var(--surface);
    border: 2px dashed var(--border);
    border-radius: 12px;
    padding: 2.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: var(--accent); }

/* Log table */
.stDataFrame { background: var(--surface) !important; }
thead tr th { background: #161b24 !important; color: var(--accent) !important; font-family: 'Space Mono', monospace !important; font-size: 0.75rem !important; }
tbody tr td { background: var(--surface) !important; color: var(--text) !important; font-size: 0.82rem !important; }
tbody tr:hover td { background: #161b24 !important; }

/* Progress */
.stProgress > div > div { background: var(--accent) !important; }

/* Buttons */
.stButton > button {
    background: var(--accent);
    color: #000;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    padding: 0.55rem 1.4rem;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

/* Info boxes */
.info-box {
    background: rgba(0,229,255,0.05);
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.85rem;
    color: var(--text);
}
.info-box b { color: var(--accent); }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── MongoDB ───────────────────────────────────────────────────────────────────
@st.cache_resource
def get_mongo():
    # Try auto-detect from env or secrets
    uri = os.getenv("MONGO_URI")
    if not uri:
        try: uri = st.secrets.get("MONGO_URI")
        except: uri = None
    
    if not uri: return None, None
    try:
        # Use dnspython for srv links
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client["seatbeltviolation"]
        return db["sessions"], db["vehicles"]
    except Exception:
        return None, None

# Initialize collections lazily
sessions_col, vehicles_col = None, None
def get_collections():
    global sessions_col, vehicles_col
    if sessions_col is None:
        sessions_col, vehicles_col = get_mongo()
    return sessions_col, vehicles_col

def save_session_data(session_doc):
    s_col, _ = get_collections()
    if s_col is not None:
        try: s_col.insert_one(session_doc)
        except: pass

def save_vehicle_data(vehicle_doc):
    _, v_col = get_collections()
    if v_col is not None:
        try:
            plate = vehicle_doc.get('license_plate')
            # If we have a valid plate, use it as the unique key for upsert
            if plate and plate != 'NOT DETECTED':
                query = {"license_plate": plate}
                # Update with latest info, but keep original session if needed
                update = {"$set": vehicle_doc}
                v_col.update_one(query, update, upsert=True)
            else:
                # Fallback to session+id to avoid duplicates in same session
                query = {
                    "session_id": vehicle_doc.get("session_id"),
                    "vehicle_id": vehicle_doc.get("vehicle_id")
                }
                v_col.update_one(query, {"$set": vehicle_doc}, upsert=True)
        except:
            pass

def fetch_sessions_data():
    s_col, _ = get_collections()
    if s_col is None: return []
    try: return list(s_col.find().sort("start_time", -1))
    except: return []

def fetch_vehicles_for_session(session_id):
    _, v_col = get_collections()
    if v_col is None: return []
    try: return list(v_col.find({"session_id": session_id}))
    except: return []

# ─── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_yolo_models():
    from ultralytics import YOLO
    v_model = YOLO('models/vehicle-detection.pt')
    p_model = YOLO('models/license_plate.pt')
    s_model = YOLO('models/seatbelt_detection.pt')
    return v_model, p_model, s_model

@st.cache_resource
def load_ocr_reader():
    import easyocr
    return easyocr.Reader(['en'], gpu=torch.cuda.is_available())

def load_models():
    v, p, s = load_yolo_models()
    ocr = load_ocr_reader()
    return v, p, s, ocr

# ─── Motion Analysis ───────────────────────────────────────────────────────────
def analyze_video_motion(video_path, sample_frames=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps   = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    flow_x_list, flow_y_list = [], []
    prev_gray = None
    step = max(1, total // sample_frames)

    for i in range(0, min(total - 1, sample_frames * step), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)
        if prev_gray is not None:
            # Resize for faster and more memory-efficient optical flow
            h, w = gray.shape
            if w > 640:
                scale = 640 / w
                gray_small = cv2.resize(gray, (0,0), fx=scale, fy=scale)
                prev_gray_small = cv2.resize(prev_gray, (0,0), fx=scale, fy=scale)
                flow = cv2.calcOpticalFlowFarneback(prev_gray_small, gray_small, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            else:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_mask = mag > 1.5
            if motion_mask.sum() > 200:
                flow_x_list.append(flow[..., 0][motion_mask].mean())
                flow_y_list.append(flow[..., 1][motion_mask].mean())
        prev_gray = gray
    cap.release()

    if not flow_x_list:
        return {'primary_direction': 'vertical', 'recommended_mode': 'HORIZONTAL',
                'flow_x': 0, 'flow_y': 0, 'angle_deg': 90, 'direction': '↑ UP',
                'width': width, 'height': height, 'fps': fps, 'total': total}

    avg_fx    = np.mean(flow_x_list)
    avg_fy    = np.mean(flow_y_list)
    h_ratio   = abs(avg_fx) / (abs(avg_fx) + abs(avg_fy) + 1e-6)
    angle_deg = math.degrees(math.atan2(abs(avg_fy), abs(avg_fx)))

    if h_ratio > 0.65:
        primary, rec_mode = 'horizontal', 'VERTICAL'
        direction = '→ RIGHT' if avg_fx > 0 else '← LEFT'
    elif h_ratio < 0.35:
        primary, rec_mode = 'vertical', 'HORIZONTAL'
        direction = '↓ DOWN' if avg_fy > 0 else '↑ UP'
    else:
        primary, rec_mode = 'diagonal', 'HORIZONTAL'
        direction = f'↘ DIAGONAL ({angle_deg:.0f}°)'

    return {'primary_direction': primary, 'recommended_mode': rec_mode,
            'flow_x': avg_fx, 'flow_y': avg_fy, 'h_ratio': h_ratio,
            'angle_deg': angle_deg, 'direction': direction,
            'width': width, 'height': height, 'fps': fps, 'total': total}


def build_counting_line(motion_info, ratio=0.9):
    W, H = motion_info['width'], motion_info['height']
    mode = motion_info['recommended_mode']
    if mode == 'VERTICAL':
        x = int(W * ratio)
        return {'mode': 'VERTICAL',  'pt1': (x, 0),  'pt2': (x, H),
                'coord': x, 'axis': 'x', 'label': f'x={x}'}
    else:
        y = int(H * ratio)
        return {'mode': 'HORIZONTAL','pt1': (0, y),  'pt2': (W, y),
                'coord': y, 'axis': 'y', 'label': f'y={y}'}

# ─── Tracker Classes ───────────────────────────────────────────────────────────
class TrackedVehicle:
    def __init__(self, vehicle_id, bbox, frame_num):
        self.vehicle_id     = vehicle_id
        self.bbox           = bbox
        self.last_seen      = frame_num
        self.first_seen     = frame_num
        self.center_history = [self._calc_center(bbox)]
        self.license_plate  = None
        self.plate_bbox     = None
        self.seatbelt_on    = None
        self.scanned        = False
        self.crossed_line   = False
        self.confidence     = 0.0

    def _calc_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    def update(self, bbox, frame_num):
        self.bbox      = bbox
        self.last_seen = frame_num
        self.center_history.append(self._calc_center(bbox))
        if len(self.center_history) > 30:
            self.center_history.pop(0)

    @property
    def center(self):
        return self._calc_center(self.bbox)

    def iou(self, other_bbox):
        x1 = max(self.bbox[0], other_bbox[0])
        y1 = max(self.bbox[1], other_bbox[1])
        x2 = min(self.bbox[2], other_bbox[2])
        y2 = min(self.bbox[3], other_bbox[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter == 0:
            return 0.0
        area_a = (self.bbox[2]-self.bbox[0]) * (self.bbox[3]-self.bbox[1])
        area_b = (other_bbox[2]-other_bbox[0]) * (other_bbox[3]-other_bbox[1])
        return inter / (area_a + area_b - inter + 1e-6)


class VehicleTracker:
    def __init__(self, iou_threshold=0.3, max_missing_frames=30):
        self.vehicles           = {}
        self.next_id            = 1
        self.iou_threshold      = iou_threshold
        self.max_missing_frames = max_missing_frames
        self.all_records        = []

    def update(self, detections, frame_num):
        matched_vehs = []
        for bbox, conf in detections:
            best_id, best_iou = None, self.iou_threshold
            for vid, veh in self.vehicles.items():
                if frame_num - veh.last_seen > self.max_missing_frames:
                    continue
                score = veh.iou(bbox)
                if score > best_iou:
                    best_iou, best_id = score, vid
            if best_id is not None:
                self.vehicles[best_id].update(bbox, frame_num)
                matched_vehs.append(self.vehicles[best_id])
            else:
                new_veh = TrackedVehicle(self.next_id, bbox, frame_num)
                new_veh.confidence = conf
                self.vehicles[self.next_id] = new_veh
                matched_vehs.append(new_veh)
                self.next_id += 1
        return matched_vehs

    def cleanup_old(self, frame_num):
        to_remove = [vid for vid, veh in self.vehicles.items()
                     if frame_num - veh.last_seen > self.max_missing_frames]
        for vid in to_remove:
            veh = self.vehicles[vid]
            self.all_records.append({
                'vehicle_id'   : veh.vehicle_id,
                'license_plate': veh.license_plate or 'NOT DETECTED',
                'seatbelt'     : 'YES' if veh.seatbelt_on else ('NO' if veh.seatbelt_on is False else 'NOT DETECTED'),
                'first_frame'  : veh.first_seen,
                'last_frame'   : veh.last_seen,
                'crossed_line' : veh.crossed_line,
                'timestamp'    : datetime.utcnow(),
            })
            del self.vehicles[vid]

    def get_all_records(self):
        current = []
        for veh in self.vehicles.values():
            current.append({
                'vehicle_id'   : veh.vehicle_id,
                'license_plate': veh.license_plate or 'NOT DETECTED',
                'seatbelt'     : 'YES' if veh.seatbelt_on else ('NO' if veh.seatbelt_on is False else 'NOT DETECTED'),
                'first_frame'  : veh.first_seen,
                'last_frame'   : veh.last_seen,
                'crossed_line' : veh.crossed_line,
                'timestamp'    : datetime.utcnow().isoformat(),
            })
        return self.all_records + current

    def get_dataframe(self):
        data = self.get_all_records()
        return pd.DataFrame(data) if data else pd.DataFrame()

# ─── Detection Helpers ─────────────────────────────────────────────────────────
def extract_license_plate_text(frame, plate_bbox, ocr_reader):
    x1, y1, x2, y2 = [int(v) for v in plate_bbox]
    pad = 3
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(frame.shape[1], x2 + pad); y2 = min(frame.shape[0], y2 + pad)
    plate_crop = frame[y1:y2, x1:x2]
    if plate_crop.size == 0:
        return None
    
    # Preprocessing
    gray    = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    results = ocr_reader.readtext(thresh, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ')
    if results:
        # Get the one with highest confidence
        text = max(results, key=lambda x: x[2])[1]
        return text.strip().upper() if text.strip() else None
    return None


def check_seatbelt(vehicle_crop, seatbelt_model):
    if vehicle_crop.size == 0: return None
    results = seatbelt_model(vehicle_crop, verbose=False)
    if not results or len(results[0].boxes) == 0: return None
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = seatbelt_model.names[cls_id].lower()
        if float(box.conf[0]) > 0.4:
            if cls_name == 'person_with_seatbelt': return True
            if cls_name == 'person_without_seatbelt': return False
    return None

def detect_license_plate_in_crop(vehicle_crop, licence_plate_model, ocr_reader):
    if vehicle_crop.size == 0: return None, None
    results = licence_plate_model(vehicle_crop, verbose=False)
    if not results or len(results[0].boxes) == 0: return None, None
    best_box = max(results[0].boxes, key=lambda x: float(x.conf[0]))
    px1, py1, px2, py2 = [int(v) for v in best_box.xyxy[0]]
    plate_text = extract_license_plate_text(vehicle_crop, (px1, py1, px2, py2), ocr_reader)
    return plate_text, (px1, py1, px2, py2)


# Replaced by detect_license_plate_in_crop


def crosses_line(vehicle_center, line_info, tolerance=22):
    if line_info['axis'] == 'x':
        return abs(vehicle_center[0] - line_info['coord']) < tolerance
    else:
        return abs(vehicle_center[1] - line_info['coord']) < tolerance

# ─── Drawing ───────────────────────────────────────────────────────────────────
def draw_vehicle(frame, veh):
    x1, y1, x2, y2 = [int(v) for v in veh.bbox]
    cx, cy = veh.center
    if veh.seatbelt_on is True:
        color, sb_text = (0, 230, 120), 'BELT: ON'
    elif veh.seatbelt_on is False:
        color, sb_text = (50, 50, 255),  'BELT: OFF'
    else:
        color, sb_text = (255, 165, 0),  'BELT: ?'

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

    # top label background
    label = f'ID:{veh.vehicle_id}  {sb_text}'
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    if veh.license_plate:
        plate_label = f'PLATE: {veh.license_plate}'
        (pw, ph), _ = cv2.getTextSize(plate_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, y2), (x1 + pw + 6, y2 + ph + 10), (0, 0, 0), -1)
        cv2.putText(frame, plate_label, (x1 + 3, y2 + ph + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    # Draw Plate Box ONLY if detected in current frame
    if hasattr(veh, 'current_plate_bbox') and veh.current_plate_bbox:
        cpx1, cpy1, cpx2, cpy2 = veh.current_plate_bbox
        # Adjust to global coordinates
        cv2.rectangle(frame, (x1 + cpx1, y1 + cpy1), (x1 + cpx2, y1 + cpy2), (0, 255, 255), 2)
        
    return frame


def draw_line(frame, line_info, count):
    cv2.line(frame, line_info['pt1'], line_info['pt2'], (0, 230, 255), 2)
    mid_x = (line_info['pt1'][0] + line_info['pt2'][0]) // 2
    mid_y = (line_info['pt1'][1] + line_info['pt2'][1]) // 2
    cv2.putText(frame, f'TOTAL: {count}', (mid_x - 55, mid_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 230, 255), 2)
    return frame

# ─── Core Processing Function ─────────────────────────────────────────────────
def process_video(video_path, output_path, line_ratio, skip_frames,
                  vehicle_model, licence_plate_model, seatbelt_model, ocr_reader,
                  progress_bar, status_text, live_frame_placeholder, detection_placeholder):

    status_text.info("🔍 Analyzing video structure and motion...")
    motion_info = analyze_video_motion(video_path)
    if motion_info is None:
        st.error("❌ Failed to read video. Please ensure it is a valid video file.")
        return None, None

    line_info = build_counting_line(motion_info, ratio=line_ratio)

    cap    = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ OpenCV could not open the video file.")
        return None, None

    fps    = motion_info['fps'] or 25
    W, H   = motion_info['width'], motion_info['height']
    total  = motion_info['total']

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    tracker     = VehicleTracker(iou_threshold=0.3, max_missing_frames=30)
    crossed_ids = set()
    frame_num   = 0
    session_id  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        if frame_num % skip_frames != 0:
            out.write(frame)
            continue

        # vehicle detection (Filtered: 2:car, 5:bus, 7:truck)
        v_results  = vehicle_model(frame, verbose=False)
        detections = []
        if v_results and len(v_results[0].boxes) > 0:
            for box in v_results[0].boxes:
                cls_idx = int(box.cls[0])
                if cls_idx in [2, 5, 7]: # Only Car, Bus, Truck
                    bbox = tuple(int(v) for v in box.xyxy[0])
                    conf = float(box.conf[0])
                    if conf > 0.4:
                        detections.append((bbox, conf))

        active = tracker.update(detections, frame_num)

        for veh in active:
            veh.current_plate_bbox = None # Reset every frame
            if frame_num % 10 == 0:
                try:
                    vx1, vy1, vx2, vy2 = [int(v) for v in veh.bbox]
                    v_crop = frame[vy1:vy2, vx1:vx2]
                    
                    if veh.license_plate is None or len(str(veh.license_plate)) < 4:
                        p_text, p_bbox = detect_license_plate_in_crop(v_crop, licence_plate_model, ocr_reader)
                        if p_text: 
                            veh.license_plate = p_text
                            veh.current_plate_bbox = p_bbox
                    
                    if veh.seatbelt_on is None:
                        veh.seatbelt_on = check_seatbelt(v_crop, seatbelt_model)
                except: pass

            if crosses_line(veh.center, line_info):
                veh.crossed_line = True
                if veh.vehicle_id not in crossed_ids:
                    crossed_ids.add(veh.vehicle_id)
                    # Use best known info
                    scan_status = f"🔍 ID:{veh.vehicle_id} | Plate:{veh.license_plate or 'NOT FOUND'} | Belt:{'YES' if veh.seatbelt_on else 'NO'}"
                    detection_placeholder.markdown(f'<div class="info-box" style="border-left: 4px solid var(--accent); background: #1a1d24;">{scan_status}</div>', unsafe_allow_html=True)


        frame = draw_line(frame, line_info, len(crossed_ids))
        for veh in active:
            frame = draw_vehicle(frame, veh)

        cv2.putText(frame, f'{frame_num}/{total}', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 1)

        out.write(frame)
        tracker.cleanup_old(frame_num)

        # update UI every 10 frames
        if frame_num % 10 == 0:
            pct = frame_num / max(total, 1)
            progress_bar.progress(min(pct, 1.0))
            status_text.markdown(
                f'<div class="info-box">⏳ Frame <b>{frame_num}</b> / {total} &nbsp;|&nbsp; '
                f'Vehicles tracked: <b>{tracker.next_id - 1}</b> &nbsp;|&nbsp; '
                f'Line crossed: <b>{len(crossed_ids)}</b></div>',
                unsafe_allow_html=True)

            # Resize image for better screen fit 
            target_h = 1000
            h, w = frame.shape[:2]
            scale = target_h / h
            new_w = int(w * scale)
            resized_frame = cv2.resize(frame, (new_w, target_h))

            rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            live_frame_placeholder.image(rgb, use_container_width=False)
            
        # Force garbage collection only every 50 frames
        if frame_num % 50 == 0:
            gc.collect()


    cap.release()
    out.release()

    df = tracker.get_dataframe()
    # Deduplicate by license plate if valid, keeping the last occurrence
    if not df.empty and 'license_plate' in df.columns:
        # Separate 'NOT DETECTED' vs valid plates
        df_valid = df[df['license_plate'] != 'NOT DETECTED']
        df_invalid = df[df['license_plate'] == 'NOT DETECTED']
        
        if not df_valid.empty:
            # Group by plate and take the last record (most recent)
            df_valid = df_valid.drop_duplicates(subset=['license_plate'], keep='last')
            
        df = pd.concat([df_valid, df_invalid]).sort_values(by='last_frame').reset_index(drop=True)

    if not df.empty:
        v_ids = []
        for _, row in df.iterrows():
            v_data = row.to_dict()
            v_data['session_id'] = session_id
            v_data['timestamp'] = datetime.utcnow()
            save_vehicle_data(v_data)
            v_ids.append(v_data['vehicle_id'])
            
        session_doc = {
            "session_id": session_id,
            "start_time": datetime.utcnow(),
            "video_name": Path(video_path).name,
            "total_vehicles": len(df),
            "total_violations": len(df[df['crossed_line'] == True]),
            "vehicle_ids": v_ids
        }
        save_session_data(session_doc)

    return df, motion_info

# ══════════════════════════════════════════════════════════════════════════════
#  UI LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-title">🚗 SmartCity Vehicle Intelligence</div>
    <p class="hero-sub">REAL-TIME VEHICLE TRACKING · LICENSE PLATE OCR · SEATBELT DETECTION · MONGODB LOGGING</p>
</div>
""", unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:

    st.markdown('<div class="section-header">📐 Line Settings</div>', unsafe_allow_html=True)
    line_ratio   = st.slider("Line Position", 0.1, 0.95, 0.85, 0.05,
                              help="0.5 = screen center, 0.9 = near end")
    skip_frames  = st.selectbox("Process Every N Frames", [1, 2, 3, 4, 5], index=0,
                                help="Higher = faster but less accurate")

# Main tabs
tab_process, tab_history = st.tabs(["🎬 Process Video", "📋 Detection History"])

# ─── TAB 1: Process ────────────────────────────────────────────────────────────
with tab_process:

    col_upload, _ = st.columns([2, 1])
    with col_upload:
        st.markdown('<div class="section-header">📤 Upload Video</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop a traffic video here",
            type=['mp4', 'avi', 'mov', 'mkv'],
            label_visibility="collapsed")

    res_dir = "results"
    if not os.path.exists(res_dir): os.makedirs(res_dir)

    if uploaded:
        # save upload to temp
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded.read())
        tfile.flush()
        video_tmp = tfile.name

        out_path = os.path.join(res_dir, f"tracked_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")

        run_col, _ = st.columns([1, 3])
        with run_col:
            run_btn = st.button("▶ RUN DETECTION", width="stretch")

        if run_btn:
            # Cleanup previous results
            if os.path.exists(res_dir):
                for f in os.listdir(res_dir):
                    try: os.unlink(os.path.join(res_dir, f))
                    except: pass
                    
            st.markdown('<div class="section-header">⚡ Live Processing</div>', unsafe_allow_html=True)

            progress_bar      = st.progress(0)
            status_text       = st.empty()
            det_placeholder   = st.empty() # For "Scan check" updates
            live_placeholder  = st.empty()

            with st.spinner("Loading models..."):
                try:
                    vehicle_model, licence_plate_model, seatbelt_model, ocr_reader = load_models()
                except Exception as e:
                    st.error(f"Model loading failed: {e}")
                    st.stop()

            # Using global mongo_col initialized at top level

            start_time = time.time()
            df, motion_info = process_video(
                video_tmp, out_path, line_ratio, skip_frames,
                vehicle_model, licence_plate_model, seatbelt_model, ocr_reader,
                progress_bar, status_text, live_placeholder, det_placeholder)

            elapsed = time.time() - start_time
            progress_bar.progress(1.0)
            status_text.markdown(
                f'<div class="info-box" style="color:var(--green)">✅ Finalized in <b>{elapsed:.1f}s</b></div>',
                unsafe_allow_html=True)
            det_placeholder.empty()

            if df is not None and not df.empty:
                st.markdown('<div class="section-header">📊 Final Statistics</div>', unsafe_allow_html=True)
                total_v  = len(df)
                belt_on  = (df['seatbelt'] == 'YES').sum()
                belt_off = (df['seatbelt'] == 'NO').sum()
                crossed  = df['crossed_line'].sum()

                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card accent"><div class="metric-label">Total Tracked</div><div class="metric-value">{total_v}</div></div>
                    <div class="metric-card green"><div class="metric-label">Belt ON</div><div class="metric-value">{belt_on}</div></div>
                    <div class="metric-card red"><div class="metric-label">Belt OFF</div><div class="metric-value">{belt_off}</div></div>
                    <div class="metric-card amber"><div class="metric-label">Line Crossed</div><div class="metric-value">{crossed}</div></div>
                </div>
                """, unsafe_allow_html=True)

                # Motion info
                if motion_info:
                    st.markdown(
                        f'<div class="info-box">🧭 Motion direction: <b>{motion_info.get("direction","N/A")}</b> &nbsp;|&nbsp; '
                        f'Line mode: <b>{motion_info.get("recommended_mode","N/A")}</b> &nbsp;|&nbsp; '
                        f'Resolution: <b>{motion_info["width"]}×{motion_info["height"]}</b> &nbsp;|&nbsp; '
                        f'FPS: <b>{motion_info["fps"]:.1f}</b></div>',
                        unsafe_allow_html=True)

                # Results section
                try:
                    if os.path.exists(out_path):
                        with open(out_path, 'rb') as f:
                            video_bytes = f.read()
                        
                        st.success("✅ Video processing complete!")
                        st.download_button(
                            label="⬇ Download Processed Video",
                            data=video_bytes,
                            file_name=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                            mime="video/mp4",
                            use_container_width=True)
                    else:
                        st.error("⚠️ Processed video file not found. Please try re-processing.")
                except Exception as e:
                    st.error(f"❌ Error preparing download: {e}")

                # Data table
                st.markdown('<div class="section-header">🗃️ Detection Data</div>', unsafe_allow_html=True)

                # Seatbelt badge rendering
                def style_seatbelt(val):
                    if val == 'YES':
                        return 'color: #00ff88; font-weight: 700;'
                    elif val == 'NO':
                        return 'color: #ff3b5c; font-weight: 700;'
                    return 'color: #ffb300;'

                # In Pandas 2.0+, Styler.applymap was renamed to Styler.map
                try:
                    styled_df = df.style.map(style_seatbelt, subset=['seatbelt'])
                except AttributeError:
                    styled_df = df.style.applymap(style_seatbelt, subset=['seatbelt'])
                st.dataframe(styled_df, width="stretch", height=800)

                # CSV download
                try:
                    csv_data = df.to_csv(index=False).encode()
                    json_data = df.to_json(orient='records', indent=2).encode()
                    
                    dl1, dl2 = st.columns(2)
                    with dl1:
                        st.download_button(
                            "⬇ Download CSV",
                            data=csv_data,
                            file_name=f"vehicle_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True)
                    with dl2:
                        st.download_button(
                            "⬇ Download JSON",
                            data=json_data,
                            file_name=f"vehicle_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True)
                except Exception as e:
                    st.error(f"⚠️ Error preparing data downloads: {e}")

                # Charts
                st.markdown('<div class="section-header">📈 Analytics</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)

                try:
                    with c1:
                        import plotly.express as px
                        sb_counts = df['seatbelt'].value_counts().reset_index()
                        sb_counts.columns = ['Status', 'Count']
                        color_map = {'YES': '#00ff88', 'NO': '#ff3b5c', 'NOT DETECTED': '#ffb300'}
                        fig = px.pie(sb_counts, names='Status', values='Count',
                                     title='Seatbelt Compliance',
                                     color='Status', color_discrete_map=color_map,
                                     hole=0.4)
                        fig.update_layout(
                            paper_bgcolor='#111318', plot_bgcolor='#111318',
                            font_color='#e8eaf0', title_font_color='#00e5ff',
                            legend_font_color='#e8eaf0')
                        st.plotly_chart(fig, width="stretch")

                    with c2:
                        df['frames_tracked'] = df['last_frame'] - df['first_frame']
                        fig2 = px.bar(df, x='vehicle_id', y='frames_tracked',
                                      title='Frames Tracked per Vehicle',
                                      color='seatbelt',
                                      color_discrete_map=color_map)
                        fig2.update_layout(
                            paper_bgcolor='#111318', plot_bgcolor='#111318',
                            font_color='#e8eaf0', title_font_color='#00e5ff',
                            xaxis=dict(gridcolor='#1e2330'),
                            yaxis=dict(gridcolor='#1e2330'))
                        st.plotly_chart(fig2, width="stretch")
                except Exception as viz_e:
                    st.error(f"Visualization error: {viz_e}")


                st.markdown(
                    f'<div class="info-box">✅ Analysis complete: <b>{len(df)}</b> vehicles tracked.</div>',
                    unsafe_allow_html=True)

            # cleanup
            try:
                os.unlink(video_tmp)
            except Exception:
                pass

# ─── TAB 2: History ────────────────────────────────────────────────────────────
with tab_history:
    st.markdown('<div class="section-header">📋 Detection History — Sessions</div>', unsafe_allow_html=True)

    if st.button("🔄 Refresh Sessions"):
        st.cache_resource.clear()

    sessions = fetch_sessions_data()

    if not sessions:
        st.info("No processing sessions found in MongoDB.")
    else:
        # Show Sessions as a selection
        session_options = {f"{s['session_id']} | {s.get('video_name','Untyped')} | {s.get('total_violations',0)} violations": s['session_id'] for s in sessions}
        if session_options:
            selected_label = st.selectbox("Select a Session to View", options=list(session_options.keys()))
            selected_sid = session_options[selected_label]

            # Display Session Summary
            sess_meta = next(s for s in sessions if s['session_id'] == selected_sid)
            c1, c2, c3 = st.columns(3)
            c1.metric("Violations", sess_meta.get('total_violations', 0))
            
            s_time = sess_meta.get('start_time')
            time_str = s_time.strftime("%Y-%m-%d %H:%M:%S") if hasattr(s_time, 'strftime') else str(s_time)
            c2.metric("Date", time_str.split(' ')[0])
            c3.metric("Time", time_str.split(' ')[1])

            st.markdown("---")
            st.markdown(f"### 🚗 Vehicles in Session: `{selected_sid}`")

            # Fetch vehicles specific to this session
            vehicles = fetch_vehicles_for_session(selected_sid)
            
            if vehicles:
                det_df = pd.DataFrame(vehicles)
                # Cleanup for display
                if '_id' in det_df.columns: det_df = det_df.drop(columns=['_id'])
                
                st.dataframe(det_df, width="stretch", height=400)
                csv = det_df.to_csv(index=False).encode()
                st.download_button("⬇ Export Session CSV", data=csv, file_name=f"session_{selected_sid}.csv", mime="text/csv")
            else:
                st.warning("No vehicle records found for this session.")
        else:
            st.warning("No sessions match your criteria.")
