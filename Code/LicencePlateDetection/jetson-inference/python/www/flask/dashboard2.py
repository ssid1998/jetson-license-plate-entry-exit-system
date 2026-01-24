# -*- coding: utf-8 -*-
import os
import time
import cv2
import numpy as np
import jetson.inference
import jetson.utils
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from threading import Thread
from flask import Flask, render_template_string, jsonify, Response, request
from datetime import datetime
import easyocr
import re
import PIL.Image
import subprocess
import sqlite3

# --- 1. CONFIGURATION ---
# >>> CHANGE YOUR PASSWORD HERE <<<
SYSTEM_PASSWORD = "nvidia" 

# File Paths
ENGINE_PATH = "/media/myssd/jetson-inference/data/networks/best.engine"
DB_PATH = "/media/myssd/jetson-inference/python/www/flask/parking.db"

# AI Settings
CONFIDENCE_THRESHOLD = 0.45
OCR_MIN_CONFIDENCE = 0.4
OCR_DELAY = 0.5 
OCR_COOLDOWN = 2.0

# --- 2. CORRECTIONS ---
OCR_CORRECTIONS = {
    "MPG": "MG", "YOB": "WOB", "VOB": "WOB", "VVOB": "WOB", "EM0": "EMD", "END": "EMD"
}
VALID_GERMAN_CITIES = {
    "B", "M", "HH", "EM", "EMD", "LER", "AUR", "WTM", "WOB", "MG", "K", "D", 
    "S", "H", "F", "HB", "L", "N", "DO", "E", "DU", "ZK"
}

# --- 3. HARDWARE OPTIMIZATIONS ---
os.environ['LD_PRELOAD'] = '/usr/lib/aarch64-linux-gnu/libgomp.so.1'
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'
if not hasattr(PIL.Image, 'Resampling'): PIL.Image.Resampling = PIL.Image

# --- GLOBAL VARIABLES ---
output_frame = None
last_valid_plate = "WAITING..."
last_valid_time = 0

# --- 4. DATABASE SETUP ---
def init_db():
    """Creates the database file if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Create table: Plate (ID), Entry Time, Last Seen Time
        c.execute('''CREATE TABLE IF NOT EXISTS visits 
                     (plate TEXT PRIMARY KEY, entry_ts REAL, last_seen_ts REAL)''')
        conn.commit()
        conn.close()
        print(f"--- DATABASE CONNECTED: {DB_PATH} ---")
    except Exception as e:
        print(f"!!! DATABASE ERROR: {e} !!!")

# Initialize DB immediately
init_db()

# --- 5. CAMERA SETUP ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Initializing OCR Engine...")
reader = easyocr.Reader(['en'], gpu=False, quantize=True)

app = Flask(__name__)

# --- 6. UI WITH SHUTDOWN BUTTON ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jetson Grid Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background-color: #000; color: #0f0; font-family: monospace; height: 100vh; overflow: hidden; }
        .main-grid { display: grid; grid-template-columns: 65fr 35fr; height: 100vh; width: 100vw; }
        
        .video-panel { background: #000; display: flex; align-items: center; justify-content: center; border-right: 2px solid #333; }
        .video-panel img { width: 100%; height: auto; max-height: 100vh; object-fit: contain; }

        .table-panel { background: #111; padding: 10px; display: flex; flex-direction: column; justify-content: space-between; height: 100vh; }
        .table-content { flex-grow: 1; overflow-y: auto; }

        table { width: 100%; border-collapse: collapse; }
        th { text-align: left; padding: 10px; border-bottom: 2px solid #444; color: #888; position: sticky; top: 0; background: #111; }
        td { padding: 10px; border-bottom: 1px solid #222; color: #fff; font-size: 1.1em; }
        .plate { background: #fc0; color: #000; padding: 2px 6px; border-radius: 4px; font-weight: bold; }
        .fee { color: #0f0; font-weight: bold; text-align: right; }

        .btn-shutdown {
            background: #d00; color: white; border: none; width: 100%; padding: 20px;
            font-size: 1.5em; font-weight: bold; cursor: pointer; margin-top: 10px;
        }
        .btn-shutdown:active { background: #900; }
    </style>
</head>
<body>
    <div class="main-grid">
        <div class="video-panel"><img src="{{ url_for('video_feed') }}"></div>
        <div class="table-panel">
            <div class="table-content">
                <h2 style="text-align:center; color:#fff; border-bottom:2px solid #0f0; padding-bottom:10px;">PARKING LOG (DB)</h2>
                <table>
                    <thead><tr><th>PLATE</th><th>TIME</th><th>FEE</th></tr></thead>
                    <tbody id="billing-body"></tbody>
                </table>
            </div>
            <button class="btn-shutdown" onclick="doShutdown()">🛑 SHUTDOWN</button>
        </div>
    </div>
    <script>
        function updateTable() {
            fetch('/api/billing')
                .then(r => r.json())
                .then(data => {
                    let html = '';
                    let sorted = Object.keys(data).sort((a,b) => data[b].entry_ts - data[a].entry_ts);
                    if (sorted.length === 0) {
                        html = '<tr><td colspan="3" style="text-align:center; color:#555;">No cars in database...</td></tr>';
                    } else {
                        for (let p of sorted) {
                            html += `<tr><td><span class="plate">${p}</span></td><td>${data[p].entry_str}</td><td class="fee">${data[p].fee.toFixed(2)}€</td></tr>`;
                        }
                    }
                    document.getElementById('billing-body').innerHTML = html;
                })
                .catch(err => console.log("API Error:", err));
        }
        setInterval(updateTable, 1000);

        function doShutdown() {
            if(confirm("POWER OFF JETSON?")) {
                fetch('/api/shutdown', {method: 'POST'});
                alert("Shutting down... Screen will go black.");
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(HTML_PAGE)

# --- API ROUTES (Connected to SQLite) ---
@app.route('/api/billing')
def get_billing():
    # Fetch data from SQLite instead of RAM
    response_data = {}
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT plate, entry_ts, last_seen_ts FROM visits")
        rows = c.fetchall()
        conn.close()

        for r in rows:
            plate, entry, last = r
            fee = (last - entry) * 0.10 # 10 cents per second (Example fee)
            response_data[plate] = {
                "entry_str": datetime.fromtimestamp(entry).strftime("%H:%M:%S"),
                "entry_ts": entry,
                "fee": fee
            }
    except Exception as e:
        print("READ ERROR:", e)
        
    return jsonify(response_data)

@app.route('/api/shutdown', methods=['POST'])
def shutdown():
    print("!!! EXECUTING SHUTDOWN !!!")
    # Uses the variable SYSTEM_PASSWORD defined at the top
    cmd = f"echo {SYSTEM_PASSWORD} | sudo -S poweroff"
    subprocess.call(cmd, shell=True)
    return jsonify({"status": "bye"})

def generate_frames():
    global output_frame
    while True:
        if output_frame is not None:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')
        time.sleep(0.08)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- HELPERS ---
def pro_preprocess(img):
    if img.size == 0: return img
    h, w = img.shape[:2]
    ratio = 64.0 / h
    new_w = int(w * ratio)
    resized = cv2.resize(img, (new_w, 64), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def clean_german_plate(raw_text):
    text = raw_text.upper().strip()
    for bad, good in OCR_CORRECTIONS.items():
        if text.startswith(bad): text = text.replace(bad, good, 1)

    found_city = None
    remaining_text = ""
    for length in [3, 2, 1]:
        if len(text) < length: continue
        prefix = text[:length]
        if prefix in VALID_GERMAN_CITIES:
            found_city = prefix
            remaining_text = text[length:] 
            break
            
    if not found_city: return False, None
    match = re.search(r'([A-Z0-9]*?)([0-9]{1,4})$', remaining_text)
    if match:
        middle_letters = re.sub(r'[^A-Z]', '', match.group(1))
        return True, f"{found_city}-{middle_letters}{match.group(2)}"
    return False, None

# --- AI LOOP (Writes to SQLite) ---
def run_yolo_ai():
    global output_frame, last_valid_plate, last_valid_time
    last_ocr_time = 0
    box_stability_start = 0  
    
    dev = cuda.Device(0)
    ctx = dev.make_context()
    
    try:
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            context = engine.create_execution_context()

        inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(cuda_mem))
            if engine.binding_is_input(binding): inputs.append({'host': host_mem, 'device': cuda_mem})
            else: outputs.append({'host': host_mem, 'device': cuda_mem})

        print("--- SYSTEM ONLINE ---")

        while True:
            ret, frame_cv = cap.read()
            if not ret: time.sleep(1); continue
            
            if (time.time() - last_valid_time) < 5.0:
                cv2.putText(frame_cv, f"READ: {last_valid_plate}", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            ctx.push()
            h, w = frame_cv.shape[:2]
            blob = cv2.resize(frame_cv, (640, 640)).astype(np.float32) / 255.0
            blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
            blob = blob.transpose(2, 0, 1).ravel()
            
            np.copyto(inputs[0]['host'], blob)
            cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
            stream.synchronize()
            
            predictions = outputs[0]['host'].reshape(1, 25200, 6)
            scores = predictions[0, :, 4]
            max_score = np.max(scores)
            
            detected_valid_box = False

            if max_score > CONFIDENCE_THRESHOLD:
                best_idx = np.argmax(scores)
                x_c, y_c, bw, bh = predictions[0, best_idx, 0:4]
                l = max(0, int((x_c-bw/2)*w/640))
                t = max(0, int((y_c-bh/2)*h/640))
                r = min(w, int((x_c+bw/2)*w/640))
                b = min(h, int((y_c+bh/2)*h/640))
                
                if not ((l < 5) or (t < 5) or (r > w - 5) or (b > h - 5)):
                    detected_valid_box = True
                    cv2.rectangle(frame_cv, (l, t), (r, b), (0, 255, 0), 2)

                    if box_stability_start == 0: box_stability_start = time.time()
                    if (time.time() - box_stability_start) > OCR_DELAY and (time.time() - last_ocr_time) > OCR_COOLDOWN:
                        cv2.rectangle(frame_cv, (l, t), (r, b), (0, 0, 255), 3)
                        crop = frame_cv[t:b, l:r]
                        if crop.size != 0:
                            processed_crop = pro_preprocess(crop)
                            results = reader.readtext(processed_crop, detail=1, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                            merged_text = "".join([res[1] for res in results if res[2] > OCR_MIN_CONFIDENCE])
                            
                            if len(merged_text) > 3:
                                valid, clean = clean_german_plate(merged_text)
                                if valid:
                                    last_valid_plate = clean
                                    last_valid_time = time.time()
                                    now = time.time()
                                    
                                    # --- DB TRANSACTION ---
                                    try:
                                        conn = sqlite3.connect(DB_PATH)
                                        c = conn.cursor()
                                        
                                        # Check if car exists
                                        c.execute("SELECT entry_ts FROM visits WHERE plate=?", (clean,))
                                        row = c.fetchone()
                                        
                                        if row is None:
                                            # INSERT NEW
                                            print(f"++ DB INSERT: {clean}")
                                            c.execute("INSERT INTO visits (plate, entry_ts, last_seen_ts) VALUES (?, ?, ?)", (clean, now, now))
                                        else:
                                            # UPDATE EXISTING
                                            print(f"** DB UPDATE: {clean}")
                                            c.execute("UPDATE visits SET last_seen_ts=? WHERE plate=?", (now, clean))
                                            
                                        conn.commit()
                                        conn.close()
                                    except Exception as e:
                                        print("DB WRITE ERROR:", e)
                                    # -----------------------

                        last_ocr_time = time.time()
            
            if not detected_valid_box: box_stability_start = 0

            ctx.pop()
            try:
                ret, buffer = cv2.imencode('.jpg', frame_cv, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ret: output_frame = buffer.tobytes()
            except: pass

    finally:
        if cap.isOpened(): cap.release()
        ctx.pop()

if __name__ == '__main__':
    Thread(target=run_yolo_ai, daemon=True).start()
    app.run(host='0.0.0.0', port=8055, threaded=True)