import os
import time
import cv2
import numpy as np
from threading import Thread
from flask import Flask, render_template_string, jsonify, Response

# 1. HARDWARE FIXES
os.environ['LD_PRELOAD'] = '/usr/lib/aarch64-linux-gnu/libgomp.so.1'
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

import jetson_inference
import jetson_utils
import easyocr

# --- CONFIGURATION ---
MODEL_PATH = "/home/nvidia/SSD/jetson-inference/data/networks/LPDNet/LPDNet.onnx.1.1.8201.GPU.FP16.engine"
LABELS_PATH = "/home/nvidia/SSD/jetson-inference/data/networks/LPDNet/labels.txt"
HOURLY_RATE = 2.50 

# --- AI & CAMERA INITIALIZATION ---
# Threshold lowered to 0.1 for maximum sensitivity to find boxes
net = jetson_inference.detectNet(argv=[
    f'--model={MODEL_PATH}', 
    f'--labels={LABELS_PATH}', 
    '--input-blob=input_1:0', 
    '--output-cvg=output_cov/Sigmoid:0', 
    '--output-bbox=output_bbox/BiasAdd:0',
    '--threshold=0.1' 
])

reader = easyocr.Reader(['en'], gpu=True)
camera = jetson_utils.videoSource("/dev/video0", argv=['--input-width=640', '--input-height=480', '--input-codec=mjpeg'])

output_frame = None
parking_lot = {}

app = Flask(__name__)

# --- WEB DASHBOARD HTML ---
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>ALPR Billing Dashboard</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    <style>
        body { background: #0b0e14; color: #e0e0e0; font-family: sans-serif; padding: 20px; }
        .video-box { border: 3px solid #00ff00; border-radius: 12px; background: black; overflow: hidden; }
    </style>
</head>
<body>
    <div class="container-fluid text-center">
        <h2 class="mb-4">🅿️ Emden Smart Parking Dashboard</h2>
        <div class="row text-start">
            <div class="col-md-8">
                <div class="video-box shadow-lg">
                    <img src="{{ url_for('video_feed') }}" width="100%">
                </div>
            </div>
            <div class="col-md-4">
                <div class="card bg-secondary text-white shadow">
                    <div class="card-header bg-dark"><h4>Active Billing</h4></div>
                    <div class="card-body">
                        <table class="table table-dark table-hover">
                            <thead><tr><th>Plate</th><th>Entry</th><th>Fee (€)</th></tr></thead>
                            <tbody id="billing-body"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        function updateTable() {
            fetch('/api/billing').then(r => r.json()).then(data => {
                let html = '';
                for (let p in data) {
                    html += `<tr><td><strong>${p}</strong></td><td>${data[p].entry}</td><td class="text-warning">${data[p].fee.toFixed(2)}</td></tr>`;
                }
                document.getElementById('billing-body').innerHTML = html;
            });
        }
        setInterval(updateTable, 1000);
    </script>
</body>
</html>
"""

def generate_frames():
    global output_frame
    while True:
        if output_frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')
        time.sleep(0.04)

@app.route('/')
def index(): return render_template_string(HTML_PAGE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/billing')
def get_billing(): return jsonify(parking_lot)

def run_ai_loop():
    global output_frame
    print("--- AI Loop Started: Watching for Plates ---")
    while True:
        img = camera.Capture()
        if img is None: continue

        # Detect potential license plates
        detections = net.Detect(img)
        
        # DEBUG: Print immediately when a box is found
        if len(detections) > 0:
            print(f"--- DETECTED {len(detections)} POTENTIAL PLATE BOX(ES) ---")
        
        for det in detections:
            # ROI and OCR logic
            roi = (int(det.Left), int(det.Top), int(det.Right), int(det.Bottom))
            img_crop = jetson_utils.cudaAllocMapped(width=det.Width, height=det.Height, format=img.format)
            jetson_utils.cudaCrop(img, img_crop, roi)
            
            jetson_utils.cudaDeviceSynchronize()
            img_np = np.array(img_crop, copy=False)
            
            # Use EasyOCR to recognize text
            results = reader.readtext(img_np)
            for (bbox, text, prob) in results:
                print(f"DEBUG: OCR Read '{text}' with {prob:.2f} confidence")
                if prob > 0.3:
                    plate = "".join(filter(str.isalnum, text)).upper()
                    if len(plate) < 4: continue
                    
                    if plate not in parking_lot:
                        parking_lot[plate] = {"entry": time.strftime("%H:%M"), "start": time.time(), "fee": 0.0}
                        print(f"REGISTERED NEW PLATE: {plate}")
                    else:
                        duration_hrs = (time.time() - parking_lot[plate]["start"]) / 3600
                        parking_lot[plate]["fee"] = duration_hrs * HOURLY_RATE

                    jetson_utils.cudaDrawText(img, det.Left, det.Top-40, f"{plate}", (0,255,0,255))
        
        # Prepare full frame for Web UI
        jetson_utils.cudaDeviceSynchronize()
        full_np = np.array(img, copy=False)
        cv_img = cv2.cvtColor(full_np, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', cv_img)
        output_frame = buffer.tobytes()

if __name__ == '__main__':
    Thread(target=run_ai_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=8050, threaded=True)