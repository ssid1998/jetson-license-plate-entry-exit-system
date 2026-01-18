import os, time, cv2
import numpy as np
import jetson.inference
import jetson.utils
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from threading import Thread
from flask import Flask, render_template_string, jsonify, Response
from datetime import datetime
import easyocr
import re

# --- THE PILLOW PATCH ---
import PIL.Image
if not hasattr(PIL.Image, 'Resampling'):
    PIL.Image.Resampling = PIL.Image

# 1. HARDWARE FIXES
os.environ['LD_PRELOAD'] = '/usr/lib/aarch64-linux-gnu/libgomp.so.1'
os.environ['OPENBLAS_CORETYPE'] = 'ARMV8'

# --- CONFIGURATION ---
ENGINE_PATH = "/media/myssd/jetson-inference/data/networks/best.engine"
CONFIDENCE_THRESHOLD = 0.45

# --- BILLING SETTINGS ---
RATE_PER_SECOND = 1.0 
OCR_COOLDOWN = 3.0     

# --- GLOBAL VARIABLES ---
parking_lot = {}  
output_frame = None
camera = jetson.utils.videoSource("/dev/video0", argv=['--input-width=640', '--input-height=480'])

print("Initializing OCR (CPU mode)...")
# 'quantize=False' can sometimes improve accuracy on small text
reader = easyocr.Reader(['en'], gpu=False, quantize=False)

app = Flask(__name__)

# --- DASHBOARD UI ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jetson AI Parking</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #0f0f0f; color: #00ff00; font-family: monospace; }
        .video-box { border: 2px solid #00ff00; border-radius: 8px; overflow: hidden; }
        .table-dark { --bs-table-bg: #1a1a1a; color: #00ff00; border-color: #333; }
        .plate-badge { background: #ffcc00; color: black; padding: 2px 8px; border-radius: 4px; font-weight: bold; }
        .money { color: #00ff00; font-size: 1.2em; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container-fluid p-4">
        <h2 class="text-center mb-4">🅿️ JETSON BILLING SYSTEM</h2>
        <div class="row">
            <div class="col-md-7">
                <div class="video-box">
                    <img src="{{ url_for('video_feed') }}" width="100%">
                </div>
            </div>
            <div class="col-md-5">
                <div class="card bg-dark border-success">
                    <div class="card-header border-success">LIVE BILLING (Rate: €1.00 / sec)</div>
                    <div class="card-body p-0">
                        <table class="table table-dark table-hover mb-0 text-center">
                            <thead>
                                <tr>
                                    <th>Plate</th>
                                    <th>In</th>
                                    <th>Last Seen</th>
                                    <th>Bill</th>
                                </tr>
                            </thead>
                            <tbody id="billing-body"></tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script>
        setInterval(() => {
            fetch('/api/billing').then(r => r.json()).then(data => {
                let html = '';
                let sorted = Object.keys(data).sort((a,b) => data[b].entry_ts - data[a].entry_ts);
                for (let p of sorted) {
                    let car = data[p];
                    html += `<tr>
                        <td><span class="plate-badge">${p}</span></td>
                        <td>${car.entry_str}</td>
                        <td>${car.last_seen_str}</td>
                        <td class="money">€${car.fee.toFixed(2)}</td>
                    </tr>`;
                }
                document.getElementById('billing-body').innerHTML = html;
            });
        }, 1000);
    </script>
</body>
</html>
"""

@app.route('/')
def index(): return render_template_string(HTML_PAGE)

@app.route('/api/billing')
def get_billing():
    response_data = {}
    for plate, info in parking_lot.items():
        duration_seconds = info['last_seen'] - info['entry']
        fee = duration_seconds * RATE_PER_SECOND
        response_data[plate] = {
            "entry_str": datetime.fromtimestamp(info['entry']).strftime("%H:%M:%S"),
            "last_seen_str": datetime.fromtimestamp(info['last_seen']).strftime("%H:%M:%S"),
            "entry_ts": info['entry'],
            "last_seen_ts": info['last_seen'],
            "fee": fee
        }
    return jsonify(response_data)

def generate_frames():
    global output_frame
    while True:
        if output_frame is not None:
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + output_frame + b'\r\n')
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- THE "ADAPTIVE" IMAGE CLEANER (BEST OPTIMIZATION) ---
def clean_and_erase_stickers(plate_img):
    """
    Advanced filter that uses Adaptive Thresholding to handle shadows
    and Aspect Ratio filtering to target square stickers.
    """
    if plate_img.size == 0: return plate_img

    # 1. Upscale (Crucial for small stickers)
    h, w = plate_img.shape[:2]
    scale = 2
    scaled = cv2.resize(plate_img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale
    gray = cv2.cvtColor(scaled, cv2.COLOR_RGB2GRAY)

    # 3. Adaptive Thresholding (The "Shadow Killer")
    # This separates text from background even in bad lighting
    # Block size 29, C=15 are tuned for license plates
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 29, 15)

    # 4. Morphological Opening (The "Separator")
    # Disconnects stickers if they are touching letters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 5. Find Contours (Crash-Proof Version)
    cnts_data = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts_data[0] if len(cnts_data) == 2 else cnts_data[1]

    if not contours: return gray

    # 6. Calculate Median Height of LETTERS only
    heights = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Only consider things that are reasonably tall (likely letters)
        if ch > (h * scale * 0.3): 
            heights.append(ch)
    
    if len(heights) == 0: return gray
    median_h = np.median(heights)

    # 7. Create Final Image for OCR (Black text on White background)
    final_img = cv2.bitwise_not(binary) 

    # 8. The Eraser Loop
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        
        # HEURISTIC A: Height Check
        # Stickers are shorter than letters
        is_short = ch < (0.70 * median_h)

        # HEURISTIC B: Shape Check
        # Stickers are often square-ish (aspect ratio ~1.0)
        # Letters are usually tall/thin (aspect ratio < 0.6)
        ratio = cw / float(ch)
        is_square_badge = (0.8 < ratio < 1.3) and (ch < 0.85 * median_h)

        # HEURISTIC C: Edge Noise
        # Remove partial shapes touching the left/right border
        is_edge_noise = (x < 5) or ((x + cw) > (w * scale - 5))

        if is_short or is_square_badge or is_edge_noise:
            # Paint pure white over the sticker
            cv2.drawContours(final_img, [cnt], -1, (255, 255, 255), -1) 

    return final_img

# --- AI LOGIC ---
def run_yolo_ai():
    global output_frame
    last_ocr_time = 0
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

        print("--- System Online: Adaptive Eraser Active ---")

        while True:
            img = camera.Capture()
            if img is None: continue
            
            ctx.push()
            img_np = jetson.utils.cudaToNumpy(img)
            h, w, _ = img_np.shape
            
            blob = cv2.resize(img_np, (640, 640)).astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1).ravel()
            
            np.copyto(inputs[0]['host'], blob)
            cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
            stream.synchronize()
            
            predictions = outputs[0]['host'].reshape(1, 25200, 6)
            
            current_time = time.time()
            if (current_time - last_ocr_time) > OCR_COOLDOWN:
                for i in range(25200):
                    conf = predictions[0, i, 4]
                    if conf > CONFIDENCE_THRESHOLD:
                        x_c, y_c, bw, bh = predictions[0, i, 0:4]
                        l, t = int((x_c-bw/2)*w/640), int((y_c-bh/2)*h/640)
                        r, b = int((x_c+bw/2)*w/640), int((y_c+bh/2)*h/640)

                        strip_w = int(bw * 0.15)
                        left_strip = img_np[max(0,t):min(h,b), max(0,l):max(0,l+strip_w)]
                        
                        is_german = False
                        if left_strip.size > 0:
                            hsv = cv2.cvtColor(left_strip, cv2.COLOR_RGB2HSV)
                            # Standard Blue Strip ranges
                            mask = cv2.inRange(hsv, np.array([100, 150, 50]), np.array([140, 255, 255]))
                            if (np.sum(mask > 0) / mask.size) > 0.15:
                                is_german = True

                        if is_german:
                            cv2.rectangle(img_np, (l, t), (r, b), (0, 255, 0), 3)
                            
                            crop = img_np[max(0,t):min(h,b), max(0,l):min(w,r)]
                            
                            # === STEP 1: CLEAN & ERASE ===
                            clean_crop = clean_and_erase_stickers(crop)
                            
                            # === STEP 2: READ TEXT ===
                            # Use strict allowlist to avoid random symbols
                            results = reader.readtext(clean_crop, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                            
                            raw_plate = "".join(results).upper()
                            
                            # === STEP 3: FINAL SANITY CHECK ===
                            # Must have at least 4 chars and contain digits
                            if len(raw_plate) >= 4 and any(c.isdigit() for c in raw_plate):
                                
                                # Logic: If the car is new or we haven't seen it in 3 seconds
                                if raw_plate not in parking_lot:
                                    parking_lot[raw_plate] = { "entry": current_time, "last_seen": current_time }
                                    print(f"[NEW CAR] {raw_plate}")
                                else:
                                    parking_lot[raw_plate]["last_seen"] = current_time
                                
                                cv2.putText(img_np, raw_plate, (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                last_ocr_time = time.time()
            else:
                pass

            ctx.pop()
            output_frame = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))[1].tobytes()

    finally:
        ctx.pop()

if __name__ == '__main__':
    Thread(target=run_yolo_ai, daemon=True).start()
    app.run(host='0.0.0.0', port=8051, threaded=True)