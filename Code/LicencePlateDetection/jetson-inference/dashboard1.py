import jetson.inference
import jetson.utils
import easyocr
import time
import cv2
import numpy as np

# 1. Initialize Detection (LPDNet)
net = jetson.inference.detectNet(argv=[
    '--model=/home/nvidia/SSD/jetson-inference/data/networks/LPDNet/LPDNet.onnx',
    '--labels=/home/nvidia/SSD/jetson-inference/data/networks/LPDNet/labels.txt',
    '--input-blob=input_1:0',
    '--output-cvg=output_cov/Sigmoid:0',
    '--output-bbox=output_bbox/BiasAdd:0'
])

# 2. Initialize OCR (German/English)
reader = easyocr.Reader(['en'], gpu=True)

# 3. Setup Camera and Output
camera = jetson.utils.videoSource("/dev/video0")
display = jetson.utils.videoOutput("rtp://192.168.55.100:1234") # Stream to your laptop

# 4. Billing Database
parking_lot = {} # { "PLATE": entry_time }

print("--- ALPR DASHBOARD STARTING ---")

while display.IsStreaming():
    img = camera.Capture()
    if img is None: continue

    # DETECT PLATE
    detections = net.Detect(img)

    for det in detections:
        # CROP PLATE FROM GPU MEMORY
        # Define the region of interest from detection
        roi = (int(det.Left), int(det.Top), int(det.Right), int(det.Bottom))
        
        # Allocate memory for the crop
        img_crop = jetson.utils.cudaAllocMapped(width=det.Width, height=det.Height, format=img.format)
        jetson.utils.cudaCrop(img, img_crop, roi)
        
        # Convert to format EasyOCR understands
        img_np = jetson.utils.cudaToNumpy(img_crop)
        
        # RUN OCR
        results = reader.readtext(img_np)
        for (bbox, text, prob) in results:
            if prob > 0.5:
                plate = text.upper().replace(" ", "")
                
                # BILLING LOGIC
                if plate not in parking_lot:
                    parking_lot[plate] = time.time()
                    status = f"ENTRY: {plate}"
                else:
                    duration_sec = time.time() - parking_lot[plate]
                    fee = (duration_sec / 60) * 0.05 # 0.05 Euro per minute
                    status = f"EXIT: {plate} | FEE: {fee:.2f} EUR"
                
                # DRAW ON SCREEN
                jetson.utils.cudaDrawText(img, det.Left, det.Top-30, status, (0,255,0,255))
    
    display.Render(img)
    display.SetStatus(f"ALPR | {len(parking_lot)} CARS | FPS: {net.GetNetworkFPS():.1f}")